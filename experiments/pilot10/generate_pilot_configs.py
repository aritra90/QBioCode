#!/usr/bin/env python
"""Generate the pilot qprofiler configs.

One YAML per dataset. The three things that vary per dataset are the dataset itself,
the ``backend`` (chosen by qubit count, see BACKEND_RULE) and whether an embedding
applies; everything else is identical across the pilot so that a difference in the
results is a difference between datasets and not between configs.

Run:  python generate_pilot_configs.py        (writes configs/, overwrites)

Two layouts, same experiment:

  --layout combined   (default) configs/pilotNN_<dataset>.yaml: one LSF job per dataset,
                      13 models fanned out over loky, embeddings looped in sequence.
  --layout split      runs/<dataset>/<dataset>_<embedding>_<model>.yaml: one LSF job per
                      (dataset, embedding, model), n_jobs 1. Every key but the model, the
                      embedding, n_jobs and the output paths is identical to the combined
                      config of that dataset (tests/test_pilot_split_contract.py checks
                      this), so the two layouts differ in scheduling and not in science.
                      The pilot's wall clock becomes its single slowest arm instead of
                      n_embeddings x the slowest arm of each dataset. Submit with
                      submit_runs.sh, watch with status.py, merge with collate_results.py.

The split layout is composed: each dataset directory has one _protocol.yaml holding every
key its jobs share -- data, split, seeds, budgets, search spaces, output location -- with
the job keys left ??? (OmegaConf's "must be set"), and each job file is a dozen lines that
pull it in through hydra's defaults list and set just those keys. So a protocol change is
made once per dataset instead of once per job, and a job file says only which job it is.
--self-contained writes every job as one full config instead; the pilot's runs/ was
generated that way. Either way a job composes to the same config, key for key (tested).

Both layouts read their embedded features from one directory, --embedding-cache
(default embeddings/ here), instead of embedding in each job. The submit scripts write
the files first, one per (dataset, embedding, split), so every job of one embedding sees
the same features whichever host runs it.
"""

import argparse
import ast
import math
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "configs")
RUNS_DIR = os.path.join(HERE, "runs")

# Pilot defaults. Override with --iter / --test-size / --n-trials-quantum for the
# full benchmark.
ITER_DEFAULT = 5
TEST_SIZE_DEFAULT = 0.20
# Quantum tuning trials. This is the knob that sets the job's wall clock, because
# freeze_quantum_params makes the per-arm cost (n_trials_quantum + iter) fits rather
# than iter x n_trials_quantum -- so at iter=5 the search is ~87% of the work and the
# resamples are the remaining ~13%. Cutting resamples to fit a deadline therefore pays
# almost nothing and costs the t-test its degrees of freedom; cutting trials pays
# almost linearly and costs only search depth. See --n-trials-quantum.
N_TRIALS_QUANTUM_DEFAULT = 32
DATA_ROOT = "/dccstor/cgq4hls/Q/qbc_data"

# ---------------------------------------------------------------------------
# Backend rule, by the number of qubits a quantum model will actually use.
#
# The qubit count equals the feature count, because every feature map in this repo is
# one qubit per feature. So the rule is a rule about feature width:
#
#   <= 13 features  -> statevector_simulator   exact, and cheaper than MPS at every
#                                              width measured up to here
#   14-20 features  -> mps_simulator           a dense statevector stops being
#                                              affordable here; MPS cost is nearly flat
#                                              in width while entanglement stays bounded
#   > 20 features   -> embed to n_components=8, then statevector_simulator
#                                              (8 qubits, so back in the first band)
#
# The 13/14 boundary is MEASURED, and it has moved twice: the rule was first written with
# 10, and the table below put it at 12/13. Per-circuit cost of the real ZZ feature map
# (reps=2, 20 circuits, 1024 shots), statevector vs Aer matrix_product_state:
#
#     q    linear: sv / mps          pairwise: sv / mps        verdict
#    10    0.0077 / 0.0302  (x3.9)   0.0078 / 0.0307  (x3.9)   MPS 4x SLOWER
#    13    0.0414 / 0.0414  (x1.0)   0.0417 / 0.0399  (x1.0)   crossover (at 20 circuits)
#    16    0.3928 / 0.0493  (x0.13)  0.4125 / 0.0492  (x0.12)  MPS 8x faster
#    19    3.9315 / 0.0586  (x0.01)  4.1019 / 0.0589  (x0.01)  MPS 67x faster
#
# MPS is essentially flat in width (0.030 -> 0.059 from 10 to 19 qubits) while the
# statevector grows ~2.5x per qubit past 13. Below the crossover MPS only adds overhead,
# so routing 10-13 qubits to MPS costs ~4x for nothing; at 19 qubits it is the difference
# between 0.06 and 3.93 s/circuit, which is what makes hepatitis runnable at all.
#
# The 13-qubit "crossover" holds only at toy row counts, which is what moved the edge from
# 12 to 13. MPS per-circuit cost grows with the training set (cost_model.MPS_ROW_ALPHA)
# while a statevector does the same FLOPs whatever the data, so a tie at 20 circuits is
# not a tie at pilot scale. On the real qsvc path at 13 qubits, heart measured MPS 73.6 ->
# 277.6 ms/circuit from n_tr=40 to n_tr=160, and statevector 71.8 -> 69.9 over the same
# step: ~4x cheaper on the statevector at 160 rows. 14 and 15 qubits were never measured
# (no pilot dataset has that width), so the edge sits at 13, the widest point at which
# the statevector is measured to win; the true crossover at these row counts may lie
# above it.
#
# EMBEDDING_MIN_FEATURES is the other half of the third case: a dataset is embedded
# only when its width EXCEEDS this number, so 20 means widths 1-20 run unembedded and
# widths 21+ embed down to N_COMPONENTS.
# ---------------------------------------------------------------------------
STATEVECTOR_MAX_QUBITS = 13         # inclusive upper bound of band 1 (measured at pilot rows)
MPS_MAX_QUBITS = 20                 # inclusive upper bound of band 2
EMBEDDING_MIN_FEATURES = 20         # embed only when width > this
N_COMPONENTS = 8                    # embedded width, so band 3 lands back in band 1
# Where every layout's jobs read their embedded features (the config's embedding_cache).
# One directory for both layouts, so the combined and the split run of one dataset score
# their models on the same features. A split's file depends on its iteration number, not
# on --iter, so a run with more resamples reuses the files for the splits it shares.
EMBEDDING_CACHE_DIR = os.path.join(HERE, "embeddings")


def _load_cost_model():
    """The measured wall-clock model, imported lazily so the generator still runs bare.

    cost_model.py sits beside this file rather than inside the package: it is calibration
    data for this experiment, not library code, and it is only needed for --budget-hours.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pilot_cost_model", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "cost_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backend_for(n_features):
    """Return (backend, qubits_used, embeddings) for a dataset of this width."""
    if n_features > EMBEDDING_MIN_FEATURES:
        return "statevector_simulator", N_COMPONENTS, "['pca', 'umap']"
    if n_features > STATEVECTOR_MAX_QUBITS:
        return "mps_simulator", n_features, "['none']"
    return "statevector_simulator", n_features, "['none']"


# Entanglement values the tuner may choose, per backend. 'full' is excluded on MPS, and
# this is a feasibility constraint rather than a preference: MPS represents a state in a
# bond dimension that grows with entanglement, and 'full' puts q(q-1)/2 two-qubit gates
# in every rep instead of q-1. Measured per-circuit cost with 'full', same setup as the
# table above:
#
#     q    sv 'full'   mps 'full'    ratio
#    10      0.0155      0.1056       x6.8
#    13      0.0581      1.1556      x19.9
#    16      0.5980     44.5572      x74.5
#
# 44.6 s/circuit at 16 qubits is not a slow run, it is a dead job: labor (16 features,
# 57 rows) needs ~208k circuits for one qnn/vqc arm, so 'full' on MPS is ~2600 CPU-hours
# against a 24h wall, versus ~2.9 CPU-hours on 'linear'. The tuner samples this space
# freely, so leaving 'full' in it means any MPS-band trial can draw the value that kills
# the job. 'pairwise' measures identically to 'linear' on MPS at every width tested, so
# the MPS space keeps two of the three values and loses nothing in expressiveness.
ENTANGLEMENT_BY_BACKEND = {
    "statevector_simulator": "['linear', 'pairwise', 'full']",
    "mps_simulator": "['linear', 'pairwise']",
}


# ---------------------------------------------------------------------------
# The pilot set: 12 datasets.
#
# Picked to span the feature-cardinality range of the full corpus (4 -> 2000), to
# cover all four data sources rather than PMLB alone, to include two synthetic
# QUANTUM datasets, and to exercise all three backend bands (6 statevector,
# 2 MPS, 4 embedded). All are binary and free of missing values -- the load path in
# qprofiler does no imputation, so a dataset with NaNs cannot run as-is.
#
# Row counts were kept low on purpose: quantum cost is (circuits x per-circuit cost)
# and circuit count is linear or quadratic in ROWS, so rows dominate width.
# ---------------------------------------------------------------------------
DATASETS = [
    # (folder, csv, rows, features, why this one is here)
    ("pmlb_data", "analcatdata_lawsuit.csv", 264, 4,
     "narrowest case in the corpus; 4 qubits"),
    ("qdata/x_view", "eng_zz_n6_gq1_s0.csv", 300, 6,
     "QUANTUM dataset: engineered ZZ correlations, balanced 150/150"),
    ("pmlb_data", "appendicitis.csv", 106, 7,
     "small and narrow; cheapest full sweep in the pilot"),
    ("pmlb_data", "glass2.csv", 163, 9,
     "upper statevector band; 9 qubits"),
    ("qdata/x_view", "te_n10_s4_seed0_tau1.csv", 400, 10,
     "QUANTUM dataset: 10 qubits, statevector band, balanced 200/200"),
    ("libsvm_data", "heart.csv", 270, 13,
     "top of the statevector band; past the cache cliff, yet ~4x cheaper than MPS"),
    ("pmlb_data", "labor.csv", 57, 16,
     "deep MPS band at only 57 rows; the cheap MPS canary"),
    ("pmlb_data", "hepatitis.csv", 155, 19,
     "top of the MPS band; infeasible on a statevector, so this is the MPS test"),
    ("pmlb_data", "spect.csv", 267, 22,
     "just past the embedding threshold; 22 -> 8 components"),
    ("openMLCC18", "wdbc.csv", 569, 30,
     "widest openML case in the pilot; 30 -> 8 components"),
    ("libsvm_data", "sonar.csv", 208, 60,
     "60 -> 8 components, a 7.5x reduction"),
    ("libsvm_data", "colon_cancer.csv", 62, 2000,
     "extreme p >> n (2000 features, 62 rows); 2000 -> 8 components"),
]

CLASSICAL_MODELS = "['lr', 'svc', 'nb', 'dt', 'rf', 'xgb', 'catboost', 'mlp', 'tabpfn']"
QUANTUM_MODELS = "['qsvc', 'pqk', 'qnn', 'vqc']"
N_MODELS = 13   # 9 classical + 4 quantum; n_jobs is capped at this downstream anyway

TEMPLATE = r"""
## @@TITLE@@
## @@SHAPE@@
## @@WHY@@
## @@BANDNOTE@@
##
## Generated by generate_pilot_configs.py@@LAYOUTFLAG@@ -- regenerate rather than editing by hand.

# Name for this run. Used in output paths and in the hydra run directory below.
config_file_name: '@@NAME@@'

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
# Directory holding the CSVs. Must be absolute: hydra changes the working directory.
folder_path: '@@FOLDER@@'
# Which CSVs in folder_path to run. Names must include the .csv extension.
file_dataset: ['@@CSV@@']
# True if column 0 is a row-name/ID column to be used as the index and not a feature.
index_col: False

# ---------------------------------------------------------------------------
# Quantum execution backend
# ---------------------------------------------------------------------------
# Where the quantum circuits run. Options:
#   'statevector_simulator'  exact, noiseless, local. Reference Qiskit primitives.
#   'mps_simulator'          Aer matrix-product-state. Exact while entanglement is
#                            bounded; the affordable choice from ~10 qubits up.
#   'ibm_<name>'             real hardware, e.g. 'ibm_cleveland'; 'ibm_least' picks
#                            the least busy device. Needs qiskit_json_path below.
# This dataset uses @@QUBITS@@ qubits -> @@BACKEND_REASON@@
backend: '@@BACKEND@@'

# IBM Quantum credentials. Only read when backend starts with 'ibm'.
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
# TabPFN API token, as {"token": "..."}. Only read when 'tabpfn' is in classical_model.
tabpfn_json_path: '~/.config/qbiocode/tabpfn.json'

# Worker processes for the model loop. Capped at the number of models downstream, so
# @@NMODELS@@ is the useful maximum here. Do NOT use -1: each worker starts its own BLAS
# pool, and -1 workers x one pool each oversubscribes the host. Ask for the slots you
# were given and keep the per-process thread count at 1 (see @@SUBMIT@@).
n_jobs: @@NMODELS@@

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Dimensionality reductions to run, each producing a separate set of results.
# Options: 'none', 'pca', 'umap', 'tsne', 'isomap', 'lle', 'mds', 'spectral',
#          'kernelpca', 'factor', 'nmf', 'quantum'. 'none' uses the raw features.
embeddings: @@EMBEDDINGS@@
# A dataset is embedded only when its feature count EXCEEDS this value; at or below it
# the embedding list collapses to a single unembedded pass. 20 = the MPS qubit ceiling.
embedding_min_features: @@MINFEAT@@
# Output width of every embedding. 8 keeps wide datasets inside the statevector band.
n_components: @@NCOMP@@
# Every embedding but 'none' is read from here, one file per (dataset, embedding, split),
# and none is computed in the job: seeded UMAP differs between CPU types, so jobs that
# embed for themselves score one arm's models on different features depending on the
# host they land on. @@SUBMIT@@ writes the files before it submits, with
#     python -m qbiocode.apps.qprofiler.embedding_cache <these configs>
# and a job whose files are missing or were written under other settings stops before
# fitting anything. null embeds in each job instead. A dataset that is not embedded
# reads nothing from here.
embedding_cache: @@EMBCACHE@@

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# The two lists are kept separate so the quantum/classical split is stated by the
# config rather than inferred from names. Internally they are concatenated into
# 'model'; do not also set 'model' here, it is derived.
# classical_model options: 'lr' 'svc' 'nb' 'dt' 'rf' 'xgb' 'catboost' 'mlp' 'tabpfn'
classical_model: @@CLASSICAL@@
# quantum_model options: 'qsvc' 'pqk' 'qnn' 'vqc' 'qpl'
#   qsvc  quantum-kernel SVC (sampler)      pqk  projected quantum kernel + classical SVC
#   qnn   EstimatorQNN classifier           vqc  variational classifier (sampler)
quantum_model: @@QUANTUM@@

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# Averaging for f1 and precision/recall. Options:
#   'binary'   score the positive class only. Valid here: targets are ordinal-encoded
#              to {0, 1}, so the positive class is 1.
#   'weighted' per-class scores weighted by support (default here; robust to imbalance)
#   'macro'    unweighted mean over classes     'micro' pooled over all samples
#   None       return the per-class vector instead of a scalar
average: 'weighted'
# Multi-class strategy for roc_auc. Options: 'raise', 'ovr', 'ovo'.
# Inert in practice: qprofiler rejects any dataset with more than two classes, so this
# never fires. Left at 'raise' to keep it that way.
multi_class: 'raise'

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed: 42          # classical RNG: splits, model init, tuner sampling
q_seed: 42        # quantum RNG: qiskit algorithm_globals
shots: 1024       # measurement samples per circuit -- on the statevector backend TOO. Its
                  # sampler (qiskit StatevectorSampler, V2) has no analytic mode, so the
                  # sampler arms, qsvc's fidelity kernel and vqc's class probabilities,
                  # are 1024-shot estimates here exactly as on MPS. The estimator arms,
                  # qnn and pqk, are exact expectation values on the statevector.
resil_level: 1    # error mitigation level on IBM hardware (0-2). Ignored on simulators.

# ---------------------------------------------------------------------------
# Train/test protocol
# ---------------------------------------------------------------------------
# Test fraction. Two pressures point in opposite directions, so this is a compromise
# rather than an optimum, and the binding one differs by dataset.
#   DOWNWARD: the corrected resampled t-test has a variance floor 1.96*sqrt(r)*s with
#   r = test_size/(1-test_size). No value of `iter` crosses it. Holding that floor under
#   an epsilon of 0.05 requires test_size <= 0.2065 when the per-iteration scatter s is
#   0.05 -- and s is an assumption here, not a measurement; this pilot is what measures
#   it. The value 0.21 used earlier came from exactly this calculation rounded to two
#   decimals, but rounding UP crosses the line it was derived from: the floor at 0.21 is
#   0.0505, marginally above the epsilon it was chosen to respect.
#   UPWARD: balanced accuracy on a test set holding m minority rows can only move in
#   steps of 0.5/m. On the smallest datasets here -- labor (20 minority rows),
#   appendicitis (21), colon_cancer (22) -- that step is 0.100 to 0.125 for ANY
#   test_size <= 0.30. There epsilon = 0.05 sits below the metric's own resolution, and
#   no choice of test_size or iter fixes it; only more data would. Those datasets are
#   granularity-limited, and analyze_pilot.py flags them so their intervals are not read
#   as if they carried that precision.
# 0.20 satisfies the floor criterion (floor 0.0490 at s=0.05) and divides evenly.
test_size: @@TEST_SIZE@@
# Stratify the split. Options: ['y'] to preserve class balance, [] for none.
stratify: ['y']
# Feature scaling. Options: ['True'] or true (MinMaxScaler), 'StandardScaler',
# 'MinMaxScaler', 'None'/false. MinMaxScaler is what the quantum feature maps want:
# it bounds features, and every feature becomes a rotation angle.
scaling: ['True']
# Independent train/test resamples. Each gets a different split; results are reported
# per iteration so the resampled t-test has something to work with. Raising 5 -> 10
# narrows every per-dataset interval by ~28% for twice the compute. Most of that gain is
# NOT the 1/I term -- r already dominates it at this test_size -- but the t-quantile,
# which falls from 2.776 to 2.262 as df goes 4 -> 9. The pilot runs at 5 because its job
# is to surface plumbing failures; the full benchmark runs at 10.
iter: @@ITER@@
validation_split: 0.25   # held-out fraction inside the training set, for tuning
cross_validation: 5      # CV folds used by the classical tuner's objective

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
# Master switch. When True, a model's results are labelled '<name>_opt' and REPLACE
# the untuned results -- so every arm in the comparison is tuned, not one of each.
grid_search: True
# Search strategy. Options: 'optuna' (sampled; honours n_trials) or 'grid'
# (exhaustive over the lists, ignores n_trials).
tuner: optuna
# Trials per classical model. Automatically reduced to the size of the search space
# when that space is finite, so a small block does not waste trials re-sampling.
n_trials: 50
# Tune the quantum models too. Without this the comparison is tuned-classical versus
# default-quantum, which is not a fair comparison.
tune_quantum: True
# Trials per quantum model. Together with freeze_quantum_params below, this is the
# dominant term in the wall clock -- see N_TRIALS_QUANTUM_DEFAULT in the generator.
n_trials_quantum: @@NTQ@@
# Tune once, on iteration 0, then reuse those parameters for the remaining iterations.
# This is what makes tuning the quantum arm affordable: cost per arm becomes
# n_trials_quantum + (iter - 1) fits instead of iter x n_trials_quantum.
# It is OPTIMISTIC for the quantum side, not a handicap: the resamples overlap (73-87% of
# a later iteration's test rows were iteration-0 training rows), while the classical arm
# re-tunes every iteration. A quantum win measured this way is not a lower bound; set
# False (or use disjoint folds) for confirmatory runs.
freeze_quantum_params: True
# Where the frozen parameters are cached. Per-config, so two configs cannot collide.
# The cache key includes the backend, so switching simulators re-tunes instead of
# silently reusing parameters chosen on a different one.
quantum_param_dir: '@@PARAMDIR@@'

# Where the fidelity Gram matrices are kept. QSVC hands its quantum kernel to libsvm as a
# callable, and sklearn does not retain what a callable returns -- so without this the most
# expensive object in the run (n_tr(n_tr-1)/2 + n_te*n_tr circuits, the arm that sets this
# job's wall clock) is built, used once and freed. Recording it adds no circuit; it only
# stops discarding what the fit already computed. Cost is one npz per arm, ~2.6 MB at the
# widest split here. Without it, kernel-target alignment, geometric separation against the
# classical kernel and the RKHS margin are unrecoverable at any price short of re-running.
# PQK writes here too, as proj_*.npz. Its kernel is cheap to recompute -- the head is an SVC
# over the cached .npy projections -- but recomputing it needs three things the cache does not
# hold: the labels in the row order of the projections (alignment is meaningless without
# them), X_train for the classical side of g(K_c || K_q), and the SVC's chosen kernel. That
# head searches kernel over ['linear','rbf','poly','sigmoid'], so it is NOT always RBF; a
# smoke run picked 'poly'. The projection cache is also keyed by a sha256 over the feature
# map AND dataset_fingerprint(X_train, X_test), so locating the right .npy after the fact
# requires already holding the exact split. The dump stores best_params alongside, so the
# kernel is rebuilt as fitted instead of assumed.
kernel_dump_dir: '@@KERNELDIR@@'

# ---------------------------------------------------------------------------
# Classical model defaults, used when grid_search is False
# ---------------------------------------------------------------------------
svc_args: { 'C': 0.01, 'gamma': 0.1, kernel: 'linear' }
dt_args: {'criterion': 'gini',
          'splitter': 'best',
          'max_depth': null,
          'min_samples_split': 2,
          'min_samples_leaf': 1
          }
nb_args: {'var_smoothing': 1.e-09}
lr_args: {'penalty': 'l2',
          'C': 1.0,
          'solver': 'liblinear',
          'max_iter': 10000
          }
rf_args: {'n_estimators': 100,
          'max_features': 'sqrt',
          'max_depth': null,
          'min_samples_split': 2,
          'min_samples_leaf': 1,
          'bootstrap': True
          }
mlp_args: {'hidden_layer_sizes': 100,
           'activation': 'relu',
           'max_iter': 10000,
           'solver': 'adam',
           'alpha': 0.0001,
           'learning_rate': 'constant',
           }
# nthread is XGBoost's own name for the cap the sklearn wrapper spells 'n_jobs'; both are
# accepted, and it is written this way here so it cannot be read as the top-level n_jobs
# above, which sizes the joblib fan-out instead. Unset, XGBoost takes omp_get_max_threads()
# -- so unlike CatBoost above, OMP_NUM_THREADS=1 from submit_pilot.sh does bound it. This
# pin is the inner belt: it makes the config correct on its own, for a qprofiler run
# launched by hand without that export. Measured on a 128-core node, one 42-row fit with no
# cap ran over 280 s and was killed; capped it took 0.06 s.
xgb_args: {'n_estimators': 200,
           'max_depth': 6,
           'learning_rate': 0.1,
           'subsample': 0.8,
           'colsample_bytree': 0.8,
           'min_child_weight': 1,
           'nthread': 1,
           }
# Parameters left out are left at CatBoost's own defaults, which it derives from the
# data and from each other -- naming one can change more than that one value.
# thread_count: 1 is not optional here. Unset, CatBoost takes every core, and it uses
# its own thread pool -- so OMP_NUM_THREADS does not bound it the way it bounds xgboost,
# torch and BLAS. With n_jobs workers each running one model, unset means n_jobs x ncores.
# 'verbose' is deliberately NOT set here. CatBoost's training chatter is already silenced
# by _QUIET inside compute_catboost, and model_run splats this block into the same kwarg
# namespace as its own 'verbose' argument (which selects the result summary) -- so naming
# it raised TypeError while the delayed() list was built, killing all 13 models before any
# of them fit. model_run now drops it with a warning; leaving it out keeps the log clean.
catboost_args: {'iterations': 200,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3.0,
                'thread_count': 1,
                }
tabpfn_args: {'model_version': 'v2',
              'n_estimators': 4,
              'softmax_temperature': 0.9,
              'balance_probabilities': False,
              'average_before_softmax': False,
              }

# ---------------------------------------------------------------------------
# Classical search spaces, used when grid_search is True
# ---------------------------------------------------------------------------
# A list is a choice; a {low, high} mapping is a range ('log: true' samples it
# logarithmically). A block of pure lists has a finite size, and n_trials is reduced
# to it; one range makes the space infinite and all n_trials are used.
gridsearch_svc_args: {'C': {low: 1.0e-3, high: 1.0e+2, log: true},
                      'gamma': {low: 1.0e-4, high: 1.0e+1, log: true},
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                      }
gridsearch_dt_args: {'criterion': ['gini', 'entropy', 'log_loss'],
                     'splitter': ['best', 'random'],
                     'max_depth': [3, 5, 10, 20, null],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4]
                     }
gridsearch_nb_args: {'var_smoothing': [1.e-09, 1.e-08, 1.e-07, 1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02]}
gridsearch_lr_args: {'penalty': ['l1', 'l2'],
                     'C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03],
                     'solver': ['liblinear', 'saga'],
                     'max_iter': [5000, 10000]
                     }
gridsearch_rf_args: {'n_estimators': [10, 50, 100, 200],
                     'max_features': ['sqrt', 'log2'],
                     'max_depth': [10, 20, 30, null],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'bootstrap': [True, False]
                     }
gridsearch_mlp_args: {'hidden_layer_sizes': [[20,],[50,],[100,]],
                      'activation': ['tanh', 'relu'],
                      'max_iter': [5000, 10000],
                      'solver': ['sgd', 'adam'],
                      'alpha': [0.0001, 0.05],
                      'learning_rate': ['constant','adaptive'],
                      }
# nthread is pinned for every trial, for the reason given on xgb_args. It has to be
# repeated here: grid_search: True replaces the untuned path rather than adding to it, so
# with tuning on this block is the only one model_run reads for xgb.
gridsearch_xgb_args: {'n_estimators': [100, 200, 400],
                      'max_depth': [3, 6, 9],
                      'learning_rate': {low: 1.0e-2, high: 3.0e-1, log: true},
                      'subsample': {low: 0.6, high: 1.0},
                      'colsample_bytree': {low: 0.6, high: 1.0},
                      'min_child_weight': [1, 3, 5],
                      'nthread': [1],
                      }
# thread_count is pinned for every trial, for the reason given on catboost_args.
gridsearch_catboost_args: {'iterations': [100, 200, 400],
                           'depth': [4, 6, 8],
                           'learning_rate': {low: 1.0e-2, high: 3.0e-1, log: true},
                           'l2_leaf_reg': {low: 1.0, high: 1.0e+1, log: true},
                           'thread_count': [1],
                           }
gridsearch_tabpfn_args: {'model_version': 'v2',
                         'n_estimators': [1, 2, 4, 8],
                         'softmax_temperature': [0.75, 0.9, 1.0],
                         'balance_probabilities': [True, False],
                         'average_before_softmax': [True, False],
                         }

# ---------------------------------------------------------------------------
# Quantum model defaults, used when tune_quantum is False
# ---------------------------------------------------------------------------
# Shared keys and their options:
#   encoding      'Z'  ZFeatureMap, product state, no entangling gates
#                 'ZZ' ZZFeatureMap, pairwise entangling phases
#                 'P'  PauliFeatureMap, arbitrary Pauli strings
#   entanglement  'linear' 'full' 'reverse_linear' 'pairwise' 'circular' 'sca'
#                 Two-qubit gate count per rep: linear/pairwise ~q, full ~q(q-1)/2.
#                 On mps_simulator this is the knob that decides cost: 'full' grows the
#                 bond dimension, measured at 44.6 s/circuit at 16 qubits against 0.049
#                 for linear (x910) and 0.60 for the same circuit on a statevector.
#                 That is why the tuning space below omits 'full' on mps_simulator; do
#                 not add it back for an MPS-band dataset.
#   reps          feature-map repetitions. Depth and expressivity scale with it; so
#                 does runtime, linearly.
#   primitive     'sampler' (measurement probabilities) or 'estimator' (expectations)
#   ansatz_type   'amp' RealAmplitudes, 'esu2' EfficientSU2, 'twolocal' TwoLocal
#   local_optimizer  'COBYLA' 'L_BFGS_B' 'SPSA' 'GradientDescent'
#   maxiter       optimizer iterations. For qnn/vqc the circuit count is
#                 maxiter x n_train, which makes these the most expensive arms.
qnn_args: {'primitive': 'estimator',
           'local_optimizer': 'COBYLA',
           'encoding': ZZ,
           'entanglement': 'linear',
           'reps': 2,
           'maxiter': 100,
           'ansatz_type': 'amp'
           }
qsvc_args: {'C': 0.01,
            'pegasos': False,
            'encoding': ZZ,
            'entanglement': 'linear',
            'reps': 2,
            'primitive': 'sampler',
            }
vqc_args: {'primitive': 'sampler',
           'local_optimizer': 'COBYLA',
           'maxiter': 100,
           'encoding': ZZ,
           'entanglement': 'linear',
           'reps': 2,
           'ansatz_type': 'amp'
           }
pqk_args: {'encoding': ZZ,
           'entanglement': 'pairwise',
           'primitive': 'estimator',
           'reps': 4
           }

# ---------------------------------------------------------------------------
# Quantum search spaces, used when grid_search AND tune_quantum are both True
# ---------------------------------------------------------------------------
# Same syntax as the classical blocks. Every trial is a full quantum fit, so the
# per-model trial budget is spent where it buys the most:
#   reps is tuned over a real range here, not just [1, 2] -- circuit depth is as
#   influential as the encoding, and for pqk it is close to free (its circuit count is
#   linear in rows and its projections are cached), so pqk explores reps furthest.
#   qnn/vqc stop at 3: their circuit count is maxiter x n_train x depth.
gridsearch_qsvc_args: {'encoding': ['Z', 'ZZ', 'P'],
                       'reps': [1, 2, 3, 4],
                       'entanglement': @@ENTSPACE@@,
                       'C': {low: 1.0e-2, high: 1.0e+2, log: true}
                      }
gridsearch_pqk_args: {'encoding': ['Z', 'ZZ', 'P'],
                      'reps': [1, 2, 3, 4, 6],
                      'entanglement': @@ENTSPACE@@
                      }
#   qnn pins 'primitive' as a one-element list on purpose: the tuned path builds its
#   kwargs from gridsearch_<model>_args only and never reads qnn_args, and qnn is the one
#   arm whose qnn_args value ('estimator') differs from its function default ('sampler').
#   local_optimizer is pinned to COBYLA the same way. L_BFGS_B is gradient-based, and with
#   no gradient supplied qiskit builds a parameter-shift one: 2 x n_params circuits per row
#   per step instead of 1. Measured on this path at 8 qubits, 40 rows, amp reps=2
#   (2026-09-28): vqc 16.8 s/iteration against 0.029 for COBYLA (x580), qnn 8.3 against
#   0.043 (x190). One L_BFGS_B trial on wdbc is then hours, and the first pilot spent its
#   whole first hour inside single trials, so do not add it back without pricing it.
gridsearch_qnn_args: {'encoding': ['Z', 'ZZ'],
                      'reps': [1, 2, 3],
                      'ansatz_type': ['amp', 'esu2'],
                      'local_optimizer': ['COBYLA'],
                      'primitive': ['estimator'],
                      'maxiter': {low: 50, high: 200}
                      }
gridsearch_vqc_args: {'encoding': ['Z', 'ZZ'],
                      'reps': [1, 2, 3],
                      'ansatz_type': ['amp', 'esu2'],
                      'local_optimizer': ['COBYLA'],
                      'maxiter': {low: 50, high: 200}
                      }

# ---------------------------------------------------------------------------
# Output location. Absolute, because hydra changes the working directory.
# ---------------------------------------------------------------------------
hydra:
  run:
    dir: @@RUNDIR@@/results/${config_file_name}/${backend}_${now:%Y-%m-%d_%H-%M-%S}
""".lstrip("\n")

BAND_NOTE = {
    "statevector_simulator": (
        "{q} qubits, under the {lim}-qubit statevector limit -> exact statevector."
    ),
    "mps_simulator": (
        "{q} qubits, inside the 10-{mps} band -> Aer MPS "
        "(a dense statevector of 2**{q} amplitudes is the thing being avoided)."
    ),
}
EMBED_NOTE = (
    "{w} features > {minf}, so PCA and UMAP reduce to {nc} components "
    "-> {nc} qubits -> exact statevector."
)


def split_jobs(feats):
    """The (embedding, model) pairs of one dataset under --layout split, heaviest first.

    Every model on every embedding the combined config would run -- no more, no fewer.
    """
    _, _, embeddings = backend_for(feats)
    models = ast.literal_eval(QUANTUM_MODELS) + ast.literal_eval(CLASSICAL_MODELS)
    return [(emb, m) for emb in ast.literal_eval(embeddings) for m in models]


#: The composed split layout's shared config, one per dataset directory. Hydra finds it
#: through each job file's defaults list; status.py and collate_results.py skip every
#: _-prefixed YAML as not a job, and submit_runs.sh submits only what MANIFEST.tsv lists.
PROTOCOL = "_protocol"
#: What the protocol holds where a job file must set the value: OmegaConf's MISSING, so a
#: job file that forgets a key -- or the protocol run by itself -- stops at it.
UNSET = "???"
#: The template tokens that say which job a split config is, and where it writes.
JOB_TOKENS = ("@@NAME@@", "@@EMBEDDINGS@@", "@@CLASSICAL@@", "@@QUANTUM@@",
              "@@PARAMDIR@@", "@@KERNELDIR@@")

JOB_TEMPLATE = r"""
## @@TITLE@@
## Only which job this is and where it writes. Everything else -- data, split, seeds,
## trial budgets, search spaces, output directory -- is this dataset's shared protocol,
## @@PROTOCOL@@.yaml beside this file; the keys below fill in the ones it leaves @@UNSET@@.
## Generated by generate_pilot_configs.py --layout split -- regenerate rather than editing by hand.
defaults:
  - @@PROTOCOL@@
  - _self_          # this file last, so its keys override the protocol's

config_file_name: '@@NAME@@'
embeddings: @@EMBEDDINGS@@
classical_model: @@CLASSICAL@@
quantum_model: @@QUANTUM@@
quantum_param_dir: '@@PARAMDIR@@'
kernel_dump_dir: '@@KERNELDIR@@'
""".lstrip("\n")


def _job_title(idx, folder, csv, emb, model):
    return (f"PILOT {idx:02d} of {len(DATASETS)} -- {folder}/{csv} -- ONE JOB: "
            f"{model} on embedding '{emb}'")


def _job_values(dataset, feats, emb, model, runs_dir):
    """{token: text} for the JOB_TOKENS of the split-layout job (emb, model)."""
    _, _, embeddings = backend_for(feats)
    if emb not in ast.literal_eval(embeddings):
        raise ValueError(f"{dataset} does not run embedding {emb!r}; it runs {embeddings}")
    classical, quantum = ast.literal_eval(CLASSICAL_MODELS), ast.literal_eval(QUANTUM_MODELS)
    if model not in classical + quantum:
        raise ValueError(f"unknown model {model!r}")
    name = f"{dataset}_{emb}_{model}"
    # Everything this job writes lives under its dataset's directory, and the two caches
    # are per CONFIG, not per dataset: the tuner's trials dump with an empty data_key
    # (proj_pqk_.npz), so the pca and umap pqk jobs of one dataset would otherwise write
    # the same file concurrently. No two jobs share a writable path.
    rundir = os.path.join(runs_dir, dataset)
    return dict(zip(JOB_TOKENS, (
        name,
        f"['{emb}']",
        f"['{model}']" if model in classical else "[]",
        f"['{model}']" if model in quantum else "[]",
        os.path.join(rundir, "quantum_tuned_params", name),
        os.path.join(rundir, "kernels", name),
    )))


def build_job(idx, folder, csv, feats, emb, model, runs_dir=RUNS_DIR):
    """The composed split layout's job file for (emb, model): the job keys, over PROTOCOL."""
    body = JOB_TEMPLATE
    for token, value in [("@@TITLE@@", _job_title(idx, folder, csv, emb, model)),
                         ("@@PROTOCOL@@", PROTOCOL), ("@@UNSET@@", UNSET),
                         *_job_values(csv[:-4], feats, emb, model, runs_dir).items()]:
        body = body.replace(token, value)
    assert "@@" not in body, f"unsubstituted token in {csv[:-4]}_{emb}_{model}"
    return f"{csv[:-4]}_{emb}_{model}", body


def load_config(path):
    """A config as hydra composes it, its own ``hydra`` node included and nothing resolved.

    A composed job file's defaults list is merged in order, _self_ where it is listed (last
    when it is not, hydra 1.1's rule). Only what this generator writes is understood --
    names of configs in the same directory, and _self_ -- and anything else is refused
    rather than guessed at. A config without a defaults list comes back as it is.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    layers = []
    for entry in defaults:
        if entry == "_self_":
            layers.append(cfg)
        elif isinstance(entry, str) and entry and not set(entry) & set("/@:= "):
            layers.append(load_config(os.path.join(os.path.dirname(path), f"{entry}.yaml")))
        else:
            raise ValueError(f"{path}: defaults entry {entry!r} is not a same-directory "
                             f"config name, which is all load_config understands")
    if "_self_" not in defaults:
        layers.append(cfg)
    return OmegaConf.merge(*layers)


def build(idx, folder, csv, rows, feats, why, n_iter=ITER_DEFAULT,
          test_size=TEST_SIZE_DEFAULT, n_trials_quantum=N_TRIALS_QUANTUM_DEFAULT,
          emb=None, model=None, runs_dir=RUNS_DIR, protocol=False,
          embedding_cache=EMBEDDING_CACHE_DIR):
    """One config. With ``model`` (and ``emb``) set, the self-contained split-layout job for
    that pair. With ``protocol``, the dataset's PROTOCOL for build_job's files: the same
    text as any of its jobs, with the job keys left UNSET. ``embedding_cache`` is the
    directory the jobs read their embedded features from, or None to embed in each job."""
    if embedding_cache and not os.path.isabs(embedding_cache):
        # qprofiler refuses a relative one: every job runs from its own output directory.
        raise ValueError(f"embedding_cache must be an absolute path, got {embedding_cache!r}")
    split = model is not None or protocol
    dataset = csv[:-4]
    backend, qubits, embeddings = backend_for(feats)
    if split:
        if protocol:
            job = dict.fromkeys(JOB_TOKENS, UNSET)
            name = PROTOCOL
            title = (f"PILOT {idx:02d} of {len(DATASETS)} -- {folder}/{csv} -- SHARED PROTOCOL "
                     f"of the {len(split_jobs(feats))} jobs in this directory.\n"
                     f"## Each job file here pulls this in (defaults: [{PROTOCOL}, _self_]) and "
                     f"sets the keys\n## left {UNSET} below: which job it is and where it "
                     f"writes. Run a job file, never this one.")
        else:
            job = _job_values(dataset, feats, emb, model, runs_dir)
            name = job["@@NAME@@"]
            title = _job_title(idx, folder, csv, emb, model)
        n_models = 1
        rundir = os.path.join(runs_dir, dataset)
        layout_flag, submit = " --layout split", "submit_runs.sh"
    else:
        name = f"pilot{idx:02d}_{dataset}"
        job = dict(zip(JOB_TOKENS, (name, embeddings, CLASSICAL_MODELS, QUANTUM_MODELS,
                                    f"{HERE}/quantum_tuned_params/{name}",
                                    f"{HERE}/kernels/{name}")))
        n_models = N_MODELS
        rundir = HERE
        title = f"PILOT {idx:02d} of {len(DATASETS)} -- {folder}/{csv}"
        layout_flag, submit = "", "submit_pilot.sh"
    embedded = feats > EMBEDDING_MIN_FEATURES
    if embedded:
        bandnote = EMBED_NOTE.format(w=feats, minf=EMBEDDING_MIN_FEATURES, nc=N_COMPONENTS)
        reason = (
            f"embedded from {feats} features to {N_COMPONENTS} components, "
            f"so the circuits are {N_COMPONENTS}-qubit and exact."
        )
    else:
        bandnote = BAND_NOTE[backend].format(
            q=qubits, lim=STATEVECTOR_MAX_QUBITS, mps=MPS_MAX_QUBITS
        )
        reason = (
            "an exact statevector is still cheap at this width."
            if backend == "statevector_simulator"
            else "a dense statevector is no longer affordable at this width."
        )
    body = TEMPLATE
    for token, value in [
        ("@@TITLE@@", title),
        ("@@LAYOUTFLAG@@", layout_flag),
        ("@@SUBMIT@@", submit),
        ("@@SHAPE@@", f"{rows} rows x {feats} features, binary target, no missing values."),
        ("@@WHY@@", f"In the pilot because: {why}."),
        ("@@BANDNOTE@@", bandnote),
        ("@@FOLDER@@", f"{DATA_ROOT}/{folder}"),
        ("@@CSV@@", csv),
        ("@@BACKEND@@", backend),
        ("@@BACKEND_REASON@@", reason),
        ("@@QUBITS@@", str(qubits)),
        # Backend-dependent, for the feasibility reason recorded at
        # ENTANGLEMENT_BY_BACKEND: an MPS-band dataset must not be able to draw
        # 'full'. Keyed on the resolved backend, so a dataset that moves band
        # because its width changed cannot keep the wrong space.
        ("@@ENTSPACE@@", ENTANGLEMENT_BY_BACKEND[backend]),
        ("@@MINFEAT@@", str(EMBEDDING_MIN_FEATURES)),
        ("@@NCOMP@@", str(N_COMPONENTS)),
        ("@@EMBCACHE@@", f"'{embedding_cache}'" if embedding_cache else "null"),
        ("@@NMODELS@@", str(n_models)),
        ("@@RUNDIR@@", rundir),
        ("@@ITER@@", str(n_iter)),
        ("@@TEST_SIZE@@", f"{test_size:g}"),
        ("@@NTQ@@", str(n_trials_quantum)),
        *job.items(),
    ]:
        body = body.replace(token, value)
    assert "@@" not in body, f"unsubstituted token in {name}: {body[body.index('@@'):][:40]}"
    return name, body


def main():
    ap = argparse.ArgumentParser(
        description="Generate one qprofiler config per pilot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # The pilot exists to catch plumbing failures, so it runs at 5 resamples. The full
    # benchmark runs at 10: the intervals come out ~28% narrower for twice the compute.
    ap.add_argument("--iter", type=int, default=ITER_DEFAULT, dest="n_iter",
                    help="independent train/test resamples per dataset")
    ap.add_argument("--test-size", type=float, default=TEST_SIZE_DEFAULT,
                    help="test fraction; see the rationale in the template")
    # Deadline knob. Per-arm cost is (n_trials_quantum + iter) fits, so this scales the
    # run almost linearly while --iter barely moves it. Lower this to fit a wall, not
    # --iter: the resamples are what the t-test's degrees of freedom are made of.
    ap.add_argument("--n-trials-quantum", type=int, default=N_TRIALS_QUANTUM_DEFAULT,
                    dest="n_trials_quantum",
                    help="optuna trials per quantum model; sets the wall clock")
    # Deadline mode. The 12 datasets run as independent LSF jobs, so the pilot's wall
    # clock is the SLOWEST job, not the total -- dropping datasets buys nothing. What does
    # buy time is giving each dataset the largest trial budget that fits, which differs by
    # two orders of magnitude across this list: colon_cancer is 62 rows and wdbc is 569,
    # and qsvc cost is quadratic in rows. A global budget would have to be set by wdbc and
    # would starve the other eleven for nothing.
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="per-job wall-clock budget; picks n_trials_quantum per dataset "
                         "from the measured cost model and overrides --n-trials-quantum")
    # Off by default because the shipped pilot was sized without it: the budget is checked
    # against the kernel arms' ZZ/linear price, which the whole encoding x entanglement grid
    # exceeds by 10-17% at 8-13 qubits (cost_model.grid_factor). The table prints both.
    ap.add_argument("--price-grid", action="store_true",
                    help="check --budget-hours against the search grid's expected cost "
                         "instead of its ZZ/linear price")
    ap.add_argument("--config-dir", default=CONFIG_DIR,
                    help="where to write the YAMLs (--layout combined)")
    # Same experiment, finer jobs: see the module docstring. The trial budget is chosen
    # exactly as for the combined layout -- including its x n_embeddings -- so a split
    # config differs from its combined parent only in model, embedding, n_jobs and paths.
    ap.add_argument("--layout", choices=("combined", "split"), default="combined",
                    help="one job per dataset, or one per (dataset, embedding, model)")
    ap.add_argument("--runs-dir", default=RUNS_DIR,
                    help="root of the per-dataset directories (--layout split)")
    ap.add_argument("--self-contained", action="store_true",
                    help="--layout split: write each job as one full config, instead of a "
                         f"job file over its dataset's {PROTOCOL}.yaml")
    ap.add_argument("--embedding-cache", default=EMBEDDING_CACHE_DIR, metavar="DIR",
                    help="where the jobs read their embedded features, written by the "
                         "submit script before it submits; '' writes null, so each job "
                         "embeds for itself")
    args = ap.parse_args()
    # Absolute, because qprofiler requires it: each job runs from its own output directory.
    args.embedding_cache = os.path.abspath(args.embedding_cache) if args.embedding_cache else None

    if not 0.0 < args.test_size < 1.0:
        ap.error(f"--test-size must lie in (0, 1), got {args.test_size}")
    if args.n_trials_quantum < 2:
        # One trial is not a search: optuna would fit once, and that single sample would
        # be reported as "tuned" against a classical arm that got n_trials=50. Refusing
        # here keeps the comparison from being quietly unfair.
        ap.error(
            f"--n-trials-quantum must be at least 2 to be a search at all, got "
            f"{args.n_trials_quantum}. To run the quantum arm untuned, set "
            f"tune_quantum: False in the template instead."
        )
    if args.budget_hours is not None and args.budget_hours <= 0:
        ap.error(f"--budget-hours must be positive, got {args.budget_hours}")
    if args.n_iter < 2:
        # n-1 degrees of freedom: one resample gives no scatter to estimate s from, and
        # the t-test is undefined. Fail here rather than after a 20-hour run.
        ap.error(f"--iter must be at least 2 for a resampled t-test, got {args.n_iter}")

    split = args.layout == "split"
    out_dir = args.runs_dir if split else args.config_dir
    if split:
        # Only the YAMLs at the top of each dataset directory are ours to replace. The
        # results/, kernels/, quantum_tuned_params/ and lsf_logs/ beside them are run
        # output and are never touched here.
        for _, csv, _, _, _ in DATASETS:
            ds_dir = os.path.join(args.runs_dir, csv[:-4])
            os.makedirs(ds_dir, exist_ok=True)
            for stale in sorted(os.listdir(ds_dir)):
                if stale.endswith((".yaml", ".yml")):
                    os.remove(os.path.join(ds_dir, stale))
    else:
        os.makedirs(args.config_dir, exist_ok=True)
        for stale in sorted(os.listdir(args.config_dir)):
            if stale.endswith((".yaml", ".yml")):
                os.remove(os.path.join(args.config_dir, stale))
    cm = _load_cost_model() if args.budget_hours is not None else None
    # The manifest's per-job hours are priced whether or not --budget-hours chose the
    # trials: submit_runs.sh orders jobs by them and status.py compares elapsed against them.
    cm_price = cm if cm is not None else (_load_cost_model() if split else None)
    manifest = []
    head = (f"iter={args.n_iter}  test_size={args.test_size:g}  ")
    if cm is None:
        print(head + f"n_trials_quantum={args.n_trials_quantum}  ->  {out_dir}")
    else:
        print(head + f"budget={args.budget_hours:g} h/job  ->  {out_dir}")
    # pred h is what the budget is checked against; exp h prices the whole search grid at its
    # mean; bound h is the same job with its frozen winner at the grid's costliest point.
    print(f"{'config':40s} {'backend':22s} {'qubits':>6s} {'embed':>12s} "
          f"{'rows':>5s} {'trials':>6s} {'pred h':>7s} {'exp h':>6s} {'bound h':>7s} {'arm':>5s}")
    walls, overruns = [], []
    for i, (folder, csv, rows, feats, why) in enumerate(DATASETS, start=1):
        backend, qubits, emb = backend_for(feats)
        ntq, pred, exp_h, bound, arm = args.n_trials_quantum, None, None, None, ""
        if cm is not None:
            bk = "mps" if backend == "mps_simulator" else "sv"
            # backend_for returns the embedding list as the literal YAML text; 'pca' in
            # it means the pca/umap pair, which qprofiler runs sequentially.
            n_emb = 2 if "pca" in emb else 1
            grid = "mean" if args.price_grid else None
            ntq = cm.choose_trials(bk, qubits, n_emb, rows, args.n_iter,
                                   args.budget_hours, test_size=args.test_size, grid=grid)
            pred, exp_h, bound = (cm.wall_hours(bk, qubits, n_emb, rows, ntq, args.n_iter,
                                                test_size=args.test_size, grid=g)
                                  for g in (grid, "mean", "bound"))
            arm = cm.slowest_arm(bk, qubits, rows, ntq, args.n_iter,
                                 test_size=args.test_size)
            walls.append((exp_h, csv[:-4], ntq, bound))
            if pred > args.budget_hours:
                overruns.append((csv[:-4], pred, ntq))
        if split:
            bk = "mps" if backend == "mps_simulator" else "sv"
            quantum = ast.literal_eval(QUANTUM_MODELS)
            same = dict(n_iter=args.n_iter, test_size=args.test_size, n_trials_quantum=ntq,
                        runs_dir=args.runs_dir, embedding_cache=args.embedding_cache)
            if not args.self_contained:
                _, pbody = build(i, folder, csv, rows, feats, why, protocol=True, **same)
                with open(os.path.join(args.runs_dir, csv[:-4], f"{PROTOCOL}.yaml"), "w") as fh:
                    fh.write(pbody)
            for e, m in split_jobs(feats):
                if args.self_contained:
                    jname, jbody = build(i, folder, csv, rows, feats, why, emb=e, model=m,
                                         **same)
                else:
                    jname, jbody = build_job(i, folder, csv, feats, e, m,
                                             runs_dir=args.runs_dir)
                path = os.path.abspath(os.path.join(args.runs_dir, csv[:-4], f"{jname}.yaml"))
                with open(path, "w") as fh:
                    fh.write(jbody)
                # Only the quantum arms are priced: the cost model is calibrated for them,
                # and the classical arms are minutes against their hours.
                jexp = jbound = None
                if cm_price is not None and m in quantum:
                    jexp, jbound = (cm_price.arm_hours(m, bk, qubits, rows, ntq, args.n_iter,
                                                       test_size=args.test_size, grid=g)
                                    for g in ("mean", "bound"))
                manifest.append((jname, csv[:-4], e, m,
                                 "quantum" if m in quantum else "classical", backend,
                                 qubits, rows, ntq, args.n_iter, jexp, jbound, path))
            name = f"{csv[:-4]}/ ({len(split_jobs(feats))} jobs)"
        else:
            name, body = build(i, folder, csv, rows, feats, why,
                               n_iter=args.n_iter, test_size=args.test_size,
                               n_trials_quantum=ntq, embedding_cache=args.embedding_cache)
            with open(os.path.join(args.config_dir, f"{name}.yaml"), "w") as fh:
                fh.write(body)
            name += ".yaml"
        print(f"{name:40s} {backend:22s} {qubits:6d} {emb:>12s} "
              f"{rows:5d} {ntq:6d} {('%.2f' % pred) if pred else '':>7s} "
              f"{('%.2f' % exp_h) if exp_h else '':>6s} {('%.2f' % bound) if bound else '':>7s} "
              f"{arm:>5s}")
    if split:
        mpath = os.path.join(args.runs_dir, "MANIFEST.tsv")
        with open(mpath, "w") as fh:
            fh.write("config\tdataset\tembedding\tmodel\tarm\tbackend\tqubits\trows\t"
                     "n_trials_quantum\titer\texp_h\tbound_h\tyaml\n")
            for row in manifest:
                fh.write("\t".join("" if v is None else (f"{v:.3f}" if isinstance(v, float)
                                                          else str(v)) for v in row) + "\n")
        print(f"\n{len(manifest)} configs written under {args.runs_dir} "
              f"({len(DATASETS)} dataset directories"
              + ("" if args.self_contained else f", each with its {PROTOCOL}.yaml")
              + f"); manifest: {mpath}")
        priced = [r for r in manifest if r[10] is not None]
        if priced:
            # One arm on one embedding is now a whole job, so the pilot's wall clock is the
            # slowest single arm, and the LSF wall is sized on that arm's frozen-winner bound.
            slow = sorted(priced, key=lambda r: r[10], reverse=True)[:5]
            print("slowest jobs (expected h / bound h): "
                  + ", ".join(f"{r[0]} {r[10]:.2f}/{r[11]:.2f}" for r in slow))
            worst = max(priced, key=lambda r: r[11])
            print(f"suggested LSF wall: {math.ceil(worst[11])}:00 (frozen-winner bound of "
                  f"{worst[0]})")
        return
    print(f"\n{len(DATASETS)} configs written to {args.config_dir}")
    if walls:
        walls.sort(reverse=True)
        print(f"slowest job (expected): {walls[0][1]} at {walls[0][0]:.2f} h "
              f"(n_trials_quantum={walls[0][2]}); 2nd {walls[1][1]} {walls[1][0]:.2f} h")
        # The wall is a kill ceiling, not a reservation, so it is sized on the tail, not on
        # the mean -- and the tail is the frozen winner. Every resample reuses whatever the
        # search picked, so a job whose search lands on ZZ/P + full + max reps pays ~3x the
        # priced fit on every full fit; a flat 1.3x on the mean used to suggest 15:00 here,
        # which a pass-by-pass simulation of this pilot kills a job at 26-41% of the time.
        # A kill is not total loss: qprofiler appends ModelResults.csv after every
        # (iteration, embedding) pass (qprofiler.py:774), so only unfinished passes are lost.
        worst = max(walls, key=lambda w: w[3])
        print(f"suggested LSF wall: {math.ceil(worst[3])}:00 (the frozen-winner bound of "
              f"{worst[1]}: search at the grid mean, every full fit at its costliest point)")
        if not args.price_grid:
            over = [f"{w[1]} {w[0]:.2f} h" for w in walls if w[0] > args.budget_hours]
            if over:
                print("expected over budget once the whole grid is priced: "
                      + ", ".join(over) + "  (--price-grid budgets against that instead)")
    for csv, pred, ntq in overruns:
        # Says so rather than quietly handing back a budget that cannot be met.
        print(f"  WARNING {csv} needs {pred:.2f} h even at the ladder minimum "
              f"n_trials_quantum={ntq}, over the {args.budget_hours:g} h budget")


if __name__ == "__main__":
    main()
