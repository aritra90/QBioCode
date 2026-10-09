# QProfiler Configuration Guide

This guide explains how to configure QProfiler using YAML configuration files for reproducible experiments and batch processing.

## Overview

QProfiler uses YAML configuration files to define:
- Input datasets and output directories
- Machine learning models to evaluate
- Quantum backend settings
- Embedding methods and parameters
- Train/test split configuration
- Model hyperparameters

An example configuration file can be found at [`qbiocode/apps/qprofiler/configs/config.yaml`](https://github.com/qiskit-community/QBioCode/blob/main/qbiocode/apps/qprofiler/configs/config.yaml).

## Quick Start

Here's a minimal configuration to get started:

```yaml
# Basic configuration
config_file_name: 'my_experiment'
folder_path: 'data/'
file_dataset: 'my_dataset.csv'
seed: 42

# Models to evaluate
model: ['rf', 'svc', 'qsvc']

# Embedding
embeddings: ['none']
n_components: 3

# Train/test split
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Quantum backend (for QML models)
backend: 'simulator'
shots: 1024
```

---

## Configuration Sections

### Input Data

Specify the location and selection of input datasets.

**Single Dataset:**

```yaml
config_file_name: 'experiment_name'
folder_path: 'data/'
file_dataset: 'dataset.csv'
```

**All Datasets in Folder:**

```yaml
folder_path: 'data/'
file_dataset: 'ALL'  # Process all CSV files in folder
```

**Multiple Specific Datasets:**

```yaml
file_dataset: ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
```

**Output Directory:**

```yaml
output_dir: 'results/'  # Where to save results
```

### Random Seeds

Set random seeds for reproducibility:

```yaml
seed: 42      # Seed for classical ML algorithms
q_seed: 42    # Seed for quantum algorithms
```

```{tip}
Always set seeds for reproducible experiments. Use the same seed across runs to compare results.
```

### Quantum Backend Configuration

Configure quantum computing backend for QML models (QSVC, VQC, QNN, PQK).

**Simulator (Default):**

```yaml
backend: 'simulator'
shots: 1024
```

Uses the Qiskit statevector simulator for exact, noiseless quantum simulation.

**AerSimulator with Custom Simulation Method:**

```yaml
backend: 'simulator_aer'
sim_method: 'statevector'  # Options: statevector, matrix_product_state, tensor_network, etc.
shots: 1024
```

Provides access to AerSimulator's various simulation methods. Useful for:
- **GPU acceleration**: Use `sim_method: 'tensor_network'` with GPU support
- **Memory efficiency**: Use `sim_method: 'matrix_product_state'` for low-entanglement circuits
- **Clifford circuits**: Use `sim_method: 'stabilizer'` for fast simulation

**Noisy Simulation Based on IBM Device:**

```yaml
backend: 'noisy_ibm_cleveland'  # Noisy simulation modeled on IBM device
sim_method: 'matrix_product_state'  # Simulation method for AerSimulator
shots: 1024
```

Simulates quantum circuits with realistic noise models from actual IBM Quantum devices. This feature:
- Extracts the noise model from a specified IBM device (e.g., `ibm_cleveland`, `ibm_kyoto`)
- Runs simulation locally using AerSimulator with the device's noise characteristics
- Allows testing quantum algorithms under realistic noise conditions without queue time
- Supports any simulation method available in AerSimulator

**Format:** `'noisy_<device_name>'` where `<device_name>` is any IBM Quantum device name.

**Examples:**
- `'noisy_ibm_cleveland'` - Noise model from IBM Cleveland
- `'noisy_ibm_kyoto'` - Noise model from IBM Kyoto
- `'noisy_ibm_sherbrooke'` - Noise model from IBM Sherbrooke

```{tip}
**Choosing a Simulation Method for Noisy Simulations:**

- **`matrix_product_state`**: Recommended for most quantum machine learning circuits. Efficient for circuits with moderate entanglement.
- **`tensor_network`**: Best for GPU-accelerated simulations. Requires `qiskit-aer-gpu` installation.
- **`statevector`**: Most accurate but memory-intensive. Limited to ~20-25 qubits depending on available RAM.
- **`automatic`**: Let AerSimulator choose the best method based on circuit properties.
```

**IBM Quantum Hardware:**

```yaml
backend: 'ibm_least'  # Use least busy device
# OR
backend: 'ibm_kyoto'  # Specific device name
shots: 4096
resil_level: 1  # Error mitigation level (1-3)
```

Runs circuits on actual IBM Quantum hardware.

**IBM Quantum Credentials:**

```yaml
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'account_qbc'  # Account alias in JSON
ibm_instance: 'ibm-q/open/main'  # Optional: specific instance
```

```{important}
**IBM Credentials Required for Noisy Simulations:**

When using `noisy_<device_name>` backends, you must provide valid IBM Quantum credentials even though the simulation runs locally. This is because QBioCode needs to:
1. Connect to IBM Quantum services to retrieve the device's noise model
2. Download the latest calibration data for accurate noise simulation

The actual circuit execution happens locally on your machine using AerSimulator, so you won't incur queue wait times or consume IBM Quantum compute credits.
```

```{note}
**Backend Options:**
- `'simulator'`: Local Qiskit statevector simulator (exact, noiseless)
- `'simulator_aer'`: AerSimulator with configurable simulation method
- `'noisy_<device_name>'`: Noisy simulation based on IBM device noise model (e.g., `'noisy_ibm_cleveland'`)
- `'ibm_least'`: Automatically select least busy IBM Quantum device
- `'ibm_<device_name>'`: Specific IBM Quantum device (e.g., 'ibm_kyoto')

**Simulation Methods (for AerSimulator):**
When using `'simulator_aer'` or `'noisy_<device_name>'` backends, specify the simulation method via `sim_method`:
- `'statevector'`: Exact statevector simulation (default, memory intensive)
- `'matrix_product_state'`: Efficient for low-entanglement circuits
- `'tensor_network'`: GPU-accelerated tensor network simulation
- `'stabilizer'`: Fast simulation for Clifford circuits
- `'extended_stabilizer'`: Approximate simulation for near-Clifford circuits
- `'automatic'`: Automatically select best method

**Shots:** Number of circuit executions. Higher = more accurate but slower.

**Resilience Level:** Error mitigation strength (1=light, 2=medium, 3=heavy). Higher = more accurate but slower.
```

### Projection Backend: how many features a quantum model can afford

`projection_backend` selects *how* the projected-kernel models (`pqk`, `qpl`) evaluate
their Pauli expectation values on a local simulator. It changes cost only — every backend
returns the same numbers, verified equal to ~1e-14 — so it is a performance switch, not a
modelling one.

It matters because **one feature is one qubit.** The default path stores all `2**n`
amplitudes, so it runs out of memory near 30 qubits and becomes impractical past about 20
on a few hundred samples. That is a hard cap on how many features a `pqk`/`qpl` model can
use, and the usual workaround — PCA down to `n_components: 3` — throws most of the dataset
away before the model sees it. Matrix-product-state backends are *linear* in qubit count
for bounded-entanglement feature maps, which lifts the cap to hundreds of features.

```yaml
backend: 'simulator'          # required: these backends are local
projection_backend: 'auto'    # omit entirely to keep the historical behaviour
projection_n_jobs: 1          # rows are independent; -1 uses every core
```

Requires the `[mps]` extra for every value except `statevector`:

```bash
pip install 'qbiocode[mps]'
```

```{note}
**Projection Backend Options:**
- *omitted* (the default): the `StatevectorEstimator` primitive, exactly as before. Kept
  as the default so existing configs and cached projections are unaffected.
- `'auto'`: pick the fastest measured backend for this feature map. Recommended.
- `'statevector'`: dense `2**n` statevector, one simulation per sample. Same result as the
  default at a fraction of the cost — the primitive re-simulates the circuit once *per
  observable*, and there are `3 x n_features` of them.
- `'aer_mps'`: Aer's matrix-product-state method. Fastest for `reps <= 6`.
- `'quimb_mps'`: quimb's MPS. Degrades far more gracefully with `reps`; use it for deep
  feature maps. Also the only option that reports whether it truncated.
- `'quimb_permmps'`: MPS with lazily tracked qubit permutation. A modest win on
  `entanglement: 'full'` only.
- `'quimb_exact'`: exact tensor-network contraction, no truncation. The fastest option for
  `entanglement: 'full'` once the circuit is wide enough that a dense statevector cannot
  cope.
```

**Which backend wins depends on the feature map, not on the data.** `'auto'` encodes the
measured rule:

| feature map | fastest backend |
|---|---|
| `entanglement` in `linear`, `reverse_linear`, `pairwise`, `circular`, `sca`; `reps <= 6` | `aer_mps` |
| the same patterns with `reps >= 7` | `quimb_mps` |
| `entanglement: 'full'`, under ~24 features | `statevector` |
| `entanglement: 'full'`, ~24 features or more | `quimb_exact` |

The reason the entanglement pattern decides this is that an MPS is only cheap when the
state's entanglement is bounded. `linear`-style patterns entangle neighbours only, and the
bond dimension stays at 2 regardless of width. `entanglement: 'full'` makes every pair
interact, which is precisely the state an MPS cannot compress — there, an MPS is *slower*
than the dense statevector.

```{warning}
`max_bond` caps the bond dimension and makes an MPS cheap on a state it cannot otherwise
compress, but it does so by discarding part of the state, and **nothing raises**. The
projections simply become approximate. `compute_qpl`/`compute_pqk` warn when the fidelity
estimate drops below 0.999; treat any such run's metrics as approximate and report the
fidelity alongside them. On `entanglement: 'full'` there is no useful operating point —
measured at 20 features, a cap of 128 out of an exact 156 still loses a third of the state.
```

See {doc}`Tutorial Notebooks <../tutorials>` for two notebooks on this: *Simulator Selection for Projections*
benchmarks every backend across qubit count, `reps` and entanglement pattern, and *MPS vs
Statevector in QProfiler* runs the comparison end to end on a single-cell classification
task. Note that the AUC gain there comes from using more genes, not from the quantum
projection: a matched classical baseline on the same raw genes does at least as well, so
on that dataset MPS buys feasibility and speed rather than accuracy.

### Embedding Methods

Dimensionality reduction techniques to apply before model training.

**No Embedding:**

```yaml
embeddings: ['none']
```

**Single Embedding Method:**

```yaml
embeddings: ['pca']
n_components: 3  # Reduce to 3 dimensions
```

**Multiple Embedding Methods:**

```yaml
embeddings: ['pca', 'nmf', 'umap', 'autoencoder']
n_components: 5
```

**Available Embedding Methods:**
- `'none'`: No dimensionality reduction
- `'pca'`: Principal Component Analysis
- `'nmf'`: Non-negative Matrix Factorization
- `'umap'`: Uniform Manifold Approximation and Projection
- `'autoencoder'`: Neural network autoencoder

```{tip}
Start with `'none'` to establish baseline performance, then try `'pca'` for faster quantum model training.
```

#### When embeddings are applied: `embedding_min_features`

An embedding is applied only to a dataset with **more than `embedding_min_features`**
features. The default is **18**.

```yaml
embeddings: ['pca', 'nmf', 'none']
embedding_min_features: 18   # the default; omit the key to get it
```

Below that width a feature reduction mostly costs information without buying anything.
Every quantum learner in QBioCode encodes one qubit per feature, so 18 features is a
circuit these models handle directly — reducing to `n_components: 3` first discards most
of the problem and then reports the models' scores on what is left. The complexity blocks
say the same thing from the other side: on the 6-feature `class_data-1`, PCA to 3
components moves 108 of the 115 `mfe.` columns, so what is being profiled after the
reduction is largely not the dataset that was loaded.

When a dataset is too narrow, the **whole `embeddings` list collapses to a single
`'none'` pass**, and the log says which methods were skipped:

```text
WARNING - class_data-1.csv: 6 features is not more than embedding_min_features=18,
so ['pca', 'nmf'] will NOT be applied and the models run on the unreduced features
instead. [...] To embed anyway, set embedding_min_features: 0 in the config.
```

It collapses rather than running each name as a no-op on purpose: three requested
embeddings would otherwise fit every model three times on identical data and write three
groups of rows distinguished only by an `embeddings` column naming a reduction that never
happened.

**To embed a narrow dataset deliberately**, set the threshold to 0:

```yaml
embedding_min_features: 0    # apply every requested embedding, whatever the width
```

That is what both QProfiler tutorial configs do — their data is 6 and 10 features wide and
the comparison they draw is `'none'` against `'pca'`.

```{note}
The threshold applies to every embedding name uniformly, including the QuVINE graph
embeddings, whose output width does not depend on the feature count. If you want a graph
embedding on a narrow table, set `embedding_min_features: 0`.
```

#### Sharing one embedding across jobs: `embedding_cache`

By default each run computes its own embeddings, seeded by the split. That is
reproducible on one machine type, which is all a single run needs. It is **not
reproducible across CPU types**: UMAP's numba kernels are compiled for the host's
instruction set, and its SGD epochs amplify the last-bit differences that leaves.
Randomized PCA, which wide datasets use, depends on BLAS the same way, though only in the
last bits. An experiment split into one cluster job per model therefore scores its models
on different features whenever the jobs land on different host types. The 2026 pilot did
exactly that, with UMAP condition numbers up to 57% apart between jobs of one
(dataset, embedding).

`embedding_cache` removes the per-job computation. Every embedding except `'none'` is
computed once, before any job starts, and read by every job:

```yaml
embedding_cache: /abs/path/to/embeddings   # null (the default): compute in each run
```

```bash
# Write the files for every config that will run: one per (dataset, embedding, split),
# however many configs share it. Files that are already current are kept.
python -m qbiocode.apps.qprofiler.embedding_cache runs/*/*.yaml
python -m qbiocode.apps.qprofiler.embedding_cache --check runs/*/*.yaml   # write nothing
```

- **The path must be absolute.** Hydra runs each job from its own output directory.
- **A job never falls back to computing.** Before fitting anything it checks that every
  file it will read exists and carries its own settings, and otherwise it stops with a
  list of the missing or stale files. The settings compared are the dataset's
  sha256, `index_col`, the split (`seed` plus the split number, `test_size`, `stratify`),
  `scaling`, the embedding name, `n_components`, `n_neighbors` and `quvine_args`. Each
  file also stores the dataset rows it was embedded from, and the job checks those
  against its own split.
- **A stale file is reported, not replaced**, because other jobs may still read it.
  Replace it with `--force`, or point the config at a new directory.
- Each file also records where, when and with which library versions it was computed.
  The job logs a sha256 of the features it used for every pass, so the logs of two jobs
  show whether they saw the same features.

#### Resuming instead of restarting: `skip_existing`

A run appends to `ModelResults.csv` as each model returns, so a job killed at a cluster
wall keeps every pass that finished. What it does not keep is the benefit: rerunning the
same config opens a **new** run directory and starts again at the first split, and the
tooling downstream reads one run directory per config — so a config that needs more than
one wall never completes, however many times it is resubmitted. On the quantum arms one
pass is hours, which makes this the difference between a run that converges and one that
does not.

`skip_existing` makes a rerun cumulative:

```yaml
skip_existing: false                 # the default: compute every pass
skip_existing: true                  # resume from the other run directories of this config
skip_existing: /abs/path/to/results  # ...or from this directory instead
```

For each (embedding, split, model) cell an earlier run already holds, the new run copies
that row into its own `ModelResults.csv` **verbatim** — the text, not a re-serialised
number — carries the model's `oof/`, `trials/` and `val_predictions/` rows across, and
does not fit the model. A pass whose every model is present is skipped whole: no embedding
is read and no complexity measure is recomputed. A pass missing some of its models fits
only those. Every adopted cell is listed in `adopted.csv` with the run directory it came
from.

- **The path must be absolute**, for the same reason `embedding_cache`'s must be. `true`
  resolves to the parent of the run directory, which is where hydra puts the sibling runs
  of one `config_file_name`. The current run directory is never read as an earlier one.
- **The newest run wins** where several hold the same cell: run directories are named
  `<backend>_%Y-%m-%d_%H-%M-%S`, so sorting them by name sorts them by time.
- **What is not checked is that the config did not change.** Nothing ties a run directory
  to a config but its name. Under `split_mode: manifest` the rows carry `dataset_sha256`
  and `manifest_sha256`, and a row disagreeing with the run's is refused and named in the
  log — which catches a dataset or a frozen split changing underneath a resume. In
  internal mode there is no such column, so **delete the old run directories rather than
  resume after editing the config.**
- `results.pkl` of a fully adopted pass is carried over. A partly adopted pass contributes
  the summary of the models that ran, because one summary describes one set of models;
  `ModelResults.csv` and the sidecars are complete either way.
- `embedding_cache` is still required to be complete for the whole run, including the
  passes that will be adopted. It is a pre-flight contract and resuming does not relax it.

On the cluster, `SKIP_EXISTING=1 ./submit_runs.sh` (or `./submit_array.sh`) passes
`++skip_existing=true` to every job it submits.

### Train/Test Split

Configure data splitting and preprocessing.

```yaml
test_size: 0.2      # 80% train, 20% test
stratify: ['y']     # Maintain class distribution
scaling: ['True']   # Standardize features
```

**Parameters:**
- `test_size`: Proportion of data for testing (0.0-1.0)
- `stratify`: `['y']` to maintain class balance, `['n']` for random split
- `scaling`: `['True']` to standardize features (recommended), `['False']` for raw data

```{warning}
Always use `stratify: ['y']` for imbalanced datasets to ensure both train and test sets have representative class distributions.
```

#### Fold-based evaluation: `split_mode: manifest`

By default (`split_mode: internal`) QProfiler draws `iter` random train/test splits
itself, at `seed + iter`, with `test_size` and `stratify` above. With
`split_mode: manifest` the outer splits come from a frozen per-dataset manifest written
by `benchmark/make_splits.py`, and every model is tuned inside every fold on the same
budget:

```yaml
split_mode: manifest            # internal (the default) | manifest
split_dir: /abs/path/splits/v2  # one <dataset stem>.json per dataset; resolved like folder_path
splits: all                     # 'all', one global iteration, or a list, e.g. [3]
grid_search: True
tuner: optuna                   # grid is refused: it ignores n_trials
n_trials: 30                    # every arm, quantum included
tune_quantum: True              # when any quantum model is listed
freeze_quantum_params: False    # freezing is refused
```

**The protocol:**
- **Outer splits:** stratified k-fold repeated R times (make_splits default 5 x 3, repeat
  r shuffled with `seed + r`; repeat 0 is `splits/v1`). A split's global iteration is
  `repeat * k + fold + 1` (1..15); it is the iteration in the `data_key`, so `splits: [i]`
  runs one fold and a manifest can be sharded one job per fold.
- **Validation rows ("next fold"):** for fold f, the validation rows are the test rows of
  fold (f + 1) mod k; the fit rows are the rest of the training fold. With k = 5 that is
  3/5 fit, 1/5 validation, 1/5 test.
- **Tuning:** every arm runs exactly `n_trials` trials (`n_trials_quantum` is ignored,
  with a warning if it differs). Trial 0 is the arm's default config (the `compute_<m>`
  defaults under `<m>_args`). Each trial is one fit on the fit rows scored on the
  validation rows with `tuning_metric`; there is no inner CV and no inner holdout. The
  best-validation config is refit on the whole training fold and scored once on the test
  fold. `tuning_score` is that trial's validation score. Every listed model needs an
  `_opt` twin (`qensemble` has none).
- **Features:** scaler and embedding are fitted on the fit rows for the trials and on the
  training fold for the refit, both at `embed_seed = seed + iteration`; no validation row
  reaches a trial. With `embedding_cache`, each pass has a second file,
  `emb_<data_key>__tune.npz`, and a file of one split mode is refused as stale by the other.
- **Checks:** the dataset CSV must be the file the manifest was computed from (sha256,
  row count and labels are verified before anything is fitted); a dataset without a
  manifest is an error. `iter`, `test_size`, `stratify`, `cross_validation` and
  `validation_split` are not used.
- **Output:** ModelResults rows gain the `PROTOCOL_COLUMNS` of
  `qbiocode.evaluation.protocol`: `split_mode`, `repeat`, `fold`, `split_k`,
  `split_repeats`, `split_validation`, `split_seed`, `manifest_sha256`, `dataset_sha256`,
  `split_generator`, `n_fit`, `n_val`, `n_test`, `seed`, `q_seed`, `embed_seed`, `host`,
  `cpu_model`, `lsf_jobid`. `results.pkl` also keeps `train_idx`, `fit_idx`, `val_idx`,
  `test_idx` and each model's `trials_<model>` log. Each pass writes three sidecars
  beside it: `oof/<data_key>.csv` (every model's test predictions with row ids),
  `trials/<data_key>.csv` (every trial's params, validation score, state, duration,
  `is_default`, `is_best`) and `val_predictions/<data_key>.csv` (each trial's
  validation predictions). Internal-mode output is unchanged.
- **RawDataEvaluation** (the dataset-level features) is still computed once, on all rows.
- **Model-specific settings:** SVC fits (`svc`, and the PQK/QPL SVC heads) are capped at
  `max_iter` = the configured value if > 0, else 10,000,000 (`pqk_args`/`qpl_args`
  `head_max_iter`); a trial that stops at the cap is recorded as failed. The PQK/QPL
  heads' own RandomizedSearchCV scores with `tuning_metric` (`head_scoring`) and counts
  as part of one trial's fit. QPL heads get their own `trials_<model>_<head>` logs.
  `compute_qnn` takes `readout: 'global'` (Z on every qubit, the default) or `'local'`
  (Z on qubit 0); search it with `gridsearch_qnn_args: {readout: ['global', 'local']}`,
  or fix it with `qnn_args: {readout: 'local'}`.
- `validation`, `default_params` and `reseed` are reserved keys in `<model>_args` and
  `gridsearch_<model>_args` blocks; they are dropped with a warning.

Manifest results are analysed with `selection='validation'` in
`qbiocode.utils.fair_selection.select_winners` (and `qc_winner_finder`): per
(dataset, embedding, fold) and side the winner is the best validation score, and the
correction uses `r = 1/(k-1)`, `n = kR` folds, `df = kR - 1`. Arms tied on the
validation score are narrowed to the best validation AUC of the refit trial (`val_auc`, a
column of manifest-mode rows; `tiebreak_col=None` turns this off), and a tie that remains
averages the tied arms' test scores. `val_log_loss` is recorded too, but only models that
output probabilities have one.

### Model Selection

Specify which machine learning models to evaluate.

**All Models:**

```yaml
model: ['svc', 'dt', 'lr', 'nb', 'rf', 'mlp', 'xgb', 'catboost', 'tabpfn',
        'qsvc', 'vqc', 'qnn', 'pqk']
```

**Classical Models Only:**

```yaml
model: ['rf', 'svc', 'lr', 'mlp', 'xgb', 'catboost']
```

**Quantum Models Only:**

```yaml
model: ['qsvc', 'vqc', 'qnn', 'pqk']
```

**Available Models:**

| Model | Type | Description |
|-------|------|-------------|
| `svc` | Classical | Support Vector Classifier |
| `dt` | Classical | Decision Tree |
| `lr` | Classical | Logistic Regression |
| `nb` | Classical | Naive Bayes |
| `rf` | Classical | Random Forest |
| `mlp` | Classical | Multi-Layer Perceptron |
| `xgb` | Classical | XGBoost |
| `catboost` | Classical | CatBoost gradient boosting |
| `tabpfn` | Classical | TabPFN pretrained tabular transformer (needs the `[tabpfn]` extra) |
| `qsvc` | Quantum | Quantum Support Vector Classifier |
| `vqc` | Quantum | Variational Quantum Classifier |
| `qnn` | Quantum | Quantum Neural Network |
| `pqk` | Quantum | Projected Quantum Kernel |

### Model Hyperparameters

Configure hyperparameters for each model. Each model has:
- **Standard arguments** (`<model>_args`): single values, used when the model is not tuned
- **Tuned arguments** (`gridsearch_<model>_args`): what to search when `grid_search: True`

```{warning}
**With tuning on and the default `split_mode: internal`, a model's `<model>_args` block is
NOT read.** When `grid_search: True` (and, for a quantum model, `tune_quantum: True`),
every trial and the final refit are built from `gridsearch_<model>_args` alone. **A
setting written only in `<model>_args` is silently ignored: no error, no warning.** The
shipped `config.yaml` has tuning on, so this is what a default run gets.

- **To fix a setting while tuning,** write it in `gridsearch_<model>_args` as a one-value
  list, e.g. `thread_count: [1]`.
- **Under `split_mode: manifest` the block IS read:** its values for the searched names
  are trial 0 (the arm's default config), and its other keys are fixed for every trial
  and the refit.
- **`qpl_args.classical_models`** is read in both modes (see below).
```

A tuned argument may be written two ways:

| Syntax | Meaning |
|---|---|
| `C: [0.1, 1, 10]` | a list -- one of these values is chosen |
| `C: {low: 0.001, high: 100}` | a range -- sampled continuously |
| `C: {low: 0.001, high: 100, log: true}` | as above, on a log scale |
| `n_estimators: {low: 10, high: 500}` | integer bounds give integer values |

Two keys control the search itself:

```yaml
grid_search: True    # tune hyperparameters at all
tuner: optuna        # 'optuna' (default) or 'grid'
n_trials: 50         # Optuna's trial budget
cross_validation: 5  # folds used to score each candidate
```

`tuner: optuna` spends `n_trials` fits, steering them with a TPE sampler toward the
region that has been scoring well. `tuner: grid` restores the exhaustive
`GridSearchCV` sweep, which fits *every* combination -- the `gridsearch_rf_args`
block below is 576 combinations, or 2,880 fits at `cross_validation: 5`. Ranges
require `tuner: optuna`; under `tuner: grid` every entry must be a list.

**What the search optimises.** `tuning_metric` names the statistic every tuner selects
on, classical and quantum alike:

```yaml
tuning_metric: balanced_accuracy   # default; or accuracy, mcc, f1_score
```

Each choice is computed identically by the classical cross-validation scorer and by
`modeleval`, which scores the quantum candidates, so both sides select on the same number.
`auc` and `pr_auc` are not offered, because the two sides would not agree on them for a
multiclass target. `f1_score` uses the run's `average`. An unknown name is refused before
any model is fitted. The default was plain accuracy before this key existed; set
`tuning_metric: accuracy` to reproduce such a run (the pilot10 runs among them).

A tuned row also records the evidence behind its parameters in three columns,
`tuning_metric`, `tuning_score` (the winning cross-validated score) and `tuning_reused`
(True when the parameters came from the frozen cache, see below). They are bookkeeping,
not meta-features, and the meta-regression excludes them.

#### Tuning the quantum models

The quantum classifiers (`qsvc`, `vqc`, `qnn`, `pqk`, `qpl`) tune through the same
`gridsearch_<model>_args` blocks, but tuning them needs a second key. A config without it
leaves them untuned; the shipped `config.yaml` sets both keys, so that the two sides are
tuned alike:

```yaml
grid_search: True        # tune at all
tune_quantum: True       # ... including the quantum models
n_trials_quantum: 10     # their budget: smaller, because each trial is a quantum fit
validation_split: 0.25   # holdout fraction used to score a quantum candidate
```

Two differences from the classical path, both driven by cost -- a quantum fit builds an
n-by-n fidelity kernel by circuit simulation, seconds rather than milliseconds:

- Each candidate is scored **once**, on a stratified holdout carved out of the training
  data, rather than on `cross_validation` folds. The test set is never touched by the
  search.
- `n_trials_quantum` defaults to 10 rather than 50.

Every quantum model named in `model` needs its own `gridsearch_<model>_args` block when
`tune_quantum` is on; a model with nothing to search is reported by name rather than
silently skipped. The five blocks, as shipped:

```yaml
gridsearch_qsvc_args:
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']
  C:            {low: 1.0e-2, high: 1.0e+2, log: true}

gridsearch_vqc_args:
  encoding:        ['Z', 'ZZ']
  reps:            [1, 2]
  ansatz_type:     ['amp']
  local_optimizer: ['COBYLA', 'L_BFGS_B']
  maxiter:         {low: 50, high: 200}

gridsearch_qnn_args:    # same keys as vqc
  encoding:        ['Z', 'ZZ']
  reps:            [1, 2]
  ansatz_type:     ['amp']
  local_optimizer: ['COBYLA', 'L_BFGS_B']
  maxiter:         {low: 50, high: 200}

gridsearch_pqk_args:
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']

gridsearch_qpl_args:    # same keys as pqk
  encoding:     ['Z', 'ZZ']
  reps:         [1, 2]
  entanglement: ['linear', 'full']
```

`pqk` and `qpl` also take a `data_map`, the rule that turns features into gate angles, in
their `pqk_args`/`qpl_args` or as a searched key:

| `data_map` | angle for a feature pair | |
|---|---|---|
| `'unit'` (default) | `x_i * x_j` | the historical map; keeps existing cache files and rows |
| `'qiskit'` | `(pi - x_i)(pi - x_j)` | qiskit's default, the map the `eng_zz` datasets are generated with |

The booleans of the `pqk` embedding are accepted too (`True` is `'unit'`, `False` is
`'qiskit'`), and any other value is refused before anything is written. The two maps never
share a projection file, and only a non-default map is recorded in the results row.

What is tunable per model:

| Model | Tunable |
|---|---|
| `qsvc` | `encoding`, `entanglement`, `reps`, `primitive`, `C`, `gamma`, `pegasos`, `bandwidth` |
| `vqc`, `qnn` | `encoding`, `entanglement`, `reps`, `primitive`, `ansatz_type`, `local_optimizer`, `maxiter` |
| `pqk`, `qpl` | `encoding`, `entanglement`, `reps`, `primitive`, `data_map`, `bandwidth` |

There is no `n_qubits`: the qubit count follows from the width of the data reaching the
model, so it is set by the embedding's `n_components`, not by tuning.

**`bandwidth` is the quantum arms' gamma.** QProfiler scales features to [0, 1], and a
feature map turns each feature into a rotation angle (`P(2 x)` for the qiskit maps). So
without it, every quantum kernel runs at one fixed angle range, while the classical SVC
tunes its gamma. `bandwidth` multiplies the features before the feature map, so they reach
it in [0, bandwidth]:

```yaml
gridsearch_qsvc_args:
  bandwidth: {low: 0.0982, high: 6.2832, log: true}   # pi/32 .. 2*pi
```

In the kernel_exps torus study this setting alone moved a fidelity kernel by up to 0.26 F1.
Its optimum depended on the encoding: about pi for `Z` and pi/32 for `ZZ`. The default 1.0
is the unscaled map and leaves results and projection caches exactly as before; any other
value is recorded in the parameter column and keys its own projection cache. `vqc` and
`qnn` do not take it.

`qpl`'s `classical_models` is **not** in that table, because it is not a hyperparameter to
search -- it selects which classical heads are fitted on the quantum projection. It stays
in `qpl_args` and is read from there whether or not tuning is on:

```yaml
qpl_args:
  classical_models: ['rf', 'lr']   # honoured with tune_quantum on or off
```

Every trial fits all of the named heads and a candidate is scored by their *mean*
accuracy, so narrowing the list makes a tuned QPL run proportionally cheaper as well as
changing what the search optimises for.

**Tuning on real hardware is refused** unless you also set `allow_hardware_tuning: True`.
Every trial is a separate queued job billed against your instance, and the failure mode
is silent -- the run simply never appears to finish. Tune on `backend: simulator`, then
run the winning configuration on the device.

`tune_quantum` without `grid_search` is an error rather than a no-op. `tuner: grid` only
ever applies to the classical models -- a quantum candidate is scored by running the whole
model, so there is no exhaustive-grid engine for one; setting both warns and still uses
Optuna for the quantum models.

QPL is scored on the **mean** accuracy across the classical heads it fits on the quantum
projection, which is what tuning the projection is meant to improve. Taking the best head
instead would let one lucky head choose the projection, and every head is reported anyway.

(The statistic averaged is the run's `tuning_metric`, which defaults to balanced accuracy;
it is accuracy only under `tuning_metric: accuracy`.)

#### Freezing the quantum search across iterations

Leaving `tune_quantum: False` while `grid_search: True` is the cheap configuration and it
is **not a fair comparison**: every classical learner is searched, no quantum learner is,
and each quantum loss is then confounded with the fact that nobody searched its space.
Turning `tune_quantum: True` on removes the confound and multiplies the quantum side by
`n_trials_quantum` *per iteration*.

`freeze_quantum_params` is the middle option -- search each quantum arm once, on iteration
0, and reuse that configuration for the rest:

```yaml
grid_search: True
tune_quantum: True
n_trials_quantum: 32
freeze_quantum_params: True
quantum_param_dir: quantum_tuned_params   # one JSON file per arm
```

Quantum fits per arm, at `iter: 5`:

| Setting | Fits | |
|---|---|---|
| `tune_quantum: False` | 5 | not comparable to a tuned classical arm |
| `tune_quantum: True` | 160 | `iter * n_trials_quantum` |
| `+ freeze_quantum_params: True` | 36 | `n_trials_quantum + (iter - 1)` |

**What it costs, and which way.** The frozen configuration was chosen on iteration 0's
training split, so on iterations 1..I-1 it is a configuration selected elsewhere -- never
better than a fresh search would have found on that split. The classical side re-searches
every iteration and keeps its full advantage. Quantum is therefore measured at a handicap
classical does not carry, so a quantum win observed under this setting is a **lower bound**
on the win a symmetric budget would show. That asymmetry is acceptable precisely because it
points away from the interesting claim, but it has to be reported as what it is, and the
protocol must not be described as symmetric.

**Correction: the handicap above does not hold, and the lower-bound claim is withdrawn.**
Iterations resample the same data, so iteration 0's training split overlaps each later
iteration's test rows heavily (73-87% in the pilot). A configuration frozen on iteration 0
has therefore been selected partly on rows it is later scored on, which is *optimistic*
for quantum, not a handicap, while the classical side's test rows never influence its own
selection. The measured effect in the pilot was small (a difference-in-differences of about
-0.02 balanced accuracy, not significant); the direction is what matters. A quantum win under
`freeze_quantum_params: True` is not a lower bound on anything. For a claim about a
quantum win, set it to `False`, or tune the frozen search on folds disjoint from every
later test split.

Frozen files also record the `tuning_metric` they were searched on. A file written before
that key existed counts as accuracy-tuned, so under any other metric it is searched again
rather than reused.

The cache key is `(dataset, embedding, n_components, model)` -- deliberately everything
except the iteration, since reuse across iterations is the point. Editing a
`gridsearch_<model>_args` block invalidates the affected files automatically, because a
frozen set whose names no longer match the current space is discarded rather than reused.
Every failure path -- missing file, truncated file, unwritable directory -- degrades to
"search this iteration again", so a corrupt cache costs one redundant search and never
aborts a sweep. Delete `quantum_param_dir` to force a fresh search throughout.

A frozen iteration still reports its model as `<model>_opt`, the same label a freshly
searched one carries. That matters for
`qbiocode.utils.select_winners`, which pairs arms across the iteration axis:
relabelling the reused iterations would split one arm in two and break the pairing.

#### Projection caches

`pqk` and `qpl` cache their projected feature matrices so a rerun does not recompute
circuits. The file name includes a fingerprint of the settings that change the circuit
(`encoding`, `entanglement`, `reps`, `primitive`, feature width), and the row count is
checked on load. Redirect either cache if you want throwaway projections kept apart from
your real ones:

```yaml
pqk_projection_dir: pqk_projections   # default, relative to the working directory
qpl_projection_dir: qpl_projections   # default
```

Tuning does this automatically: every trial writes to a temporary directory that is
deleted when the search ends, so trial projections never collide with the final run's.

#### The results column

With tuning on, `ModelResults.csv` records the chosen hyperparameters in a
**`BestParams_Tuned`** column; with tuning off it records `Model_Parameters` instead,
never both. `BestParams_Tuned` was called `BestParams_GridSearch` before Optuna became
the default engine, and every reader (`qbiocode.utils.qc_winner_finder`, `QuantumSage`)
still accepts the old name, so results files written earlier keep working.

**Example: Support Vector Classifier (SVC)**

```yaml
# Standard run with fixed parameters
svc_args:
  C: 1.0
  gamma: 0.1
  kernel: 'rbf'

# Tuned: lists and ranges may be mixed freely
gridsearch_svc_args:
  C: {low: 0.001, high: 100, log: true}
  gamma: {low: 0.0001, high: 1, log: true}
  kernel: ['linear', 'rbf', 'poly', 'sigmoid']
```

**Example: Random Forest (RF)**

```yaml
rf_args:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15, 20]
  min_samples_split: [2, 5, 10]
```

**Example: XGBoost (XGB)**

```yaml
xgb_args:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6

gridsearch_xgb_args:
  n_estimators: [50, 100, 200]
  learning_rate: [0.01, 0.1, 0.3]
  max_depth: [3, 6, 9]
```

**Example: CatBoost**

Any parameter you leave out stays at CatBoost's own default. That is not the same as
writing the documented default in: CatBoost *derives* several defaults from the data and
from each other, so naming one can change more than the one value. It auto-selects
`learning_rate` for `Logloss` and `MultiClass` unless `l2_leaf_reg` is set, and it
chooses `bootstrap_type` from the loss it inferred.

```yaml
catboost_args:
  iterations: 200
  learning_rate: 0.1
  depth: 6
  l2_leaf_reg: 3.0

gridsearch_catboost_args:
  iterations: [100, 200, 400]
  learning_rate: {low: 1.0e-2, high: 3.0e-1, log: true}
  depth: [4, 6, 8]
  l2_leaf_reg: {low: 1.0, high: 10.0}
  random_strength: [0.5, 1.0, 2.0]
```

```{note}
`min_data_in_leaf` is accepted but **not searchable**. CatBoost honours it only under
`grow_policy: Depthwise` or `Lossguide`; at the default `SymmetricTree` every value produces
an identical model, so searching it multiplies the fits for nothing. Give it a single value
alongside a `grow_policy` that honours it — several values are refused with a message saying
so.
```

```{warning}
**`subsample` and `bagging_temperature` are not interchangeable, and which one is legal
depends on the loss.** They belong to mutually exclusive CatBoost bootstrap schemes, and
CatBoost picks the default scheme from the loss: `MVS` under `Logloss`, `Bayesian` under
`MultiClass`. QProfiler is a binary-classification tool, so the inferred loss is
`Logloss` and `subsample` works at the default — but setting `loss_function: MultiClass`,
which is legal even on a two-class target, flips the scheme and fails:

    CatBoostError: default bootstrap type is Bayesian, which does not support subsample

QBioCode pins `bootstrap_type` for you as soon as you name either parameter —
`Bernoulli` for `subsample`, `Bayesian` for `bagging_temperature` — so the behaviour no
longer depends on the loss. Naming *both*, or searching `bootstrap_type` across values
that contradict the one you named, is rejected before the search starts with a message
naming the config key. The shipped block above simply stays clear of the area.

`loss_function: MultiClass` also makes CatBoost's `predict()` return an `(n, 1)` column
rather than a flat array. QBioCode flattens it, so the stored predictions keep the same
shape as every other model's — scikit-learn scores a column vector correctly either way,
so this affects the results frame rather than the metrics.
```

**Example: TabPFN**

TabPFN needs only the optional extra — no API key, no license acceptance:

```bash
pip install "qbiocode[tabpfn]"
```

QBioCode pins `model_version: v2`, whose weights are published under the Prior Labs
License (Apache 2.0 plus an attribution clause) and download anonymously on first fit.

```{warning}
**The model version is a licensing choice.** TabPFN's *code* is Apache 2.0 plus
attribution, but its *weights* are licensed per version and the regimes differ sharply:

| `model_version` | Weights license | Commercial use |
| --- | --- | --- |
| `v2` (default) | Prior Labs License v1.1 (Apache 2.0 + attribution) | **Permitted** |
| `v2.5` | TABPFN-2.5 Non-Commercial License | No |
| `v2.6` | TABPFN-2.6 Non-Commercial License | No |
| `v3` | TABPFN-3 Non-Commercial License | No |

The three newest are **non-production as well as non-commercial**: their license permits
testing, evaluation, internal benchmarking and academic research, but not revenue-generating
activity, production systems, or training other models for commercial use. They also require
accepting that license against a Prior Labs account, which upstream does interactively — so
they cannot be fetched unattended, and an API key alone is not enough. Setting
`model_version` to one of them warns and names the license.

`v2` is the default precisely because QBioCode is Apache-2.0 software whose users include
companies; a default that quietly imposed a non-commercial license on them would be the
wrong default whatever its accuracy.
```

Only if you opt into a restricted version do you need an API key. There are two ways to
supply one, and the first is preferred:

```yaml
# The key lives in a file OUTSIDE the repository, so it cannot be committed.
tabpfn_json_path: '~/.config/qbiocode/tabpfn.json'
```

Create that file with the correct permissions rather than by hand:

```bash
python -c "from qbiocode.utils import write_token_template; print(write_token_template())"
# -> ~/.config/qbiocode/tabpfn.json, created mode 0600
```

then paste the key into its `token` field:

```json
{
  "token": "<your api key>"
}
```

QProfiler reads it automatically whenever `tabpfn` is in the `model` list. Outside
QProfiler, call `qbiocode.utils.load_tabpfn_token()` before fitting.

The alternative is to export the variable TabPFN itself reads, which takes precedence over
the file:

```bash
export TABPFN_TOKEN="<your api key>"
```

```{warning}
**Do not put the key in `config.yaml`, or in any file inside the repository.** The
`~/.config/` location is the supported one specifically because it is outside the
checkout: a gitignored file in the tree is *unlikely* to be committed, not unable to be —
`git add -f` overrides the rule, a rewritten `.gitignore` stops covering it, and a copy
made into a sibling clone is not covered at all.

QBioCode never logs or prints the token. `qbiocode.utils.describe_token_source()` reports
*whether* a token is configured and where it came from, with no key material at all — not
even a prefix or fingerprint — because tutorial notebooks are published with their
committed outputs.
```

```yaml
tabpfn_args:
  n_estimators: 4
  softmax_temperature: 0.9
  balance_probabilities: False
  average_before_softmax: False
  device: cpu

gridsearch_tabpfn_args:
  n_estimators: [1, 4, 8]
  softmax_temperature: {low: 0.5, high: 1.5}
  balance_probabilities: [True, False]
  device: cpu
```

```{note}
**Nothing in the TabPFN block is a training hyperparameter.** The weights are frozen and
pretrained; `fit` only memorises the training rows, and every setting above is an
inference knob -- none of them changes model capacity. `n_estimators` buys ensemble
members over differently preprocessed views of the same rows.

Two consequences for tuning. A trial costs `cross_validation` full transformer forward
passes rather than five cheap tree fits, so keep `gridsearch_tabpfn_args` small. And
`device` is pinned to `cpu` above deliberately: MPS and CUDA are not numerically
identical to CPU, which would make a benchmark irreproducible across machines.

TabPFN also supports **at most 10 classes**. Unlike its row and feature limits, that one
cannot be waived with `ignore_pretraining_limits`; a dataset with more is rejected up
front, naming its class count.
```

```{seealso}
For detailed parameter descriptions, see the upstream documentation:
- [SVC Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Random Forest Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [CatBoost Training Parameters](https://catboost.ai/docs/en/references/training-parameters/common)
- [TabPFN](https://github.com/PriorLabs/TabPFN)
```

### Quantum Model Hyperparameters

For quantum models, hyperparameter tuning requires generating separate config files for each combination.

```{important}
**QML Grid Search:**

Quantum model grid search is handled differently than classical models. Use the `generate_experiments.ipynb` notebook in `archive/tutorial_notebooks/qml_experiment_generators/` to generate individual config files for each parameter combination.

This approach is necessary because:
1. Quantum jobs are submitted to IBM Quantum queue
2. Each configuration may take hours to complete
3. Separate configs allow parallel job submission
```

**Example: Quantum SVC (QSVC)**

```yaml
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
  entanglement: 'linear'
```

**Example: Variational Quantum Classifier (VQC)**

```yaml
vqc_args:
  feature_map: 'ZZFeatureMap'
  ansatz: 'RealAmplitudes'
  reps: 3
  optimizer: 'COBYLA'
```

---

## Complete Example Configuration

Here's a comprehensive example combining all sections:

```yaml
# Experiment identification
config_file_name: 'comprehensive_experiment'

# Input data
folder_path: 'datasets/'
file_dataset: ['cancer_data.csv', 'diabetes_data.csv']
output_dir: 'results/experiment_001/'

# Reproducibility
seed: 42
q_seed: 42

# Quantum backend - Noisy simulation example
backend: 'noisy_ibm_cleveland'
sim_method: 'matrix_product_state'
shots: 1024
resil_level: 1
qiskit_json_path: '~/.qiskit/qiskit-ibm.json'
name: 'my_ibm_account'

# Dimensionality reduction
embeddings: ['none', 'pca']
n_components: 5

# Data splitting
test_size: 0.2
stratify: ['y']
scaling: ['True']

# Models to evaluate
model: ['rf', 'svc', 'mlp', 'xgb', 'catboost', 'qsvc', 'pqk']

# Classical model parameters
rf_args:
  n_estimators: 100
  max_depth: 10

gridsearch_rf_args:
  n_estimators: [50, 100, 200]
  max_depth: [5, 10, 15]

svc_args:
  C: 1.0
  kernel: 'rbf'

gridsearch_svc_args:
  C: [0.1, 1, 10]
  kernel: ['linear', 'rbf']

# Quantum model parameters
qsvc_args:
  feature_map: 'ZZFeatureMap'
  reps: 2
```

**Alternative Backend Configurations:**

```yaml
# For exact noiseless simulation
backend: 'simulator'

# For AerSimulator with GPU acceleration
backend: 'simulator_aer'
sim_method: 'tensor_network'

# For actual IBM Quantum hardware
backend: 'ibm_kyoto'
shots: 4096
resil_level: 2
```

---

## Best Practices

```{tip}
**Configuration Tips:**

1. **Start Simple**: Begin with a minimal config and add complexity gradually
2. **Use Descriptive Names**: Name configs by experiment purpose (e.g., `cancer_baseline.yaml`)
3. **Version Control**: Keep configs in git to track experiment history
4. **Document Changes**: Add comments in YAML to explain non-obvious choices
5. **Test Locally First**: Use `backend: 'simulator'` before submitting to quantum hardware
```

```{warning}
**Common Pitfalls:**

- **Missing Seeds**: Always set `seed` and `q_seed` for reproducibility
- **Too Many Combinations Under `tuner: grid`**: the exhaustive sweep fits every
  combination; start small to estimate runtime, or use the default `tuner: optuna`
```

---

## Troubleshooting

**Problem: "Config file not found"**
- Ensure config file is in `configs/` directory
- Check file name matches `--config-name` argument
- Use relative path from project root

**Problem: "Invalid backend"**
- Verify IBM Quantum credentials are configured
- Check device name spelling (use `ibm_<device>` format)
- Ensure you have access to the specified instance

**Problem: "Hyperparameter tuning taking too long"**
- Lower `n_trials` (the Optuna tuner's budget maps directly onto fits)
- Use fewer cross-validation folds
- If you set `tuner: grid`, the cost is the *whole* cross product regardless of
  `n_trials` -- switch back to `tuner: optuna` unless you specifically need the
  exhaustive sweep to reproduce an older result

**Problem: "Out of memory"**
- Reduce `n_components` for embeddings
- Use smaller `test_size` to reduce data size
- Process datasets one at a time instead of batch

---

## See Also

- :doc:`QProfiler Usage Guide <profiler>` - How to run QProfiler
- :doc:`QSage Configuration <sage>` - Meta-learning model selection
- :doc:`Tutorial Notebooks <../tutorials>` - Step-by-step examples
