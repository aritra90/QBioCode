(benchmarking)=

# Benchmarking classical vs quantum models

This page takes a benchmark from raw tabular data to two results: a per-dataset verdict
(did the best quantum model beat the best classical one?), and a meta-analysis of
*which dataset properties* move that verdict. Every step writes files that the next one
reads, and every file the result depends on is pinned by a hash. A collaborator who
reruns the same commands on the same inputs gets the same splits, features and
configurations.

::::{grid} 2 2 3 4
:gutter: 2

:::{grid-item-card} {octicon}`database;1.3em` 1 · Curate
:link: bench-curate
:link-type: ref
Real datasets in one numeric format, hashed.
:::

:::{grid-item-card} {octicon}`beaker;1.3em` 2 · Synthesize
:link: bench-synthetic
:link-type: ref
Shape and quantum families, controls gated.
:::

:::{grid-item-card} {octicon}`git-branch;1.3em` 3 · Split
:link: bench-splits
:link-type: ref
5-fold × 3 repeats, frozen as manifests.
:::

:::{grid-item-card} {octicon}`lock;1.3em` 4 · Hold out
:link: bench-holdout
:link-type: ref
Draw the meta-analysis hold-out, pre-register.
:::

:::{grid-item-card} {octicon}`file-code;1.3em` 5 · Configure
:link: bench-configs
:link-type: ref
One YAML per job, priced.
:::

:::{grid-item-card} {octicon}`stack;1.3em` 6 · Embed once
:link: bench-cache
:link-type: ref
The embedding cache every job reads.
:::

:::{grid-item-card} {octicon}`rocket;1.3em` 7 · Run
:link: bench-run
:link-type: ref
QProfiler on every YAML, on LSF or by hand.
:::

:::{grid-item-card} {octicon}`inbox;1.3em` 8 · Collect
:link: bench-collate
:link-type: ref
Merge and verify all job outputs.
:::

:::{grid-item-card} {octicon}`trophy;1.3em` 9 · Winners
:link: bench-winners
:link-type: ref
Validation-selected, corrected CV test.
:::

:::{grid-item-card} {octicon}`telescope;1.3em` 10 · Kernels
:link: bench-kernels
:link-type: ref
Huang et al. geometry from the kernel dumps.
:::

:::{grid-item-card} {octicon}`graph;1.3em` 11 · Meta-analysis
:link: bench-meta
:link-type: ref
Which properties predict a quantum win.
:::

:::{grid-item-card} {octicon}`checklist;1.3em` Checklist
:link: bench-checklist
:link-type: ref
Everything above on one screen.
:::
::::

## Before you start

The benchmark tools live in a QBioCode **source checkout**, not in the wheel:
`benchmark/` builds the inputs, and `experiments/pilot10/` writes, submits and collects
the jobs. Install the checkout with the apps extras, then name two directories:

```bash
git clone https://github.com/qiskit-community/QBioCode.git && cd QBioCode
pip install -e ".[apps]"
export REPO=$PWD                      # the checkout; every command below runs from it
export BENCH=/scratch/me/bench1       # a run root with room for ~10 GB
```

```{tip}
**On a shared login node, cap the threads.** Prefix heavy commands with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Many clusters kill a
login-node process tree that uses more than a few cores, and the numerical libraries
start one thread per core by default.
```

At the end, `$BENCH` holds this tree. Each step below says which part it writes.

```text
$BENCH/
├── data/
│   ├── datasets/<id>/<id>.csv, meta.yaml (+ latent.npz)   ← steps 1, 2
│   ├── inventory.csv, inventory_synthetic.csv, duplicates.csv
│   └── splits/v2/<id>.json                                 ← step 3
├── cluster_map.csv, holdout.csv, prereg.yaml               ← step 4
├── code/qbiocode/                                          ← frozen package (step 7)
├── runs_cv/<run_id>/
│   ├── MANIFEST.tsv                                        ← step 5
│   ├── embeddings/emb_<id>_<emb>_8_<split>[__tune].npz     ← step 6
│   └── <id>/
│       ├── <id>_<emb>_i01-15_classical.yaml, …_qsvc.yaml, …_pqk.yaml
│       ├── results/<config>/<backend>_<timestamp>/         ← step 7
│       ├── kernels/<config>/gram_*.npz, proj_*.npz         ← step 7 (step 10 reads)
│       ├── quantum_tuned_params/<config>/
│       └── lsf_logs/<config>.<jobid>.out, .err
└── collated/ModelResults.csv, oof.csv, trials.csv, …       ← step 8
```

(bench-curate)=

## 1 · Curate the real datasets

`benchmark/curate.py` reads a source tree with `pmlb_data/`, `openMLCC18/` and
`libsvm_data/` subfolders. It:
- drops ID and constant columns;
- one-hot encodes categorical columns;
- maps the label to `{0, 1}` and shuffles the rows once;
- writes **one CSV per dataset**, label last, with a `meta.yaml` that records every
  choice and the CSV's sha256.

```bash
python benchmark/curate.py --source-root /path/to/sources --out-root $BENCH/data/datasets
```

{bdg-primary}`writes` `data/datasets/<id>/<id>.csv`, `meta.yaml`, plus
`data/inventory.csv` (one row per dataset) and `data/duplicates.csv`.

:::{dropdown} What a `meta.yaml` looks like
:icon: file

```yaml
dataset_id: pmlb__labor
source: pmlb
n: 57
p: 16
class_counts: {0: 20, 1: 37}
minority_n: 20
minority_fraction: 0.350877
family: real_tabular
group_col: null
dropped_id_columns: []
one_hot_columns: []
sha256: 7d6c77b2978093…       # the CSV's bytes; the split manifests pin this
```
:::

```{tip}
**One CSV per dataset directory, and nothing train- or test-shaped beside it.**
QProfiler treats every `*.csv` it finds as a complete dataset and splits it itself. A
stray `labor_train.csv` would be re-split and reported as a dataset of its own, so
`curate.py` refuses to finish if one appears.
```

```{tip}
**Curation is deterministic.** The same sources and `--seed` give byte-identical CSVs,
which is what the recorded sha256 is for. `duplicates.csv` lists datasets that share a
stem across sources (`libsvm__sonar` and `pmlb__sonar`). Treat each such pair as one
cluster in step 4.
```

(bench-synthetic)=

## 2 · Add synthetic datasets (optional)

`benchmark/create_synthetic_datasets.py` writes synthetic datasets in exactly the same
format, into the same tree. The labels are known functions of latent coordinates, so you
know beforehand what each dataset is hard *for*.

```bash
python benchmark/create_synthetic_datasets.py --list            # the catalogue
python benchmark/create_synthetic_datasets.py --out-root $BENCH/data/datasets \
    --shapes all --k 2,5,8 --d 8 --n 400 --seeds 0,1,2
python benchmark/create_synthetic_datasets.py --out-root $BENCH/data/datasets \
    --quantum ql_zz,eng_qiskit --k 2,5,8 --d 4,6,8 --bandwidth 1,0.5,0.25 --n 400 --seeds 0,1,2
```

| Flag | Families | Role written into `meta.yaml` |
|---|---|---|
| `--shapes` | torus, sphere, concentric circles and spheres, checkerboard, random manifold, swiss roll, half moons, parity, permutation parity, simple linear | `shape`, `classical_favoured`, `negative_control` |
| `--quantum` | angle encoding, quantum labels (`ql_zz`, `ql_evo`), engineered kernels (`eng_*`), time evolution (`te`), ground state (`gs_*`), Hamiltonian learning (`hl`) | `positive_control`, `negative_control`, `product_kernel_control`, `difficulty_ladder`, `classically_easy` |

{bdg-primary}`writes` the same `<id>/<id>.csv` + `meta.yaml` (now with `role`,
`matched_arm`, `prediction`, `n_cells`, `train_per_cell`, and for controls
`control_gates`), a `latent.npz` with the latent coordinates and continuous target, and
`data/inventory_synthetic.csv`.

```{tip}
**`k` is each family's own complexity knob**: frequency for the torus, rings for the
circles, bits for parity, evolution time for `te` and `ql_*`. To compare difficulty
across families, use `train_per_cell` in `meta.yaml`. Below about one training point
per label cell, every method is at chance.
```

```{tip}
**Positive controls are gated, and the gates never look at a classical result.** A
quantum positive control is written only if a pilot draw of its generator passes
`benchmark/control_gates.py`:
- G1: the matched feature map entangles;
- G2: the matched kernel is not concentrated;
- G3: the matched kernel alone can learn the labels at this sample size;
- G4: it is far enough from an RBF kernel for any advantage to be possible.

A failure is skipped with its reasons. The thresholds are a pre-registration: fix them
before any result exists (step 4).
```

(bench-splits)=

## 3 · Freeze the splits

`benchmark/make_splits.py` turns each dataset into a **split manifest**: stratified
5-fold cross-validation repeated 3 times (repeat *r* shuffled with seed + *r*). Each
fold's validation rows are the test rows of the next fold of the same repeat, so every
row is tested once per repeat and validated once.

```bash
python benchmark/make_splits.py --datasets $BENCH/data/datasets --out $BENCH/data/splits/v2
```

{bdg-primary}`writes` `data/splits/v2/<id>.json`, 15 splits per dataset.

:::{dropdown} What a split manifest looks like
:icon: file

Shown with the index lists cut short and only the first of the 15 folds; the real file
lists every row index of every fold.

```json
{
  "schema_version": 2,
  "dataset_id": "pmlb__labor",
  "sha256": "7d6c77b297809335b2c445ca5b143f7c7f175231467d29552438905209eda9f9",
  "n": 57, "k": 5, "n_repeats": 3,
  "protocol": "StratifiedKFold", "validation": "next_fold",
  "seed": 42, "repeat_seeds": [42, 43, 44],
  "folds": [
    {"repeat": 0, "fold": 0,
     "train": [3, 4, 5, 6, 8, 9, 10], "val": [4, 12, 13, 16], "test": [0, 1, 2, 7, 11]}
  ]
}
```
:::

```{tip}
**The manifest is the contract.** It pins the CSV's sha256, row count and labels, and a
job refuses to run on a CSV that has changed since. Commit the manifests, or record
their hashes, as soon as the dataset set is final. Datasets whose validation minority
class falls below 3 rows are reported and kept.
```

(bench-holdout)=

## 4 · Draw the hold-out and pre-register

The meta-analysis holds back a random share of the corpus. The meta-model is frozen on
the discovery datasets and then scored **once** on the held-back ones. The draw is by
cluster: every variant of a synthetic family falls on one side, and so do datasets
listed in a cluster map.

```bash
printf 'dataset_id,cluster\nlibsvm__sonar,real__sonar\npmlb__sonar,real__sonar\n' > $BENCH/cluster_map.csv
python benchmark/holdout.py --datasets $BENCH/data/datasets --fraction 0.2 --seed 20261007 \
    --cluster-map $BENCH/cluster_map.csv --out $BENCH/holdout.csv
```

{bdg-primary}`writes` `holdout.csv` (`dataset_id, family, cluster, holdout`) and prints
its sha256.

```{tip}
**Draw once, before any result exists, and do not redraw to get a nicer split.**
`holdout.py` refuses to overwrite its output without `--force`. Write down the rest of
the plan in the same breath, in a `prereg.yaml` with the hashes of the inventory, the
split manifests and the hold-out:
- the gate thresholds, the arms, the trial budget;
- the primary metric, the margin and the FDR level;
- which roles count as controls.
```

(bench-configs)=

## 5 · Write the job YAMLs

`experiments/pilot10/generate_pilot_configs.py --split-mode manifest` writes one QProfiler
YAML per job. A job is one (dataset, embedding, model group, chunk of splits). Every
path inside a YAML is absolute, so a job can run from anywhere.

```bash
cd $REPO/experiments/pilot10
python generate_pilot_configs.py --split-mode manifest \
    --datasets-root $BENCH/data/datasets --split-dir $BENCH/data/splits/v2 \
    --runs-dir $BENCH/runs_cv --run-id full1 \
    --datasets all --splits all --n-trials 30 \
    --splits-per-job 'classical=all,qsvc=all,pqk=5' \
    --embed-above 13 --wall auto
```

| Flag | Meaning |
|---|---|
| `--datasets all` / `--datasets-file F` | every curated dataset with a manifest, or a list (one id per line, or a CSV with `dataset_id`) |
| `--models` | default: qsvc, pqk and the classical group (lr, svc, nb, dt, rf, xgb, catboost, mlp, tabpfn). Add `qnn` or `vqc` by naming them |
| `--n-trials 30` | the same tuning budget for every arm; trial 0 is each arm's default configuration |
| `--splits-per-job` | how many splits one job runs in turn; batching cheap arms saves per-job overhead |
| `--embed-above 13` | datasets wider than 13 features are embedded (PCA and UMAP, 8 components), so every quantum arm stays on the exact statevector simulator |
| `--wall auto` | each job's LSF wall from the measured cost model (3 × expected + 15 min) |

{bdg-primary}`writes` `runs_cv/<run>/<id>/<config>.yaml` and `runs_cv/<run>/MANIFEST.tsv`
(one row per job: dataset, embedding, group, backend, qubits, rows, splits, expected
hours, wall). It also prints the **price** of the run, its expected core-hours.

:::{dropdown} The paths a YAML carries
:icon: file-code

```yaml
config_file_name: 'openml__pc4_pca_i01-05_pqk'
folder_path: '$BENCH/data/datasets/openml__pc4'
file_dataset: ['openml__pc4.csv']
embeddings: ['pca']
n_components: 8
embedding_min_features: 13
embedding_cache: '$BENCH/runs_cv/full1/embeddings'
quantum_model: ['pqk']
split_mode: manifest
split_dir: '$BENCH/data/splits/v2'
splits: [1, 2, 3, 4, 5]
quantum_param_dir: '$BENCH/runs_cv/full1/openml__pc4/quantum_tuned_params/openml__pc4_pca_i01-05_pqk'
kernel_dump_dir: '$BENCH/runs_cv/full1/openml__pc4/kernels/openml__pc4_pca_i01-05_pqk'
```
:::

```{tip}
**In manifest mode a model's `<model>_args` block is read.** Its searched names are
trial 0, and its other keys are fixed for every trial and the refit. Outside manifest
mode a tuned model ignores it entirely; see the warning in the
{doc}`configuration guide <apps/config>`. The quantum arms also search `bandwidth`, the
quantum counterpart of an RBF kernel's gamma.
```

(bench-cache)=

## 6 · Compute the embedding cache, once

Every embedded split is computed **once, before any job runs**. That covers both stages:
the final one (fit on the training rows, applied to the test rows) and the tuning one
(fit on the fit rows, applied to the validation rows). Jobs read these files and never
embed for themselves.

```bash
cd $REPO/experiments/pilot10
PRECOMPUTE_ONLY=1 RUNS=$BENCH/runs_cv/full1 ./submit_runs.sh      # writes, submits nothing
# or directly, e.g. one dataset per cluster job:
python -m qbiocode.apps.qprofiler.embedding_cache $BENCH/runs_cv/full1/openml__pc4/*_pca_*.yaml
```

{bdg-primary}`writes` `runs_cv/<run>/embeddings/emb_<id>_<emb>_8_<split>.npz` and
`…__tune.npz`. Each holds `X_train`, `X_test`, `train_idx`, `test_idx`, a `spec` (dataset
hash, seed, scaling and embedding settings) and a `provenance` record (host, CPU, time).

```{tip}
**This is what keeps every job of a pass on identical features.** Seeded UMAP still
differs between CPU types, so features computed inside the jobs would depend on where
each job landed. A job checks every file it needs against its own settings before
fitting anything, and stops if one is missing or stale.
```

```{tip}
**It is slow to start but never recomputes.** Importing UMAP takes about 90 s, and each
config check takes about 0.3 s. On a large run, compute the cache in parallel cluster
jobs, one per dataset, so no two jobs write the same file.
```

(bench-run)=

## 7 · Run QProfiler on the YAMLs

::::{tab-set}

:::{tab-item} On LSF
```bash
cd $REPO/experiments/pilot10
cp -r $REPO/qbiocode $BENCH/code/ && export PYTHONPATH=$BENCH/code   # freeze the code
HOSTS="$(LIST_HOSTS=Intel_Platinum:128 ./submit_runs.sh | tail -1)"
DRY=1 RUNS=$BENCH/runs_cv/full1 HOSTS="$HOSTS" ./submit_runs.sh      # preview the bsub lines
RUNS=$BENCH/runs_cv/full1 HOSTS="$HOSTS" ./submit_runs.sh            # submit
```

| Variable | Effect |
|---|---|
| `DRY=1` | print what would be written and submitted |
| `PRECOMPUTE_ONLY=1` | write the embedding cache, submit nothing |
| `HOSTS=` / `LIST_HOSTS=` | pin the jobs to one host group / print that group's regex |
| `DATASET=`, `EMB=`, `MODEL=` | submit a subset (anchored regexes on MANIFEST.tsv) |
| `FORCE=1` | resubmit jobs that are done or queued |
| `WALL=`, `MEM=` | override every selected job's wall and memory (GB) |
:::

:::{tab-item} One job by hand
Any YAML is a complete QProfiler run, so a single job needs no scheduler:

```bash
cd $BENCH/runs_cv/full1/pmlb__labor
python -m qbiocode.apps.qprofiler.cli --config-dir=$PWD --config-name=pmlb__labor_pca_i01-15_qsvc
```

The embedding cache must exist first (step 6).
:::

:::{tab-item} Watching
```bash
./status.py --runs-dir $BENCH/runs_cv/full1       # per dataset: done / running / pending / failed
bjobs -noheader -o 'job_name stat run_time' | grep '^p10_full1_'
```
:::
::::

Each job writes into its dataset's folder:

```text
runs_cv/full1/pmlb__labor/
├── results/pmlb__labor_pca_i01-15_qsvc/statevector_simulator_2026-10-07_02-34-44/
│   ├── ModelResults.csv        one row per (split, model): scores, tuned params, 141 complexity measures
│   ├── RawDataEvaluation.csv   complexity of the raw, unembedded dataset
│   ├── results.pkl             per-split summaries: fit/val/train/test indices, trial logs
│   ├── oof/<key>.csv           test-row predictions and scores (row_id, y_true, y_pred, y_score)
│   ├── trials/<key>.csv        every tuning trial: params, validation score, is_default, is_best
│   ├── val_predictions/<key>.csv  validation predictions of every trial
│   └── qprofiler.log
├── kernels/pmlb__labor_pca_i01-15_qsvc/gram_qsvc_opt_pmlb__labor_pca_8_<split>.npz
├── kernels/pmlb__labor_pca_i01-05_pqk/proj_pqk_opt_pmlb__labor_pca_8_<split>.npz
├── quantum_tuned_params/<config>/
└── lsf_logs/<config>.<jobid>.out, .err
```

:::{dropdown} The columns of `ModelResults.csv`
:icon: table

Besides the 141 {doc}`complexity measures <dataset_metrics>`:

| Group | Columns |
|---|---|
| Identity | `Dataset`, `embeddings`, `iteration`, `repeat`, `fold`, `model` |
| Scores (test rows) | `accuracy`, `balanced_accuracy`, `f1_score`, `mcc`, `auc`, `pr_auc`, `time` |
| Tuning | `BestParams_Tuned`, `tuning_metric`, `tuning_score`, `tuning_reused`, `val_auc`, `val_log_loss` |
| Protocol and provenance | `split_mode`, `split_k`, `split_repeats`, `split_seed`, `manifest_sha256`, `dataset_sha256`, `n_fit`, `n_val`, `n_test`, `seed`, `q_seed`, `embed_seed`, `host`, `cpu_model`, `lsf_jobid` |

For example, a tuned qsvc row records
`{'encoding': 'ZZ', 'reps': 1, 'entanglement': 'pairwise', 'C': 24.1, 'bandwidth': 0.43, …}`.
:::

```{tip}
**Freeze the code and pin the hosts.** `bsub` passes your environment to the job, so
exporting `PYTHONPATH` to a frozen copy keeps later edits away from running jobs. UMAP,
TabPFN and the variational arms vary with the CPU type, so pin one host group, and read
the real CPU from the `cpu_model` column.
```

```{tip}
**A wall is a kill limit, not a reservation.** If a job is slower than its estimate,
extend it in place with `bmod -W 18:00 <jobid>`, so no work is lost. To rerun a failed
job, move its `results/<config>` aside, then
`FORCE=1 DATASET='<id>' WALL=… ./submit_runs.sh`. `submit_runs.sh` sets
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` in every job, because TabPFN otherwise refuses more
than 1000 training rows on a CPU.
```

(bench-collate)=

## 8 · Collect

```bash
./collate_results.py --runs-dir $BENCH/runs_cv/full1 --out-dir $BENCH/collated
```

{bdg-primary}`writes` `collated/ModelResults.csv`, `oof.csv`, `trials.csv`,
`val_predictions.csv`, `RawDataEvaluation.csv`, `results.pkl` and `collate_manifest.csv`
(which job each row came from).

```{tip}
**Collate checks the run; take a refusal seriously.** It refuses duplicate rows, and
jobs of one pass whose features differ (relative tolerance 1e-6). A mismatch is a
determinism bug: find the unseeded step, fix it, and rerun the affected passes.
`--allow-mismatch` is for inspection only. `--out-dir` must lie outside `--runs-dir`.
```

(bench-winners)=

## 9 · Winners

```python
import pandas as pd
from qbiocode.utils.fair_selection import select_winners

results = pd.read_csv("collated/ModelResults.csv")
inv = pd.read_csv("data/inventory_synthetic.csv")
controls = [f"{d}.csv" for d, r in zip(inv.dataset_id, inv.role) if r.endswith("_control")]
report = select_winners(results, metric="balanced_accuracy", selection="validation",
                        k=5, margin=0.0, fdr=0.10, epsilon=0.05, controls=controls)
report.per_dataset          # delta, CI, p-value and verdict per dataset
report.selection            # which arm was chosen on each fold, and why
```

1. **Pick on validation.** For each (dataset, embedding, fold) and each side, the arm
   with the best validation score is chosen. A tie is broken on validation AUC.
2. **Score on test.** Δ = classical − quantum on that fold's test rows.
3. **Test per dataset.** A repeated-CV t-test, corrected for overlapping training
   sets (Nadeau–Bengio; r = 1/(k−1), df = kR−1), against the pre-registered margin.
4. **Correct for multiplicity.** Benjamini–Hochberg over the discovery datasets; the
   controls form their own Holm family.

```{tip}
**Report balanced accuracy first; use MCC and PR-AUC on imbalanced data.** Validation
picks the arm and the metric only scores it, so changing the reported metric never
changes which model was selected. Judge each control against its own pre-registered
prediction, not against the discovery results.
```

(bench-kernels)=

## 10 · Kernel geometry (Huang et al.)

A winner table says *whether* the quantum arm won. The kernel dumps let you ask
*whether the quantum kernel sees structure a classical kernel cannot reach*. Every
tuned qsvc and pqk refit writes its kernel for each split:

| File | Contents |
|---|---|
| `gram_qsvc_opt_<key>.npz` | `K_train` (train × train), `K_test` (test × train), `X_train`, `y_train`, `y_test` |
| `proj_pqk_opt_<key>.npz` | `Z_train`, `Z_test` (one-qubit Pauli expectations, all X, then Y, then Z), `X_train`, `y_train`, `y_test`, `best_params` |

```python
import numpy as np
from qbiocode.utils import kernel_diagnostics as kd

z = np.load("kernels/pmlb__labor_pca_i01-15_qsvc/gram_qsvc_opt_pmlb__labor_pca_8_1.npz")
Kq, X, y = z["K_train"], z["X_train"], 2 * z["y_train"] - 1
sq = ((X[:, None] - X[None]) ** 2).sum(-1)
Kc = np.exp(-sq / np.median(sq[sq > 0]))             # an RBF classical kernel
kd.kta(kd.normalize_trace(Kq), y)                   # kernel-target alignment
kd.geometric_separation(Kc, Kq, lam=1e-3)           # g(Kc || Kq)
kd.kernel_report(Kc, Kq, y)                         # both alignments and g over a lambda sweep
```

For the whole run, `./analyze_pilot.py --results-root $BENCH/collated --kernels-root
$BENCH/runs_cv/full1 --config-dir $BENCH/runs_cv/full1` adds a kernel-geometry table
beside the accuracy table.

```{tip}
**Large g is room for an advantage, not evidence of one.** g near 1 means the classical
kernel already spans the quantum one, so a quantum win there is about tuning, not
quantum structure. A fidelity kernel built from a Z-only feature map is a product of
one-qubit kernels with an exact classical twin. Kernels estimated from shots are
eigen-clipped to be positive semi-definite, so their diagonal can sit slightly off 1.
```

(bench-meta)=

## 11 · Meta-analysis: which properties predict a quantum win

The meta-features are the 141 {doc}`dataset complexity measures <dataset_metrics>`
written beside every score. The response is the per-fold contrast Δ from step 9.
`experiments/pilot10/meta_analysis.ipynb` runs the analysis end to end. Set its
parameters cell (`RESULTS_CSV`, `CONFIG_DIR`, `METRIC`, the screen thresholds, the FDR
levels) and freeze it before the results arrive. The functions it uses are in
`qbiocode.utils.meta_regression`:

| Stage | What it does | Function |
|---|---|---|
| Quality | canonical per-pass values, reliability, host noise | `canonical_meta_features`, `feature_reliability` |
| Screen | outcome-blind, ending in a VIF ceiling | `screen_features`, `variance_inflation` |
| Design | one row per (dataset, embedding, fold), clustered on the dataset | `build_design` |
| Tests | one model per feature with a wild cluster bootstrap; joint ridge tests | `marginal_tests`, `joint_tests` |
| Calibration | block-permuted null responses, power and minimum detectable effect | `block_permutations` |

```{tip}
**The unit of independence is the dataset, not the row.** Folds are resamples of the
same rows, and a dataset's PCA and UMAP passes share their splits. So every standard
error clusters on the dataset, with G − 1 degrees of freedom. More folds tighten each
dataset's contrast; only more datasets tighten a meta-feature's coefficient.
```

```{tip}
**Keep the plan honest.** Screen features without looking at the outcome, and score the
hold-out from step 4 once, after the meta-model is frozen. Leave the control datasets
out of both sets. Treat label-reading features, such as `task.graph_hf_mass_z`, with
care when the response is itself built from the labels.
```

(bench-checklist)=

## Checklist

- [ ] `data/datasets` curated (and synthetic families generated); `inventory*.csv` saved
- [ ] `data/splits/v2` written, and its hashes recorded or committed
- [ ] gate thresholds, arms, budget, metric, margin, FDR and control roles written in `prereg.yaml`
- [ ] hold-out drawn once (`holdout.csv` and its sha256 in `prereg.yaml`)
- [ ] configs generated; `MANIFEST.tsv` priced and reviewed
- [ ] embedding cache complete
- [ ] code frozen (`PYTHONPATH`), hosts pinned, jobs submitted
- [ ] every job DONE in `status.py`; failures rerun after moving their results aside
- [ ] `collate_results.py` accepted the run
- [ ] winners computed; controls judged against their predictions
- [ ] kernel geometry and meta-analysis run; hold-out scored once

## Where everything lives

| Artefact | Written by | Path | Read by |
|---|---|---|---|
| Curated datasets | `benchmark/curate.py`, `create_synthetic_datasets.py` | `data/datasets/<id>/` | every later step |
| Split manifests | `benchmark/make_splits.py` | `data/splits/v2/<id>.json` | configs, every job |
| Hold-out | `benchmark/holdout.py` | `holdout.csv` | the meta-analysis |
| Job YAMLs | `generate_pilot_configs.py` | `runs_cv/<run>/<id>/*.yaml`, `MANIFEST.tsv` | cache, `submit_runs.sh`, jobs |
| Embedding cache | `qbiocode.apps.qprofiler.embedding_cache` | `runs_cv/<run>/embeddings/` | every embedded job |
| Job results | QProfiler (`qbiocode.apps.qprofiler.cli`) | `runs_cv/<run>/<id>/results/` | `status.py`, `collate_results.py` |
| Kernel dumps | qsvc and pqk refits | `runs_cv/<run>/<id>/kernels/` | `analyze_pilot.py`, `kernel_diagnostics` |
| Collated tables | `collate_results.py` | `collated/` | winners, meta-analysis |

```{seealso}
- {doc}`Dataset complexity metrics <dataset_metrics>`: what each of the 141 measures means.
- {doc}`Simulated quantum datasets <quantum_datasets>`: the quantum families in depth.
- {doc}`QProfiler <apps/profiler>` and its {doc}`configuration guide <apps/config>`.
```
