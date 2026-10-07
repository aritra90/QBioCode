# The QBioCode classical-vs-quantum benchmark: workflow

This folder builds the benchmark's inputs. `experiments/pilot10/` turns them into LSF jobs
and collects the results, and `qbiocode.utils.fair_selection` and
`experiments/pilot10/meta_analysis.ipynb` analyse them. Every command below runs from the
repository root, with the project interpreter (`/dccstor/boseukb/Q/envs/qbc/bin/python` on
CCC). On a login node, prefix heavy commands with `OMP_NUM_THREADS=1`, because the node
kills any process tree that uses more than 8 cores.

```
curate.py ─┐                                  ┌─ holdout.py (meta-analysis hold-out)
           ├─> datasets/<id>/{<id>.csv, meta.yaml} ─> make_splits.py ─> splits/v2/<id>.json
create_synthetic_datasets.py ─┘                                            │
                                                                           v
generate_pilot_configs.py --split-mode manifest  ─>  runs_cv/<run>/{<id>/*.yaml, MANIFEST.tsv}
submit_runs.sh  (embedding cache, then one bsub per job)  ─>  results/, oof/, trials/, val_predictions/
status.py  ─  collate_results.py  ─>  collated/ModelResults.csv (+ oof, trials, val_predictions)
select_winners(selection='validation')  ─  meta_analysis.ipynb
```

## 1. Datasets

All datasets share one format: `datasets/<id>/<id>.csv` (features, then an integer `label`
in {0, 1} as the last column, rows shuffled once) and a `meta.yaml` holding `n`, `p`, the
class counts, `family`, `group_col` and the `sha256` of the CSV bytes. The split manifests
pin that sha256, so a CSV that changes afterwards is refused at run time.

**Real datasets.** `curate.py` reads the source tree (PMLB, OpenML, libsvm, ...), drops
known id columns and constant columns, one-hot encodes categoricals, maps labels to 0/1 and
writes `datasets/` plus `inventory.csv`:

```bash
python benchmark/curate.py --out-root benchmark/datasets
```

**Synthetic datasets.** `create_synthetic_datasets.py` generates shape families
(`--shapes`) and quantum families (`--quantum`). Each flag takes a comma list or `all`, and
every family is built at every combination of `--k`, `--d`, `--n` and `--seeds` it accepts.
Combinations a family cannot take are skipped with the reason.

```bash
python benchmark/create_synthetic_datasets.py --list          # the catalogue
python benchmark/create_synthetic_datasets.py --out-root benchmark/datasets \
    --shapes torus,sphere,concentric_circles,half_moons,checkerboard \
    --quantum angle_encoding,ql_zz,ql_evo,eng_qiskit,eng_unit,te \
    --k 2,5,8 --d 4,6,8 --bandwidth 1,0.5,0.25 --n 400 --seeds 0,1,2
```

- `k` is each family's own complexity knob: frequency for the torus, rings for the
  circles, bits for parity, qubits in the product for angle encoding, and evolution time
  τ = k/2 for `te` and `ql_*`. To compare complexity across families, use `n_cells` and
  `train_per_cell` in `meta.yaml`. Below about one training point per cell, every method
  is at chance.
- `d` is the feature count, which is also the qubit count when the dataset is not
  embedded. The ground-state family writes 2⌈d/2⌉−1 features, and `te` can draw at most 2^d
  distinct rows.
- Labels depend only on latent coordinates. `latent.npz` stores those coordinates and the
  continuous target `F`, so a label can be re-derived independently
  (`tests/test_synthetic_datasets.py` does this). For the shape families and
  `angle_encoding`, the embedding map is fixed per `(family, d, k)`, so the seeds are
  samples of one manifold. The `ql_*` families draw their concept (the Heisenberg
  Hamiltonian) from a `concept_seed` fixed per `(family, d, k)`, so their seeds are samples
  of one concept too. **The other `qbiocode.data_generation` families (`gs_*`, `te`, `hl`,
  `eng_*`) draw their concept from the data seed, so each of their seeds is a different
  concept.**
- `--bandwidth` (the `ql_*` families) encodes `b * x` and still writes `x`. The id gains
  `_bw<b>`, and the matched qsvc reaches the labels only at that `bandwidth` (section 4).
- Every dataset records its `role`:
  - `shape`, `positive_control`, `negative_control`, `product_kernel_control`,
    `classically_easy`, `difficulty_ladder` or `classical_favoured`;
  - for a control, also `matched_arm` and a pre-registered `prediction`.

  For example, `eng_unit` is matched to *pqk with data_map unit, reps 2, linear, gamma 1*.
  Its labels are built for a projected kernel, so qsvc is not its matched arm.
  `angle_encoding` is a `product_kernel_control`: its kernel is a product of one-qubit
  kernels with an exact classical twin, so a win on it says nothing quantum.
- **Positive controls are gated, without looking at any classical result**
  (`control_gates.py`). A configuration of `ql_zz`, `eng_qiskit` or `eng_unit` is written
  only if a pilot draw of its generator (seed `PILOT_SEED`, never a data seed) clears:
  - G1, the matched map entangles;
  - G2, the matched kernel is not concentrated (participation ratio / n, mean off-diagonal);
  - G3, the matched kernel alone learns the labels at this n (nested-CV balanced accuracy);
  - G4, its geometric difference from an RBF kernel, g / √n_train, leaves room for any
    advantage.

  No gate fits a classical model or reads a benchmark row, so the choice is not circular.
  A failure is skipped with its reasons; `--keep-failed-controls` writes it as
  `gate_rejected`, which belongs in neither the control family nor the discovery family.
  `meta.yaml` records the gate values, thresholds and version under `control_gates`.
  **The thresholds are the pre-registration: fix them before any benchmark result
  exists.** The gates guarantee that the matched arm can learn the labels, not that it
  beats tree ensembles or TabPFN, and the prediction says so. Example (k=2, n=400): the
  dry run's `ql_zz` at 8 qubits and bandwidth 1 is rejected (concentrated, matched kernel
  0.795), while 8 qubits at bandwidth 0.25 is accepted.
- The synthetic rows are listed in `inventory_synthetic.csv`. `curate.py` owns
  `inventory.csv`.

## 2. Splits

`make_splits.py` freezes the outer folds of every dataset as a schema-2 manifest:

- stratified 5-fold × 3 repeats (repeat r is shuffled with seed + r);
- each fold's validation rows are the test rows of the next fold of the same repeat;
- the dataset's sha256 is pinned.

```bash
python benchmark/make_splits.py --datasets benchmark/datasets --out benchmark/splits/v2
```

Datasets whose validation minority class falls below 3 rows are reported and kept.
`datasets/` and `splits/` are gitignored. Once a set is chosen, freeze it: commit the
manifests, or record their sha256s.

## 3. The meta-analysis hold-out, drawn before any result

```bash
python benchmark/holdout.py --datasets benchmark/datasets --fraction 0.2 --seed <recorded seed> \
    --out benchmark/holdout.csv
```

The draw is by cluster:
- a synthetic family counts as one cluster across all its variants;
- each real dataset is its own cluster, unless `--cluster-map` says otherwise.

The script refuses to overwrite an existing draw. Every dataset is still run; the
discovery/hold-out split exists only in the analysis.

## 4. Job configs

```bash
python experiments/pilot10/generate_pilot_configs.py --split-mode manifest \
    --datasets-root benchmark/datasets --split-dir benchmark/splits/v2 \
    --runs-dir <runs> --run-id full1 \
    --datasets all                       # or --datasets-file <list or inventory CSV>
    --splits all --n-trials 30 \
    --splits-per-job 'classical=all,qsvc=all,pqk=5' \
    --embed-above 13 --wall auto
```

- **Arms.** Without `--models`, a run gets qsvc, pqk and the classical group (lr, svc, nb,
  dt, rf, xgb, catboost, mlp, tabpfn). qnn and vqc are left out: qnn was the weakest and
  costliest arm on ctrl1, and its noisy validation scores pulled the quantum side down.
  Both stay available by naming them, e.g. `--models qsvc,pqk,qnn`.
- **`--embed-above 13`** embeds every dataset wider than 13 features (pca and umap, to 8
  components), which is every width the statevector cannot take, so no job needs MPS.
  The default, 20, keeps 14-20 features unembedded on MPS; in the 84-dataset corpus those 9
  datasets alone would cost about 1.48M core-h.

- **One job per (dataset, embedding, model group, chunk of splits).** The classical group
  (lr, svc, nb, dt, rf, xgb, catboost, mlp, tabpfn) is one job.
  `--splits-per-job classical=all` runs all 15 splits of a pass in one classical job. A
  classical job takes minutes per split, so this removes most of the jobs and their
  per-job overhead.
- **Embeddings.** A dataset with more than 20 features gets pca and umap at 8 components.
  Otherwise its features go into the models directly, one qubit per feature (statevector
  up to 13 qubits, MPS above).
- **`--wall`** is the LSF `-W` kill limit, not a reservation. It takes one `H:MM`,
  per-group values (`classical=1:00,qsvc=3:00`), or `auto`.
  - `auto` prices each job from `experiments/pilot10/cost_model.py`
    (`manifest_job_hours`): rows, qubits, backend, n_trials and the splits in the job.
  - The wall is 3 × expected + 15 min, between 0:30 and 72:00.
  - MANIFEST.tsv then carries `exp_h` (expected) and `bound_h` (the wall).
  - The laws were fitted on controlled run ctrl1 (2026-10-05) and reproduce its 300 jobs
    within 6%. Jobs outside that calibration (more than 267 rows, or other widths) are
    flagged in `cost_extrapolated`.
  - Recalibrate with `cost_model.calibrate_manifest` once a run covers them.
  - Jobs expected to exceed the 72 h cap are listed at generation time.
- **The quantum arms search `bandwidth`** (qsvc and pqk, π/32 to 2π on a log scale). It
  multiplies the [0, 1] features before the feature map: the quantum counterpart of the
  RBF SVC's gamma. Without it, every quantum kernel ran at one fixed angle range. On the
  dry run's 8-feature torus, the Z map gained 0.06 balanced accuracy from it (0.854 at 1,
  0.914 with the bandwidth chosen on validation), still a product kernel.
- **`<model>_args` is read here because these configs run in manifest mode.** Its searched
  names are trial 0 and its other keys are fixed for every trial and the refit. **Outside
  manifest mode (QProfiler's default `split_mode: internal`) a tuned model does NOT read
  `<model>_args` at all;** see the warning in `docs/source/apps/config.md`.
- Every config runs `split_mode: manifest`: each trial is tuned on the fit rows and
  scored on the validation rows, then the refit is on the training rows and the test runs
  once. There is no parameter freezing, every arm gets the same n_trials, and trial 0 is
  the arm's default config. `MANIFEST.tsv` records every choice.

## 5. Embeddings and submission

```bash
cd experiments/pilot10
DRY=1 RUNS=<runs>/full1 ./submit_runs.sh                     # what it would write and submit
PRECOMPUTE_ONLY=1 RUNS=<runs>/full1 ./submit_runs.sh         # write the embedding cache only
HOSTS="$(LIST_HOSTS=Intel_Platinum:128 ./submit_runs.sh | tail -1)" RUNS=<runs>/full1 ./submit_runs.sh
```

- **Embeddings are computed once.** Every embedded split is written before submission,
  final and tuning stage separately, to `<runs>/<run>/embeddings`. The YAMLs point there,
  and a job never embeds for itself. Seeded UMAP differs between CPU types, so this is what
  keeps the jobs of one pass on identical features. `collate_results.py` verifies it.
- **The cache step is slow, but it does not recompute anything.** Importing UMAP costs
  about 90 s per call, and checking costs about 0.3 s per config (Hydra composes each one).
  So a 5,000-job run spends roughly half an hour here before it submits anything. Run it
  once with `PRECOMPUTE_ONLY=1`; the submit then only confirms that every file is current.
- **Pin hosts.** UMAP, tabpfn and the variational arms vary by CPU type, so pin to one
  host group. The LSF model name is a label only: the 128-cpu "Intel_Platinum" hosts are
  AMD EPYC 7763. The `cpu_model` column of the results records what actually ran.
- **Freeze the code.** For a long run, export `PYTHONPATH` to a frozen copy of `qbiocode/`
  before submitting (bsub passes the environment to the jobs), so later edits cannot reach
  running jobs.

## 6. Tracking and collecting

```bash
./status.py --runs-dir <runs>/full1
./collate_results.py --runs-dir <runs>/full1 --out-dir <runs>/full1/collated
```

`collate_results.py` refuses duplicate keys and jobs that trained on different features. It
merges the `oof/`, `trials/` and `val_predictions/` sidecars, and fills the validation
tie-break columns (`val_auc`, `val_log_loss`) for runs written before they existed.

When it refuses, it names the passes and the column that differs. That is a determinism
bug, not noise to waive: find the unseeded draw, fix it, and rerun the affected passes
(`FORCE=1 DATASET='<id>' RUNS=... ./submit_runs.sh`, after moving their old `results/`
aside). Use `--allow-mismatch` only to inspect. The synthetic dry run of 2026-10-06 hit
this once: Isomap's ARPACK start vector came from numpy's global RNG, and it is now seeded.

## 7. Winners

```python
from qbiocode.utils.fair_selection import select_winners
rep = select_winners(results, metric="balanced_accuracy", epsilon=0.05, selection="validation",
                     k=5, margin=0.0, fdr=0.10, controls=[...])
```

1. **Pick on validation.** Per (dataset, embedding, fold) and side, the arm with the best
   validation score is chosen. A tie is broken on the refit trial's validation AUC; arms
   still tied are averaged.
2. **Score on test.** The chosen arms are scored on that fold's test rows. Δ = classical −
   quantum per fold.
3. **Per-dataset test.** The corrected repeated-CV t (Nadeau–Bengio / Bouckaert–Frank),
   with r = 1/(k−1) and df = kR−1, tested against the pre-registered margin.
4. **Multiple testing.** BH across the discovery datasets in both directions; controls
   (the synthetic `*_control` roles) form their own Holm family.

Report balanced accuracy as the primary metric. Use MCC and PR-AUC as secondaries for
imbalanced data: validation picks the arm, the metric only scores it.

## 8. Meta-analysis

`experiments/pilot10/meta_analysis.ipynb`: set the parameters cell (`RESULTS_CSV`,
`CONFIG_DIR`, the screen thresholds, FDR levels) and freeze it before the results arrive.

1. **Screen.** It screens meta-features without looking at the outcome, ending in a VIF
   ceiling.
2. **Primary test.** One model per feature, with a wild cluster bootstrap-t.
3. **Ridge.** A held-out (leave-one-cluster-out) ridge test, and the pairs-bootstrap ridge
   reported beside it.
4. **Checks.** It calibrates every test on block-permuted null responses and reports
   power.

With few datasets the cluster-robust tests are not identified: their p-values come out
NaN, with a warning, instead of 0. The hold-out from step 3 is scored once, after the
meta-model is frozen.

## Known limits (2026-10-06)

- **MPS qsvc is quadratic in rows.** Datasets with 14–20 features stay unembedded, so qsvc
  runs on MPS at one qubit per feature. In the 84-dataset corpus, 7 such datasets (540–1600
  rows) are expected to need far more than 72 h per job. Decide before a full run: embed
  datasets with more than 13 features, run pqk only on them, or drop them.
- **qnn** is the weakest and costliest arm on the controlled run, and its noisy validation
  scores pull the quantum side down. The recommendation is `--models qsvc,pqk` for the
  corpus.
- **`/tmp` is local to each login node.** Keep run and work directories on `/dccstor`.
