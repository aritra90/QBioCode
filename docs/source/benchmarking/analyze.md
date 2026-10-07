# Collect and analyze

Four steps turn the job outputs into results: merge them, pick the winners, look at the
kernels, and ask which dataset properties predict the outcome.

(bench-collate)=

## 8 · Collect

```bash
./collate_results.py --runs-dir $BENCH/runs_cv/full1 --out-dir $BENCH/collated
```

**Writes:** `collated/ModelResults.csv`, `oof.csv`, `trials.csv`, `val_predictions.csv`,
`RawDataEvaluation.csv` and `collate_manifest.csv` (which job each row came from).

```{tip}
Collate refuses duplicate rows, and jobs of one pass whose features differ. A
mismatch is a determinism bug: fix it and rerun the affected passes, rather than
waiving it with `--allow-mismatch`. The output directory must lie outside the runs
directory.
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
report.per_dataset    # delta, confidence interval, p-value and verdict per dataset
```

1. **Pick on validation.** For each fold and each side, choose the arm with the best
   validation score, breaking ties on validation AUC.
2. **Score on test.** Δ = classical − quantum, on that fold's test rows.
3. **Test each dataset.** Use a repeated-CV t-test, corrected for overlapping training
   sets (r = 1/(k−1), df = kR−1), against the pre-registered margin.
4. **Correct for multiplicity.** Apply Benjamini–Hochberg across the datasets; the
   controls form their own Holm family.

```{tip}
Report balanced accuracy first, with MCC and PR-AUC on imbalanced data. Validation
picks the arm, so changing the reported metric never changes which model was chosen.
```

(bench-kernels)=

## 10 · Kernel geometry (Huang et al.)

A winner table says *whether* the quantum arm won. The kernels tell you whether the
quantum kernel sees structure a classical kernel cannot reach. Every tuned qsvc and pqk
refit writes its kernel for each split:

| File | Contents |
|---|---|
| `gram_qsvc_opt_<key>.npz` | `K_train`, `K_test`, `X_train`, `y_train`, `y_test` |
| `proj_pqk_opt_<key>.npz` | `Z_train`, `Z_test` (one-qubit Pauli expectations), `X_train`, `y_train`, `y_test`, `best_params` |

```python
import numpy as np
from qbiocode.utils import kernel_diagnostics as kd

z = np.load("kernels/pmlb__labor_pca_i01-15_qsvc/gram_qsvc_opt_pmlb__labor_pca_8_1.npz")
Kq, X, y = z["K_train"], z["X_train"], 2 * z["y_train"] - 1
sq = ((X[:, None] - X[None]) ** 2).sum(-1)
Kc = np.exp(-sq / np.median(sq[sq > 0]))       # an RBF classical kernel
kd.geometric_separation(Kc, Kq, lam=1e-3)     # g(Kc || Kq)
kd.kernel_report(Kc, Kq, y)                   # both alignments, and g over a lambda sweep
```

For the whole run, `./analyze_pilot.py --results-root $BENCH/collated --kernels-root
$BENCH/runs_cv/full1` adds a kernel table beside the accuracy table.

```{tip}
**A large g is room for an advantage, not evidence of one.** g near 1 means the
classical kernel already spans the quantum one. A fidelity kernel built from a Z-only
map is a product of one-qubit kernels, with an exact classical twin.
```

(bench-meta)=

## 11 · Meta-analysis

The meta-features are the 141 {doc}`dataset complexity measures <../dataset_metrics>`
written beside every score. The response is the per-fold contrast Δ from step 9.
`experiments/pilot10/meta_analysis.ipynb` runs the analysis. Set its parameters cell
(the results file, the metric, the screen thresholds, the FDR levels) before the results
arrive. It calls these functions from `qbiocode.utils.meta_regression`:

| Stage | What it does | Function |
|---|---|---|
| Quality | per-pass values, reliability, host noise | `canonical_meta_features`, `feature_reliability` |
| Screen | outcome-blind, ending in a VIF ceiling | `screen_features`, `variance_inflation` |
| Design | one row per fold, clustered on the dataset | `build_design` |
| Tests | per-feature wild cluster bootstrap, joint ridge | `marginal_tests`, `joint_tests` |
| Calibration | block-permuted nulls, power | `block_permutations` |

```{tip}
**The unit of independence is the dataset, not the row.** Standard errors cluster on
the dataset, so only more datasets, not more folds, tighten a meta-feature's
coefficient.

Score the hold-out once, after the meta-model is frozen, and leave the control
datasets out of both sets. Be careful with label-reading features such as
`task.graph_hf_mass_z`, because the response is built from the same labels.
```
