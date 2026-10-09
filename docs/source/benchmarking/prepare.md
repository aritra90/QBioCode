# Prepare the data

Four steps turn source data into frozen inputs. Nothing here fits a model, and
everything here is deterministic.

(bench-curate)=

## 1 · Curate the real datasets

`benchmark/curate.py` reads a source tree with `pmlb_data/`, `openMLCC18/` and
`libsvm_data/` folders. It drops ID and constant columns, one-hot encodes categorical
columns, maps the label to {0, 1}, and shuffles the rows once. Then it writes one numeric
CSV per dataset, label last, with a `meta.yaml` that records every choice and the CSV's
sha256.

```bash
python benchmark/curate.py --source-root /path/to/sources --out-root $BENCH/data/datasets
```

**Writes:** `data/datasets/<id>/<id>.csv` and `meta.yaml`; `data/inventory.csv`, one row
per dataset; and `data/duplicates.csv`, the datasets that share a name across sources.

:::{dropdown} A `meta.yaml`
:icon: file

```yaml
dataset_id: pmlb__labor
source: pmlb
n: 57
p: 16
class_counts: {0: 20, 1: 37}
minority_fraction: 0.350877
family: real_tabular
dropped_id_columns: []
one_hot_columns: []
sha256: 7d6c77b297809335…   # the CSV's bytes; the split manifests pin it
```
:::

```{tip}
Keep **one CSV per dataset directory**, and nothing train- or test-shaped beside it.
QProfiler treats every `*.csv` it finds as a complete dataset, so a stray
`labor_train.csv` would be re-split and reported as a dataset of its own. `curate.py`
refuses to finish if one appears.
```

(bench-synthetic)=

## 2 · Add synthetic datasets (optional)

`benchmark/create_synthetic_datasets.py` writes synthetic datasets in the same format,
into the same tree. Their labels are known functions of latent coordinates, so you know
in advance what each dataset is hard *for*. `--list` prints the catalogue.

```bash
python benchmark/create_synthetic_datasets.py --out-root $BENCH/data/datasets \
    --shapes all --k 2,5,8 --d 8 --n 400 --seeds 0,1,2
python benchmark/create_synthetic_datasets.py --out-root $BENCH/data/datasets \
    --quantum ql_zz,eng_qiskit --k 2,5,8 --d 4,6,8 --bandwidth 1,0.5,0.25 --n 400 --seeds 0,1,2
```

| Flag | Families |
|---|---|
| `--shapes` | torus, sphere, concentric circles and spheres, checkerboard, random manifold, swiss roll, half moons, parity, permutation parity, simple linear |
| `--quantum` | angle encoding, quantum labels (`ql_zz`, `ql_evo`), engineered kernels (`eng_*`), time evolution (`te`), ground state (`gs_*`), Hamiltonian learning (`hl`) |

`k` is each family's complexity knob, for example the torus frequency, or the
evolution time for `ql_*`. To compare difficulty across families, use `train_per_cell` in
`meta.yaml`. Each dataset also records its `role` (shape, positive or negative control,
and so on), and, for controls, the matched arm and a pre-registered prediction.

**Writes:** the same `<id>/<id>.csv` and `meta.yaml`, plus `latent.npz` (the latent
coordinates and continuous target) and `data/inventory_synthetic.csv`.

```{tip}
**Positive controls are gated, and the gates never look at a classical result.** A
quantum control is written only if a pilot draw of its generator clears
`benchmark/control_gates.py`:
- the matched map entangles;
- its kernel is not concentrated;
- that kernel alone learns the labels;
- it is far enough from an RBF kernel for any advantage to be possible.

The thresholds are part of the pre-registration (step 4).
```

(bench-splits)=

## 3 · Freeze the splits

`benchmark/make_splits.py` writes one **split manifest** per dataset: stratified 5-fold
cross-validation, repeated 3 times. Each fold's validation rows are the test rows of the
next fold, so every row is tested once and validated once per repeat.

```bash
python benchmark/make_splits.py --datasets $BENCH/data/datasets --out $BENCH/data/splits/v2
```

**Writes:** `data/splits/v2/<id>.json`, 15 splits per dataset.

:::{dropdown} A split manifest (lists cut short)
:icon: file

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
their hashes, as soon as the dataset set is final.
```

(bench-holdout)=

## 4 · Draw the hold-out and pre-register

The meta-analysis holds back a random share of the corpus. The meta-model is frozen on
the rest, then scored **once** on the held-back datasets. The draw is by cluster:
- every variant of a synthetic family falls on the same side;
- so do datasets grouped in a cluster map, such as the pairs listed in `duplicates.csv`.

First write the cluster map. `benchmark/review_duplicates.py` emits one covering exactly
the corpus on disk, with the `dataset_id,cluster` columns `holdout.py` requires — which
`experiments/cluster_map_draft.csv` does not have, and does not cover the synthetic corpus
with at all. Read its report before using the file: it prints the resulting **G**, every
cluster holding more than one dataset, and any dataset left alone in a cluster.

```bash
python benchmark/review_duplicates.py --datasets $BENCH/data/datasets \
    --emit-cluster-map $BENCH/cluster_map.csv
python benchmark/holdout.py --datasets $BENCH/data/datasets --fraction 0.2 --seed 20261007 \
    --cluster-map $BENCH/cluster_map.csv --out $BENCH/holdout.csv
```

```{warning}
`--cluster-map` **replaces** the clustering `holdout.py` derives from each `meta.yaml`,
rather than adding to it, so a map that mis-groups the synthetic families is worse than
no map at all. Check the `-> N clusters` line before passing the file on.
```

**Writes:** `holdout.csv` (`dataset_id, family, cluster, holdout`), and prints its sha256.

```{tip}
**Draw once, before any result exists, and never redraw for a nicer split.** At the same
time, write a `prereg.yaml` with the plan and the hashes of the inventory, the
manifests and the hold-out:
- the gate thresholds, the arms and the trial budget;
- the primary metric, the margin and the FDR level;
- which roles count as controls.
```
