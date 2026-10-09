(benchmarking)=

# Benchmarking

The benchmark answers two questions over a whole corpus of datasets. **Does the best
quantum model beat the best classical one on each dataset?** And **which dataset
properties move that verdict?** It runs in eleven steps, grouped in three stages. Each
step writes files that the next one reads, and everything the result depends on is
pinned by a hash, so a collaborator who reruns the commands gets the same splits,
features and configurations.

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-mb-4

:::{grid-item-card} 1 · Prepare the data
:link: benchmarking/prepare
:link-type: doc
Curate the real datasets, add synthetic ones, freeze the splits, draw the hold-out.
+++
Steps 1–4
:::

:::{grid-item-card} 2 · Configure and run
:link: benchmarking/run
:link-type: doc
Write one YAML per job, compute the embeddings once, run QProfiler.
+++
Steps 5–7
:::

:::{grid-item-card} 3 · Collect and analyze
:link: benchmarking/analyze
:link-type: doc
Merge the outputs, pick the winners, read the kernels, run the meta-analysis.
+++
Steps 8–11
:::
::::

## Before you start

The benchmark tools live in a QBioCode **source checkout**, not in the wheel:
`benchmark/` builds the inputs, and `experiments/pilot10/` writes, submits and collects
the jobs.

```bash
git clone https://github.com/qiskit-community/QBioCode.git && cd QBioCode
pip install -e ".[apps]"
export REPO=$PWD                  # every command in this section runs from here
export BENCH=/scratch/me/bench1   # the run root, with room for about 10 GB
```

```{tip}
On a shared login node, prefix heavy commands with `OMP_NUM_THREADS=1`. Numerical
libraries start one thread per core, and many clusters kill a login process that uses
more than a few.
```

## Where everything lands

```text
$BENCH/
├── data/datasets/<id>/        <id>.csv, meta.yaml          steps 1–2
├── data/splits/v2/            <id>.json                    step 3
├── holdout.csv, prereg.yaml                                step 4
├── runs_cv/<run>/
│   ├── MANIFEST.tsv           one row per job              step 5
│   ├── embeddings/            emb_*.npz                    step 6
│   └── <id>/                  *.yaml, results/, kernels/   step 7
└── collated/                  ModelResults.csv, …          step 8
```

## The steps at a glance

| | Step | Command | Writes |
|---|---|---|---|
| 1 | Curate | `benchmark/curate.py` | `data/datasets/<id>/` |
| 2 | Synthesize (optional) | `benchmark/create_synthetic_datasets.py` | `data/datasets/<id>/` |
| 3 | Split | `benchmark/make_splits.py` | `data/splits/v2/` |
| 4 | Hold out | `benchmark/holdout.py` | `holdout.csv` |
| 5 | Configure | `generate_pilot_configs.py --split-mode manifest` | `runs_cv/<run>/` |
| 6 | Embed once | `submit_runs.sh` with `PRECOMPUTE_ONLY=1` | `runs_cv/<run>/embeddings/` |
| 7 | Run | `submit_runs.sh`, `submit_array.sh`, or `qbiocode.apps.qprofiler.cli` | `runs_cv/<run>/<id>/results/` |
| 8 | Collect | `collate_results.py` | `collated/` |
| 9 | Winners | `select_winners` | per-dataset verdicts |
| 10 | Kernel geometry | `qbiocode.utils.kernel_diagnostics` | alignment and g tables |
| 11 | Meta-analysis | `meta_analysis.ipynb` | meta-feature effects |

:::{dropdown} Checklist
:icon: checklist

- [ ] datasets curated (and synthetic families generated); `inventory*.csv` saved
- [ ] split manifests written, and their hashes recorded or committed
- [ ] gate thresholds, arms, budget, metric, margin, FDR and control roles in `prereg.yaml`
- [ ] hold-out drawn once, its sha256 in `prereg.yaml`
- [ ] configs generated, `MANIFEST.tsv` priced and reviewed
- [ ] embedding cache complete
- [ ] code frozen, hosts pinned, jobs submitted
- [ ] every job done; failures rerun after moving their results aside
- [ ] `collate_results.py` accepted the run
- [ ] winners computed; controls judged against their predictions
- [ ] kernel geometry and meta-analysis run; hold-out scored once
:::

```{toctree}
:hidden:
:maxdepth: 1

Prepare the data <benchmarking/prepare>
Configure and run <benchmarking/run>
Collect and analyze <benchmarking/analyze>
Dataset complexity metrics <dataset_metrics>
Simulated quantum datasets <quantum_datasets>
```
