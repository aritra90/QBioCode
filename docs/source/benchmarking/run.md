# Configure and run

Three steps turn the frozen inputs into results: write one QProfiler YAML per job,
compute every embedding once, and run the jobs.

(bench-configs)=

## 5 · Write the job YAMLs

`generate_pilot_configs.py --split-mode manifest` writes one YAML per job. A job is one
(dataset, embedding, model group, chunk of splits). Every path inside a YAML is
absolute, so a job can run from anywhere.

```bash
cd $REPO/experiments/pilot10
python generate_pilot_configs.py --split-mode manifest \
    --datasets-root $BENCH/data/datasets --split-dir $BENCH/data/splits/v2 \
    --runs-dir $BENCH/runs_cv --run-id full1 \
    --datasets all --splits all --n-trials 30 \
    --splits-per-job 'classical=all,qsvc=all,pqk=5' --embed-above 13 --wall auto
```

| Flag | Meaning |
|---|---|
| `--datasets all` | every curated dataset with a manifest (or `--datasets-file` with a list) |
| `--models` | default: qsvc, pqk and the nine classical models; name `qnn` or `vqc` to add them |
| `--n-trials` | the same tuning budget for every arm; trial 0 is each arm's default |
| `--splits-per-job` | how many splits one job runs in turn, to save per-job overhead |
| `--embed-above 13` | wider datasets are embedded to 8 components, so quantum arms stay on the exact simulator |
| `--wall auto` | each job's LSF wall from the cost model |

**Writes:** `runs_cv/<run>/<id>/<config>.yaml`, and `runs_cv/<run>/MANIFEST.tsv` with one
row per job. The script also prints the run's expected cost in core-hours.

:::{dropdown} The paths a YAML carries
:icon: file-code

```yaml
config_file_name: 'openml__pc4_pca_i01-05_pqk'
folder_path: '$BENCH/data/datasets/openml__pc4'
file_dataset: ['openml__pc4.csv']
embeddings: ['pca']
n_components: 8
embedding_cache: '$BENCH/runs_cv/full1/embeddings'
quantum_model: ['pqk']
split_mode: manifest
split_dir: '$BENCH/data/splits/v2'
splits: [1, 2, 3, 4, 5]
kernel_dump_dir: '$BENCH/runs_cv/full1/openml__pc4/kernels/openml__pc4_pca_i01-05_pqk'
```
:::

```{tip}
In manifest mode, a model's `<model>_args` block **is** read: its searched names are
trial 0, and its other keys stay fixed. Outside manifest mode, a tuned model ignores it;
see the warning in the {doc}`configuration guide <../apps/config>`.
```

(bench-cache)=

## 6 · Compute the embedding cache, once

Every embedded split is computed before any job runs, in both stages:
- the final stage, fit on the training rows and applied to the test rows;
- the tuning stage, fit on the fit rows and applied to the validation rows.

Jobs read these files and never embed for themselves.

```bash
PRECOMPUTE_ONLY=1 RUNS=$BENCH/runs_cv/full1 ./submit_runs.sh       # writes, submits nothing
python -m qbiocode.apps.qprofiler.embedding_cache $BENCH/runs_cv/full1/openml__pc4/*.yaml
```

**Writes:** `runs_cv/<run>/embeddings/emb_<id>_<emb>_8_<split>.npz` and `…__tune.npz`. Each
holds the embedded rows, their indices, a `spec` of everything they depend on, and a
`provenance` record.

```{tip}
This is what keeps every job of a pass on identical features. Seeded UMAP still
differs between CPU types, so features computed inside the jobs would depend on where
each job landed. On a large run, compute the cache as parallel cluster jobs, one per
dataset.
```

(bench-run)=

## 7 · Run QProfiler on the YAMLs

::::{tab-set}

:::{tab-item} On LSF
```bash
cp -r $REPO/qbiocode $BENCH/code/ && export PYTHONPATH=$BENCH/code    # freeze the code
HOSTS="$(LIST_HOSTS=Intel_Platinum:128 ./submit_runs.sh | tail -1)"  # one host group
DRY=1 RUNS=$BENCH/runs_cv/full1 HOSTS="$HOSTS" ./submit_runs.sh       # preview
RUNS=$BENCH/runs_cv/full1 HOSTS="$HOSTS" ./submit_runs.sh             # submit
```
:::

:::{tab-item} One job by hand
Every YAML is a complete QProfiler run, so one job needs no scheduler:

```bash
cd $BENCH/runs_cv/full1/pmlb__labor
python -m qbiocode.apps.qprofiler.cli --config-dir=$PWD --config-name=pmlb__labor_pca_i01-15_qsvc
```
:::

:::{tab-item} Watching
```bash
./status.py --runs-dir $BENCH/runs_cv/full1     # done / running / pending / failed per dataset
```
:::
::::

Each job writes into its dataset's folder:

```text
runs_cv/full1/pmlb__labor/
├── results/<config>/<backend>_<timestamp>/
│   ├── ModelResults.csv          scores, tuned parameters, 141 complexity measures
│   ├── RawDataEvaluation.csv     complexity of the raw, unembedded data
│   ├── oof/  trials/  val_predictions/
│   └── results.pkl, qprofiler.log
├── kernels/<config>/             gram_*.npz (qsvc), proj_*.npz (pqk)
├── quantum_tuned_params/<config>/
└── lsf_logs/
```

:::{dropdown} The columns of `ModelResults.csv`
:icon: table

Besides the 141 {doc}`complexity measures <../dataset_metrics>`:

| Group | Columns |
|---|---|
| Identity | `Dataset`, `embeddings`, `iteration`, `repeat`, `fold`, `model` |
| Scores | `accuracy`, `balanced_accuracy`, `f1_score`, `mcc`, `auc`, `pr_auc`, `time` |
| Tuning | `BestParams_Tuned`, `tuning_metric`, `tuning_score`, `val_auc`, `val_log_loss` |
| Provenance | `manifest_sha256`, `dataset_sha256`, `n_fit`, `n_val`, `n_test`, `seed`, `host`, `cpu_model`, `lsf_jobid` |

The sidecar folders hold one CSV per split:
- `oof/`: the test-row predictions;
- `trials/`: every tuning trial and its validation score;
- `val_predictions/`: each trial's validation predictions.
:::

:::{dropdown} `submit_runs.sh` settings, and when a job fails
:icon: gear

| Variable | Effect |
|---|---|
| `DRY=1` | print what would be written and submitted |
| `PRECOMPUTE_ONLY=1` | write the embedding cache, submit nothing |
| `HOSTS=`, `LIST_HOSTS=` | pin to one host group; print that group |
| `DATASET=`, `EMB=`, `MODEL=` | submit a subset (anchored regexes on `MANIFEST.tsv`) |
| `FORCE=1` | resubmit jobs that are done or queued |
| `WALL=`, `MEM=` | override every selected job's wall and memory (GB) |

A wall is a kill limit, not a reservation. Extend a slow job in place with
`bmod -W 18:00 <jobid>`. To rerun a failed job, move its `results/<config>` aside, then
run `FORCE=1 DATASET='<id>' WALL=… ./submit_runs.sh`.
:::

```{tip}
**Freeze the code and pin the hosts.** `bsub` passes your environment to the job, so a
frozen `PYTHONPATH` keeps later edits away from running jobs. UMAP, TabPFN and the
variational arms vary with the CPU type; the `cpu_model` column records what ran.
```
