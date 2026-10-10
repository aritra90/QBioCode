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

:::{tab-item} As job arrays
One `bsub`/`sbatch` per 400 jobs instead of one per job, for a sweep of thousands. The
configs are frozen into a numbered task list, so each user can take a range of it:

```bash
./submit_array.sh --count                        # the task total and the list's sha256
RUNS=$BENCH/runs_cv/full1 DRY=1 ./submit_array.sh 1 2000      # preview
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh 1 2000            # user A
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh 2001 4000         # user B
RUNS=$BENCH/runs_cv/full1 SCHED=slurm ./submit_array.sh       # sbatch --array
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
│   ├── adopted.csv               only with skip_existing: which cells came from an
│   │                             earlier run, and from which directory
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
| `SKIP_EXISTING=1` | each job resumes instead of restarting (see below) |

A wall is a kill limit, not a reservation. Extend a slow job in place with
`bmod -W 18:00 <jobid>`. To rerun a failed job, move its `results/<config>` aside, then
run `FORCE=1 DATASET='<id>' WALL=… ./submit_runs.sh`.
:::

(bench-resume)=

## Resuming a job that was killed at its wall

A job appends to `ModelResults.csv` as each model returns, so a wall kill keeps every
pass that finished. But a resubmit opens a **new** run directory and starts again at the
first split, and `collate_results.py` takes one run directory per config — so a config
that needs more than one wall never completes, however often it is resubmitted.

`skip_existing` makes resubmission cumulative. The new run reads the earlier run
directories of the same config and, for every (embedding, split, model) cell they already
hold, copies that row in verbatim, carries its `oof/`, `trials/` and `val_predictions/`
rows across, and fits only what is missing. A pass whose every model is present is skipped
whole: no embedding is read and no complexity measure is recomputed.

```bash
SKIP_EXISTING=1 RUNS=$BENCH/runs_cv/full1 FORCE=1 DATASET='<id>' ./submit_runs.sh
```

Equivalently, in a YAML or as a hydra override — `skip_existing: true` for the sibling run
directories, or an absolute path to look elsewhere:

```bash
python -m qbiocode.apps.qprofiler.cli --config-dir=$PWD --config-name=<config> \
    ++skip_existing=true
```

**Writes:** `adopted.csv` in the new run directory, one row per adopted cell with the run
directory it came from.

```{warning}
Nothing ties a run directory to a config but its name, so resuming **across an edit to the
config** mixes two protocols in one table. Under `split_mode: manifest` the rows carry
`dataset_sha256` and `manifest_sha256`, and a row that disagrees with the run's is refused
and named in the log — which covers a dataset or a frozen split changing underneath. In
internal mode there is no such column: delete the old run directories rather than resume
after changing the config.
```

```{tip}
`results.pkl` of a fully adopted pass is carried over from the earlier run. A pass that was
only *partly* adopted contributes the summary of the models that ran there, because one
summary describes one set of models. `ModelResults.csv` and the sidecars are complete
either way.
```

```{tip}
**Freeze the code and pin the hosts.** `bsub` passes your environment to the job, so a
frozen `PYTHONPATH` keeps later edits away from running jobs. UMAP, TabPFN and the
variational arms vary with the CPU type; the `cpu_model` column records what ran.
```

(bench-array)=

## Job arrays, and splitting a sweep between users

`submit_runs.sh` sends one job per config. At a few hundred jobs that is the right shape;
at a few hundred thousand it is not — every `bsub` is a round trip, and a cluster's
per-user pending limit (400 here) is reached long before the sweep is in.
`submit_array.sh` submits the same configs as **arrays**, waits for the pending count to
drain, and submits the next round. It drives LSF (`bsub -J name[1-400]%200`) or Slurm
(`sbatch --array=1-400%200`).

```{tip}
For a run several users share a results tree, `experiments/pilot10/SHARED_RUN.md` is the
step-by-step: making the tree group-writable (which `UMASK=002` alone does not do), the
smoke test to run before handing the range out, tmux so the submit loop outlives your ssh
session, and why `status.py` needs `--no-lsf` when someone else's jobs are in flight.
```

### Freeze the numbering first

The configs are written once into a numbered task list, `tasks.tsv` beside
`MANIFEST.tsv`. After that, task 5000 means the same config for everyone — which is what
makes dividing the work by range safe. Before anyone submits, have every user run:

```bash
cd $REPO/experiments/pilot10
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh --count
# 568260 tasks in …/runs_cv/full1/tasks.tsv
# sha256 3f9c…
```

Everyone must see the **same total and the same sha256**. A different sha256 means that
list was built from another manifest, or with different `DATASET=`/`EMB=`/`MODEL=`
filters, and that user's task 5000 is not yours. A later call with different filters is
refused rather than renumbering the list under someone mid-submission; `REBUILD=1` forces
it, and should be run only when nobody is submitting.

### Then take a range each

```bash
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh 1 200000          # user A
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh 200001 400000      # user B
RUNS=$BENCH/runs_cv/full1 ./submit_array.sh 400001 568260      # user C
```

`SHARE=k/n` computes the same contiguous blocks, so three users can run `SHARE=1/3`,
`SHARE=2/3`, `SHARE=3/3` without anyone doing the arithmetic. `./submit_array.sh --list N`
prints task N, which is how a failing element is identified from its log.

| Variable | Effect |
|---|---|
| `SCHED=slurm` | `sbatch --array` instead of `bsub`; `PARTITION=` names the partition |
| `MAX_INDEX=400` | elements per array — the per-user pending limit |
| `THROTTLE=200` | elements of one array running at once |
| `BATCH_SLICES=1` | arrays queued per round before waiting |
| `PEND_THRESHOLD=200`, `POLL_INTERVAL=60` | when to submit the next round, and how often to look |
| `SKIP_EXISTING=1` | every element resumes rather than restarts ({ref}`above <bench-resume>`) |
| `SKIP_DONE=1` | an element whose config is already complete exits at once (default) |
| `WALL=`, `MEM=`, `SLOTS=`, `QUEUE=` | as in `submit_runs.sh` |
| `DRY=1` | print the submit lines and leave nothing behind |

```{warning}
`submit_array.sh` does **not** write the embedding cache. Several users submitting
overlapping ranges would each compute it, and a job whose files are missing stops before
fitting anything — so write it once first with
`PRECOMPUTE_ONLY=1 RUNS=$BENCH/runs_cv/full1 ./submit_runs.sh`, and pass `CACHE_CHECK=1`
to confirm a range is served before queueing it.
```

```{note}
An array's elements share one job name, so `status.py` cannot attribute a live LSF state
to a config the way it can for `submit_runs.sh` jobs. Read progress from the result files
(`./status.py --runs-dir … --no-lsf`) and the queue from `bjobs -J <name>` or
`squeue -n <name>`.
```
