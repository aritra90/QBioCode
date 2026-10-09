# Testing the two skips on the cluster

A 20-minute check that a resubmitted job does not redo work. There are **two** independent
skips at two granularities, and they compose — this tests both:

| | Skip | Granularity | Where it lives | Phases |
|---|---|---|---|---|
| **A** | `skip_existing` | one (embedding, split, model) **cell** | inside QProfiler | 1-4 |
| **B** | the done-check | a whole **config** / job | `submit_runs.sh`, `array_task.sh` | 6 |

**B** is the cheap one: it never starts a Python process for a config whose results are
already complete. **A** is the fine one: it starts the job but adopts the cells an earlier
run left behind. A config that is *partially* done is deliberately **not** skipped by B —
it is handed to A to finish, which is the whole point of the pair.

Phases 1-4 run the mechanism by hand on the login node (fast, deterministic, nothing
queued), phase 5 checks the LSF submission path, phase 6 checks the done-check.

Everything is written under one scratch directory, so **nothing touches `runs/` or the
shipped `results/` trees**. Delete the directory and the test is gone.

## 0 · Set up

```bash
cd $REPO/experiments/pilot10
export PY=/dccstor/boseukb/Q/envs/qbc/bin/python     # same interpreter submit_runs.sh uses
export T=/dccstor/$USER/skip_test                    # scratch; anywhere writable
mkdir -p $T
```

The test uses `runs/heart/heart_none_nb.yaml`, deliberately the cheapest shipped config:
naive Bayes on 270 rows, `embeddings: ['none']`, `iter: 5`. Being unembedded it reads
**nothing** from the embedding cache, so there is no precompute step, and being classical
it touches neither `quantum_param_dir` nor `kernel_dump_dir`.

Check the dataset it needs is readable, and keep thread counts at 1 on a login node:

```bash
ls -l /dccstor/cgq4hls/Q/qbc_data/libsvm_data/heart.csv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Two helpers used throughout — row counts, and which (split, model) cells are present:

```bash
cells () { $PY - "$1" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{len(rows)} rows:", sorted({(r["iteration"], r["model"]) for r in rows}))
EOF
}
resume_log () { grep -E "skip_existing|adopts" "$1"/qprofiler.log || echo "(no resume lines)"; }
```

`hydra.run.dir` is overridden on every command below so the three runs land as **siblings**
under `$T/results/heart_none_nb/`. That is the real layout: `skip_existing: true` resolves
to the parent of the current run directory, i.e. the other runs of the same config.

## 1 · A complete baseline run

```bash
time $PY -m qbiocode.apps.qprofiler.cli \
    --config-dir=$PWD/runs/heart --config-name=heart_none_nb \
    hydra.run.dir=$T/results/heart_none_nb/run1

cells $T/results/heart_none_nb/run1/ModelResults.csv
```

**Expect** `5 rows:` with `('1','nb_opt') … ('5','nb_opt')` — `iter: 5` × one embedding ×
one model. Note the wall time; phase 3 is compared against it. No `adopted.csv` is written
(nothing was adopted). The model label is `nb_opt`, not `nb`, because `grid_search: True`
tunes it.

## 2 · Make it look like a job killed at its wall

A job killed part-way leaves a short `ModelResults.csv`. Truncate the baseline to its
first two splits to reproduce that exactly:

```bash
R1=$T/results/heart_none_nb/run1
wc -l < $R1/ModelResults.csv                 # 6 = header + 5
head -3 $R1/ModelResults.csv > $R1/tmp && mv $R1/tmp $R1/ModelResults.csv
cells $R1/ModelResults.csv                   # -> 2 rows: splits 1 and 2
```

`results.pkl` is deliberately left alone; a real wall kill leaves whatever it had, and
phase 3 shows the pass summaries being carried across regardless.

## 3 · Resume — the measurement

```bash
time $PY -m qbiocode.apps.qprofiler.cli \
    --config-dir=$PWD/runs/heart --config-name=heart_none_nb \
    ++skip_existing=true \
    hydra.run.dir=$T/results/heart_none_nb/run2

cells      $T/results/heart_none_nb/run2/ModelResults.csv
cells      $T/results/heart_none_nb/run2/adopted.csv
resume_log $T/results/heart_none_nb/run2
```

`++` rather than `+`, so the key is set whether or not the config already defines one.

**Expect, and this is the whole test:**

| Check | Expected |
|---|---|
| `run2/ModelResults.csv` | **5 rows**, splits 1-5 — complete, though only 3 were fitted |
| `run2/adopted.csv` | **2 rows**, splits 1 and 2, model `nb_opt`, `source_run_dir` = `…/run1` |
| resume log | one `… 2 result row(s) found in 1 earlier run(s) …` line, then **two** `adopts` lines |
| wall time | roughly 3/5 of phase 1 |

The log lines read like this (the `data_key` is `<dataset>_<embedding>_<n_components>_<split>`,
so `heart_none_8_1`):

```text
skip_existing: 2 result row(s) found in 1 earlier run(s) under …/results/heart_none_nb;
the (embedding, split, model) cells they cover will not be recomputed. …
heart_none_8_1: skip_existing adopts ['nb_opt'] from ['…/run1']; nothing left to fit
heart_none_8_2: skip_existing adopts ['nb_opt'] from ['…/run1']; nothing left to fit
```

Splits 3-5 print **no** `adopts` line — nothing was adopted for them, so they ran normally.
That asymmetry is the point: two lines, not five.

Confirm the adopted rows were copied and not recomputed — they must be byte-identical to
the originals, because an adopted row is copied as text:

```bash
$PY - $T/results/heart_none_nb <<'EOF'
import csv, sys, pathlib
root = pathlib.Path(sys.argv[1])
old = {r["iteration"]: r for r in csv.DictReader(open(root / "run1" / "ModelResults.csv"))}
new = {r["iteration"]: r for r in csv.DictReader(open(root / "run2" / "ModelResults.csv"))}
for it in sorted(old):
    same = all(old[it][k] == new[it].get(k) for k in old[it])
    print(f"split {it}: identical to run1 = {same}")
EOF
```

**Expect** `identical to run1 = True` for splits 1 and 2.

## 4 · Resume again — nothing left to do

`run2` is now complete, so a third run should fit **nothing at all**:

```bash
time $PY -m qbiocode.apps.qprofiler.cli \
    --config-dir=$PWD/runs/heart --config-name=heart_none_nb \
    ++skip_existing=true \
    hydra.run.dir=$T/results/heart_none_nb/run3

cells      $T/results/heart_none_nb/run3/ModelResults.csv
cells      $T/results/heart_none_nb/run3/adopted.csv
resume_log $T/results/heart_none_nb/run3
```

**Expect** 5 rows in both files, **five** `adopts … nothing left to fit` lines, and a wall
time of **seconds** — a fully adopted pass reads no features and recomputes none of the 141
complexity measures, which is where phase 1's time went. The `found` line should say
`5 result row(s) … in 2 earlier run(s)`: run1 and run2 both contribute, and where they
overlap the newer directory wins (run directories sort by their timestamp, so later is
newer).

`run3/results.pkl` should hold 5 pass summaries, carried over from run2 — a fully adopted
pass takes the earlier run's summary so this run's `results.pkl` is as complete as its
table:

```bash
$PY -c "import pickle; print(len(pickle.load(open('$T/results/heart_none_nb/run3/results.pkl','rb'))), 'passes')"
```

## 5 · Under LSF

### 5a · Straight `bsub` — no manifest needed

The quickest confirmation that the resume works inside a batch job rather than on the login
node. Nothing here reads `MANIFEST.tsv`, so it works in any checkout:

```bash
mkdir -p $T/lsf_logs
bsub -J skip_a -q normal -n 1 -R "span[hosts=1] rusage[mem=8]" \
     -o $T/lsf_logs/skip_a.%J.out -e $T/lsf_logs/skip_a.%J.err \
     "cd $PWD && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
      $PY -m qbiocode.apps.qprofiler.cli --config-dir=$PWD/runs/heart \
      --config-name=heart_none_nb ++skip_existing=true \
      hydra.run.dir=$T/results/heart_none_nb/lsf1"

bjobs -J skip_a            # wait for DONE
resume_log $T/results/heart_none_nb/lsf1
cells      $T/results/heart_none_nb/lsf1/adopted.csv
```

**Expect** 5 `adopts … nothing left to fit` lines and 5 rows in `adopted.csv`, as in phase
4 — run1/2/3 from the earlier phases are its siblings, and run3 is complete. The LSF
summary in the `.out` file should show a CPU time of a second or two.

### 5b · Through `submit_runs.sh`

This is the part that checks `SKIP_EXISTING=1` reaches the job. `submit_runs.sh` selects
jobs by joining the YAML path you hand it against `MANIFEST.tsv`'s absolute `yaml` column,
and the shipped manifest holds the paths of the machine the configs were **generated** on.
If your checkout is somewhere else you get:

```text
no config matched the selection (DATASET='' EMB='' MODEL='')
```

Check before anything else, and skip to 5c if they differ:

```bash
awk -F'\t' 'NR>1 && $1=="heart_none_nb" {print "manifest:", $13}' runs/MANIFEST.tsv
echo "yours:    $PWD/runs/heart/heart_none_nb.yaml"
```

If they match, the shipped tree works directly — `DRY=1` first, which submits nothing:

```bash
DRY=1 FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh runs/heart/heart_none_nb.yaml
DRY=1 FORCE=1 SPREAD=0                 ./submit_runs.sh runs/heart/heart_none_nb.yaml
```

**Expect** `++skip_existing=true` at the end of the printed `bsub` command in the first,
and absent in the second. That one flag is the entire submit-side change. Then drop `DRY=1`
to submit; the job writes into `runs/heart/results/heart_none_nb/`, which `.gitignore`
covers (`experiments/**/results/`).

### 5c · A one-job manifest for this checkout

If the paths differ, build a scratch tree whose manifest names *your* YAML. Three commands,
no editing by hand — the old prefix is read out of the manifest rather than typed:

```bash
mkdir -p $T/runs/heart
OLDDIR=$(dirname "$(awk -F'\t' 'NR>1 && $1=="heart_none_nb" {print $13; exit}' runs/MANIFEST.tsv)")
sed "s|$OLDDIR|$T/runs/heart|g" runs/heart/heart_none_nb.yaml > $T/runs/heart/heart_none_nb.yaml
awk -F'\t' -v OFS='\t' -v new="$T/runs/heart/heart_none_nb.yaml" \
    'NR==1 {print; next} $1=="heart_none_nb" {$13=new; print}' runs/MANIFEST.tsv > $T/runs/MANIFEST.tsv
```

That one `sed` moves `hydra.run.dir`, `quantum_param_dir` and `kernel_dump_dir` into `$T`
as well, since all three sit under the same prefix. `folder_path` is left alone — it points
at the shared dataset directory, which is correct. `embedding_cache` is also left pointing
at the generating machine's path and that is harmless here: with `embeddings: ['none']`
nothing is ever read from it (the log names the directory but every pass says *"No feature
reduction"* and *"computed in this run"*).

Check what it selected, then submit:

```bash
RUNS=$T/runs DRY=1 FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh $T/runs/heart/heart_none_nb.yaml
RUNS=$T/runs      FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh $T/runs/heart/heart_none_nb.yaml
bjobs -J p10_heart_none_nb
```

**Expect** `submitted 1 jobs (0 quantum, 1 classical)`. Run it **twice**: the second job
finds the first's run directory as a sibling and adopts all five cells.

```bash
R=$T/runs/heart/results/heart_none_nb
ls -1 $R                                   # two timestamped run directories
resume_log $R/$(ls -1 $R | tail -1)
cells      $R/$(ls -1 $R | tail -1)/adopted.csv
```

> **Why `FORCE=1`.** `submit_runs.sh` otherwise skips a config whose results are already
> complete — the coarser, per-config skip that has always been there. `SKIP_EXISTING=1` is
> the finer, per-cell one under test. The two compose, so without `FORCE=1` the second
> submission would not go out at all. Phase 6 tests that skip on its own.

## 6 · Skip B: a completed job is never started

Everything above ran with `FORCE=1`, which **turns this off**. Now test it on its own.

### Through `submit_runs.sh`

Phase 5 left a complete run, so ask without `FORCE`. Use whichever tree phase 5 worked in
— `RUNS=runs` for 5b, or `RUNS=$T/runs` for 5c (shown here):

```bash
./status.py --runs-dir $T/runs $T/runs/heart/heart_none_nb.yaml --list all
RUNS=$T/runs SKIP_EXISTING=1 SPREAD=0 ./submit_runs.sh $T/runs/heart/heart_none_nb.yaml
```

**Expect** `status.py` to show the config as `done` with `rows 5/5`, and the submit to print
**`nothing to submit: every selected config is done, running or pending`** and exit 0
having queued nothing. That is skip B: no job, no Python process, no queue slot.

The decision comes from `status.py --todo`, which prints a path only for a config that is
`todo`, `failed` or `partial`. You can see it directly — empty output means "nothing to
do":

```bash
./status.py --runs-dir $T/runs --no-lsf --todo $T/runs/heart/heart_none_nb.yaml | wc -l   # 0
```

### Through the array runner

`array_task.sh` makes the same check per array element, so a resubmitted range costs a
second per finished task instead of a process. `QBC_SKIP_DONE=1` is its default. Drive one
element by hand:

```bash
printf '%s\t%s\t%s\t\n' "$T/runs/heart/heart_none_nb.yaml" heart_none_nb "$T/runs/heart" > $T/tasks.tsv
export QBC_TASKS=$T/tasks.tsv QBC_RUNS=$T/runs QBC_STATUS=$PWD/status.py
export QBC_PY=$PY QBC_ENVV="OMP_NUM_THREADS=1" QBC_TASK_OFFSET=0

QBC_TASK_INDEX=1 QBC_SKIP_DONE=1 ./array_task.sh; echo "exit=$?"
```

Nothing here reads `MANIFEST.tsv` — the task list is the three columns above — so this part
works whether or not 5b applied to your checkout.

**Expect** `already done (status.py --todo lists it as neither todo, failed nor partial);
skipping` and **exit 0**, in about a second. With `QBC_SKIP_DONE=0` the same command runs
the job instead.

### The interaction that matters

A **partially** finished config must *not* be skipped by B — it has to reach the job so A
can adopt what landed and finish the rest. Truncate the complete results and re-ask:

```bash
B=$T/runs/heart/results/heart_none_nb
R=$B/$(ls -1 $B | tail -1)
cp $R/ModelResults.csv $T/full_backup.csv
head -3 $R/ModelResults.csv > $R/tmp && mv $R/tmp $R/ModelResults.csv   # 2 of 5 rows

./status.py --runs-dir $T/runs $T/runs/heart/heart_none_nb.yaml --list all   # partial, rows 2/5
QBC_TASK_INDEX=1 QBC_SKIP_DONE=1 ./array_task.sh 2>&1 | tail -3
```

**Expect** `partial  heart_none_nb  rows 2/5` from `status.py`, and the runner to **run the
job** rather than skip it — the two skips layering correctly. Restore afterwards if you
want to keep the completed results:

```bash
cp $T/full_backup.csv $R/ModelResults.csv
```

## 7 · Clean up

```bash
rm -rf $T
```

If you also ran phases 5-6 against the shipped tree, remove the run directories they
created (that path is gitignored, so this is tidiness rather than hygiene):

```bash
rm -rf runs/heart/results/heart_none_nb runs/heart/lsf_logs/heart_none_nb.*
```

And unset the array-runner variables if you are staying in the same shell:

```bash
unset QBC_TASKS QBC_RUNS QBC_STATUS QBC_PY QBC_ENVV QBC_TASK_OFFSET
```

## If a phase does not match

| Symptom | Cause |
|---|---|
| No `skip_existing:` line in the log at all | the override did not arrive. Check `++`, not `+`, and that the log is the one in the run directory you just wrote |
| `0 result row(s) found` | `hydra.run.dir` is not a sibling of the earlier run. `skip_existing: true` looks at the **parent** of the current run directory and one level below it |
| `will NOT adopt … its dataset_sha256 is …` | the dataset changed since the earlier run. This is the guard working; it only fires under `split_mode: manifest`, which this test does not use |
| `adopted.csv` absent after phase 3 | nothing was adopted — usually the truncation in phase 2 did not take, so check `cells $R1/ModelResults.csv` says 2 |
| `skip_existing must be true, false, or an ABSOLUTE directory` | a relative path was passed. Raised during validation, before any data is read |
| `no config matched the selection (DATASET='' EMB='' MODEL='')` | **The common one.** `MANIFEST.tsv` stores absolute YAML paths from the machine the configs were generated on, and this checkout is elsewhere. Use **5c**, which builds a one-job manifest naming your path; or regenerate with `generate_pilot_configs.py --layout split`. Phases 1-4, 5a and the array part of 6 need no manifest at all |
| phase 5: `nothing to submit: every selected config is done` | `FORCE=1` was omitted. That is skip **B**, which phase 5 is not testing — it is the *expected* result in phase 6 |
| phase 6: the config reads `todo`, not `done` | `--runs-dir` is wrong, or the results are not under `<config dir>/results/<config_file_name>/`. `status.py` globs exactly that, and counts distinct (embedding, iteration, model) rows against `iter x embeddings x models` |
| phase 6: skipped when you expected it to run | the truncation did not take. `status.py … --list all` must say `partial  rows 2/5` before the runner will run it |

## What this does and does not establish

**Skip A** (`skip_existing`): that cells are adopted, that adopted rows are byte-identical
to the originals, that only the missing models are fitted, that a fully-adopted config
costs seconds, and that `SKIP_EXISTING=1` reaches the job.

**Skip B** (the done-check): that a complete config is never submitted and a complete array
element exits without starting Python, and — the interaction that matters — that a
*partial* config is **not** skipped by B but handed to A to finish.

It does **not** exercise the sidecar carry-over (`oof/`, `trials/`, `val_predictions/`),
because those are written only under `split_mode: manifest` and this config is internal
mode. That path is covered by
`tests/test_qprofiler_skip_existing.py::TestTheSidecarsComeAcross`; to check it on the
cluster, run phases 1-4 against a manifest-mode job from a `runs_cv/<run-id>/` tree and
additionally confirm each sidecar CSV names **every** model of the pass, not only the ones
this run fitted.

Nor does it cover `status.py`'s *live* states (`running`, `pending`), which need real
queued jobs: skip B also drops a config that is currently in flight, so a second
`./submit_runs.sh` while the first is running queues nothing. Phase 6 exercises only the
`done` and `partial` paths, which are the ones decided from the result files alone.
