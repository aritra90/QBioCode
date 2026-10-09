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
: "${T:?set T before continuing}"                    # see the warning below
mkdir -p $T
```

> **Mind the case of `$T`.** Every path in this file hangs off it, and an unset variable
> expands to nothing rather than erroring — so a stray `rm -fr $t/lsf_logs` (lowercase)
> becomes `rm -fr /lsf_logs`, and `rm -fr $t/*` would become `rm -fr /*`. The `${T:?…}`
> line above fails loudly if `T` is unset, and the deletions in step 7 are written
> `"${T:?}"` for the same reason. Use `$T`, and if a command ever prints a path starting
> `//` or `/runs`, stop.

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

`-K` makes `bsub` block until the job finishes, so the checks below it run *after* the
results exist rather than racing them. And `$RUN` is a fresh directory per submission, for
the reason in the warning underneath.

```bash
mkdir -p $T/lsf_logs
RUN=lsf_$(date +%H%M%S)
bsub -K -J skip_a -q normal -n 1 -R "span[hosts=1] rusage[mem=8]" \
     -o $T/lsf_logs/skip_a.%J.out -e $T/lsf_logs/skip_a.%J.err \
     "cd $PWD && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
      $PY -m qbiocode.apps.qprofiler.cli --config-dir=$PWD/runs/heart \
      --config-name=heart_none_nb ++skip_existing=true \
      hydra.run.dir=$T/results/heart_none_nb/$RUN"

resume_log $T/results/heart_none_nb/$RUN
cells      $T/results/heart_none_nb/$RUN/adopted.csv
```

**Expect** 5 `adopts … nothing left to fit` lines and 5 rows in `adopted.csv`, as in phase
4 — run1/2/3 from the earlier phases are its siblings, and run3 is complete. The LSF
summary in the `.out` file should show a CPU time of a second or two.

Without `-K` the job is queued and the two checks run immediately against a directory that
does not exist yet:

```text
grep: …/lsf1/qprofiler.log: No such file or directory
FileNotFoundError: …/lsf1/adopted.csv
```

That is the race, not a failure — rerun the two checks once `bjobs` is empty. If `-K`
blocks longer than you want (a busy queue), Ctrl-C only abandons the wait; the job keeps
running.

> **Never point two live jobs at one `hydra.run.dir`.** They would both append to the same
> `ModelResults.csv` and `adopted.csv` and race `results.pkl`, and `skip_existing` would
> have each adopting rows the other was still writing. `$RUN` above carries a timestamp so
> a resubmission gets its own directory; `submit_runs.sh` does the same with
> `${now:%Y-%m-%d_%H-%M-%S}` in the shipped configs. If you do submit twice by accident,
> `bkill` the second and delete the directory rather than trusting what is in it.

### 5b · Through `submit_runs.sh`

This is the part that checks `SKIP_EXISTING=1` reaches the job. `submit_runs.sh` selects
jobs by joining the YAML path you hand it against `MANIFEST.tsv`'s absolute `yaml` column,
and the shipped manifest holds the paths of the machine the configs were **generated** on.
In any other checkout every selection is empty:

```text
no config matched the selection (DATASET='' EMB='' MODEL='')
```

**Step 1 — let the shell decide which case you are in.** This sets `RUNS` and `CFG` for
everything below, and prints whether you need step 2:

```bash
STORED=$(awk -F'\t' 'NR>1 && $1=="heart_none_nb" {print $13; exit}' runs/MANIFEST.tsv)
if [ "$STORED" = "$PWD/runs/heart/heart_none_nb.yaml" ]; then
    export RUNS=$PWD/runs CFG=$PWD/runs/heart/heart_none_nb.yaml
    echo "MATCHES this checkout -> SKIP step 2, go straight to step 3"
else
    echo "manifest was generated elsewhere:"
    echo "  it says $STORED"
    echo "  you are $PWD/runs/heart/heart_none_nb.yaml"
    echo "-> RUN step 2 before step 3"
fi
```

**Step 2 — a one-job manifest for this checkout.** Only if step 1 told you to. Three
commands, nothing typed by hand: the old prefix is read out of the manifest.

This is scoped to the test. The real fix for a checkout that is not the one the configs
were generated in is to **regenerate them**, which rewrites every absolute path to your
tree — see "Whose paths are in `runs/`?" at the end of this file. Step 2 exists so you can
finish the test without doing that first.

```bash
mkdir -p $T/runs/heart
sed "s|$(dirname "$STORED")|$T/runs/heart|g" \
    runs/heart/heart_none_nb.yaml > $T/runs/heart/heart_none_nb.yaml
awk -F'\t' -v OFS='\t' -v new="$T/runs/heart/heart_none_nb.yaml" \
    'NR==1 {print; next} $1=="heart_none_nb" {$13=new; print}' runs/MANIFEST.tsv > $T/runs/MANIFEST.tsv
export RUNS=$T/runs CFG=$T/runs/heart/heart_none_nb.yaml
echo "RUNS=$RUNS"; echo "CFG=$CFG"
```

That one `sed` moves `hydra.run.dir`, `quantum_param_dir` and `kernel_dump_dir` into `$T`
as well, since all three sit under the same prefix. `folder_path` is left alone — it points
at the shared dataset directory, which is correct. `embedding_cache` is also left pointing
at the generating machine's path, and that is harmless here: with `embeddings: ['none']`
nothing is ever read from it (the log names the directory, then every pass says *"No
feature reduction"* and *"computed in this run"*).

**Step 3 — the actual check.** Identical either way, because `RUNS` and `CFG` carry the
difference. `DRY=1` submits nothing:

```bash
DRY=1 FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh $CFG
DRY=1 FORCE=1 SPREAD=0                 ./submit_runs.sh $CFG
```

**Expect** `++skip_existing=true` at the end of the printed `bsub` command in the first and
absent in the second. That one flag is the entire submit-side change.

**Step 4 — submit, twice.** The second job finds the first's run directory as a sibling:

```bash
FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh $CFG
bjobs -J p10_heart_none_nb                 # wait for it to clear, then repeat the line above
```

`submit_runs.sh` does **not** take the `hydra.run.dir` override that phases 1-5a used — it
uses the one baked into the YAML, under `runs/heart/results/heart_none_nb/`. That is a
different parent from `$T`, so the first of these two jobs starts with **no siblings** and
computes all five splits from scratch: five *"computed in this run"* lines and no `adopts`
line at all. That is correct, not a regression. The resume is what the **second** job does.

**Expect** `submitted 1 jobs (0 quantum, 1 classical)` each time, then all five cells
adopted by the second:

```bash
R=$(dirname $CFG)/results/heart_none_nb
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

Phase 5 left a complete run, so ask without `FORCE`. `RUNS` and `CFG` are still the ones
5b exported, so this is the same command either way:

```bash
./status.py --runs-dir $RUNS $CFG --list all
SKIP_EXISTING=1 SPREAD=0 ./submit_runs.sh $CFG
```

**Expect** `status.py` to show the config as `done` with `rows 5/5`, and the submit to print
**`nothing to submit: every selected config is done, running or pending`** and exit 0
having queued nothing. That is skip B: no job, no Python process, no queue slot.

The decision comes from `status.py --todo`, which prints a path only for a config that is
`todo`, `failed` or `partial`. You can see it directly — empty output means "nothing to
do":

```bash
./status.py --runs-dir $RUNS --no-lsf --todo $CFG | wc -l          # 0
```

### Through the array runner

`array_task.sh` makes the same check per array element, so a resubmitted range costs a
second per finished task instead of a process. `QBC_SKIP_DONE=1` is its default. Drive one
element by hand:

```bash
printf '%s\t%s\t%s\t\n' "$CFG" heart_none_nb "$(dirname $CFG)" > $T/tasks.tsv
export QBC_TASKS=$T/tasks.tsv QBC_RUNS=$RUNS QBC_STATUS=$PWD/status.py
export QBC_PY=$PY QBC_ENVV="OMP_NUM_THREADS=1" QBC_TASK_OFFSET=0

QBC_TASK_INDEX=1 QBC_SKIP_DONE=1 ./array_task.sh; echo "exit=$?"
```

Nothing here reads `MANIFEST.tsv` — the task list is the three columns above — so this part
works whether or not 5b needed its step 2.

**Expect** `already done (status.py --todo lists it as neither todo, failed nor partial);
skipping` and **exit 0**, in about a second. With `QBC_SKIP_DONE=0` the same command runs
the job instead.

### The interaction that matters

A **partially** finished config must *not* be skipped by B — it has to reach the job so A
can adopt what landed and finish the rest. Truncate the complete results and re-ask:

```bash
B=$(dirname $CFG)/results/heart_none_nb
R=$B/$(ls -1 $B | tail -1)
cp $R/ModelResults.csv $T/full_backup.csv
head -3 $R/ModelResults.csv > $R/tmp && mv $R/tmp $R/ModelResults.csv   # 2 of 5 rows

./status.py --runs-dir $RUNS $CFG --list all          # partial, rows 2/5
QBC_TASK_INDEX=1 QBC_SKIP_DONE=1 ./array_task.sh 2>&1 | tail -3
```

**Expect** `partial  heart_none_nb  rows 2/5` from `status.py`, and the runner to **run the
job** rather than skip it — the two skips layering correctly. Restore afterwards if you
want to keep the completed results:

```bash
cp $T/full_backup.csv $R/ModelResults.csv
```

## 7 · Clean up

`"${T:?}"`, not `$T`: if the variable has been lost the command fails instead of deleting
from `/`.

```bash
echo "about to remove: ${T:?set T first}" && rm -rf "${T:?}"
```

If 5b took the step-1 "MATCHES" branch, phases 5-6 wrote into the shipped tree instead;
remove those too (the path is gitignored, so this is tidiness rather than hygiene):

```bash
rm -rf runs/heart/results/heart_none_nb runs/heart/lsf_logs/heart_none_nb.*
```

And drop the variables if you are staying in the same shell:

```bash
unset QBC_TASKS QBC_RUNS QBC_STATUS QBC_PY QBC_ENVV QBC_TASK_OFFSET RUNS CFG STORED RUN
```

## If a phase does not match

| Symptom | Cause |
|---|---|
| `Additional config directory '…/runs/heart' not found` | you are in the wrong directory. Every command here is run from `experiments/pilot10`, so that `$PWD/runs/heart` resolves; from the repository root `$PWD/runs` does not exist |
| No `skip_existing:` line in the log at all | the override did not arrive. Check `++`, not `+`, and that the log is the one in the run directory you just wrote |
| `0 result row(s) found` | `hydra.run.dir` is not a sibling of the earlier run. `skip_existing: true` looks at the **parent** of the current run directory and one level below it |
| `will NOT adopt … its dataset_sha256 is …` | the dataset changed since the earlier run. This is the guard working; it only fires under `split_mode: manifest`, which this test does not use |
| `adopted.csv` absent after phase 3 | nothing was adopted — usually the truncation in phase 2 did not take, so check `cells $R1/ModelResults.csv` says 2 |
| `skip_existing must be true, false, or an ABSOLUTE directory` | a relative path was passed. Raised during validation, before any data is read |
| `no config matched the selection (DATASET='' EMB='' MODEL='')` | **The common one.** `MANIFEST.tsv` stores absolute YAML paths from the machine the configs were generated on, and this checkout is elsewhere. 5b step 1 detects it; step 2 works around it for the test, and *Whose paths are in `runs/`?* below is the real fix. Phases 1-4, 5a and the array part of 6 need no manifest at all |
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

## Whose paths are in `runs/`?

Not necessarily yours. `generate_pilot_configs.py` bakes **absolute** paths into every
config, because hydra changes the working directory and a relative one would mean something
different in each job. The `runs/` tree committed here was generated on one machine, so in
anybody else's checkout four things point at that machine:

| In each YAML | Points at | Matters? |
|---|---|---|
| `hydra.run.dir` | the generating machine's `runs/<ds>/results/` | **yes** — the job writes here |
| `quantum_param_dir` | …`/quantum_tuned_params/` | yes, for quantum arms |
| `kernel_dump_dir` | …`/kernels/` | yes, for qsvc/pqk |
| `embedding_cache` | …`/embeddings/` | yes, for any embedded (non-`none`) arm |
| `folder_path` | `/dccstor/cgq4hls/Q/qbc_data/...` | no — shared, readable |

`MANIFEST.tsv`'s `yaml` column is absolute for the same reason, which is what makes
`submit_runs.sh` print *"no config matched the selection"* elsewhere.

**Regenerate, rather than editing by hand.** Run this from your own checkout:

```bash
cd $REPO/experiments/pilot10
./generate_pilot_configs.py --layout split --self-contained --n-trials-quantum 12
```

Every absolute path is then derived from the script's own location, so `runs/`,
`embeddings/` and `MANIFEST.tsv` all land in your tree, and `folder_path` stays on the
shared data root. After that `submit_runs.sh` works normally and step 2 of 5b is
unnecessary.

The three flags are not decoration — they are what makes the result the *same experiment*:

- `--self-contained`, because this `runs/` holds whole configs rather than job files over a
  `_protocol.yaml`;
- `--n-trials-quantum 12`, because the generator's default is 32 and these were built at 12;
- `--layout split`, one job per (dataset, embedding, model).

`--iter` and `--test-size` already default to the shipped 5 and 0.20. Checked: with those
three flags the regenerated `heart_none_nb.yaml` is byte-identical to the committed one
apart from `hydra.run.dir`, `quantum_param_dir` and `kernel_dump_dir` — so no search space,
seed, model list or split setting moves.

> **Note.** `runs/*/*.yaml` and `runs/MANIFEST.tsv` are tracked in git, so regenerating
> shows up as modified tracked files. That is a local adaptation, not a change anyone else
> wants: keep it out of shared branches, or commit it on your own branch knowing it
> re-points the tree at your paths.

### After regenerating: rebuild the embedding cache

`embedding_cache` now points at **your** `experiments/pilot10/embeddings`, which is
gitignored and therefore empty. Of the 208 split-layout jobs, 104 are `embeddings: ['none']`
and read nothing from it — `heart_none_nb`, so this whole test, is one of those. The other
104 (the `pca`/`umap` arms of `spect`, `wdbc`, `sonar`, `colon_cancer`) read it for every
split, and a job whose file is missing **stops before fitting anything** rather than
embedding for itself. So before running those:

```bash
PRECOMPUTE_ONLY=1 ./submit_runs.sh          # writes the cache, submits nothing
```

Check it without writing anything:

```bash
$PY -m qbiocode.apps.qprofiler.embedding_cache --check runs/*/*_pca_*.yaml runs/*/*_umap_*.yaml
```

This is not a `skip_existing` interaction — `skip_existing` still requires the cache to be
complete for the whole run, including the passes it is going to adopt. It is a pre-flight
contract and resuming does not relax it.
