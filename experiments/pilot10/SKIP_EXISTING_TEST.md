# Testing `skip_existing` on the cluster

A 15-minute check that a resubmitted job adopts what an earlier run already computed
instead of recomputing it. Four phases: the mechanism by hand on the login node (fast,
deterministic, nothing queued), then the LSF submission path.

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

## 5 · The LSF path

Only now check the submission wiring. First with no jobs at risk — `DRY=1` prints the
`bsub` lines and submits nothing:

```bash
DRY=1 FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh runs/heart/heart_none_nb.yaml
```

**Expect** the printed command to end in `++skip_existing=true`. Without
`SKIP_EXISTING=1` it must not appear — check both ways; that one flag is the entire
submit-side change.

Then submit for real:

```bash
FORCE=1 SPREAD=0 SKIP_EXISTING=1 ./submit_runs.sh runs/heart/heart_none_nb.yaml
bjobs -J p10_heart_none_nb
```

This one **must** run from the shipped `runs/` tree, not a copy: `submit_runs.sh` selects
jobs by joining the YAML path you give it against `MANIFEST.tsv`'s absolute `yaml` column,
so a YAML copied into `$T` matches no row and the script exits with *"no config matched the
selection"*. The job therefore writes into `runs/heart/results/heart_none_nb/` — that path
is covered by `.gitignore` (`experiments/**/results/`), so nothing becomes git-visible, and
step 6 removes it.

To see the resume work through LSF, run that command **twice**. The second job finds the
first one's run directory as a sibling and adopts whatever had landed:

```bash
R=runs/heart/results/heart_none_nb
ls -1 $R                                   # two timestamped run directories
resume_log $R/$(ls -1 $R | tail -1)        # the newer one
cells      $R/$(ls -1 $R | tail -1)/adopted.csv
```

**Expect** 5 `adopts … nothing left to fit` lines in the second job and 5 rows in its
`adopted.csv`, exactly as in phase 4 — the first job having completed all five splits. The
job's stdout in `runs/heart/lsf_logs/heart_none_nb.*.out` should show the same, plus a
"Max Memory"/wall summary from LSF that is far below the first job's.

> **Why `FORCE=1`.** `submit_runs.sh` otherwise skips a config whose results are already
> complete — the coarser, per-config skip that has always been there. `SKIP_EXISTING=1` is
> the finer, per-cell one under test. The two compose, so without `FORCE=1` the second
> submission would not go out at all.

## 6 · Clean up

```bash
rm -rf $T
```

If you also ran phase 5 against the shipped tree, remove the run directory it created:

```bash
rm -rf runs/heart/results/heart_none_nb
```

## If a phase does not match

| Symptom | Cause |
|---|---|
| No `skip_existing:` line in the log at all | the override did not arrive. Check `++`, not `+`, and that the log is the one in the run directory you just wrote |
| `0 result row(s) found` | `hydra.run.dir` is not a sibling of the earlier run. `skip_existing: true` looks at the **parent** of the current run directory and one level below it |
| `will NOT adopt … its dataset_sha256 is …` | the dataset changed since the earlier run. This is the guard working; it only fires under `split_mode: manifest`, which this test does not use |
| `adopted.csv` absent after phase 3 | nothing was adopted — usually the truncation in phase 2 did not take, so check `cells $R1/ModelResults.csv` says 2 |
| `skip_existing must be true, false, or an ABSOLUTE directory` | a relative path was passed. Raised during validation, before any data is read |
| phase 5: `no config matched the selection` | `MANIFEST.tsv` stores absolute YAML paths, and this checkout is not at the path they were generated against. Regenerate with `generate_pilot_configs.py --layout split`, or run phases 1-4 only — they need no manifest |
| phase 5: `nothing to submit: every selected config is done` | `FORCE=1` was omitted. That is the older per-config skip, not the per-cell one under test |

## What this does and does not establish

It establishes that cells are adopted, that adopted rows are byte-identical, that only the
missing models are fitted, that a complete config costs seconds, and that
`SKIP_EXISTING=1` reaches the job.

It does **not** exercise the sidecar carry-over (`oof/`, `trials/`, `val_predictions/`),
because those are written only under `split_mode: manifest` and this config is internal
mode. That path is covered by
`tests/test_qprofiler_skip_existing.py::TestTheSidecarsComeAcross`; to check it on the
cluster, run the same four phases against a manifest-mode job from a `runs_cv/<run-id>/`
tree and additionally confirm each sidecar CSV names **every** model of the pass, not only
the ones this run fitted.
