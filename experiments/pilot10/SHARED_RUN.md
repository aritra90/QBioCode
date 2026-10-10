# Running one sweep across several users

One frozen task list, divided by range, written into one results tree that everybody can
write to. Nobody runs the same config twice, and anyone can resume anyone else's work.

This covers only what is different about sharing a run. The pipeline itself is
`docs/source/benchmarking/prepare.md` (steps 1-4) and `run.md` (steps 5-8); the settings
tables for the submitters are in the header of each script.

**Read the roles before you start.** The *owner* does sections 1-4 once. Everyone else
starts at section 5 and never touches the task list.

---

## Why permissions come first

A job writes with the umask of the shell that submitted it. The usual default is `022`,
which produces files like

```
-rw-r--r--  1 futro  cgq4hls   ModelResults.csv
```

group-**readable**, not group-writable. Everything looks fine until the moment someone
resumes, re-runs or collates a config a different user started — and then it fails inside
a job, hours later, as a permission error in a log nobody is watching. Section 1 is not
optional setup; it is the part that decides whether section 7 works.

Three separate things have to be right, and fixing one does not fix the others:

| | what it controls | set by |
|---|---|---|
| **group** | who *could* write | `chgrp`, and setgid for new files |
| **write bit** | whether they actually can | `umask` / default ACL |
| **existing files** | already written, unaffected by either | `chmod` after the fact |

---

## 1 · Owner: make the tree group-writable

```bash
export BENCH=/dccstor/cgq4hls/Q/bench1          # the shared results tree
export REPO=/dccstor/cgq4hls/Q/futro/QBioCode   # your checkout

id -Gn                      # the groups you are in
ls -ld $BENCH               # the group it has now
```

Pick the group all of you share — below it is `cgq4hls`; substitute yours.

```bash
GRP=cgq4hls

chgrp -R  "$GRP" $BENCH
chmod -R  g+rwX  $BENCH                                  # capital X: dirs, not data files
find $BENCH -type d -print0 | xargs -0 chmod g+s         # new entries inherit the group
```

`g+s` (setgid) makes a *new* file inherit the directory's group. It does **not** make it
group-writable — that is the umask. If your filesystem supports POSIX ACLs, a default ACL
covers both and survives somebody forgetting their umask:

```bash
setfacl -R -m   g:"$GRP":rwX $BENCH     # existing entries
setfacl -R -d -m g:"$GRP":rwX $BENCH    # default: inherited by everything created later
```

On GPFS (which `/dccstor` is) `setfacl` may be absent; `mmputacl` is the equivalent, or
rely on setgid plus the `UMASK=002` in sections 4-6, which every command below passes.

Verify as **another user**, not as yourself — this is the only check that means anything:

```bash
# coworker runs:
touch $BENCH/runs_cv/.writetest && rm $BENCH/runs_cv/.writetest && echo "writable"
```

> Two users must never point `--runs-dir` at the same `--run-id` from *different*
> `generate_pilot_configs.py` invocations. One tree, generated once, is the whole premise:
> the task numbers come from it.

## 2 · Owner: write the job YAMLs, once

```bash
cd $REPO/experiments/pilot10
python generate_pilot_configs.py --split-mode manifest \
    --datasets-root $BENCH/data/datasets --split-dir $BENCH/data/splits/v2 \
    --runs-dir $BENCH/runs_cv --run-id full1 \
    --datasets all --splits all --n-trials 30 \
    --splits-per-job 'classical=all,qsvc=all,pqk=5' --embed-above 13 --wall auto
```

Writes `$BENCH/runs_cv/full1/` with one YAML per job and `MANIFEST.tsv`. It prints the
run's expected core-hours — read that number before committing a cluster to it.

```bash
export RUNS=$BENCH/runs_cv/full1
```

## 3 · Owner: smoke-test on 3 datasets before anyone else sees it

Do this in a **throwaway run tree**, so the real `tasks.tsv` is never built from a
half-tested configuration. Three of the smallest datasets, two splits, two trials:

```bash
cd $REPO/experiments/pilot10
python generate_pilot_configs.py --split-mode manifest \
    --datasets-root $BENCH/data/datasets --split-dir $BENCH/data/splits/v2 \
    --runs-dir $BENCH/runs_cv --run-id smoke1 \
    --datasets pmlb__analcatdata_aids,pmlb__labor,pmlb__analcatdata_bankruptcy \
    --splits 1-2 --n-trials 2 \
    --splits-per-job 'classical=all,qsvc=all,pqk=all' --embed-above 13 --wall auto

export SMOKE=$BENCH/runs_cv/smoke1
UMASK=002 PRECOMPUTE_ONLY=1 RUNS=$SMOKE ./submit_runs.sh     # cache; submits nothing
RUNS=$SMOKE ./submit_array.sh --count                        # how many tasks
RUNS=$SMOKE DRY=1 UMASK=002 ./submit_array.sh 1 5            # print, submit nothing
RUNS=$SMOKE UMASK=002 ./submit_array.sh 1 5                  # really submit 5
```

Then check, in this order:

```bash
./status.py --runs-dir $SMOKE --no-lsf                       # 5 should reach done
ls -l $SMOKE/*/results/*/*/ModelResults.csv | head           # want -rw-rw-r--
RUNS=$SMOKE UMASK=002 SKIP_DONE=1 ./submit_array.sh 1 5      # re-run: all 5 skip in <1s
```

That last line is the one worth watching. Each element should print

```
already done (status.py --todo lists it as neither todo, failed nor partial); skipping
```

and exit. If instead they start fitting, the done-check is not working and a shared
re-submission will duplicate work rather than absorb it — stop and fix that before
section 5.

Have **one coworker** repeat the re-run line above. If it skips for them too, group
permissions are right. Then throw the tree away:

```bash
rm -rf $SMOKE
```

## 4 · Owner: write the embedding cache, once

`submit_array.sh` deliberately does not write the cache: several users submitting
overlapping ranges would each compute it, and a job whose cache files are missing stops
before fitting anything.

```bash
cd $REPO/experiments/pilot10
UMASK=002 PRECOMPUTE_ONLY=1 RUNS=$RUNS ./submit_runs.sh
```

## 5 · Everyone: agree one number before anyone submits

```bash
export BENCH=/dccstor/cgq4hls/Q/bench1
export REPO=/dccstor/cgq4hls/Q/futro/QBioCode     # or your own checkout, same commit
export RUNS=$BENCH/runs_cv/full1
cd $REPO/experiments/pilot10

./submit_array.sh --count
# 568260 tasks in /dccstor/.../full1/tasks.tsv
# sha256 3f9c…
```

The first person to run this **freezes** `tasks.tsv`. Everyone else reads the same file
and must see the **same total and the same sha256**. A different sha256 means somebody's
list came from a different manifest or different filters, and their task 5000 is not
yours — compare before submitting, not after.

Spot-check that a number means the same config for both of you:

```bash
./submit_array.sh --list 5000
```

> Never pass `REBUILD=1` on a shared tree. It renumbers every task, and from that moment
> two users running "their range" are running overlapping work.

## 6 · Everyone: take a range, in tmux

`submit_array.sh` does not submit and exit — it submits a slice, waits for your pending
queue to drain below `PEND_THRESHOLD`, submits the next, and so on for as long as your
range lasts. That can be hours or days, so it must outlive your ssh session.

tmux is per login node. Note which one you are on; you can only reattach from there.

```bash
hostname                       # e.g. ccc-login2 -- remember this
tmux new -s qbc
```

Inside the session:

```bash
source /dccstor/cgq4hls/Q/env_qbc/bin/activate
export BENCH=/dccstor/cgq4hls/Q/bench1
export RUNS=$BENCH/runs_cv/full1
umask 002
cd /dccstor/cgq4hls/Q/futro/QBioCode/experiments/pilot10

UMASK=002 ./submit_array.sh 1 200000        # <-- YOUR range only
```

Detach with `Ctrl-b` then `d`; the loop keeps running. Reattach, from the same login node:

```bash
tmux attach -t qbc
tmux ls                        # what is running here
```

Divide the range however you like, as long as the slices do not overlap:

| user | command |
|---|---|
| A | `UMASK=002 ./submit_array.sh 1 200000` |
| B | `UMASK=002 ./submit_array.sh 200001 400000` |
| C | `UMASK=002 ./submit_array.sh 400001 568260` |

Or let the script do the arithmetic — `SHARE=k/n` is the k-th of n equal contiguous
shares, so nobody has to compute boundaries:

```bash
UMASK=002 SHARE=1/3 ./submit_array.sh      # user A
UMASK=002 SHARE=2/3 ./submit_array.sh      # user B
UMASK=002 SHARE=3/3 ./submit_array.sh      # user C
```

`SHARE` and an explicit range must agree across the team: three users on `SHARE=k/3`
covers everything exactly once, but `SHARE=1/3` next to a hand-typed `1 200000` does not.

Settings worth knowing (full table in the script header): `MAX_INDEX` elements per array
(default 400, this cluster's per-user MPJOBS), `THROTTLE` how many run at once,
`PEND_THRESHOLD` how deep your queue gets before the loop waits, `SCHED=slurm` for
`sbatch --array`, `QUEUE`, `MEM`, `SLOTS`.

## 7 · The two skips

They are independent and answer different questions.

**`SKIP_DONE=1` — on by default.** Before starting Python, each element asks
`status.py --todo` whether its config is already complete, and exits in under a second if
so. It reads the *result files*, never the scheduler, so it is correct while other users'
jobs are running. This is what makes the recovery procedure for a shared run be "everyone
re-runs their own range" — finished work is absorbed, not repeated.

**`SKIP_EXISTING=1` — off by default.** Passes `++skip_existing=true`, so a job *adopts*
the (embedding, split, model) cells its config's earlier run directories already hold and
fits only the rest. Use it for a config that died part-way through, rather than restarting
it from split 1.

It is off by default because adopting rows is only sound if the config did not change
between the two runs. This run is `--split-mode manifest`, so the split manifest pins the
dataset's sha256 and a changed CSV is caught — but **nothing checks that you kept the same
`--n-trials`, models or embeddings**. If the YAMLs were regenerated with different
settings, do not adopt; start the config clean.

```bash
UMASK=002 SKIP_EXISTING=1 ./submit_array.sh 1 200000
```

## 8 · Watching a shared run

```bash
./status.py --runs-dir $RUNS --no-lsf
watch -n 300 ./status.py --runs-dir $RUNS --no-lsf
bjobs -A                      # YOUR arrays, by job id
```

Use `--no-lsf` and mean it. Two reasons, and the second one bites:

1. An array's elements share one job name, so `status.py` cannot attribute a live state to
   a single config even for your own jobs.
2. `bjobs` shows **only your own** jobs. A config a *coworker* is running right now has
   result rows but no job you can see, so without `--no-lsf` it is reported as `partial`
   or `failed`. Neither is true.

So in a shared run, trust `done` and read everything else as "not finished here yet". The
totals are still exact — they come from the result files, which all of you write into the
same tree. Only re-judge `failed` once everyone's queues are empty.

## 9 · When the queues are empty

Each user re-runs their own range once. `SKIP_DONE` absorbs everything complete, so this
costs seconds per finished config and picks up whatever died:

```bash
UMASK=002 ./submit_array.sh 1 200000
```

When a pass adds nothing, merge:

```bash
./collate_results.py --runs-dir $RUNS
```

---

## Failure modes, and what they look like

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied` in a job log, on another user's directory | umask was `022` when those files were written | section 1, then `chmod -R g+rwX` the affected tree; add `UMASK=002` |
| Two users' totals disagree in `--count` | different manifest or filters, or someone passed `REBUILD=1` | compare sha256, agree one tree, never rebuild |
| Elements re-fit work that is already done | the done-check failed — often `status.py` not finding `--runs-dir` | run section 3's re-run check; read the `note: the done-check failed` line in the log |
| Jobs stop immediately, having fitted nothing | embedding cache missing | section 4; `CACHE_CHECK=1` to verify a range first |
| `status.py` reports a coworker's running config as `failed` | `bjobs` is per-user | `--no-lsf`, and re-judge only when all queues are drained |
| A config keeps dying at the wall | the array carries one wall for the whole slice | submit that config on its own: `FORCE=1 DATASET='<id>' WALL=… ./submit_runs.sh` |
| The submit loop died when ssh dropped | not in tmux | section 6 |
