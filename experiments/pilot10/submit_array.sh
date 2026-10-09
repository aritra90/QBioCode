#!/bin/bash
# =============================================================================
# Submit a whole run as job ARRAYS, in slices, with the work split between users.
#
# submit_runs.sh submits one job per config -- which is the right shape for a few hundred
# jobs and the wrong one for a few hundred thousand: each bsub is a round trip, every job
# needs a name, and a user's pending queue fills long before the sweep is in. This submits
# the same configs as arrays of MAX_INDEX elements, waits for the pending count to drain,
# and submits the next round.
#
# The configs are frozen into a numbered task list ONCE (tasks.tsv, beside MANIFEST.tsv).
# After that a task number means the same config for everybody, which is what lets the work
# be divided by range: each user runs the same command with their own slice and no two of
# them submit the same config.
#
#   ./submit_array.sh                        every task (1 .. N)
#   ./submit_array.sh 1 200000               tasks 1-200000  -- user A
#   ./submit_array.sh 200001 400000          tasks 200001-400000  -- user B
#   SHARE=2/3 ./submit_array.sh              the second of three equal contiguous shares
#   SCHED=slurm ./submit_array.sh 1 1000     sbatch --array instead of bsub -J name[..]
#   DRY=1 ./submit_array.sh                  build the list, print the submit lines, submit nothing
#   REBUILD=1 ./submit_array.sh              rebuild the task list (CHANGES EVERY TASK NUMBER)
#   ./submit_array.sh --list 7               print task 7 and exit
#
# Dividing the work
#   Agree one number: `./submit_array.sh --count` prints the task total and the list's
#   sha256. Every user should see the same pair before anyone submits -- a different sha256
#   means somebody's list was built from a different manifest or different filters, and
#   their task 5000 is not yours. Then each takes a range, or SHARE=k/n.
#
# Settings
#   SCHED            lsf (default) or slurm
#   MAX_INDEX        elements per array (default 400 = this cluster's per-user MPJOBS)
#   THROTTLE         max elements of one array running at once (LSF %N, Slurm --array=..%N)
#   BATCH_SLICES     arrays queued per round before waiting (default 1)
#   PEND_THRESHOLD   wait until fewer than this many of your tasks are pending (default 200)
#   POLL_INTERVAL    seconds between pending-count checks (default 60)
#   QUEUE/PARTITION  LSF queue (default normal) / Slurm partition (default unset)
#   SLOTS, MEM       cores and GB per element (default 1 and 8, as submit_runs.sh)
#   WALL             one wall for every element. There is NO default: an array carries a
#                    single limit, so each slice takes the longest wall among its own tasks
#                    from MANIFEST.tsv's wall column, and a slice whose tasks name no wall
#                    is submitted with no limit at all -- which is right on a queue that
#                    imposes none, and is why nothing is invented here.
#   THREADS          per-process thread cap (default 1)
#   SKIP_EXISTING    1 to make every element adopt the (embedding, split, model) cells its
#                    config's earlier run directories already hold and fit only the rest
#                    (qbiocode.apps.qprofiler.resume). OFF by default: it is only sound
#                    when the config has not changed between the two runs, and in internal
#                    split mode nothing checks that for you.
#   SKIP_DONE        1 (default) to let each element exit immediately when its config's
#                    results are already complete. Costs one status.py call per element.
#   DATASET/EMB/MODEL  anchored regexes on MANIFEST.tsv, applied when the list is BUILT
#   CACHE_CHECK      1 to run embedding_cache --check over the selected range first
#
# The embedding cache is NOT written here. Several users submitting overlapping ranges
# would each compute it, and a job whose files are missing stops before fitting anything --
# so write it once, before anyone submits:
#     PRECOMPUTE_ONLY=1 RUNS=$RUNS ./submit_runs.sh
#
# Watching: ./status.py --runs-dir $RUNS --no-lsf  (an array's elements share one job name,
# so status.py cannot attribute live states to configs -- read progress from the results,
# and the queue from bjobs/squeue directly). Merge: ./collate_results.py --runs-dir $RUNS
# =============================================================================
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/dccstor/boseukb/Q/envs/qbc/bin/python}
RUNS=${RUNS:-$HERE/runs}
MANIFEST=$RUNS/MANIFEST.tsv
TASKS=${TASKS:-$RUNS/tasks.tsv}
META=$TASKS.meta
TASK_SCRIPT=${TASK_SCRIPT:-$HERE/array_task.sh}
LOGDIR=${LOGDIR:-$RUNS/array_logs}

SCHED=${SCHED:-lsf}
MAX_INDEX=${MAX_INDEX:-400}
THROTTLE=${THROTTLE:-200}
BATCH_SLICES=${BATCH_SLICES:-1}
PEND_THRESHOLD=${PEND_THRESHOLD:-200}
POLL_INTERVAL=${POLL_INTERVAL:-60}
QUEUE=${QUEUE:-normal}
PARTITION=${PARTITION:-}
SLOTS=${SLOTS:-1}
MEM=${MEM:-8}
# Empty means "no wall": neither -W nor --time is passed, and the queue's own policy (none,
# on this cluster) applies. An explicit WALL= overrides every slice's own longest wall.
WALL_SET=${WALL:-}
THREADS=${THREADS:-1}
ENVV="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS TABPFN_ALLOW_CPU_LARGE_DATASET=1"

case $SCHED in
  lsf|slurm) ;;
  *) echo "SCHED must be lsf or slurm; got '$SCHED'" >&2; exit 1 ;;
esac
[ -f "$MANIFEST" ] || { echo "no $MANIFEST -- generate the configs first (generate_pilot_configs.py)" >&2; exit 1; }
[ -f "$TASK_SCRIPT" ] || { echo "no task script at $TASK_SCRIPT" >&2; exit 1; }

# A manifest-mode tree names its run; the array's job name carries it so two runs' arrays
# are told apart in bjobs/squeue. 'a' distinguishes them from submit_runs.sh's p10_ jobs.
RUN_ID=$(awk -F'\t' 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "run_id") c = i; next}
                     c && $c != "" {print $c; exit}' "$MANIFEST")
JOB_NAME=${JOB_NAME:-p10a${RUN_ID:+_$RUN_ID}}

# ---------------------------------------------------------------------------
# 1. The frozen task list. Built once, from MANIFEST.tsv, in the same heaviest-first order
#    submit_runs.sh uses (quantum arms by expected hours, then the classical models by a
#    fixed rank), with the config name as the tie-break so the order is total and does not
#    depend on the filesystem. Line N is task N, for everyone, until REBUILD=1.
#
#    The build parameters go in tasks.tsv.meta. A later call with DIFFERENT filters would
#    produce a different numbering, and silently renumbering the list under a user who has
#    already submitted half of it would make two users run the same configs while others
#    ran none -- so that is refused rather than rebuilt.
#
#    Columns: yaml, config, the directory the job runs in, wall ('' where the manifest
#    gives none).
# ---------------------------------------------------------------------------
want_meta="manifest=$MANIFEST DATASET=${DATASET:-} EMB=${EMB:-} MODEL=${MODEL:-}"

build_tasks() {
  local tmp=$TASKS.$$.tmp
  mkdir -p "$(dirname "$TASKS")"
  awk -F'\t' -v ds="${DATASET:-}" -v em="${EMB:-}" -v mo="${MODEL:-}" '
    BEGIN { split("tabpfn catboost mlp xgb rf svc lr dt nb", order, " ")
            for (i in order) rank[order[i]] = i
            OFS = "\t" }
    FNR == 1 { for (i = 1; i <= NF; i++) if ($i == "wall") wc = i; next }
    ds != "" && $2 !~ ("^(" ds ")$") { next }
    em != "" && $3 !~ ("^(" em ")$") { next }
    mo != "" && $4 !~ ("^(" mo ")$") { next }
    { w = (wc && $wc != "") ? $wc : ""
      # Expected hours, for ordering only: the manifest column, else the wall when there
      # is one, else 0 -- which leaves those tasks ordered by config name alone.
      h = $11
      if (h == "") { if (w != "") {split(w, hm, ":"); h = hm[1] + hm[2] / 60} else h = 0 }
      key = ($5 == "quantum") ? sprintf("0 %012.3f", 1e6 - h) \
                              : sprintf("1 %012d", (($4 in rank) ? rank[$4] : 99))
      dir = $13; sub(/\/[^\/]*$/, "", dir)
      print key, $1, $13, dir, w }
  ' "$MANIFEST" |
  LC_ALL=C sort -t$'\t' -k1,1 -k2,2 |
  # Drop the two sort keys and put the columns in the order array_task.sh reads them.
  awk -F'\t' 'BEGIN {OFS = "\t"} {print $3, $2, $4, $5}' > "$tmp"
  if [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    echo "no config in $MANIFEST matched DATASET='${DATASET:-}' EMB='${EMB:-}' MODEL='${MODEL:-}'" >&2
    return 1
  fi
  mv "$tmp" "$TASKS"
  printf '%s\n' "$want_meta" > "$META"
  return 0
}

tasks_sha() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$TASKS" | cut -d' ' -f1
  elif command -v shasum   >/dev/null 2>&1; then shasum -a 256 "$TASKS" | cut -d' ' -f1
  else echo "(no sha256 tool)"; fi
}

if [ "${REBUILD:-0}" = "1" ] || [ ! -s "$TASKS" ]; then
  build_tasks || exit 1
  echo "built $TASKS" >&2
else
  have_meta=$(cat "$META" 2>/dev/null || echo "")
  if [ "$have_meta" != "$want_meta" ]; then
    echo "!! $TASKS was built with [$have_meta]" >&2
    echo "!! this call asks for      [$want_meta]" >&2
    echo "!! Renumbering the list would give every user a different task 5000. Use the" >&2
    echo "!! same filters, or REBUILD=1 once nobody is mid-submission." >&2
    exit 1
  fi
fi

TOTAL=$(wc -l < "$TASKS" | tr -d ' ')

# --count / --list: agree on the numbering before anyone submits.
if [ $# -ge 1 ] && [ "$1" = "--count" ]; then
  echo "$TOTAL tasks in $TASKS"
  echo "sha256 $(tasks_sha)"
  exit 0
fi
if [ $# -ge 1 ] && [ "$1" = "--list" ]; then
  [ $# -ge 2 ] || { echo "--list needs a task number" >&2; exit 1; }
  sed -n "$2{p;q;}" "$TASKS"
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Which tasks are mine: positional LO HI, SHARE=k/n, or everything.
# ---------------------------------------------------------------------------
if [ $# -eq 2 ]; then
  LO=$1; HI=$2
  echo "range given: tasks ${LO}-${HI} of $TOTAL" >&2
elif [ $# -ne 0 ]; then
  echo "usage: $0 [LO HI] | --count | --list N   (see the header for the env settings)" >&2
  exit 1
elif [ -n "${SHARE:-}" ]; then
  case $SHARE in
    [0-9]*/[0-9]*) ;;
    *) echo "SHARE must be k/n, e.g. SHARE=2/3; got '$SHARE'" >&2; exit 1 ;;
  esac
  k=${SHARE%%/*}; n=${SHARE##*/}
  { [ "$n" -ge 1 ] && [ "$k" -ge 1 ] && [ "$k" -le "$n" ]; } || {
    echo "SHARE=k/n needs 1 <= k <= n; got '$SHARE'" >&2; exit 1; }
  # Contiguous blocks, remainder spread over the first few shares, so the n blocks
  # partition 1..TOTAL exactly whatever the arithmetic.
  base=$((TOTAL / n)); rem=$((TOTAL % n))
  LO=$(( (k - 1) * base + (k - 1 < rem ? k - 1 : rem) + 1 ))
  HI=$(( LO + base + (k <= rem ? 1 : 0) - 1 ))
  echo "SHARE=$SHARE of $TOTAL tasks: mine are ${LO}-${HI}" >&2
else
  LO=1; HI=$TOTAL
  echo "no range given: every task, 1-$TOTAL" >&2
fi

case $LO$HI in *[!0-9]*) echo "LO and HI must be task numbers" >&2; exit 1 ;; esac
[ "$LO" -ge 1 ] || { echo "LO must be at least 1" >&2; exit 1; }
[ "$HI" -le "$TOTAL" ] || { echo "HI ($HI) is past the last task ($TOTAL)" >&2; exit 1; }
[ "$LO" -le "$HI" ] || { echo "empty range $LO-$HI" >&2; exit 0; }

# Slurm array ids must stay below MaxArraySize, so a slice can never be longer than that
# however large MAX_INDEX is set. Asked of the controller where it will answer.
if [ "$SCHED" = "slurm" ] && command -v scontrol >/dev/null 2>&1; then
  maxarr=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3; exit}')
  case ${maxarr:-} in
    ''|*[!0-9]*) ;;
    *) [ "$MAX_INDEX" -ge "$maxarr" ] && {
         MAX_INDEX=$((maxarr - 1))
         echo "MAX_INDEX capped at $MAX_INDEX by the cluster's MaxArraySize=$maxarr" >&2; } ;;
  esac
fi

# ---------------------------------------------------------------------------
# 3. Optional: confirm the embedding cache can serve this range before queueing it.
# ---------------------------------------------------------------------------
if [ "${CACHE_CHECK:-0}" = "1" ]; then
  echo "checking the embedding cache for tasks ${LO}-${HI}..." >&2
  # Through xargs rather than one argv: a range of a few hundred thousand configs is past
  # ARG_MAX, and xargs splits it into as many calls as it takes. It returns 123 when any
  # of them exited 1-125, which is how --check reports a missing or stale file. (xargs,
  # not `mapfile` into an array -- this script stays usable under bash 3.2, which has no
  # mapfile; see submit_runs.sh, which does need bash 4.)
  sed -n "${LO},${HI}p" "$TASKS" | cut -f1 |
    xargs env $ENVV "$PY" -m qbiocode.apps.qprofiler.embedding_cache --check >&2
  if [ $? -ne 0 ]; then
    echo "!! the embedding cache cannot serve these jobs; write it first with" >&2
    echo "!!     PRECOMPUTE_ONLY=1 RUNS=$RUNS $HERE/submit_runs.sh" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 4. Submit, in slices, waiting for the pending count to drain between rounds.
# ---------------------------------------------------------------------------
# Not under DRY=1: a preview should leave nothing behind.
[ "${DRY:-0}" = "1" ] || mkdir -p "$LOGDIR"
export QBC_TASKS="$TASKS" QBC_PY="$PY" QBC_RUNS="$RUNS" QBC_ENVV="$ENVV"
export QBC_STATUS="$HERE/status.py"
export QBC_SKIP_EXISTING="${SKIP_EXISTING:-0}"
export QBC_SKIP_DONE="${SKIP_DONE:-1}"

pending_count() {
  if [ "$SCHED" = "slurm" ]; then
    # -r expands an array into one line per element; without it a pending array counts as 1.
    squeue -h -r -u "$USER" -n "$JOB_NAME" -t PENDING 2>/dev/null | wc -l | tr -d ' '
  else
    bjobs -noheader -J "$JOB_NAME" 2>/dev/null | awk '$3 == "PEND"' | wc -l | tr -d ' '
  fi
}

slice_wall() {   # the longest wall among tasks $1..$2 as H:MM, or '' when none names one
  sed -n "${1},${2}p" "$TASKS" | cut -f4 |
  awk -F: 'BEGIN {best = -1} NF >= 2 {m = $1 * 60 + $2; if (m > best) {best = m}}
           END {if (best >= 0) printf "%d:%02d\n", best / 60, best % 60}'
}

submit_slice() {   # $1 = first task, $2 = last task, $3 = wall ('' for no limit)
  local lo=$1 hi=$2 wall=${3:-}
  # Separate statements: `local` expands ALL of its arguments before it assigns any of
  # them, so a count computed in the same `local` as hi reads hi while it is still unset.
  local count=$((hi - lo + 1))
  local offset=$((lo - 1))
  local args
  export QBC_TASK_OFFSET=$offset
  if [ "$SCHED" = "slurm" ]; then
    args=(--job-name "$JOB_NAME" --array "1-${count}%${THROTTLE}"
          --cpus-per-task "$SLOTS" --mem "${MEM}G" --export ALL
          --output "$LOGDIR/${JOB_NAME}_%A_%a.out"
          --error  "$LOGDIR/${JOB_NAME}_%A_%a.err")
    # Slurm wants D-HH:MM or HH:MM:SS, so H:MM becomes H:MM:00. Omitted entirely when no
    # task in the slice names a wall, leaving the partition's own limit in force.
    [ -n "$wall" ] && args+=(--time "${wall}:00")
    [ -n "$PARTITION" ] && args+=(--partition "$PARTITION")
    args+=("$TASK_SCRIPT")
    if [ "${DRY:-0}" = "1" ]; then
      echo "QBC_TASK_OFFSET=$offset sbatch ${args[*]}"
    else
      sbatch "${args[@]}" || return 1
    fi
  else
    args=(-J "${JOB_NAME}[1-${count}]%${THROTTLE}" -q "$QUEUE" -n "$SLOTS"
          -R "span[hosts=1] rusage[mem=$MEM]")
    [ -n "$wall" ] && args+=(-W "$wall")
    args+=(-o "$LOGDIR/${JOB_NAME}.%J.%I.out" -e "$LOGDIR/${JOB_NAME}.%J.%I.err"
           "$TASK_SCRIPT")
    if [ "${DRY:-0}" = "1" ]; then
      printf 'QBC_TASK_OFFSET=%s bsub' "$offset"
      for a in "${args[@]}"; do
        case $a in
          *[[:space:]\[\]\&]*) printf " '%s'" "$a" ;;
          *)                   printf ' %s' "$a" ;;
        esac
      done
      printf '\n'
    else
      bsub "${args[@]}" || return 1
    fi
  fi
  return 0
}

echo "$JOB_NAME: tasks ${LO}-${HI} as arrays of up to $MAX_INDEX, $THROTTLE running at once," \
     "$SLOTS slot and ${MEM} GB each, skip_existing=${SKIP_EXISTING:-0}" >&2

SLICE_LO=$LO
n_slices=0; n_tasks=0; n_fail=0; ROUND=1
while [ "$SLICE_LO" -le "$HI" ]; do
  echo "-- round $ROUND: up to $BATCH_SLICES slice(s) from task $SLICE_LO --" >&2
  this_round=0
  while [ "$SLICE_LO" -le "$HI" ] && [ "$this_round" -lt "$BATCH_SLICES" ]; do
    SLICE_HI=$((SLICE_LO + MAX_INDEX - 1))
    [ "$SLICE_HI" -gt "$HI" ] && SLICE_HI=$HI
    wall=${WALL_SET:-$(slice_wall "$SLICE_LO" "$SLICE_HI")}
    if submit_slice "$SLICE_LO" "$SLICE_HI" "$wall"; then
      n_tasks=$((n_tasks + SLICE_HI - SLICE_LO + 1))
    else
      echo "!! submit FAILED for tasks ${SLICE_LO}-${SLICE_HI}; rerun this script with" \
           "that range to retry exactly those" >&2
      n_fail=$((n_fail + 1))
    fi
    n_slices=$((n_slices + 1))
    this_round=$((this_round + 1))
    SLICE_LO=$((SLICE_HI + 1))
  done

  # More to come: wait for the queue to drain below the threshold. DRY=1 submits nothing,
  # so there is nothing to wait for and the loop would otherwise poll a count that never
  # moves for as long as the sweep is long.
  if [ "$SLICE_LO" -le "$HI" ] && [ "${DRY:-0}" != "1" ]; then
    PEND=$(pending_count)
    echo "  queued $n_slices slice(s) so far; pending $PEND, waiting for < $PEND_THRESHOLD" >&2
    while [ "${PEND:-0}" -ge "$PEND_THRESHOLD" ]; do
      sleep "$POLL_INTERVAL"
      PEND=$(pending_count)
      echo "  $(date '+%H:%M:%S')  pending: $PEND" >&2
    done
    echo "  pending down to $PEND -- next round" >&2
  fi
  ROUND=$((ROUND + 1))
done

verb=$([ "${DRY:-0}" = "1" ] && echo "would submit" || echo "submitted")
echo "$verb $n_slices slice(s), $n_tasks task(s), covering ${LO}-${HI} of $TOTAL" >&2
[ "$n_fail" -gt 0 ] && echo "!! $n_fail slice(s) FAILED to submit" >&2
if [ "${DRY:-0}" != "1" ]; then
  if [ "$SCHED" = "slurm" ]; then
    echo "watch:  squeue -u $USER -n $JOB_NAME        kill: scancel -n $JOB_NAME" >&2
  else
    echo "watch:  bjobs -J $JOB_NAME                  kill: bkill -J $JOB_NAME" >&2
    echo "states: bjobs -noheader -J $JOB_NAME | awk '{print \$3}' | sort | uniq -c" >&2
  fi
  echo "results: $HERE/status.py --runs-dir $RUNS --no-lsf" >&2
fi
[ "$n_fail" -eq 0 ]
