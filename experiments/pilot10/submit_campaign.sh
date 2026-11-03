#!/bin/bash
# =============================================================================
# Submit every run tree of one campaign, one after another, through submit_array.sh.
#
#   ./submit_campaign.sh full1-qt-rdm1 full1-qt-rdm2nn ... tcga1-qt-rdm2
#   DRY=1 ./submit_campaign.sh <run ids...>     preview: build task lists, submit nothing
#   ./submit_campaign.sh --status <run ids...>  done / partial / failed per run, no submit
#
# Between runs it WAITS until the user's pending jobs -- all of them, not one run's --
# drop below PEND_THRESHOLD (default 200). submit_array.sh cannot do that for us: its own
# wait only runs between the slices of ONE run, and a run of <= MAX_INDEX (400) tasks is
# a single slice, so it submits and returns at once. Six such runs back to back would
# put ~1400 elements in the queue against a per-user MPJOBS of 400. Counting every
# pending job of the user, rather than one job name, is what makes the limit hold across
# runs; it also counts jobs this script did not submit, which is the conservative side.
#
# Every run is checked BEFORE anything is submitted: that its tree exists, and a DRY pass
# of submit_array.sh, which runs its interpreter and embedding-cache checks. One bad run
# tree stops the campaign before the first job is queued, rather than after half of it.
#
# Settings pass through to submit_array.sh: UMASK, SHARE, MEM, SCHED, THROTTLE,
# PEND_THRESHOLD, QUEUE, MAX_INDEX, ... RUNS_ROOT is where the run trees live.
# =============================================================================
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
RUNS_ROOT=${RUNS_ROOT:-/dccstor/cgq4hls/Q/bench1/runs_cv}

STATUS_ONLY=0
if [ "${1:-}" = "--status" ]; then
  STATUS_ONLY=1; shift
fi
if [ $# -eq 0 ]; then
  echo "usage: $0 [--status] <run id> [<run id> ...]   (RUNS_ROOT=$RUNS_ROOT)" >&2
  exit 1
fi

# 1. Every run tree must exist before any is submitted.
missing=0
for rid in "$@"; do
  if [ ! -f "$RUNS_ROOT/$rid/MANIFEST.tsv" ]; then
    echo "!! $rid: no $RUNS_ROOT/$rid/MANIFEST.tsv -- generate it first" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1

if [ "$STATUS_ONLY" -eq 1 ]; then
  for rid in "$@"; do
    echo "=== $rid"
    "$HERE/status.py" --runs-dir "$RUNS_ROOT/$rid" --no-lsf | tail -2
  done
  exit 0
fi

# 2. Every run checked up front, by a DRY pass of submit_array.sh over it. Its
#    interpreter check (import qbiocode) and its CACHE_CHECK both run before DRY takes
#    effect, so a dry pass performs every pre-flight check and queues nothing. Without
#    this, run 4's missing cache would surface only after runs 1-3 were submitted.
for rid in "$@"; do
  echo "--- pre-flight: $rid" >&2
  if ! out=$(RUNS="$RUNS_ROOT/$rid" DRY=1 "$HERE/submit_array.sh" 2>&1); then
    printf '%s\n' "$out" | tail -20 >&2
    echo "!! $rid failed its pre-flight checks; NOTHING has been submitted" >&2
    exit 1
  fi
  printf '%s\n' "$out" | grep -E "tasks in|would submit|cannot serve|SHARE" >&2
done
if [ "${DRY:-0}" = "1" ]; then
  echo "DRY=1: every run passed its pre-flight checks; nothing submitted" >&2
  exit 0
fi

# 3. Submit, one run at a time, waiting between runs for the user's queue to drain.
PEND_THRESHOLD=${PEND_THRESHOLD:-200}
POLL_INTERVAL=${POLL_INTERVAL:-120}
user_pending() {
  if [ "${SCHED:-lsf}" = "slurm" ]; then
    squeue -h -r -u "$USER" -t PENDING 2>/dev/null | wc -l | tr -d ' '
  else
    bjobs -noheader -u "$USER" -p 2>/dev/null | wc -l | tr -d ' '
  fi
}
wait_for_queue() {
  local pend
  pend=$(user_pending)
  while [ "${pend:-0}" -ge "$PEND_THRESHOLD" ]; do
    echo "  $(date '+%H:%M:%S')  $pend of your jobs pending; waiting for < $PEND_THRESHOLD" >&2
    sleep "$POLL_INTERVAL"
    pend=$(user_pending)
  done
}
n=0
for rid in "$@"; do
  n=$((n + 1))
  [ "$n" -gt 1 ] && wait_for_queue
  echo
  echo "=================== $n/$# : $rid ===================" >&2
  if ! RUNS="$RUNS_ROOT/$rid" "$HERE/submit_array.sh"; then
    echo "!! $rid: submit_array.sh failed; stopping here. Runs 1..$((n - 1)) are" \
         "queued; rerun this script with the remaining ids" >&2
    exit 1
  fi
done
echo
echo "submitted $# run(s). Watch them with:  $0 --status $*" >&2
