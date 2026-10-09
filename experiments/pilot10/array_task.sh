#!/bin/bash
# One element of a job array submitted by submit_array.sh: run QProfiler on one config.
#
# Not meant to be run by hand, though it can be -- QBC_TASK_INDEX=7 ./array_task.sh runs
# task 7 of the frozen task list, which is how a single failing task is reproduced
# interactively.
#
# The scheduler supplies the element's index; everything else arrives in the environment,
# which both bsub and sbatch propagate from the submitting shell by default:
#
#   QBC_TASKS         the frozen task list (TSV: yaml, config, dataset dir, wall)
#   QBC_TASK_OFFSET   added to the array index to get the line number in that file. A
#                     slice of tasks 801-1200 is submitted as array indices 1-400 with an
#                     offset of 800, because Slurm array ids must stay under MaxArraySize
#                     (1001 by default) -- the global index cannot be the array index.
#   QBC_PY            the interpreter
#   QBC_SKIP_EXISTING 1 to pass ++skip_existing=true, so the job adopts the cells its
#                     config's earlier run directories already hold
#   QBC_STATUS        status.py, used for the pre-flight "is this config already done?"
#                     check; empty to skip that check
#   QBC_RUNS          the runs tree, for status.py
#   QBC_ENVV          the thread-pinning exports, as one string of KEY=VALUE pairs
#
# Exit status is the job's: 0 when QProfiler finished (or when there was nothing to do),
# non-zero when it failed, so the scheduler records the element as EXIT/FAILED.
set -u

index=${QBC_TASK_INDEX:-${LSB_JOBINDEX:-${SLURM_ARRAY_TASK_ID:-}}}
offset=${QBC_TASK_OFFSET:-0}
tasks=${QBC_TASKS:-}
PY=${QBC_PY:-python}

[ -n "$index" ] || { echo "no array index: none of QBC_TASK_INDEX, LSB_JOBINDEX or SLURM_ARRAY_TASK_ID is set" >&2; exit 2; }
# Indices start at 1, as submit_array.sh submits them. LSF sets LSB_JOBINDEX=0 for a job
# that is NOT an array element, which would otherwise reach `sed -n "0{p;q;}"` -- an
# invalid line address -- and look like a task that does not exist.
case $index in
  ''|*[!0-9]*) echo "array index '$index' is not a number" >&2; exit 2 ;;
esac
[ "$index" -ge 1 ] || { echo "array index $index is not an element of an array: submit_array.sh numbers them from 1, and LSF uses 0 for a plain job" >&2; exit 2; }
[ -n "$tasks" ] && [ -f "$tasks" ] || { echo "QBC_TASKS does not name a task list: '${tasks}'" >&2; exit 2; }

line=$((index + offset))
# sed, not awk: one seek to the line wanted and quit, which matters when the list holds
# half a million lines and every element of the array reads it.
task=$(sed -n "${line}{p;q;}" "$tasks")
if [ -z "$task" ]; then
  echo "task $line is past the end of $tasks ($(wc -l < "$tasks" | tr -d ' ') tasks); nothing to do" >&2
  exit 0
fi

IFS=$'\t' read -r cfg name dsdir wall <<< "$task"
[ -f "$cfg" ] || { echo "task $line names a config that does not exist: $cfg" >&2; exit 2; }
echo "task $line (array index $index + offset $offset): $name  [wall $wall]"

# Pre-flight: a config whose results are already complete is not run again. This is what
# makes a resubmission of the whole range cheap -- the elements whose work is done exit in
# under a second instead of starting a Python process that would do nothing. It reads the
# result files, never the scheduler, so it is correct even while other elements run.
if [ -n "${QBC_STATUS:-}" ] && [ "${QBC_SKIP_DONE:-1}" = "1" ]; then
  todo=$("$PY" "$QBC_STATUS" --runs-dir "${QBC_RUNS:-$(dirname "$dsdir")}" --no-lsf --todo "$cfg" 2>/dev/null)
  status=$?
  if [ "$status" -eq 0 ] && [ -z "$todo" ]; then
    echo "already done (status.py --todo lists it as neither todo, failed nor partial); skipping"
    exit 0
  fi
  [ "$status" -ne 0 ] && echo "note: the done-check failed (status $status); running the job anyway" >&2
fi

cd "$dsdir" || exit 2
# shellcheck disable=SC2086  # QBC_ENVV is a deliberate list of KEY=VALUE words
[ -n "${QBC_ENVV:-}" ] && export ${QBC_ENVV}

cmd=("$PY" -m qbiocode.apps.qprofiler.cli "--config-dir=$dsdir" "--config-name=$name")
# '++', not '+': it sets the key whether or not the composed config already defines one.
[ "${QBC_SKIP_EXISTING:-0}" = "1" ] && cmd+=("++skip_existing=true")

echo "+ ${cmd[*]}"
"${cmd[@]}"
