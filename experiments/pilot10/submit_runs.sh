#!/bin/bash
# Submit the split-layout configs: one single-slot LSF job per (dataset, embedding, model).
#
#   ./submit_runs.sh                         every config that is not done and not live
#   ./submit_runs.sh runs/heart/*.yaml       exactly these (still skipping done/live ones)
#   ./status.py --todo | ./submit_runs.sh -  read the list from stdin
#   DATASET=heart MODEL=qsvc ./submit_runs.sh     filters: anchored regexes on the manifest
#   EMB=umap ./submit_runs.sh                     columns dataset / embedding / model
#   DRY=1 ./submit_runs.sh                   print the bsub lines, submit nothing
#   FORCE=1 ./submit_runs.sh ...             submit even configs that are done or live
#
# Generate the configs first: ./generate_pilot_configs.py --budget-hours 10.9 --layout split
# Watch: ./status.py      Merge: ./collate_results.py
#
# Why one job per model: submit_pilot.sh ran a dataset as one 16-slot job whose wall was
# n_embeddings x its slowest arm, so wdbc (pca then umap, qsvc 5.9 h each) took 11.9 h while
# its nine classical arms finished in minutes and then held their slots idle. Split, every
# arm of every embedding runs at once and the pilot's wall is the single slowest arm. The
# science is unchanged -- each YAML equals its combined parent except for the model, the
# embedding, n_jobs and its output paths (tests/test_pilot_split_contract.py) -- and every
# job owns every path it writes, so no two jobs share a file.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/dccstor/boseukb/Q/envs/qbc/bin/python
RUNS=${RUNS:-$HERE/runs}
MANIFEST=$RUNS/MANIFEST.tsv
QUEUE=${QUEUE:-normal}

# One model, n_jobs 1: one process, one core. The tuners are sequential already
# (_tuning.py passes n_jobs=1), and catboost's thread_count and xgb's n_jobs are pinned to
# 1 in the configs, so a second slot would sit idle.
SLOTS=${SLOTS:-1}
# GB (LSF_UNIT_FOR_LIMITS=GB -- do not write MB values; see submit_pilot.sh). A whole
# 13-model combined job peaked at 5-8 GB ("Max Memory" in lsf_logs/pilot*.out), so one
# model fits in 8 under either reading of rusage[mem] (per job or per slot; SLOTS=1 makes
# them the same). No -M, for the reason in submit_pilot.sh.
MEM=${MEM:-8}
# ABS_RUNLIMIT=Y: wall-clock. The slowest arm is heart qsvc at 12.8 h expected and 21.9 h
# frozen-winner bound (MANIFEST.tsv exp_h / bound_h; the generator prints the suggestion),
# so 24:00 keeps a margin over the worst case. A kill ceiling, not a reservation.
WALL=${WALL:-24:00}
THREADS=${THREADS:-1}
ENVV="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS"
JOB_PREFIX=p10_

# ---------------------------------------------------------------------------
# Spread. 208 single-slot jobs left to LSF land wherever the one "least loaded" host of
# that scheduling cycle is -- the 2026-09-28 14:59 pilot put 5 of 12 jobs on one node that
# way -- and a 56-core node would take 56 of these. Node-shared resources (memory
# bandwidth, which statevector simulation leans on; local disk) are not in the cost model.
#
# So each job gets an -m list: a preferred host (+1) and SPREAD_GROUP-1 backups, so that it
# is never pinned to one host that might fill. Hosts are ranked by how many of YOUR jobs
# already run there (fewest first), then CPU factor (fastest first), then utilisation.
# With N jobs and H hosts:
#   N <= H  disjoint slices: job k gets hosts k, k+N, k+2N, ... -- no two jobs share a host
#   N >  H  job k prefers host k mod H, backups strided H/SPREAD_GROUP apart, so every host
#           is preferred by N/H jobs (about 2 per host at 208 jobs on ~100 hosts).
# Jobs go out heaviest first (quantum arms by expected hours, then tabpfn, catboost, mlp,
# xgb, rf, svc, lr, dt, nb), so the long arms both dispatch first and take the fastest
# hosts: longest-first is the classic makespan heuristic, and here it also means the
# 12-hour qsvc jobs are not queued behind 2-minute naive-Bayes ones.
#
# Candidates: bhosts ok with >= MIN_FREE free slots, maxmem >= MEM, lsload ok, not a GPU or
# management host. SPREAD=0 drops -m and lets LSF place everything.
# ---------------------------------------------------------------------------
SPREAD=${SPREAD:-1}
SPREAD_GROUP=${SPREAD_GROUP:-6}
MIN_FREE=${MIN_FREE:-$SLOTS}

candidate_hosts() {
  local mine
  # exec_host is "h", "h:h" or "4*h"; one count per job, on its first host.
  mine=$(bjobs -r -noheader -o "exec_host" 2>/dev/null |
         awk '{split($1, h, ":"); sub(/^[0-9]+\*/, "", h[1]); print h[1]}' | LC_ALL=C sort | uniq -c |
         awk '{print $2 "=" $1}')
  LC_ALL=C join \
      <(bhosts -w | awk -v s="$MIN_FREE" 'NR > 1 && $2 == "ok" && $4 - $5 >= s {print $1}' |
        LC_ALL=C sort) \
      <(lshosts -w | awk -v m="$MEM" 'NR > 1 && $8 == "Yes" {
            r = ""; for (i = 9; i <= NF; i++) r = r $i
            if (r ~ /gpu|mg/) next
            u = substr($6, length($6)); x = substr($6, 1, length($6) - 1) + 0
            g = (u == "T") ? x * 1024 : (u == "G") ? x : (u == "M") ? x / 1024 : 0
            if (g >= m) print $1, $4 }' | LC_ALL=C sort) |
  LC_ALL=C join - <(lsload -w -I ut 2>/dev/null |
                    awk 'NR > 1 && $2 == "ok" {sub("%", "", $3); print $1, $3}' |
                    LC_ALL=C sort) |
  awk -v mine="$mine" 'BEGIN {n = split(mine, m, /[[:space:]]+/)
                              for (i = 1; i <= n; i++) {split(m[i], kv, "="); c[kv[1]] = kv[2]}}
                       {print $1, ($1 in c) ? c[$1] : 0, $2, $3}' |
  sort -k2,2n -k3,3gr -k4,4g | awk '{print $1}'
}

[ -f "$MANIFEST" ] || { echo "no $MANIFEST -- run generate_pilot_configs.py --layout split first"; exit 1; }

# 1. Candidates: the arguments, stdin ("-"), or every config.
shopt -s nullglob
cands=()
if [ $# -eq 1 ] && [ "$1" = "-" ]; then
  mapfile -t cands
elif [ $# -ge 1 ]; then
  cands=("$@")
else
  cands=("$RUNS"/*/*.yaml)
fi
[ ${#cands[@]} -eq 0 ] && { echo "no configs given or found under $RUNS"; exit 1; }
abs=()
for c in "${cands[@]}"; do
  [ -f "$c" ] || { echo "no such config: $c" >&2; exit 1; }
  abs+=("$(cd "$(dirname "$c")" && pwd)/$(basename "$c")")
done

# 2. Unless FORCE=1, drop what is done or already queued/running -- so this is safe to
#    re-run after a partial failure, and never puts a second copy of a live job in flight.
if [ "${FORCE:-0}" != "1" ]; then
  mapfile -t abs < <("$PY" "$HERE/status.py" --runs-dir "$RUNS" --todo "${abs[@]}")
  [ ${#abs[@]} -eq 0 ] && { echo "nothing to submit: every selected config is done, running or pending"; exit 0; }
fi

# 3. Filter and order through the manifest. Output: yaml <TAB> config <TAB> dataset.
mapfile -t selected < <(
  printf '%s\n' "${abs[@]}" |
  awk -F'\t' -v ds="${DATASET:-}" -v em="${EMB:-}" -v mo="${MODEL:-}" '
    BEGIN { split("tabpfn catboost mlp xgb rf svc lr dt nb", order, " ")
            for (i in order) rank[order[i]] = i }
    NR == FNR { want[$0] = 1; next }
    FNR == 1  { next }                       # manifest header
    !($13 in want) { next }
    ds != "" && $2 !~ ("^(" ds ")$") { next }
    em != "" && $3 !~ ("^(" em ")$") { next }
    mo != "" && $4 !~ ("^(" mo ")$") { next }
    { key = ($5 == "quantum") ? sprintf("0 %012.3f", 1e6 - $11) \
                              : sprintf("1 %012d", (($4 in rank) ? rank[$4] : 99))
      print key "\t" $13 "\t" $1 "\t" $2 }
  ' - "$MANIFEST" | LC_ALL=C sort -t$'\t' -k1,1 -k3,3 | cut -f2-)
[ ${#selected[@]} -eq 0 ] && { echo "no config matched the selection (DATASET='${DATASET:-}' EMB='${EMB:-}' MODEL='${MODEL:-}')"; exit 1; }

N=${#selected[@]}
hosts=()
if [ "$SPREAD" = "1" ]; then
  mapfile -t hosts < <(candidate_hosts)
  [ ${#hosts[@]} -eq 0 ] && echo "WARNING: SPREAD found no candidate host; submitting without -m" >&2
fi
H=${#hosts[@]}
G=$SPREAD_GROUP; [ "$G" -gt "$H" ] && G=$H
STRIDE=1; [ "$G" -gt 0 ] && [ $((H / G)) -gt 1 ] && STRIDE=$((H / G))

n_sub=0; n_q=0; n_fail=0
for k in "${!selected[@]}"; do
  IFS=$'\t' read -r cfg name dataset <<< "${selected[$k]}"
  dsdir=$(dirname "$cfg")
  mkdir -p "$dsdir/lsf_logs"
  payload="cd $dsdir && export $ENVV && $PY -m qbiocode.apps.qprofiler.cli --config-dir=$dsdir --config-name=$name"

  mopt=()
  if [ "$H" -gt 0 ]; then
    group=()
    if [ "$H" -ge "$N" ]; then
      for ((i = k; i < H && ${#group[@]} < G; i += N)); do group+=("${hosts[$i]}"); done
    else
      for ((j = 0; j < G; j++)); do group+=("${hosts[$(((k + j * STRIDE) % H))]}"); done
    fi
    mopt=(-m "${group[0]}+1${group[1]:+ ${group[*]:1}}")
  fi

  bsub_args=(-J "$JOB_PREFIX$name" -q "$QUEUE" -n "$SLOTS" -R "span[hosts=1] rusage[mem=$MEM]"
             -W "$WALL" "${mopt[@]}"
             -o "$dsdir/lsf_logs/$name.%J.out" -e "$dsdir/lsf_logs/$name.%J.err"
             "$payload")
  if [ "${DRY:-0}" = "1" ]; then
    printf 'bsub'
    for a in "${bsub_args[@]}"; do
      case $a in
        *[[:space:]\[\]\&]*) printf " '%s'" "$a" ;;
        *)                   printf ' %s' "$a" ;;
      esac
    done
    printf '\n'
  elif ! bsub "${bsub_args[@]}"; then
    echo "!! bsub failed for $name" >&2
    n_fail=$((n_fail + 1))
    continue
  fi
  n_sub=$((n_sub + 1))
  grep -qP "^$name\t[^\t]*\t[^\t]*\t[^\t]*\tquantum\t" "$MANIFEST" && n_q=$((n_q + 1))
done
verb=$([ "${DRY:-0}" = "1" ] && echo "would submit" || echo "submitted")
echo "$verb $n_sub jobs ($n_q quantum, $((n_sub - n_q)) classical) over $H candidate hosts; wall $WALL, $SLOTS slot, ${MEM} GB each" >&2
[ "$n_fail" -gt 0 ] && echo "!! $n_fail bsub calls FAILED -- ./submit_runs.sh again resubmits exactly those" >&2
[ "${DRY:-0}" != "1" ] && echo "watch: $HERE/status.py   (placement: bjobs -o 'jobid job_name stat first_host')" >&2
[ "$n_fail" -eq 0 ]
