#!/bin/bash
# Submit the split-layout configs: one single-slot LSF job per (dataset, embedding, model).
#
#   ./submit_runs.sh                         every config that is not done and not live
#   ./submit_runs.sh runs/heart/*.yaml       exactly these (still skipping done/live ones)
#   ./status.py --todo | ./submit_runs.sh -  read the list from stdin
#   DATASET=heart MODEL=qsvc ./submit_runs.sh     filters: anchored regexes on the manifest
#   EMB=umap ./submit_runs.sh                     columns dataset / embedding / model
#   HOSTS='cccxc4[0-9]+' ./submit_runs.sh ...    only these candidate hosts (anchored regex)
#   DRY=1 ./submit_runs.sh                   print the bsub lines, submit nothing
#   PRECOMPUTE_ONLY=1 ./submit_runs.sh       write the embedding cache, submit nothing
#   FORCE=1 ./submit_runs.sh ...             submit even configs that are done or live
#   SKIP_EXISTING=1 ./submit_runs.sh         each job adopts the (embedding, split, model)
#                                            cells its config's earlier run directories
#                                            already hold, and fits only the rest. This is
#                                            what makes resubmitting a job killed at its
#                                            wall cumulative: without it the new run starts
#                                            at the first split again, so a config needing
#                                            two walls never finishes. See
#                                            qbiocode.apps.qprofiler.resume.
#   LIST_HOSTS=Intel_Platinum:128 ./submit_runs.sh   print a HOSTS= regex of every host of
#                                            that lshosts model (and ncpus), minus
#                                            advance-reserved (brsvs) hosts; submits nothing
#
# Generate the configs first: ./generate_pilot_configs.py --budget-hours 10.9 --layout split
# A manifest-mode run (generate_pilot_configs.py --split-mode manifest --run-id ID) lives in
# its own tree: RUNS=runs_cv/ID ./submit_runs.sh. Its jobs are named p10_ID_<config> (so two
# runs trees never collide in bjobs), and each job's -W comes from the MANIFEST.tsv wall
# column (per group, from --wall). An explicit WALL= overrides every job's wall.
# Before submitting, this writes the embedded features the jobs read (step 4 below).
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
# The interpreter the jobs run, and the one status.py is called with.
#
# Taken from the environment, not baked in. This line used to be a plain assignment of one
# person's virtualenv path, which was wrong twice over: an exported PY was silently
# discarded, and every other user got bsub payloads naming an interpreter they cannot
# execute -- `env: '...': Permission denied` from the cache step, then a job that dies the
# moment it is dispatched. Swapping in a different absolute path would only move the
# problem to the next person, so there is no default path at all: activate the environment,
# or export PY.
#
# It must be ABSOLUTE, because it is substituted into the bsub payload below and that runs
# on a compute node where the environment is not activated. `command -v` returns an
# absolute path, and the resolution happens here, on the submitting host.
#
# Deliberately NOT verified by importing qbiocode: tests/test_pilot_split_contract.py
# drives this script with DRY=1 and a stub CACHE_PY, under a PATH whose python need not
# have the package installed, and an import check would fail those runs for no reason.
PY=${PY:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "no usable python: PY='${PY}'." >&2
  echo "Activate the environment that has qbiocode, or export PY=/abs/path/to/python." >&2
  exit 1
fi
case $PY in
  /*) ;;
   *) PY=$(command -v "$PY") ;;   # the payload runs elsewhere; relative would not resolve
esac
# The interpreter of the embedding-cache step (4) only; the jobs always run $PY. A stub
# here lets a test drive DRY=1 without importing the package per call.
CACHE_PY=${CACHE_PY:-$PY}
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
# Manifest-mode runs carry a per-job wall in MANIFEST.tsv; WALL= set here overrides it, and
# 24:00 is the fallback for a job the manifest gives none.
WALL_SET=${WALL:-}
WALL=${WALL:-24:00}
THREADS=${THREADS:-1}
# '++' rather than '+': it sets the key whether or not the composed config defines it, so
# one spelling works both for the packaged config (which ships skip_existing: false) and
# for the generated protocols (which do not name it at all).
SKIP_OVERRIDE=""
[ "${SKIP_EXISTING:-0}" = "1" ] && SKIP_OVERRIDE=" ++skip_existing=true"
# TABPFN_ALLOW_CPU_LARGE_DATASET: TabPFN refuses more than 1000 training rows on a CPU, and
# every trial of the run's 27 largest classical jobs (1324-2600 rows) failed on it, killing
# the job ("Every tuning trial for 'tabpfn' failed"). The guard is about speed, not validity;
# a 1000-1200-row job measured tabpfn at 2.4-3.6 h, so those jobs need a long wall.
ENVV="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS TABPFN_ALLOW_CPU_LARGE_DATASET=1"
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
#
# HOSTS keeps only the candidates it matches, to put jobs on one host type on purpose --
# e.g. rerunning a job on the CPU type where it went wrong, to show the fix holds there.
# The ranking above prefers the fastest type, so without it the rerun would land on
# whichever type is fastest this cycle. No match is an error rather than a fallback to
# LSF's placement, which would quietly run the jobs somewhere else.
# ---------------------------------------------------------------------------
SPREAD=${SPREAD:-1}
HOSTS=${HOSTS:-}
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
  sort -k2,2n -k3,3gr -k4,4g | awk -v re="$HOSTS" 're == "" || $1 ~ ("^(" re ")$") {print $1}'
}

# LIST_HOSTS=MODEL[:NCPUS]: the hosts to pin a run to one CPU type, as a regex for HOSTS=.
# lshosts' model alone is not a CPU type here -- Intel_Platinum covers both the 56- and the
# 128-core nodes, at the same cpuf -- so give ncpus too. Same filters as candidate_hosts
# (server, no gpu/mg, maxmem >= MEM), and hosts inside an advance reservation (brsvs
# RSV_HOSTS, "host:used / total") are left out: jobs of ours cannot run there.
list_hosts() {
  local model=${LIST_HOSTS%%:*} ncpus="" reserved
  case $LIST_HOSTS in *:*) ncpus=${LIST_HOSTS#*:} ;; esac
  reserved=$(brsvs -w 2>/dev/null |
             awk '{for (i = 1; i <= NF; i++) if ($i ~ /^[^:\/]+:[0-9]+$/) {split($i, a, ":"); print a[1]}}' |
             LC_ALL=C sort -u)
  lshosts -w | awk -v mo="$model" -v nc="$ncpus" -v m="$MEM" -v rsv="$reserved" '
    BEGIN { n = split(rsv, rv, /[[:space:]]+/); for (i = 1; i <= n; i++) if (rv[i] != "") R[rv[i]] = 1 }
    NR > 1 && $8 == "Yes" && $3 == mo && (nc == "" || $5 == nc) {
      r = ""; for (i = 9; i <= NF; i++) r = r $i
      if (r ~ /gpu|mg/) next
      u = substr($6, length($6)); x = substr($6, 1, length($6) - 1) + 0
      g = (u == "T") ? x * 1024 : (u == "G") ? x : (u == "M") ? x / 1024 : 0
      if (g < m) next
      if ($1 in R) { nr++; next }
      h[++k] = $1 }
    END { for (i = 1; i <= k; i++) printf "%s%s", (i > 1 ? "|" : ""), h[i]
          if (k) printf "\n"
          printf "%d hosts of model %s%s (%d advance-reserved left out)\n", k, mo,
                 (nc == "" ? "" : ", ncpus " nc), nr > "/dev/stderr"
          exit (k ? 0 : 1) }'
}
if [ -n "${LIST_HOSTS:-}" ]; then list_hosts; exit; fi

[ -f "$MANIFEST" ] || { echo "no $MANIFEST -- run generate_pilot_configs.py --layout split first"; exit 1; }
# A manifest-mode tree names its run (MANIFEST.tsv run_id column): prefix the job names with
# it, so this tree's jobs are told apart from another run's in bjobs and status.py.
RUN_ID=$(awk -F'\t' 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "run_id") c = i; next}
                     c && $c != "" {print $c; exit}' "$MANIFEST")
[ -n "$RUN_ID" ] && JOB_PREFIX=p10_${RUN_ID}_
[ -n "$HOSTS" ] && [ "$SPREAD" != "1" ] && { echo "HOSTS needs SPREAD=1 (it filters SPREAD's candidates)"; exit 1; }

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
#    status.py's exit status is checked: read through `mapfile < <(...)` alone, a status.py
#    that died (a traceback, a missing runs dir) produced an empty list, which is
#    indistinguishable here from "everything is done" -- so the script said so and exited 0,
#    having submitted nothing and reported success.
if [ "${FORCE:-0}" != "1" ]; then
  todo=$("$PY" "$HERE/status.py" --runs-dir "$RUNS" --todo "${abs[@]}") || {
    echo "!! status.py failed (above); nothing submitted. FORCE=1 skips this step." >&2; exit 1; }
  mapfile -t abs <<< "$todo"
  # A single empty line is what `mapfile` makes of empty input.
  [ ${#abs[@]} -eq 1 ] && [ -z "${abs[0]}" ] && abs=()
  [ ${#abs[@]} -eq 0 ] && { echo "nothing to submit: every selected config is done, running or pending"; exit 0; }
fi

# 3. Filter and order through the manifest. Output: yaml <TAB> config <TAB> dataset <TAB> wall.
#    The wall is WALL= if set, else the manifest's wall column (manifest mode), else 24:00.
#    Quantum jobs go by expected hours, or by their wall where the manifest has none.
mapfile -t selected < <(
  printf '%s\n' "${abs[@]}" |
  awk -F'\t' -v ds="${DATASET:-}" -v em="${EMB:-}" -v mo="${MODEL:-}" \
      -v wset="$WALL_SET" -v wdef="$WALL" '
    BEGIN { split("tabpfn catboost mlp xgb rf svc lr dt nb", order, " ")
            for (i in order) rank[order[i]] = i }
    NR == FNR { want[$0] = 1; next }
    FNR == 1  { for (i = 1; i <= NF; i++) if ($i == "wall") wc = i; next }   # manifest header
    !($13 in want) { next }
    ds != "" && $2 !~ ("^(" ds ")$") { next }
    em != "" && $3 !~ ("^(" em ")$") { next }
    mo != "" && $4 !~ ("^(" mo ")$") { next }
    { w = (wset != "") ? wset : (wc && $wc != "") ? $wc : wdef
      h = $11
      if (h == "") { split(w, hm, ":"); h = hm[1] + hm[2] / 60 }
      key = ($5 == "quantum") ? sprintf("0 %012.3f", 1e6 - h) \
                              : sprintf("1 %012d", (($4 in rank) ? rank[$4] : 99))
      print key "\t" $13 "\t" $1 "\t" $2 "\t" w }
  ' - "$MANIFEST" | LC_ALL=C sort -t$'\t' -k1,1 -k3,3 | cut -f2-)
[ ${#selected[@]} -eq 0 ] && { echo "no config matched the selection (DATASET='${DATASET:-}' EMB='${EMB:-}' MODEL='${MODEL:-}')"; exit 1; }

# 4. The embedded features, before any job goes out. A job reads them from its config's
#    embedding_cache and never computes them, so every job of one (dataset, embedding) sees
#    the same features, whichever host it lands on. Computed in the jobs, seeded UMAP
#    differed between CPU types. Written here, once, on this host, for exactly the jobs
#    selected. Files already current are kept, so a resubmit only reads them. A job whose
#    files are missing stops before fitting anything, so this is not optional. DRY=1 only
#    lists what it would write and writes nothing. It runs --check, not --dry-run: both
#    list the same, but only --check exits non-zero on a missing or stale file, which is
#    what raises the WARNING. For split_mode: manifest configs the tool writes both stages
#    (final and tuning) of each config's own splits, so a short run's --splits subset is
#    all it builds.
cfgs=()
for s in "${selected[@]}"; do cfgs+=("${s%%$'\t'*}"); done
if [ "${DRY:-0}" = "1" ]; then
  echo "would precompute the embedding cache for ${#cfgs[@]} configs:" \
       "env $ENVV NUMBA_NUM_THREADS=$THREADS $CACHE_PY -m qbiocode.apps.qprofiler.embedding_cache CONFIG..." >&2
  env $ENVV NUMBA_NUM_THREADS=$THREADS "$CACHE_PY" -m qbiocode.apps.qprofiler.embedding_cache --check "${cfgs[@]}" >&2 ||
    echo "WARNING: the embedding cache cannot serve these jobs yet; a real submit writes it first" >&2
elif ! env $ENVV NUMBA_NUM_THREADS=$THREADS "$CACHE_PY" -m qbiocode.apps.qprofiler.embedding_cache "${cfgs[@]}" >&2; then
  echo "!! the embedding cache could not be written (above), so nothing was submitted" >&2
  exit 1
fi
# PRECOMPUTE_ONLY=1: stop here, with the cache written and nothing submitted -- e.g. to build a
# large corpus's cache once (as its own LSF job if it is long), then submit with a second call,
# whose step 4 then only confirms every file is current.
if [ "${PRECOMPUTE_ONLY:-0}" = "1" ] && [ "${DRY:-0}" != "1" ]; then
  echo "embedding cache written for ${#cfgs[@]} configs; PRECOMPUTE_ONLY=1, so nothing was submitted" >&2
  exit 0
fi

N=${#selected[@]}
hosts=()
if [ "$SPREAD" = "1" ]; then
  mapfile -t hosts < <(candidate_hosts)
  [ ${#hosts[@]} -eq 0 ] && [ -n "$HOSTS" ] && { echo "no candidate host matches HOSTS='$HOSTS'; nothing submitted" >&2; exit 1; }
  [ ${#hosts[@]} -eq 0 ] && echo "WARNING: SPREAD found no candidate host; submitting without -m" >&2
fi
H=${#hosts[@]}
G=$SPREAD_GROUP; [ "$G" -gt "$H" ] && G=$H
STRIDE=1; [ "$G" -gt 0 ] && [ $((H / G)) -gt 1 ] && STRIDE=$((H / G))

n_sub=0; n_q=0; n_fail=0; walls=""
for k in "${!selected[@]}"; do
  IFS=$'\t' read -r cfg name dataset wall <<< "${selected[$k]}"
  dsdir=$(dirname "$cfg")
  mkdir -p "$dsdir/lsf_logs"
  payload="cd $dsdir && export $ENVV && $PY -m qbiocode.apps.qprofiler.cli --config-dir=$dsdir --config-name=$name$SKIP_OVERRIDE"

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
             -W "$wall" "${mopt[@]}"
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
  case " $walls " in *" $wall "*) ;; *) walls="${walls:+$walls }$wall" ;; esac
  # awk, not `grep -qP`. -P is not portable: stock BSD grep (macOS /usr/bin/grep, "BSD
  # grep, GNU compatible 2.6.0-FreeBSD") rejects it outright -- "invalid option -- P",
  # exit 2 -- so there the quantum/classical tally in the closing summary was always
  # 0/all, silently, and the DRY=1 run that is meant to preview a submission printed a
  # usage block per job. It survives on a developer machine only when something like
  # ugrep shadows grep on PATH. Comparing the two fields is what the ordering step above
  # already does, and it does not depend on how an implementation spells \t either.
  awk -F'\t' -v n="$name" -v arm="quantum" '$1 == n && $5 == arm {found = 1}
                                            END {exit !found}' "$MANIFEST" &&
    n_q=$((n_q + 1))
done
verb=$([ "${DRY:-0}" = "1" ] && echo "would submit" || echo "submitted")
echo "$verb $n_sub jobs ($n_q quantum, $((n_sub - n_q)) classical) over $H candidate hosts; wall ${walls:-$WALL}, $SLOTS slot, ${MEM} GB each" >&2
[ "$n_fail" -gt 0 ] && echo "!! $n_fail bsub calls FAILED -- ./submit_runs.sh again resubmits exactly those" >&2
[ "${DRY:-0}" != "1" ] && echo "watch: $HERE/status.py   (placement: bjobs -o 'jobid job_name stat first_host')" >&2
[ "$n_fail" -eq 0 ]
