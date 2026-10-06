#!/bin/bash
# Submit the 12 pilot configs as 12 independent LSF jobs -- one per dataset.
#
# One job per config, not one job for the sweep, because a job that dies takes exactly
# one dataset with it and is resubmitted on its own. The quantum parameter cache
# (quantum_param_dir) is also per-config, so there are no concurrent writers to it.
#
# Usage:  ./submit_pilot.sh            submit all 12
#         ./submit_pilot.sh 5          submit only pilot05
#         DRY=1 ./submit_pilot.sh      print the bsub lines without submitting
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/dccstor/boseukb/Q/envs/qbc/bin/python

# `normal` is this cluster's default queue and the only Open:Active general one alongside
# `night` (checked with bqueues). No queue here defines a RUNLIMIT or a MEMLIMIT, so -W
# below is the job's own wall limit and the memory figure is a scheduling reservation.
QUEUE=${QUEUE:-normal}

# ---------------------------------------------------------------------------
# Slots. n_jobs in the configs is 13; 16 leaves headroom for the parent and for
# catboost/xgb. This is not just a request: lsf.conf sets
#     LSB_RESOURCE_ENFORCE="cpu memory gpu"
# so the job is confined to a cpuset cgroup holding exactly the cores it was granted.
# Asking for fewer slots than n_jobs does not slow things down gracefully -- it packs 13
# worker processes onto fewer cores and multiplies every wall-clock estimate.
# ---------------------------------------------------------------------------
SLOTS=${SLOTS:-16}

# ---------------------------------------------------------------------------
# Memory, in GB. lsf.conf sets LSF_UNIT_FOR_LIMITS=GB, so this is a bare number and
# 'mem=32' means 32 GB. Do NOT write MB-style values: past jobs on this cluster
# submitted with rusage[mem=16000] and rusage[mem=4000] -- read as 16000 GB and 4000 GB
# against a 755 GB host -- and exited, while mem=64 and mem=8.00 ran fine.
#
# 32 is chosen to be safe under BOTH readings of rusage[mem]. lsb.params does not set
# RESOURCE_RESERVE_PER_SLOT, so the reservation should be per-job, but if it is per-slot
# then 32 x 16 = 512 GB still fits a 755 GB host, whereas the 64 this script used before
# would have asked 1024 GB and pended forever. Measured need is far below either: the
# tree is 13 loky workers, each a fresh interpreter with numpy/sklearn/xgboost/catboost
# (+torch for tabpfn/mlp), and the data itself is trivial -- the widest pilot dataset is
# 62 x 2000 and the largest kernel matrix is 569 x 569.
#
# Deliberately NOT passing -M: with memory enforcement on, a hard limit would have the
# cgroup OOM-kill a job that spikes near the end of a 20-hour run. The reservation gets
# the scheduling right without that failure mode.
MEM=${MEM:-32}

# A kill ceiling, not a reservation: a job that finishes early releases its slots, so
# headroom here costs nothing. Nor is an overrun total loss. qprofiler appends to
# ModelResults.csv at qprofiler.py:774, inside both the iteration and the embedding loop,
# so a job killed at the wall keeps every pass that finished. The checkpoint unit is the
# (iteration, embedding) pass rather than the model, despite what `_append_model_row`'s
# docstring suggests: the call sits *after* `model_run` returns, and `model_run` returns
# only once all 13 models of that pass have joined. So the loss from a kill is one pass,
# not one model and not the dataset. Still worth sizing above the prediction, though:
# each lost resample costs the paired comparison a degree of freedom. The PQK kernels are
# not exposed to a kill at all -- compute_pqk writes one independent .npz per pass into
# kernel_dump_dir.
#
# Sized from the measured cost model (cost_model.py; generate_pilot_configs.py --budget-hours
# prints each job's budgeted, expected and bound hours and suggests a wall). Measured
# 2026-09-28 for the shipped configs, generated at --budget-hours 10.9:
#
#   job          trials  budgeted  expected  bound
#   heart          12     10.87     12.77    21.86   statevector, 13 qubits
#   wdbc            6     10.78     11.88    23.92
#   hepatitis       8     10.41     10.41    13.97   mps, 19 qubits
#   te_n10         24      8.75     10.23    15.48
#   the other 8          <= 6.79   <= 7.48  <= 10.13
#
# "budgeted" is what --budget-hours checks: the kernel arms priced at ZZ/linear. "expected"
# prices their whole encoding x entanglement grid (cost_model.grid_factor). "bound" is the
# same job with its frozen winner at the grid's costliest point: every resample reuses what
# the iteration-0 search picked, so a search that lands on ZZ/P + full + max reps pays ~3x on
# every full fit. Simulated pass by pass with the winner drawn uniformly, and counted from
# when the jobs start (queue wait is extra), the LAST of the 12 finishes at a median of
# 13-14.5 h, and the chance that at least one job is killed is 26-41% at WALL=15:00, 10-16%
# at 18:00, 5-9% at 20:00 and 0.1-1.7% at 24:00. Hence the default below.
#
# The two bands carry very different confidence, and the MPS one was wrong until measured
# properly. STATEVECTOR is solid: per-circuit cost was validated on wdbc at 40, 120 and 200
# training rows and held to within 1% across a 25x span in circuit count, and re-confirmed at
# 13 qubits (71.8 ms at 40 rows, 69.9 ms at 160 -- 0.97x for 4x the rows). That is what theory
# demands: a statevector does the same FLOPs whatever the parameter values, so its cost cannot
# depend on the data.
#
# MPS is the opposite: cost tracks the bond dimension the feature map drives, so it DOES
# depend on the feature vectors, and it grows with training-set size. Fitting that growth on
# the 40->80 pairs alone (heart 13q 1.35x, hepatitis 19q 1.43x) gives an exponent of 0.51 and
# predicts 150 ms/circuit for heart at 160 rows. The measured value is 277.6 ms -- the
# two-point law understates the only regime the pilot actually runs in by 1.85x. The 40->160
# pair implies 0.958, near-linear in rows, and cost_model.MPS_ROW_ALPHA now takes that.
#
# The consequence was that the qubit band rule inverted at its own lower edge: at 13 qubits
# MPS costs ~4x the statevector at realistic row counts, and the two are within 2% at 40
# rows. The cache cliff between 10 and 13 qubits is real (16.4 -> 69.9 ms/circuit) but it
# does not make MPS the cheaper choice there. So the generator's statevector band now ends
# at 13 (STATEVECTOR_MAX_QUBITS), which moves heart -- the only job at that width -- onto
# the statevector. At 19 qubits MPS still wins by roughly 24x, so hepatitis and labor stay.
#
# Neither band models the classical arms or scheduler contention.
#
# If a job does run over, lower --budget-hours and regenerate -- do NOT cap maxiter. COBYLA
# needs about n_params+1 evaluations just to build its initial simplex (39 parameters for
# RealAmplitudes reps=2 at 13 qubits, 78 for EfficientSU2), so a low cap leaves vqc and qnn
# effectively untrained and manufactures a classical win. It would not buy time anyway:
# those arms measured flat in maxiter, their cost being circuit construction rather than
# the optimizer loop. That holds for COBYLA only. L_BFGS_B costs 190-580x per iteration
# (parameter-shift gradients), which is why the generator pins the qnn/vqc search to COBYLA.
WALL=${WALL:-24:00}

# One BLAS/OpenMP thread per worker. model_run.py fans the 13 models out over loky, and
# each worker otherwise starts its own pool sized to the machine (56 cores here), not to
# the job's slot allocation -- 13 x 56 threads on 16 slots is pure context-switching. The
# tuners are already sequential (_tuning.py passes n_jobs=1 on purpose), so nothing above
# this line wants the extra threads. CatBoost is the exception that these variables cannot
# reach: it uses its own pool, which is why thread_count is pinned to 1 in the configs.
# Raise this only if you also lower n_jobs in the configs.
THREADS=${THREADS:-1}
ENVV="OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS"

# ---------------------------------------------------------------------------
# Spread: one job per host. The pilot submitted 2026-09-28 14:59 put 5 of its 12 jobs on
# cccxc515. Jobs dispatched in one scheduling cycle all see the same "least loaded" host,
# because load indices do not refresh between them. It did not measurably slow them
# (a slot is a physical core there), but it put 5 datasets behind one node, and contention
# for what a node shares -- memory bandwidth, local disk -- is nothing the cost model prices.
#
# So every job gets its own -m host list, and the lists are DISJOINT slices of one ranked
# candidate list: of N jobs, job k gets candidates k, k+N, k+2N, ... (at most SPREAD_GROUP
# of them), the first marked preferred (+1). No two jobs of one submission can then share
# a host, and none is pinned to a single host that might fill before it dispatches. Hosts
# already running one of your pilot* jobs are left out, so a single-dataset resubmit does
# not land next to the survivors either.
#
# Candidates: bhosts status ok with room for the whole job (MAX - NJOBS >= SLOTS), maxmem
# >= MEM, lsload status ok, not a GPU or management host. Faster CPU factor first (the
# Intel_Platinum nodes are 15.0 against 12.5 for the E5s), then the idlest by ut. Ranked
# once, at submission. This does not move the kill time: lsb.params sets ABS_RUNLIMIT=Y,
# so -W is wall-clock on every host, not scaled by its CPU factor.
#
# SPREAD=0 drops -m and lets LSF place the jobs, as before.
# ---------------------------------------------------------------------------
SPREAD=${SPREAD:-1}
SPREAD_GROUP=${SPREAD_GROUP:-8}

candidate_hosts() {
  local busy
  busy=$(bjobs -r -noheader -o "job_name exec_host" 2>/dev/null |
         awk '$1 ~ /^pilot/ {split($2, h, ":"); print h[1]}')
  LC_ALL=C join \
      <(bhosts -w | awk -v s="$SLOTS" 'NR > 1 && $2 == "ok" && $4 - $5 >= s {print $1}' |
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
  awk -v busy="$busy" 'BEGIN {n = split(busy, b, /[[:space:]]+/); for (i = 1; i <= n; i++) x[b[i]]}
                       !($1 in x)' |
  sort -k2,2gr -k3,3g | awk '{print $1}'
}

mkdir -p "$HERE/lsf_logs"

shopt -s nullglob
configs=("$HERE"/configs/pilot*.yaml)
[ ${#configs[@]} -eq 0 ] && { echo "no configs found -- run generate_pilot_configs.py first"; exit 1; }

# Select first, submit second: the host slices below are cut by how many jobs go out.
selected=()
for cfg in "${configs[@]}"; do
  name=$(basename "$cfg" .yaml)
  # Positional args select datasets by number ("./submit_pilot.sh 5 7 9"); SKIP excludes
  # them ("SKIP=6 ./submit_pilot.sh"). Both exist because the 12 datasets are 12 INDEPENDENT
  # jobs, so a question about one of them -- an unresolved cost estimate, a rerun of a single
  # failure -- should never hold up the other eleven. Selecting a subset was previously one
  # dataset at a time, which made "all but heart" eleven invocations.
  if [ $# -ge 1 ]; then
    match=0
    for want in "$@"; do
      printf '%s' "$name" | grep -qE "^pilot0*${want}_" && { match=1; break; }
    done
    [ "$match" -eq 1 ] || continue
  fi
  skipped=0
  for skip in ${SKIP:-}; do
    printf '%s' "$name" | grep -qE "^pilot0*${skip}_" && { skipped=1; break; }
  done
  if [ "$skipped" -eq 1 ]; then
    echo "skipping $name (SKIP='$SKIP')" >&2
    continue
  fi
  selected+=("$cfg")
done
[ ${#selected[@]} -eq 0 ] && { echo "no config matched the selection"; exit 1; }

N=${#selected[@]}
hosts=()
if [ "$SPREAD" = "1" ]; then
  mapfile -t hosts < <(candidate_hosts)
  if [ ${#hosts[@]} -eq 0 ]; then
    echo "WARNING: SPREAD found no candidate host; submitting without -m" >&2
  elif [ ${#hosts[@]} -lt "$N" ]; then
    echo "WARNING: only ${#hosts[@]} candidate hosts for $N jobs; some will share one" >&2
  fi
fi
H=${#hosts[@]}

for k in "${!selected[@]}"; do
  cfg=${selected[$k]}
  name=$(basename "$cfg" .yaml)

  # Warn if the config would oversubscribe the cgroup we are about to ask for.
  njobs=$(awk '/^n_jobs:/{print $2; exit}' "$cfg")
  if [ -n "${njobs:-}" ] && [ "$njobs" -gt "$SLOTS" ] 2>/dev/null; then
    echo "WARNING: $name sets n_jobs=$njobs but SLOTS=$SLOTS; cpu enforcement will pack them." >&2
  fi

  cmd=("$PY" -m qbiocode.apps.qprofiler.cli
       "--config-dir=$HERE/configs" "--config-name=$name")

  # span[hosts=1] is REQUIRED, not tidiness. This cluster does scatter multi-slot
  # allocations -- a concurrent 4-slot job here held cccxc435:cccxc435:cccxc435:cccxc436.
  # qprofiler runs as ONE local process tree, so slots on a second host are unreachable
  # while the cpuset cgroup on the first host confines 13 workers to whatever landed
  # there. Without this the job silently runs at a fraction of the assumed parallelism.
  RES="span[hosts=1] rusage[mem=$MEM]"

  # Build the payload once so DRY prints exactly what a real submit sends. These used to
  # differ -- DRY showed an `env ...` exec form while bsub was handed a `cd && export &&`
  # shell string, and DRY omitted -o/-e entirely -- so the preview could not be trusted to
  # show what would run.
  payload="cd $HERE && export $ENVV && ${cmd[*]}"
  # An array, NOT `set --`: the script's own positional parameters are the dataset filter
  # ("./submit_pilot.sh 5"), read as $# and $1 at the top of this loop, so overwriting them
  # here would break selective submission on the second iteration onward.
  # This job's slice of the candidates (see Spread above). With fewer hosts than jobs a
  # slice would be empty, so it falls back to one host each, round-robin.
  mopt=()
  if [ "$H" -gt 0 ]; then
    group=()
    if [ "$H" -ge "$N" ]; then
      for ((i = k; i < H && ${#group[@]} < SPREAD_GROUP; i += N)); do group+=("${hosts[$i]}"); done
    else
      group=("${hosts[$((k % H))]}")
    fi
    mopt=(-m "${group[0]}+1${group[1]:+ ${group[*]:1}}")
  fi

  bsub_args=(-J "$name" -q "$QUEUE" -n "$SLOTS" -R "$RES" -W "$WALL" "${mopt[@]}"
             -o "$HERE/lsf_logs/$name.%J.out" -e "$HERE/lsf_logs/$name.%J.err"
             "$payload")

  if [ "${DRY:-0}" = "1" ]; then
    # Single-quote only the arguments that need it, rather than %q-escaping everything:
    # the point of DRY is that a human can read all 12 lines and also paste one verbatim.
    printf 'bsub'
    for a in "${bsub_args[@]}"; do
      case $a in
        *[[:space:]\[\]\&]*) printf " '%s'" "$a" ;;
        *)                     printf ' %s' "$a" ;;
      esac
    done
    printf '\n'
    continue
  fi
  bsub "${bsub_args[@]}"
done
if [ "${DRY:-0}" != "1" ] && [ "$H" -gt 0 ]; then
  echo "placement once dispatched: bjobs -o 'jobid job_name stat first_host'"
fi
