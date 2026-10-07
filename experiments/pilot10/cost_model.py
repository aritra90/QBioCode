"""Measured wall-clock model for the quantum arms, used to size the trial budget.

Everything here is calibrated against the real code path: /tmp/cost_probe.py timed one
fit of each quantum arm through the real ``compute_*`` functions, with args loaded from
the real configs, on row-subsampled data at five widths (8 and 10 qubits statevector, 13
and 19 qubits MPS). The numbers below are those measurements. Re-derive them by rerunning
that probe if the simulator, qiskit version or host class changes.

Two cost laws, because the arms do not behave alike:

  qsvc, pqk   cost = circuits x seconds-per-circuit. Circuit count is closed form --
              n_tr(n_tr-1)/2 + n_te*n_tr kernel entries for qsvc, one embedding per row
              for pqk -- so a per-circuit cost measured at 40 rows transfers to 455 rows
              exactly, and the only thing extrapolated is a constant.

  qnn, vqc    cost = a fixed per-fit overhead + an optimizer loop. Two-point probes
              (maxiter 10 vs 30) measured the loop directly:
                  vqc  heart 13q MPS   131.5s -> 132.2s    overhead 131.1s, 0.035s/iter
                  vqc  labor 16q MPS   217.9s -> 215.6s    overhead 219.0s, flat
                  qnn  heart 13q MPS     2.8s ->   2.8s    overhead   2.8s, flat
              These arms are dominated by circuit construction and transpilation, which
              happens once per fit, not by the variational loop. A single-point probe
              cannot see that and forces you to assume cost scales with maxiter, which
              overstates vqc by ~57x -- that error is what produced an earlier estimate of
              47h for heart and nearly sent the pilot in over a wall it could not meet.

The pilot's wall clock is the SLOWEST JOB, not the total: the datasets run as independent
LSF jobs, so cutting the dataset count buys nothing and only per-dataset work matters.
"""
import math

TEST_SIZE = 0.20
VALIDATION_SPLIT = 0.25            # configs set this; _tuning defaults to it too
MAXITER_MEAN = 125                 # mean of the searched maxiter range {low: 50, high: 200}
PROBE_NTR = 40                     # training rows the calibration probe subsampled to

# The searched reps grid (the template's qsvc/pqk gridsearch blocks), its mean, and the
# probe's reps=2 (qsvc) / reps=4 (pqk) baseline.
REPS_GRID = {"qsvc": (1, 2, 3, 4), "pqk": (1, 2, 3, 4, 6)}
REPS_MEAN = {a: sum(g) / len(g) for a, g in REPS_GRID.items()}      # 2.5, 3.2
REPS_PROBE = {"qsvc": 2.0, "pqk": 4.0}

KERNEL_ARMS = ("qsvc", "pqk")
VAR_ARMS = ("qnn", "vqc")
ARMS = KERNEL_ARMS + VAR_ARMS

# ---------------------------------------------------------------------------
# Calibration. (arm, backend, qubits) -> seconds per circuit, at reps = REPS_PROBE.
# Backed out of the probe as measured_seconds / (circuits x reps_factor).
# ---------------------------------------------------------------------------
UNIT_S_PER_CIRCUIT = {
    ("qsvc", "sv", 8): 13.5 / 1180 / 1.0,
    ("qsvc", "sv", 10): 19.4 / 1180 / 1.0,
    # Measured, not interpolated, and it has to be here: without it _interp clamps a 13q
    # statevector to the 10q value of 16.4 ms/circuit, which is 4.25x optimistic against the
    # 69.9 ms actually measured -- the cache cliff between 10 and 13 qubits is steeper than
    # the 8->10 slope predicts, so extrapolating across it understates the cost badly.
    # Two legs, n_tr=40 and n_tr=160, came in at 71.8 and 69.9 ms: flat in rows at 0.97x for
    # 4x the rows, which is why no row_scale applies to sv and why the MPS row law does not
    # leak into this entry. The 40-row leg also read 1.00x against a reference taken days
    # earlier on the same node, so the pair doubles as the control clearing that node of
    # throttling -- the same window that measured MPS at 277.6 ms.
    ("qsvc", "sv", 13): 84.7670545578 / 1180 / 1.0,
    ("qsvc", "mps", 13): 86.8 / 1180 / 1.0,
    ("qsvc", "mps", 19): 118.7 / 1180 / 1.0,
    ("pqk", "sv", 8): 5.9 / 50 / 1.0,
    ("pqk", "sv", 10): 10.1 / 50 / 1.0,
    ("pqk", "mps", 13): 1.9 / 50 / 1.0,
    ("pqk", "mps", 19): 2.6 / 50 / 1.0,
}

# (arm, backend, qubits) -> measured seconds of fixed per-fit overhead. Width-bound, not
# row-bound: wdbc (455 rows) and spect (214 rows) both measured 7.3s for vqc at 8 qubits,
# which is the evidence that this term really is setup and not work over rows.
OVERHEAD_S = {
    ("qnn", "mps", 13): 3.3, ("vqc", "mps", 13): 129.3,
    ("qnn", "mps", 16): 4.6, ("vqc", "mps", 16): 219.0,
    ("qnn", "mps", 19): 6.7, ("vqc", "mps", 19): 314.5,
    ("qnn", "sv", 10): 7.4, ("vqc", "sv", 10): 13.7,
    ("qnn", "sv", 8): 4.55, ("vqc", "sv", 8): 7.3,
}

# Seconds per row per optimizer iteration, from the vqc two-point slope at 13q MPS:
# (132.2 - 131.5) / (30 - 10) = 0.035 s/iter over 40 rows. Applied to qnn as well, whose
# own slope measured 0.000 within a 0.05s resolution -- so this is a conservative bound
# for qnn, not a fit.
SLOPE_S_PER_ROW_ITER = 0.035 / PROBE_NTR
SLOPE_CALIBRATED_AT = ("mps", 13)


def kernel_circuits(n_tr, n_te):
    """Fidelity-kernel evaluations: the training Gram triangle plus the test block."""
    return n_tr * (n_tr - 1) / 2 + n_te * n_tr


def circuits(arm, n_tr, n_te):
    return kernel_circuits(n_tr, n_te) if arm == "qsvc" else n_tr + n_te


# ---------------------------------------------------------------------------
# Row dependence of the MPS band
# ---------------------------------------------------------------------------
# Statevector simulation performs the same FLOPs whatever the parameter values, so its
# per-circuit cost CANNOT depend on the data -- and wdbc confirmed that empirically, holding
# 11.44 / 11.46 / 11.28 ms/circuit at n_tr = 40, 120, 200 (within 1% across a 25x span in
# circuit count). MPS is different in kind: its cost tracks the bond dimension the feature
# map drives, and that depends on the actual feature vectors. Measured, one qsvc fit each:
#
#   heart      13q  n_tr= 40  seed 43   73.5593 ms/circuit   <- the original calibration
#   heart      13q  n_tr= 80  seed 43   99.2720 ms/circuit   1.35x
#   heart      13q  n_tr= 80  seed  7   98.5221 ms/circuit   1.34x  reproducible, not noise
#   hepatitis  19q  n_tr= 40  seed 43  100.5932 ms/circuit   <- the original calibration
#   hepatitis  19q  n_tr= 80  seed 43  143.7309 ms/circuit   1.43x  and not a heart quirk
#
# Two seeds at the same row count agree to 0.8%, which is what rules out run-to-run variance
# on a shared host -- the first reading of the 1.35x. A fixed per-fit overhead is ruled out
# too: dividing one by a 4x larger circuit count would push ms/circuit DOWN, not up.
#
# Modelled as a power law in n_tr, since that is what two decades of rows can support and it
# extrapolates monotonically rather than flattening by assumption. Fitted by least squares on
# log(ms/circuit) vs log(n_tr), per (arm, width), and the WORST exponent is used everywhere in
# the band. That is deliberate: a wall estimate that is too high costs nothing -- LSF releases
# a job's slots the moment it exits -- while one that is too low costs a resample to a wall
# kill. The spread between the two datasets' fits is itself the honest error bar.
#
# CAVEAT, because it bounds how far this should be trusted: the probe fits its MinMaxScaler on
# the SUBSAMPLE's training rows (cost_probe_lib.py:58), so the 40- and 80-row points scale
# their features from 40 and 80 observed ranges. Part of the growth is therefore the scaler
# moving rather than the pair set growing. It is not an artefact -- the real qprofiler run fits
# its scaler on the real X_train as well -- but both calibration points are scaled from fewer
# rows than heart's real 216, so this is the right question asked imperfectly. Treat the
# exponent as a conservative bound, not a measurement of a physical law.
MPS_ROW_POINTS = {
    # The 160-row heart point is the one that sets MPS_ROW_ALPHA, and it is worth saying why
    # it is kept. Fitted on the 40/80 pairs alone the exponent is 0.51, which predicts 150
    # ms/circuit at n_tr=160; the measured value is 277.6, so the two-point law understates
    # the cost of the only regime the pilot actually runs in by 1.85x. The 40->160 pair alone
    # implies 0.958 -- near-linear in rows. That measurement shared an 8-core-capped login
    # node with other work, so it is an upper bound rather than a clean number, but this
    # module's rule is to take the worst exponent: an over-estimated wall costs nothing
    # because LSF frees the slot on exit, while an under-estimate costs a resample to a kill.
    ("qsvc", 13): [(40, 0.0735593), (80, 0.0992720), (80, 0.0985221), (160, 0.2776124)],
    ("qsvc", 19): [(40, 0.1005932), (80, 0.1437309)],
}


def _fit_alpha(points):
    """Least-squares slope of log(cost) against log(n_tr); 0.0 if the rows do not vary."""
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(v) for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


MPS_ROW_ALPHA = max(_fit_alpha(pts) for pts in MPS_ROW_POINTS.values())


def row_scale(bk, n_tr):
    """Per-circuit multiplier for `n_tr` rows, relative to the PROBE_NTR calibration.

    Clamped below at PROBE_NTR rather than extrapolated downward: every entry in
    UNIT_S_PER_CIRCUIT *is* the cost at PROBE_NTR, so a dataset training on fewer rows than
    the probe is already described by its table value and must not be discounted further.
    """
    if bk != "mps":
        return 1.0
    return (max(n_tr, PROBE_NTR) / PROBE_NTR) ** MPS_ROW_ALPHA


def _interp(table, arm, bk, q):
    """Log-linear interpolation in qubit count, clamped outside the calibrated range.

    Log-linear because statevector cost is exponential in width while the measured MPS
    points are close to flat, and both are straight lines in log(cost).
    """
    pts = sorted((qq, v) for (a, b, qq), v in table.items() if a == arm and b == bk)
    if not pts:
        pts = sorted((qq, v) for (a, b, qq), v in table.items() if a == arm)
    if not pts:
        raise KeyError(f"no calibration for arm {arm!r}")
    for qq, v in pts:
        if qq == q:
            return v
    if q <= pts[0][0]:
        return pts[0][1]
    if q >= pts[-1][0]:
        return pts[-1][1]
    for (q0, v0), (q1, v1) in zip(pts, pts[1:]):
        if q0 <= q <= q1:
            f = (q - q0) / (q1 - q0)
            return math.exp(math.log(v0) + f * (math.log(v1) - math.log(v0)))
    return pts[-1][1]


def loop_s_per_row_iter(bk, q):
    """Scale the measured optimizer-loop slope to another width.

    The loop runs circuits on the same simulator the kernel arms use, so the per-circuit
    cost ratio is the right transfer function between widths.
    """
    ref = _interp(UNIT_S_PER_CIRCUIT, "qsvc", *SLOPE_CALIBRATED_AT)
    return SLOPE_S_PER_ROW_ITER * _interp(UNIT_S_PER_CIRCUIT, "qsvc", bk, q) / ref


# ---------------------------------------------------------------------------
# The rest of the kernel arms' search grid: encoding and entanglement
# ---------------------------------------------------------------------------
# UNIT_S_PER_CIRCUIT is calibrated at ZZ/linear and REPS_MEAN prices the reps grid, but qsvc
# and pqk also search encoding (Z, ZZ, P) and entanglement (linear, pairwise, and 'full' on
# the statevector only), and on the statevector those do NOT average out to ZZ/linear.
# Measured 2026-09-28 through the real compute_qsvc path (/tmp/ent_probe.py, 50-row
# subsample, one fit each), relative to ZZ/linear at reps 2 on the same width:
#
#    q   Z/linear  ZZ/full  ZZ/full reps 4   grid mean
#    4    0.81      1.22                       0.99
#    6    0.76      1.50                       1.03
#    7    0.73      1.71                       1.07
#    8    0.70      1.91       3.27            1.10
#    9    0.65      1.94                       1.09
#   10    0.69      2.23       3.91            1.17
#   13    0.83      2.04       3.59            1.17
#
# P is PauliFeatureMap with its default paulis, which builds the same circuit as ZZ
# (identical gate counts at 8 and 13 qubits). Pairwise entangles the same n-1 pairs as
# linear in a different order, so it costs the same. Z has no entanglers at all, so for it
# the entanglement choice is moot. Two things follow, and they pull in different directions:
#
#   the mean  a search that samples the grid uniformly (TPE's first 10 trials are random)
#             pays the grid mean per trial -- 10-17% above what fit_seconds prices at 8-13q.
#   the tail  freeze_quantum_params reuses ONE winner for every resample, so a job whose
#             search lands on ZZ/P + full + max reps pays ~2.6-3.1x the priced fit on every
#             full fit. That is what sizes the wall, not the mean: at the pilot's widths it
#             is most of the spread between jobs that finish on time and jobs that do not.
#
# The 13q 'full' ratio is 2.04 on this path, not the 1.40 a bare 20-circuit timing gave: the
# kernel path builds and simulates each fidelity circuit U(x')^dagger U(x) at double depth,
# which the bare timing did not. MPS is not measured; it gets 1.0, which is conservative for
# Z (no entanglers leave the bond dimension at 1) and exact for pairwise vs linear.
# pqk borrows the qsvc ratios: it runs the same feature map, once per row rather than once
# per pair.
ENC_ENT_SV = {  # q -> (Z/linear, ZZ/full), relative to ZZ/linear
    4: (0.81, 1.22), 6: (0.76, 1.50), 7: (0.73, 1.71), 8: (0.70, 1.91),
    9: (0.65, 1.94), 10: (0.69, 2.23), 13: (0.83, 2.04),
}


def _enc_ent(bk, q):
    """(Z, full) cost ratios at width q: linear in q between measured widths, clamped outside."""
    if bk != "sv":
        return 1.0, 1.0
    pts = sorted(ENC_ENT_SV.items())
    if q <= pts[0][0]:
        return pts[0][1]
    if q >= pts[-1][0]:
        return pts[-1][1]
    for (q0, v0), (q1, v1) in zip(pts, pts[1:]):
        if q0 <= q <= q1:
            f = (q - q0) / (q1 - q0)
            return tuple(a + f * (b - a) for a, b in zip(v0, v1))
    return pts[-1][1]


def grid_factor(arm, bk, q, point="mean"):
    """Cost of the searched grid relative to what fit_seconds prices for this arm.

    fit_seconds prices ZZ/linear with reps at REPS_MEAN. point='mean' averages over encoding
    x entanglement (reps already sits at its mean); point='max' is the grid's costliest
    point -- ZZ or P, the widest entanglement the backend searches, the largest reps.
    Variational arms do not search this grid and get 1.0.
    """
    if arm not in KERNEL_ARMS:
        return 1.0
    z, full = _enc_ent(bk, q)
    ents = [1.0, 1.0] + ([full] if bk == "sv" else [])      # linear, pairwise(, full)
    if point == "mean":
        return z / 3 + (2 / 3) * sum(ents) / len(ents)
    return max(ents + [z]) * max(REPS_GRID[arm]) / REPS_MEAN[arm]


def fit_seconds(arm, bk, q, n_tr, n_te):
    """Seconds for ONE fit of this arm at this width and split, at ZZ/linear (see grid_factor)."""
    if arm in KERNEL_ARMS:
        reps = REPS_MEAN[arm] / REPS_PROBE[arm]
        return (circuits(arm, n_tr, n_te) * _interp(UNIT_S_PER_CIRCUIT, arm, bk, q)
                * reps * row_scale(bk, n_tr))
    # The variational arms' fixed term is circuit construction and transpilation, which the
    # row count does not touch, so row_scale is applied only to the per-row loop term. Those
    # arms are a small share of every MPS job's bill anyway (vqc is 0.54 h of heart's 8.91 h),
    # so this distinction cannot move a wall decision; the kernel arms are what bind.
    return (_interp(OVERHEAD_S, arm, bk, q)
            + MAXITER_MEAN * loop_s_per_row_iter(bk, q) * n_tr * row_scale(bk, n_tr))


def split_rows(rows, test_size=TEST_SIZE):
    n_tr = int(round(rows * (1 - test_size)))
    return n_tr, rows - n_tr


def arm_hours(arm, bk, q, rows, ntq, n_iter, test_size=TEST_SIZE, grid=None):
    """Hours this arm contributes: ntq discounted search trials + n_iter full fits.

    grid=None prices every fit at ZZ/linear, which is what the trial budget is checked
    against (choose_trials) unless the caller asks otherwise. grid='mean' prices the search
    grid at its mean everywhere -- the expected cost. grid='bound' prices the search at the
    mean and every full fit at the grid's costliest point: the job whose frozen winner is
    the most expensive configuration it could have picked.

    freeze_quantum_params makes the cost (ntq + n_iter) fits rather than ntq x n_iter:
    embedding 1 runs the search then refits, and every later resample reuses the frozen
    winner. So at n_iter=5 the search is most of the bill, which is why the trial budget
    is the deadline knob and n_iter is not.

    A search trial is cheaper than a full fit: search_hyperparameters carves an inner
    validation_split out of X_train, so a trial trains on 0.75*n_tr and scores on the
    remaining 0.25*n_tr, while the refit and every frozen resample see the real n_tr/n_te.
    For an arm whose cost is quadratic in rows that is a ~0.63 discount; ignoring it
    overstates the search term by ~1.6x.
    """
    n_tr, n_te = split_rows(rows, test_size)
    n_inner_tr = int(round(n_tr * (1 - VALIDATION_SPLIT)))
    n_inner_val = n_tr - n_inner_tr
    full = fit_seconds(arm, bk, q, n_tr, n_te)
    trial = fit_seconds(arm, bk, q, n_inner_tr, n_inner_val)
    search = fits = 1.0
    if grid is not None:
        search = grid_factor(arm, bk, q, "mean")
        fits = grid_factor(arm, bk, q, "mean" if grid == "mean" else "max")
    return (ntq * trial * search + n_iter * full * fits) / 3600.0


def wall_hours(bk, q, n_emb, rows, ntq, n_iter, test_size=TEST_SIZE, grid=None):
    """Hours for one dataset's LSF job (grid: see arm_hours).

    max over arms, not sum: model_run fans the 13 models out over joblib, so the arms run
    concurrently and the slowest sets the clock. x n_emb because qprofiler loops
    embeddings sequentially and the freeze cache is keyed per embedding, so each embedding
    pays for its own search.
    """
    return n_emb * max(arm_hours(a, bk, q, rows, ntq, n_iter, test_size, grid) for a in ARMS)


def slowest_arm(bk, q, rows, ntq, n_iter, test_size=TEST_SIZE):
    return max(ARMS, key=lambda a: arm_hours(a, bk, q, rows, ntq, n_iter, test_size))


TRIAL_LADDER = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32)


def choose_trials(bk, q, n_emb, rows, n_iter, budget_hours,
                  ladder=TRIAL_LADDER, test_size=TEST_SIZE, grid=None):
    """Largest trial budget on the ladder whose predicted wall fits budget_hours.

    Per dataset rather than global because the jobs are independent: a 62-row dataset
    finishing in 20 minutes gains nothing from being starved to fit the same budget as a
    569-row one. Returns the ladder minimum if even that overruns -- the caller is told,
    rather than silently handed a budget that cannot be met. grid=None (the default, and
    what sized the shipped pilot) checks the ZZ/linear price; grid='mean' checks the
    expected cost of the whole search grid instead.
    """
    best = ladder[0]
    for ntq in ladder:
        if wall_hours(bk, q, n_emb, rows, ntq, n_iter, test_size, grid) <= budget_hours:
            best = ntq
        else:
            break
    return best


# ---------------------------------------------------------------------------
# split_mode: manifest jobs (generate_pilot_configs.py --split-mode manifest --wall auto).
#
# A manifest job runs one model group on one or more outer splits of one (dataset,
# embedding). Per split: n_trials fits on the fit rows scored on the validation rows, then
# one refit on the training rows -- no freeze, so the search is paid on every split. Rows
# split as k-fold: n_test = n_val = rows/k, n_fit = rows - 2 rows/k.
#
# Calibrated against controlled run ctrl1 (2026-10-05, 300 LSF jobs on AMD EPYC 7763,
# 30 trials, k=5; analysis/job_runtimes.csv under the run): labor 57 rows x 16 qubits mps,
# colon_cancer 62 rows and spect 267 rows at 8 qubits sv. Against those runs the circuit
# model above holds for pqk and mps qsvc (measured / predicted 0.9-1.1 once the per-job
# startup is added) but not for:
#   sv qsvc   priced as n(n-1)/2 circuits; StatevectorFidelityKernel computes one
#             statevector per row, so the real cost is ~linear (0.22x at 62 rows,
#             0.014x at 267 rows).
#   qnn       4.6-11.6x underpriced; per fit it is ~linear in rows.
#   classical not modelled above; all nine arms in one job.
# Those three get the measured linear laws below. Rows beyond 267 and widths other than
# 8 sv / 16 mps are extrapolated: the generator flags such jobs, and their walls carry the
# same safety factor. Re-calibrate with calibrate_manifest() on a run that covers them.
# ---------------------------------------------------------------------------
JOB_OVERHEAD_S = 100.0             # python start, imports, data load, sidecar writes
#: Per-fit seconds = a + b * rows for the arms the circuit model misprices, at 8 sv qubits.
MANIFEST_LINEAR_S = {"qsvc": (0.65, 0.0036), "qnn": (2.4, 0.49)}   # qsvc: per row fitted+scored; qnn: per fit row
#: qnn per fit on mps relative to the 8-qubit sv law (labor 16q mps: 39.8 s vs 19.6 s).
QNN_MPS_FACTOR = 2.0
#: Classical group, all nine arms, per split: a + b * rows (the startup is extra).
CLASSICAL_SPLIT_S = (45.0, 0.36)
#: Measured / predicted per (arm, bk) after the laws above, from ctrl1; 1.0 = trusted.
MANIFEST_CALIBRATION = {("pqk", "sv"): 1.0, ("pqk", "mps"): 1.07, ("qsvc", "mps"): 1.07,
                        ("qsvc", "sv"): 1.0, ("qnn", "sv"): 1.0, ("qnn", "mps"): 1.0,
                        ("vqc", "sv"): 1.0, ("vqc", "mps"): 1.0}
#: The (arm-or-group, bk, qubits) points and the largest row count the laws were fitted on.
MANIFEST_CALIBRATED = {"widths": {("sv", 8), ("mps", 16)}, "max_rows": 267}
AUTO_WALL_SAFETY = 3.0             # wall = safety x expected + margin
AUTO_WALL_MARGIN_H = 0.25
AUTO_WALL_MIN_H = 0.5
AUTO_WALL_MAX_H = 72.0             # the queue's ABS_RUNLIMIT ceiling


def manifest_rows(rows, k=5):
    """(n_fit, n_val, n_train, n_test) of one outer split of a ``rows``-row dataset."""
    n_te = max(1, int(round(rows / k)))
    n_tr = rows - n_te
    return n_tr - n_te, n_te, n_tr, n_te


def manifest_split_seconds(group, bk, q, rows, k=5, n_trials=30):
    """Expected seconds for ONE outer split of one model group in a manifest job.

    ``group`` is a quantum arm ('qsvc', 'pqk', 'qnn', 'vqc') or 'classical'. Excludes the
    per-job startup (JOB_OVERHEAD_S).
    """
    n_fit, n_val, n_tr, n_te = manifest_rows(rows, k)
    if group == "classical":
        a, b = CLASSICAL_SPLIT_S
        return a + b * rows
    if group == "qsvc" and bk == "sv":
        a, b = MANIFEST_LINEAR_S["qsvc"]
        # One statevector per row; the per-row cost grows with the 2^q amplitudes.
        width = 2.0 ** (q - 8) * q / 8.0
        per_fit = lambda n: a + b * n * width  # noqa: E731
        secs = n_trials * per_fit(n_fit + n_val) + per_fit(n_tr + n_te)
    elif group == "qnn":
        a, b = MANIFEST_LINEAR_S["qnn"]
        scale = QNN_MPS_FACTOR if bk == "mps" else 2.0 ** max(0, q - 8) * q / 8.0
        secs = scale * (n_trials * (a + b * n_fit) + (a + b * n_tr))
    else:
        g = grid_factor(group, bk, q, "mean")
        secs = (n_trials * fit_seconds(group, bk, q, n_fit, n_val)
                + fit_seconds(group, bk, q, n_tr, n_te)) * g
    return secs * MANIFEST_CALIBRATION.get((group, bk), 1.0)


def manifest_job_hours(group, bk, q, rows, k=5, n_trials=30, n_splits=1):
    """Expected wall-clock hours of one manifest job: startup + n_splits splits."""
    return (JOB_OVERHEAD_S + n_splits * manifest_split_seconds(group, bk, q, rows, k, n_trials)) / 3600.0


def manifest_extrapolated(bk, q, rows):
    """True when (bk, q, rows) lies outside what the manifest laws were fitted on."""
    return (bk, q) not in MANIFEST_CALIBRATED["widths"] or rows > MANIFEST_CALIBRATED["max_rows"]


def auto_wall(hours):
    """LSF -W for an expected run of ``hours``: ``(H:MM, capped)``.

    safety x expected + margin, rounded up to 15 minutes, at least AUTO_WALL_MIN_H and at
    most AUTO_WALL_MAX_H; ``capped`` says the ceiling cut it, i.e. the job is expected to
    need more than the queue allows and should be split (fewer splits per job) or dropped.
    """
    want = AUTO_WALL_SAFETY * hours + AUTO_WALL_MARGIN_H
    capped = want > AUTO_WALL_MAX_H
    h = min(max(want, AUTO_WALL_MIN_H), AUTO_WALL_MAX_H)
    quarters = math.ceil(h * 4 - 1e-9)
    return f"{quarters // 4}:{(quarters % 4) * 15:02d}", capped


def calibrate_manifest(manifest_tsv, runtimes):
    """Measured / expected per (group, bk) for a finished manifest run.

    Args:
        manifest_tsv: the run's MANIFEST.tsv (pandas DataFrame).
        runtimes: {config: seconds} from the LSF logs ("Run time").

    Returns:
        A DataFrame per (group, bk, qubits, rows): jobs, median measured and expected
        seconds and their ratio, to update MANIFEST_CALIBRATION and the laws above.
    """
    import pandas as pd

    rows = []
    for _, r in manifest_tsv.iterrows():
        if r["config"] not in runtimes:
            continue
        bk = "mps" if r["backend"] == "mps_simulator" else "sv"
        n_splits = len(str(r["iteration"]).split(";"))
        exp = manifest_job_hours(r["group"], bk, int(r["qubits"]), int(r["rows"]),
                                 n_trials=int(r["n_trials"]), n_splits=n_splits) * 3600
        rows.append(dict(group=r["group"], bk=bk, qubits=int(r["qubits"]), rows=int(r["rows"]),
                         measured_s=runtimes[r["config"]], expected_s=exp))
    df = pd.DataFrame(rows)
    out = df.groupby(["group", "bk", "qubits", "rows"]).agg(
        jobs=("measured_s", "size"), measured_s=("measured_s", "median"),
        expected_s=("expected_s", "median")).reset_index()
    out["ratio"] = out["measured_s"] / out["expected_s"]
    return out
