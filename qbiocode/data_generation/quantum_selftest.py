"""
Numerical self-checks for the quantum data generators.

The generators in this package compute expectation values with bit-trick Pauli
algebra, project onto a symmetry sector, and build labels from a kernel identity.
None of that is verifiable by eye, and a mistake in any of it produces a dataset
that is *plausible* rather than wrong-looking -- a benchmark quietly measuring
something other than what it claims. So the physics is pinned by checks that
compare each shortcut against a slower construction that is obviously correct:

=============================== ========================================================
:func:`check_pauli_action`      the bit-trick Pauli action against dense Kronecker
                                products, over all :math:`4^4` Pauli strings
:func:`check_sparse_pauli_sum`  the sparse Hamiltonian against a dense sum
:func:`check_ground_state`      the even-sector ground state's eigen-residual, its
                                :math:`\\prod X` parity, and that it really is the
                                global minimum
:func:`check_walsh_transform`   the fast Walsh-Hadamard transform's round trip and
                                Parseval identity
:func:`check_short_time_limit`  the quench expansion
                                :math:`\\langle Z_i(t)\\rangle = 1 - 2h_i^2t^2 + O(t^4)`
:func:`check_engineered_labels` that engineered labels hit :math:`s_Q = 1` and
                                :math:`s_C = g^2`
:func:`check_zz_feature_map`    the native ``ZZFeatureMap`` against Qiskit's
=============================== ========================================================

Each function returns the quantities it measured and raises :class:`AssertionError`
naming the residual when one is out of tolerance, so a failure says *what* drifted.
:func:`run_selftest` runs them all; the test suite calls them individually, and the
``qdata-gen selftest`` command calls :func:`run_selftest`, so the same checks guard
the repository and a user's own install without being written twice.

Two of these are not just internal consistency. :func:`check_short_time_limit`
pins the property that makes ``hl`` a *classically easy* control -- the expansion
hands a learner :math:`h_i` almost directly -- and
:func:`check_engineered_labels` pins the identity that makes ``eng`` a positive
control. Without them, those two roles would be assumptions rather than facts.
"""

import numpy as np

from .quantum_core import (
    Pauli,
    dense_pauli,
    engineered_labels,
    even_sector_ground_state,
    expvals,
    fwht,
    ising_terms,
    pauli_ops,
    pauli_sum,
    rbf,
    zz_feature_state,
)

#: Seed used by every check, so a reported residual is reproducible.
SELFTEST_SEED = 1


def _require_assertions(check):
    """Refuse to run ``check`` when ``assert`` statements have been compiled out.

    Every check below signals failure with a bare ``assert``, which reads well and lets
    pytest introspect the comparison. ``python -O`` (and ``PYTHONOPTIMIZE``) removes
    ``assert`` statements outright, and this module is library code, so -- unlike the
    asserts inside a test file, which pytest rewrites into real raises -- there is nothing
    left to fail. Each check would then return its measured quantities no matter how wrong
    they were, and its caller would conclude the physics is right.

    :func:`run_selftest` has refused under -O since it was written, but the test suite
    calls the ``check_*`` functions *directly* and so bypassed that guard: measured on
    this tree, ``python -O -m pytest -k TestThePhysicsIsRight`` reported 7 passed and only
    the one test that goes through ``run_selftest`` failed. The guard therefore belongs on
    each check rather than on the runner.

    Args:
        check (str): the check's name, for the message.

    Raises:
        RuntimeError: if the interpreter is running with ``-O`` / ``PYTHONOPTIMIZE``.
    """
    if not __debug__:
        raise RuntimeError(
            f"{check} cannot run under python -O: it asserts its results, and -O strips "
            f"assert statements, so it would pass vacuously having verified nothing. "
            f"Re-run without -O (or unset PYTHONOPTIMIZE)."
        )

#: Qubit count, feature-map repetitions and entanglement pattern of each
#: configuration the Qiskit cross-check compares, covering one and several
#: repetitions and all three entanglement patterns.
ZZ_CROSSCHECK_CASES = (
    (3, 1, "linear"),
    (5, 2, "linear"),
    (4, 2, "full"),
    (11, 2, "linear"),
    (5, 4, "pairwise"),
    (7, 4, "pairwise"),
)


def check_pauli_action(n=4, tol=1e-12):
    """Compare the bit-trick Pauli action with dense Kronecker products.

    Every one of the :math:`4^n` Pauli strings on ``n`` qubits is applied to the same
    random complex state both ways. This is the check that matters most: every
    feature in every family is an expectation value computed by the fast path.

    Parameters
    ----------
    n : int, default=4
        Qubit count. The cost is :math:`4^n` dense :math:`2^n \\times 2^n` products,
        so 4 is already 256 comparisons.
    tol : float, default=1e-12
        Tolerance on the largest disagreement.

    Returns
    -------
    dict
        ``max_error`` and the number of strings compared.

    Raises
    ------
    AssertionError
        If any string disagrees by more than ``tol``.
    """
    _require_assertions("check_pauli_action")
    rng = np.random.default_rng(SELFTEST_SEED)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    err = 0.0
    for code in range(4 ** n):
        ops = {q: "IXYZ"[(code >> (2 * q)) & 3] for q in range(n)}
        ops = {q: p for q, p in ops.items() if p != "I"}
        fast = expvals(psi, [Pauli(n, ops)])[0, 0]
        dense = np.real(psi.conj() @ dense_pauli(n, ops) @ psi)
        err = max(err, abs(fast - dense))
    assert err < tol, f"Pauli action disagrees with dense kron by {err:.3e} > {tol:.0e}"
    return {"max_error": err, "n_strings": 4 ** n}


def check_sparse_pauli_sum(n=4, tol=1e-12):
    """Compare the sparse Hamiltonian assembly with a dense term-by-term sum.

    Parameters
    ----------
    n : int, default=4
        Chain length.
    tol : float, default=1e-12
        Tolerance on the largest entrywise disagreement.

    Returns
    -------
    dict
        ``max_error``.

    Raises
    ------
    AssertionError
        If the two matrices disagree by more than ``tol``.
    """
    _require_assertions("check_sparse_pauli_sum")
    rng = np.random.default_rng(SELFTEST_SEED)
    J, h = rng.uniform(0.5, 1.5, n - 1), rng.uniform(0, 2, n)
    terms = ising_terms(n, J, h, kappa=0.3)
    dense = sum(c * dense_pauli(n, {q: s for q, s in zip(*pauli_ops(P))}) for c, P in terms)
    err = float(np.abs(pauli_sum(n, terms).toarray() - dense).max())
    assert err < tol, f"sparse and dense Hamiltonians disagree by {err:.3e} > {tol:.0e}"
    return {"max_error": err}


def check_ground_state(n=4, residual_tol=1e-8, parity_tol=1e-10, energy_tol=1e-9):
    """Verify the even-sector ground state three ways.

    The generator finds the ground state *within* the :math:`\\prod X = +1` sector by
    penalising the odd one, which is faster than projecting but only correct if the
    result is still an eigenvector, still in the right sector, and still the global
    minimum. All three are checked, the last against a full dense diagonalisation.

    Parameters
    ----------
    n : int, default=4
        Chain length.
    residual_tol : float, default=1e-8
        Tolerance on :math:`\\|H\\psi - E\\psi\\|`.
    parity_tol : float, default=1e-10
        Tolerance on :math:`\\langle \\prod X \\rangle - 1`.
    energy_tol : float, default=1e-9
        Tolerance on the gap to the true spectral minimum.

    Returns
    -------
    dict
        ``residual``, ``parity``, ``energy_above_minimum`` and the sector ``gap``.

    Raises
    ------
    AssertionError
        If the state is not an eigenvector, is not in the even sector, or is not the
        global minimum.
    """
    _require_assertions("check_ground_state")
    rng = np.random.default_rng(SELFTEST_SEED)
    J, h = rng.uniform(0.5, 1.5, n - 1), rng.uniform(0, 2, n)
    terms = ising_terms(n, J, h, kappa=0.3)
    dense = sum(c * dense_pauli(n, {q: s for q, s in zip(*pauli_ops(P))}) for c, P in terms)
    psi, gap, residual = even_sector_ground_state(n, terms)
    parity = expvals(psi, [Pauli(n, {i: "X" for i in range(n)})])[0, 0]
    energy = float(np.real(psi.conj() @ dense @ psi))
    above = energy - float(np.linalg.eigvalsh(dense)[0])
    assert residual < residual_tol, f"ground state is not an eigenvector: residual {residual:.3e}"
    assert abs(parity - 1) < parity_tol, f"ground state is not in the even sector: <prod X> = {parity:+.12f}"
    assert abs(above) < energy_tol, f"even-sector state is not the global minimum: E0 - min(spec) = {above:.3e}"
    return {"residual": residual, "parity": parity, "energy_above_minimum": above, "gap": gap}


def check_walsh_transform(n=6, tol=1e-10):
    """Verify the fast Walsh-Hadamard transform's round trip and Parseval identity.

    The ``te`` family reports an effective degree read off this transform, which is
    meaningless if the transform or its normalisation is wrong.

    Parameters
    ----------
    n : int, default=6
        Number of bits, i.e. a vector of length :math:`2^n`.
    tol : float, default=1e-10
        Tolerance on both residuals.

    Returns
    -------
    dict
        ``round_trip_error`` and ``parseval_error``.

    Raises
    ------
    AssertionError
        If either residual exceeds ``tol``.
    """
    _require_assertions("check_walsh_transform")
    rng = np.random.default_rng(SELFTEST_SEED)
    F = rng.normal(size=1 << n)
    coeffs = fwht(F) / (1 << n)
    round_trip = float(np.abs(fwht(coeffs) - F).max())
    parseval = float(abs(np.sum(coeffs ** 2) - np.mean(F ** 2)))
    assert round_trip < tol, f"FWHT round trip is off by {round_trip:.3e} > {tol:.0e}"
    assert parseval < tol, f"FWHT violates Parseval by {parseval:.3e} > {tol:.0e}"
    return {"round_trip_error": round_trip, "parseval_error": parseval}


def check_short_time_limit(n=4, t=0.02, tol=1e-5):
    """Verify the short-time quench expansion that makes ``hl`` classically easy.

    From :math:`|0\\ldots0\\rangle`, :math:`\\langle Z_i(t)\\rangle = 1 - 2h_i^2t^2 +
    O(t^4)`. The residual should therefore scale as :math:`t^4` -- about 1e-6 at the
    default ``t`` -- and this being true is *why* classical models are expected to do
    well on the ``hl`` family, which is its role as a control.

    Parameters
    ----------
    n : int, default=4
        Chain length.
    t : float, default=0.02
        Evolution time. The tolerance assumes it is small.
    tol : float, default=1e-5
        Tolerance on the largest deviation from the quadratic expansion.

    Returns
    -------
    dict
        ``max_error`` and the ``t`` used.

    Raises
    ------
    AssertionError
        If any site deviates from the expansion by more than ``tol``.
    """
    _require_assertions("check_short_time_limit")
    rng = np.random.default_rng(SELFTEST_SEED)
    J, h = rng.uniform(0.5, 1.5, n - 1), rng.uniform(0, 2, n)
    H = pauli_sum(n, ising_terms(n, J, h, g=np.full(n, 0.5), sign=1.0)).toarray()
    E, V = np.linalg.eigh(H)
    psi0 = np.zeros(1 << n, complex)
    psi0[0] = 1
    z = expvals(V @ (np.exp(-1j * E * t) * (V.conj().T @ psi0)),
                [Pauli(n, {i: "Z"}) for i in range(n)])[0]
    err = float(np.abs(z - (1 - 2 * h ** 2 * t ** 2)).max())
    assert err < tol, (
        f"short-time expansion is off by {err:.3e} > {tol:.0e} at t={t}; "
        f"expected O(t^4) ~ {t ** 4:.0e}"
    )
    return {"max_error": err, "t": t}


def check_engineered_labels(n_rows=60, n_features=3, sq_tol=1e-6, sc_tol=1e-8):
    """Verify that engineered labels saturate the geometric-difference bound.

    For the continuous target the construction should give exactly :math:`s_Q = 1`
    and :math:`s_C = g^2`. This is the identity the ``eng`` family's role as a
    positive control rests on, so it is asserted rather than assumed.

    Parameters
    ----------
    n_rows : int, default=60
        Kernel size.
    n_features : int, default=3
        Input dimension.
    sq_tol : float, default=1e-6
        Tolerance on :math:`s_Q - 1`. Looser than ``sc_tol`` because it goes through
        a pseudo-inverse of a kernel that is numerically rank-deficient.
    sc_tol : float, default=1e-8
        Tolerance on :math:`s_C / g^2 - 1`.

    Returns
    -------
    dict
        ``s_q``, ``s_c_over_g2`` and ``g``.

    Raises
    ------
    AssertionError
        If either complexity misses its target.
    """
    _require_assertions("check_engineered_labels")
    rng = np.random.default_rng(SELFTEST_SEED)
    X = rng.uniform(0, 1, (n_rows, n_features))
    K_Q, K_C = rbf(X, 5.0), rbf(X, 0.5)
    y, g, K_Cn, K_Qn = engineered_labels(K_C, K_Q, 1e-3)
    s_q = float(y @ np.linalg.pinv(K_Qn, rcond=1e-12, hermitian=True) @ y)
    s_c = float(y @ np.linalg.solve(K_Cn + 1e-3 * np.eye(n_rows), y))
    ratio = s_c / g ** 2
    assert abs(s_q - 1) < sq_tol, f"engineered labels give s_Q = {s_q:.6f}, expected 1"
    assert abs(ratio - 1) < sc_tol, f"engineered labels give s_C/g^2 = {ratio:.6f}, expected 1"
    return {"s_q": s_q, "s_c_over_g2": ratio, "g": g}


def check_zz_feature_map(tol=1e-10, cases=ZZ_CROSSCHECK_CASES, draws=3):
    """Compare the native ``ZZFeatureMap`` statevector with Qiskit's.

    The native implementation exists so the generators do not build a circuit per
    sample, and the ``ql``/``eng`` labels depend on it being the *same* feature map a
    learner will use through Qiskit. That equivalence is checked here across
    repetitions and entanglement patterns rather than trusted.

    Both data maps are covered, because QProfiler's two quantum models disagree:
    ``qsvc`` uses Qiskit's default and ``pqk`` uses
    :func:`~qbiocode.utils.qutils.unit_coefficient_data_map`. A family generated
    against one is anti-aligned with the other, so ``data_map='unit'`` must track
    Qiskit's ``data_map_func`` just as tightly as the default does.

    Parameters
    ----------
    tol : float, default=1e-10
        Tolerance on ``1 - |overlap|``.
    cases : sequence of tuple, default=:data:`ZZ_CROSSCHECK_CASES`
        ``(n_qubits, reps, entanglement)`` configurations to compare.
    draws : int, default=3
        Random inputs per configuration.

    Returns
    -------
    dict
        ``max_infidelity`` and the number of comparisons.

    Raises
    ------
    AssertionError
        If any configuration disagrees by more than ``tol``.

    Notes
    -----
    Qiskit's ``zz_feature_map`` function is used when available; the deprecated
    ``ZZFeatureMap`` class is the fallback for Qiskit older than 2.1.
    """
    _require_assertions("check_zz_feature_map")
    try:                                    # the class is deprecated as of Qiskit 2.1
        from qiskit.circuit.library import zz_feature_map as _zz
    except ImportError:                     # pragma: no cover - older Qiskit
        from qiskit.circuit.library import ZZFeatureMap as _zz
    from qiskit.quantum_info import Statevector

    from qbiocode.utils.qutils import unit_coefficient_data_map

    rng = np.random.default_rng(SELFTEST_SEED)
    worst, count = {"qiskit": 0.0, "unit": 0.0}, 0
    for n_qubits, reps, entanglement in cases:
        for _ in range(draws):
            x = rng.uniform(0, 1, n_qubits)
            for data_map, data_map_func in (("qiskit", None), ("unit", unit_coefficient_data_map)):
                kwargs = {} if data_map_func is None else {"data_map_func": data_map_func}
                reference = Statevector(
                    _zz(n_qubits, reps=reps, entanglement=entanglement,
                        **kwargs).assign_parameters(x)
                ).data
                infidelity = 1 - abs(np.vdot(
                    reference, zz_feature_state(x, reps, entanglement, data_map)
                ))
                worst[data_map] = max(worst[data_map], infidelity)
            count += 1
    for data_map, value in worst.items():
        assert value < tol, (
            f"native ZZFeatureMap with data_map={data_map!r} differs from Qiskit by "
            f"{value:.3e} > {tol:.0e}"
        )
    return {"max_infidelity": max(worst.values()),
            "max_infidelity_qiskit_map": worst["qiskit"],
            "max_infidelity_unit_map": worst["unit"],
            "n_comparisons": 2 * count}


#: Every check, in the order :func:`run_selftest` runs them.
CHECKS = (
    ("Pauli action vs dense kron", check_pauli_action),
    ("sparse vs dense Hamiltonian", check_sparse_pauli_sum),
    ("even-sector ground state", check_ground_state),
    ("Walsh-Hadamard transform", check_walsh_transform),
    ("short-time quench expansion", check_short_time_limit),
    ("engineered-label bound", check_engineered_labels),
    ("native vs Qiskit ZZFeatureMap", check_zz_feature_map),
)


def run_selftest(verbose=True):
    """Run every check and report.

    Parameters
    ----------
    verbose : bool, default=True
        Print one line per check, with the quantities it measured, and a final
        ``SELFTEST PASS``/``SELFTEST FAIL``.

    Returns
    -------
    tuple of (bool, dict)
        Whether everything passed, and a mapping from check name to either the
        measured quantities or the :class:`AssertionError` message.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_selftest import run_selftest
    >>> ok, results = run_selftest()                               # doctest: +SKIP
    >>> ok                                                         # doctest: +SKIP
    True

    Raises
    ------
    RuntimeError
        If the interpreter is running with ``-O`` / ``PYTHONOPTIMIZE``, which strips
        the ``assert`` statements every check is built on.
    """
    # Every check signals failure with a bare ``assert``, which reads well and lets
    # pytest introspect it -- but ``python -O`` removes assert statements outright, so
    # under -O each check would return its measured quantities no matter how wrong they
    # were and this function would print "SELFTEST PASS" having verified nothing. A
    # self-test that cannot fail is worse than no self-test, so refuse instead.
    if not __debug__:
        raise RuntimeError(
            "run_selftest() cannot run under python -O: the checks assert their "
            "results, and -O strips assert statements, so every check would pass "
            "vacuously. Re-run without -O (or unset PYTHONOPTIMIZE)."
        )
    ok, results = True, {}
    for label, check in CHECKS:
        try:
            measured = check()
        except AssertionError as failure:
            ok = False
            results[label] = str(failure)
            if verbose:
                print(f"  FAIL {label}: {failure}")
        else:
            results[label] = measured
            if verbose:
                summary = ", ".join(
                    f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in measured.items()
                )
                print(f"  ok   {label}: {summary}")
    if verbose:
        print("SELFTEST", "PASS" if ok else "FAIL")
    return ok, results
