"""
Exact statevector primitives shared by the quantum dataset generators.

This module holds the physics and I/O machinery behind the five quantum dataset
families in :mod:`qbiocode.data_generation` (ground states, time evolution,
Hamiltonian learning, quantum labels and engineered kernels). Everything is exact
statevector simulation in NumPy/SciPy, with no quantum backend involved, and is
intended for ``n <= 12`` qubits: the Hilbert space is built densely in several
places, so cost and memory grow as ``2 ** n``.

Conventions
-----------
Little-endian, as in Qiskit: qubit ``q`` corresponds to bit ``q`` of the
computational basis index, so ``dense_pauli`` builds its Kronecker product with
qubit ``n - 1`` as the leftmost factor. Pauli labels in dataset metadata read
``X3Z4``, meaning X on qubit 3 and Z on qubit 4. Ground states in the ``gs``
family are taken in the :math:`\\prod_i X_i = +1` sector, which removes the
finite-size :math:`Z_2` near-degeneracy that otherwise makes the ground state
ill-defined at small ``n``.

Datasets are written in the layout QProfiler can consume::

    <save_path>/x_view/<name>.csv     features = classical description x
    <save_path>/phi_view/<name>.csv   features = local Pauli expectation values
    <save_path>/meta/<name>.json      parameters, label rule, threshold, diagnostics
    <save_path>/meta/<name>_F.npy     continuous pre-threshold quantity F

The two feature views live in separate directories because QProfiler reads every
``*.csv`` in ``folder_path`` non-recursively and would otherwise treat both views
of the same dataset as two unrelated datasets. Sidecars go to ``meta/`` for the
same reason. Point ``folder_path`` at a *view* directory, never at ``save_path``.

Notes
-----
BLAS thread oversubscription dominates the runtime of these generators. Each
dataset row needs one eigendecomposition of a ``2 ** n`` matrix, and for the
matrix sizes involved (256 x 256 at ``n = 8``) a threaded BLAS spends far more
time synchronising than computing: measured on a 128-core machine, 60
decompositions of a 256 x 256 matrix took 9.01 s with default threading and
0.39 s pinned to a single thread. :func:`blas_limit` therefore pins the small-
matrix loops to one thread and leaves larger problems alone. Single-threaded BLAS
is also the more reproducible choice, since reduction order no longer depends on
how work was partitioned across threads.
"""

import contextlib
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

#: Hilbert-space dimension up to which :func:`blas_limit` pins BLAS to one thread.
#: Above this the matrices are large enough for threading to pay for itself.
SMALL_HILBERT_DIM = 1024


@contextlib.contextmanager
def blas_limit(n_qubits, blas_threads=1):
    """Pin BLAS to ``blas_threads`` while the Hilbert space is small.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, so the Hilbert-space dimension is ``2 ** n_qubits``.
    blas_threads : int or None, default=1
        Thread limit to apply. ``None`` disables limiting entirely, which is the
        setting to use when reproducing output generated without it.

    Yields
    ------
    None
        The body runs with the limit applied; the previous limits are restored on
        exit.

    Notes
    -----
    The limit is applied only when ``2 ** n_qubits <= SMALL_HILBERT_DIM``. Thread
    count can change the summation order inside BLAS kernels, so results may
    differ in the last bits between limited and unlimited runs; pass
    ``blas_threads=None`` when bit-for-bit agreement with an unpinned run matters.
    """
    if blas_threads is None or (1 << n_qubits) > SMALL_HILBERT_DIM:
        yield
    else:
        with threadpool_limits(limits=blas_threads):
            yield


# ----------------------------------------------------------------------------- Pauli algebra


def _popcount(a):
    """Number of set bits in each element of ``a``."""
    a = np.asarray(a, dtype=np.uint64).copy()
    c = np.zeros(a.shape, dtype=np.int64)
    one = np.uint64(1)
    while np.any(a):
        c += (a & one).astype(np.int64)
        a >>= one
    return c


class Pauli:
    """Pauli string ``{qubit: 'X'|'Y'|'Z'}`` with ``P|i> = phase[i] |target[i]>``.

    A Pauli string is stored as two bitmasks and a count of Y factors rather than
    as a matrix, so its action on a statevector is a permutation of amplitudes
    times a phase. That keeps expectation values cheap at the ``2 ** n`` sizes
    these generators use, where building the dense operator for every observable
    would dominate the cost.

    Parameters
    ----------
    n : int
        Number of qubits.
    ops : dict
        Mapping from qubit index to ``'X'``, ``'Y'`` or ``'Z'``. Qubits absent
        from the mapping carry the identity.

    Attributes
    ----------
    mx, mz : int
        Bitmasks of the qubits carrying an X-type and a Z-type factor
        respectively; a Y factor sets both.
    ny : int
        Number of Y factors, which fixes the overall power of ``i``.
    label : str
        Canonical label such as ``'X3Z4'``, used in dataset metadata and as the
        column name of the corresponding feature in ``phi_view``.

    Raises
    ------
    ValueError
        If a qubit index is out of range, or an operator is not X, Y or Z.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_core import Pauli
    >>> Pauli(5, {3: 'X', 4: 'Z'}).label
    'X3Z4'
    """

    def __init__(self, n, ops):
        self.n, mx, mz, ny = n, 0, 0, 0
        for q, p in ops.items():
            if not 0 <= q < n:
                raise ValueError(f"qubit {q} out of range for n={n}")
            b = 1 << q
            if p == "X":
                mx |= b
            elif p == "Z":
                mz |= b
            elif p == "Y":
                mx, mz, ny = mx | b, mz | b, ny + 1
            else:
                raise ValueError(f"bad Pauli {p!r}")
        self.mx, self.mz, self.ny = mx, mz, ny
        self.label = "".join(f"{p}{q}" for q, p in sorted(ops.items()))
        self._act = None

    def action(self):
        """Return ``(target, phase)`` such that ``P|i> = phase[i] |target[i]>``.

        Returns
        -------
        target : ndarray of shape (2 ** n,) of int
            Basis index each basis state is mapped to.
        phase : ndarray of shape (2 ** n,) of complex
            Amplitude picked up by each basis state.

        Notes
        -----
        The result is computed once and cached on the instance, so a Pauli reused
        across many rows of a dataset pays for this only on first use.
        """
        if self._act is None:
            idx = np.arange(1 << self.n, dtype=np.uint64)
            parity = _popcount(idx & np.uint64(self.mz)) & 1
            phase = (1j) ** self.ny * (1 - 2 * parity)
            self._act = ((idx ^ np.uint64(self.mx)).astype(np.int64), phase)
        return self._act


def expvals(states, paulis):
    """Expectation values of ``paulis`` in each of ``states``.

    Parameters
    ----------
    states : ndarray of shape (D,) or (D, M) of complex
        One statevector, or ``M`` statevectors as columns.
    paulis : sequence of Pauli
        Observables to evaluate.

    Returns
    -------
    ndarray of shape (M, len(paulis)) of float
        Real expectation values, row per state and column per observable.
    """
    S = states.reshape(states.shape[0], -1)
    out = np.empty((S.shape[1], len(paulis)))
    for k, P in enumerate(paulis):
        tgt, ph = P.action()
        # `tgt` is a permutation of range(D): a Pauli string sends each basis state to
        # exactly one other, so every row of PS is written before the einsum reads it.
        # An operator with a non-bijective target would leave rows at their initial
        # value and silently contribute zeros to the sum. zeros_like rather than
        # empty_like keeps that failure mode benign, and measurably preserves the
        # shipped corpus: the calloc'd buffer's alignment is what the generated
        # meta/*_F.npy diagnostics were computed against, and swapping in empty_like
        # shifted them by ~1e-14 for no gain beyond a skipped allocation.
        PS = np.zeros_like(S, dtype=complex)
        PS[tgt] = ph[:, None] * S
        out[:, k] = np.real(np.einsum("dm,dm->m", S.conj(), PS))
    return out


def pauli_sum(n, terms):
    """Sparse matrix of ``sum_k c_k P_k``.

    Parameters
    ----------
    n : int
        Number of qubits.
    terms : list of tuple
        ``(coefficient, Pauli)`` pairs.

    Returns
    -------
    scipy.sparse.csr_matrix
        The operator, cast to a real matrix when its imaginary part is
        numerically zero (which it is for every Hamiltonian built here).
    """
    D = 1 << n
    idx = np.arange(D)
    r, c, d = [], [], []
    for coef, P in terms:
        tgt, ph = P.action()
        r.append(tgt)
        c.append(idx)
        d.append(coef * ph)
    H = sp.csr_matrix((np.concatenate(d), (np.concatenate(r), np.concatenate(c))), shape=(D, D))
    if H.nnz and abs(H.imag).max() < 1e-14:
        H = H.real.tocsr()
    return H


def dense_pauli(n, ops):
    """Dense Kronecker-product construction of a Pauli string.

    This duplicates what :class:`Pauli` does by bit tricks, deliberately: it is
    the independent implementation the test suite compares against, and it is not
    used to generate data.

    Parameters
    ----------
    n : int
        Number of qubits.
    ops : dict
        Mapping from qubit index to ``'I'``, ``'X'``, ``'Y'`` or ``'Z'``; missing
        qubits are treated as identity.

    Returns
    -------
    ndarray of shape (2 ** n, 2 ** n) of complex
        The dense operator, little-endian.
    """
    m = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}
    out = np.array([[1.0]])
    for q in reversed(range(n)):             # little-endian: qubit n-1 is the leftmost factor
        out = np.kron(out, m[ops.get(q, "I")])
    return out


def pauli_ops(P):
    """Recover the ``(qubits, operators)`` of a :class:`Pauli` from its bitmasks.

    Parameters
    ----------
    P : Pauli
        Pauli string to decode.

    Returns
    -------
    qubits : list of int
        Qubits carrying a non-identity factor, ascending.
    operators : list of str
        The matching ``'X'``, ``'Y'`` or ``'Z'`` factors.
    """
    qs, ss = [], []
    for q in range(P.n):
        b = 1 << q
        x, z = bool(P.mx & b), bool(P.mz & b)
        if x or z:
            qs.append(q)
            ss.append("Y" if x and z else ("X" if x else "Z"))
    return qs, ss


def shot_noise(phi, shots, rng):
    """Replace exact expectation values by finite-shot estimates.

    Each :math:`\\pm 1`-valued Pauli expectation is re-estimated from ``shots``
    independent Bernoulli draws, which is the estimation noise a real measurement
    of that observable would carry.

    Parameters
    ----------
    phi : ndarray
        Exact expectation values in ``[-1, 1]``.
    shots : int
        Shots per observable. ``0`` (or any falsy value) returns ``phi``
        unchanged, i.e. the exact-expectation limit.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    ndarray
        Estimates on the grid ``2 k / shots - 1``, same shape as ``phi``.
    """
    if not shots:
        return phi
    p = np.clip((1 + phi) / 2, 0, 1)
    return 2 * rng.binomial(shots, p) / shots - 1


def fwht(a):
    """Unnormalised Walsh-Hadamard transform along axis 0 (length ``2 ** n``).

    Parameters
    ----------
    a : array_like
        Real vector of length ``2 ** n``.

    Returns
    -------
    ndarray
        The transform. Applying ``fwht`` twice multiplies by ``2 ** n``.
    """
    a = np.array(a, dtype=float)
    h, D = 1, a.shape[0]
    while h < D:
        a = a.reshape(D // (2 * h), 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(D)
        h *= 2
    return a


def walsh_degree_profile(F, n):
    """Distribute the variance of ``F`` over Walsh (Fourier) degrees.

    Parameters
    ----------
    F : ndarray of shape (2 ** n,)
        The target evaluated on *all* ``2 ** n`` bitstring inputs.
    n : int
        Number of bits.

    Returns
    -------
    profile : ndarray of shape (n,)
        Fraction of ``Var(F)`` carried by degree ``1 .. n``.
    effective_degree : float
        Mean degree under ``profile``, a single-number difficulty summary: a
        target concentrated on low degrees is learnable from few samples, and one
        spread to high degrees is not.
    """
    fh = fwht(F) / (1 << n)
    deg = _popcount(np.arange(1 << n))
    w = np.array([np.sum(fh[deg == d] ** 2) for d in range(n + 1)])
    var = w[1:].sum()
    prof = (w[1:] / var) if var > 0 else np.zeros(n)
    return prof, float(np.sum(np.arange(1, n + 1) * prof))


def level_spacing_ratio(E):
    """Mean adjacent-gap ratio over the middle half of a spectrum.

    Parameters
    ----------
    E : ndarray
        Eigenvalues, in any order.

    Returns
    -------
    float
        Mean of ``min(s_i, s_{i+1}) / max(s_i, s_{i+1})`` over consecutive gaps.
        Roughly 0.39 for an integrable (Poisson) spectrum and 0.53 for a
        chaotic (GOE) one, so it says whether the chosen Hamiltonian is actually
        in the non-integrable regime the ``te`` family assumes. ``nan`` when the
        middle half of the spectrum holds fewer than two resolvable gaps, which
        needs a Hilbert space far smaller than any the ``te`` family uses.
    """
    E = np.sort(E)
    k = len(E)
    s = np.diff(E[k // 4: 3 * k // 4])
    s = s[s > 1e-12]
    if len(s) < 2:                                # nothing to take a ratio of
        return float("nan")
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(r.mean())


# ----------------------------------------------------------------------------- pools & models


def pool_z2_even(n):
    """Real, :math:`Z_2`-even local Paulis, well defined on an even-sector state.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    list of Pauli
        Single-site X, nearest-neighbour XX, YY and ZZ, and next-nearest ZZ.
        Every element commutes with :math:`\\prod_i X_i`, so its expectation is
        unambiguous on the ``prod X = +1`` ground state; odd observables would
        average to zero there and carry no signal.
    """
    P = [Pauli(n, {i: "X"}) for i in range(n)]
    for a, b in (("Z", "Z"), ("X", "X"), ("Y", "Y")):
        P += [Pauli(n, {i: a, i + 1: b}) for i in range(n - 1)]
    P += [Pauli(n, {i: "Z", i + 2: "Z"}) for i in range(n - 2)]
    return P


def pool_local(n):
    """All 1-local Paulis and nearest-neighbour XX, YY, ZZ.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    list of Pauli
        ``3 n + 3 (n - 1)`` observables.
    """
    P = [Pauli(n, {i: s}) for s in "XYZ" for i in range(n)]
    for s in "XYZ":
        P += [Pauli(n, {i: s, i + 1: s}) for i in range(n - 1)]
    return P


def sparse_observable(pool, s, rng):
    """Draw a hidden sparse observable from ``pool``.

    Parameters
    ----------
    pool : list of Pauli
        Candidate terms.
    s : int
        Number of terms to keep.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    support : ndarray of shape (s,) of int
        Indices into ``pool``, ascending.
    alpha : ndarray of shape (s,) of float
        Coefficients drawn uniformly from ``[-1, 1]``.
    """
    S = np.sort(rng.choice(len(pool), size=s, replace=False))
    alpha = rng.uniform(-1, 1, size=s)
    return S, alpha


def ising_terms(n, J, h, kappa=0.0, g=None, sign=-1.0):
    """Terms of a mixed-field Ising chain.

    Builds ``sign * (sum_i J_i Z_i Z_{i+1} + sum_i h_i X_i + kappa sum_i X_i
    X_{i+1} + sum_i g_i Z_i)``.

    Parameters
    ----------
    n : int
        Number of qubits.
    J : array_like of shape (n - 1,)
        Nearest-neighbour ZZ couplings.
    h : array_like of shape (n,)
        Transverse fields.
    kappa : float, default=0.0
        Nearest-neighbour XX coupling. A non-zero value makes the model
        interacting rather than free-fermion, which is the point of the knob.
    g : array_like of shape (n,), optional
        Longitudinal fields. Omitting them keeps the :math:`Z_2` symmetry
        :func:`even_sector_ground_state` relies on.
    sign : float, default=-1.0
        Overall sign; ``-1`` gives the conventional ferromagnetic form.

    Returns
    -------
    list of tuple
        ``(coefficient, Pauli)`` pairs, ready for :func:`pauli_sum`.
    """
    t = [(sign * J[i], Pauli(n, {i: "Z", i + 1: "Z"})) for i in range(n - 1)]
    t += [(sign * h[i], Pauli(n, {i: "X"})) for i in range(n)]
    if kappa:
        t += [(sign * kappa, Pauli(n, {i: "X", i + 1: "X"})) for i in range(n - 1)]
    if g is not None:
        t += [(sign * g[i], Pauli(n, {i: "Z"})) for i in range(n)]
    return t


def even_sector_ground_state(n, terms):
    """Ground state in the :math:`\\prod_i X_i = +1` sector.

    The two lowest states of a transverse-field Ising chain are exponentially
    close in ``n``, so "the" ground state is numerically arbitrary at the sizes
    used here and small parameter changes would flip which of the two is
    returned. Projecting onto the even sector by penalising the odd one removes
    that ambiguity, and the returned gap is the gap *within* the sector.

    Parameters
    ----------
    n : int
        Number of qubits.
    terms : list of tuple
        ``(coefficient, Pauli)`` pairs, as from :func:`ising_terms`. Every term
        must commute with :math:`\\prod_i X_i`.

    Returns
    -------
    psi : ndarray of shape (2 ** n,) of complex
        Normalised even-sector ground state.
    gap : float
        Energy gap to the next even-sector state.
    residual : float
        ``norm(H psi - E0 psi)``, a check that ``psi`` is an eigenvector of the
        *unpenalised* Hamiltonian. This is recorded per dataset as
        ``diagnostics.max_eigen_residual``.
    """
    H = pauli_sum(n, terms)
    Pall = pauli_sum(n, [(1.0, Pauli(n, {i: "X" for i in range(n)}))])
    c = 2 * sum(abs(t[0]) for t in terms) + 1.0
    Hp = H + (c / 2) * (sp.identity(1 << n, format="csr") - Pall)
    if (1 << n) <= 1024:
        E, V = np.linalg.eigh(Hp.toarray())
        e0, e1, psi = E[0], E[1], V[:, 0]
    else:
        E, V = eigsh(Hp, k=2, which="SA", tol=1e-12)
        o = np.argsort(E)
        e0, e1, psi = E[o[0]], E[o[1]], V[:, o[0]]
    res = np.linalg.norm(H @ psi - e0 * psi)
    return psi.astype(complex), float(e1 - e0), float(res)


def heisenberg_terms(n, rng):
    """Random nearest-neighbour Heisenberg chain terms.

    Parameters
    ----------
    n : int
        Number of qubits.
    rng : numpy.random.Generator
        Source of randomness; each of the ``3 (n - 1)`` couplings is drawn
        uniformly from ``[-1, 1]``.

    Returns
    -------
    list of tuple
        ``(coefficient, Pauli)`` pairs.
    """
    t = []
    for i in range(n - 1):
        for s in "XYZ":
            t.append((rng.uniform(-1, 1), Pauli(n, {i: s, i + 1: s})))
    return t


#: Entanglement patterns :func:`zz_feature_state` accepts. Named here so the
#: generators can reject a typo before any simulation runs -- the ``evo`` encoding
#: ignores the pattern, so a bad value would otherwise pass silently.
ENTANGLEMENTS = ("linear", "pairwise", "full")

#: Data maps :func:`zz_feature_state` accepts -- the function Qiskit's
#: ``PauliFeatureMap`` applies to the features before they become rotation angles.
#:
#: ``'qiskit'`` is Qiskit's documented default,
#: ``phi(x_i) = x_i`` and ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)``.
#: ``'unit'`` halves at every step, ``phi(x_i) = x_i / 2`` and
#: ``phi(x_i, x_j) = x_i x_j / 2``, reproducing
#: :func:`qbiocode.utils.qutils.unit_coefficient_data_map`.
#:
#: The two are *different unitaries*, not a reparameterisation, and which one a
#: consumer uses decides whether a kernel-aligned family is aligned at all. See
#: the note in :func:`zz_feature_state` for which QProfiler model uses which.
DATA_MAPS = ("qiskit", "unit")


def zz_feature_state(x, reps=2, entanglement="linear", data_map="qiskit"):
    """Statevector of Qiskit's ``ZZFeatureMap`` applied to ``x``.

    Native re-implementation. Per repetition: H on every qubit, followed by the
    diagonal phase ``exp(i 2 [sum_i phi(x_i) b_i
    + sum_(i,j) phi(x_i, x_j) (b_i xor b_j)])``, where ``phi`` is the data map
    selected by ``data_map``. Agreement with Qiskit is asserted by the test
    suite to ~1e-15 for both data maps.

    .. important::
       **QProfiler's two quantum models do not use the same data map**, so there
       is no single choice here that aligns with both:

       * ``qsvc`` calls :func:`qbiocode.utils.qutils.get_feature_map` without a
         ``data_map_func``, so it gets Qiskit's default -- ``data_map='qiskit'``.
       * ``pqk`` (both :func:`qbiocode.learning.compute_pqk.compute_pqk` and
         :func:`qbiocode.embeddings.embed.pqk`) passes
         :func:`~qbiocode.utils.qutils.unit_coefficient_data_map` -- ``data_map='unit'``.

       A family whose labels are tuned to one kernel is a *negative* control for
       the other. Measured on ``eng_zz_n4_gq1_s0``, projected-kernel accuracy is
       0.797 +/- 0.073 under ``'qiskit'`` and 0.403 +/- 0.108 under ``'unit'``
       (40 stratified 70/30 splits) -- below chance, because the labels are
       anti-aligned with the other encoding's geometry.

    Parameters
    ----------
    x : array_like of shape (n,)
        Input features, one per qubit.
    reps : int, default=2
        Number of repetitions. QProfiler's QSVC default is 2 and its PQK default
        is 4; a mismatch here changes the labelling and is enough on its own to
        remove the intended advantage.
    entanglement : {'linear', 'pairwise', 'full'}, default='linear'
        Pair set for the two-qubit phases. ``'pairwise'`` is an alias of
        ``'linear'``: it generates the same pairs, and since the phases are
        diagonal their order is irrelevant.
    data_map : {'qiskit', 'unit'}, default='qiskit'
        Which data map to apply; see :data:`DATA_MAPS` and the note above. The
        default reproduces Qiskit's stock ``ZZFeatureMap`` and keeps every
        previously generated dataset bit-for-bit reproducible.

    Returns
    -------
    ndarray of shape (2 ** n,) of complex
        The encoded statevector.

    Raises
    ------
    ValueError
        If ``entanglement`` is not one of the three accepted values, or
        ``data_map`` is not one of :data:`DATA_MAPS`.
    """
    if data_map not in DATA_MAPS:
        raise ValueError(
            f"data_map must be one of {list(DATA_MAPS)}, got {data_map!r}. "
            "'qiskit' matches QProfiler's qsvc, 'unit' matches its pqk."
        )
    x = np.asarray(x, float)
    n = len(x)
    idx = np.arange(1 << n)
    bits = (idx[:, None] >> np.arange(n)) & 1
    if entanglement in ("linear", "pairwise"):
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif entanglement == "full":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError("entanglement must be 'linear', 'pairwise' or 'full'")
    if data_map == "qiskit":
        theta = 2 * (bits @ x)
        for i, j in pairs:
            theta = theta + 2 * (np.pi - x[i]) * (np.pi - x[j]) * (bits[:, i] ^ bits[:, j])
    else:
        # unit_coefficient_data_map halves at every step: phi(x_i) = x_i / 2 and
        # phi(x_i, x_j) = x_i x_j / 2, and qiskit applies 2 * phi as the angle.
        theta = bits @ x
        for i, j in pairs:
            theta = theta + x[i] * x[j] * (bits[:, i] ^ bits[:, j])
    ph = np.exp(1j * theta)
    psi = np.zeros(1 << n, complex)
    psi[0] = 1.0
    for _ in range(reps):
        psi = ph * (fwht(psi.real) + 1j * fwht(psi.imag)) / np.sqrt(1 << n)
    return psi


def evo_encoding_state(x, t_enc=1.0, layers=2):
    """Hamiltonian-evolution encoding of ``x``.

    Starting from ``|+>^n``, applies ``layers`` repetitions of
    ``exp(-i t_enc sum_i X_i X_{i+1}) exp(-i t_enc sum_i x_i Z_i)`` -- reading the
    product right to left, as it acts on the state: the data-dependent Z phase
    first, then the entangling XX evolution.

    Parameters
    ----------
    x : array_like of shape (n,)
        Input features.
    t_enc : float, default=1.0
        Evolution time per layer.
    layers : int, default=2
        Number of layers.

    Returns
    -------
    ndarray of shape (2 ** n,) of complex
        The encoded statevector.

    Notes
    -----
    This encoding is defined here and is *not* the E3 encoding of Huang et al.
    Its role is to be deliberately misaligned with QProfiler's ZZ-based kernels,
    giving a negative control: a dataset labelled through it should be no easier
    for the quantum arms than for the classical ones.
    """
    n = len(x)
    D = 1 << n
    bits = (np.arange(D)[:, None] >> np.arange(n)) & 1
    zdiag = np.exp(-1j * t_enc * ((1 - 2 * bits) @ np.asarray(x, float)))
    Hxx = pauli_sum(n, [(1.0, Pauli(n, {i: "X", i + 1: "X"})) for i in range(n - 1)]).toarray()
    w, V = np.linalg.eigh(Hxx)
    Uxx = (V * np.exp(-1j * t_enc * w)) @ V.conj().T
    psi = np.full(D, 1 / np.sqrt(D), complex)
    for _ in range(layers):
        psi = Uxx @ (zdiag * psi)
    return psi


# ----------------------------------------------------------------------------- engineered kernels


def sqrtm_psd(K):
    """Symmetric square root of a positive semi-definite matrix.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Symmetric matrix; negative eigenvalues from round-off are clipped to zero.

    Returns
    -------
    ndarray of shape (N, N)
        The symmetric square root.
    """
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def engineered_labels(K_C, K_Q, lam=1e-3):
    """Continuous labels maximising the advantage of ``K_Q`` over ``K_C``.

    Constructs a target that saturates ``s_C(y) = g ** 2 s_Q(y)`` with
    ``s_Q(y) = 1``, where ``s`` is the kernel-regression model complexity and
    ``g`` is the geometric difference with a ridge, in the sense of equation (5)
    of Huang et al. (2021). Concretely ``y = sqrt(K_Q) v`` where ``v`` is the top
    eigenvector of ``sqrt(K_Q) (K_C + lam I)^{-1} sqrt(K_Q)``.

    Parameters
    ----------
    K_C, K_Q : ndarray of shape (N, N)
        Classical and quantum kernel matrices. Both are rescaled to trace ``N``
        before use, so ``g`` does not depend on their overall normalisation.
    lam : float, default=1e-3
        Ridge on ``K_C``, which keeps ``g`` finite when ``K_C`` is near-singular.

    Returns
    -------
    y : ndarray of shape (N,)
        Continuous target.
    g : float
        Geometric difference ``g(K_C || K_Q)``; larger means the quantum kernel
        has more room to outperform.
    K_C, K_Q : ndarray of shape (N, N)
        The trace-normalised kernels, returned so the complexities can be
        recomputed against exactly the matrices used here.

    Notes
    -----
    This formula was derived to satisfy that equation; the Supplementary Section 7
    procedure of the same paper was not consulted. The guarantee also covers the
    *continuous* target only -- binarising at the median gives a label whose
    separation must be measured, not assumed.
    """
    N = K_C.shape[0]
    K_C = K_C * N / np.trace(K_C)
    K_Q = K_Q * N / np.trace(K_Q)
    Qs = sqrtm_psd(K_Q)
    M = Qs @ np.linalg.solve(K_C + lam * np.eye(N), Qs)
    w, Vv = np.linalg.eigh((M + M.T) / 2)
    return Qs @ Vv[:, -1], float(np.sqrt(w[-1])), K_C, K_Q


def rbf(A, gamma):
    """Gaussian (RBF) kernel matrix of the rows of ``A``.

    Parameters
    ----------
    A : ndarray of shape (N, d)
        Feature vectors.
    gamma : float
        Kernel bandwidth parameter.

    Returns
    -------
    ndarray of shape (N, N)
        ``exp(-gamma ||a_i - a_j|| ** 2)``.
    """
    d2 = np.sum(A ** 2, 1)[:, None] + np.sum(A ** 2, 1)[None] - 2 * A @ A.T
    return np.exp(-gamma * np.clip(d2, 0, None))


# ----------------------------------------------------------------------------- labels & I/O


def threshold(F, margin):
    """Median threshold for binarising ``F``, plus a margin mask.

    Parameters
    ----------
    F : ndarray of shape (N,)
        Continuous target.
    margin : float
        Rows with ``|F - median| < margin`` are excluded, which removes the
        ambiguous band around the decision boundary and makes the task easier
        without changing the label rule.

    Returns
    -------
    thr : float
        ``median(F)``, so the resulting labels are balanced by construction.
    keep : ndarray of shape (N,) of bool
        Rows to keep.

    Raises
    ------
    ValueError
        If no row survives ``margin``, or if the surviving rows all carry the same
        label. Either way the dataset written would be unusable -- a zero-row CSV,
        or one class -- and QProfiler would fail far from the cause, on a
        stratified split it cannot make.
    """
    F = np.asarray(F, float)
    thr = float(np.median(F))
    keep = np.abs(F - thr) >= margin
    if not keep.any():
        raise ValueError(
            f"margin={margin:g} removed all {len(F)} rows: every target lies within it of "
            f"the median {thr:g} (targets span {float(F.min()):g} to {float(F.max()):g}). "
            "Lower margin, or generate more rows so the target spreads further."
        )
    labels = F[keep] > thr
    if labels.all() or not labels.any():
        raise ValueError(
            f"the {int(keep.sum())} rows surviving margin={margin:g} all carry label "
            f"{int(labels[0])}: the continuous target is constant or has too few distinct "
            f"values around its median {thr:g} to binarise. Generate more rows."
        )
    return thr, keep


def write_dataset(save_path, name, X, xcols, y, F, meta, phi=None, phicols=None):
    """Write one dataset in the layout QProfiler reads.

    Parameters
    ----------
    save_path : str
        Directory to write into. ``x_view/`` and ``meta/`` are created as needed,
        and ``phi_view/`` only when ``phi`` is given -- an empty one would read to
        QProfiler as a corpus containing no datasets rather than as a family with
        no second view.
    name : str
        Dataset name, used as the CSV and sidecar basename.
    X : ndarray of shape (N, d)
        Features for ``x_view``.
    xcols : list of str
        Column names for ``X``.
    y : array_like of shape (N,)
        Binary labels, written as the LAST column because QProfiler splits
        features from labels positionally.
    F : ndarray of shape (N,)
        Continuous pre-threshold target, saved for margin or regression analysis.
    meta : dict
        Parameters, label rule, threshold and diagnostics. ``n_rows`` and
        ``class_balance`` are added here.
    phi : ndarray of shape (N, k), optional
        Features for ``phi_view``; omitted for families with a single view.
    phicols : list of str, optional
        Column names for ``phi``.

    Returns
    -------
    dict
        The metadata as written.
    """
    views = ("x_view", "meta") if phi is None else ("x_view", "phi_view", "meta")
    for d in views:
        os.makedirs(os.path.join(save_path, d), exist_ok=True)
    y = np.asarray(y, int)
    df = pd.DataFrame(X, columns=xcols)
    df["label"] = y
    df.to_csv(os.path.join(save_path, "x_view", f"{name}.csv"), index=False)
    if phi is not None:
        dp = pd.DataFrame(phi, columns=phicols)
        dp["label"] = y
        dp.to_csv(os.path.join(save_path, "phi_view", f"{name}.csv"), index=False)
    np.save(os.path.join(save_path, "meta", f"{name}_F.npy"), np.asarray(F, float))
    meta = dict(meta, n_rows=int(len(y)), class_balance=float(y.mean()))
    with open(os.path.join(save_path, "meta", f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"[{name}] rows={len(y)} balance={y.mean():.3f} -> {save_path}")
    return meta


def as_list(value):
    """Wrap a scalar in a list, leaving sequences alone.

    Lets the dataset generators accept either ``n_qubits=8`` or
    ``n_qubits=[6, 8]`` without the caller having to care.

    Parameters
    ----------
    value : object
        A scalar, or any iterable of them -- a list, tuple, :class:`range`, NumPy
        array or Hydra ``ListConfig``. Strings, bytes, mappings and 0-dimensional
        arrays count as scalars.

    Returns
    -------
    list
        ``list(value)`` for a non-scalar iterable, else ``[value]``.

    Notes
    -----
    The iterable test is deliberately broader than ``isinstance(value, (list,
    tuple))``. QProfiler is Hydra-driven, and a list in a YAML config arrives as
    an ``omegaconf.ListConfig``, which is not a :class:`list`; a ``dtype``-typed
    ``np.linspace`` of taus is not one either. Wrapping such a value instead of
    iterating it would sweep over a single configuration whose "qubit count" is a
    whole sequence, which fails much later and confusingly.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_core import as_list
    >>> as_list(8)
    [8]
    >>> as_list((6, 8))
    [6, 8]
    >>> as_list("sparse")
    ['sparse']
    >>> as_list(range(4, 6))
    [4, 5]
    """
    if isinstance(value, (str, bytes, bytearray, dict)):
        return [value]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [value.item()]
    try:
        entries = list(value)
    except TypeError:                             # not iterable: a genuine scalar
        entries = [value]
    # NumPy scalars are not JSON-serialisable, and every knob passed through here is
    # recorded verbatim in the dataset metadata, so an ndarray of qubit counts would
    # otherwise land in the JSON as 6.0 rather than 6 via ``json.dump(default=float)``.
    return [v.item() if isinstance(v, np.generic) else v for v in entries]


def ensure_unique_names(names, unnamed_knobs=()):
    """Raise if a sweep would write two datasets to the same file.

    Dataset names encode the physical parameters that identify a dataset, but not
    every sweepable knob appears in them -- ``n_samples``, for instance, is in no
    family's name. A sweep over such a knob therefore produces repeated names, and
    since each dataset is written by path the second silently overwrites the first
    while both are still reported as generated. That is worth an exception rather
    than a corrupt corpus.

    Parameters
    ----------
    names : sequence of str
        The dataset names a sweep is about to write, in order.
    unnamed_knobs : sequence of str, optional
        Names of the swept parameters that do *not* appear in the dataset name, to
        quote in the error message as the likely cause.

    Raises
    ------
    ValueError
        If any name occurs more than once.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_core import ensure_unique_names
    >>> ensure_unique_names(["gs_sparse_n8_k0.5_s0", "gs_e2e_n8_k0.5_s0"])
    >>> ensure_unique_names(["a", "a"], unnamed_knobs=["n_samples"])
    Traceback (most recent call last):
        ...
    ValueError: this sweep would write 2 datasets to the name 'a' ...
    """
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    repeated = {name: count for name, count in counts.items() if count > 1}
    if not repeated:
        return
    name, count = next(iter(repeated.items()))
    because = (
        f" Dataset names do not encode {', '.join(unnamed_knobs)}, so sweeping over "
        f"{'them' if len(unnamed_knobs) > 1 else 'it'} collides."
        if unnamed_knobs else ""
    )
    raise ValueError(
        f"this sweep would write {count} datasets to the name {name!r} "
        f"({len(repeated)} name(s) repeat in total), and each would overwrite the "
        f"last.{because} Vary a parameter that appears in the name, or make one call "
        f"per dataset passing an explicit name."
    )


def reject_name_for_sweep(name, n_configurations, knobs):
    """Raise if an explicit ``name`` is combined with a multi-configuration sweep.

    ``name`` overrides the generated, parameter-encoding dataset name. That is useful
    for a single dataset and incoherent for a sweep: every configuration would be
    written under the one name, so all but the last would be silently overwritten.
    :func:`ensure_unique_names` cannot catch it, because with ``name`` given there is
    only ever one name to compare.

    Checked before anything is written, so a rejected sweep leaves no partial output.

    Parameters
    ----------
    name : str or None
        The caller's explicit dataset name. ``None`` means "generate names", and is
        always accepted.
    n_configurations : int
        How many configurations the sweep expanded to.
    knobs : sequence of str
        The swept parameter names, to quote back as the ones to collapse.

    Raises
    ------
    ValueError
        If ``name`` is given and ``n_configurations`` exceeds one.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_core import reject_name_for_sweep
    >>> reject_name_for_sweep(None, 4, ["n_qubits"])
    >>> reject_name_for_sweep("mine", 1, ["n_qubits"])
    >>> reject_name_for_sweep("mine", 2, ["n_qubits", "random_state"])
    Traceback (most recent call last):
        ...
    ValueError: name='mine' was given but the sweep has 2 configurations ...
    """
    if name is None or n_configurations <= 1:
        return
    knobs = list(knobs)
    listed = (f"{', '.join(knobs[:-1])} and {knobs[-1]}") if len(knobs) > 1 else knobs[0]
    raise ValueError(
        f"name={name!r} was given but the sweep has {n_configurations} configurations, "
        f"whose outputs would collide under that one name. Pass a single value for each "
        f"of {listed}, or drop name to have each configuration named after its parameters."
    )


def as_float_list(value):
    """:func:`as_list` for a float-valued knob, coercing every entry to ``float``.

    Float knobs are recorded verbatim in the dataset metadata, and several appear in
    dataset names. Without coercion ``tau=1`` and ``tau=1.0`` produce metadata that
    differs (``"tau": 1`` against ``"tau": 1.0``) for runs that are numerically
    identical, which makes a byte comparison of two corpora report a change that is
    only a caller's choice of literal. Coercing here also makes the Python API agree
    with the command line, where ``argparse`` has already applied ``type=float``.

    Parameters
    ----------
    value : float or sequence of float
        A single value or a sequence of them.

    Returns
    -------
    list of float

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_core import as_float_list
    >>> as_float_list(1)
    [1.0]
    >>> as_float_list([0.25, 1, 2])
    [0.25, 1.0, 2.0]
    """
    return [float(v) for v in as_list(value)]
