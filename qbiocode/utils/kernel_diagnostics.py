# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kernel-level diagnostics: alignment, geometric separation, and margin.

These answer a different question from the accuracy table. A delta in balanced accuracy
says *whether* the quantum arm won; these say *whether the quantum kernel sees structure
the classical one cannot reach*, which is the claim a paper actually wants to defend. A
quantum win with g_cq near 1 is a win the classical kernel could have had -- so it is
evidence about the tuning, not about quantum advantage.

Three quantities, following Huang et al., *Power of data in quantum machine learning*
(Nat. Commun. 12, 2631, 2021):

``kta``
    Kernel-target alignment. How much of the kernel's structure is label structure.
``geometric_separation``
    g(K_c || K_q): how much better a quantum model *could* do than the best classical
    model using K_c. g near 1 means the classical kernel already spans it; large g means
    there is room, though room is not the same as realised advantage.
``margin`` / ``model_complexity``
    The RKHS norm of the fitted separator, which bounds generalisation.

Normalisation is not optional here -- see :func:`normalize_trace`.
"""

import numpy as np

__all__ = [
    "psd_sqrt",
    "normalize_trace",
    "kta",
    "geometric_separation",
    "compute_margin",
    "margin_from_kernel",
    "model_complexity_sK",
    "kta_inverse",
    "kernel_report",
]


def psd_sqrt(K, eps=1e-10):
    """PSD square root via eigendecomposition with clipping.

    Symmetrising first is load-bearing: a fidelity kernel estimated from finite shots is
    only symmetric up to sampling noise, and ``eigh`` on a slightly asymmetric matrix
    silently uses the lower triangle rather than complaining.

    Clipping to ``eps`` rather than to 0 keeps the result positive *definite*, so the
    inverse taken in :func:`geometric_separation` stays finite. Shot noise routinely puts
    a few eigenvalues slightly below zero, and those are noise around zero, not signal.
    """
    K = (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T) / 2
    w, V = np.linalg.eigh(K)
    return (V * np.sqrt(np.clip(w, eps, None))) @ V.T


def normalize_trace(K):
    """Scale ``K`` so ``trace(K) == N``, i.e. unit average self-similarity.

    :func:`geometric_separation` is **not scale invariant**: ``K_q -> c K_q`` sends
    ``g -> sqrt(c) g``. So comparing a raw quantum kernel against a raw classical one
    measures their relative scale as much as their relative structure, and comparing g
    across datasets of different width measures nothing at all.

    Fidelity kernels and RBF kernels both have unit diagonal already, so this is a no-op
    for them -- but a linear or polynomial classical kernel does not, and the PQK kernel's
    scale moves with the tuned ``gamma``. Normalising unconditionally costs nothing and
    removes the trap.
    """
    K = np.asarray(K, dtype=float)
    tr = float(np.trace(K))
    if not np.isfinite(tr) or tr <= 0:
        return K
    return K * (K.shape[0] / tr)


def kta(K, y):
    """Kernel-target alignment: ``y^T K y / (N * ||K||_F)``.

    This is Cristianini's alignment ``<K, yy^T>_F / (||K||_F ||yy^T||_F)`` specialised to
    labels in ``{-1, +1}``, where ``<K, yy^T>_F = y^T K y`` and ``||yy^T||_F = N``. Labels
    given as ``{0, 1}`` are mapped to ``{-1, +1}`` first; passing them through unmapped
    would silently drop every negative-class contribution and inflate the alignment.

    Note on a formula that looks like this one but is not::

        kta_q = float(y @ y) / (N * np.sqrt(N))     # <- no kernel in it

    With labels in ``{-1, +1}`` that is ``N / (N sqrt(N)) = 1/sqrt(N)`` for *every*
    kernel and every dataset -- it is the alignment of the identity matrix, since
    ``||I||_F = sqrt(N)``. It is a constant, so a table of it reports only sample size.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y).ravel()
    u = np.unique(y)
    if u.size == 2 and not np.array_equal(np.sort(u), np.array([-1.0, 1.0])):
        y = np.where(y == u[0], -1.0, 1.0)
    y = y.astype(float)
    denom = y.size * np.linalg.norm(K, "fro")
    return float(y @ K @ y / denom) if denom > 0 else np.nan


def geometric_separation(Kc, Kq, lam=1e-3, form="standard", normalize=True):
    """g(K_c || K_q) -- the geometric difference, ``sqrt(lambda_max(M))``.

    ``standard`` is Huang et al.'s asymmetric difference with the classical kernel
    regularised::

        M = sqrt(Kq) (Kc + lam I)^-1 sqrt(Kq)

    Large g means there exists a labelling the quantum kernel separates and the classical
    one does not. It is a statement about *potential*, one-sided: g must be large for
    quantum advantage to be possible, but a large g with no accuracy gain simply means the
    potential went unused.

    ``symmetric`` replaces the hard inverse with ``Kc^{1/2}(Kc+lam I)^{-2}Kc^{1/2}``, which
    tends to the same limit as ``lam -> 0`` but damps the smallest classical eigenvalues
    smoothly instead of letting ``lam`` alone hold them off zero. Prefer it when Kc is
    near-singular, which is the normal case once the embedding has fewer components than
    rows.

    ``lam`` matters as much as the kernels do, and a single value is not a result: g decays
    monotonically in ``lam``, from "the classical kernel is invertible noise" at one end to
    "everything is classical" at the other. Report :func:`kernel_report`'s sweep, and read
    the trend, not a point.

    The inverse goes through the same clipped eigendecomposition as :func:`psd_sqrt`
    rather than ``numpy.linalg.inv``: with ``lam`` small and Kc rank-deficient, ``inv``
    returns large finite garbage instead of failing, and that garbage lands in g as a
    spuriously large separation.
    """
    Kc = np.asarray(Kc, dtype=float)
    Kq = np.asarray(Kq, dtype=float)
    if normalize:
        Kc, Kq = normalize_trace(Kc), normalize_trace(Kq)
    N = Kc.shape[0]
    Kq_sqrt = psd_sqrt(Kq)

    Kc_sym = (Kc + Kc.T) / 2
    w, V = np.linalg.eigh(Kc_sym)
    if form == "standard":
        inner = (V * (1.0 / np.clip(w + lam, 1e-300, None))) @ V.T
    elif form == "symmetric":
        wc = np.clip(w, 0.0, None)
        scale = wc / np.clip(w + lam, 1e-300, None) ** 2
        inner = (V * scale) @ V.T
    else:
        raise ValueError("form must be 'standard' or 'symmetric'")

    M = Kq_sqrt @ inner @ Kq_sqrt
    M = (M + M.T) / 2
    return float(np.sqrt(max(float(np.linalg.eigvalsh(M).max()), 0.0)))


def compute_margin(svc, K_train):
    """``1/||w||`` in the RKHS for a trained precomputed-kernel SVM.

    ``dual_coef_`` holds ``y_i * alpha_i`` for the support vectors, so
    ``w = sum_i dual_coef_i phi(x_i)`` and ``||w||^2 = d^T K_sv d``.

    For a soft-margin SVM this is the inverse weight norm, which is the geometric distance
    to the separating hyperplane only for the points that sit exactly on the margin. It is
    the quantity that appears in the generalisation bound, which is why it is the one worth
    reporting -- but it is not "the distance to the nearest training point" once slack is
    active, and describing it that way in a paper would be wrong.
    """
    sv_idx = svc.support_
    K_sv = np.asarray(K_train, dtype=float)[np.ix_(sv_idx, sv_idx)]
    d = np.asarray(svc.dual_coef_).reshape(-1)
    w2 = float(d @ K_sv @ d)
    return 1.0 / np.sqrt(w2) if w2 > 0 else np.nan


def margin_from_kernel(K, y, C=1.0):
    """Fit a precomputed-kernel SVM on ``K`` and return its RKHS margin ``1/||w||``.

    :func:`compute_margin` needs a *trained* machine, and nothing in the benchmark persists
    one -- ``qprofiler`` pickles the results frame, not the estimators. So a margin computed
    from a dumped Gram is necessarily the margin of a machine fitted here, and ``C`` is a
    choice this function makes rather than one it recovers.

    That choice matters: ``1/||w||`` grows with the slack ``C`` allows, so margins are
    comparable **only** at equal ``C`` and only between kernels on the same scale. Callers
    that compare a classical against a quantum margin must therefore trace-normalise both
    (:func:`normalize_trace`) and hold ``C`` fixed -- which is what :func:`kernel_report`
    does, recording the value under ``margin_C`` so the number is never read as
    C-independent.

    ``C=1.0`` is sklearn's default and is *not* the tuned ``C`` from the run; the tuned value
    lives in the results frame and can be passed in when a per-arm margin is wanted.
    """
    from sklearn.svm import SVC

    K = np.asarray(K, dtype=float)
    y = np.asarray(y).ravel()
    if np.unique(y).size < 2:
        return np.nan
    try:
        # random_state is inert here -- libsvm draws from it only for Platt scaling, which
        # probability=False never runs, so it cannot move the margin. Pinned anyway so the
        # package-wide contract in test_split_reproducibility holds without an exemption,
        # and so turning probability on later cannot quietly start drawing from the
        # global RNG.
        svc = SVC(kernel="precomputed", C=C, random_state=0).fit((K + K.T) / 2, y)
    except Exception:
        return np.nan
    return compute_margin(svc, K)


def model_complexity_sK(svc, K_train):
    """``sK = sqrt(alpha^T K alpha) = ||w|| = 1 / margin``.

    Exactly the reciprocal of :func:`compute_margin`, kept because the literature names
    both. Reporting the pair as two independent columns would overstate how much is being
    measured -- they carry one number between them.
    """
    m = compute_margin(svc, K_train)
    return 1.0 / m if np.isfinite(m) and m > 0 else np.nan


def kta_inverse(Kc, y, lam=1e-3):
    """Alignment of the *regularised inverse* ``(K_c + lam I)^-1`` with the labels.

    This is the second of the two alignment lines in the pasted reference code::

        kta_c = float(y @ Ainv @ y) / (N * np.linalg.norm(Ainv, 'fro'))

    It is a different quantity from :func:`kta`, not a variant spelling of it, and the two
    are not comparable: alignment with ``K`` asks whether similar points share a label,
    while alignment with ``K^-1`` is dominated by the *smallest* eigenvalues of ``K`` --
    the directions the kernel considers least, i.e. the ones the regulariser controls.
    Large ``y^T A^-1 y`` is the standard kernel-ridge complexity term and reads as the
    labels being *hard* for that kernel, so its sign as evidence is the opposite of
    :func:`kta`'s. It is also strongly ``lam``-dependent, where :func:`kta` is not.

    Reported alongside ``kta_classical`` rather than in place of it, so whichever of the
    two was intended is present and neither is silently substituted for the other.
    """
    Kc = np.asarray(Kc, dtype=float)
    y = np.asarray(y).ravel()
    u = np.unique(y)
    if u.size == 2 and not np.array_equal(np.sort(u), np.array([-1.0, 1.0])):
        y = np.where(y == u[0], -1.0, 1.0)
    y = y.astype(float)
    # Same clipped eigendecomposition used by geometric_separation, and for the same
    # reason: on a rank-deficient Kc, np.linalg.inv returns large finite garbage that
    # looks like a result.
    w, V = np.linalg.eigh((Kc + Kc.T) / 2)
    Ainv = (V * (1.0 / np.clip(w + lam, 1e-300, None))) @ V.T
    denom = y.size * np.linalg.norm(Ainv, "fro")
    return float(y @ Ainv @ y / denom) if denom > 0 else np.nan


def kernel_report(
    Kc, Kq, y, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), form="symmetric", margin_C=1.0
):
    """Alignment for both kernels plus g over a ``lam`` sweep, as one flat dict.

    Flat because this becomes one row per (dataset, embedding, iteration) in the
    diagnostics table that sits beside the accuracy table.
    """
    Kc_n, Kq_n = normalize_trace(Kc), normalize_trace(Kq)
    out = {
        "kta_classical": kta(Kc_n, y),
        "kta_quantum": kta(Kq_n, y),
        "n_samples": int(np.asarray(Kc).shape[0]),
    }
    out["kta_ratio"] = (
        out["kta_quantum"] / out["kta_classical"]
        if out["kta_classical"] not in (0.0, np.nan) and np.isfinite(out["kta_classical"])
        else np.nan
    )
    # Margins are of machines fitted here at a fixed C -- see margin_from_kernel. Both
    # kernels are trace-normalised above, without which the comparison is a scale artefact.
    out["margin_C"] = float(margin_C)
    out["margin_classical"] = margin_from_kernel(Kc_n, y, C=margin_C)
    out["margin_quantum"] = margin_from_kernel(Kq_n, y, C=margin_C)
    for lam in lams:
        out[f"g_cq_lam{lam:g}"] = geometric_separation(
            Kc_n, Kq_n, lam=lam, form=form, normalize=False
        )
        out[f"kta_classical_inv_lam{lam:g}"] = kta_inverse(Kc_n, y, lam=lam)
    return out
