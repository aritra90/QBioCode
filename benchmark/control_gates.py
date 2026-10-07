"""Accept a quantum positive control on its generator and matched kernel alone.

A positive control is meant to show that the pipeline can find and score a quantum arm
that works. Choosing one because the quantum arm beat the classical arms on it would be
circular. Nothing here fits a classical learner or reads a benchmark row. Every number
comes from a PILOT draw of the generator (seed ``PILOT_SEED``, never a data seed) and the
kernel of the matched arm, the arm the family's labels were built for:

  G1  not classical by construction: the matched map entangles. A Z-only (product) map
      is a product of one-qubit kernels with an exact classical twin (kernel_exps
      ANALYSIS_F F1: identical to 9e-15, and the torus "win" collapsed once that twin
      was a classical arm).
  G2  not concentrated, from the inputs alone: participation ratio PR = tr(K)^2 / ||K||_F^2
      over n, and the mean off-diagonal kernel value. For a fidelity kernel that mean is
      compared with 2^-d, the overlap of random states on d qubits, so the test scales
      with the qubit count; other kernels use a fixed floor. A near-identity kernel
      memorises; it cannot generalise from a few hundred rows.
  G3  learnable by the matched kernel at the benchmark's size: nested cross-validated
      balanced accuracy of SVC(precomputed K), C chosen in the inner folds.
  G4  far enough from a classical kernel for any advantage to be possible, from the
      inputs alone: Huang et al.'s geometric difference g(K_RBF || K_Q), trace-normalised,
      lambda pinned at 1e-3, over sqrt(n_train). Necessary, not sufficient: as a
      predictor of the winner it was a coin flip in kernel_exps (11 of 21).

G2-G4 pull against each other: across 144 ql_zz configurations, log g and the matched
accuracy correlated at -0.67. A control has to clear all four.

``THRESHOLDS``, ``PILOT_SEED`` and ``GATE_VERSION`` are the pre-registration. Fix them
before any benchmark result exists, and bump the version when one changes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

GATE_VERSION = "control_gates/1.1"
PILOT_SEED = 900001
#: Pre-registered 2026-10-07 for full run full1. Why each value (threshold_sensitivity.csv
#: in qbc_work/controls has the grid):
#:   pr_over_n_max 0.25         ~4 training points per effective kernel mode; no config lies
#:                              between 0.17 and 0.47, so the cutoff is not knife-edge.
#:   overlap_over_random_min 2  fidelity kernels: mean overlap at least twice that of random
#:                              states (2^-d); at bandwidth 1 every d sits at 1.6-1.8x.
#:   offdiag_mean_min 0.05      other kernels (the eng RBF on Bloch vectors).
#:   matched_ba_min 0.85        the matched kernel closes 70% of the gap from chance to the
#:                              Bayes rate (1, the labels are deterministic). The most
#:                              sensitive gate: 0.80 admits 16 configs, 0.90 only 7.
#:   g_over_sqrt_n_min 0.5      half of the sqrt(n) scale of Huang et al.'s bound (slack for
#:                              its constants); it removes kernels an RBF reproduces.
THRESHOLDS = {
    "pr_over_n_max": 0.25,            # G2
    "overlap_over_random_min": 2.0,   # G2, fidelity kernels
    "offdiag_mean_min": 0.05,         # G2, other kernels
    "matched_ba_min": 0.85,           # G3
    "g_over_sqrt_n_min": 0.5,         # G4
}
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
LAMBDA = 1e-3


def minmax(X):
    """What QProfiler does to the features before a quantum arm sees them."""
    X = np.asarray(X, float)
    lo, span = X.min(0), np.ptp(X, 0)
    return (X - lo) / np.where(span > 0, span, 1.0)


def fidelity_kernel_zz(X, reps=2, entanglement="linear", data_map="qiskit", bandwidth=1.0):
    """|<psi(b x)|psi(b x')>|^2 for the ZZ feature map: qsvc's kernel."""
    from qbiocode.data_generation.quantum_core import zz_feature_state
    S = np.array([zz_feature_state(bandwidth * x, reps, entanglement, data_map) for x in X])
    return np.abs(S.conj() @ S.T) ** 2


def projected_kernel_zz(X, reps=2, entanglement="linear", data_map="qiskit", gamma=1.0):
    """RBF(gamma) on the one-qubit Bloch vectors of the ZZ state: the eng family's K_Q."""
    from qbiocode.data_generation.quantum_core import Pauli, expvals, rbf, zz_feature_state
    n = X.shape[1]
    paulis = [Pauli(n, {i: s}) for i in range(n) for s in "XYZ"]
    B = np.array([expvals(zz_feature_state(x, reps, entanglement, data_map), paulis)[0]
                  for x in X])
    return rbf(B, gamma)


def _nested_ba(K, y, seed=0):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in outer.split(K, y):
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        best, best_c = -1.0, CS[0]
        for C in CS:
            s = np.mean([balanced_accuracy_score(
                y[tr][b], SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr[a], tr[a])], y[tr][a])
                .predict(K[np.ix_(tr[b], tr[a])])) for a, b in inner.split(tr, y[tr])])
            if s > best:
                best, best_c = s, C
        model = SVC(kernel="precomputed", C=best_c).fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(balanced_accuracy_score(y[te], model.predict(K[np.ix_(te, tr)])))
    return float(np.mean(scores))


def geometric_difference(K_q, X):
    """g(K_RBF || K_q), RBF at the median-distance bandwidth, both trace-normalised."""
    n = len(X)
    sq = np.sum((X[:, None] - X[None]) ** 2, -1)
    K_c = np.exp(-sq / np.median(sq[sq > 0]))
    K_qn, K_cn = K_q * n / np.trace(K_q), K_c * n / np.trace(K_c)
    w, Q = np.linalg.eigh((K_qn + K_qn.T) / 2)
    root = (Q * np.sqrt(np.clip(w, 0, None))) @ Q.T
    M = root @ np.linalg.solve(K_cn + LAMBDA * np.eye(n), root)
    return float(np.sqrt(np.linalg.eigvalsh((M + M.T) / 2).max()))


def evaluate(X, y, K, n_train, entangling, n_qubits=None):
    """The four gates on one pilot draw: values, thresholds, pass flags, and the verdict.

    ``n_qubits`` marks ``K`` as a fidelity kernel on that many qubits, whose mean overlap
    G2 compares with 2^-n_qubits; None applies the fixed off-diagonal floor.
    """
    y = np.asarray(y).astype(int)
    off = K[~np.eye(len(K), dtype=bool)]
    pr = float(np.trace(K) ** 2 / np.sum(K ** 2))
    values = {
        "entangling_map": bool(entangling),
        "offdiag_mean": round(float(off.mean()), 6),
        "overlap_over_random": (None if n_qubits is None
                                else round(float(off.mean()) * 2.0 ** n_qubits, 6)),
        "pr_over_n": round(pr / len(K), 6),
        "matched_ba": round(_nested_ba(K, y), 6),
        "g_over_sqrt_n": round(geometric_difference(K, X) / np.sqrt(n_train), 6),
    }
    overlap_ok = (values["offdiag_mean"] >= THRESHOLDS["offdiag_mean_min"] if n_qubits is None
                  else values["overlap_over_random"] >= THRESHOLDS["overlap_over_random_min"])
    passed = {
        "G1": values["entangling_map"],
        "G2": values["pr_over_n"] <= THRESHOLDS["pr_over_n_max"] and overlap_ok,
        "G3": values["matched_ba"] >= THRESHOLDS["matched_ba_min"],
        "G4": values["g_over_sqrt_n"] >= THRESHOLDS["g_over_sqrt_n_min"],
    }
    return {"version": GATE_VERSION, "pilot_seed": PILOT_SEED, "pilot_rows": int(len(K)),
            "thresholds": dict(THRESHOLDS), "values": values, "passed": passed,
            "accepted": all(passed.values())}


def reasons(result):
    """A one-line account of the failed gates, for the skip message."""
    v, t = result["values"], result["thresholds"]
    why = {
        "G1": "matched map is a product (classical) kernel",
        "G2": (f"concentrated (PR/n {v['pr_over_n']:.3f} vs <= {t['pr_over_n_max']}; "
               + (f"off-diagonal {v['offdiag_mean']:.4f} vs >= {t['offdiag_mean_min']})"
                  if v["overlap_over_random"] is None else
                  f"overlap {v['overlap_over_random']:.2f}x random vs >= "
                  f"{t['overlap_over_random_min']})")),
        "G3": f"matched kernel learns it to {v['matched_ba']:.3f} < {t['matched_ba_min']}",
        "G4": f"g/sqrt(n) {v['g_over_sqrt_n']:.2f} < {t['g_over_sqrt_n_min']} (an RBF kernel can match it)",
    }
    return "; ".join(f"{g} {why[g]}" for g, ok in result["passed"].items() if not ok)
