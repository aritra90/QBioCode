"""Quantum families for create_synthetic_datasets.py.

Two kinds:

- ``angle_encoding``: defined here (numpy only). The reference ``Angle_Encoding`` of
  kernel_exps/ultra_hard_datasets.py, generalised from one qubit to k.
- The five families of :mod:`qbiocode.data_generation` (exact statevector simulation):
  ground state, time evolution, Hamiltonian learning, quantum labels and engineered
  kernel. They are called through their public generators, which already balance the
  classes (median threshold) and record their diagnostics. This module maps the
  benchmark's ``(d, k)`` onto their parameters and reads their x_view CSV back.

Each family has a ``role`` and, where one applies, the arm it is matched to and a
pre-registered prediction. ``select_winners(controls=...)`` should judge the controls as
their own family, apart from the discovery datasets.

A ``positive_control`` with a ``matched_kernel`` is written only if it clears the gates of
:mod:`control_gates`, which look at the generator and that kernel alone, never at a
classical result.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import control_gates as cg

#: The prediction of a gated positive control. The gates guarantee that the matched arm
#: can learn the labels (G3); they cannot guarantee that it beats tree ensembles or
#: TabPFN, so that is measured, not predicted.
GATED_PREDICTION = ("the matched arm reaches the accuracy its gate measured (G3, within "
                    "0.05) and the tuner finds it; a win over the classical arms is measured, "
                    "not predicted")


# ---- angle encoding ------------------------------------------------------------------

def angle_encoding(n, d, k, rng, srng):
    """Product of k one-qubit fidelities with fixed target states, thresholded at its median.

    Qubit i encodes features (x_{2i}, x_{2i+1}) as
    psi_i = cos(pi x_{2i}) |0> + e^{2 pi i x_{2i+1}} sin(pi x_{2i}) |1>, and
    F = prod_i |<phi_i|psi_i>|^2 - median, phi_i = a_i |0> + b_i |1> fixed per (d, k).
    k = 1 with (a, b) = (0.8, 0.6) is the reference Angle_Encoding. Features 2k.. are
    irrelevant U(0, 1) columns. A positive control for kernels built on that one-qubit
    angle encoding (a product-state fidelity kernel). It is not matched to qsvc's ZZ map.
    """
    if 2 * k > d:
        raise ValueError(f"angle_encoding: k={k} qubits need d >= {2 * k} features, got d={d}")
    X = rng.uniform(0.0, 1.0, (n, d))
    angles = srng.uniform(0.0, np.pi / 2, k)
    a, b = np.cos(angles), np.sin(angles)
    if k == 1:
        a, b = np.array([0.8]), np.array([0.6])
    fid = np.ones(n)
    for i in range(k):
        c, s = np.cos(np.pi * X[:, 2 * i]), np.sin(np.pi * X[:, 2 * i])
        c2 = np.cos(2 * np.pi * X[:, 2 * i + 1])
        fid *= a[i] ** 2 * c ** 2 + 2 * a[i] * b[i] * c * s * c2 + b[i] ** 2 * s ** 2
    return X, fid - np.median(fid), X[:, :2 * k]


# ---- the qbiocode.data_generation families -------------------------------------------

def _qbiocode_family(kind: str, n: int, d: int, k: int, seed: int, concept_seed=None,
                     bandwidth: float = 1.0) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Run one qbiocode generator into a scratch directory and read its x_view back.

    ``concept_seed`` and ``bandwidth`` reach the ql generators only (the others draw their
    concept from ``seed`` and have no bandwidth). Returns (features + 'label' frame,
    continuous target F, the generator's meta).
    """
    from qbiocode import data_generation as dg

    with tempfile.TemporaryDirectory(prefix="qbc_synth_") as tmp:
        name = f"{kind}_tmp"
        if kind in ("gs_sparse", "gs_e2e"):
            dg.generate_ground_state_datasets(
                n_qubits=[(d + 1) // 2], n_samples=[n], label=[kind[3:]], save_path=tmp,
                name=name, random_state=seed)
        elif kind == "te":
            dg.generate_time_evolution_datasets(
                n_qubits=[d], n_samples=[n], taus=[k / 2.0], save_path=tmp, name=name,
                random_state=seed)
            name = f"{name}_tau{k / 2.0:g}"
        elif kind == "hl":
            dg.generate_hamiltonian_learning_datasets(
                n_qubits=[d // 2], n_samples=[n], times=[0.5], save_path=tmp, name=name,
                random_state=seed)
        elif kind in ("ql_zz", "ql_evo"):
            dg.generate_quantum_label_datasets(
                n_qubits=[d], n_samples=[n], encoding=[kind[3:]], tau=[k / 2.0],
                save_path=tmp, name=name, random_state=seed, concept_seed=concept_seed,
                bandwidth=bandwidth)
        elif kind in ("eng_qiskit", "eng_unit"):
            dg.generate_engineered_kernel_datasets(
                n_qubits=[d], n_samples=[n], save_path=tmp, name=name, random_state=seed,
                data_map=kind[4:])
        else:
            raise ValueError(f"unknown qbiocode family {kind!r}")
        frame = pd.read_csv(os.path.join(tmp, "x_view", f"{name}.csv"))
        F = np.load(os.path.join(tmp, "meta", f"{name}_F.npy"))
        with open(os.path.join(tmp, "meta", f"{name}.json")) as fh:
            meta = json.load(fh)
    return frame, F, meta


@dataclass(frozen=True)
class QuantumFamily:
    """One quantum family: how (d, k) map onto it, and what it is for."""

    d_min: int
    d_max: int
    uses_k: bool
    p_of_d: Callable[[int], int]            # feature count the family writes for a given d
    label_rule: str
    role: str
    matched_arm: str | None = None
    prediction: str = ""
    k_meaning: str = ""
    native: Callable | None = None           # angle_encoding; None = a qbiocode family
    oversample: int = 1
    n_max: Callable[[int], int] | None = None  # most rows the family can draw at a given d
    # The matched arm's kernel, (X, k, d, bandwidth) -> K, for the control gates.
    matched_kernel: Callable | None = None
    entangling: bool | None = None           # G1: does the matched map entangle?
    uses_bandwidth: bool = False             # takes --bandwidth (the ql generators)
    fidelity: bool = False                   # matched kernel is a fidelity kernel (G2's 2^-d)
    fixed_concept: bool = False              # one concept per (family, d, k), not per seed


QUANTUM = {
    "angle_encoding": QuantumFamily(
        2, 64, True, lambda d: d,
        "1[prod_i |<phi_i|psi_i(x_2i, x_2i+1)>|^2 > median], k one-qubit angle encodings",
        # Fails gate G1 by construction: the state is a product, so its fidelity kernel is a
        # product of one-qubit kernels with an exact classical twin. A control for kernel
        # design, not for a quantum effect.
        "product_kernel_control",
        "fidelity kernel of the one-qubit angle encoding (product state; classical twin)",
        "a kernel on that encoding >= RBF; a win here is not quantum (product kernel)",
        "k = qubits in the product (2k signal features)", native=angle_encoding,
        entangling=False),
    "gs_sparse": QuantumFamily(
        3, 23, False, lambda d: 2 * ((d + 1) // 2) - 1,
        "sparse observable of the even-sector TFIM ground state, median-thresholded",
        "classically_easy", None, "classical >= qsvc/pqk; any quantum win -> audit",
        "n = (d + 1) // 2 qubits, features J (n-1) and h (n)"),
    "gs_e2e": QuantumFamily(
        3, 23, False, lambda d: 2 * ((d + 1) // 2) - 1,
        "Z0 Z_{n-1} of the even-sector TFIM ground state, median-thresholded",
        "classically_easy", None, "a closed-form product criterion captures most of it",
        "n = (d + 1) // 2 qubits"),
    "te": QuantumFamily(
        2, 12, True, lambda d: d,
        "sparse observable after exp(-iH tau) on a basis state, median-thresholded",
        "difficulty_ladder", None, "accuracy falls as the Walsh effective degree grows",
        "tau = k / 2 (k = 2: the pilot's tau 1); d = qubits = input bits",
        n_max=lambda d: 2 ** d),           # inputs are distinct basis states
    "hl": QuantumFamily(
        4, 24, False, lambda d: 2 * (d // 2),
        "1[mean(J) - mean(h) > median] from shot-noisy quench measurements",
        "classically_easy", None, "classical >= quantum (short-time expansion gives h)",
        "n = d // 2 qubits, one time point, Z and X per qubit"),
    "ql_zz": QuantumFamily(
        2, 12, True, lambda d: d,
        "<Z0> after a random Heisenberg evolution of the ZZFeatureMap state (qiskit map, "
        "reps 2, linear), median-thresholded",
        "positive_control", "qsvc (ZZ, reps 2, linear, data_map qiskit, bandwidth b)",
        GATED_PREDICTION, "tau = k / 2 of the Heisenberg evolution",
        matched_kernel=lambda X, k, d, b: cg.fidelity_kernel_zz(X, 2, "linear", "qiskit", b),
        entangling=True, uses_bandwidth=True, fixed_concept=True, fidelity=True),
    "ql_evo": QuantumFamily(
        2, 12, True, lambda d: d,
        "<Z0> after Heisenberg evolution of an evolution-encoded state (misaligned with ZZ)",
        "negative_control", None, "classical >= quantum", "tau = k / 2",
        uses_bandwidth=True, fixed_concept=True),
    "eng_qiskit": QuantumFamily(
        2, 12, False, lambda d: d,
        "labels engineered to saturate g(K_C||K_Q) for K_Q = RBF(gamma 1) on the Bloch "
        "vectors of the ZZFeatureMap state (qiskit map, reps 2, linear)",
        "positive_control", "pqk with data_map qiskit, reps 2, linear, gamma 1",
        GATED_PREDICTION + "; qsvc is not matched", "n = d qubits; k unused",
        matched_kernel=lambda X, k, d, b: cg.projected_kernel_zz(X, 2, "linear", "qiskit", 1.0),
        entangling=True),
    "eng_unit": QuantumFamily(
        2, 12, False, lambda d: d,
        "as eng_qiskit with the unit data map",
        "positive_control", "pqk with data_map unit, reps 2, linear, gamma 1",
        GATED_PREDICTION, "n = d qubits; k unused",
        matched_kernel=lambda X, k, d, b: cg.projected_kernel_zz(X, 2, "linear", "unit", 1.0),
        entangling=True),
}
