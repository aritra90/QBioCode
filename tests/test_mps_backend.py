"""Tests for :mod:`qbiocode.utils.mps_backend`.

These lock in the three things an audit of the MPS bridge found worth guarding:

1. **Every row of the gate translation table is numerically correct.** The table is a
   hand-written qiskit-name -> quimb-label map with per-gate parameter counts. Only the
   handful of gates a ``ZZFeatureMap`` happens to emit are exercised by ordinary use, so
   a wrong angle sign or swapped parameter order in any other row would sit undetected
   until some future feature map emitted that gate.
2. **Parameter binding order.** ``project_row`` binds a plain array via
   ``assign_parameters``, which uses ``circuit.parameters`` order. If that view sorted
   ``a[0]..a[n-1]`` as strings, ``n >= 11`` would silently permute features. (It does
   not -- qiskit sorts ``ParameterVectorElement`` numerically -- but that is a qiskit
   implementation detail this module depends on, so it is pinned here.)
3. **Truncation is reported, and only when it happens.** ``fidelity_estimate()`` is the
   only signal that a capped ``max_bond`` produced approximate features; nothing raises.
"""

import numpy as np
import pytest

quimb = pytest.importorskip("quimb", reason="MPS backend requires the optional quimb dependency")
import quimb.tensor as qtn  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import CU1Gate, U1Gate, U2Gate, U3Gate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

from qbiocode.utils import qutils  # noqa: E402
from qbiocode.utils.mps_backend import (  # noqa: E402
    _QISKIT_TO_QUIMB,
    TRANSPILE_BASIS,
    MPSFeatureMapProjector,
    qiskit_circuit_to_quimb_gates,
)

#: Generic, mutually distinct angles. Values like pi/2 can make a wrong parameter order
#: or a flipped sign coincidentally agree.
ANGLES = (0.7123, 1.9871, 2.4567)

#: Gates needing more than one qubit, and how many.
_WIDTHS = {"ccx": 3}


def _n_qubits(name):
    if name in _WIDTHS:
        return _WIDTHS[name]
    if name.startswith("c") or name in ("swap", "iswap", "rxx", "ryy", "rzz"):
        return 2
    return 1


def _entangled_prelude(total):
    """A scrambled, entangled starting state, so no gate can look right by accident."""
    qc = QuantumCircuit(total)
    for q in range(total):
        qc.ry(0.4 + 0.3 * q, q)
        qc.rz(0.25 + 0.5 * q, q)
    for q in range(total - 1):
        qc.cx(q, q + 1)
    return qc


def _single_site_evs(state_or_circuit, total):
    sv = Statevector(state_or_circuit)
    return np.array([
        [sv.expectation_value(
            SparsePauliOp.from_sparse_list([(p, [k], 1.0)], num_qubits=total)).real
         for k in range(total)]
        for p in "XYZ"
    ])


def _quimb_evs(circuit, total):
    circ = qtn.CircuitMPS(N=total, cutoff=1e-14)
    circ.apply_gates(qiskit_circuit_to_quimb_gates(circuit))
    return np.array([
        [circ.local_expectation(quimb.pauli(p), k).real for k in range(total)]
        for p in "XYZ"
    ])


# ---------------------------------------------------------------- gate table

#: Rows whose qiskit convenience method was removed in qiskit 2.x -- they are still
#: reachable via their gate classes (and via QASM2 import), so they still need checking.
_CLASS_ONLY = {
    "u1": (U1Gate(ANGLES[0]), [0]),
    "u2": (U2Gate(ANGLES[0], ANGLES[1]), [0]),
    "u3": (U3Gate(*ANGLES), [0]),
    "cu1": (CU1Gate(ANGLES[0]), [0, 1]),
}


@pytest.mark.parametrize("name", sorted(_QISKIT_TO_QUIMB))
def test_gate_table_row_matches_qiskit(name):
    """Each translation-table row reproduces qiskit's own matrix for that gate."""
    label, n_params = _QISKIT_TO_QUIMB[name]
    total = max(3, _n_qubits(name))
    qc = _entangled_prelude(total)

    if name in _CLASS_ONLY:
        gate, qubits = _CLASS_ONLY[name]
        qc.append(gate, qubits)
    else:
        method = getattr(qc, name, None)
        assert method is not None, (
            f"table row {name!r} is neither a QuantumCircuit method nor listed in "
            f"_CLASS_ONLY; it cannot be verified and may be unreachable"
        )
        method(*ANGLES[:n_params], *range(_n_qubits(name)))

    assert qc.data[-1].operation.name == name, (
        f"expected the appended instruction to be named {name!r}, got "
        f"{qc.data[-1].operation.name!r} -- the table key does not match the "
        f"instruction name quimb will be asked to translate"
    )
    np.testing.assert_allclose(
        _quimb_evs(qc, total), _single_site_evs(qc, total), atol=1e-9,
        err_msg=f"quimb {label} disagrees with qiskit {name} -- check angle sign "
                f"and parameter order",
    )


def test_transpile_basis_names_are_all_translatable():
    """Nothing can be produced by TRANSPILE_BASIS that the table cannot translate."""
    missing = sorted(set(TRANSPILE_BASIS) - set(_QISKIT_TO_QUIMB))
    assert not missing, (
        f"TRANSPILE_BASIS may emit {missing}, which qiskit_circuit_to_quimb_gates "
        f"would reject at runtime"
    )


def test_unknown_instruction_is_rejected_with_a_useful_message():
    qc = QuantumCircuit(2)
    qc.ecr(0, 1)                      # real gate, deliberately not in the table
    with pytest.raises(ValueError, match="no quimb equivalent"):
        qiskit_circuit_to_quimb_gates(qc)


def test_unbound_parameters_are_rejected():
    fm, _ = qutils.get_feature_map("ZZ", 4, reps=1, entanglement="linear")
    with pytest.raises(ValueError, match="unbound parameter"):
        qiskit_circuit_to_quimb_gates(fm)


# ------------------------------------------------------- ordering & exactness

@pytest.mark.parametrize("n", [8, 12, 20])
def test_feature_parameters_are_indexed_in_order(n):
    """assign_parameters(array) relies on circuit.parameters being index-ordered."""
    fm, _ = qutils.get_feature_map("ZZ", n, reps=1, entanglement="linear")
    order = [int(p.name.split("[")[1][:-1]) for p in fm.parameters]
    assert order == list(range(n)), (
        "circuit.parameters is not in feature order, so binding a plain array would "
        "permute features (this bites only at n >= 11, where string sorting differs "
        "from numeric)"
    )


@pytest.mark.parametrize(
    "encoding,n,reps,entanglement",
    [("Z", 8, 1, "linear"),
     ("ZZ", 8, 1, "linear"),
     ("ZZ", 12, 1, "linear"),      # > 10: catches a string-sorted parameter order
     ("ZZ", 14, 2, "pairwise"),
     ("ZZ", 12, 1, "circular"),    # wrap-around gate is non-local in the MPS chain
     ("ZZ", 10, 1, "full"),
     ("P", 8, 1, "linear")],
)
def test_projection_is_exact_against_statevector(encoding, n, reps, entanglement):
    """Uncapped MPS projections equal the dense statevector to floating point."""
    # Strictly increasing features, so any permutation of them changes the answer.
    x = np.linspace(0.1, 3.0, n)

    proj = MPSFeatureMapProjector(n_qubits=n, encoding=encoding, reps=reps,
                                  entanglement=entanglement)
    got = proj.project_row(x)

    fm, _ = qutils.get_feature_map(encoding, n, reps=reps, entanglement=entanglement)
    # Bind by name so the reference cannot share a parameter-ordering bug.
    bound = fm.assign_parameters(
        {p: x[int(p.name.split("[")[1][:-1])] for p in fm.parameters})

    np.testing.assert_allclose(got, _single_site_evs(bound, n), atol=1e-9)
    assert proj.fidelity_estimate() == pytest.approx(1.0, abs=1e-9), (
        "an uncapped MPS truncated nothing, so fidelity must be 1.0"
    )


def test_project_shape_and_row_agreement():
    n = 8
    X = np.random.default_rng(0).uniform(0, np.pi, (5, n))
    proj = MPSFeatureMapProjector(n_qubits=n, encoding="ZZ", reps=1)
    out = proj.project(X, progress_every=None)
    assert out.shape == (5, 3, n)
    # Layout must match what compute_qpl/compute_pqk save and then flatten.
    np.testing.assert_allclose(out[2], proj.project_row(X[2]), atol=1e-12)


def test_wrong_feature_width_is_rejected():
    proj = MPSFeatureMapProjector(n_qubits=4, encoding="ZZ", reps=1)
    with pytest.raises(ValueError, match="takes 4 parameter"):
        proj.project_row([1.0, 2.0, 3.0])


# ------------------------------------------------------------- truncation

def test_truncation_is_reported_not_raised():
    """A too-small max_bond silently returns approximate features; fidelity is the tell."""
    n = 12
    x = np.linspace(0.1, 3.0, n)
    proj = MPSFeatureMapProjector(n_qubits=n, encoding="ZZ", reps=1,
                                  entanglement="full", max_bond=2)
    proj.project_row(x)                                   # must NOT raise
    assert proj.observed_max_bond() <= 2
    assert proj.fidelity_estimate() < 0.99, (
        "capping max_bond at 2 on a full-entanglement map must register as lost "
        "fidelity -- otherwise callers have no way to detect approximate features"
    )


def test_observed_max_bond_tracks_the_pattern():
    """linear/pairwise stay at bond 2; full does not. This is the whole cost model."""
    n = 10
    x = np.linspace(0.1, 3.0, n)
    bonds = {}
    for ent in ("linear", "pairwise", "full"):
        proj = MPSFeatureMapProjector(n_qubits=n, encoding="ZZ", reps=1, entanglement=ent)
        proj.project_row(x)
        bonds[ent] = proj.observed_max_bond()
    assert bonds["linear"] == 2
    assert bonds["pairwise"] == 2
    assert bonds["full"] > bonds["linear"]
