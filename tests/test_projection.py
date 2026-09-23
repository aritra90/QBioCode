"""Tests for :mod:`qbiocode.utils.projection`.

The point of this module is that five different simulators must produce the *same*
features, so a config change is a pure performance decision. These tests pin that
equivalence, the dispatch rule's boundaries, and the parallel path.
"""

import numpy as np
import pytest

pytest.importorskip("quimb", reason="projection backends require the optional quimb dependency")

from qbiocode.utils.projection import (  # noqa: E402
    AER_MAX_REPS,
    EXACT_MIN_QUBITS,
    MPS_FRIENDLY_ENTANGLEMENT,
    SUPPORTED_BACKENDS,
    AerMPSFeatureMapProjector,
    StatevectorFeatureMapProjector,
    choose_backend,
    make_projector,
)

#: Small enough to stay fast, large enough that 'full' differs from 'linear'.
N = 10

#: Strictly increasing, so any permutation of features changes the answer.
X_ROW = np.linspace(0.1, 3.0, N)


@pytest.fixture(scope="module")
def reference():
    """Dense statevector projections -- the ground truth every backend must match."""
    return StatevectorFeatureMapProjector(n_qubits=N, encoding="ZZ", reps=1,
                                         entanglement="linear").project_row(X_ROW)


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_all_backends_agree(backend, reference):
    """Switching backend must be a performance decision, never a numerical one."""
    proj = make_projector(N, encoding="ZZ", reps=1, entanglement="linear", backend=backend)
    np.testing.assert_allclose(proj.project_row(X_ROW), reference, atol=1e-9)


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_project_shape_and_row_consistency(backend):
    X = np.random.default_rng(0).uniform(0, np.pi, (4, N))
    proj = make_projector(N, encoding="ZZ", reps=1, entanglement="linear", backend=backend)
    out = proj.project(X, progress_every=None)
    assert out.shape == (4, 3, N)
    np.testing.assert_allclose(out[1], proj.project_row(X[1]), atol=1e-12)


@pytest.mark.parametrize("backend", ["statevector", "quimb_mps", "aer_mps"])
def test_parallel_matches_sequential(backend):
    """n_jobs must change only wall clock. Uses 2 workers to keep the test cheap."""
    X = np.random.default_rng(1).uniform(0, np.pi, (6, N))
    proj = make_projector(N, encoding="ZZ", reps=1, entanglement="linear", backend=backend)
    np.testing.assert_array_equal(
        proj.project(X, progress_every=None, n_jobs=1),
        proj.project(X, progress_every=None, n_jobs=2),
    )


def test_non_2d_input_is_rejected():
    proj = make_projector(N, backend="quimb_mps")
    with pytest.raises(ValueError, match="must be 2-D"):
        proj.project(X_ROW)          # a single row, not a matrix


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_wrong_feature_width_is_rejected(backend):
    proj = make_projector(4, encoding="ZZ", reps=1, backend=backend)
    with pytest.raises(ValueError):
        proj.project_row([1.0, 2.0, 3.0])


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported backend"):
        make_projector(4, backend="does_not_exist")


def test_unknown_simulator_is_rejected():
    from qbiocode.utils.mps_backend import MPSFeatureMapProjector
    with pytest.raises(ValueError, match="Unsupported simulator"):
        MPSFeatureMapProjector(n_qubits=4, simulator="tensor_soup")


# --------------------------------------------------------------- dispatch rule

@pytest.mark.parametrize("entanglement", sorted(MPS_FRIENDLY_ENTANGLEMENT))
def test_bounded_patterns_route_to_an_mps(entanglement):
    assert choose_backend(64, entanglement, reps=1) == "aer_mps"
    assert choose_backend(64, entanglement, reps=AER_MAX_REPS + 1) == "quimb_mps"


def test_reps_boundary_is_where_measured():
    """Aer up to AER_MAX_REPS, quimb past it -- the crossover section 3 measures."""
    assert choose_backend(20, "pairwise", reps=AER_MAX_REPS) == "aer_mps"
    assert choose_backend(20, "pairwise", reps=AER_MAX_REPS + 1) == "quimb_mps"


def test_full_entanglement_switches_on_width():
    """Dense while it is feasible; exact contraction once it is not (section 6)."""
    assert choose_backend(EXACT_MIN_QUBITS - 1, "full", reps=1) == "statevector"
    assert choose_backend(EXACT_MIN_QUBITS, "full", reps=1) == "quimb_exact"


def test_unknown_pattern_is_treated_as_unbounded():
    """An uncharacterised pattern must not be assumed MPS-friendly."""
    assert choose_backend(8, "some_custom_pattern", reps=1) == "statevector"


# ------------------------------------------------------------- diagnostics

def test_statevector_reports_no_truncation():
    proj = StatevectorFeatureMapProjector(n_qubits=6, encoding="ZZ", reps=1)
    assert proj.fidelity_estimate() == 1.0
    assert proj.observed_max_bond() is None


def test_aer_reports_no_fidelity():
    """Documented gap: Aer is fastest at low reps but cannot tell you it truncated."""
    proj = AerMPSFeatureMapProjector(n_qubits=6, encoding="ZZ", reps=1)
    assert proj.fidelity_estimate() is None


def test_exact_backend_never_truncates():
    proj = make_projector(8, entanglement="full", reps=1, backend="quimb_exact")
    proj.project_row(np.linspace(0.1, 3.0, 8))
    assert proj.fidelity_estimate() == 1.0


# ------------------------------------------------- parameters that must not be ignored

@pytest.mark.parametrize("backend", ["statevector", "quimb_exact"])
def test_max_bond_is_refused_where_it_cannot_apply(backend):
    """A bond-dimension cap on a backend with no bond dimension must raise, not vanish.

    Silently dropping it would let a caller believe they had requested an approximation
    and were measuring its cost, when they were measuring the exact computation.
    """
    with pytest.raises(ValueError, match="max_bond"):
        make_projector(6, backend=backend, max_bond=8)


@pytest.mark.parametrize("backend", ["quimb_mps", "quimb_permmps", "aer_mps"])
def test_max_bond_is_accepted_where_it_applies(backend):
    make_projector(6, backend=backend, max_bond=8)          # must not raise


def test_exact_simulator_refuses_max_bond():
    from qbiocode.utils.mps_backend import MPSFeatureMapProjector
    with pytest.raises(ValueError, match="max_bond"):
        MPSFeatureMapProjector(n_qubits=6, simulator="exact", max_bond=4)
