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

"""The QELM building blocks, and what they must not change about PQK/QPL.

:mod:`qbiocode.utils.qelm_features` adds two things to a feature map -- a fixed random
reservoir and an observable pool wider than ``3n`` -- and the projectors in
:mod:`qbiocode.utils.projection` / :mod:`qbiocode.utils.mps_backend` learned to accept
both. Nothing here fits a model; that is a later phase. What these tests pin is the
part that is easy to break and hard to notice:

* **PQK and QPL are untouched.** Both pass ``observables=None`` and no
  ``extra_circuit``, so their projections must stay bit-identical in value and shape.
  A regression here is silent -- the features stay plausible and the accuracy moves a
  little -- so it is checked against the pre-existing default rather than against a
  stored constant.

* **The qubit convention.** Character ``i`` of a Pauli string acts on ``qubits[i]``.
  Qiskit's ``SparsePauliOp`` has *two* spellings that disagree about this: the
  index-based ``from_sparse_list`` (what every path here uses) and the little-endian
  label ``SparsePauliOp("XY")``, whose rightmost character is qubit 0. Mixing them
  mirrors every multi-qubit observable, which is invisible in a score: a permutation of
  feature columns leaves a rotationally symmetric readout unchanged. So the convention
  is pinned directly, on a state with ``<X_0> = 1`` and ``<Y_1> = 1``, where the two
  spellings give different answers.

* **The ablation.** A zero-depth reservoir with the ``'xyz'`` pool must reproduce the
  PQK projection *exactly*, because that cell is what isolates the readout from the
  reservoir in the planned comparison. Qiskit's ``efficient_su2(reps=0)`` still emits a
  final rotation layer, so the natural implementation of "no reservoir" silently
  measures a rotated feature map instead; :func:`qelm_reservoir` returns ``None`` for
  that reason and this test is what keeps it doing so.

* **MPS compressibility.** The reservoir is nearest-neighbour so that a QELM stays
  simulable past the ~20 qubits a dense statevector reaches. If it ever stopped being
  so, the arm would keep working and quietly lose the scaling the ``[mps]`` extra buys.
"""

from __future__ import annotations

import numpy as np
import pytest

from qbiocode.utils.qelm_features import (
    DEFAULT_MAX_WEIGHT,
    MEASUREMENTS,
    check_observables,
    default_observables,
    observables_fingerprint,
    qelm_observables,
    qelm_reservoir,
)

#: Wide enough that 'full' differs from 'linear' and weight-2 pools are non-trivial,
#: small enough to simulate in a unit test.
N = 6

#: Strictly increasing, so any permutation of the features changes the answer -- which
#: is what makes the ordering tests below able to fail.
X_ROW = np.linspace(0.1, 3.0, N)


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).uniform(0, np.pi, (4, N))


# ----------------------------------------------------------------------------------
# The observable pools
# ----------------------------------------------------------------------------------


class TestObservablePools:
    def test_default_is_the_projected_kernel_pool_in_its_order(self):
        """``3n`` single-qubit Paulis, X on every qubit then Y then Z.

        The order is what lets a flat read reshape to the ``(3, n_qubits)`` layout
        ``compute_pqk``/``compute_qpl`` save, so it is part of the contract rather than
        an implementation detail.
        """
        assert default_observables(2) == [
            ("X", (0,)), ("X", (1,)),
            ("Y", (0,)), ("Y", (1,)),
            ("Z", (0,)), ("Z", (1,)),
        ]

    def test_xyz_is_the_default_pool(self):
        assert qelm_observables(N, "xyz") == default_observables(N)

    def test_xyz_zz_adds_the_chain(self):
        pool = qelm_observables(N, "xyz+zz")
        assert pool[: 3 * N] == default_observables(N)
        assert pool[3 * N:] == [("ZZ", (k, k + 1)) for k in range(N - 1)]
        assert len(pool) == 4 * N - 1

    def test_random_pauli_is_reproducible_and_seed_sensitive(self):
        first = qelm_observables(N, "random_pauli", n_features=10, seed=1)
        assert first == qelm_observables(N, "random_pauli", n_features=10, seed=1)
        assert first != qelm_observables(N, "random_pauli", n_features=10, seed=2)

    def test_random_pauli_draws_distinct_operators(self):
        """Two features that are the same operator would be a duplicated column: a
        measurement paid for twice, and a readout handed a rank-deficient design."""
        pool = qelm_observables(N, "random_pauli", n_features=30, seed=5)
        assert len(set(pool)) == len(pool) == 30

    def test_random_pauli_respects_the_weight_cap(self):
        pool = qelm_observables(N, "random_pauli", n_features=20, seed=5, max_weight=2)
        assert max(len(pauli) for pauli, _ in pool) <= DEFAULT_MAX_WEIGHT

    @pytest.mark.parametrize("measurement", MEASUREMENTS)
    def test_every_pool_validates(self, measurement):
        extra = {"n_features": 8, "seed": 3} if measurement == "random_pauli" else {}
        pool = qelm_observables(N, measurement, **extra)
        assert check_observables(pool, N) == pool


class TestObservableValidation:
    """Every message names the observable and what was wrong with it: these are
    rejected at construction, before a single circuit is simulated."""

    @pytest.mark.parametrize(
        "spec,match",
        [
            (("I", (0,)), "Pauli character"),
            (("ZZ", (0,)), "must match"),
            (("Z", (N + 3,)), "outside"),
            (("ZZ", (1, 1)), "twice"),
            (("", ()), "empty Pauli"),
        ],
    )
    def test_malformed_specs_are_rejected(self, spec, match):
        with pytest.raises(ValueError, match=match):
            check_observables([spec], N)

    def test_a_bare_string_is_rejected_as_one(self):
        """``"ZZ"`` unpacks into ``("Z", "Z")`` -- two characters, so a pair check passes
        -- and used to fail later as ``int('Z')``: "invalid literal for int() with base
        10", naming neither the observable nor the shape it should have had."""
        with pytest.raises(ValueError, match="bare string"):
            check_observables(["ZZ"], N)

    def test_an_empty_pool_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            check_observables([], N)

    def test_random_pauli_requires_a_width(self):
        with pytest.raises(ValueError, match="needs n_features"):
            qelm_observables(N, "random_pauli", seed=1)

    def test_random_pauli_requires_a_seed(self):
        with pytest.raises(ValueError, match="seeded"):
            qelm_observables(N, "random_pauli", n_features=4)

    def test_asking_for_more_than_exists_is_refused(self):
        with pytest.raises(ValueError, match="only"):
            qelm_observables(3, "random_pauli", n_features=10_000, seed=1)

    def test_a_structured_pool_refuses_a_width_it_cannot_honour(self):
        """Truncating ``'xyz'`` would drop every Z before any X -- a different
        measurement, not a smaller one -- so the mismatch is reported instead."""
        with pytest.raises(ValueError, match="fixed pool"):
            qelm_observables(N, "xyz", n_features=5)

    def test_a_structured_pool_accepts_its_own_size(self):
        assert len(qelm_observables(N, "xyz", n_features=3 * N)) == 3 * N

    def test_an_unknown_measurement_names_the_options(self):
        with pytest.raises(ValueError, match="must be one of"):
            qelm_observables(N, "kitchen_sink")


class TestFingerprint:
    """The pool decides what a cached projection holds, so it has to be part of the key
    that names the file -- see :func:`observables_fingerprint`."""

    def test_different_pools_fingerprint_differently(self):
        assert observables_fingerprint(qelm_observables(N, "xyz")) != \
            observables_fingerprint(qelm_observables(N, "xyz+zz"))

    def test_the_fingerprint_is_order_sensitive(self):
        """Order fixes the column order of the projection, so two pools differing only
        by a permutation are different caches."""
        pool = qelm_observables(N, "xyz")
        assert observables_fingerprint(pool) != \
            observables_fingerprint(list(reversed(pool)))

    def test_the_fingerprint_is_stable_across_calls(self):
        assert observables_fingerprint(qelm_observables(N, "xyz")) == \
            observables_fingerprint(qelm_observables(N, "xyz"))

    def test_a_seeded_pool_fingerprints_by_its_seed(self):
        draw = lambda seed: observables_fingerprint(  # noqa: E731
            qelm_observables(N, "random_pauli", n_features=8, seed=seed)
        )
        assert draw(1) == draw(1)
        assert draw(1) != draw(2)


# ----------------------------------------------------------------------------------
# The reservoir
# ----------------------------------------------------------------------------------


class TestReservoir:
    def test_zero_depth_is_no_circuit_at_all(self):
        """Not ``efficient_su2(reps=0)``, which still emits a final rotation layer and
        so is a random single-qubit rotation rather than the identity. See
        :func:`qelm_reservoir`; :meth:`TestTheAblation.test_zero_depth_reproduces_pqk`
        is what that choice buys."""
        assert qelm_reservoir(N, 0, 42) is None

    def test_a_reservoir_is_fully_bound(self):
        """It is composed onto the *parameterised* feature map, so any free parameter of
        its own would have a data row bound into it by position."""
        assert qelm_reservoir(N, 2, 42).num_parameters == 0

    def test_a_reservoir_is_reproducible_from_its_seed(self):
        from qiskit.quantum_info import Operator

        np.testing.assert_allclose(
            Operator(qelm_reservoir(N, 1, 42)).data,
            Operator(qelm_reservoir(N, 1, 42)).data,
        )

    def test_a_different_seed_is_a_different_reservoir(self):
        from qiskit.quantum_info import Operator

        assert not np.allclose(
            Operator(qelm_reservoir(N, 1, 42)).data,
            Operator(qelm_reservoir(N, 1, 43)).data,
        )

    def test_depth_grows_with_reps(self):
        assert qelm_reservoir(N, 2, 1).depth() > qelm_reservoir(N, 1, 1).depth()

    @pytest.mark.parametrize("reps,seed,match", [
        (-1, 1, "non-negative"),
        (1, None, "seeded"),
    ])
    def test_bad_arguments_are_rejected(self, reps, seed, match):
        with pytest.raises(ValueError, match=match):
            qelm_reservoir(N, reps, seed)

    def test_a_non_integer_width_is_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            qelm_reservoir(0, 1, 1)

    def test_the_reservoir_is_nearest_neighbour(self):
        """What keeps a QELM MPS-simulable. An unstructured reservoir would cap the arm
        at the ~20-24 qubits a dense statevector reaches, giving up the scaling the
        optional ``[mps]`` extra buys for ``pqk``/``qpl`` -- and it would do so
        silently, since the features stay perfectly valid."""
        reservoir = qelm_reservoir(8, 2, 1)
        for instruction in reservoir.data:
            qubits = [reservoir.find_bit(q).index for q in instruction.qubits]
            if len(qubits) > 1:
                assert abs(qubits[0] - qubits[1]) == 1, (
                    f"{instruction.operation.name} acts on {qubits}, which is not a "
                    f"nearest-neighbour pair"
                )


# ----------------------------------------------------------------------------------
# The projectors
# ----------------------------------------------------------------------------------

# The projector backends other than the two qiskit ones need quimb, which is the
# optional [mps] extra -- so the OPTIONAL dependency is guarded and the first-party
# module is then imported directly. Guarding `qbiocode.utils.projection` itself instead
# would turn a real regression in it (an ImportError from a typo, a broken sibling
# import) into a skip rather than a failure; tests/test_suite_hygiene.py pins that
# distinction. `qiskit_aer` needs no guard at all: it is a core dependency
# (requirements-base.txt), so its absence is a broken install, not a configuration.
pytest.importorskip("quimb", reason="projection backends require the optional quimb dependency")

from qbiocode.utils import projection  # noqa: E402

#: Backends that need no optional dependency beyond quimb. 'aer_mps' is parametrized
#: separately so a missing qiskit_aer skips only its own nodes.
QUIMB_BACKENDS = ("statevector", "quimb_mps", "quimb_permmps", "quimb_exact")


def _make(backend, **kwargs):
    from qbiocode.utils import qutils

    return projection.make_projector(
        N, encoding="ZZ", reps=1, entanglement="linear", backend=backend,
        # The projected-kernel paths' own map; omitting it produces different features
        # from those pipelines, so the comparisons below would not be like for like.
        data_map_func=qutils.unit_coefficient_data_map, **kwargs
    )


@pytest.fixture(scope="module")
def pqk_reference(X):
    """The projection ``compute_pqk``/``compute_qpl`` compute -- default pool, no
    reservoir. The baseline every "unchanged" claim below is measured against."""
    return _make("statevector").project(X, progress_every=None)


class TestPqkAndQplAreUnchanged:
    """``observables=None`` and no ``extra_circuit`` is what both callers pass."""

    def test_the_default_row_shape_is_unchanged(self, pqk_reference, X):
        assert pqk_reference.shape == (len(X), 3, N)

    @pytest.mark.parametrize("backend", QUIMB_BACKENDS)
    def test_every_backend_still_agrees_on_the_default_pool(self, backend, X,
                                                            pqk_reference):
        np.testing.assert_allclose(
            _make(backend).project(X, progress_every=None), pqk_reference, atol=1e-9
        )

    def test_resolve_observables_keeps_the_two_dimensional_row(self):
        """The default keeps ``(3, n)`` rather than flattening, because both modules,
        their ``.npy`` caches and their width validation are written against it."""
        specs, row_shape = projection.resolve_observables(None, N)
        assert specs == default_observables(N)
        assert row_shape == (3, N)

    def test_an_explicit_pool_switches_to_a_flat_row(self):
        pool = qelm_observables(N, "xyz+zz")
        specs, row_shape = projection.resolve_observables(pool, N)
        assert specs == pool
        assert row_shape == (len(pool),)


class TestQubitOrderingConvention:
    """Character ``i`` acts on ``qubits[i]``, in every backend.

    The trap is that ``SparsePauliOp("XY")`` means the opposite (rightmost character is
    qubit 0), and that getting it backwards only permutes feature columns -- which no
    score can see. So this is pinned against a state whose two conventions disagree.
    """

    #: Two observables that are each other's mirror, so reading them under the wrong
    #: convention returns the other one's value rather than a harmless permutation.
    MIRRORED = [("XY", (0, 1)), ("YX", (0, 1))]

    #: A (reservoir seed, data row) at which ``<X_0 Y_1>`` and ``<Y_0 X_1>`` are far
    #: apart on the 2-qubit ZZ circuit below (measured: -0.497 vs +0.261). A state where
    #: they agreed would make every assertion here pass under either convention.
    SEED, ROW = 1, np.array([0.7, 2.0])

    @classmethod
    def _circuit(cls):
        """The exact circuit the projectors build: feature map, then the reservoir.

        Built independently of ``projection`` so the reference below is a second opinion
        rather than the same code path restated. Note the feature map is *not* the
        identity at any row -- the qiskit maps open with a Hadamard layer -- so the
        reference has to be taken on the real circuit.
        """
        from qbiocode.utils import qutils

        feature_map, _ = qutils.get_feature_map(
            feature_map="Z", feat_dimension=2, reps=1, entanglement="linear",
            data_map_func=qutils.unit_coefficient_data_map,
        )
        circuit = feature_map.copy()
        circuit.compose(qelm_reservoir(2, 1, cls.SEED), inplace=True)
        return circuit

    @classmethod
    def _reference(cls):
        """``[<X_0 Y_1>, <Y_0 X_1>]`` from a plain qiskit ``Statevector``."""
        from qiskit.quantum_info import SparsePauliOp, Statevector

        state = Statevector(cls._circuit().assign_parameters(cls.ROW))
        return np.array([
            state.expectation_value(
                SparsePauliOp.from_sparse_list([(pauli, list(qubits), 1.0)],
                                               num_qubits=2)
            ).real
            for pauli, qubits in cls.MIRRORED
        ])

    def test_the_two_conventions_really_disagree_here(self):
        """The premise of every assertion below, pinned so it cannot quietly lapse.

        ``from_sparse_list`` is index-based; a bare ``SparsePauliOp("XY")`` label is
        little-endian and means the mirrored operator. If those two ever returned the
        same number on this circuit, the tests below would pass under either convention
        and would be protecting nothing.
        """
        from qiskit.quantum_info import SparsePauliOp, Statevector

        state = Statevector(self._circuit().assign_parameters(self.ROW))
        index_based = state.expectation_value(
            SparsePauliOp.from_sparse_list([("XY", [0, 1], 1.0)], num_qubits=2)
        ).real
        little_endian = state.expectation_value(SparsePauliOp("XY")).real
        assert abs(index_based - little_endian) > 0.3
        # And the index-based reading is the one that agrees with the mirrored spec,
        # which is what makes the reference above meaningful.
        assert index_based == pytest.approx(self._reference()[0])

    @pytest.mark.parametrize("backend", QUIMB_BACKENDS)
    def test_every_backend_reads_a_multi_qubit_observable_the_same_way(self, backend):
        """Each backend must match the independent qiskit reference, not just itself.

        A mirrored operator in any one backend swaps these two values, which this
        catches and a cross-backend comparison alone would not -- if every backend
        mirrored, they would still agree with each other.
        """
        from qbiocode.utils import qutils

        projector = projection.make_projector(
            2, encoding="Z", reps=1, entanglement="linear", backend=backend,
            data_map_func=qutils.unit_coefficient_data_map,
            observables=self.MIRRORED,
            extra_circuit=qelm_reservoir(2, 1, self.SEED),
        )
        np.testing.assert_allclose(
            projector.project_row(self.ROW), self._reference(), atol=1e-9
        )


class TestExplicitPoolsAgreeAcrossBackends:
    """An observable pool must be a measurement choice, never a backend choice."""

    @pytest.mark.parametrize("measurement", MEASUREMENTS)
    @pytest.mark.parametrize("backend", QUIMB_BACKENDS)
    def test_pools_agree_with_the_statevector(self, backend, measurement, X):
        extra = {"n_features": 12, "seed": 3} if measurement == "random_pauli" else {}
        pool = qelm_observables(N, measurement, **extra)
        reference = _make("statevector", observables=pool).project(X, progress_every=None)
        np.testing.assert_allclose(
            _make(backend, observables=pool).project(X, progress_every=None),
            reference, atol=1e-9,
        )

    @pytest.mark.parametrize("backend", QUIMB_BACKENDS)
    def test_pools_agree_with_a_reservoir_attached(self, backend, X):
        """The reservoir is composed before each backend's own basis translation, so
        this is also the check that every translation received it."""
        pool = qelm_observables(N, "xyz+zz")
        reservoir = qelm_reservoir(N, 2, 7)
        reference = _make(
            "statevector", observables=pool, extra_circuit=reservoir
        ).project(X, progress_every=None)
        np.testing.assert_allclose(
            _make(backend, observables=pool, extra_circuit=reservoir)
            .project(X, progress_every=None),
            reference, atol=1e-9,
        )

    @pytest.mark.parametrize("backend", QUIMB_BACKENDS)
    def test_an_explicit_pool_gives_a_flat_matrix(self, backend, X):
        pool = qelm_observables(N, "random_pauli", n_features=9, seed=2)
        out = _make(backend, observables=pool).project(X, progress_every=None)
        assert out.shape == (len(X), 9)


class TestTheAblation:
    """What the planned PQK-vs-QELM comparison rests on."""

    def test_zero_depth_reproduces_pqk(self, pqk_reference, X):
        """A zero-depth reservoir with the ``'xyz'`` pool is the PQK projection, flat.

        Exactly -- not approximately. This is the cell that isolates the *readout* from
        the *reservoir*: if it drifted, "reservoir adds nothing" and "the ridge readout
        is weaker than the SVC" would be indistinguishable in the results. It holds
        because :func:`qelm_reservoir` returns ``None`` at depth 0 rather than a qiskit
        ``reps=0`` circuit, which still carries a rotation layer.
        """
        out = _make(
            "statevector",
            observables=qelm_observables(N, "xyz"),
            extra_circuit=qelm_reservoir(N, 0, 42),
        ).project(X, progress_every=None)
        np.testing.assert_array_equal(out, pqk_reference.reshape(len(X), -1))

    def test_a_reservoir_changes_the_features(self, X):
        """Otherwise the reservoir row of the ablation would be measuring nothing."""
        pool = qelm_observables(N, "xyz")
        without = _make("statevector", observables=pool).project(X, progress_every=None)
        with_reservoir = _make(
            "statevector", observables=pool, extra_circuit=qelm_reservoir(N, 1, 42)
        ).project(X, progress_every=None)
        assert not np.allclose(without, with_reservoir)

    def test_the_same_seed_gives_the_same_features(self, X):
        pool = qelm_observables(N, "xyz")
        project = lambda seed: _make(  # noqa: E731
            "statevector", observables=pool, extra_circuit=qelm_reservoir(N, 1, seed)
        ).project(X, progress_every=None)
        np.testing.assert_array_equal(project(42), project(42))
        assert not np.allclose(project(42), project(43))


class TestExtraCircuitValidation:
    def test_an_unbound_extra_circuit_is_refused(self):
        """It is composed onto the parameterised feature map and qiskit binds by
        position, so a row would be split between the map and the reservoir -- and the
        reservoir would stop being fixed."""
        from qiskit.circuit.library import efficient_su2

        with pytest.raises(ValueError, match="unbound"):
            _make("statevector", extra_circuit=efficient_su2(N, reps=1))

    def test_a_narrower_extra_circuit_is_refused(self):
        """Composing it would leave the remaining qubits untouched rather than fail,
        which is a different circuit and not an error."""
        with pytest.raises(ValueError, match="qubit"):
            _make("statevector", extra_circuit=qelm_reservoir(N - 1, 1, 3))


class TestMpsScaling:
    """The premise of the ``[mps]`` extra, which the reservoir must not break."""

    @pytest.mark.parametrize("n_qubits", [8, 14])
    def test_the_bond_dimension_stays_flat_in_width(self, n_qubits):
        """A compressible state has a bond dimension that does not grow with ``n``.

        Measured here against ``2**(n//2)``, the uncompressed worst case: at n=8 and
        n=14 the observed bond stays in single digits, so a QELM keeps the linear-in-``n``
        cost that lifts the ~20-feature statevector ceiling. An unstructured reservoir
        would fail this while still producing valid features.
        """
        from qbiocode.utils import qutils

        rows = np.random.default_rng(1).uniform(0, np.pi, (2, n_qubits))
        projector = projection.make_projector(
            n_qubits, encoding="ZZ", reps=1, entanglement="linear", backend="quimb_mps",
            data_map_func=qutils.unit_coefficient_data_map,
            observables=qelm_observables(n_qubits, "xyz+zz"),
            extra_circuit=qelm_reservoir(n_qubits, 2, 7),
        )
        projector.project(rows, progress_every=None)
        assert projector.observed_max_bond() < 2 ** (n_qubits // 2)
        # Exact: nothing was truncated, so these features are not an approximation.
        assert projector.fidelity_estimate() == pytest.approx(1.0, abs=1e-9)


class TestAerBackend:
    """``aer_mps`` kept apart, because its measurement path is a different one.

    Its ``save_expectation_value`` is the one place a bare ``SparsePauliOp`` label would
    have been the natural thing to write, and the one place that would have been
    little-endian -- so of all the backends, the ordering check matters here most.

    No ``importorskip``: ``qiskit-aer`` is a core dependency, so its absence is a broken
    install rather than a configuration to skip around.
    """

    def test_the_default_pool_is_unchanged(self, X, pqk_reference):
        np.testing.assert_allclose(
            _make("aer_mps").project(X, progress_every=None), pqk_reference, atol=1e-9
        )

    @pytest.mark.parametrize("measurement", MEASUREMENTS)
    def test_pools_agree_with_the_statevector(self, measurement, X):
        extra = {"n_features": 12, "seed": 3} if measurement == "random_pauli" else {}
        pool = qelm_observables(N, measurement, **extra)
        np.testing.assert_allclose(
            _make("aer_mps", observables=pool).project(X, progress_every=None),
            _make("statevector", observables=pool).project(X, progress_every=None),
            atol=1e-9,
        )

    def test_it_reads_a_multi_qubit_observable_index_first(self):
        """Against the same independent reference the other backends are held to.

        This path is where a bare ``SparsePauliOp`` label would have been the natural
        thing to write for ``save_expectation_value``, and it would have been
        little-endian -- so of all the backends, this is the one the check is for.
        """
        from qbiocode.utils import qutils

        convention = TestQubitOrderingConvention
        projector = projection.make_projector(
            2, encoding="Z", reps=1, entanglement="linear", backend="aer_mps",
            data_map_func=qutils.unit_coefficient_data_map,
            observables=convention.MIRRORED,
            extra_circuit=qelm_reservoir(2, 1, convention.SEED),
        )
        np.testing.assert_allclose(
            projector.project_row(convention.ROW), convention._reference(), atol=1e-9
        )

    def test_observables_sharing_a_pauli_string_are_not_collapsed(self, X):
        """Aer's result dict is keyed by label, so labelling by the Pauli string would
        silently merge ``ZZ`` on (0,1) with ``ZZ`` on (1,2) into one saved value."""
        pool = [("ZZ", (0, 1)), ("ZZ", (1, 2))]
        out = _make("aer_mps", observables=pool).project(X, progress_every=None)
        assert out.shape == (len(X), 2)
        np.testing.assert_allclose(
            out, _make("statevector", observables=pool).project(X, progress_every=None),
            atol=1e-9,
        )
