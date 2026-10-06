"""``StatevectorFidelityKernel`` is the circuit path's kernel, to the last bit.

qsvc on the statevector backend used to cost one simulated circuit per kernel entry --
46,548 of them per fit+predict+score at heart's 216/54 split, ~41 min at 13 qubits -- and
was the pilot's critical path. ``StatevectorFidelityKernel`` computes the same matrix from
one statevector per row, 147-370x faster there. That is only an acceleration, and not a
methods change, if the matrices are *identical*: same sampling noise, same diagonal, same
symmetrisation, same PSD projection. So every comparison here is ``assert_array_equal``,
never a tolerance.

The identity rests on two library internals (``StatevectorSampler`` reseeding each pub,
and how ``Generator.choice`` consumes its uniforms), so the kernel checks a few entries
against real circuits on first use and falls back when they disagree. That fallback is
tested by breaking the identity on purpose.
"""

import itertools
import logging

import numpy as np
import pytest

pytest.importorskip("qiskit_machine_learning")

from qiskit.primitives import StatevectorSampler  # noqa: E402
from qiskit_machine_learning.kernels import FidelityQuantumKernel  # noqa: E402
from qiskit_machine_learning.state_fidelities import ComputeUncompute  # noqa: E402

import qbiocode.utils.qutils as qutils  # noqa: E402
from qbiocode.learning.compute_qsvc import (  # noqa: E402
    StatevectorFidelityKernel,
    _fidelity_kernel,
    compute_qsvc,
)

#: The pilot's gridsearch_qsvc_args (experiments/pilot10/runs/*/*_qsvc.yaml).
GRID = list(itertools.product(["Z", "ZZ", "P"], [1, 2, 3, 4], ["linear", "pairwise", "full"]))


def _rows(n, q, seed):
    # Angles spread over [0, 2): fidelities land across (0, 1), where the sampling
    # noise is at its widest and a mismatch would show.
    return np.random.default_rng(seed).uniform(0, 2, size=(n, q))


def _pair(encoding="ZZ", reps=2, entanglement="linear", q=4, seed=42, shots=1024):
    fm, _ = qutils.get_feature_map(feature_map=encoding, feat_dimension=q, reps=reps,
                                   entanglement=entanglement)
    sampler = StatevectorSampler(seed=seed, default_shots=shots)
    fidelity = ComputeUncompute(sampler=sampler)
    circuits = FidelityQuantumKernel(fidelity=fidelity, feature_map=fm)
    fast = StatevectorFidelityKernel(feature_map=fm, fidelity=fidelity, shots=shots, seed=seed)
    return circuits, fast


class TestTheSameMatrix:
    @pytest.mark.parametrize("encoding,reps,entanglement", GRID,
                             ids=[f"{e}-r{r}-{t}" for e, r, t in GRID])
    def test_every_setting_the_pilot_searches(self, encoding, reps, entanglement):
        circuits, fast = _pair(encoding, reps, entanglement)
        X_tr, X_te = _rows(12, 4, 0), _rows(5, 4, 1)
        # Train Gram (the symmetric branch), then the test block (the rectangular one).
        np.testing.assert_array_equal(fast.evaluate(X_tr), circuits.evaluate(X_tr))
        np.testing.assert_array_equal(fast.evaluate(X_te, X_tr), circuits.evaluate(X_te, X_tr))
        assert fast.evaluation == "statevector", "the canary must have passed, not fallen back"

    @pytest.mark.parametrize("seed,shots", [(0, 100), (7, 4096), (np.int64(3), 1)])
    def test_other_seeds_and_shot_counts(self, seed, shots):
        circuits, fast = _pair(seed=seed, shots=shots)
        X_tr, X_te = _rows(10, 4, 2), _rows(4, 4, 3)
        np.testing.assert_array_equal(fast.evaluate(X_tr), circuits.evaluate(X_tr))
        np.testing.assert_array_equal(fast.evaluate(X_te, X_tr), circuits.evaluate(X_te, X_tr))
        assert fast.evaluation == "statevector"

    def test_y_equal_to_x_takes_the_symmetric_branch_on_both_paths(self):
        circuits, fast = _pair()
        X = _rows(9, 4, 4)
        np.testing.assert_array_equal(fast.evaluate(X, X.copy()), circuits.evaluate(X, X.copy()))
        # Before the PSD projection (whose eig round trip moves the diagonal ~1e-15, on
        # both paths alike): ones on the diagonal, the upper triangle mirrored.
        circuits._enforce_psd = fast._enforce_psd = False
        K = fast.evaluate(X, X.copy())
        np.testing.assert_array_equal(K, circuits.evaluate(X, X.copy()))
        np.testing.assert_array_equal(np.diag(K), 1.0)
        np.testing.assert_array_equal(K, K.T)

    def test_without_psd_projection(self):
        circuits, fast = _pair()
        circuits._enforce_psd = fast._enforce_psd = False
        X = _rows(10, 4, 5)
        np.testing.assert_array_equal(fast.evaluate(X), circuits.evaluate(X))

    def test_one_row(self):
        # No off-diagonal entry to check: nothing is verified yet, and the next call checks.
        circuits, fast = _pair()
        X = _rows(1, 4, 6)
        np.testing.assert_array_equal(fast.evaluate(X), circuits.evaluate(X))
        assert not fast._verified
        np.testing.assert_array_equal(fast.evaluate(_rows(6, 4, 7)),
                                      circuits.evaluate(_rows(6, 4, 7)))
        assert fast._verified and fast.evaluation == "statevector"


class TestTheCanary:
    def test_a_broken_identity_falls_back_to_circuits_and_says_so(self, caplog):
        circuits, fast = _pair()
        # What a qiskit change to per-job seeding would look like from here: different
        # uniforms from the ones the sampler actually uses.
        fast._uniforms = np.sort(np.random.default_rng(43).random(fast._shots))
        X_tr, X_te = _rows(10, 4, 8), _rows(4, 4, 9)
        with caplog.at_level(logging.WARNING, logger="qbiocode.learning.compute_qsvc"):
            K = fast.evaluate(X_tr)
        np.testing.assert_array_equal(K, circuits.evaluate(X_tr))
        assert fast.evaluation == "circuits"
        assert any("Falling back to circuits" in r.getMessage() for r in caplog.records)
        # ...for good: later calls go straight to circuits.
        np.testing.assert_array_equal(fast.evaluate(X_te, X_tr), circuits.evaluate(X_te, X_tr))

    def test_the_check_runs_once(self, monkeypatch):
        _, fast = _pair()
        calls = []
        real = fast._agrees_with_circuits
        monkeypatch.setattr(fast, "_agrees_with_circuits",
                            lambda *a: calls.append(1) or real(*a))
        X = _rows(8, 4, 10)
        fast.evaluate(X)
        fast.evaluate(_rows(3, 4, 11), X)
        fast.evaluate(X)
        assert len(calls) == 1

    def test_too_many_states_for_memory_uses_the_circuit_path(self, monkeypatch):
        circuits, fast = _pair()
        monkeypatch.setattr(fast, "MAX_STATE_BYTES", 16)
        X = _rows(6, 4, 12)
        np.testing.assert_array_equal(fast.evaluate(X), circuits.evaluate(X))


class TestWhereItApplies:
    def _fid(self, **kw):
        return ComputeUncompute(sampler=StatevectorSampler(**kw))

    @pytest.mark.parametrize("seed", [42, 0, np.int64(5)])
    def test_an_integer_seeded_statevector_sampler(self, seed):
        f = self._fid(seed=seed)
        assert StatevectorFidelityKernel.applies(f, f._sampler)

    @pytest.mark.parametrize("seed", [None, np.random.default_rng(1), True],
                             ids=["unseeded", "generator", "bool"])
    def test_not_a_seed_that_is_consumed_or_absent(self, seed):
        # None draws fresh entropy per pub; a Generator is advanced pub after pub. Either
        # way an entry's noise depends on its position in the job, not the entry alone.
        f = self._fid(seed=seed)
        assert not StatevectorFidelityKernel.applies(f, f._sampler)

    def test_not_the_local_fidelity(self):
        s = StatevectorSampler(seed=1)
        assert not StatevectorFidelityKernel.applies(ComputeUncompute(sampler=s, local=True), s)

    def test_not_a_fidelity_with_run_options_of_its_own(self):
        s = StatevectorSampler(seed=1)
        assert not StatevectorFidelityKernel.applies(
            ComputeUncompute(sampler=s, options={"shots": 64}), s)

    def test_the_config_can_force_circuits(self):
        fm, _ = qutils.get_feature_map(feature_map="Z", feat_dimension=2)
        f = self._fid(seed=1)
        assert type(_fidelity_kernel(fm, f, f._sampler, {})) is StatevectorFidelityKernel
        forced = _fidelity_kernel(fm, f, f._sampler, {"qsvc_kernel_evaluation": "circuits"})
        assert type(forced) is FidelityQuantumKernel


class TestTheModelIsUnchanged:
    """compute_qsvc end to end: same predictions, scores and metrics either way."""

    @pytest.fixture
    def data(self):
        from qbiocode import scale_train_test

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        return scale_train_test(X[:30], X[30:], scaling="MinMaxScaler") + (y[:30], y[30:])

    def _run(self, data, **extra):
        X_tr, X_te, y_tr, y_te = data
        args = {"backend": "simulator", "shots": 1024, "seed": 42, **extra}
        out = compute_qsvc(X_tr, X_te, y_tr, y_te, args, encoding="ZZ", reps=2, C=1.0)
        return out.iloc[0].to_dict()

    def test_same_outputs(self, data):
        fast = self._run(data)
        slow = self._run(data, qsvc_kernel_evaluation="circuits")
        np.testing.assert_array_equal(fast["y_predicted_qsvc"], slow["y_predicted_qsvc"])
        np.testing.assert_array_equal(fast["y_score_qsvc"], slow["y_score_qsvc"])
        rf, rs = dict(fast["results_qsvc"]), dict(slow["results_qsvc"])
        pf, ps = rf.pop("Model_Parameters"), rs.pop("Model_Parameters")
        rf.pop("time"), rs.pop("time")
        assert rf == rs
        # The record names the estimator, and separately how its matrix was computed.
        assert pf["quantum_kernel"] == ps["quantum_kernel"] == "FidelityQuantumKernel"
        assert (pf["kernel_evaluation"], ps["kernel_evaluation"]) == ("statevector", "circuits")
