"""The 2-RDM measurement for compute_pqk / compute_qpl.

Four properties, each built so that the obvious wrong implementation fails it:

1. **The 2-RDM features are the reduced state.** Each two-body column is checked against
   ``Tr(rho_ij (P x Q))`` computed independently by partial-tracing the full statevector,
   not against the projector's own code -- which would only test it against itself.
2. **The 2-RDM adds columns and changes none.** The first ``3n`` columns of a 2-RDM
   projection equal the 1-RDM projection, entry for entry.
3. **rdm1 is the shipped behaviour, byte for byte**, including its cache file name, so
   every projection already on disk keeps serving.
4. **The 2-RDM is refused where it cannot be measured** -- the hand-built estimator path,
   which builds single-qubit observables only.
"""

import os

import numpy as np
import pytest

from qbiocode.learning.compute_pqk import compute_pqk
from qbiocode.learning.compute_qpl import compute_qpl
from qbiocode.utils import qutils
from qbiocode.utils.qelm_features import RDM_MEASUREMENTS, qelm_observables

N = 4  # qubits: small enough to partial-trace by hand, big enough to have non-neighbours


def _problem(n_features=N, n_train=10, n_test=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_train + n_test, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    y[:2] = [0, 1]                       # both classes in train, whatever the draw
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _args(tmp_path, **extra):
    args = dict(backend="simulator", seed=42, q_seed=42, shots=1024, resil_level=1,
                average="weighted", multi_class="raise", grid_search=False,
                cross_validation=3, qpl_projection_dir=str(tmp_path),
                pqk_projection_dir=str(tmp_path), projection_backend="statevector")
    args.update(extra)
    return args


def _projection(tmp_path, measurement, fn=compute_qpl, **extra):
    X_train, X_test, y_train, y_test = _problem()
    kwargs = dict(data_key="t", encoding="ZZ", entanglement="linear", reps=1,
                  measurement=measurement)
    if fn is compute_qpl:
        kwargs["classical_models"] = ["lr"]
    fn(X_train, X_test, y_train, y_test, _args(tmp_path, **extra), **kwargs)
    train = [f for f in sorted(os.listdir(tmp_path)) if f.endswith("_train.npy")]
    assert len(train) == 1, train
    return np.load(os.path.join(tmp_path, train[0])), X_train, train[0]


# ---------------------------------------------------------------------------
# 1. The pools
# ---------------------------------------------------------------------------

class TestPools:
    @pytest.mark.parametrize("n, nn, full", [(2, 15, 15), (4, 39, 66), (8, 87, 276)])
    def test_sizes(self, n, nn, full):
        assert len(qelm_observables(n, "rdm2_nn")) == nn == 3 * n + 9 * (n - 1)
        assert len(qelm_observables(n, "rdm2")) == full == 3 * n + 9 * n * (n - 1) // 2

    @pytest.mark.parametrize("pool", ["rdm2", "rdm2_nn"])
    def test_starts_with_the_1rdm_block_in_its_order(self, pool):
        assert qelm_observables(N, pool)[:3 * N] == qelm_observables(N, "xyz")

    def test_every_pair_carries_all_nine_correlators_once(self):
        two_body = [spec for spec in qelm_observables(N, "rdm2") if len(spec[1]) == 2]
        assert len(two_body) == len(set(two_body)) == 9 * N * (N - 1) // 2
        pairs = {qubits for _, qubits in two_body}
        assert len(pairs) == N * (N - 1) // 2
        for qubits in pairs:
            assert sorted(p for p, q in two_body if q == qubits) == sorted(
                a + b for a in "XYZ" for b in "XYZ")

    def test_nn_pool_is_neighbours_only(self):
        pairs = {q for _, q in qelm_observables(N, "rdm2_nn") if len(q) == 2}
        assert pairs == {(k, k + 1) for k in range(N - 1)}

    def test_rdm1_is_the_shipped_pool(self):
        assert RDM_MEASUREMENTS["rdm1"] == "xyz"


# ---------------------------------------------------------------------------
# 2. The values are the reduced state
# ---------------------------------------------------------------------------

def _paulis():
    return {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}


def _reduced(psi, keep, n):
    """rho of qubits ``keep`` (sorted), by partial trace of |psi><psi|.

    qiskit's statevector is little-endian: amplitude index bit k is qubit k, so the
    reshaped tensor's axis for qubit k is ``n - 1 - k``.
    """
    t = psi.reshape([2] * n)
    axes = [n - 1 - k for k in keep]
    rest = [a for a in range(n) if a not in axes]
    t = np.transpose(t, axes + rest).reshape(2 ** len(keep), -1)
    return t @ t.conj().T                    # rows: keep[0] is the MOST significant


def _expectation(psi, pauli, qubits, n):
    """<psi| P_qubits |psi> from the reduced density matrix of those qubits."""
    order = sorted(range(len(qubits)), key=lambda i: qubits[i])
    keep = [qubits[i] for i in order]
    ops = [_paulis()[pauli[i]] for i in order]
    op = ops[0]
    for o in ops[1:]:
        op = np.kron(op, o)
    return float(np.real(np.trace(_reduced(psi, keep, n) @ op)))


class TestValuesAreTheReducedState:
    def test_every_rdm2_column_matches_an_independent_partial_trace(self, tmp_path):
        from qiskit.quantum_info import Statevector

        proj, X_train, _ = _projection(tmp_path, "rdm2")
        pool = qelm_observables(N, "rdm2")
        assert proj.shape == (len(X_train), len(pool))

        feature_map, _ = qutils.get_feature_map(
            "ZZ", N, reps=1, entanglement="linear",
            data_map_func=qutils.unit_coefficient_data_map)
        for row in range(3):
            psi = Statevector(feature_map.assign_parameters(X_train[row])).data
            want = [_expectation(psi, p, q, N) for p, q in pool]
            np.testing.assert_allclose(
                proj[row], want, atol=1e-9,
                err_msg="a 2-RDM column is not the expectation of its operator on the "
                        "reduced state -- check the qubit order of two-body terms")

    def test_an_entangling_map_has_correlators_its_marginals_do_not_explain(self, tmp_path):
        """If every correlator factorised, the 2-RDM would carry nothing the 1-RDM does
        not. Checked over ALL nine, not ZZ alone: a ZZ feature map at reps=1 applies only
        exp(i phi Z_i Z_j) between qubits, which commutes with Z_i Z_j, so <Z_i Z_j> does
        factorise exactly there -- while XZ, YZ, ZX, ... are strongly correlated."""
        proj, _, _ = _projection(tmp_path, "rdm2")
        pool = qelm_observables(N, "rdm2")
        one = {(p, q[0]): c for c, (p, q) in enumerate(pool) if len(q) == 1}
        gaps = [np.abs(proj[:, c] - proj[:, one[(p[0], q[0])]] * proj[:, one[(p[1], q[1])]]).max()
                for c, (p, q) in enumerate(pool) if len(q) == 2]
        assert max(gaps) > 0.05

    def test_the_Z_encoding_is_a_product_state_so_its_2rdm_is_redundant(self, tmp_path):
        """ZFeatureMap has no entangling gate, so every correlator is the product of its
        marginals at any reps. Recorded as a test because it bears on experiment design:
        measurement=rdm2 with encoding=Z adds columns that are exact functions of the
        rdm1 columns, so a search pairing them is spending trials on nothing."""
        X_train, X_test, y_train, y_test = _problem()
        compute_qpl(X_train, X_test, y_train, y_test, _args(tmp_path), data_key="t",
                    encoding="Z", entanglement="linear", reps=2,
                    classical_models=["lr"], measurement="rdm2")
        f = [x for x in os.listdir(tmp_path) if x.endswith("_train.npy")][0]
        proj = np.load(os.path.join(tmp_path, f))
        pool = qelm_observables(N, "rdm2")
        one = {(p, q[0]): c for c, (p, q) in enumerate(pool) if len(q) == 1}
        for c, (p, q) in enumerate(pool):
            if len(q) == 2:
                np.testing.assert_allclose(
                    proj[:, c], proj[:, one[(p[0], q[0])]] * proj[:, one[(p[1], q[1])]],
                    atol=1e-10)


# ---------------------------------------------------------------------------
# 3. rdm1 is unchanged; the 2-RDM only adds
# ---------------------------------------------------------------------------

class TestCompatibility:
    @pytest.mark.parametrize("pool", ["rdm2", "rdm2_nn"])
    def test_rdm2_prefix_equals_rdm1(self, tmp_path, pool):
        one, _, _ = _projection(tmp_path / "one", "rdm1")
        two, _, _ = _projection(tmp_path / "two", pool)
        # rdm1 is saved (n, 3, N) and qubit-flipped to match the legacy path; flattening
        # it gives X on qubits N-1..0, then Y, then Z. Undo the flip to compare.
        one_flat = one[:, :, ::-1].reshape(len(one), -1)
        np.testing.assert_allclose(two[:, :3 * N], one_flat, atol=1e-12)

    def test_rdm1_keeps_the_shipped_cache_name(self, tmp_path):
        """The default must not change the fingerprint, or every cached projection on
        disk is orphaned."""
        _, _, default_name = _projection(tmp_path / "a", "rdm1")
        X_train, X_test, y_train, y_test = _problem()
        compute_qpl(X_train, X_test, y_train, y_test, _args(tmp_path / "b"),
                    data_key="t", encoding="ZZ", entanglement="linear", reps=1,
                    classical_models=["lr"])           # no measurement= at all
        legacy = [f for f in os.listdir(tmp_path / "b") if f.endswith("_train.npy")]
        assert legacy == [default_name]

    def test_measurements_never_share_a_cache_file(self, tmp_path):
        names = {m: _projection(tmp_path / m, m)[2] for m in RDM_MEASUREMENTS}
        assert len(set(names.values())) == 3

    @pytest.mark.parametrize("fn", [compute_pqk, compute_qpl])
    def test_both_arms_take_it(self, tmp_path, fn):
        proj, X_train, _ = _projection(tmp_path, "rdm2_nn", fn=fn)
        assert proj.shape == (len(X_train), 3 * N + 9 * (N - 1))


# ---------------------------------------------------------------------------
# 4. Refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_rdm2_needs_projection_backend(self, tmp_path):
        X_train, X_test, y_train, y_test = _problem()
        with pytest.raises(ValueError, match="needs projection_backend"):
            compute_qpl(X_train, X_test, y_train, y_test,
                        _args(tmp_path, projection_backend=None), data_key="t",
                        encoding="ZZ", reps=1, classical_models=["lr"],
                        measurement="rdm2")

    def test_unknown_measurement_is_named(self, tmp_path):
        X_train, X_test, y_train, y_test = _problem()
        with pytest.raises(ValueError, match="measurement must be one of"):
            compute_pqk(X_train, X_test, y_train, y_test, _args(tmp_path),
                        data_key="t", encoding="ZZ", reps=1, measurement="rdm3")
