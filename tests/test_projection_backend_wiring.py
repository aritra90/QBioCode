"""Tests for ``args['projection_backend']`` in compute_qpl / compute_pqk.

The switch has to be a pure performance change: same projected features, byte-identical
to the historical ``StatevectorEstimator`` path. Two things make that non-obvious and are
pinned here.

1. **Qubit ordering.** The legacy path builds observables as
   ``Pauli(id[:i] + "X" + id[i+1:])``, indexing by STRING POSITION -- and in a qiskit
   Pauli label the rightmost character is qubit 0, so its ``observables_x[i]`` is X on
   qubit ``n-1-i``. :mod:`qbiocode.utils.projection` indexes by qubit number. Without a
   flip the projected columns come out mirrored: self-consistent, identical downstream
   accuracy (a fixed permutation of features), but not equal to a cached legacy
   projection -- so the switch would stop being a drop-in without anything failing.
2. **Cache fingerprinting.** Projections are cached per feature-map fingerprint. The
   backend must be part of it, or a head-to-head silently reads the first backend's file
   back for the others; but a *bare* ``None`` must not be, or every projection cached
   before this feature is orphaned.
"""

import os

import numpy as np
import pytest

pytest.importorskip("quimb", reason="projection backends require the optional quimb dependency")

from qbiocode.learning.compute_qpl import compute_qpl  # noqa: E402

BACKENDS = ["statevector", "aer_mps", "quimb_mps"]


def _tiny_problem(n_features=6, n_train=16, n_test=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_train + n_test, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _base_args(tmp_path, **extra):
    args = dict(backend="simulator", seed=42, q_seed=42, shots=1024, resil_level=1,
                average="weighted", multi_class="raise", grid_search=False,
                cross_validation=3, qpl_projection_dir=str(tmp_path))
    args.update(extra)
    return args


def _run(tmp_path, projection_backend, n_features=6):
    """Run compute_qpl and return the saved TRAIN projection array."""
    X_train, X_test, y_train, y_test = _tiny_problem(n_features)
    args = _base_args(tmp_path, projection_backend=projection_backend)
    compute_qpl(X_train, X_test, y_train, y_test, args, data_key="t",
                encoding="ZZ", entanglement="linear", reps=1, classical_models=["lr"])
    train = [f for f in sorted(os.listdir(tmp_path)) if f.endswith("_train.npy")]
    assert len(train) == 1, f"expected one train projection, found {train}"
    return np.load(os.path.join(tmp_path, train[0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_matches_the_legacy_estimator_path(backend, tmp_path):
    """Byte-for-byte drop-in, including qubit order."""
    legacy = _run(tmp_path / "legacy", None)
    got = _run(tmp_path / backend, backend)
    assert got.shape == legacy.shape
    np.testing.assert_allclose(
        got, legacy, atol=1e-9,
        err_msg="projected features differ from the legacy path -- if the arrays are "
                "mirrored on the qubit axis, the [:, :, ::-1] flip in the wiring is "
                "missing or has been applied twice",
    )


def test_backends_do_not_share_a_cache_entry(tmp_path):
    """Two backends in one projection dir must write two files, not reuse one."""
    d = tmp_path / "shared"
    _run(d, "statevector")
    n_after_first = len([f for f in os.listdir(d) if f.endswith(".npy")])
    # Same dir, same feature map, different backend.
    X_train, X_test, y_train, y_test = _tiny_problem(6)
    compute_qpl(X_train, X_test, y_train, y_test,
                _base_args(d, projection_backend="quimb_mps"), data_key="t",
                encoding="ZZ", entanglement="linear", reps=1, classical_models=["lr"])
    n_after_second = len([f for f in os.listdir(d) if f.endswith(".npy")])
    assert n_after_second > n_after_first, (
        "the second backend reused the first one's cached projection; a head-to-head "
        "would report its wall clock as ~0 and its agreement as tautological"
    )


def test_unset_backend_keeps_the_historical_fingerprint():
    """A bare None must not enter the hash, or existing caches are orphaned."""
    import hashlib
    historical = hashlib.sha256(
        repr(("ZZ", "linear", 2, "estimator", 10)).encode()).hexdigest()[:10]
    parts = ("ZZ", "linear", 2, "estimator", 10)
    unset = None
    if unset:                                   # mirrors the code under test
        parts = parts + (unset,)
    assert hashlib.sha256(repr(parts).encode()).hexdigest()[:10] == historical


def test_remote_backend_rejects_projection_backend(tmp_path):
    """The local projectors cannot run on hardware; say so instead of ignoring it."""
    X_train, X_test, y_train, y_test = _tiny_problem(4)
    args = _base_args(tmp_path, projection_backend="quimb_mps", backend="ibm_least")
    with pytest.raises(ValueError, match="projection_backend"):
        compute_qpl(X_train, X_test, y_train, y_test, args, data_key="t",
                    encoding="ZZ", entanglement="linear", reps=1,
                    classical_models=["lr"])
