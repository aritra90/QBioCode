"""compute_qpl drops the tabpfn head when the projection is wider than TabPFN accepts.

Without the guard every tuning trial raised TabPFNValidationError ("Number of features
528 ... maximum ... 500") inside the search, and the job reported only that every trial
failed. tabpfn need not be installed: the guard runs before any head is fitted, and
availability is patched so the width check is the one exercised.
"""

import importlib
import warnings

import numpy as np
import pytest

# The package re-exports the function under the module's name, so `from ... import`
# would bind the function; the module itself is what the patch must reach.
qpl_module = importlib.import_module("qbiocode.learning.compute_qpl")
from qbiocode.learning.compute_tabpfn import TABPFN_MAX_FEATURES


def _run(tmp_path, n_features, measurement, **extra):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (14, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    y[:2] = [0, 1]
    args = dict(backend="simulator", seed=42, q_seed=42, shots=1024, resil_level=1,
                average="weighted", multi_class="raise", grid_search=False,
                cross_validation=3, qpl_projection_dir=str(tmp_path),
                projection_backend="statevector")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            qpl_module.compute_qpl(X[:10], X[10:], y[:10], y[10:], args, data_key="t",
                                   encoding="Z", reps=1, measurement=measurement,
                                   classical_models=["lr", "tabpfn"], **extra)
        except Exception:
            pass   # a kept tabpfn head fails to fit here (not installed); not under test
    return [str(w.message) for w in caught if "TabPFN accepts at most" in str(w.message)]


@pytest.fixture(autouse=True)
def _tabpfn_available(monkeypatch):
    monkeypatch.setattr(qpl_module, "tabpfn_is_available", lambda: True)


def test_limit_is_v2s():
    assert TABPFN_MAX_FEATURES == 500


def test_eleven_qubit_rdm2_drops_tabpfn(tmp_path):
    msgs = _run(tmp_path, 11, "rdm2")
    assert len(msgs) == 1 and "528" in msgs[0]


@pytest.mark.parametrize("n, measurement", [(10, "rdm2"), (11, "rdm2_nn"), (11, "rdm1")])
def test_narrower_projections_keep_it(tmp_path, n, measurement):
    assert _run(tmp_path, n, measurement) == []


def test_the_waiver_keeps_it(tmp_path):
    assert _run(tmp_path, 11, "rdm2", classical_models_args={
        "tabpfn": {"ignore_pretraining_limits": True}}) == []
