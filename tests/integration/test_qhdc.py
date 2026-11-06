"""Real hdlib/Aer runs; no IBM account or network is used by these tests."""

import numpy as np
import pytest

import qbiocode
from .conftest import run_qprofiler

pytestmark = pytest.mark.usefixtures("qhdc_ready")


def test_real_simulator_dispatch_and_reproducibility():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [0.9, 0.8], [1.0, 1.0]] * 3)
    y = np.array([0, 0, 1, 1] * 3)
    args = {
        "backend": "simulator",
        "seed": 7,
        "q_seed": 11,
        "shots": 128,
        "model": ["qhdc", "lr"],
        "n_jobs": 1,
        "grid_search": False,
        "qhdc_args": {"size": 8, "levels": 2, "retrain_epochs": 1},
    }
    a = qbiocode.model_run(X[:8], X[8:], y[:8], y[8:], "qhdc-test", args)
    b = qbiocode.model_run(X[:8], X[8:], y[:8], y[8:], "qhdc-test", args)
    for name in ("lr", "qhdc"):
        row = a[f"results_{name}"][0]
        assert all(0 <= row[m] <= 1 for m in ("accuracy", "f1_score", "auc"))
        np.testing.assert_array_equal(a[f"y_test_{name}"][0], y[8:])
        np.testing.assert_array_equal(a[f"y_predicted_{name}"][0], b[f"y_predicted_{name}"][0])
    assert a["results_qhdc"][0]["auc"] == b["results_qhdc"][0]["auc"]


def test_real_optuna_refit():
    X = np.array([[0.0, 0.0], [0.1, 0.2], [0.9, 0.8], [1.0, 1.0]] * 4)
    y = np.array([0, 0, 1, 1] * 4)
    frame = qbiocode.compute_qhdc_opt(
        X[:12],
        X[12:],
        y[:12],
        y[12:],
        {"backend": "simulator", "seed": 7, "shots": 64},
        size=[8],
        levels=[2, 3],
        n_trials=2,
    )
    row = frame["results_qhdc_opt"][0]
    assert row["BestParams_Tuned"]["size"] == 8
    assert np.isfinite(row["auc"])
    np.testing.assert_array_equal(frame["y_test_qhdc_opt"][0], y[12:])


def test_qprofiler_writes_qhdc_alongside_classical_model(tmp_path):
    rows = run_qprofiler(
        tmp_path,
        [
            "model=[lr,qhdc]",
            "embeddings=[none,pca]",
            "embedding_min_features=0",
            "n_components=2",
            "iter=1",
            "n_jobs=2",
            "grid_search=False",
            "tune_quantum=False",
            "shots=64",
            "qhdc_args.size=8",
            "qhdc_args.levels=2",
        ],
    )
    assert len(rows) == 4
    assert set(rows["model"]) == {"lr", "qhdc"}
    assert set(rows["embeddings"]) == {"none", "pca"}
    assert np.isfinite(rows[["accuracy", "f1_score", "auc", "time"]].to_numpy()).all()
    assert rows.loc[rows.model == "qhdc", "Model_Parameters"].str.contains("hdlib_version").all()
