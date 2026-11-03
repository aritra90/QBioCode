"""qHDC's split, metric, backend and tuning contracts, without hardware calls."""

import builtins
import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import qbiocode

qhdc = importlib.import_module("qbiocode.learning.compute_qhdc")


@pytest.fixture
def data():
    # Test extrema deliberately exceed the training range. Fitting the scaler
    # on both sets would change every training coordinate and fail this test.
    return (
        np.array([[-4.0, 10.0], [-2.0, 11.0], [0.0, 13.0], [2.0, 14.0]]),
        np.array([[10.0, 12.0], [-20.0, 15.0], [1.0, 11.0]]),
        np.array([0, 1, 0, 1]),
        np.array([1, 0, 1]),
    )


@pytest.fixture
def recording_model(monkeypatch):
    instances = []

    class Model:
        version = "test-hdlib"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.retraining = None
            instances.append(self)

        def fit(self, X, y):
            self.train = np.array(X)
            self.labels = np.array(y)
            self.classes_ = [1, 0]  # Reverse order catches a hardcoded score column.

        def retrain(self, X, y, **kwargs):
            self.retraining = (np.array(X), np.array(y), kwargs)

        def predict(self, X):
            self.test = np.array(X)
            positive = np.linspace(0.2, 0.9, len(X))
            negative = np.linspace(0.8, 0.1, len(X))
            self.similarities = np.column_stack([positive, negative])
            return np.array(self.classes_)[self.similarities.argmax(axis=1)], self.similarities

    monkeypatch.setattr(qhdc, "_load_classifier", lambda: Model)
    monkeypatch.setattr(qhdc, "_backend", lambda *args: SimpleNamespace(name="fake"))
    return instances


def test_preserves_split_and_fits_scaler_only_on_training(data, recording_model):
    originals = [x.copy() for x in data]
    frame = qhdc.compute_qhdc(*data, {"q_seed": 19, "shots": 256}, retrain_epochs=2)
    fitted = recording_model[0]
    expected = MinMaxScaler(clip=True).fit(data[0])
    np.testing.assert_allclose(fitted.train, expected.transform(data[0]))
    np.testing.assert_allclose(fitted.test, expected.transform(data[1]))
    np.testing.assert_array_equal(fitted.labels, data[2])
    for before, after in zip(originals, data):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(frame["y_test_qhdc"][0], data[3])
    assert fitted.kwargs["seed"] == 19 and fitted.kwargs["shots"] == 256
    train, labels, kwargs = fitted.retraining
    np.testing.assert_array_equal(train, fitted.train)
    np.testing.assert_array_equal(labels, data[2])
    assert set(kwargs) == {"epochs", "lr", "early_stop", "verbose"}


def test_auc_uses_similarity_margin_and_class_order(data, recording_model):
    frame = qhdc.compute_qhdc(*data, {})
    similarities = recording_model[0].similarities
    expected = roc_auc_score(data[3], similarities[:, 0] - similarities[:, 1])
    row = frame["results_qhdc"][0]
    assert row["auc"] == expected
    assert {"model", "accuracy", "f1_score", "auc", "time", "Model_Parameters"} <= set(row)
    assert recording_model[0].retraining is None


def test_missing_hdlib_explains_source_install(monkeypatch):
    original = builtins.__import__

    def without_hdlib(name, *args, **kwargs):
        if name.startswith("hdlib"):
            raise ImportError("hdlib absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hdlib)
    message = "requires Python" if sys.version_info < (3, 11) else "requirements-qhdc.txt"
    with pytest.raises(ImportError, match=message):
        qhdc._load_classifier()


def test_package_import_does_not_import_hdlib():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, qbiocode; assert 'hdlib' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"size": 12}, "power of two"),
        ({"size": True}, "size"),
        ({"levels": 1}, "levels"),
        ({"retrain_epochs": -1}, "retrain_epochs"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"input_scaling": "none"}, "all features in"),
    ],
)
def test_invalid_inputs_fail_before_constructing_model(data, recording_model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        qhdc.compute_qhdc(*data, {}, **kwargs)
    assert not recording_model


@pytest.mark.parametrize("name", ["ibm_example", "ibm_least"])
def test_backend_reuses_named_account_and_config_overrides(tmp_path, monkeypatch, name):
    account = importlib.import_module("qbiocode.utils.ibm_account")
    path = tmp_path / "accounts.json"
    path.write_text(
        json.dumps(
            {
                "chosen": {
                    "token": "file-secret",
                    "channel": "ibm_quantum_platform",
                    "instance": "file-instance",
                },
                "other": {"token": "wrong-secret"},
            }
        )
    )
    seen = {}
    backend = SimpleNamespace(num_qubits=127)

    class Service:
        def __init__(self, **kwargs):
            seen["credentials"] = kwargs

        def backend(self, **kwargs):
            seen["backend"] = kwargs
            return backend

        def least_busy(self, **kwargs):
            seen["least_busy"] = kwargs
            return backend

    monkeypatch.setattr(account, "QiskitRuntimeService", Service)
    args = dict(backend=name, qiskit_json_path=str(path), name="chosen", ibm_instance="override")
    assert qhdc._backend(args, 6, "CPU") is backend
    assert seen["credentials"]["token"] == "file-secret"
    assert seen["credentials"]["instance"] == "override"
    assert seen.get("backend", {}).get("name", name) == name
    if name == "ibm_least":
        assert seen["least_busy"]["min_num_qubits"] == 6


def test_credentials_do_not_enter_metrics(data, recording_model):
    frame = qhdc.compute_qhdc(*data, {"ibm_token": "do-not-record"})
    assert "do-not-record" not in str(frame.to_dict())


def test_tuning_never_sees_outer_test_rows_and_keeps_execution_settings(recording_model):
    X = np.arange(40, dtype=float).reshape(20, 2) / 40
    y = np.tile([0, 1], 10)
    test = np.array([[0.99, 0.98], [0.97, 0.96]])
    args = {
        "backend": "simulator",
        "seed": 3,
        "q_seed": 17,
        "model": ["qhdc"],
        "n_jobs": 1,
        "grid_search": True,
        "tune_quantum": True,
        "n_trials_quantum": 2,
        "qhdc_args": {"size": 8, "input_scaling": "none", "early_stop": False, "seed": 23},
        "gridsearch_qhdc_args": {"levels": [2, 3]},
    }
    result = qbiocode.model_run(X, test, y, np.array([1, 0]), "same-split", args)
    assert len(recording_model) == 3  # Two trials and the final refit.
    for trial in recording_model[:-1]:
        assert len(trial.train) < len(X)
        assert all(any(np.array_equal(row, original) for original in X) for row in trial.test)
        assert trial.kwargs["size"] == 8
        assert trial.kwargs["seed"] == 23
    final = recording_model[-1]
    assert final.kwargs["seed"] == 23
    np.testing.assert_array_equal(final.train, X)
    np.testing.assert_array_equal(final.test, test)
    row = result["results_qhdc_opt"][0]
    assert row["BestParams_Tuned"]["size"] == 8
    assert row["BestParams_Tuned"]["input_scaling"] == "none"
    assert row["BestParams_Tuned"]["early_stop"] is False


@pytest.mark.parametrize("grid_search", [False, True])
def test_quantum_tuning_off_dispatches_untuned_qhdc(data, recording_model, grid_search):
    args = {
        "model": ["qhdc"],
        "n_jobs": 1,
        "grid_search": grid_search,
        "tune_quantum": False,
        "seed": 7,
        "qhdc_args": {"seed": 29},
    }
    result = qbiocode.model_run(*data, "split", args)
    assert "results_qhdc" in result
    assert len(recording_model) == 1
    assert recording_model[0].kwargs["seed"] == 29


def test_hardware_tuning_requires_explicit_opt_in(data):
    with pytest.raises(ValueError, match="allow_hardware_tuning"):
        qhdc.compute_qhdc_opt(*data, {"backend": "ibm_example"}, size=[8])


@pytest.mark.parametrize("label", ["qhdc", "qhdc_opt"])
def test_qhdc_is_counted_as_a_quantum_winner(tmp_path, label):
    from qbiocode.utils.qc_winner_finder import qml_winner

    results = pd.DataFrame(
        [
            {
                "Dataset": "tiny.csv",
                "embeddings": "none",
                "model": name,
                "f1_score": score,
                "Model_Parameters": "{}",
            }
            for name, score in [(label, 0.9), ("lr", 0.7)]
        ]
    )
    raw = pd.DataFrame({"Dataset": ["tiny.csv"]})
    found = qml_winner(results, raw, str(tmp_path), "qhdc")
    assert found is not None
    assert (tmp_path / "qhdc_qml_winners.csv").exists()
