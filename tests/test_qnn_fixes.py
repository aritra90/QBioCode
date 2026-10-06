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

"""Regression tests for three defects in ``compute_qnn``.

* **Labels.** ``NeuralNetworkClassifier`` does not encode integer targets: it trained
  the one-output EstimatorQNN against ``{0, 1}`` and ``predict`` returned
  ``sign(raw)`` in ``{-1, +1}``, so class 0 was never predicted and balanced accuracy
  was at most 0.5 by construction. The sampler returned the argmax column index, which
  equals the label only for a ``{0, 1}`` target.
* **Noise.** EstimatorQNN's ``default_precision=0.015625`` made every V2 estimator add
  Gaussian noise to its expectation values, unseeded on Aer -- the cause of qnn's
  irreproducibility on ``'simulator_aer'``.
* **Backend rule.** The sampler branch tested ``'simulator' in backend``, which also
  matched ``'simulator_aer'`` and so dropped the configured Aer primitive and its pass
  manager.

Everything runs on 30 rows and 2 features with a handful of optimizer iterations.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import qbiocode  # noqa: F401  (orders the OpenMP runtimes; see test_openmp_import_order)
from qbiocode.utils import qutils

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: The module, not the function of the same name that ``qbiocode.learning`` re-exports.
QNN_MODULE = importlib.import_module("qbiocode.learning.compute_qnn")


@pytest.fixture(scope="module")
def toy():
    """30 rows in [0, 1]^2, labelled by the first feature: separable along one axis."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(30, 2))
    y = (X[:, 0] > 0.5).astype(int)
    return X[:20], X[20:], y[:20], y[20:]


#: The Aer method the pilot's wide feature maps would use.
AER_MPS = {"sim_method": "matrix_product_state"}


def _args(backend="simulator", **extra):
    return {"backend": backend, "seed": 3, "shots": 256, **extra}


def _seed_globals(seed=5):
    """Seed every RNG the variational fit reads, as ``model_run`` does."""
    import qiskit_algorithms.utils as qa_utils
    import qiskit_machine_learning.utils as qml_utils

    np.random.seed(seed)
    qa_utils.algorithm_globals.random_seed = seed
    qml_utils.algorithm_globals.random_seed = seed


def _run(monkeypatch, X_train, X_test, y_train, y_test, args, **kwargs):
    """Run ``compute_qnn`` and return the predictions and scores it evaluated.

    ``modeleval`` itself is stubbed out: these tests are about what ``compute_qnn``
    hands it, not about scoring (which :func:`test_modeleval_scores_string_labels`
    covers for the string parametrization used here).
    """
    captured = {}

    def spy(y_true, y_predicted, *rest, **kw):
        captured["y_predicted"] = np.asarray(y_predicted)
        captured["y_score"] = np.asarray(kw.get("y_score"), dtype=float)

    monkeypatch.setattr(QNN_MODULE, "modeleval", spy)
    _seed_globals()
    kwargs.setdefault("reps", 1)
    kwargs.setdefault("maxiter", 8)
    QNN_MODULE.compute_qnn(X_train, X_test, y_train, y_test, args, **kwargs)
    return captured["y_predicted"], captured["y_score"]


@pytest.mark.parametrize("primitive", ["estimator", "sampler"])
@pytest.mark.parametrize(
    "labels", [(0, 1), (3, 7), ("benign", "malignant")], ids=["01", "int37", "str"]
)
def test_predictions_are_in_the_training_label_set(monkeypatch, toy, primitive, labels):
    """Not {-1, +1}, not column indices: the caller's own labels, in their own dtype."""
    X_train, X_test, y_train, y_test = toy
    mapping = np.asarray(labels)
    y_pred, y_score = _run(
        monkeypatch, X_train, X_test, mapping[y_train], mapping[y_test], _args(),
        primitive=primitive,
    )
    assert set(y_pred.tolist()) <= set(mapping.tolist())
    assert y_pred.dtype == mapping.dtype
    assert y_pred.shape == (len(X_test),)
    assert y_score.shape == (len(X_test),) and np.all(np.isfinite(y_score))


@pytest.mark.parametrize("primitive", ["estimator", "sampler"])
def test_the_score_is_oriented_towards_the_larger_label(monkeypatch, toy, primitive):
    """Higher score means the predicted class is classes_[1], for both primitives."""
    X_train, X_test, y_train, y_test = toy
    y_pred, y_score = _run(
        monkeypatch, X_train, X_test, y_train, y_test, _args(),
        primitive=primitive, maxiter=40,
    )
    threshold = 0.0 if primitive == "estimator" else 0.5
    # The sampler predicts by argmax over two probabilities summing to 1, so column 1
    # above 0.5 is exactly "predicted parity 1"; ties are not expected on this data.
    np.testing.assert_array_equal(y_pred == 1, y_score > threshold)


def test_the_estimator_learns_a_separable_toy(monkeypatch, toy):
    """The pilot's estimator qnn never predicted class 0; on a separable toy it must."""
    from sklearn.metrics import balanced_accuracy_score

    X_train, X_test, y_train, y_test = toy
    y_pred, _ = _run(
        monkeypatch, X_train, X_test, y_train, y_test, _args(),
        primitive="estimator", maxiter=40,
    )
    assert 0 in y_pred, "class 0 was never predicted"
    assert balanced_accuracy_score(y_test, y_pred) > 0.5


def test_a_non_binary_target_is_refused(toy):
    X_train, X_test, y_train, y_test = toy
    y_three = y_train.copy()
    y_three[:3] = 2
    with pytest.raises(ValueError, match="exactly two classes"):
        QNN_MODULE.compute_qnn(X_train, X_test, y_three, y_test, _args(), reps=1, maxiter=1)


def test_the_target_is_checked_before_any_backend_is_opened(monkeypatch, toy):
    # On an IBM backend get_backend_session opens a runtime Session that only the
    # success path closes, so a bad target must be refused before it is called.
    X_train, X_test, y_train, y_test = toy
    y_three = y_train.copy()
    y_three[:3] = 2

    def no_session(*a, **k):
        raise AssertionError("get_backend_session reached with a non-binary target")

    monkeypatch.setattr(QNN_MODULE.qutils, "get_backend_session", no_session)
    with pytest.raises(ValueError, match="exactly two classes"):
        QNN_MODULE.compute_qnn(X_train, X_test, y_three, y_test, _args(), reps=1, maxiter=1)


def test_modeleval_scores_string_labels():
    # compute_qnn returns the caller's own labels, so modeleval must score strings;
    # its positive-label helpers used ndarray.max(), which raises on '<U' arrays.
    import time

    from qbiocode.evaluation.model_evaluation import modeleval

    y = np.array(["no", "yes", "yes", "no", "yes", "no"])
    pred = np.array(["no", "yes", "no", "no", "yes", "yes"])
    score = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])
    out = modeleval(y, pred, time.time(), {}, {}, "qnn", verbose=False, y_score=score)
    res = out["results_qnn"].iloc[0]
    assert res["accuracy"] == pytest.approx(4 / 6)
    # 'yes' (the larger label) is positive for both threshold-free metrics.
    assert res["auc"] == pytest.approx(8 / 9)
    assert res["pr_auc"] > 0.5


@pytest.mark.parametrize(
    "backend,primitive,sim_method",
    [
        ("simulator", "estimator", None),
        ("simulator", "sampler", None),
        ("simulator_aer", "estimator", "matrix_product_state"),
    ],
)
def test_two_seeded_runs_agree(monkeypatch, toy, backend, primitive, sim_method):
    """Identical predictions and scores -- including qnn on Aer MPS, once irreproducible."""
    X_train, X_test, y_train, y_test = toy
    extra = {"sim_method": sim_method} if sim_method else {}
    args = _args(backend, **extra)
    first = _run(monkeypatch, X_train, X_test, y_train, y_test, args, primitive=primitive)
    second = _run(monkeypatch, X_train, X_test, y_train, y_test, args, primitive=primitive)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_the_aer_estimator_matches_the_statevector_one(monkeypatch, toy):
    """Exact on both, so equal: covers default_precision and the observable's layout."""
    X_train, X_test, y_train, y_test = toy
    on_sv = _run(monkeypatch, X_train, X_test, y_train, y_test, _args(), primitive="estimator")
    on_aer = _run(
        monkeypatch, X_train, X_test, y_train, y_test,
        _args("simulator_aer", sim_method="matrix_product_state"), primitive="estimator",
    )
    np.testing.assert_array_equal(on_sv[0], on_aer[0])
    np.testing.assert_allclose(on_sv[1], on_aer[1], atol=1e-9)


class _Stop(Exception):
    """Raised by a stub QNN constructor once it has recorded its arguments."""


def _capture_constructor(monkeypatch, name, backend_session=None):
    """Replace ``name`` (EstimatorQNN or SamplerQNN) with a recorder that stops the fit.

    Optionally also stub ``get_backend_session`` to return ``backend_session``, so the
    test asserts on the primitive object itself without running Aer.
    """
    seen = {}

    def recorder(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(QNN_MODULE, name, recorder)
    if backend_session is not None:
        monkeypatch.setattr(
            QNN_MODULE.qutils, "get_backend_session", lambda *a, **k: backend_session
        )
    return seen


@pytest.mark.parametrize(
    "primitive,cls,key", [("estimator", "EstimatorQNN", "estimator"),
                          ("sampler", "SamplerQNN", "sampler")]
)
def test_simulator_aer_gets_the_aer_primitive_and_a_pass_manager(
    monkeypatch, toy, primitive, cls, key
):
    """The sampler branch used to match 'simulator' in 'simulator_aer' and drop both."""
    from qiskit_aer import AerSimulator

    marker = object()
    seen = _capture_constructor(
        monkeypatch, cls, (AerSimulator(method="matrix_product_state"), None, marker)
    )
    X_train, X_test, y_train, y_test = toy
    with pytest.raises(_Stop):
        QNN_MODULE.compute_qnn(
            X_train, X_test, y_train, y_test,
            _args("simulator_aer", sim_method="matrix_product_state"),
            primitive=primitive, reps=1, maxiter=1,
        )
    assert seen[key] is marker
    assert seen["pass_manager"] is not None


@pytest.mark.parametrize(
    "primitive,cls,key", [("estimator", "EstimatorQNN", "estimator"),
                          ("sampler", "SamplerQNN", "sampler")]
)
def test_simulator_gets_the_seeded_primitive_and_no_pass_manager(
    monkeypatch, toy, primitive, cls, key
):
    """'simulator' used to pass no primitive at all, so args['seed'] never reached it."""
    marker = object()
    seen = _capture_constructor(monkeypatch, cls, (None, None, marker))
    X_train, X_test, y_train, y_test = toy
    with pytest.raises(_Stop):
        QNN_MODULE.compute_qnn(
            X_train, X_test, y_train, y_test, _args(), primitive=primitive, reps=1, maxiter=1
        )
    assert seen[key] is marker
    assert seen["pass_manager"] is None


@pytest.mark.parametrize("backend", ["simulator", "simulator_aer"])
def test_the_estimator_qnn_adds_no_precision_noise(monkeypatch, toy, backend):
    """default_precision=0.0 on every EstimatorQNN compute_qnn builds."""
    seen = _capture_constructor(monkeypatch, "EstimatorQNN")
    X_train, X_test, y_train, y_test = toy
    with pytest.raises(_Stop):
        QNN_MODULE.compute_qnn(
            X_train, X_test, y_train, y_test,
            _args(backend, **({} if backend == "simulator" else AER_MPS)),
            primitive="estimator", reps=1, maxiter=1,
        )
    assert seen["default_precision"] == 0.0


def test_simulator_aer_estimator_no_longer_warns():
    """The "not root-caused" RuntimeWarning is gone: its cause was default_precision."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        qutils.get_backend_session(
            {"backend": "simulator_aer", "seed": 1, "sim_method": "matrix_product_state"},
            "estimator",
            num_qubits=2,
        )
