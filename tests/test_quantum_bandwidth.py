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

"""qsvc, pqk and qpl take a ``bandwidth``: features reach the feature map as ``bandwidth * x``.

QProfiler scales features to [0, 1] and the qiskit maps rotate by ``2 x``, so every quantum
kernel ran at one fixed angle range while the classical SVC tuned its gamma. In the
kernel_exps torus study the bandwidth alone moved a fidelity kernel by up to 0.26 F1, and
the winning setting (B = pi) was out of QProfiler's reach. The contract:

1. ``bandwidth=b`` on X is exactly the unscaled model on ``b * X``.
2. 1.0 is the old behaviour bit for bit: the same metrics, no new parameter in the
   results, and the same projection-cache filename.
3. Two bandwidths never share a projection cache file.
4. The tuned twins search it like any other hyperparameter and record the chosen value.
5. Anything but a positive finite number is refused.
"""

from __future__ import annotations

import ast
import hashlib
import os

import numpy as np
import pytest

import qbiocode.utils.qutils as qutils
from qbiocode.learning._tuning import build_search_space
from qbiocode.learning.compute_pqk import compute_pqk
from qbiocode.learning.compute_qpl import compute_qpl
from qbiocode.learning.compute_qsvc import compute_qsvc, compute_qsvc_opt

N_TRAIN, N_TEST, N_FEATURES = 12, 6, 2
METRICS = ("accuracy", "f1_score", "balanced_accuracy", "mcc", "auc", "pr_auc")


def _dataset():
    rng = np.random.default_rng(3)
    X_train = rng.uniform(0, 1, (N_TRAIN, N_FEATURES))
    X_test = rng.uniform(0, 1, (N_TEST, N_FEATURES))
    return X_train, X_test, np.array([0, 1] * (N_TRAIN // 2)), np.array([0, 1] * (N_TEST // 2))


def _args(tmp_path, **extra):
    return {"backend": "simulator", "seed": 42, "shots": 1024, "grid_search": False,
            "pqk_projection_dir": str(tmp_path / "pqk"), "qpl_projection_dir": str(tmp_path / "qpl"),
            **extra}


def _row(frame, model):
    return frame["results_" + model].iloc[0]


def _params(row):
    value = row.get("BestParams_Tuned", row.get("Model_Parameters"))
    return ast.literal_eval(value) if isinstance(value, str) else dict(value)


def _metrics(row):
    return {m: row[m] for m in METRICS}


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "1", None])
def test_anything_but_a_positive_finite_number_is_refused(bad):
    with pytest.raises(ValueError, match="bandwidth"):
        qutils.apply_bandwidth(bad, np.ones((2, 2)))


def test_one_returns_the_arrays_unchanged_and_another_value_scales():
    X = np.array([[0.25, 0.5]])
    (same,) = qutils.apply_bandwidth(1.0, X)
    np.testing.assert_array_equal(same, X)
    (scaled,) = qutils.apply_bandwidth(np.pi, X)
    np.testing.assert_allclose(scaled, np.pi * X)


class TestQSVC:
    def test_bandwidth_b_is_the_unscaled_model_on_b_x(self, tmp_path):
        X_train, X_test, y_train, y_test = _dataset()
        scaled = compute_qsvc(X_train, X_test, y_train, y_test, _args(tmp_path), bandwidth=2.5)
        direct = compute_qsvc(2.5 * X_train, 2.5 * X_test, y_train, y_test, _args(tmp_path))
        assert _metrics(_row(scaled, "qsvc")) == _metrics(_row(direct, "qsvc"))
        assert _params(_row(scaled, "qsvc"))["bandwidth"] == 2.5

    def test_one_is_the_old_behaviour(self, tmp_path):
        X_train, X_test, y_train, y_test = _dataset()
        old = compute_qsvc(X_train, X_test, y_train, y_test, _args(tmp_path))
        one = compute_qsvc(X_train, X_test, y_train, y_test, _args(tmp_path), bandwidth=1.0)
        assert _metrics(_row(old, "qsvc")) == _metrics(_row(one, "qsvc"))
        assert "bandwidth" not in _params(_row(one, "qsvc"))

    def test_the_tuned_twin_searches_and_records_it(self, tmp_path):
        X_train, X_test, y_train, y_test = _dataset()
        frame = compute_qsvc_opt(X_train, X_test, y_train, y_test, _args(tmp_path),
                                 reps=[1], bandwidth=[0.5, 3.0], n_trials=2)
        assert _params(_row(frame, "qsvc_opt"))["bandwidth"] in (0.5, 3.0)

    def test_a_log_range_is_a_valid_search_space(self):
        space = build_search_space("qsvc", {"bandwidth": {"low": 0.098, "high": 6.283, "log": True}})
        assert "bandwidth" in space


class TestProjectedKernels:
    @staticmethod
    def _files(directory):
        return sorted(f for f in os.listdir(directory) if f.endswith(".npy"))

    @staticmethod
    def _legacy_fingerprint():
        """compute_pqk's cache digest before ``bandwidth`` existed, written out again."""
        X_train, X_test, _, _ = _dataset()
        parts = ("ZZ", "linear", 2, "estimator", N_FEATURES,
                 qutils.dataset_fingerprint(X_train, X_test), 1024, "simulator")
        return hashlib.sha256(repr(parts).encode()).hexdigest()[:10]

    def _pqk(self, tmp_path, X_train, X_test, **kw):
        _, _, y_train, y_test = _dataset()
        return compute_pqk(X_train, X_test, y_train, y_test, _args(tmp_path), data_key="ds",
                           encoding="ZZ", entanglement="linear", reps=2, **kw)

    def test_one_keeps_the_historical_cache_file_and_the_results(self, tmp_path):
        X_train, X_test, _, _ = _dataset()
        row = _row(self._pqk(tmp_path, X_train, X_test, bandwidth=1.0), "pqk")
        fp = self._legacy_fingerprint()
        assert self._files(tmp_path / "pqk") == [f"pqk_projection_ds_{fp}_test.npy",
                                                 f"pqk_projection_ds_{fp}_train.npy"]
        assert "bandwidth" not in _params(row)

    def test_two_bandwidths_never_share_a_cache_file(self, tmp_path):
        X_train, X_test, _, _ = _dataset()
        self._pqk(tmp_path, X_train, X_test)
        row = _row(self._pqk(tmp_path, X_train, X_test, bandwidth=0.5), "pqk")
        assert len(self._files(tmp_path / "pqk")) == 4
        assert _params(row)["bandwidth"] == 0.5

    def test_bandwidth_b_projects_b_x(self, tmp_path):
        X_train, X_test, _, _ = _dataset()
        self._pqk(tmp_path / "a", X_train, X_test, bandwidth=0.5)
        self._pqk(tmp_path / "b", 0.5 * X_train, 0.5 * X_test)
        a = [np.load(tmp_path / "a" / "pqk" / f) for f in self._files(tmp_path / "a" / "pqk")]
        b = [np.load(tmp_path / "b" / "pqk" / f) for f in self._files(tmp_path / "b" / "pqk")]
        for left, right in zip(sorted(a, key=len), sorted(b, key=len)):
            np.testing.assert_allclose(left, right, atol=1e-12)

    def test_qpl_bandwidth_b_is_qpl_on_b_x(self, tmp_path):
        X_train, X_test, y_train, y_test = _dataset()
        kw = dict(data_key="ds", encoding="ZZ", reps=2, classical_models=["lr"])
        scaled = compute_qpl(X_train, X_test, y_train, y_test, _args(tmp_path / "a"),
                             bandwidth=2.0, **kw)
        direct = compute_qpl(2.0 * X_train, 2.0 * X_test, y_train, y_test,
                             _args(tmp_path / "b"), **kw)
        assert _metrics(_row(scaled, "qpl_lr")) == _metrics(_row(direct, "qpl_lr"))
        assert _params(_row(scaled, "qpl_lr"))["bandwidth"] == 2.0


def test_in_manifest_mode_trial_0_is_the_default_bandwidth(tmp_path):
    """The protocol's trial 0 is the arm's default config, so bandwidth 1.0 even when the
    searched values leave it out (the space is widened by the default, and logged)."""
    from qbiocode.evaluation.model_run import _default_params, _Reseed
    from qbiocode.evaluation.protocol import TRIALS_PREFIX, ValidationSplit

    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 1.0, size=(48, 2))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    fit, val, test = np.arange(24), np.arange(24, 36), np.arange(36, 48)
    split = ValidationSplit(X[fit], X[val], y[fit], y[val], fit_idx=fit, val_idx=val)
    train = np.concatenate([fit, val])
    args = {**_args(tmp_path), "shots": 64, "q_seed": 7, "grid_search": True, "tune_quantum": True,
            "kernel_dump_dir": str(tmp_path / "kernels")}
    defaults = _default_params(compute_qsvc, {"reps": 1})
    assert defaults["bandwidth"] == 1.0
    frame = compute_qsvc_opt(X[train], X[test], y[train], y[test], args, bandwidth=[0.5, 3.0],
                             n_trials=3, validation=split, default_params=defaults,
                             reseed=_Reseed(7, 7))
    log = next(c for c in frame[TRIALS_PREFIX + "qsvc_opt"] if isinstance(c, dict))
    assert log["trials"][0]["is_default"] and log["trials"][0]["params"]["bandwidth"] == 1.0
    assert {t["params"]["bandwidth"] for t in log["trials"]} <= {0.5, 1.0, 3.0}
