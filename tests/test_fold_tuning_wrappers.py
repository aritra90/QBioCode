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

"""The ``compute_<m>_opt`` wrappers under ``split_mode: manifest`` (G4).

The tuner core is pinned by ``test_fold_tuning.py``. These tests pin what each wrapper
adds on top: it forwards ``validation``/``default_params`` (and ``reseed`` for the
quantum ones), fixes the unsearched configured defaults for every trial and the refit,
caps libsvm, scores the PQK/QPL head searches with the tuning metric, gives each QPL
head its own trial log, and writes no kernel dump from a trial. One tiny call per
family: a classical tree, a linear model, nb, a quantum kernel and a variational
model. The qnn ``readout`` switch is tested here as well.
"""

import importlib
import inspect
import logging
import math
import os
import warnings

import numpy as np
import pytest

from qbiocode.evaluation.model_run import _default_params, _Reseed
from qbiocode.evaluation.protocol import TRIALS_PREFIX, ValidationSplit
from qbiocode.learning.compute_fold import FOLD_SVC_MAX_ITER, fold_fixed, fold_svc_max_iter


def _module(name):
    """The ``compute_<name>`` module; the package re-exports the function by that name."""
    return importlib.import_module(f"qbiocode.learning.compute_{name}")


@pytest.fixture(scope="module")
def toy():
    """48 rows, 2 features in [0, 1], a clean diagonal boundary; fit 24 / val 12 / test 12."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0.0, 1.0, size=(48, 2))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    fit, val, test = np.arange(24), np.arange(24, 36), np.arange(36, 48)
    split = ValidationSplit(X[fit], X[val], y[fit], y[val], fit_idx=fit, val_idx=val)
    train = np.concatenate([fit, val])
    return {
        "split": split, "X_train": X[train], "y_train": y[train],
        "X_test": X[test], "y_test": y[test],
    }


@pytest.fixture
def qargs(tmp_path):
    """Simulator args with every cache and dump directory under ``tmp_path``."""
    return {
        "backend": "simulator", "shots": 64, "seed": 7, "q_seed": 7, "n_jobs": 1,
        "grid_search": True, "tune_quantum": True,
        "pqk_projection_dir": str(tmp_path / "pqk"),
        "qpl_projection_dir": str(tmp_path / "qpl"),
        "kernel_dump_dir": str(tmp_path / "kernels"),
    }


CARGS = {"seed": 0, "n_jobs": 1, "grid_search": True}


def _call(name, toy, args, default_params, **kwargs):
    fn = getattr(_module(name), f"compute_{name}_opt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(
            toy["X_train"], toy["X_test"], toy["y_train"], toy["y_test"], args,
            validation=toy["split"], default_params=default_params, **kwargs,
        )


def _log(frame, label):
    cells = [c for c in frame[TRIALS_PREFIX + label] if isinstance(c, dict)]
    assert len(cells) == 1, f"expected one trials_{label} cell, got {len(cells)}"
    return cells[0]


def _metrics(frame, label):
    return next(v for v in frame["results_" + label] if isinstance(v, dict))


class TestHelpers:
    def test_fold_fixed_keeps_unsearched_valid_non_none_defaults(self):
        fixed = fold_fixed(
            "lr", {"C": [0.1, 1.0], "solver": None},
            {"C": 1.0, "solver": "saga", "max_iter": 10000, "l1_ratio": None, "bogus": 3},
            {"C", "solver", "max_iter", "l1_ratio"}, {"random_state": 0},
        )
        assert fixed == {"solver": "saga", "max_iter": 10000, "random_state": 0}

    def test_explicit_fixed_wins(self):
        assert fold_fixed("lr", {}, {"max_iter": 5}, {"max_iter"}, {"max_iter": 9}) == {
            "max_iter": 9}

    @pytest.mark.parametrize("configured,expected", [
        (None, FOLD_SVC_MAX_ITER), (-1, FOLD_SVC_MAX_ITER), (0, FOLD_SVC_MAX_ITER),
        (500, 500), ("x", FOLD_SVC_MAX_ITER),
    ])
    def test_fold_svc_max_iter(self, configured, expected):
        assert fold_svc_max_iter(configured) == expected


class TestClassicalWrappers:
    def test_rf_trial_zero_is_the_default_and_the_fixed_reach_the_log(self, toy):
        from qbiocode.learning.compute_rf import compute_rf

        defaults = _default_params(
            compute_rf, {"n_estimators": 10, "max_depth": 3, "criterion": "entropy"})
        frame = _call("rf", toy, CARGS, defaults, n_estimators=[5, 10],
                      max_depth=[2, 3], random_state=0, n_trials=3)
        log = _log(frame, "Random Forest")
        assert len(log["trials"]) == 3
        assert log["trials"][0]["is_default"]
        assert log["trials"][0]["params"] == {"n_estimators": 10, "max_depth": 3}
        # criterion was not searched: fixed at the configured default, for trials and refit.
        assert log["fixed"]["criterion"] == "entropy"
        assert log["fixed"]["random_state"] == 0

    def test_lr_unsearched_solver_and_max_iter_are_fixed(self, toy):
        from qbiocode.learning.compute_lr import compute_lr

        defaults = _default_params(compute_lr, {})
        frame = _call("lr", toy, CARGS, defaults, C=[0.1, 1.0, 10.0], random_state=0,
                      n_trials=3)
        log = _log(frame, "Logistic Regression")
        assert len(log["trials"]) == 3
        assert log["trials"][0]["is_default"]
        assert log["trials"][0]["params"] == {"C": defaults["C"]}
        assert log["fixed"]["solver"] == defaults["solver"]
        assert log["fixed"]["max_iter"] == defaults["max_iter"]
        # Every trial kept its validation predictions, aligned with val_idx.
        assert all(t["y_pred"] is not None and len(t["y_pred"]) == 12 for t in log["trials"])
        assert _metrics(frame, "Logistic Regression")["tuning_metric"] == "balanced_accuracy"

    def test_nb_records_every_trial(self, toy):
        from qbiocode.learning.compute_nb import compute_nb

        defaults = _default_params(compute_nb, {})
        frame = _call("nb", toy, CARGS, defaults, var_smoothing=[1e-9, 1e-3, 1e-1], n_trials=3)
        log = _log(frame, "Naive Bayes")
        assert len(log["trials"]) == 3
        assert log["trials"][0]["is_default"]
        assert log["trials"][0]["params"] == {"var_smoothing": defaults["var_smoothing"]}

    def test_svc_max_iter_is_fixed_at_the_fold_cap(self, toy):
        from qbiocode.learning.compute_svc import compute_svc

        frame = _call("svc", toy, CARGS, _default_params(compute_svc, {}),
                      C=[0.1, 1.0], random_state=0, n_trials=2)
        assert _log(frame, "SVC")["fixed"]["max_iter"] == FOLD_SVC_MAX_ITER

    def test_a_capped_svc_trial_fails_and_the_study_still_refits(self, toy, caplog):
        from qbiocode.learning.compute_svc import compute_svc

        defaults = _default_params(compute_svc, {"max_iter": 1})
        with caplog.at_level(logging.WARNING):
            frame = _call("svc", toy, CARGS, defaults, C=[0.1, 1.0, 10.0],
                          kernel=["rbf"], random_state=0, n_trials=3)
        log = _log(frame, "SVC")
        assert log["fixed"]["max_iter"] == 1
        assert [t["state"] for t in log["trials"]] == ["FAIL"] * 3
        assert all(math.isnan(t["value"]) for t in log["trials"])
        metrics = _metrics(frame, "SVC")
        assert math.isnan(metrics["tuning_score"])
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert "max_iter" in caplog.text

    def test_internal_mode_adds_no_trials_column(self, toy):
        from qbiocode.learning.compute_nb import compute_nb_opt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_nb_opt(toy["X_train"], toy["X_test"], toy["y_train"],
                                   toy["y_test"], CARGS, var_smoothing=[1e-9, 1e-3],
                                   n_trials=2)
        assert not [c for c in frame.columns if c.startswith(TRIALS_PREFIX)]


class TestQuantumWrappers:
    def test_qsvc_trials_write_no_kernel_dump(self, toy, qargs):
        from qbiocode.learning.compute_qsvc import compute_qsvc

        defaults = _default_params(compute_qsvc, {"reps": 1})
        frame = _call("qsvc", toy, qargs, defaults, C=[0.1, 1.0], n_trials=2,
                      data_key="toy_none_2_0", reseed=_Reseed(7, 7))
        log = _log(frame, "qsvc_opt")
        assert len(log["trials"]) == 2
        assert log["trials"][0]["is_default"]
        assert log["fixed"]["reps"] == 1
        dumps = sorted(os.listdir(qargs["kernel_dump_dir"]))
        assert dumps and all(d.startswith("gram_qsvc_opt_toy_none_2_0") for d in dumps), dumps

    def test_qnn_trials_record_predictions_and_fixed(self, toy, qargs):
        from qbiocode.learning.compute_qnn import compute_qnn

        defaults = _default_params(compute_qnn, {"maxiter": 3, "reps": 1})
        frame = _call("qnn", toy, qargs, defaults, maxiter=[2, 3], n_trials=2,
                      reseed=_Reseed(7, 7))
        log = _log(frame, "qnn_opt")
        assert len(log["trials"]) == 2
        assert log["trials"][0]["is_default"]
        assert log["trials"][0]["params"] == {"maxiter": 3}
        assert log["fixed"]["reps"] == 1 and log["fixed"]["readout"] == "global"
        assert all(t["y_pred"] is not None for t in log["trials"])

    def test_qnn_readout_is_searchable(self, toy, qargs):
        from qbiocode.learning.compute_qnn import compute_qnn

        defaults = _default_params(compute_qnn, {"maxiter": 2, "reps": 1})
        frame = _call("qnn", toy, qargs, defaults, readout=["global", "local"], n_trials=2,
                      reseed=_Reseed(7, 7))
        log = _log(frame, "qnn_opt")
        assert [t["params"]["readout"] for t in log["trials"]][0] == "global"
        assert {t["params"]["readout"] for t in log["trials"]} == {"global", "local"}
        assert "readout" not in log["fixed"]

    def test_pqk_head_search_uses_the_tuning_metric_and_the_cap(self, toy, qargs):
        from qbiocode.learning.compute_pqk import compute_pqk

        # A configured head_max_iter of -1 is libsvm's "no cap": replaced by the fold cap.
        defaults = _default_params(compute_pqk, {"reps": 1, "head_max_iter": -1})
        frame = _call("pqk", toy, qargs, defaults, entanglement=["linear", "full"],
                      n_trials=2, data_key="toy_none_2_0", reseed=_Reseed(7, 7))
        log = _log(frame, "pqk_opt")
        assert len(log["trials"]) == 2
        assert log["fixed"]["head_scoring"] == "balanced_accuracy"
        assert log["fixed"]["head_max_iter"] == FOLD_SVC_MAX_ITER
        params = _metrics(frame, "pqk_opt")["BestParams_Tuned"]
        assert params["head_scoring"] == "balanced_accuracy"
        dumps = sorted(os.listdir(qargs["kernel_dump_dir"]))
        assert dumps and all(d.startswith("proj_pqk_opt_toy_none_2_0") for d in dumps), dumps

    def test_qpl_heads_get_their_own_logs_and_scores(self, toy, qargs):
        from qbiocode.learning.compute_qpl import compute_qpl

        defaults = _default_params(compute_qpl, {"reps": 1, "classical_models": ["lr", "svc"],
                                                 "head_max_iter": 0})
        frame = _call("qpl", toy, qargs, defaults, entanglement=["linear", "full"],
                      n_trials=2, reseed=_Reseed(7, 7))
        scores = {}
        for head in ("lr", "svc"):
            log = _log(frame, f"qpl_opt_{head}")
            assert len(log["trials"]) == 2
            assert log["fixed"]["classical_models"] == ["lr", "svc"]
            assert all(t["y_pred"] is not None and len(t["y_pred"]) == 12 for t in log["trials"])
            best = next(t for t in log["trials"] if t["is_best"])
            metrics = _metrics(frame, f"qpl_opt_{head}")
            assert metrics["tuning_score"] == pytest.approx(best["value"])
            scores[head] = [t["value"] for t in log["trials"]]
            assert log["fixed"]["head_max_iter"] == FOLD_SVC_MAX_ITER
        assert TRIALS_PREFIX + "qpl_opt" not in frame.columns

    def test_a_qpl_trial_failing_before_its_fit_costs_only_its_own_entry(self, toy, qargs):
        """Head logs pair frames to trials by number, not by count."""
        from qbiocode.learning.compute_qpl import compute_qpl

        reseed, calls = _Reseed(7, 7), []

        def failing_second_trial():
            calls.append(None)
            if len(calls) == 2:  # trial 1's reseed, before compute_qpl is called
                raise RuntimeError("reseed failed")
            reseed()

        defaults = _default_params(compute_qpl, {"reps": 1, "classical_models": ["lr", "svc"]})
        frame = _call("qpl", toy, qargs, defaults, entanglement=["linear", "full"],
                      n_trials=2, reseed=failing_second_trial)
        for head in ("lr", "svc"):
            first, second = _log(frame, f"qpl_opt_{head}")["trials"]
            assert first["state"] == "COMPLETE" and len(first["y_pred"]) == 12
            assert second["state"] == "FAIL" and second["y_pred"] is None
            assert math.isnan(second["value"])

    def test_quantum_internal_signatures_keep_their_positions(self):
        for name in ("qsvc", "qnn", "vqc", "pqk", "qpl"):
            params = inspect.signature(getattr(_module(name), f"compute_{name}_opt")).parameters
            for key in ("validation", "default_params", "reseed"):
                assert params[key].kind is inspect.Parameter.KEYWORD_ONLY
                assert params[key].default is None
        for name in ("pqk", "qpl"):
            params = inspect.signature(getattr(_module(name), f"compute_{name}")).parameters
            for key in ("head_scoring", "head_max_iter"):
                assert params[key].kind is inspect.Parameter.KEYWORD_ONLY


class TestQnnReadout:
    @pytest.mark.parametrize("primitive", ["estimator", "sampler"])
    @pytest.mark.parametrize("readout", ["global", "local"])
    def test_readout_labels_and_accuracy(self, readout, primitive):
        """Labels stay {0, 1} and an easy set is learned under either readout.

        Separable on x0 alone, at x0 near 0.3 or 1.3. Under the Z feature map qubit 0
        sits at azimuth 2 * x0 and a one-rep RealAmplitudes (qubit 0 only ever a CX
        control) brings only its X component, cos(2 * x0), into Z_0 -- with no bias term, so a local readout can
        only separate classes on which cos(2 * x0) changes sign (0.83 vs -0.86 here).
        """
        from qbiocode.learning.compute_qnn import compute_qnn
        from sklearn.metrics import balanced_accuracy_score

        rng = np.random.RandomState(1)
        X = rng.uniform(0.0, 1.0, size=(40, 2))
        y = (X[:, 0] > 0.5).astype(int)
        X = np.column_stack([np.where(y == 1, 1.3, 0.3) + 0.1 * rng.randn(40), X[:, 1]])
        args = {"backend": "simulator", "shots": 256, "seed": 3, "q_seed": 3,
                "grid_search": False}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = compute_qnn(X[:28], X[28:], y[:28], y[28:], args, primitive=primitive,
                                maxiter=40, reps=1, readout=readout)
        y_pred = np.asarray(next(v for v in frame["y_predicted_qnn"] if v is not None))
        assert set(np.unique(y_pred)) <= {0, 1}
        assert balanced_accuracy_score(y[28:], y_pred) > 0.5
        params = _metrics(frame, "qnn")["Model_Parameters"]
        assert ("readout" in params) == (readout == "local")

    def test_unknown_readout_raises(self):
        from qbiocode.learning.compute_qnn import compute_qnn

        with pytest.raises(ValueError, match="readout"):
            compute_qnn(np.zeros((4, 2)), np.zeros((2, 2)), np.array([0, 1, 0, 1]),
                        np.array([0, 1]), {"backend": "simulator"}, readout="parity")

    def test_readout_default_is_global(self):
        from qbiocode.learning.compute_qnn import _readout_label, compute_qnn

        assert inspect.signature(compute_qnn).parameters["readout"].default == "global"
        assert _readout_label("global", 3) == "ZZZ"
        assert _readout_label("local", 3) == "IIZ"

    @pytest.mark.parametrize("primitive", ["estimator", "sampler"])
    @pytest.mark.parametrize("readout", ["global", "local"])
    def test_readout_follows_the_transpiled_layout(self, monkeypatch, readout, primitive):
        """Under a pass manager with a non-trivial layout, each readout reads virtual qubit 0.

        The network is captured before it is built; the estimator's observable and the
        sampler's interpret function are checked against the circuit's own layout.
        """
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        cq = _module("qnn")
        device = GenericBackendV2(7, seed=1)
        monkeypatch.setattr(cq.qutils, "normalize_backend",
                            lambda args: {"backend": "fake_device"})
        monkeypatch.setattr(cq.qutils, "get_backend_session",
                            lambda args, primitive, num_qubits=None: (device, None, None))
        monkeypatch.setattr(
            cq, "generate_preset_pass_manager",
            lambda backend, optimization_level: generate_preset_pass_manager(
                backend=device, optimization_level=optimization_level, seed_transpiler=0,
                initial_layout=[4, 2, 6]),
        )
        seen = {}

        class _Captured(Exception):
            pass

        def _capture(**kwargs):
            seen.update(kwargs)
            raise _Captured

        monkeypatch.setattr(cq, "EstimatorQNN", _capture)
        monkeypatch.setattr(cq, "SamplerQNN", _capture)
        X = np.random.RandomState(0).uniform(size=(6, 3))
        with pytest.raises(_Captured):
            cq.compute_qnn(X, X[:2], np.array([0, 1] * 3), np.array([0, 1]),
                           {"backend": "fake_device", "seed": 0}, primitive=primitive,
                           reps=1, readout=readout)
        n_phys = device.num_qubits
        if primitive == "estimator":
            layout = seen["circuit"].layout.final_index_layout()
            (label,) = seen["observables"].paulis.to_labels()
            z_on = {n_phys - 1 - i for i, c in enumerate(label) if c == "Z"}
            assert z_on == ({layout[0]} if readout == "local" else set(layout))
        else:
            interpret = seen["interpret"]
            if readout == "local":
                # Transpiled here, so SamplerQNN keeps this layout and measures every
                # physical qubit: bit k of the outcome is physical qubit k.
                phys0 = seen["circuit"].layout.final_index_layout()[0]
                assert phys0 != 0
                assert interpret(1 << phys0) == 1 and interpret(1) == 0
                assert interpret((1 << n_phys) - 1 - (1 << phys0)) == 0
            else:
                assert interpret(0b101) == 0 and interpret(0b100) == 1
