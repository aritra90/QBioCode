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

"""Tuning inside the fold (``split_mode: manifest``): the tuner core.

Under the fold-based protocol every trial is ONE fit on the fit rows, scored on the
validation rows with the tuning metric; trial 0 is the arm's default config; every
trial is kept (params, validation score, state, duration, validation predictions);
the best config is refit by the caller on the whole outer training fold. These tests
pin that contract for :func:`run_study` / :func:`search_hyperparameters` (classical),
:func:`run_function_study` (quantum-shaped compute functions), the ``trials_<model>``
cell that carries the log through ``modeleval`` and ``model_run``, and the guard rails
``model_run`` puts around the mode. The last test pins ``split_mode: internal`` (no
validation split) to the numbers the tree produced before the change.
"""

import json
import logging
import pickle
import time

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.evaluation.protocol import TRIALS_PREFIX, ValidationSplit, trial_frames
from qbiocode.learning import _tuning
from qbiocode.learning._tuning import (
    TunedParams,
    build_search_space,
    check_fit_status,
    record_tuned_params,
    run_function_study,
    run_study,
    search_hyperparameters,
    tuning_scorer,
)

TUNING_LOGGER = "qbiocode.learning._tuning"


@pytest.fixture(scope="module")
def toy():
    """60 rows, 3 features, a noisy linear boundary; fit 36 / val 12 / test 12."""
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = (X[:, 0] + 0.8 * rng.randn(60) > 0).astype(int)
    fit, val, test = np.arange(36), np.arange(36, 48), np.arange(48, 60)
    split = ValidationSplit(X[fit], X[val], y[fit], y[val], fit_idx=fit, val_idx=val)
    train = np.concatenate([fit, val])
    return {
        "X": X, "y": y, "split": split,
        "X_train": X[train], "y_train": y[train], "X_test": X[test], "y_test": y[test],
    }


def _lr_study(toy, n_trials=5, default_params=None, space=None, **kwargs):
    space = space or build_search_space("lr", {"C": {"low": 0.01, "high": 10.0, "log": True}})
    return run_study(
        LogisticRegression, space, toy["X_train"], toy["y_train"], cv=3,
        n_trials=n_trials, model="lr", seed=0, fixed={"max_iter": 200},
        scoring=tuning_scorer({}), validation=toy["split"],
        default_params=default_params, **kwargs,
    )


# ---------------------------------------------------------------------------------------
# run_study / search_hyperparameters with a validation split
# ---------------------------------------------------------------------------------------


class TestValidationStudy:
    def test_trial_zero_is_the_default_config_and_counts_in_the_budget(self, toy):
        tuned = _lr_study(toy, n_trials=5, default_params={"C": 1.0, "solver": "lbfgs"})
        assert tuned.n_trials == 5
        assert [t.number for t in tuned.trials] == [0, 1, 2, 3, 4]
        # Restricted to the searched names: 'solver' is not searched.
        assert tuned.trials[0].params == {"C": 1.0}
        assert [t.is_default for t in tuned.trials] == [True] + [False] * 4

    def test_without_default_params_no_trial_is_default(self, toy):
        tuned = _lr_study(toy, n_trials=3)
        assert not any(t.is_default for t in tuned.trials)

    def test_values_are_validation_scores(self, toy):
        tuned = _lr_study(toy, n_trials=4, default_params={"C": 1.0})
        split = toy["split"]
        for record in tuned.trials:
            estimator = LogisticRegression(**record.params, max_iter=200)
            estimator.fit(split.X_fit, split.y_fit)
            by_hand = balanced_accuracy_score(split.y_val, estimator.predict(split.X_val))
            assert record.value == pytest.approx(by_hand)
            assert record.state == "COMPLETE"
            np.testing.assert_array_equal(record.y_pred, estimator.predict(split.X_val))
            np.testing.assert_allclose(
                record.y_score, extract_binary_scores(estimator, split.X_val)
            )
            assert record.duration_s >= 0

    def test_best_trial_is_the_refit_params_and_the_score(self, toy):
        tuned = _lr_study(toy, n_trials=6, default_params={"C": 1.0})
        best = tuned.trials[tuned.best_trial]
        assert dict(tuned) == best.params
        assert tuned.score == best.value == max(t.value for t in tuned.trials)
        assert tuned.evidence() == {
            "tuning_metric": "balanced_accuracy", "tuning_score": best.value,
            "tuning_reused": False,
        }
        np.testing.assert_array_equal(tuned.val_idx, toy["split"].val_idx)
        np.testing.assert_array_equal(tuned.y_val, toy["split"].y_val)

    def test_trials_do_not_see_the_callers_X(self, toy):
        # Garbage X/y: a validation search must not read them.
        split = toy["split"]
        tuned = run_study(
            LogisticRegression, build_search_space("lr", {"C": [0.1, 1.0]}),
            np.full((5, 3), np.nan), np.zeros(5), cv=3, n_trials=2, model="lr", seed=0,
            validation=split,
        )
        assert len(tuned.trials) == 2

    def test_out_of_space_categorical_default_extends_the_space(self, toy, caplog):
        space = build_search_space("lr", {"C": [0.1, 0.5]})
        with caplog.at_level(logging.INFO, logger=TUNING_LOGGER):
            tuned = _lr_study(toy, n_trials=10, default_params={"C": 1.0}, space=space)
        assert tuned.trials[0].params == {"C": 1.0}
        assert "not among the configured choices" in caplog.text
        # The finite-space cap holds, and counts the added value: 3 points, 3 trials.
        assert tuned.n_trials == 3
        assert {t.params["C"] for t in tuned.trials} <= {0.1, 0.5, 1.0}
        # The caller's space is not modified.
        assert space["C"].values == [0.1, 0.5]

    def test_out_of_range_default_widens_the_range(self, toy, caplog):
        space = build_search_space("lr", {"C": {"low": 0.01, "high": 0.1, "log": True}})
        with caplog.at_level(logging.INFO, logger=TUNING_LOGGER):
            tuned = _lr_study(toy, n_trials=4, default_params={"C": 1.0}, space=space)
        assert tuned.trials[0].params == {"C": 1.0}
        assert "widened it to [0.01, 1.0]" in caplog.text
        assert all(0.01 <= t.params["C"] <= 1.0 for t in tuned.trials)

    def test_unrepresentable_default_is_sampled_and_logged(self, caplog):
        space = build_search_space("rf", {"max_depth": {"low": 2, "high": 8}, "C": [1]})
        with caplog.at_level(logging.INFO, logger=TUNING_LOGGER):
            enqueue, widened, whole = _tuning._default_trial(
                "rf", space, {"max_depth": None, "C": 1}
            )
        assert enqueue == {"C": 1} and whole is False
        assert widened["max_depth"] is space["max_depth"]
        assert "cannot be represented" in caplog.text

    def test_int_range_widens_as_int(self):
        space = build_search_space("rf", {"n_estimators": {"low": 10, "high": 50}})
        enqueue, widened, whole = _tuning._default_trial("rf", space, {"n_estimators": 100})
        assert enqueue == {"n_estimators": 100} and whole is True
        assert widened["n_estimators"].is_int
        assert (widened["n_estimators"].low, widened["n_estimators"].high) == (10, 100)

    def test_failed_trial_is_nan_and_the_study_continues(self, toy):
        # l1 is not supported by lbfgs: that trial raises inside fit.
        space = build_search_space("lr", {"penalty": ["l1", "l2"]})
        tuned = run_study(
            LogisticRegression, space, toy["X_train"], toy["y_train"], cv=3,
            n_trials=2, model="lr", seed=0, fixed={"solver": "lbfgs"},
            validation=toy["split"], default_params={"penalty": "l1"},
        )
        first, second = tuned.trials
        assert first.state == "FAIL" and np.isnan(first.value) and first.y_pred is None
        assert second.state == "COMPLETE" and dict(tuned) == {"penalty": "l2"}
        assert tuned.best_trial == 1

    def test_unconverged_svc_trial_is_a_failure(self, toy, caplog):
        space = build_search_space("svc", {"max_iter": [1, -1]})
        with caplog.at_level(logging.INFO, logger=TUNING_LOGGER):
            tuned = run_study(
                SVC, space, toy["X_train"], toy["y_train"], cv=3, n_trials=2,
                model="svc", seed=0, validation=toy["split"],
                default_params={"max_iter": 1},
            )
        capped, free = tuned.trials
        assert capped.params == {"max_iter": 1}
        assert capped.state == "FAIL" and np.isnan(capped.value)
        assert free.state == "COMPLETE"
        assert dict(tuned) == {"max_iter": -1}
        assert "did not converge" in caplog.text and "'max_iter': 1" in caplog.text

    def test_every_trial_failing_raises(self, toy):
        # l1 is not supported by lbfgs: every trial raises inside fit.
        space = build_search_space("lr", {"penalty": ["l1"], "C": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Every tuning trial for 'lr' failed"):
            run_study(
                LogisticRegression, space, toy["X_train"], toy["y_train"], cv=3,
                n_trials=2, model="lr", seed=0, fixed={"solver": "lbfgs"},
                validation=toy["split"],
            )

    def test_every_trial_unconverged_falls_back_to_trial_zero(self, toy, caplog):
        # A non-converged fit must not cost the dataset's whole model_run: when every
        # trial stopped at max_iter, trial 0 (the default) is refit, with a warning.
        space = build_search_space("svc", {"C": [0.5, 1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger=TUNING_LOGGER):
            tuned = run_study(
                SVC, space, toy["X_train"], toy["y_train"], cv=3, n_trials=3,
                model="svc", seed=0, fixed={"max_iter": 1}, validation=toy["split"],
                default_params={"C": 2.0},
            )
        assert dict(tuned) == {"C": 2.0} and tuned.best_trial == 0
        assert np.isnan(tuned.score)
        assert all(t.state == "FAIL" for t in tuned.trials) and len(tuned.trials) == 3
        assert tuned.trial_log()["best"] == 0
        assert "stopped at max_iter without converging" in caplog.text

    def test_unrepresentable_default_makes_trial_zero_not_the_default(self, toy):
        # compute_svc's own gamma='scale' cannot sit in a numeric range: trial 0
        # enqueues C and samples gamma, so it is not recorded as the default config.
        space = build_search_space("svc", {
            "C": {"low": 0.1, "high": 10.0, "log": True},
            "gamma": {"low": 1e-3, "high": 1.0, "log": True},
        })
        tuned = run_study(
            SVC, space, toy["X_train"], toy["y_train"], cv=3, n_trials=2, model="svc",
            seed=0, validation=toy["split"], default_params={"C": 1.0, "gamma": "scale"},
        )
        first = tuned.trials[0]
        assert first.params["C"] == 1.0 and isinstance(first.params["gamma"], float)
        assert not any(t.is_default for t in tuned.trials)

    def test_a_searched_name_without_default_makes_trial_zero_not_the_default(self, toy):
        space = build_search_space("lr", {
            "C": {"low": 0.01, "high": 10.0, "log": True}, "tol": [1e-4, 1e-3],
        })
        tuned = _lr_study(toy, n_trials=2, default_params={"C": 1.0}, space=space)
        assert tuned.trials[0].params["C"] == 1.0
        assert not any(t.is_default for t in tuned.trials)

    def test_numpy_scalar_defaults_are_numbers(self, toy):
        space = build_search_space("rf", {"max_depth": {"low": 2, "high": 5}})
        enqueue, widened, whole = _tuning._default_trial(
            "rf", space, {"max_depth": np.int64(8)}
        )
        assert enqueue == {"max_depth": 8} and type(enqueue["max_depth"]) is int
        assert (widened["max_depth"].low, widened["max_depth"].high) == (2, 8) and whole
        tuned = _lr_study(toy, n_trials=2, default_params={"C": np.float64(1.0)})
        assert tuned.trials[0].params == {"C": 1.0} and tuned.trials[0].is_default

    def test_equivalent_layer_spelling_reuses_the_configured_choice(self, caplog):
        # mlp_args hidden_layer_sizes: 100 is the network [100] already in the grid.
        space = build_search_space(
            "mlp", {"hidden_layer_sizes": [[20], [50], [100]]}
        )
        with caplog.at_level(logging.INFO, logger=TUNING_LOGGER):
            enqueue, widened, whole = _tuning._default_trial(
                "mlp", space, {"hidden_layer_sizes": 100}
            )
        assert enqueue == {"hidden_layer_sizes": [100]} and whole
        assert widened["hidden_layer_sizes"].values == [[20], [50], [100]]
        assert "not among the configured choices" not in caplog.text
        # A genuinely new network is still added.
        enqueue, widened, _ = _tuning._default_trial(
            "mlp", space, {"hidden_layer_sizes": (30, 10)}
        )
        assert len(widened["hidden_layer_sizes"].values) == 4

    def test_check_fit_status_warns_after_an_unconverged_refit(self, toy, caplog):
        estimator = SVC(max_iter=1).fit(toy["X_train"], toy["y_train"])
        with caplog.at_level(logging.WARNING, logger=TUNING_LOGGER):
            assert check_fit_status(estimator, "svc_opt", {"C": 1.0}) is False
        assert "did not converge" in caplog.text
        assert check_fit_status(LogisticRegression().fit(toy["X_train"], toy["y_train"]),
                                "lr") is True

    def test_search_hyperparameters_forwards_and_refuses_grid(self, toy):
        tuned = search_hyperparameters(
            "nb", GaussianNB, {"var_smoothing": [1e-9, 1e-3, 1e-1]},
            toy["X_train"], toy["y_train"], cv=3, n_trials=30, seed=0,
            validation=toy["split"], default_params={"var_smoothing": 1e-9},
        )
        assert tuned.n_trials == 3 and tuned.trials[0].params == {"var_smoothing": 1e-9}
        with pytest.raises(ValueError, match="tuner: 'grid'"):
            search_hyperparameters(
                "nb", GaussianNB, {"var_smoothing": [1e-9]}, toy["X_train"],
                toy["y_train"], cv=3, tuner="grid", validation=toy["split"],
            )

    def test_default_params_without_validation_enqueues_trial_zero(self, toy):
        # Internal mode is only changed when a caller passes default_params.
        space = build_search_space("lr", {"C": [0.1, 1.0, 5.0]})
        tuned = run_study(
            LogisticRegression, space, toy["X_train"], toy["y_train"], cv=3,
            n_trials=1, model="lr", seed=0, default_params={"C": 5.0},
        )
        assert dict(tuned) == {"C": 5.0} and tuned.trials is None


# ---------------------------------------------------------------------------------------
# TunedParams and the trials_<model> cell
# ---------------------------------------------------------------------------------------


class TestTrialLog:
    def test_trial_log_carries_the_fixed_params(self, toy):
        tuned = _lr_study(toy, n_trials=2, default_params={"C": 1.0})
        assert tuned.fixed == {"max_iter": 200}
        assert tuned.trial_log()["fixed"] == {"max_iter": 200}
        # A TunedParams pickled before the field existed still gives a log.
        del tuned.fixed
        assert "fixed" not in tuned.trial_log()

    def test_trial_log_and_pickle(self, toy):
        tuned = _lr_study(toy, n_trials=3, default_params={"C": 1.0})
        log = tuned.trial_log()
        assert log["metric"] == "balanced_accuracy" and log["best"] == tuned.best_trial
        assert [e["is_best"] for e in log["trials"]].count(True) == 1
        assert log["trials"][0]["is_default"] is True
        again = pickle.loads(pickle.dumps(tuned))
        assert again == tuned and again.best_trial == tuned.best_trial
        assert len(again.trials) == 3
        assert TunedParams({"C": 1.0}).trial_log() is None

    def test_modeleval_adds_the_cell_only_with_trials(self, toy):
        tuned = _lr_study(toy, n_trials=2, default_params={"C": 1.0})
        y = toy["y_test"]
        frame = modeleval(y, y, time.time(), tuned, {}, model="lr_opt", verbose=False,
                          tuned=True)
        cell = frame[TRIALS_PREFIX + "lr_opt"].iloc[0]
        assert cell["best"] == tuned.best_trial and len(cell["trials"]) == 2
        assert frame["results_lr_opt"].iloc[0]["tuning_score"] == tuned.score
        plain = modeleval(y, y, time.time(), TunedParams({"C": 1.0}), {}, model="lr_opt",
                          verbose=False, tuned=True)
        assert list(plain.columns) == [
            "y_test_lr_opt", "y_predicted_lr_opt", "y_score_lr_opt", "results_lr_opt",
        ]

    def test_record_tuned_params_adds_the_cell(self, toy):
        tuned = _lr_study(toy, n_trials=2, default_params={"C": 1.0})
        y = toy["y_test"]
        frame = modeleval(y, y, time.time(), {"C": 1.0}, {}, model="qsvc_opt",
                          verbose=False, tuned=True)
        frame = record_tuned_params(frame, tuned, time.time())
        assert frame[TRIALS_PREFIX + "qsvc_opt"].iloc[0]["best"] == tuned.best_trial
        untouched = modeleval(y, y, time.time(), {"C": 1.0}, {}, model="qsvc_opt",
                              verbose=False, tuned=True)
        record_tuned_params(untouched, TunedParams({"C": 1.0}), time.time())
        assert TRIALS_PREFIX + "qsvc_opt" not in untouched.columns


# ---------------------------------------------------------------------------------------
# run_function_study with a validation split
# ---------------------------------------------------------------------------------------


def _lr_compute(calls):
    """A quantum-shaped compute function: raw arrays in, one modeleval frame out."""

    def compute(X_train, X_test, y_train, y_test, args, C=1.0, model="fake", **kwargs):
        calls.append({"X_train": X_train, "X_test": X_test, "args": dict(args), "C": C,
                      "draw": float(np.random.rand())})
        estimator = LogisticRegression(C=C).fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        return modeleval(y_test, y_pred, time.time(), {"C": C}, args, model=model,
                         verbose=False, y_score=extract_binary_scores(estimator, X_test))

    return compute


class TestValidationFunctionStudy:
    def _run(self, toy, calls, args=None, reseed=None, n_trials=4, data_key="d_none_0_1"):
        return run_function_study(
            _lr_compute(calls), build_search_space("fake", {"C": {"low": 0.01, "high": 10.0}}),
            toy["X_train"], toy["y_train"],
            {"seed": 0, "kernel_dump_dir": "/nonexistent/dumps", **(args or {})},
            model="fake", n_trials=n_trials, seed=0, data_key=data_key,
            validation=toy["split"], default_params={"C": 1.0}, reseed=reseed,
        )

    def test_trials_fit_on_fit_rows_without_dumps_and_record_predictions(self, toy):
        calls = []
        tuned = self._run(toy, calls)
        split = toy["split"]
        assert len(calls) == 4 and tuned.n_trials == 4
        for call in calls:
            assert call["X_train"] is split.X_fit and call["X_test"] is split.X_val
            assert "kernel_dump_dir" not in call["args"]
        assert tuned.trials[0].is_default and tuned.trials[0].params == {"C": 1.0}
        record = tuned.trials[0]
        estimator = LogisticRegression(C=1.0).fit(split.X_fit, split.y_fit)
        np.testing.assert_array_equal(record.y_pred, estimator.predict(split.X_val))
        assert record.value == pytest.approx(
            balanced_accuracy_score(split.y_val, estimator.predict(split.X_val))
        )
        assert tuned.score == tuned.trials[tuned.best_trial].value
        assert tuned.trial_log()["fixed"] == {}

    def test_reseed_before_every_trial_and_before_the_refit(self, toy):
        calls, reseeds = [], []

        def reseed():
            reseeds.append(len(calls))
            np.random.seed(123)

        self._run(toy, calls, reseed=reseed, n_trials=3)
        # Before each of the 3 trials, then once more after the last one.
        assert reseeds == [0, 1, 2, 3]
        assert len({call["draw"] for call in calls}) == 1

    def test_freeze_is_neither_loaded_nor_allowed(self, toy, tmp_path):
        with pytest.raises(ValueError, match="freeze_quantum_params"):
            self._run(toy, [], args={"freeze_quantum_params": True,
                                     "quantum_param_dir": str(tmp_path)})
        self._run(toy, [], args={"quantum_param_dir": str(tmp_path)})
        assert list(tmp_path.iterdir()) == []

    def test_failed_trial_is_nan_and_the_study_continues(self, toy):
        def compute(X_train, X_test, y_train, y_test, args, C=1.0, **kwargs):
            if C == 1.0:
                raise RuntimeError("unbuildable corner")
            return _lr_compute([])(X_train, X_test, y_train, y_test, args, C=C)

        tuned = run_function_study(
            compute, build_search_space("fake", {"C": [1.0, 2.0]}), toy["X_train"],
            toy["y_train"], {"seed": 0}, model="fake", n_trials=5, seed=0,
            validation=toy["split"], default_params={"C": 1.0},
        )
        assert [t.state for t in tuned.trials] == ["FAIL", "COMPLETE"]
        assert np.isnan(tuned.trials[0].value) and dict(tuned) == {"C": 2.0}


def test_param_cache_refuses_a_validation_split(toy, tmp_path):
    from qbiocode.learning._param_cache import load_frozen_params, save_frozen_params

    args = {"freeze_quantum_params": True, "quantum_param_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="split_mode: manifest"):
        load_frozen_params(args, "d_pca_2_1", "qsvc", validation=toy["split"])
    with pytest.raises(ValueError, match="split_mode: manifest"):
        save_frozen_params(args, "d_pca_2_1", "qsvc", {"C": 1}, validation=toy["split"])


# ---------------------------------------------------------------------------------------
# model_run(..., validation=...)
# ---------------------------------------------------------------------------------------


def _stub_classical_opt(estimator_cls, name, calls):
    """What a G4-style ``compute_<m>_opt`` does with the fold keywords."""

    def compute_opt(X_train, X_test, y_train, y_test, args, verbose=False, cv=5,
                    model=name, random_state=None, *, tuner="optuna", n_trials=50,
                    validation=None, default_params=None, **grid):
        calls.append({"name": name, "n_trials": n_trials, "validation": validation,
                      "default_params": default_params, "random_state": random_state})
        best = search_hyperparameters(
            name, estimator_cls, grid, X_train, y_train, cv=cv, tuner=tuner,
            n_trials=n_trials, seed=random_state, scoring=tuning_scorer(args),
            validation=validation, default_params=default_params,
        )
        estimator = estimator_cls(**best).fit(X_train, y_train)
        return modeleval(y_test, estimator.predict(X_test), time.time(), best, args,
                         model=model, verbose=False, tuned=True,
                         y_score=extract_binary_scores(estimator, X_test))

    return compute_opt


@pytest.fixture
def stubbed(monkeypatch):
    import importlib

    # The package re-exports the functions under the module names; go to the modules.
    compute_lr = importlib.import_module("qbiocode.learning.compute_lr")
    compute_nb = importlib.import_module("qbiocode.learning.compute_nb")
    calls = []
    monkeypatch.setattr(compute_lr, "compute_lr_opt",
                        _stub_classical_opt(LogisticRegression, "lr", calls))
    monkeypatch.setattr(compute_nb, "compute_nb_opt",
                        _stub_classical_opt(GaussianNB, "nb", calls))
    return calls


def _fold_args(**overrides):
    args = {
        "model": ["lr", "nb"], "n_jobs": 1, "grid_search": True, "tuner": "optuna",
        "n_trials": 3, "seed": 7, "q_seed": 7, "cross_validation": 3,
        "gridsearch_lr_args": {"C": [0.1, 1.0, 10.0]},
        "gridsearch_nb_args": {"var_smoothing": [1e-9, 1e-2]},
        "lr_args": {"C": 10.0, "max_iter": 500, "verbose": 0},
    }
    args.update(overrides)
    return args


class TestModelRunFold:
    def _run(self, toy, args):
        from qbiocode.evaluation.model_run import model_run

        return model_run(toy["X_train"], toy["X_test"], toy["y_train"], toy["y_test"],
                         "toy_none_0_1", args, validation=toy["split"])

    def test_trials_cell_survives_into_the_summary(self, toy, stubbed):
        summary = self._run(toy, _fold_args())
        for label in ("lr_opt", "nb_opt"):
            log = summary[TRIALS_PREFIX + label][0]
            assert log["trials"][0]["is_default"]
            best = [e for e in log["trials"] if e["is_best"]]
            assert len(best) == 1
            row = summary["results_" + label][0]
            assert row["tuning_score"] == best[0]["value"]
            assert row["BestParams_Tuned"] == best[0]["params"]
        # And the protocol's writer reads it back.
        trials, predictions = trial_frames(summary, "toy_none_0_1")
        assert set(trials["model"]) == {"lr_opt", "nb_opt"}
        assert len(predictions) == len(trials) * toy["split"].n_val
        json.loads(trials["params"].iloc[0])

    def test_trial_rows_carry_the_fixed_params(self):
        from qbiocode.evaluation import protocol

        records = [protocol.TrialRecord(t, {"C": float(t + 1)}, 0.5, is_default=(t == 0))
                   for t in range(2)]
        summary = {
            TRIALS_PREFIX + "svc": {0: protocol.trial_log(
                records, metric="balanced_accuracy", best=0,
                fixed={"max_iter": 10_000_000, "kernel": "rbf"})},
            TRIALS_PREFIX + "nb": {0: protocol.trial_log(records, metric="f1", best=0)},
        }
        trials, _ = trial_frames(summary, "toy_none_0_1")
        assert list(trials.columns) == list(protocol.TRIAL_COLUMNS)
        svc = trials[trials["model"] == "svc"]
        assert [json.loads(f) for f in svc["fixed"]] == [
            {"kernel": "rbf", "max_iter": 10_000_000}] * 2
        assert trials.loc[trials["model"] == "nb", "fixed"].isna().all()

    def test_default_params_and_budget(self, toy, stubbed):
        self._run(toy, _fold_args())
        by_name = {call["name"]: call for call in stubbed}
        lr = by_name["lr"]["default_params"]
        # compute_lr's signature defaults, overridden by lr_args, minus reserved keys.
        assert lr["C"] == 10.0 and lr["max_iter"] == 500 and lr["solver"] == "saga"
        for reserved in ("verbose", "model", "data_key", "random_state", "validation"):
            assert reserved not in lr
        assert by_name["nb"]["default_params"] == {"var_smoothing": 1e-9}
        assert {call["n_trials"] for call in stubbed} == {3}
        assert all(call["validation"] is toy["split"] for call in stubbed)
        assert by_name["lr"]["random_state"] == 7

    @pytest.mark.parametrize("overrides, match", [
        ({"grid_search": False}, "grid_search is off"),
        ({"tuner": "grid"}, "tuner: 'grid'"),
        ({"freeze_quantum_params": True}, "freeze_quantum_params"),
        ({"model": ["lr", "qsvc"], "gridsearch_qsvc_args": {"C": [1.0]}},
         "tune_quantum is off"),
        ({"model": ["lr", "qensemble"]}, "qensemble"),
    ])
    def test_config_errors_before_any_fit(self, toy, stubbed, overrides, match):
        with pytest.raises(ValueError, match=match):
            self._run(toy, _fold_args(**overrides))
        assert stubbed == []

    def test_quantum_gets_one_budget_and_a_reseed(self, toy, stubbed, monkeypatch, caplog):
        import importlib

        from qbiocode.evaluation.model_run import _Reseed

        compute_qsvc = importlib.import_module("qbiocode.learning.compute_qsvc")

        seen = {}

        def fake_qsvc_opt(X_train, X_test, y_train, y_test, args, verbose=False,
                          model="qsvc", data_key="", n_trials=10, validation_split=0.25,
                          validation=None, default_params=None, reseed=None, **grid):
            seen.update(n_trials=n_trials, reseed=reseed, default_params=default_params,
                        validation=validation)
            return _stub_classical_opt(GaussianNB, "nb", [])(
                X_train, X_test, y_train, y_test, {}, model=model, n_trials=1,
                validation=validation, var_smoothing=[1e-9],
            )

        monkeypatch.setattr(compute_qsvc, "compute_qsvc_opt", fake_qsvc_opt)
        args = _fold_args(model=["lr", "qsvc"], tune_quantum=True, n_trials_quantum=32,
                          gridsearch_qsvc_args={"C": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger="qbiocode.evaluation.model_run"):
            self._run(toy, args)
        assert "n_trials_quantum=32 is ignored" in caplog.text
        assert seen["n_trials"] == 3 and seen["validation"] is toy["split"]
        assert isinstance(seen["reseed"], _Reseed)
        assert seen["default_params"]["encoding"] == "ZZ"
        assert "local_optimizer" in seen["default_params"]
        assert "reseed" not in {c["name"] for c in stubbed}  # classical gets none


def test_reseed_resets_the_global_streams():
    import random

    from qbiocode.evaluation.model_run import _Reseed

    reseed = pickle.loads(pickle.dumps(_Reseed(3, 5)))
    reseed()
    first = (np.random.rand(), random.random())
    np.random.rand(10)
    reseed()
    assert (np.random.rand(), random.random()) == first


def test_reserved_fold_keys_are_dropped_from_config_blocks(toy, caplog):
    from qbiocode.evaluation.model_run import model_run

    args = {"model": ["nb"], "n_jobs": 1, "seed": 0,
            "nb_args": {"validation": 1, "default_params": {}, "reseed": None}}
    with caplog.at_level(logging.WARNING, logger="qbiocode.evaluation.model_run"):
        summary = model_run(toy["X_train"], toy["X_test"], toy["y_train"], toy["y_test"],
                            "toy_none_0_1", args)
    for key in ("validation", "default_params", "reseed"):
        assert f"ignoring {key!r}" in caplog.text
    assert not any(k.startswith(TRIALS_PREFIX) for k in summary)


# ---------------------------------------------------------------------------------------
# split_mode: internal is unchanged
# ---------------------------------------------------------------------------------------

# Golden values: taken from the tree before the fold-based protocol landed (the same
# calls, same seeds), so a change to the internal path shows up here as a number moving.


def test_internal_run_study_matches_the_pre_change_numbers():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = (X[:, 0] + 0.8 * rng.randn(60) > 0).astype(int)
    space = build_search_space("lr", {"C": {"low": 0.01, "high": 10.0, "log": True}})
    tuned = run_study(LogisticRegression, space, X[:48], y[:48], cv=3, n_trials=4,
                      model="lr", seed=0, fixed={"max_iter": 200},
                      scoring=tuning_scorer({}))
    assert dict(tuned) == {"C": pytest.approx(0.44303752452182665)}
    assert tuned.score == pytest.approx(0.7804232804232805)
    assert tuned.n_trials == 4
    assert tuned.trials is None and tuned.trial_log() is None

    searched = search_hyperparameters(
        "nb", GaussianNB, {"var_smoothing": [1e-9, 1e-2, 1e-1]}, X[:48], y[:48], cv=3,
        n_trials=3, seed=0, scoring=tuning_scorer({}),
    )
    assert dict(searched) == {"var_smoothing": 0.01}
    assert searched.score == pytest.approx(0.7804232804232805)
    assert searched.n_trials == 3
    assert searched.trial_log() is None
