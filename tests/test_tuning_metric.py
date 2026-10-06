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

"""Every tuner selects on ``tuning_metric``, and reports how well its choice scored.

Before this, the classical tuners scored candidates with the estimator's own
``score`` (accuracy), ``GridSearchCV`` likewise, and the quantum objective averaged
``metrics["accuracy"]`` -- while the study reports balanced accuracy on a corpus with
several 90/10 datasets. Tuning on accuracy there prefers the configuration that
predicts the majority class. The pins below:

* on a 90/10 toy, ``class_weight`` in ``[None, 'balanced']`` is chosen differently under
  ``accuracy`` and ``balanced_accuracy``, for both classical engines, and ``accuracy``
  restores the old choice;
* the quantum objective reads the configured modeleval key;
* an unknown metric is refused, by ``model_run`` before anything is fitted;
* the returned :class:`TunedParams` is a plain dict to every consumer, and carries the
  score -- which reaches the results row as ``tuning_metric``/``tuning_score``/
  ``tuning_reused`` for tuned models only;
* frozen reuse marks ``tuning_reused``, logs its optimism warning once, and an old
  payload without a score still loads.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.learning import _param_cache
from qbiocode.learning._param_cache import (
    CACHE_DIR_KEY,
    FREEZE_KEY,
    LEGACY_TUNING_METRIC,
    load_frozen_params,
    save_frozen_params,
)
from qbiocode.learning._tuning import (
    DEFAULT_TUNING_METRIC,
    TunedParams,
    TuningScorer,
    build_search_space,
    record_tuned_params,
    run_function_study,
    search_hyperparameters,
    tuning_metric,
    tuning_scorer,
)

CANDIDATES = {"class_weight": [None, "balanced"]}


@pytest.fixture
def imbalanced():
    """200 rows, ~10% positives, weakly separable: accuracy prefers the unweighted
    model (it predicts the majority class) and balanced accuracy the weighted one."""
    rng = np.random.RandomState(0)
    y = (rng.rand(200) < 0.1).astype(int)
    X = rng.randn(200, 3) + y[:, None] * 0.8
    return X, y


def _tune(X, y, tuner, metric):
    kwargs = {} if metric is None else {"scoring": tuning_scorer({"tuning_metric": metric})}
    return search_hyperparameters(
        "lr", LogisticRegression, CANDIDATES, X, y, cv=3, tuner=tuner, seed=0, **kwargs
    )


class TestTheMetricChangesTheChoice:
    @pytest.mark.parametrize("tuner", ["optuna", "grid"])
    def test_accuracy_and_balanced_accuracy_pick_differently(self, imbalanced, tuner):
        X, y = imbalanced
        by_accuracy = _tune(X, y, tuner, "accuracy")
        by_balanced = _tune(X, y, tuner, "balanced_accuracy")
        assert by_accuracy == {"class_weight": None}
        assert by_balanced == {"class_weight": "balanced"}
        assert by_accuracy.metric == "accuracy"
        assert by_balanced.metric == "balanced_accuracy"

    @pytest.mark.parametrize("tuner", ["optuna", "grid"])
    def test_the_default_is_balanced_accuracy(self, imbalanced, tuner):
        X, y = imbalanced
        default = _tune(X, y, tuner, None)
        assert default == {"class_weight": "balanced"}
        assert default.metric == DEFAULT_TUNING_METRIC == "balanced_accuracy"

    @pytest.mark.parametrize("tuner", ["optuna", "grid"])
    def test_accuracy_reproduces_the_old_default_scorer(self, imbalanced, tuner):
        """'accuracy' is the estimator's own ``score``, which is what ran before."""
        from sklearn.model_selection import GridSearchCV, cross_val_score

        X, y = imbalanced
        best = _tune(X, y, tuner, "accuracy")
        if tuner == "grid":
            old = GridSearchCV(LogisticRegression(), param_grid={"class_weight": [None, "balanced"]}, cv=3)
            old.fit(X, y)
            assert best == old.best_params_
            assert best.score == pytest.approx(old.best_score_, abs=0)
        else:
            old = cross_val_score(LogisticRegression(**best), X, y, cv=3).mean()
            assert best.score == pytest.approx(old, abs=0)

    def test_the_score_is_the_best_mean_cv_score(self, imbalanced):
        from sklearn.model_selection import cross_val_score

        X, y = imbalanced
        best = _tune(X, y, "optuna", "balanced_accuracy")
        expected = cross_val_score(
            LogisticRegression(**best), X, y, cv=3, scoring="balanced_accuracy"
        ).mean()
        assert best.score == pytest.approx(expected)
        assert best.n_trials == 2
        assert best.reused is False


class TestTheMetricIsValidated:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown tuning_metric 'roc'"):
            tuning_metric({"tuning_metric": "roc"})

    def test_absent_means_the_default(self):
        assert tuning_metric({}) == "balanced_accuracy"
        assert tuning_metric(None) == "balanced_accuracy"

    def test_model_run_rejects_it_before_fitting(self, imbalanced):
        """The upfront check fires in ``model_run`` itself, not inside a worker."""
        from qbiocode.evaluation.model_run import model_run

        X, y = imbalanced
        args = {
            "model": ["dt"], "n_jobs": 1, "grid_search": True, "seed": 0,
            "tuning_metric": "precision", "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with pytest.raises(ValueError, match="Unknown tuning_metric 'precision'"):
            model_run(X[:150], X[150:], y[:150], y[150:], "k_pca_2_0", args)

    def test_f1_follows_the_configured_average(self):
        scorer = tuning_scorer({"tuning_metric": "f1_score", "average": "macro"})
        assert scorer.metric == "f1_score" and scorer.average == "macro"
        assert tuning_scorer({"tuning_metric": "f1_score"}).average == "weighted"

    @pytest.mark.parametrize("metric", ["accuracy", "balanced_accuracy", "mcc", "f1_score"])
    def test_each_scorer_equals_the_modeleval_key(self, imbalanced, metric):
        """The table's promise: the classical scorer and modeleval compute one number."""
        X, y = imbalanced
        estimator = LogisticRegression().fit(X, y)
        row = modeleval(y, estimator.predict(X), 0.0, {}, {}, "m", verbose=False)
        expected = row["results_m"][0][metric]
        assert TuningScorer(metric)(estimator, X, y) == pytest.approx(expected)


def _stub_frame(accuracy, balanced):
    return pd.DataFrame(
        {"results_stub": [{"accuracy": accuracy, "balanced_accuracy": balanced}]}
    )


def _stub(X_train, X_test, y_train, y_test, args, *, c):
    # c=0 wins on accuracy, c=1 on balanced accuracy.
    return _stub_frame(0.9, 0.5) if c == 0 else _stub_frame(0.8, 0.7)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 2)), np.array([0, 1] * 20)


class TestTheQuantumObjective:
    @pytest.mark.parametrize("metric,expected", [
        ("accuracy", 0), ("balanced_accuracy", 1), (None, 1),
    ])
    def test_it_reads_the_configured_key(self, small, metric, expected):
        X, y = small
        args = {"backend": "simulator"}
        if metric is not None:
            args["tuning_metric"] = metric
        best = run_function_study(
            _stub, build_search_space("stub", {"c": [0, 1]}), X, y, args,
            model="stub", n_trials=2, seed=0,
        )
        assert best == {"c": expected}
        assert best.metric == (metric or "balanced_accuracy")
        assert best.score == pytest.approx(0.9 if expected == 0 else 0.7)
        assert best.reused is False

    def test_an_unknown_metric_raises(self, small):
        X, y = small
        with pytest.raises(ValueError, match="Unknown tuning_metric"):
            run_function_study(
                _stub, build_search_space("stub", {"c": [0, 1]}), X, y,
                {"backend": "simulator", "tuning_metric": "auc"},
                model="stub", n_trials=2, seed=0,
            )

    def test_multi_head_frames_are_averaged(self):
        from qbiocode.learning._tuning import _metric_of

        frame = pd.DataFrame({
            "results_qpl_rf": [{"balanced_accuracy": 0.6}, np.nan],
            "results_qpl_lr": [np.nan, {"balanced_accuracy": 0.8}],
        })
        assert _metric_of(frame, "qpl", "balanced_accuracy") == pytest.approx(0.7)

    def test_record_tuned_params_writes_the_evidence(self):
        frame = pd.DataFrame({"results_q": [{"accuracy": 1.0, "BestParams_Tuned": {"k": 1}}]})
        best = TunedParams({"reps": 2}, metric="balanced_accuracy", score=0.75)
        row = record_tuned_params(frame, best, 0.0)["results_q"][0]
        assert row["tuning_metric"] == "balanced_accuracy"
        assert row["tuning_score"] == 0.75
        assert row["tuning_reused"] is False
        assert type(row["BestParams_Tuned"]) is dict
        assert row["BestParams_Tuned"] == {"k": 1, "reps": 2}


class TestTunedParamsIsAPlainDict:
    def _params(self):
        return TunedParams(
            {"C": 1.5, "kernel": "rbf"}, metric="mcc", score=0.4, n_trials=7, reused=True
        )

    def test_text_and_equality(self):
        p = self._params()
        plain = {"C": 1.5, "kernel": "rbf"}
        assert str(p) == str(plain) and repr(p) == repr(plain)
        assert p == plain and plain == p
        assert json.dumps(p) == json.dumps(plain)
        assert LogisticRegression(**{"C": p["C"]}).C == 1.5

    def test_pickle_keeps_the_attributes(self):
        p = pickle.loads(pickle.dumps(self._params()))
        assert isinstance(p, TunedParams) and p == {"C": 1.5, "kernel": "rbf"}
        assert (p.metric, p.score, p.n_trials, p.reused) == ("mcc", 0.4, 7, True)

    def test_it_survives_a_loky_worker(self):
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=2, backend="loky")(
            delayed(_identity)(self._params()) for _ in range(2)
        )
        for p in out:
            assert isinstance(p, TunedParams)
            assert (p.metric, p.score, p.reused) == ("mcc", 0.4, True)


def _identity(value):
    return value


class TestResultsRows:
    def test_a_tuned_row_carries_the_three_columns(self, imbalanced):
        X, y = imbalanced
        best = TunedParams({"C": 1.0}, metric="balanced_accuracy", score=0.61)
        row = modeleval(y, y, 0.0, best, {}, "lr_opt", verbose=False)["results_lr_opt"][0]
        assert (row["tuning_metric"], row["tuning_score"], row["tuning_reused"]) == (
            "balanced_accuracy", 0.61, False,
        )
        assert type(row["BestParams_Tuned"]) is dict
        assert str(row["BestParams_Tuned"]) == "{'C': 1.0}"

    def test_an_untuned_row_carries_none_of_them(self, imbalanced):
        X, y = imbalanced
        row = modeleval(y, y, 0.0, {"C": 1.0}, {}, "lr", verbose=False)["results_lr"][0]
        assert not {"tuning_metric", "tuning_score", "tuning_reused"} & set(row)

    def test_model_run_reports_them_for_a_tuned_model(self, imbalanced):
        from qbiocode.evaluation.model_run import model_run

        X, y = imbalanced
        args = {
            "model": ["dt"], "n_jobs": 1, "grid_search": True, "seed": 0,
            "cross_validation": 3, "n_trials": 2, "tuning_metric": "accuracy",
            "gridsearch_dt_args": {"max_depth": [2, 3]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = model_run(X[:150], X[150:], y[:150], y[150:], "k_pca_2_0", args)
        row = out["results_dt_opt"][0]
        assert row["tuning_metric"] == "accuracy"
        assert 0.0 < row["tuning_score"] <= 1.0
        assert row["tuning_reused"] is False
        assert type(row["BestParams_Tuned"]) is dict
        assert row["BestParams_Tuned"]["max_depth"] in (2, 3)


@pytest.fixture
def freeze_args(tmp_path):
    return {"backend": "simulator", "seed": 1, FREEZE_KEY: True,
            CACHE_DIR_KEY: str(tmp_path / "frozen")}


class TestFrozenReuse:
    def test_reuse_is_marked_and_warned_about_once(self, small, freeze_args, caplog, monkeypatch):
        monkeypatch.setattr(_param_cache, "_REUSE_WARNED", False)
        X, y = small
        space = build_search_space("stub", {"c": [0, 1]})
        study = lambda key: run_function_study(  # noqa: E731
            _stub, space, X, y, freeze_args, model="stub", n_trials=2, seed=0, data_key=key,
        )
        first = study("d_pca_2_0")
        assert first.reused is False
        with caplog.at_level(logging.WARNING, logger=_param_cache.__name__):
            second = study("d_pca_2_1")
            third = study("d_pca_2_2")
        assert second == first == third == {"c": 1}
        assert second.reused is True and third.reused is True
        assert second.score == pytest.approx(first.score)
        assert second.metric == "balanced_accuracy"
        optimistic = [r for r in caplog.records if "OPTIMISTIC" in r.getMessage()]
        assert len(optimistic) == 1
        assert "freeze_quantum_params: False" in optimistic[0].getMessage()

    def test_the_payload_stores_score_and_metric_but_params_stay_clean(self, freeze_args):
        path = save_frozen_params(freeze_args, "d_pca_2_0", "stub", {"c": 1},
                                  metric="mcc", score=0.3)
        with open(path) as handle:
            payload = json.load(handle)
        assert payload["params"] == {"c": 1}
        assert payload["tuning_metric"] == "mcc" and payload["tuning_score"] == 0.3
        loaded = load_frozen_params(freeze_args, "d_pca_2_3", "stub", metric="mcc")
        assert loaded == {"c": 1} and set(loaded) == {"c"}
        assert loaded.reused and loaded.score == 0.3

    def test_an_old_payload_without_a_score_still_loads(self, freeze_args):
        path = save_frozen_params(freeze_args, "d_pca_2_0", "stub", {"c": 0})
        with open(path) as handle:
            payload = json.load(handle)
        assert "tuning_score" not in payload and "tuning_metric" not in payload
        loaded = load_frozen_params(freeze_args, "d_pca_2_1", "stub", metric="accuracy")
        assert loaded == {"c": 0} and loaded.reused
        assert math.isnan(loaded.score)
        # Pre-``tuning_metric`` payloads were all accuracy-tuned, and the row says so.
        assert loaded.metric == LEGACY_TUNING_METRIC == "accuracy"

    def test_an_old_payload_is_retuned_under_a_non_accuracy_metric(self, freeze_args, caplog):
        save_frozen_params(freeze_args, "d_pca_2_0", "stub", {"c": 0})
        with caplog.at_level(logging.WARNING, logger="qbiocode.learning._param_cache"):
            assert load_frozen_params(
                freeze_args, "d_pca_2_1", "stub", metric="balanced_accuracy"
            ) is None
        assert any("selected on accuracy" in r.getMessage() for r in caplog.records)

    def test_an_old_payload_loaded_without_a_metric_reports_accuracy(self, freeze_args):
        save_frozen_params(freeze_args, "d_pca_2_0", "stub", {"c": 0})
        loaded = load_frozen_params(freeze_args, "d_pca_2_1", "stub")
        assert loaded == {"c": 0} and loaded.metric == "accuracy"

    def test_a_payload_tuned_on_another_metric_is_retuned(self, freeze_args):
        save_frozen_params(freeze_args, "d_pca_2_0", "stub", {"c": 0},
                           metric="accuracy", score=0.9)
        assert load_frozen_params(
            freeze_args, "d_pca_2_1", "stub", metric="balanced_accuracy"
        ) is None


def test_tuning_evidence_keys_match_the_shared_column_constant():
    """Consumers exclude the evidence block by ``TUNING_EVIDENCE_COLUMNS``; the row
    keys ``TunedParams.evidence`` writes must be exactly those names."""
    from qbiocode.evaluation.model_evaluation import TUNING_EVIDENCE_COLUMNS

    assert tuple(TunedParams({"a": 1}).evidence()) == TUNING_EVIDENCE_COLUMNS
