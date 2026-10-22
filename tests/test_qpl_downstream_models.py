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

"""QPL's configurable downstream models: selection, per-model tuning, and reuse.

QPL is the generalised PQK -- project the rows through a quantum feature map, then fit a
*configurable* downstream classical model on the projection, where ``compute_pqk`` fits
one fixed SVC. Three properties of that arrangement are asserted here, and they are
easy to confuse with each other:

* **The projection is not optimised by** :func:`~qbiocode.learning.compute_qpl.compute_qpl`
  -- it runs at whatever it is given. :func:`~qbiocode.learning.compute_qpl.compute_qpl_opt`
  is what searches ``encoding``/``reps``/``entanglement``/``data_map``/``bandwidth``,
  with Optuna, scoring a candidate by the mean tuning metric across the downstream
  models (``tests/test_quantum_tuning.py`` covers that search itself).
* **Each downstream model IS optimised**, inside one ``compute_qpl`` fit, over its own
  model-specific grid. So an untuned QPL run still tunes its downstream models; the two
  optimisations are independent and stack.
* **The projection is computed once** however many downstream models are selected. This
  is the expensive half -- one circuit simulation per row -- and the tests below pin it
  by counting backend sessions and cached projection files, not by timing.

The per-model grids reach the library's shared validation
(:func:`qbiocode.learning._grid.build_param_grid`), the same one every classical
``compute_*_opt`` uses, so a malformed QPL head block and a malformed
``gridsearch_<model>_args`` block are refused identically.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.model_selection import RandomizedSearchCV

import qbiocode.utils.qutils as qutils
from qbiocode.learning.compute_qpl import (
    HEAD_CV,
    HEAD_N_ITER,
    QPL_DEFAULT_HEADS,
    QPL_HEADS,
    build_head,
    compute_qpl,
)
from qbiocode.learning.compute_tabpfn import TABPFN_MAX_CLASSES, tabpfn_is_available

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: The cheap downstream models. ``xgb``/``catboost``/``tabpfn`` are exercised
#: individually where they are the point; using them everywhere would make this module
#: cost minutes for nothing.
CHEAP = ("lr", "svc")


def _problem(n_features=2, n_train=10, n_test=4, seed=0, n_classes=2):
    """A tiny binary problem. Deliberately small: these tests assert on structure --
    which columns come back, how many projections were written -- not on accuracy, and
    every extra row is another circuit simulation."""
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.uniform(0, 1, (total, n_features))
    y = np.array([i % n_classes for i in range(total)])
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _args(tmp_path, **extra):
    """The keys ``compute_qpl`` reads, with the projection cache under a temp directory.

    ``compute_qpl`` defaults its cache to a CWD-relative ``qpl_projections``, so without
    the redirect these tests would write into the checkout and, worse, read each other's
    projections.
    """
    args = dict(
        backend="simulator", seed=42, q_seed=42, shots=100, resil_level=1,
        average="weighted", multi_class="raise", grid_search=False,
        cross_validation=2, qpl_projection_dir=str(tmp_path),
    )
    args.update(extra)
    return args


def _run(tmp_path, **kwargs):
    """One real ``compute_qpl`` run: quantum projection, then the downstream models."""
    X_train, X_test, y_train, y_test = _problem(**kwargs.pop("problem", {}))
    kwargs.setdefault("classical_models", list(CHEAP))
    # setdefault rather than positional, so a caller may pin the projection settings it
    # is asserting on without colliding with these.
    for key, value in (("encoding", "ZZ"), ("entanglement", "linear"), ("reps", 1)):
        kwargs.setdefault(key, value)
    return compute_qpl(
        X_train, X_test, y_train, y_test, _args(tmp_path), data_key="t", **kwargs,
    )


def _result_labels(frame):
    """The ``results_<label>`` labels of a frame, which are what downstream readers key on."""
    return sorted(
        c[len("results_"):] for c in frame.columns if c.startswith("results_")
    )


def _projection_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".npy"))


def _cell(frame, column):
    """The one populated value of ``column``, whatever row it is on.

    ``compute_qpl`` concatenates one ``modeleval`` frame per downstream model, so a
    multi-model frame has one row per model and each ``results_<label>`` column is
    populated on its own row and NaN on the others. ``iloc[0]`` therefore reads NaN for
    every model but the first -- the same reason
    :func:`qbiocode.learning._tuning._metric_dicts` scans for the populated cell.
    """
    for value in frame[column]:
        if isinstance(value, (dict, np.ndarray, list)):
            return value
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    raise AssertionError(f"{column} is populated on no row of the frame")


def _available(heads):
    """``heads`` minus the ones this install cannot provide.

    ``compute_qpl`` warns and drops a model whose library is unusable -- an xgboost
    whose native extension cannot load (no libomp on macOS), a broken catboost, an
    absent ``[tabpfn]`` extra. That is the behaviour under test elsewhere in this
    module, so here it is accounted for rather than asserted against: a test of *which
    models were selected* must not fail because of what is installed.
    """
    from qbiocode.learning.compute_qpl import CATBOOST_AVAILABLE, XGBOOST_AVAILABLE

    usable = {
        "xgb": XGBOOST_AVAILABLE,
        "catboost": CATBOOST_AVAILABLE,
        "tabpfn": tabpfn_is_available(),
    }
    return [head for head in heads if usable.get(head, True)]


# ----------------------------------------------------------------------------------
# Default behaviour, and backward compatibility
# ----------------------------------------------------------------------------------


class TestDefaultBehaviour:
    """What a QPL run does when nothing new is configured, which must not have moved."""

    def test_the_default_model_list_is_unchanged(self):
        """The six models QPL has always run, in order, with TabPFN still opt-in.

        TabPFN needs the optional ``[tabpfn]`` extra, and ``compute_qpl`` warns and
        drops an unavailable model -- so defaulting it on would make every QPL run in an
        ordinary install warn about something the user never asked for.
        """
        assert QPL_DEFAULT_HEADS == ("rf", "mlp", "svc", "lr", "xgb", "catboost")
        assert "tabpfn" in QPL_HEADS and "tabpfn" not in QPL_DEFAULT_HEADS

    def test_omitting_classical_models_runs_the_default_list(self, tmp_path):
        """``classical_models=None`` must still mean the default six, not the new path."""
        frame = _run(tmp_path, classical_models=None)
        assert _result_labels(frame) == sorted(
            f"qpl_{head}" for head in _available(QPL_DEFAULT_HEADS)
        )

    def test_a_default_run_records_no_new_parameter_keys(self, tmp_path):
        """A row from a default run must be comparable with rows recorded before this
        change, so the new per-model block appears only when it was actually used."""
        frame = _run(tmp_path)
        for head in CHEAP:
            params = _cell(frame, f"results_qpl_{head}")["Model_Parameters"]
            assert "head_search_args" not in params
            assert set(params) == {"feature_map", "feature_map_reps", "entanglement",
                                   "best_params"}

    def test_the_legacy_factories_still_build_their_models(self):
        """``create_<head>_model`` is public API and is called from outside this module
        (``compute_pqk`` has its own ``create_svc_model``; the tests and CHANGELOG name
        these). They are now thin delegates to ``build_head``, which must not change
        what they return."""
        from qbiocode.learning.compute_qpl import (
            create_catboost_model, create_lr_model, create_mlp_model, create_rf_model,
            create_svc_model, create_xgb_model,
        )

        for head, factory, expected in (
            ("rf", create_rf_model, "RandomForestClassifier"),
            ("mlp", create_mlp_model, "MLPClassifier"),
            ("svc", create_svc_model, "SVC"),
            ("lr", create_lr_model, "LogisticRegression"),
            ("xgb", create_xgb_model, "XGBClassifier"),
            ("catboost", create_catboost_model, "CatBoostClassifier"),
        ):
            if head not in _available([head]):
                continue  # this install cannot load that library; see _available
            model = factory(42)
            assert isinstance(model, RandomizedSearchCV)
            assert type(model.estimator).__name__ == expected
            assert model.random_state == 42


# ----------------------------------------------------------------------------------
# Selecting one or several downstream models
# ----------------------------------------------------------------------------------


class TestSelectingDownstreamModels:

    @pytest.mark.parametrize("head", CHEAP)
    def test_a_single_downstream_model_yields_exactly_one_row(self, tmp_path, head):
        frame = _run(tmp_path / head, classical_models=[head])
        assert _result_labels(frame) == [f"qpl_{head}"]

    def test_a_bare_string_selects_one_model(self, tmp_path):
        """``classical_models: svc`` in YAML is a string, not a list. Read as a sequence
        it would have been four unknown models ('s', 'v', 'c'), so this is accepted
        explicitly rather than left to iterate."""
        frame = _run(tmp_path, classical_models="svc")
        assert _result_labels(frame) == ["qpl_svc"]

    def test_several_downstream_models_each_get_their_own_row(self, tmp_path):
        frame = _run(tmp_path, classical_models=["lr", "svc", "rf"])
        assert _result_labels(frame) == ["qpl_lr", "qpl_rf", "qpl_svc"]

    def test_every_row_carries_its_own_predictions_and_metrics(self, tmp_path):
        """One row per downstream model is only useful if the rows differ: a fan-out
        that reported the same numbers under several names would be worse than one row."""
        frame = _run(tmp_path, classical_models=["lr", "svc"])
        for head in ("lr", "svc"):
            metrics = _cell(frame, f"results_qpl_{head}")
            assert metrics["model"] == f"qpl_{head}"
            assert np.isfinite(metrics["accuracy"])
            assert _cell(frame, f"y_predicted_qpl_{head}") is not None


# ----------------------------------------------------------------------------------
# The projection is computed once, whatever the downstream models
# ----------------------------------------------------------------------------------


class TestTheProjectionIsComputedOnce:
    """The reason a multi-model QPL run is affordable at all.

    One circuit simulation per row dominates a QPL fit; the downstream searches are
    milliseconds of classical work on an already-projected matrix. If the projection
    were recomputed per downstream model, selecting six would cost six projections --
    and nothing in the reported numbers would say so, because every projection of the
    same data at the same settings is identical. Hence these assertions count *work*,
    not wall clock.
    """

    def test_one_backend_session_serves_every_downstream_model(self, tmp_path, monkeypatch):
        """The projection is built inside one ``get_backend_session``; a per-model
        projection would open one session per model."""
        calls = []
        original = qutils.get_backend_session

        def spy(*args, **kwargs):
            calls.append(kwargs.get("num_qubits"))
            return original(*args, **kwargs)

        monkeypatch.setattr(qutils, "get_backend_session", spy)
        frame = _run(tmp_path, classical_models=["lr", "svc", "rf"])
        assert len(_result_labels(frame)) == 3
        assert len(calls) == 1, (
            f"three downstream models opened {len(calls)} backend sessions; the quantum "
            f"projection is being recomputed per model"
        )

    def test_one_train_test_projection_pair_however_many_models(self, tmp_path):
        """The cache holds exactly one (train, test) pair: the models share a fingerprint
        because the projection does not depend on which model consumes it."""
        _run(tmp_path, classical_models=["lr", "svc", "rf"])
        files = _projection_files(tmp_path)
        assert len(files) == 2, files
        assert sum(f.endswith("_train.npy") for f in files) == 1

    def test_each_split_is_projected_into_its_own_file(self, tmp_path):
        """Which split a cache file holds must not be re-derived from its path.

        The two write loops used to decide with ``"train" in f_tr`` against the WHOLE
        path, directory included. So any projection directory whose path contained
        "train" -- ``~/runs/trained/``, a ``training_data`` dataset directory, or a
        pytest ``tmp_path`` for a test whose own name says "train" -- matched the train
        branch for the *test* file too, and the test projection was filled with the
        TRAINING rows.

        Where the splits differ in size that surfaces far away as "Found input variables
        with inconsistent numbers of samples" from inside a downstream model's fit,
        naming neither the cache nor the path. Where they happen to match in size there
        is no error at all and every model is scored on its own training rows. Hence the
        directory name here is deliberate, not incidental.
        """
        directory = tmp_path / "trained_models"
        directory.mkdir()
        X_train, X_test, y_train, y_test = _problem(n_train=10, n_test=4)
        compute_qpl(
            X_train, X_test, y_train, y_test, _args(directory), data_key="t",
            encoding="ZZ", entanglement="linear", reps=1, classical_models=["lr"],
        )
        rows = {
            ("train" if f.endswith("_train.npy") else "test"): len(np.load(directory / f))
            for f in _projection_files(directory)
        }
        assert rows == {"train": len(X_train), "test": len(X_test)}

    def test_the_projection_is_identical_whichever_models_are_selected(self, tmp_path):
        """Selecting more models must not change the projection, only what reads it.

        Both runs share one cache directory, so the second finds the first's file by
        fingerprint and must neither rewrite nor invalidate it.
        """
        _run(tmp_path, classical_models=["lr"])
        first = _projection_files(tmp_path)
        before = np.load(tmp_path / first[0])
        mtimes = {f: os.path.getmtime(tmp_path / f) for f in first}

        _run(tmp_path, classical_models=["lr", "svc", "rf"])
        assert _projection_files(tmp_path) == first, "a second run added a projection"
        assert np.array_equal(np.load(tmp_path / first[0]), before)
        assert {f: os.path.getmtime(tmp_path / f) for f in first} == mtimes, (
            "the cached projection was rewritten rather than reused"
        )

    def test_every_downstream_model_is_fitted_on_the_projection_not_the_raw_features(
        self, tmp_path
    ):
        """The projection is 3n wide (``<X>``, ``<Y>``, ``<Z>`` per qubit), the raw data
        n. A model fitted on the raw features would be a silently different experiment,
        and on these shapes would not even raise."""
        X_train, X_test, y_train, y_test = _problem(n_features=2)
        frame = compute_qpl(
            X_train, X_test, y_train, y_test, _args(tmp_path), data_key="t",
            encoding="ZZ", entanglement="linear", reps=1, classical_models=["lr"],
        )
        best = _cell(frame, "results_qpl_lr")["Model_Parameters"]["best_params"]
        assert best  # the search ran
        train_file = next(f for f in _projection_files(tmp_path) if "train" in f)
        projection = np.load(tmp_path / train_file)
        assert projection.reshape(len(X_train), -1).shape[1] == 3 * X_train.shape[1]


# ----------------------------------------------------------------------------------
# Per-downstream-model hyperparameter optimisation
# ----------------------------------------------------------------------------------


class TestDownstreamModelOptimisation:
    """Each downstream model is searched over ITS OWN hyperparameters.

    This is what makes QPL's downstream stage comparable with a standalone classical
    arm rather than a single default-configured fit, and it happens whether or not the
    projection is being tuned.
    """

    @pytest.mark.parametrize("head", ["rf", "mlp", "svc", "lr"])
    def test_each_model_is_searched_over_its_own_grid(self, head):
        """A shared grid would be meaningless -- ``C`` means nothing to a random forest
        -- so the grids must differ per model and name that model's parameters."""
        model = build_head(head, 42)
        assert isinstance(model, RandomizedSearchCV)
        grid = model.param_distributions
        assert grid, f"{head} is not searched over anything"
        expected = {
            "rf": "n_estimators", "mlp": "hidden_layer_sizes",
            "svc": "kernel", "lr": "penalty",
        }[head]
        assert expected in grid, sorted(grid)

    def test_every_searched_model_shares_one_budget(self):
        """Equal budgets are what make the reported rows comparable with each other.
        The seven factories used to spell ``n_iter=40, cv=5`` out individually."""
        for head in _available(QPL_HEADS):
            if head == "tabpfn":
                continue  # no search; see TestTabPfnAsADownstreamModel
            model = build_head(head, 42)
            assert (model.n_iter, model.cv) == (HEAD_N_ITER, HEAD_CV), head

    def test_the_search_reports_the_configuration_it_chose(self, tmp_path):
        """``best_params`` must be the search's winner, drawn from its own grid -- not
        the estimator's defaults, which is what an unfitted or bypassed search gives."""
        frame = _run(tmp_path, classical_models=["svc"])
        best = _cell(frame, "results_qpl_svc")["Model_Parameters"]["best_params"]
        assert set(best) == {"C", "gamma", "kernel"}
        assert best["kernel"] in ("linear", "rbf", "poly", "sigmoid")

    def test_a_config_can_replace_one_model_s_grid(self, tmp_path):
        """The point of routing these through the shared validation: a QPL downstream
        grid is configurable, as a standalone model's ``gridsearch_<model>_args`` is."""
        frame = _run(
            tmp_path, classical_models=["svc"],
            classical_models_args={"svc": {"C": [10], "kernel": ["linear"]}},
        )
        params = _cell(frame, "results_qpl_svc")["Model_Parameters"]
        assert params["best_params"] == {"C": 10, "kernel": "linear"}
        # Recorded, because `best_params` alone cannot say what was searched to find it.
        assert params["head_search_args"] == {"C": [10], "kernel": ["linear"]}

    def test_an_override_reaches_only_the_model_it_names(self, tmp_path):
        frame = _run(
            tmp_path, classical_models=["svc", "lr"],
            classical_models_args={"svc": {"kernel": ["linear"]}},
        )
        svc = _cell(frame, "results_qpl_svc")["Model_Parameters"]
        lr = _cell(frame, "results_qpl_lr")["Model_Parameters"]
        assert svc["best_params"] == {"kernel": "linear"}
        assert "head_search_args" not in lr
        # lr keeps its own default grid.
        assert set(lr["best_params"]) == {"C", "penalty", "solver"}

    def test_an_override_goes_through_the_shared_grid_validation(self):
        """A ``{low, high}`` range is Optuna-only syntax and a ``RandomizedSearchCV``
        cannot sample it. Left unvalidated it is passed to the estimator as a literal
        dict value, which surfaces as an error about the estimator naming neither the
        config block nor the tuner that would accept it."""
        from qbiocode.learning._grid import build_param_grid

        with pytest.raises(ValueError, match="only the Optuna tuner can sample"):
            build_param_grid("qpl head 'svc'", {"C": {"low": 1, "high": 10}})

    def test_the_head_scoring_metric_reaches_every_searched_model(self):
        """Under ``split_mode: manifest`` the downstream searches select on the run's
        tuning metric rather than accuracy, so the two optimisation levels agree."""
        from qbiocode.learning.compute_fold import head_scorer

        scoring = head_scorer("balanced_accuracy", {"average": "weighted"})
        for head in ("rf", "svc", "lr"):
            assert build_head(head, 42, scoring=scoring).scoring is scoring

    def test_only_the_svc_model_reads_the_libsvm_cap(self):
        """``head_max_iter`` is an SVC-specific escape hatch (an uncapped poly-kernel fit
        once ran for 24 hours in a pilot job), so it must not leak into other models."""
        assert build_head("svc", 42, max_iter=5000).estimator.max_iter == 5000
        assert build_head("svc", 42).estimator.max_iter == -1
        # Accepted and ignored by every other model rather than raising, because
        # compute_qpl passes it for all of them.
        assert build_head("rf", 42, max_iter=5000).param_distributions


# ----------------------------------------------------------------------------------
# Projection optimisation is a SEPARATE level
# ----------------------------------------------------------------------------------


class TestProjectionOptimisationIsSeparate:
    """``compute_qpl`` does not tune the projection; ``compute_qpl_opt`` does.

    Keeping this distinction asserted matters because both are "QPL tuning" in casual
    speech, and conflating them is how a run ends up reported as tuned when only its
    downstream models were searched.
    """

    def test_compute_qpl_runs_the_projection_at_what_it_was_given(self, tmp_path):
        """No search over encoding/reps: the row reports exactly the asked-for map."""
        frame = _run(tmp_path, classical_models=["lr"], reps=1, entanglement="linear")
        params = _cell(frame, "results_qpl_lr")["Model_Parameters"]
        assert params["feature_map_reps"] == 1
        assert params["entanglement"] == "linear"
        # Untuned: the parameter column is Model_Parameters, not BestParams_Tuned.
        assert "BestParams_Tuned" not in _cell(frame, "results_qpl_lr")

    def test_the_projection_search_space_is_the_projection_s_own_parameters(self):
        """What ``compute_qpl_opt`` searches, from ``gridsearch_qpl_args`` -- the feature
        map, never the downstream model list."""
        from qbiocode.learning._tuning import build_search_space

        space = build_search_space("qpl", {
            "encoding": ["Z", "ZZ"], "reps": [1, 2],
            "entanglement": ["linear", "full"], "data_map": ["unit", "qiskit"],
            "bandwidth": {"low": 0.1, "high": 6.0, "log": True},
        })
        assert set(space) == {"encoding", "reps", "entanglement", "data_map", "bandwidth"}

    def test_the_model_selection_keys_are_not_searchable_projection_parameters(self):
        """``classical_models`` selects which models run and ``classical_models_args``
        how each is searched; neither is a candidate value. ``compute_qpl_opt`` passes
        both to every trial as ``fixed`` instead -- scoring candidates on a different
        set of downstream models than the refit uses would choose a projection suiting
        models the config excluded."""
        import inspect

        from qbiocode.learning.compute_qpl import compute_qpl_opt

        params = inspect.signature(compute_qpl_opt).parameters
        for key in ("classical_models", "classical_models_args"):
            assert key in params
        assert params["classical_models_args"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_every_trial_is_given_the_same_downstream_models_as_the_refit(
        self, tmp_path, monkeypatch
    ):
        """Both selection keys reach every trial, not just the final fit.

        Asserted by recording what each call to ``compute_qpl`` was given rather than by
        matching the assignment in ``compute_qpl_opt``'s source. The substring version
        pinned one of two spellings of one line, which is not what matters: the property
        is that a trial searches the same downstream models, over the same grids, as the
        refit does. Scoring candidates against a different downstream pipeline would
        choose a projection suiting models the config excluded -- the defect
        ``_QUANTUM_PASSTHROUGH`` exists to prevent, one level down.
        """
        import sys

        from qbiocode.learning.compute_qpl import compute_qpl_opt

        module = sys.modules["qbiocode.learning.compute_qpl"]
        seen = []
        original = module.compute_qpl

        def recording(*args, **kwargs):
            seen.append((kwargs.get("classical_models"),
                         kwargs.get("classical_models_args")))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "compute_qpl", recording)
        X_train, X_test, y_train, y_test = _problem(n_train=12, n_test=4)
        compute_qpl_opt(
            X_train, X_test, y_train, y_test, _args(tmp_path), data_key="t",
            encoding=["ZZ"], reps=[1, 2], classical_models=["svc"],
            classical_models_args={"svc": {"kernel": ["linear"]}},
            n_trials=2,
        )
        # Two trials plus the refit, every one of them given both keys.
        assert len(seen) >= 2, seen
        assert all(models == ["svc"] for models, _ in seen), seen
        assert all(
            blocks == {"svc": {"kernel": ["linear"]}} for _, blocks in seen
        ), seen


# ----------------------------------------------------------------------------------
# TabPFN as a downstream model
# ----------------------------------------------------------------------------------


class TestTabPfnAsADownstreamModel:
    """TabPFN integrated through the same registry, with its constraints respected.

    It is the one downstream model with no hyperparameter search, and that is a property
    of the model rather than a special case in the pipeline: the weights are pretrained
    and frozen, so what it exposes are inference settings, not training
    hyperparameters. Searching them would cost ``HEAD_N_ITER * HEAD_CV`` transformer
    forward passes per projection to choose between near-identical candidates.
    """

    def test_it_is_a_selectable_model_but_not_a_default(self):
        assert "tabpfn" in QPL_HEADS
        assert "tabpfn" not in QPL_DEFAULT_HEADS

    def test_it_goes_through_the_same_builder_as_every_other_model(self):
        """No parallel pipeline: one dispatch builds every downstream model.

        Asserted by building each model through ``build_head`` rather than by grepping
        ``compute_qpl``'s source for the call. What matters is that the one builder
        answers for TabPFN as it does for the others -- so a model cannot be added to
        the registry and then quietly fail to be reachable.
        """
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        for head in _available(QPL_HEADS):
            assert build_head(head, 42) is not None, head
        # And the registry is what the loop iterates, so a name it does not know is a
        # refusal from the builder rather than a silently skipped model.
        with pytest.raises(ValueError, match="Unknown QPL head"):
            build_head("tabpfn_v2", 42)

    def test_it_is_not_wrapped_in_a_search(self):
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        model = build_head("tabpfn", 42)
        assert not isinstance(model, RandomizedSearchCV)
        assert not hasattr(model, "best_params_")
        assert model.get_params()["random_state"] == 42

    def test_it_pins_qbiocode_s_own_checkpoint_rather_than_upstream_s_default(self):
        """Upstream's ``model_path='auto'`` resolves to ITS default version, which is v3
        -- non-commercial, non-production, and gated behind an interactive licence
        acceptance. Left at 'auto' this downstream model either failed with a licence
        error *after* the quantum projection had been paid for, or quietly ran a
        differently-licensed model than ``compute_tabpfn`` did on the same data."""
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        from qbiocode.learning.compute_tabpfn import (
            TABPFN_DEFAULT_VERSION, resolve_model_path,
        )

        path = str(build_head("tabpfn", 42).get_params()["model_path"])
        assert path != "auto", "left at upstream's default, which is the gated version"
        # The same checkpoint `compute_tabpfn` resolves, so a QPL downstream TabPFN and
        # a standalone TabPFN arm are the same model on the same data.
        assert path == str(resolve_model_path("auto", TABPFN_DEFAULT_VERSION))
        assert f"-{TABPFN_DEFAULT_VERSION}-" in os.path.basename(path), path

    def test_its_block_configures_inference_settings_rather_than_a_grid(self):
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        model = build_head("tabpfn", 42, search_args={"n_estimators": 2})
        assert model.get_params()["n_estimators"] == 2

    def test_the_class_limit_is_respected_rather_than_hit_as_a_tensor_error(self, tmp_path):
        """TabPFN's pretrained head cannot represent more than ten classes, and unlike
        its row and feature limits that one is not waivable. An unsuitable target drops
        the model with an explanation instead of failing a run that has already paid for
        its projection."""
        if not tabpfn_is_available():
            pytest.skip("the [tabpfn] extra is not installed")
        n_classes = TABPFN_MAX_CLASSES + 1
        # Enough rows per class for the surviving model's own HEAD_CV-fold search; the
        # point here is the class count, not a small-sample edge case.
        problem = {
            "n_train": n_classes * HEAD_CV, "n_test": n_classes, "n_classes": n_classes,
        }
        with pytest.warns(UserWarning, match=f"at most {TABPFN_MAX_CLASSES} classes"):
            frame = _run(tmp_path, problem=problem, classical_models=["rf", "tabpfn"])
        assert _result_labels(frame) == ["qpl_rf"], (
            "an unsuitable target must drop only TabPFN, not the whole run"
        )

    def test_it_reports_its_own_settings_where_the_others_report_a_winner(
        self, tmp_path, tabpfn_ready
    ):
        """With no search there is no ``best_params_``; ``get_params()`` is the honest
        answer, and ``compute_qpl``'s defaulting read is what allows it."""
        frame = _run(tmp_path, classical_models=["tabpfn"])
        assert _result_labels(frame) == ["qpl_tabpfn"]
        best = _cell(frame, "results_qpl_tabpfn")["Model_Parameters"]["best_params"]
        assert "random_state" in best and "model_path" in best

    def test_it_shares_the_one_projection_with_the_other_models(
        self, tmp_path, tabpfn_ready
    ):
        """The integration must not cost an extra projection."""
        frame = _run(tmp_path, classical_models=["lr", "tabpfn"])
        assert _result_labels(frame) == ["qpl_lr", "qpl_tabpfn"]
        assert len(_projection_files(tmp_path)) == 2

    def test_an_absent_extra_drops_it_rather_than_failing_the_run(self, tmp_path, monkeypatch):
        """An optional extra being absent is an ordinary configuration state, and by the
        time the models run the projection is already paid for."""
        import sys

        # The MODULE, not the function of the same name imported above.
        module = sys.modules["qbiocode.learning.compute_qpl"]
        monkeypatch.setattr(module, "tabpfn_is_available", lambda: False)
        with pytest.warns(UserWarning, match="TabPFN is not installed"):
            frame = _run(tmp_path, classical_models=["lr", "tabpfn"])
        assert _result_labels(frame) == ["qpl_lr"]


# ----------------------------------------------------------------------------------
# Output naming
# ----------------------------------------------------------------------------------


class TestOutputNaming:
    """``<projection label>_<downstream model>``, which is the existing convention.

    Every downstream reader keys on these strings -- the fair-selection pairing, the
    quantum-winner finder, the correlation figures -- so the label is a contract, not a
    display detail.
    """

    def test_a_row_is_named_projection_then_downstream_model(self, tmp_path):
        frame = _run(tmp_path, classical_models=["svc", "rf"])
        assert _result_labels(frame) == ["qpl_rf", "qpl_svc"]
        assert _cell(frame, "results_qpl_svc")["model"] == "qpl_svc"

    def test_there_is_no_bare_projection_row(self, tmp_path):
        """A QPL run fans out; a bare ``results_qpl`` would be a seventh result with no
        model behind it, and consumers test for the fan-out."""
        frame = _run(tmp_path, classical_models=["svc"])
        assert "results_qpl" not in frame.columns

    def test_the_label_follows_the_caller_s_model_name(self, tmp_path):
        """So a tuned run reads ``qpl_opt_<model>``. The suffix used to be hardcoded to
        ``qpl_``, which threw the argument away and made a tuned QPL run produce exactly
        the columns an untuned one did -- so the results table could not say whether the
        projection had been searched."""
        frame = _run(tmp_path, classical_models=["svc"], model="qpl_opt")
        assert _result_labels(frame) == ["qpl_opt_svc"]
        assert "BestParams_Tuned" in _cell(frame, "results_qpl_opt_svc")

    def test_every_companion_column_carries_the_same_label(self, tmp_path):
        """``model_run`` pivots on these, so a label that disagreed across the four
        columns would split one model's result across several rows."""
        frame = _run(tmp_path, classical_models=["svc"])
        for prefix in ("results_", "y_test_", "y_predicted_", "y_score_"):
            assert f"{prefix}qpl_svc" in frame.columns

    @pytest.mark.parametrize("head", QPL_HEADS)
    def test_each_model_s_label_is_the_name_the_config_uses(self, head):
        """No renaming between config and results: ``classical_models: ['randomforest']``
        is not a thing, and ``qpl-svc``-style hyphens are not this library's convention.
        """
        assert head == head.lower() and "-" not in head
        assert f"qpl_{head}".startswith("qpl_")


# ----------------------------------------------------------------------------------
# Invalid configurations
# ----------------------------------------------------------------------------------


class TestInvalidConfigurations:
    """Rejected up front, before the projection -- which is the expensive half.

    An unknown model name used to be warned about and skipped inside the model loop, so
    a typo in the only requested model paid for the whole projection and then raised a
    message about nothing having been fitted.
    """

    def _never(self, tmp_path):
        """A projection directory that must not come into existence."""
        return tmp_path / "never"

    @pytest.mark.parametrize("models, match", [
        (["nope"], "Unknown QPL downstream model"),
        (["lr", "nope"], r"\['nope'\]"),
        ([], "classical_models is empty"),
        (["svc", "svc"], "Duplicate QPL downstream model"),
        (42, "must be a head name or a list"),
    ])
    def test_a_bad_model_list_is_refused_before_the_projection(
        self, tmp_path, models, match
    ):
        with pytest.raises(ValueError, match=match):
            _run(self._never(tmp_path), classical_models=models)
        assert not os.path.exists(self._never(tmp_path))

    @pytest.mark.parametrize("blocks, match", [
        ({"bogus": {"C": [1]}}, "unknown QPL downstream model"),
        ({"svc": [1, 2]}, "must be a mapping of settings"),
        ({"svc": {"C": {"low": 1, "high": 10}}}, "only the Optuna tuner can sample"),
        ([("svc", {})], "must map a head name"),
    ])
    def test_a_bad_per_model_block_is_refused_before_the_projection(
        self, tmp_path, blocks, match
    ):
        with pytest.raises(ValueError, match=match):
            _run(
                self._never(tmp_path), classical_models=["svc"],
                classical_models_args=blocks,
            )
        assert not os.path.exists(self._never(tmp_path)), (
            "the projection was computed before the config was validated"
        )

    def test_a_block_for_an_unselected_model_is_not_an_error(self, tmp_path):
        """The usual cause is a model missing from ``classical_models``, which is worth
        a log line rather than a refusal -- a config may legitimately carry blocks for
        models it is not running this time."""
        frame = _run(
            tmp_path, classical_models=["svc"],
            classical_models_args={"lr": {"C": [1]}},
        )
        assert _result_labels(frame) == ["qpl_svc"]

    def test_an_unknown_model_named_directly_to_the_builder_is_refused(self):
        with pytest.raises(ValueError, match="Unknown QPL head"):
            build_head("nope", 42)


# ----------------------------------------------------------------------------------
# Config plumbing
# ----------------------------------------------------------------------------------


class TestConfigPlumbing:
    """``qpl_args`` reaches both the tuned and the untuned path.

    ``gridsearch_qpl_args`` says what to search; ``qpl_args`` says how to run the model.
    Model selection is unambiguously the second, so it has to survive turning tuning on
    -- it used to be honoured with tuning off and silently dropped with it on, which
    meant a tuned run searched the projection against models the config had excluded.
    """

    def test_both_model_selection_keys_pass_through_to_the_tuned_path(self):
        from qbiocode.evaluation.model_run import _QUANTUM_PASSTHROUGH

        assert set(_QUANTUM_PASSTHROUGH["qpl"]) == {
            "classical_models", "classical_models_args",
        }

    def test_both_keys_are_accepted_by_both_qpl_entry_points(self):
        import inspect

        from qbiocode.learning.compute_qpl import compute_qpl_opt

        for function in (compute_qpl, compute_qpl_opt):
            params = inspect.signature(function).parameters
            assert "classical_models" in params
            assert "classical_models_args" in params
            assert params["classical_models_args"].default is None

    def test_the_shipped_config_documents_the_selection_keys(self):
        """A key nobody can find is a key nobody uses, and the QPL block is where a
        reader looks for it."""
        from pathlib import Path

        import qbiocode

        config = (
            Path(qbiocode.__file__).parent / "apps" / "qprofiler" / "configs" / "config.yaml"
        ).read_text()
        assert "classical_models" in config
        assert "classical_models_args" in config
