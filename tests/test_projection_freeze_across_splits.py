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

"""``freeze_quantum_params`` across splits, for the two projected-kernel models.

Two optimisation levels run inside a tuned QPL or PQK arm, and this module is about
keeping them apart:

* **projection-level** -- ``encoding``, ``reps``, ``entanglement``, ``data_map``,
  ``bandwidth``: searched with Optuna by ``compute_<m>_opt``, scored on a holdout carved
  out of that split's *training* rows.
* **downstream-model** -- each classical head's own hyperparameters: searched by a
  ``RandomizedSearchCV`` inside every ``compute_<m>`` fit, on the projected features.

``freeze_quantum_params`` governs the **first** only:

* ``False`` (the library default) -- *regular* optimisation. Every split searches the
  projection itself, on its own training rows::

      split 1 -> optimise projection -> evaluate
      split 2 -> optimise projection -> evaluate
      split 3 -> optimise projection -> evaluate

* ``True`` -- *fast* optimisation. The first split searches, and its winner is frozen to
  a JSON file and reused by every later split, which then pays one refit instead of a
  whole study::

      split 1 -> optimise projection -> evaluate
      split 2 -> reuse split 1's projection params -> evaluate
      split 3 -> reuse split 1's projection params -> evaluate

  The downstream searches still run per split, on that split's own projected training
  rows.

:mod:`tests.test_quantum_param_freeze` already covers the cache itself -- key scoping,
staleness, corruption, and the cost claim -- against a recording stub. What it cannot
show is the interaction with the real two-level models, which is what is asserted here:
that the saving lands on the projection study and not on the downstream searches, and
that QPL still computes one projection per split however many heads it fans out to.

The assertions count *work* rather than measuring time: a spy on
``optuna.create_study`` counts projection studies, and a spy on
``qutils.get_backend_session`` counts projections. A test that inferred the difference
from a wall clock would pass on a slow machine and on a broken freeze alike.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import qbiocode.learning._tuning as tuning_module
import qbiocode.utils.qutils as qutils
from qbiocode.learning._param_cache import FREEZE_KEY, freezing_enabled
from qbiocode.learning.compute_pqk import compute_pqk_opt
from qbiocode.learning.compute_qelm import compute_qelm_opt
from qbiocode.learning.compute_qpl import compute_qpl_opt

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: The projection search space these runs use. Two values per key, so a study has
#: something to choose between while staying cheap.
SPACE = {"encoding": ["Z", "ZZ"], "reps": [1, 2]}

#: Trials per split's projection study. The assertions are about how many *studies* run,
#: not how well they search, so this is the smallest budget that still searches.
N_TRIALS = 2

#: Splits per scenario. Three rather than two so "once" is distinguishable from "every
#: other one".
N_SPLITS = 3

#: The projection parameters a frozen file carries, in the order the report reads them.
PROJECTION_KEYS = ("encoding", "reps")


@pytest.fixture(autouse=True)
def _quiet_freeze_warning():
    """Silence the once-per-process "frozen results are optimistic" warning.

    It is a real and deliberate warning (see :mod:`qbiocode.learning._param_cache`), but
    it is logged, not raised, and it would otherwise print on every parametrisation.
    """
    logger = logging.getLogger("qbiocode.learning._param_cache")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _dataset(n=44, n_features=3, seed=0):
    """A small separable binary problem, resampled per split by ``_split``."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _split(X, y, iteration):
    """Split ``iteration`` of ``X``, scaled -- the protocol ``qprofiler`` follows.

    ``random_state=42 + iteration`` mirrors ``qprofiler._split_seed``'s ``seed + iter``,
    so each split really is a different resample. That matters here: a freeze that was
    silently keyed on the data would look identical to a working one if every split were
    the same rows.

    Scaled with the scaler fitted on the training rows only, as ``scale_train_test``
    does, so no test statistic reaches the training features.
    """
    from qbiocode import scale_train_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42 + iteration
    )
    X_train, X_test = scale_train_test(X_train, X_test, scaling="MinMaxScaler")
    return X_train, X_test, y_train, y_test


def _args(root, freeze, **extra):
    """A tuned-quantum config, with both caches under ``root``.

    ``data_key`` is what enables freezing at all, and the cache directory has to be
    per-scenario: a frozen file from the regular-mode run would turn the fast-mode run
    into a cache hit on its very first split.
    """
    args = {
        "backend": "simulator", "seed": 7, "q_seed": 7, "shots": 100, "resil_level": 1,
        "average": "weighted", "multi_class": "raise",
        "grid_search": True, "tune_quantum": True, "cross_validation": 2,
        "tuning_metric": "balanced_accuracy",
        FREEZE_KEY: freeze,
        "quantum_param_dir": str(root / "params"),
        "pqk_projection_dir": str(root / "proj"),
        "qpl_projection_dir": str(root / "proj"),
        # Named for the same reason as the other two: the default is a RELATIVE
        # directory, so an arm whose key is missing writes its cache into whatever the
        # working directory happens to be -- the repository root, under pytest.
        "qelm_projection_dir": str(root / "proj"),
    }
    args.update(extra)
    return args


class _Spies:
    """Counts of the two kinds of work a split can do.

    ``studies`` counts Optuna studies created, which is one per projection-level search
    -- the thing ``freeze_quantum_params`` is meant to skip. ``sessions`` counts
    projections actually computed: a trial's, and the refit's.

    A projection is counted at BOTH of the places an arm can compute one. ``pqk`` and
    ``qpl`` go through ``qutils.get_backend_session`` (one session per projection);
    ``qelm`` never opens a session at all -- it always uses
    ``qbiocode.utils.projection.make_projector``, because its observable pool can hold
    multi-qubit Paulis and the per-row Estimator loop the other two use is written
    around 3n single-qubit groups. Counting only sessions therefore scored every QELM
    split as zero work, which reads as "it reused the cached projection" -- the exact
    failure this spy exists to detect, reported against the one arm that cannot have it.
    """

    def __init__(self):
        self.studies = []
        self.sessions = []

    def install(self, monkeypatch):
        from qbiocode.utils import projection as projection_module

        real_study = tuning_module.optuna.create_study
        real_session = qutils.get_backend_session
        real_projector = projection_module.make_projector

        def study(*args, **kwargs):
            self.studies.append(1)
            return real_study(*args, **kwargs)

        def session(*args, **kwargs):
            self.sessions.append(1)
            return real_session(*args, **kwargs)

        def projector(*args, **kwargs):
            self.sessions.append(1)
            return real_projector(*args, **kwargs)

        monkeypatch.setattr(tuning_module.optuna, "create_study", study)
        monkeypatch.setattr(qutils, "get_backend_session", session)
        # Patched on the module rather than on the importing function's namespace:
        # compute_qelm imports it inside the call, so the attribute is looked up afresh.
        monkeypatch.setattr(projection_module, "make_projector", projector)
        return self


def _metrics(frame, label):
    """The populated metrics dict of ``results_<label>``.

    One row per reported model, each column populated on its own row (see
    ``_tuning._metric_dicts``), so a positional read would find NaN for all but the
    first.
    """
    column = f"results_{label}"
    for value in frame[column]:
        if isinstance(value, dict):
            return value
    raise AssertionError(f"{column} is populated on no row")


def _labels(frame):
    return sorted(c[len("results_"):] for c in frame.columns if c.startswith("results_"))


def _projection_params(frame, label):
    """``(encoding, reps)`` as the tuned row reports them for one model."""
    tuned = _metrics(frame, label)["BestParams_Tuned"]
    return tuple(tuned.get(key) for key in PROJECTION_KEYS)


#: The two arms under test. ``label`` is the results label each writes, and ``extra``
#: the model-specific kwargs: QPL needs its downstream model list, PQK takes none.
ARMS = {
    "pqk": {
        "run": compute_pqk_opt,
        "label": "pqk_opt",
        "extra": {},
    },
    "qpl": {
        "run": compute_qpl_opt,
        "label": "qpl_opt_svc",
        "extra": {"classical_models": ["svc"]},
    },
    # QELM freezes the same projection keys plus its own reservoir_reps; its ridge
    # readout is re-selected per split, as pqk's and qpl's heads are.
    "qelm": {
        "run": compute_qelm_opt,
        "label": "qelm_opt",
        "extra": {},
    },
}


def _run_splits(arm, tmp_path, freeze, monkeypatch, splits=N_SPLITS, **extra):
    """Run ``splits`` resamples of one arm, returning the spies and each split's frame.

    One cache directory across the splits -- that is the channel freezing uses -- and a
    fresh one per call, so scenarios cannot read each other's frozen files.
    """
    spec = ARMS[arm]
    root = tmp_path / f"{arm}_{freeze}"
    shutil.rmtree(root, ignore_errors=True)
    spies = _Spies().install(monkeypatch)
    X, y = _dataset()
    # The caller's kwargs win over the arm's defaults, so a test may override
    # `classical_models` without colliding with the entry in `spec["extra"]`.
    model_kwargs = {**spec["extra"], **extra}
    frames, per_split = [], []
    for iteration in range(1, splits + 1):
        X_train, X_test, y_train, y_test = _split(X, y, iteration)
        before = (len(spies.studies), len(spies.sessions))
        frames.append(spec["run"](
            X_train, X_test, y_train, y_test, _args(root, freeze),
            # The iteration token is the part freeze_key drops, so these keys differ per
            # split and still map to one frozen file.
            data_key=f"ds_none_3_{iteration}",
            n_trials=N_TRIALS, **SPACE, **model_kwargs,
        ))
        per_split.append((len(spies.studies) - before[0], len(spies.sessions) - before[1]))
    return spies, frames, per_split


# ----------------------------------------------------------------------------------
# Regular mode: every split optimises its own projection
# ----------------------------------------------------------------------------------


class TestRegularModeOptimisesEverySplit:
    """``freeze_quantum_params: False`` -- the default, and the symmetric protocol."""

    def test_it_is_the_library_default(self):
        """A config that says nothing must get per-split optimisation.

        The shipped ``apps/qprofiler/configs/config.yaml`` deliberately sets ``True``
        for the sweep's cost; the *library* default is the honest one.
        """
        assert freezing_enabled({}) is False
        assert freezing_enabled({"backend": "simulator"}) is False

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_every_split_runs_its_own_projection_study(self, arm, tmp_path, monkeypatch):
        spies, _, per_split = _run_splits(arm, tmp_path, False, monkeypatch)
        assert len(spies.studies) == N_SPLITS, (
            f"{arm} ran {len(spies.studies)} projection studies across {N_SPLITS} "
            f"splits; regular mode must search once per split"
        )
        assert [studies for studies, _ in per_split] == [1] * N_SPLITS, per_split

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_no_split_reports_reused_parameters(self, arm, tmp_path, monkeypatch):
        """``tuning_reused`` is the row's own account of whether it searched."""
        _, frames, _ = _run_splits(arm, tmp_path, False, monkeypatch)
        label = ARMS[arm]["label"]
        assert [_metrics(f, label)["tuning_reused"] for f in frames] == [False] * N_SPLITS

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_nothing_is_written_to_the_frozen_cache(self, arm, tmp_path, monkeypatch):
        """Regular mode must leave no file a later run could pick up as frozen."""
        _run_splits(arm, tmp_path, False, monkeypatch)
        params = tmp_path / f"{arm}_False" / "params"
        assert not params.exists() or not list(params.glob("*.json"))


# ----------------------------------------------------------------------------------
# Fast mode: the projection is optimised once, on the first split
# ----------------------------------------------------------------------------------


class TestFastModeOptimisesOnlyTheFirstSplit:
    """``freeze_quantum_params: True`` -- search once, reuse for the rest."""

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_exactly_one_projection_study_runs_and_it_is_the_first_split_s(
        self, arm, tmp_path, monkeypatch
    ):
        spies, _, per_split = _run_splits(arm, tmp_path, True, monkeypatch)
        assert len(spies.studies) == 1, (
            f"{arm} ran {len(spies.studies)} projection studies; fast mode must search "
            f"exactly once"
        )
        studies = [count for count, _ in per_split]
        assert studies == [1] + [0] * (N_SPLITS - 1), (
            f"the study must fall on the FIRST split; got {studies}"
        )

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_later_splits_reuse_the_first_split_s_parameters_unchanged(
        self, arm, tmp_path, monkeypatch
    ):
        _, frames, _ = _run_splits(arm, tmp_path, True, monkeypatch)
        label = ARMS[arm]["label"]
        chosen = [_projection_params(f, label) for f in frames]
        assert all(value is not None for value in chosen[0]), chosen[0]
        assert chosen == [chosen[0]] * N_SPLITS, (
            f"{arm} did not reuse the first split's projection parameters: {chosen}"
        )
        reused = [_metrics(f, label)["tuning_reused"] for f in frames]
        assert reused == [False] + [True] * (N_SPLITS - 1), reused

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_a_later_split_still_computes_its_own_projection(
        self, arm, tmp_path, monkeypatch
    ):
        """Frozen *parameters*, not a frozen projection.

        What is reused is the configuration; split 2's rows are different rows and must
        be projected through it. A later split paying zero sessions would mean it had
        loaded split 1's cached projection -- the row-count guard usually catches that,
        but not when two splits happen to share a length.
        """
        _, _, per_split = _run_splits(arm, tmp_path, True, monkeypatch)
        sessions = [count for _, count in per_split]
        assert sessions[0] == N_TRIALS + 1, (
            f"the first split should pay {N_TRIALS} trial projections plus the refit; "
            f"got {sessions[0]}"
        )
        assert sessions[1:] == [1] * (N_SPLITS - 1), (
            f"a reusing split should pay exactly one projection (its own refit); "
            f"got {sessions[1:]}"
        )

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_the_frozen_file_is_written_once_and_names_the_arm(
        self, arm, tmp_path, monkeypatch
    ):
        _run_splits(arm, tmp_path, True, monkeypatch)
        files = sorted((tmp_path / f"{arm}_True" / "params").glob("*.json"))
        assert len(files) == 1, files
        # The iteration token is dropped from the key, which is what lets every split
        # reach one file; the model name is kept, which keeps arms apart.
        assert files[0].name == f"ds_none_3__{arm}.json", files[0].name

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_the_frozen_file_holds_the_parameters_the_rows_report(
        self, arm, tmp_path, monkeypatch
    ):
        """The file is the propagation channel, so it must agree with the results."""
        import json

        _, frames, _ = _run_splits(arm, tmp_path, True, monkeypatch)
        payload = json.loads(
            next((tmp_path / f"{arm}_True" / "params").glob("*.json")).read_text()
        )
        reported = _projection_params(frames[0], ARMS[arm]["label"])
        assert tuple(payload["params"][key] for key in PROJECTION_KEYS) == reported
        assert payload["model"] == arm
        assert payload["tuning_metric"] == "balanced_accuracy"


# ----------------------------------------------------------------------------------
# The flag governs the projection level only
# ----------------------------------------------------------------------------------


class TestDownstreamOptimisationIsUnaffected:
    """Fast mode must not freeze the downstream models too.

    Those searches are cheap -- a ``RandomizedSearchCV`` over an already-projected
    matrix -- so there is nothing to save by skipping them, and skipping them would
    change what the arm is: a model fitted at another split's hyperparameters.
    """

    @pytest.mark.parametrize("freeze", [False, True], ids=["regular", "fast"])
    def test_each_split_searches_its_downstream_model_in_both_modes(
        self, freeze, tmp_path, monkeypatch
    ):
        """Every split reports a ``best_params`` drawn from the head's own grid.

        Asserted per split rather than by requiring the splits to *differ*: identical
        winners on different rows are possible and would make such a test flaky. What
        must hold is that each split ran its own search and reported its own result.
        """
        _, frames, _ = _run_splits("qpl", tmp_path, freeze, monkeypatch)
        for index, frame in enumerate(frames, start=1):
            best = _metrics(frame, "qpl_opt_svc")["BestParams_Tuned"]["best_params"]
            assert set(best) == {"C", "gamma", "kernel"}, (split := index) and best
            assert best["kernel"] in ("linear", "rbf", "poly", "sigmoid"), index

    def test_the_downstream_search_is_not_cached_across_splits(self, tmp_path, monkeypatch):
        """Only the projection parameters are frozen; the cache holds nothing else.

        A frozen payload naming a downstream hyperparameter would mean the head search
        had been frozen too -- the failure this whole class exists to exclude.
        """
        import json

        _run_splits("qpl", tmp_path, True, monkeypatch)
        payload = json.loads(
            next((tmp_path / "qpl_True" / "params").glob("*.json")).read_text()
        )
        assert set(payload["params"]) <= set(SPACE), payload["params"]
        for leaked in ("C", "gamma", "kernel", "n_estimators", "best_params"):
            assert leaked not in payload["params"], payload["params"]

    def test_the_head_list_is_not_frozen_either(self, tmp_path, monkeypatch):
        """``classical_models`` selects which models run; it is not a tuned value."""
        import json

        _run_splits("qpl", tmp_path, True, monkeypatch)
        payload = json.loads(
            next((tmp_path / "qpl_True" / "params").glob("*.json")).read_text()
        )
        assert "classical_models" not in payload["params"]
        assert "classical_models_args" not in payload["params"]


class TestQplProjectionReuseWithinASplitStillHolds:
    """Fast mode across splits must not disturb projection sharing within a split.

    Within one split a QPL fit projects once and fans the result out to every downstream
    model. That property is tested on its own in
    ``tests/test_qpl_downstream_models.py``; here it is checked under freezing, because
    the two mechanisms both touch how often a projection is computed and a regression in
    either would look like one in the other.
    """

    @pytest.mark.parametrize("freeze", [False, True], ids=["regular", "fast"])
    def test_several_downstream_models_share_one_projection_per_split(
        self, freeze, tmp_path, monkeypatch
    ):
        _, frames, per_split = _run_splits(
            "qpl", tmp_path, freeze, monkeypatch, classical_models=["svc", "lr", "rf"],
        )
        for frame in frames:
            assert _labels(frame) == ["qpl_opt_lr", "qpl_opt_rf", "qpl_opt_svc"]
        sessions = [count for _, count in per_split]
        # One session per projection: the trials' plus the refit's on the searching
        # split, the refit's alone on a reusing one. Three heads, not three projections.
        expected_first = N_TRIALS + 1
        assert sessions[0] == expected_first, sessions
        later = [1] * (N_SPLITS - 1) if freeze else [expected_first] * (N_SPLITS - 1)
        assert sessions[1:] == later, sessions

    def test_every_head_of_a_reusing_split_reports_the_frozen_projection(
        self, tmp_path, monkeypatch
    ):
        """One projection means one projection configuration across the heads."""
        _, frames, _ = _run_splits(
            "qpl", tmp_path, True, monkeypatch, classical_models=["svc", "lr"],
        )
        first = _projection_params(frames[0], "qpl_opt_svc")
        for frame in frames:
            for head in ("svc", "lr"):
                assert _projection_params(frame, f"qpl_opt_{head}") == first


# ----------------------------------------------------------------------------------
# Split isolation
# ----------------------------------------------------------------------------------


class TestNoTestRowsReachTheProjectionSearch:
    """The projection study must never see the rows it is later scored on.

    In regular mode that is guaranteed per split; in fast mode the first split's search
    is the only one, so it is the only place a leak could enter. Both are checked by
    recording the arrays each trial was handed.
    """

    @pytest.mark.parametrize("freeze", [False, True], ids=["regular", "fast"])
    def test_a_trial_is_fitted_and_scored_inside_the_training_rows(
        self, freeze, tmp_path, monkeypatch
    ):
        """Every trial's fit and validation rows are subsets of that split's X_train.

        ``run_function_study`` carves its scoring holdout out of ``X_train`` with
        ``train_test_split``, so a trial sees training rows only. Pinned by recording
        what the compute function was actually called with -- the alternative is reading
        the tuner and trusting it.
        """
        import sys

        # The MODULE, not the re-exported function of the same name: `qbiocode.learning`
        # does `from .compute_qpl import compute_qpl`, so the attribute path resolves to
        # the function.
        module = sys.modules["qbiocode.learning.compute_qpl"]
        seen = []
        real = module.compute_qpl

        def recording(X_train, X_test, y_train, y_test, args, **kwargs):
            seen.append((np.asarray(X_train), np.asarray(X_test)))
            return real(X_train, X_test, y_train, y_test, args, **kwargs)

        monkeypatch.setattr(module, "compute_qpl", recording)
        X, y = _dataset()
        root = tmp_path / f"leak_{freeze}"
        rows = []
        for iteration in range(1, N_SPLITS + 1):
            X_train, X_test, y_train, y_test = _split(X, y, iteration)
            rows.append((X_train, X_test))
            compute_qpl_opt(
                X_train, X_test, y_train, y_test, _args(root, freeze),
                data_key=f"ds_none_3_{iteration}", n_trials=N_TRIALS,
                classical_models=["lr"], **SPACE,
            )

        def row_set(array):
            return {tuple(np.round(row, 12)) for row in np.asarray(array)}

        # The refit of split i is called with (X_train_i, X_test_i); every other call is
        # a trial, whose two arrays must both sit inside some split's training rows.
        refits = {(len(a), len(b)) for a, b in
                  ((tr, te) for tr, te in rows)}
        training_rows = [row_set(tr) for tr, _ in rows]
        test_rows = [row_set(te) for _, te in rows]
        trials = 0
        for fit, validation in seen:
            if (len(fit), len(validation)) in refits and any(
                row_set(fit) == train for train in training_rows
            ):
                continue  # a refit: it is given the real test split, as it must be
            trials += 1
            combined = row_set(fit) | row_set(validation)
            assert any(combined <= train for train in training_rows), (
                "a projection trial was given rows outside every split's training set"
            )
            for index, held_out in enumerate(test_rows):
                assert not (combined & held_out), (
                    f"a projection trial saw rows from split {index + 1}'s TEST set"
                )
        assert trials, "no trial calls were recorded, so nothing was checked"

    def test_fast_mode_searches_on_the_first_split_only_and_not_on_later_test_rows(
        self, tmp_path, monkeypatch
    ):
        """The frozen parameters are a function of split 1's training rows alone.

        This is the documented cost of fast mode rather than a bug: split 1's training
        rows overlap later splits' test rows under repeated holdout resampling, so a
        frozen arm's later scores are optimistic. What must not happen is the stronger
        failure -- the search *reading* a later split's test rows. Since the search runs
        once, before any later split exists, that is structural, and this test pins it by
        showing no study runs after the first split.
        """
        spies, _, per_split = _run_splits("qpl", tmp_path, True, monkeypatch)
        assert [count for count, _ in per_split] == [1, 0, 0]
        assert len(spies.studies) == 1


# ----------------------------------------------------------------------------------
# Backward compatibility
# ----------------------------------------------------------------------------------


class TestBackwardCompatibility:

    def test_an_existing_config_without_the_key_tunes_every_split(
        self, tmp_path, monkeypatch
    ):
        """Omitting ``freeze_quantum_params`` entirely is regular mode."""
        spies = _Spies().install(monkeypatch)
        X, y = _dataset()
        root = tmp_path / "nokey"
        for iteration in (1, 2):
            X_train, X_test, y_train, y_test = _split(X, y, iteration)
            args = _args(root, False)
            del args[FREEZE_KEY]
            compute_pqk_opt(
                X_train, X_test, y_train, y_test, args,
                data_key=f"ds_none_3_{iteration}", n_trials=N_TRIALS, **SPACE,
            )
        assert len(spies.studies) == 2

    @pytest.mark.parametrize("arm", sorted(ARMS))
    def test_a_direct_caller_that_passes_no_data_key_shares_one_frozen_entry(
        self, arm, tmp_path, monkeypatch
    ):
        """``data_key`` defaults to ``''``, which freezes under a single shared key.

        Worth pinning because it is easy to assume the opposite -- that an absent
        ``data_key`` disables freezing the way ``None`` does in
        :func:`~qbiocode.learning._tuning.run_function_study`. It does not: ``''`` is not
        ``None``, so the cache is consulted and every such call maps to one file (
        ``freeze_key('')`` is ``''``). Under ``qprofiler`` this never arises -- it always
        passes a real per-pass key -- but a direct caller that loops over splits while
        leaving ``data_key`` at its default and freezing on gets ONE search for all of
        them, across datasets as readily as across splits.
        """
        spec = ARMS[arm]
        spies = _Spies().install(monkeypatch)
        X, y = _dataset()
        root = tmp_path / f"nokey_{arm}"
        for iteration in (1, 2):
            X_train, X_test, y_train, y_test = _split(X, y, iteration)
            spec["run"](
                X_train, X_test, y_train, y_test, _args(root, True),
                n_trials=N_TRIALS, **SPACE, **spec["extra"],
            )
        assert len(spies.studies) == 1, (
            "an empty data_key still freezes; if this is now 2, the default changed to "
            "None and the docstring above is stale"
        )
        files = sorted((root / "params").glob("*.json"))
        assert [f.name for f in files] == [f"__{arm}.json"], files

    def test_the_shipped_sweep_config_still_opts_in(self):
        """The sweep's cost model depends on it; a silent flip would multiply its cost.

        The shipped config deliberately differs from the library default, so both are
        pinned -- the default in ``TestRegularModeOptimisesEverySplit``, the config here.
        """
        from pathlib import Path

        import yaml

        import qbiocode

        config = yaml.safe_load(
            (Path(qbiocode.__file__).parent / "apps" / "qprofiler" / "configs"
             / "config.yaml").read_text()
        )
        assert config[FREEZE_KEY] is True
        assert "quantum_param_dir" in config
