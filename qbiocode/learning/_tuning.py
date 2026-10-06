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

"""Tuning a partly-filled config block with Optuna instead of an exhaustive grid.

``GridSearchCV`` fits every point of the cross product. ``gridsearch_rf_args`` as
shipped is 4x2x4x3x3x2 = 576 combinations, so at ``cross_validation: 5`` a single
random forest costs 2880 fits -- and the grid is searched blind, spending as much
time in a hopeless corner as in a promising one. The grid is also only ever
discrete: ``C`` can be tried at each decade from 1e-3 to 1e+3 but never at 3.7.

Optuna's TPE sampler spends a fixed budget (``n_trials``) and steers it toward the
region that has been scoring well, which is what makes a continuous range worth
expressing at all. So this module accepts *both* config shapes::

    gridsearch_svc_args:
      kernel: ['linear', 'rbf', 'poly']        # a list is still a list
      C:      {low: 1.0e-3, high: 1.0e+2, log: true}   # a range is now a range

A list becomes a categorical choice, which is exactly what the old grid did with
it, so every config already in the tree keeps its meaning. A ``{low, high}`` mapping
becomes a real int or float distribution.

``build_param_grid`` in :mod:`qbiocode.learning._grid` stays as the ``tuner: grid``
path -- reproducing a number published against the exhaustive search has to remain
possible, and the two helpers deliberately agree on which entries count as "not
tuned" so switching ``tuner`` cannot change *which* hyperparameters are searched.
"""

import contextlib
import io
import logging
import math
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence

import optuna
from sklearn.metrics import f1_score, get_scorer, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

from qbiocode.learning._grid import build_param_grid, to_plain

# Optuna logs one INFO line per trial. With the default budget across seven models,
# each running inside a joblib worker that interleaves its stdout with the others,
# that is a few hundred lines of noise around the one number anybody wanted.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

#: Config key naming the metric every tuner selects hyperparameters on.
TUNING_METRIC_KEY = "tuning_metric"
#: What an absent ``tuning_metric`` means. Balanced accuracy, because that is what the
#: study reports on a corpus where several datasets are 90/10 or worse: tuning on plain
#: accuracy there rewards a configuration for predicting the majority class, and then
#: the headline column scores it on something else. ``'accuracy'`` reproduces the pilot.
DEFAULT_TUNING_METRIC = "balanced_accuracy"

#: The one place a tuning metric is defined. Each name is BOTH a key of the metrics
#: dict ``modeleval`` writes (which is what the quantum tuner reads back) AND something
#: the classical tuners can hand to ``cross_val_score``/``GridSearchCV`` as ``scoring``,
#: and the two compute the same number from the same predictions. The value takes the
#: run's ``average`` (only ``f1_score`` reads it) and returns the sklearn ``scoring``.
#:
#: Deliberately left out: ``auc`` and ``pr_auc``. ``modeleval`` records NaN for them on
#: a multiclass target and for an estimator that publishes no ranking, where sklearn's
#: ``roc_auc``/``average_precision`` scorers raise or switch to a one-vs-rest average --
#: so the classical and quantum sides would not be selecting on the same statistic.
_TUNING_METRICS = {
    "accuracy": lambda average: "accuracy",
    # adjusted=False on both sides: modeleval calls balanced_accuracy_score bare.
    "balanced_accuracy": lambda average: "balanced_accuracy",
    "mcc": lambda average: "matthews_corrcoef",
    # modeleval's F1 is averaged by args['average'] with zero_division=0; the stock
    # 'f1_weighted' scorer would agree only at one averaging and warn where this does not.
    "f1_score": lambda average: make_scorer(f1_score, average=average, zero_division=0),
}


def tuning_metric(args):
    """The validated ``tuning_metric`` of a run, defaulting to balanced accuracy.

    Args:
        args (Mapping or None): The run's config; a Hydra ``DictConfig`` works too.

    Returns:
        str: One of the keys of ``_TUNING_METRICS``, which is also the ``modeleval``
        metrics-dict key the quantum tuner scores on.

    Raises:
        ValueError: If the config names a metric that has no exact equivalent on both
            the classical and the quantum side. ``model_run`` calls this before any
            model is fitted.
    """
    metric = args.get(TUNING_METRIC_KEY) if isinstance(args, Mapping) else None
    if metric is None:
        return DEFAULT_TUNING_METRIC
    if metric not in _TUNING_METRICS:
        raise ValueError(
            f"Unknown tuning_metric {metric!r}. Choose one of {sorted(_TUNING_METRICS)}: "
            f"each is computed identically by the classical cross-validation scorer and "
            f"by modeleval, which scores the quantum candidates. 'accuracy' reproduces "
            f"runs made before this key existed; {DEFAULT_TUNING_METRIC!r} is the default."
        )
    return str(metric)


class TuningScorer:
    """A sklearn ``scoring`` callable that also knows which tuning metric it is.

    Built by :func:`tuning_scorer` from a run's config and handed to
    :func:`search_hyperparameters`, so the classical tuners score exactly what
    ``tuning_metric`` names and :class:`TunedParams` can report it by that name. The
    underlying sklearn scorer is resolved on each call rather than stored, which keeps
    the object trivially picklable into a joblib worker.

    Args:
        metric (str): A key of ``_TUNING_METRICS``.
        average (str): ``f1_score``'s averaging, as ``modeleval`` reads it from
            ``args['average']``. Ignored by the other metrics.
    """

    def __init__(self, metric=DEFAULT_TUNING_METRIC, average="weighted"):
        self.metric = tuning_metric({TUNING_METRIC_KEY: metric})
        self.average = average

    def sklearn_scoring(self):
        """The scorer sklearn would build for this metric."""
        scoring = _TUNING_METRICS[self.metric](self.average)
        return get_scorer(scoring) if isinstance(scoring, str) else scoring

    def __call__(self, estimator, X, y):
        return self.sklearn_scoring()(estimator, X, y)

    def __repr__(self):
        return f"TuningScorer(metric={self.metric!r}, average={self.average!r})"


def tuning_scorer(args):
    """The :class:`TuningScorer` a run's config asks for.

    Reads ``tuning_metric`` and, for ``f1_score``, ``average`` -- the same key and the
    same fallback (``'weighted'``) ``modeleval`` uses, so a tuned F1 and a reported F1
    are one statistic.
    """
    average = (args.get("average") if isinstance(args, Mapping) else None) or "weighted"
    return TuningScorer(tuning_metric(args), average=average)


def _scoring_parts(scoring):
    """``(metric name, sklearn scoring)`` for whatever a caller passed as ``scoring``.

    ``None`` means the default metric; a :class:`TuningScorer` or a tuning-metric name
    is resolved through the table; anything else (a sklearn scorer name such as
    ``'roc_auc'``, or a callable) is passed to sklearn untouched and reported under its
    own name.
    """
    if scoring is None:
        scoring = TuningScorer()
    elif isinstance(scoring, str) and scoring in _TUNING_METRICS:
        scoring = TuningScorer(scoring)
    if isinstance(scoring, TuningScorer):
        return scoring.metric, scoring
    return (scoring if isinstance(scoring, str) else repr(scoring)), scoring


class TunedParams(dict):
    """The best hyperparameters of a search, plus how well they scored.

    A plain ``dict`` of the chosen hyperparameters in every respect a caller relies on
    -- ``Estimator(**params)``, ``==``, ``str`` and ``repr`` (so the ``BestParams_Tuned``
    CSV text is unchanged), JSON -- with the selection evidence carried as attributes
    rather than keys. Keys would reach the estimator as keyword arguments.

    Defined at module level so it pickles by reference through joblib's loky workers,
    and ``dict`` subclasses pickle their instance ``__dict__`` along with the items, so
    the attributes survive the trip.

    Attributes:
        metric (str or None): The tuning metric the search maximised.
        score (float): Mean cross-validated score (classical) or inner-holdout score
            (quantum) of the chosen configuration; NaN when unknown -- a frozen payload
            written before scores were recorded.
        n_trials (int or None): Configurations evaluated, where cheaply known.
        reused (bool): True when the parameters came from the frozen cache of an
            earlier resample rather than a search on this one.
    """

    def __init__(self, params=(), *, metric=None, score=float("nan"), n_trials=None,
                 reused=False):
        super().__init__(params)
        self.metric = metric
        self.score = float(score) if score is not None else float("nan")
        self.n_trials = n_trials
        self.reused = bool(reused)

    def evidence(self):
        """The three results-row fields this search reports alongside its parameters."""
        return {
            "tuning_metric": self.metric,
            "tuning_score": self.score,
            "tuning_reused": self.reused,
        }


#: Keys a range mapping may carry. Anything else is a typo worth reporting: Optuna
#: would otherwise raise about a distribution the user never named.
_RANGE_KEYS = frozenset({"low", "high", "log", "step"})


class _Categorical:
    """A fixed set of values to choose among -- what a config list has always meant."""

    def __init__(self, values):
        # Kept exactly as configured, including a choice that is itself a list --
        # `gridsearch_mlp_args` writes `hidden_layer_sizes: [[20], [50], [100]]`, and
        # `best_params` should report the value the user wrote. See
        # `_suppress_unstorable_choice_warning` for why that does not produce a warning.
        #
        # "As configured" means the *values*, not the config library's wrapper types:
        # `build_search_space` has already put them through `_grid.to_plain`, so a choice
        # that is a container is a plain `list` and not a `ListConfig`. That conversion is
        # load-bearing rather than cosmetic -- Optuna puts the chosen value through
        # `json.dumps`, which a `list` survives and a `ListConfig` does not. `list(values)`
        # here is only the outer copy; it cannot reach the elements.
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def suggest(self, trial, name):
        return trial.suggest_categorical(name, self.values)


class _Range:
    """A half-open interval sampled as an int or a float.

    ``int`` when both bounds are integral and no fractional ``step`` was given --
    ``n_estimators: {low: 10, high: 500}`` must not propose 214.7 trees.
    """

    def __init__(self, low, high, log=False, step=None):
        self.low = low
        self.high = high
        self.log = bool(log)
        self.step = step
        self.is_int = (
            isinstance(low, int)
            and isinstance(high, int)
            and not isinstance(low, bool)
            and not isinstance(high, bool)
            and (step is None or isinstance(step, int))
        )

    def __len__(self):
        # Deliberately not the number of representable values: a range is treated as
        # infinite for budget purposes even when it is an int range, because the
        # sampler is free to revisit a value and the cap below is only an optimisation.
        raise TypeError("a range has no finite length")

    def suggest(self, trial, name):
        if self.is_int:
            kwargs = {"log": self.log}
            if self.step is not None:
                kwargs["step"] = self.step
            return trial.suggest_int(name, self.low, self.high, **kwargs)
        kwargs = {"log": self.log}
        if self.step is not None:
            kwargs["step"] = self.step
        return trial.suggest_float(name, float(self.low), float(self.high), **kwargs)


def _parse_range(model, name, spec):
    """Validate one ``{low, high, ...}`` mapping from the config."""
    unknown = sorted(set(spec) - _RANGE_KEYS)
    if unknown:
        raise ValueError(
            f"{model!r} hyperparameter {name!r} was given a range with unrecognised "
            f"key(s) {unknown}. A range takes {sorted(_RANGE_KEYS)}, as in "
            f"{name}: {{low: 0.01, high: 10, log: true}}."
        )
    if "low" not in spec or "high" not in spec:
        raise ValueError(
            f"{model!r} hyperparameter {name!r} was given a mapping without both "
            f"'low' and 'high' ({dict(spec)!r}). Write a range as "
            f"{name}: {{low: 0.01, high: 10}}, or a list of values to choose among."
        )
    low, high = spec["low"], spec["high"]
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(
            f"{model!r} hyperparameter {name!r} has non-numeric range bounds "
            f"(low={low!r}, high={high!r}). Only numbers can be sampled from a "
            f"range; use a list for categorical values such as kernel names."
        )
    if high <= low:
        raise ValueError(
            f"{model!r} hyperparameter {name!r} has an empty range: high={high!r} is "
            f"not above low={low!r}."
        )
    if spec.get("log") and spec.get("step") is not None:
        raise ValueError(
            f"{model!r} hyperparameter {name!r} asks for both 'log' and 'step'. Optuna "
            f"cannot combine them -- a log scale has no constant spacing. Drop one."
        )
    if spec.get("log") and low <= 0:
        raise ValueError(
            f"{model!r} hyperparameter {name!r} asks for a log scale but its range "
            f"starts at low={low!r}. A log scale needs a positive lower bound."
        )
    return _Range(low, high, log=spec.get("log", False), step=spec.get("step"))


def build_search_space(model, candidates):
    """Build an Optuna search space from the values actually supplied.

    The counterpart of :func:`qbiocode.learning._grid.build_param_grid`, and
    deliberately agreeing with it on what counts as "not tuned" so that flipping
    ``tuner`` never changes which hyperparameters are searched.

    Args:
        model (str): Model name, used only to make error messages specific.
        candidates (dict): Maps hyperparameter name to what the config asked for. A
            list, tuple or set becomes a categorical choice; a ``{low, high}``
            mapping (optionally with ``log`` or ``step``) becomes an int or float
            range; a bare scalar or string is a one-value categorical, leaving it
            reported in ``best_params`` at the value the user pinned. ``None`` or an
            empty sequence means "not tuned" and is dropped, leaving the estimator's
            own default in force.

    Returns:
        dict: Maps hyperparameter name to a spec object with a ``suggest(trial, name)``
        method. Only the entries worth searching are present.

    Raises:
        ValueError: If nothing at all was supplied, or a range is malformed. An
            empty space is a config mistake rather than a one-point search, and
            saying so here names the config block instead of letting Optuna report
            an empty study.
    """
    space = {}
    for name, values in candidates.items():
        # Before any type test below, because every one of them passes for an OmegaConf
        # node and the damage is done later, in Optuna. See `to_plain`.
        values = to_plain(values)
        if values is None:
            continue
        if isinstance(values, Mapping):
            space[name] = _parse_range(model, name, values)
            continue
        # A string is a Sequence, so `max_features: sqrt` would otherwise be searched
        # as ['s', 'q', 'r', 't'] -- four invalid values, no error, and a best_params
        # that means nothing. Same guard as _grid.build_param_grid.
        if isinstance(values, str) or not isinstance(values, (Sequence, set, frozenset)):
            values = [values]
        values = list(values)
        if not values:
            continue
        space[name] = _Categorical(values)

    if not space:
        raise ValueError(
            f"Hyperparameter tuning was requested for {model!r} but no hyperparameter "
            f"values were given, so there is nothing to search. Either add a "
            f"'gridsearch_{model}_args' block to the config naming at least one "
            f"hyperparameter and the values to try, or set grid_search: False to "
            f"run {model!r} at its default hyperparameters. "
            f"Recognised hyperparameters for this model: "
            f"{', '.join(sorted(candidates))}."
        )
    return space


@contextlib.contextmanager
def _suppress_unstorable_choice_warning():
    """Silence Optuna's warning about choices it could not persist.

    A categorical choice that is not None/bool/int/float/str makes Optuna warn once per
    trial: "Choices for a categorical distribution should be a tuple of None, bool, int,
    float and str for persistent storage but contains [20] which is of type list." The
    shipped `gridsearch_mlp_args` hits it on every run -- its `hidden_layer_sizes` is
    `[[20], [50], [100]]` -- so a default config buried its own output under warnings.

    The warning is about writing a study to a database. These studies are created
    in memory and discarded when the search returns, so the limitation it describes
    cannot be reached from here. Coercing the values to tuples does *not* silence it
    (the element type is what it objects to) and would make `best_params` report
    something other than what the config said, so the warning is suppressed by message
    instead -- narrowly, and only around the search.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Choices for a categorical distribution should be a tuple.*",
        )
        yield


def _validate_budget(model, n_trials):
    """A budget below one trial is a config mistake, and it used to look like a crash.

    ``n_trials: 0`` reached ``study.best_params`` with nothing completed, so Optuna
    raised "No trials are completed yet" -- and in ``run_function_study`` the
    all-trials-failed branch caught it first and reported that every trial had *failed*
    and the search space was probably unbuildable. Both blame the wrong thing.
    """
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError(
            f"Tuning {model!r} needs at least one trial, but n_trials is {n_trials!r}. "
            f"Set it to a positive integer, or set grid_search: False to run at the "
            f"default hyperparameters."
        )


def _finite_size(space):
    """How many distinct points the space holds, or ``None`` if any dimension is a range."""
    total = 1
    for spec in space.values():
        if not isinstance(spec, _Categorical):
            return None
        total *= len(spec)
    return total


def run_study(
    estimator_cls, space, X, y, *, cv, n_trials, model=None, seed=None, fixed=None,
    scoring=None,
):
    """Search ``space`` with Optuna and return the best hyperparameters found.

    Scored by ``cross_val_score(..., scoring=scoring).mean()``. The scorer defaults to
    balanced accuracy, the statistic the study reports; it used to be the estimator's
    own ``score`` -- accuracy for a classifier -- which on an imbalanced dataset selects
    for majority-class prediction. Pass ``scoring='accuracy'`` (or set
    ``tuning_metric: accuracy``) to reproduce a search made before that change.

    Args:
        estimator_cls (type): Estimator to construct for each trial.
        space (dict): From :func:`build_search_space`.
        X (array-like): Training features.
        y (array-like): Training labels.
        cv (int): Number of cross-validation folds.
        n_trials (int): Trial budget. Lowered to the size of the space when the
            space is entirely categorical and smaller than the budget, so an
            8-value block does not spend 50 fits re-evaluating 8 models.
        model (str or None): Model name as the config spells it ('rf'), used only to
            make a rejected budget name the ``gridsearch_<model>_args`` block it came
            from. Falls back to the estimator's class name.
        seed (int or None): Seeds the sampler, so a run repeats at a given
            ``args['seed']``. ``None`` leaves Optuna drawing from the global RNG.
        fixed (dict or None): Passed to every trial's estimator but not searched --
            ``random_state`` in practice.
        scoring (TuningScorer, str, callable or None): What each trial maximises. A
            :class:`TuningScorer` (see :func:`tuning_scorer`) or a tuning-metric name;
            any other sklearn ``scoring`` is passed through. ``None`` means balanced
            accuracy.

    Returns:
        TunedParams: The best trial's hyperparameters, in the same shape
        ``GridSearchCV.best_params_`` returned, so callers refit unchanged -- with the
        metric, its best mean CV score and the number of trials run as attributes.
    """
    fixed = dict(fixed or {})
    metric, scoring = _scoring_parts(scoring)
    # The config key where the caller supplied one, not `estimator_cls.__name__`: every
    # other message in this module names the model as the config spells it ('rf'), which
    # is what makes "gridsearch_rf_args" findable. 'RandomForestClassifier' appears in no
    # config, so it left the one message that rejects a budget unable to point anywhere.
    _validate_budget(model or estimator_cls.__name__, n_trials)

    size = _finite_size(space)
    if size is not None:
        n_trials = min(n_trials, size)

    def objective(trial):
        params = {name: spec.suggest(trial, name) for name, spec in space.items()}
        estimator = estimator_cls(**params, **fixed)
        with warnings.catch_warnings():
            # A trial landing on a non-converging corner (saga at max_iter=5000, say)
            # is information, not a problem to report: the score it earns is what
            # steers the sampler away. GridSearchCV was equally quiet about it.
            warnings.simplefilter("ignore")
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        score = float(scores.mean())
        return score if math.isfinite(score) else float("-inf")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # n_jobs=1 on purpose. model_run.py already fans the models out over joblib
    # workers, and on macOS a QBioCode process has three LLVM OpenMP runtimes mapped
    # in under one install name (torch, qiskit-aer, xgboost); adding another layer of
    # threads is what kills the process in __kmp_fork_barrier, below Python and so
    # without a traceback. Parallelism belongs at the model level, where it already is.
    with _suppress_unstorable_choice_warning():
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return TunedParams(
        study.best_params, metric=metric, score=study.best_value, n_trials=len(study.trials)
    )


def search_hyperparameters(
    model,
    estimator_cls,
    candidates,
    X_train,
    y_train,
    *,
    cv,
    tuner="optuna",
    n_trials=50,
    seed=None,
    fixed=None,
    scoring=None,
):
    """Run whichever search ``tuner`` names, and return the best hyperparameters.

    The whole of what the eight classical ``compute_*_opt`` functions used to spell out
    individually. Each held the same twenty lines -- build a grid or a search space from
    the same ``candidates``, fit ``GridSearchCV`` or drive :func:`run_study`, take the
    best parameters -- differing only in the estimator class and the model name. Eight
    copies of a branch is eight places for the two engines to drift apart, and they had
    already started to: ``compute_nb_opt`` reads its seed from ``args`` because
    ``GaussianNB`` has no ``random_state`` for ``model_run`` to fill in.

    Args:
        model (str): Model name as the config spells it ('rf', 'catboost'), so an error
            names the ``gridsearch_<model>_args`` block the user has to edit.
        estimator_cls (type): Estimator to search over and construct.
        candidates (dict): Hyperparameter name -> what the config asked for. Passed
            unchanged to whichever of :func:`build_param_grid` or
            :func:`build_search_space` this engine uses, which is what keeps the two
            agreeing on which entries count as "not tuned".
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        cv (int): Number of cross-validation folds.
        tuner (str): ``'optuna'`` (default) samples ``n_trials`` configurations;
            ``'grid'`` restores the exhaustive ``GridSearchCV`` sweep. Any other value
            is treated as ``'optuna'`` -- ``model_run`` rejects an unknown name up
            front, and a direct caller passing one gets the default engine.
        n_trials (int): Trial budget, used by the Optuna engine only.
        seed (int or None): Seeds the sampler.
        fixed (dict or None): Settings every candidate estimator is built with but which
            are not searched -- ``random_state``, and CatBoost's quiet flags and resolved
            ``bootstrap_type``. The grid engine gets them by constructing its estimator
            with them, which is what the per-model code was already doing by hand.
        scoring (TuningScorer, str, callable or None): What both engines maximise --
            ``cross_val_score(scoring=...)`` for Optuna, ``GridSearchCV(scoring=...)``
            for the grid. Each ``compute_*_opt`` passes ``tuning_scorer(args)``. ``None``
            means balanced accuracy; ``'accuracy'`` restores the old default.

    Returns:
        TunedParams: The best hyperparameters found, in the shape
        ``GridSearchCV.best_params_`` returned, so callers refit unchanged, carrying
        the metric and the best mean cross-validated score as attributes.
    """
    fixed = dict(fixed or {})
    metric, scoring = _scoring_parts(scoring)
    # Optuna by default; the exhaustive grid stays reachable so a number published
    # against it can still be reproduced. Both engines are handed the same `candidates`,
    # so switching `tuner` never changes *which* hyperparameters are searched -- only how
    # the search spends its fits.
    if tuner == "grid":
        search = GridSearchCV(
            estimator_cls(**fixed),
            param_grid=build_param_grid(model, candidates),
            cv=cv,
            scoring=scoring,
        )
        search.fit(X_train, y_train)
        return TunedParams(
            search.best_params_,
            metric=metric,
            score=search.best_score_,
            n_trials=len(search.cv_results_["params"]),
        )
    return run_study(
        estimator_cls,
        build_search_space(model, candidates),
        X_train,
        y_train,
        cv=cv,
        n_trials=n_trials,
        model=model,
        seed=seed,
        fixed=fixed,
        scoring=scoring,
    )


# --------------------------------------------------------------------------------------
# Quantum models: tuning a whole compute function rather than an estimator
# --------------------------------------------------------------------------------------
#
# The classical `_opt` learners hand `run_study` an estimator class and let
# `cross_val_score` fit it. The quantum learners have no estimator to hand over: a
# hyperparameter like `encoding` or `reps` selects a *feature map*, which selects a
# kernel or an ansatz, and `compute_qsvc` and friends build that chain internally and
# go straight from raw arrays to a `modeleval` frame. Exposing an sklearn-compatible
# estimator from each would mean restructuring five functions that already work.
#
# So the objective calls the compute function itself and reads the configured
# `tuning_metric` back out of the frame's metrics dict. Two consequences worth knowing:
#
#   * Scoring is a single stratified holdout carved from the training data, not k-fold.
#     A quantum fit computes an n-by-n fidelity kernel by circuit simulation -- seconds,
#     not milliseconds -- so k-fold would multiply an already expensive search by k. A
#     noisier score per trial buys more trials for the same wall clock.
#   * A hyperparameter combination the stack rejects is pruned rather than fatal. Not
#     every (encoding, entanglement, primitive) triple is constructible, and one bad
#     corner should cost a trial, not the run.

#: Backends that cost nothing but local CPU time. Tuning against anything else means
#: one queued hardware job per trial, so it has to be asked for explicitly.
# Both the internal names and the config-facing aliases: qprofiler resolves aliases
# before tuning runs, but a direct caller of the tuning API may not have.
_FREE_BACKENDS = frozenset(
    {"simulator", "simulator_aer", "statevector_simulator", "mps_simulator"}
)


def _metric_dicts(frame, model):
    """Every metrics dict in a ``modeleval`` frame, one per ``results_<label>`` column.

    Usually there is exactly one. ``compute_qpl`` is the exception: it fits a classical
    head per entry in ``classical_models`` on the same quantum projection and
    ``pd.concat``s a frame per head, so the result carries a ``results_qpl_rf``, a
    ``results_qpl_svc`` and so on, each populated on its own row and NaN elsewhere.

    Hence the columns are *found* rather than derived from the label the caller passed:
    a QPL frame appends the head name to it, so ``model='qpl_opt'`` produces
    ``results_qpl_opt_rf`` and five siblings, none of them named ``results_qpl_opt``.
    """
    dicts = []
    for column in [c for c in frame.columns if c.startswith("results_")]:
        for value in frame[column]:
            if isinstance(value, dict):
                dicts.append(value)
                break
    if not dicts:
        raise ValueError(
            f"scoring {model!r} found no populated 'results_' column in the frame "
            f"modeleval returned; its columns are {sorted(frame.columns)}."
        )
    return dicts


def _metric_of(frame, model, metric=DEFAULT_TUNING_METRIC):
    """Score one trial: the mean of ``metric`` across whatever models the frame reports.

    ``metric`` is a ``modeleval`` metrics-dict key -- every name in ``_TUNING_METRICS``
    is one. For everything but QPL that is a single value. For QPL it averages over the
    classical heads fitted on the quantum projection, which is the point being tuned --
    a projection that is broadly informative. Taking the *best* head instead would let
    one lucky head choose the projection, and the frame reports every head either way.
    """
    values = [float(metrics[metric]) for metrics in _metric_dicts(frame, model)]
    return sum(values) / len(values)


def _accuracy_of(frame, model):
    """:func:`_metric_of` at ``'accuracy'`` -- the objective every run used before
    ``tuning_metric`` existed."""
    return _metric_of(frame, model, "accuracy")


def ensure_tuning_is_affordable(args, model):
    """Refuse to tune against real quantum hardware unless it was asked for.

    Every trial is a separate fit, and on a device each fit is a queued job billed
    against the user's instance. Tuning a quantum model on hardware by leaving
    ``tune_quantum: True`` in a config written for the simulator is a mistake that
    costs money and hours, and it is silent -- the run simply never seems to finish.
    """
    backend = args.get("backend", "simulator")
    if backend in _FREE_BACKENDS or args.get("allow_hardware_tuning", False):
        return
    raise ValueError(
        f"Refusing to tune {model!r} on backend {backend!r}: hyperparameter tuning runs "
        f"one fit per trial, and on a real device every fit is a queued job. Either set "
        f"backend: simulator for the tuning run, drop tune_quantum, or -- if you really "
        f"mean to spend device time -- set allow_hardware_tuning: True."
    )


def seed_from(args):
    """The run's ``seed`` for a tuner, from a plain dict or a Hydra ``DictConfig``.

    Not ``isinstance(args, dict)``: ``DictConfig`` is a ``MutableMapping`` and not a
    ``dict``, so under the CLI -- the only way the pilot runs -- that test failed and
    the qsvc, pqk, qnn, vqc, qpl and nb tuners were given ``seed=None``. TPE then drew
    its trials from OS entropy: four tunings of one sonar split, on features with one
    sha256, picked four different ``C``. The unit tests pass plain dicts, so they never
    saw it.
    """
    return args.get("seed") if isinstance(args, Mapping) else None


def run_function_study(
    compute_fn,
    space,
    X_train,
    y_train,
    args,
    *,
    model,
    n_trials,
    seed=None,
    validation_split=0.25,
    fixed=None,
    data_key=None,
):
    """Tune a ``compute_*`` function by scoring it on an inner validation split.

    Args:
        compute_fn (callable): A ``compute_*`` function taking
            ``(X_train, X_test, y_train, y_test, args, **hyperparameters)`` and
            returning a ``modeleval`` frame.
        space (dict): From :func:`build_search_space`.
        X_train (array-like): Training features. Split again internally; the caller's
            test set is never touched, so the reported score stays honest.
        y_train (array-like): Training labels.
        args (dict): The run's config. Passed through to ``compute_fn`` because the
            quantum functions read ``backend``, ``shots`` and ``seed`` from it. Its
            ``tuning_metric`` (default ``'balanced_accuracy'``) names the ``modeleval``
            metric each trial is scored on.
        model (str): Model name, for error messages.
        n_trials (int): Trial budget, lowered to the size of a finite space.
        seed (int or None): Seeds both the sampler and the inner split.
        validation_split (float): Fraction of the training data held out to score on.
        fixed (dict or None): Passed to every trial but not searched.
        data_key (str or None): The per-pass key from ``model_run``. Only used to
            freeze and reuse the search across resamples when the config sets
            ``freeze_quantum_params: True``; ``None`` disables that entirely, so a
            caller that does not pass it behaves exactly as before.

    Returns:
        TunedParams: The best trial's hyperparameters -- from a fresh search, or from
        the frozen file written by the first resample when freezing is on
        (``.reused`` is then True and ``.score`` is the score recorded when the file
        was written, NaN for a file from before scores were recorded).

    Raises:
        ValueError: If ``tuning_metric`` is unknown, if tuning would run against real
            hardware without ``allow_hardware_tuning``, if the data cannot be split so
            that both sides carry every class, or if every trial failed.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    from qbiocode.learning._param_cache import load_frozen_params, save_frozen_params

    # Checked before every guard below, deliberately. On a hit no search runs at all, so
    # the hardware-affordability check, the trial budget and the class-count feasibility
    # of the inner split are all moot -- they constrain searching, not refitting. The
    # caller then refits on the real split at these parameters exactly as it would have
    # done with a freshly searched set, and the row keeps its '_opt' label, which matters:
    # a frozen resample labelled '<model>' instead would split one arm across the
    # iteration axis and break the pairing that fair_selection depends on.
    metric = tuning_metric(args)
    if data_key is not None:
        frozen = load_frozen_params(args, data_key, model, space, metric=metric)
        if frozen is not None:
            return frozen

    ensure_tuning_is_affordable(args, model)
    _validate_budget(model, n_trials)
    # Validated here rather than left to train_test_split, which reports it against
    # `test_size` -- a parameter name that appears nowhere in the user's config.
    if not isinstance(validation_split, float) or not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be a fraction strictly between 0 and 1 to score "
            f"{model!r}, got {validation_split!r}. It is the share of the training data "
            f"held back to score each candidate; 0.25 is the default."
        )
    fixed = dict(fixed or {})

    y_array = np.asarray(y_train)
    classes, counts = np.unique(y_array, return_counts=True)
    if counts.min() < 2:
        raise ValueError(
            f"Cannot tune {model!r}: class {classes[counts.argmin()]!r} has only "
            f"{counts.min()} training sample, so no validation split can contain it. "
            f"Use more data per class, or turn tuning off for this model."
        )
    X_inner, X_val, y_inner, y_val = train_test_split(
        X_train, y_train, test_size=validation_split, stratify=y_array, random_state=seed
    )

    # modeleval branches on this key, and reads it with [] rather than .get. The inner
    # score only needs the metrics row, so take the cheaper branch.
    scoring_args = {**args, "grid_search": False}

    size = _finite_size(space)
    if size is not None:
        n_trials = min(n_trials, size)

    failures = []

    def objective(trial):
        params = {name: spec.suggest(trial, name) for name, spec in space.items()}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The quantum functions narrate to stdout -- qubit and parameter counts
                # per fit. Useful once, unreadable n_trials times inside a joblib worker.
                with contextlib.redirect_stdout(io.StringIO()):
                    frame = compute_fn(
                        X_inner, X_val, y_inner, y_val, scoring_args,
                        **params, **fixed,
                    )
            score = _metric_of(frame, model, metric)
        except Exception as error:  # noqa: BLE001 -- an unbuildable corner costs a trial
            failures.append(f"{params}: {type(error).__name__}: {error}")
            raise optuna.TrialPruned() from error
        # Outside the try: a NaN metric is a scored trial, not a failed one. Same
        # reading as run_study -- the worst possible score, so the sampler moves away.
        return score if math.isfinite(score) else float("-inf")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # Serial for the same reason as run_study: model_run already parallelises over
    # models, and the quantum stack maps its own OpenMP runtime in alongside torch's.
    #
    # The scratch projection directory is what makes PQK and QPL tunable at all. They
    # cache the projected feature matrix under a name built from `data_key` and a
    # fingerprint of the feature map -- not the row count. A trial projects the inner
    # split (22 rows, say) and writes that file; the final fit then projects the whole
    # training set (30 rows), matches the same name, and is stopped by PQK's own
    # row-count guard: "Projection file ... has 22 rows, but the current dataset expects
    # 30 rows. Remove this projection file or use a different pqk_projection_dir." Trial
    # projections are throwaway, so they get their own directory and are deleted with it.
    with tempfile.TemporaryDirectory(prefix="qbiocode_tuning_") as scratch:
        scoring_args["pqk_projection_dir"] = scratch
        scoring_args["qpl_projection_dir"] = scratch
        with _suppress_unstorable_choice_warning():
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        detail = "\n  ".join(failures[:5]) or "no trials ran"
        raise ValueError(
            f"Every tuning trial for {model!r} failed, so there is no best "
            f"configuration to report. The first failures were:\n  {detail}\n"
            f"This usually means the search space names a combination the quantum "
            f"stack cannot build -- check the 'gridsearch_{model}_args' block."
        )
    if failures:
        warnings.warn(
            f"{len(failures)} of {n_trials} tuning trials for {model!r} failed and were "
            f"skipped; the best of the {len(completed)} that ran was used. First "
            f"failure: {failures[0]}",
            UserWarning,
            stacklevel=2,
        )
    best = TunedParams(
        study.best_params, metric=metric, score=study.best_value, n_trials=len(study.trials)
    )
    if data_key is not None:
        # Best-effort: a failed write is logged and ignored, costing later resamples a
        # redundant search rather than aborting a sweep hours in over a cache file.
        save_frozen_params(args, data_key, model, best, metric=metric, score=best.score)
    return best


def record_tuned_params(frame, best_params, beg_time):
    """Report the tuned hyperparameters and the whole search's wall clock.

    A ``compute_*_opt`` wrapper for a quantum model tunes, then calls the base
    function once on the real split. That base call writes its own parameter dict --
    the feature map and kernel class names it happened to build -- and its own
    ``time``, which covers only the final fit and so understates the run by the whole
    search. Both are corrected here in place.

    The parameter entry ends up as the base function's description with the tuned
    values laid over it, which is strictly more informative than either alone: the
    tuned block names ``encoding`` and ``reps``, the base block names the
    ``ZZFeatureMap`` they produced.

    When ``best_params`` is a :class:`TunedParams` -- what :func:`run_function_study`
    returns -- its ``tuning_metric``, ``tuning_score`` and ``tuning_reused`` are written
    into each metrics row as well, so the validation score of the chosen configuration
    reaches ModelResults.csv. The parameter entry itself stays a plain dict.
    """
    elapsed = time.time() - beg_time
    evidence = best_params.evidence() if isinstance(best_params, TunedParams) else {}
    # QPL reports one row per classical head; each carries its own parameter dict and
    # each understates the time by the whole search, so every one is corrected.
    for metrics in _metric_dicts(frame, "tuned model"):
        key = "BestParams_Tuned" if "BestParams_Tuned" in metrics else "Model_Parameters"
        existing = metrics.get(key)
        metrics[key] = (
            {**existing, **best_params} if isinstance(existing, dict) else dict(best_params)
        )
        metrics["time"] = elapsed
        metrics.update(evidence)
    return frame
