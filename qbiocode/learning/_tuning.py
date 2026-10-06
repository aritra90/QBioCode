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
import contextvars
import io
import logging
import math
import numbers
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import optuna
from sklearn.metrics import f1_score, get_scorer, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

from qbiocode.evaluation.protocol import TRIALS_PREFIX, TrialRecord
from qbiocode.evaluation.protocol import trial_log as protocol_trial_log
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
        trials (list of TrialRecord or None): Every trial of a search scored on a
            :class:`~qbiocode.evaluation.protocol.ValidationSplit` (``split_mode:
            manifest``), in trial order; ``None`` for every other search, which keeps
            only its winner.
        best_trial (int or None): ``number`` of the trial whose parameters these are.
        val_idx (array-like or None): Row ids of the validation rows the trials' ``y_pred``
            and ``y_score`` are aligned with.
        y_val (array-like or None): Labels of those rows.
        fixed (dict or None): The non-searched params every trial was fitted with (the
            trials' ``params`` hold only the searched names); ``None`` without trials.
    """

    def __init__(self, params=(), *, metric=None, score=float("nan"), n_trials=None,
                 reused=False, trials=None, best_trial=None, val_idx=None, y_val=None,
                 fixed=None):
        super().__init__(params)
        self.metric = metric
        self.score = float(score) if score is not None else float("nan")
        self.n_trials = n_trials
        self.reused = bool(reused)
        self.trials = None if trials is None else list(trials)
        self.best_trial = None if best_trial is None else int(best_trial)
        self.val_idx = val_idx
        self.y_val = y_val
        self.fixed = None if fixed is None else dict(fixed)

    def trial_log(self):
        """The ``trials_<model>`` cell for these trials (see
        :func:`qbiocode.evaluation.protocol.trial_log`), or ``None`` without trials."""
        if not self.trials:
            return None
        return protocol_trial_log(
            self.trials, metric=self.metric, best=self.best_trial,
            val_idx=self.val_idx, y_val=self.y_val,
            # getattr: a TunedParams pickled before the field existed has no attribute.
            fixed=getattr(self, "fixed", None),
        )

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


def _plain_scalar(value):
    """A numpy scalar as the Python number it holds; anything else unchanged.

    A config built in code can carry ``np.int64(8)``, which is no ``int`` to an
    ``isinstance`` test yet is the same default as ``8``.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _extend_range(spec, value):
    """``spec`` widened just enough to hold ``value``, or ``None`` if no range can.

    A range holds numbers only: ``None``, a string or a bool default has no place in it,
    nor has a fractional value in an int range, a non-positive one on a log scale, or one
    off the ``step`` lattice.
    """
    value = _plain_scalar(value)
    if not _is_number(value):
        return None
    if not math.isfinite(value):
        return None
    if spec.is_int:
        if float(value) != int(value):
            return None
        value = int(value)
    else:
        value = float(value)
    if spec.log and value <= 0:
        return None
    if spec.step is not None:
        offset = (value - spec.low) / spec.step
        if not math.isclose(offset, round(offset), abs_tol=1e-9):
            return None
    low, high = min(spec.low, value), max(spec.high, value)
    if not spec.is_int:
        low, high = float(low), float(high)
    return _Range(low, high, log=spec.log, step=spec.step)


def _range_holds(spec, value):
    value = _plain_scalar(value)
    if not _is_number(value):
        return False
    if spec.is_int and float(value) != int(value):
        return False
    return spec.low <= value <= spec.high


def _as_layers(value):
    """``value`` as a tuple of layer widths when it spells one, else ``None``.

    ``hidden_layer_sizes=100`` and ``[100]`` (or ``(100,)``) build the same network;
    sklearn reads a bare int as one hidden layer.
    """
    value = _plain_scalar(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return (int(value),)
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(_plain_scalar(v), numbers.Integral)
        and not isinstance(_plain_scalar(v), bool) for v in value
    ):
        return tuple(int(_plain_scalar(v)) for v in value)
    return None


def _matching_choice(values, value):
    """The configured choice equal to ``value``, or equivalent to it, else a sentinel.

    Equal first; then, for a list or tuple choice, the same layer widths (so a default
    ``100`` finds a configured ``[100]`` rather than being added beside it as a second
    spelling of one network).
    """
    for choice in values:
        try:
            if bool(choice == value):
                return choice
        except (TypeError, ValueError):
            continue
    layers = _as_layers(value)
    if layers is not None:
        for choice in values:
            if isinstance(choice, (list, tuple)) and _as_layers(choice) == layers:
                return choice
    return _NO_CHOICE


#: Returned by :func:`_matching_choice` when no configured choice matches.
_NO_CHOICE = object()


def _default_trial(model, space, default_params):
    """The trial-0 config, the space it needs, and whether it is the whole default.

    Restricts ``default_params`` to the searched names -- the rest are not the tuner's
    business; a caller that wants them honoured passes them as ``fixed``. A default the
    space cannot represent makes Optuna refuse the enqueued trial (a categorical value
    outside the choices raises in ``to_internal_repr``) or report a point outside the
    distribution, so the space is widened by exactly that value, and the widening is
    logged: the search is then over what the config asked for plus the default. A
    categorical default equivalent to a configured choice (``100`` beside ``[100]``)
    enqueues that choice instead.

    A range default no range can hold (``gamma='scale'`` against a numeric range,
    ``max_depth=None``) is not enqueued, nor is a searched name with no default; trial
    0 samples those names, and the third return value is False so the trial is not
    recorded as the default config.

    Args:
        model (str): Model name, for the log.
        space (dict): From :func:`build_search_space`. Not modified.
        default_params (Mapping or None): The arm's default config.

    Returns:
        tuple: ``(enqueue, space, complete)`` -- the values to enqueue (empty when no
        searched name has a usable default), the possibly widened copy of ``space``,
        and True when every searched name was enqueued (trial 0 is then the whole
        default config).
    """
    space = dict(space)
    enqueue = {}
    for name, value in dict(to_plain(dict(default_params or {}))).items():
        if name not in space:
            continue
        value = _plain_scalar(value)
        spec = space[name]
        if isinstance(spec, _Categorical):
            choice = _matching_choice(spec.values, value)
            if choice is _NO_CHOICE:
                space[name] = _Categorical(spec.values + [value])
                logger.info(
                    "tuning %r: default %s=%r is not among the configured choices %r; "
                    "added it so the default config can be trial 0.",
                    model, name, value, spec.values,
                )
                choice = value
            enqueue[name] = choice
            continue
        if _range_holds(spec, value):
            enqueue[name] = value
            continue
        widened = _extend_range(spec, value)
        if widened is None:
            logger.info(
                "tuning %r: default %s=%r cannot be represented in the range "
                "[%r, %r]; trial 0 samples %s instead.",
                model, name, value, spec.low, spec.high, name,
            )
            continue
        space[name] = widened
        enqueue[name] = int(value) if widened.is_int else float(value)
        logger.info(
            "tuning %r: default %s=%r is outside the configured range [%r, %r]; "
            "widened it to [%r, %r] so the default config can be trial 0.",
            model, name, value, spec.low, spec.high, widened.low, widened.high,
        )
    sampled = [name for name in space if name not in enqueue]
    if enqueue and sampled:
        logger.info(
            "tuning %r: trial 0 samples %s, so it is not the default config and is not "
            "recorded as one.", model, ", ".join(sampled),
        )
    return enqueue, space, not sampled


def _predictions_score(scoring, estimator, X_val, y_val, y_pred):
    """One validation score, from predictions already made where the metric allows.

    A :class:`TuningScorer` names a label metric, which is a function of ``y_pred``
    alone -- the same function sklearn's scorer would reach after predicting again.
    Anything else goes through sklearn's own scorer resolution.
    """
    if isinstance(scoring, TuningScorer):
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, matthews_corrcoef,
        )

        if scoring.metric == "accuracy":
            return float(accuracy_score(y_val, y_pred))
        if scoring.metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_val, y_pred))
        if scoring.metric == "mcc":
            return float(matthews_corrcoef(y_val, y_pred))
        return float(f1_score(y_val, y_pred, average=scoring.average, zero_division=0))
    from sklearn.metrics import check_scoring

    return float(check_scoring(estimator, scoring=scoring)(estimator, X_val, y_val))


def check_fit_status(estimator, model, params=None, *, level=logging.WARNING):
    """Log, and report, a fit that libsvm stopped before it converged.

    ``SVC`` (and anything else built on libsvm) sets ``fit_status_`` to 1 when the
    solver hit ``max_iter`` rather than its tolerance. The fitted model is usable but it
    is not the optimum of its own objective, so a validation score earned by it measures
    the cap, not the config. :func:`run_study` fails such a trial; a ``compute_*_opt``
    wrapper calls this after its refit, which is kept and returned all the same.

    Args:
        estimator: A fitted estimator. One without ``fit_status_`` counts as converged.
        model (str): Model name, for the message.
        params (Mapping or None): The config fitted, for the message.
        level (int): Logging level of the message.

    Returns:
        bool: True when the fit converged (or reports nothing).
    """
    status = getattr(estimator, "fit_status_", 0)
    if status == 0:
        return True
    logger.log(
        level,
        "%r did not converge (fit_status_=%r: the solver stopped at max_iter) "
        "with params %r.", model, status, dict(params or {}),
    )
    return False


def _validation_study(estimator_cls, space, validation, *, n_trials, model, seed, fixed,
                      scoring, metric, default_params):
    """:func:`run_study` on a fixed validation split: one fit per trial, every trial kept."""
    from qbiocode.evaluation.model_evaluation import extract_binary_scores

    enqueue, space, whole = _default_trial(model, space, default_params)
    # Trial 0 is the default config only when every searched default was enqueued.
    is_default = bool(enqueue) and whole
    size = _finite_size(space)
    if size is not None:
        n_trials = min(n_trials, size)
    records = {}
    unconverged = set()

    def objective(trial):
        params = {name: spec.suggest(trial, name) for name, spec in space.items()}
        start = time.perf_counter()
        record = TrialRecord(
            number=trial.number, params=params, value=math.nan, state="FAIL",
            is_default=is_default and trial.number == 0,
        )
        records[trial.number] = record
        try:
            estimator = estimator_cls(**params, **fixed)
            with warnings.catch_warnings():
                # Same reading as the cross-validated objective: a non-converging corner
                # is information for the sampler, not something to print.
                warnings.simplefilter("ignore")
                estimator.fit(validation.X_fit, validation.y_fit)
                if not check_fit_status(estimator, model, params, level=logging.INFO):
                    unconverged.add(trial.number)
                    return float("nan")
                y_pred = estimator.predict(validation.X_val)
                y_score = extract_binary_scores(estimator, validation.X_val)
                score = _predictions_score(
                    scoring, estimator, validation.X_val, validation.y_val, y_pred
                )
        except Exception as error:  # noqa: BLE001 -- an unfittable corner costs a trial
            logger.info("tuning %r: trial %d with params %r failed: %s: %s",
                        model, trial.number, params, type(error).__name__, error)
            return float("nan")
        finally:
            record.duration_s = time.perf_counter() - start
        value = score if math.isfinite(score) else float("-inf")
        record.value, record.state = value, "COMPLETE"
        record.y_pred, record.y_score = y_pred, y_score
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if enqueue:
        study.enqueue_trial(enqueue)
    # A NaN return is how a trial is marked FAIL without stopping the study: Optuna
    # records it as failed, TPE ignores it, and the next trial runs.
    with _suppress_unstorable_choice_warning():
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return _tuned_from_study(study, records, model, metric, validation, fixed=fixed,
                             unconverged=unconverged)


def _tuned_from_study(study, records, model, metric, validation, *, fixed=None,
                      unconverged=()):
    """The :class:`TunedParams` of a validation-scored study, trials attached.

    ``unconverged`` names the trials that failed only because libsvm stopped at
    ``max_iter``. When every trial failed that way the arm is not lost: trial 0 (the
    default config where one was enqueued) is returned with a NaN score and a warning,
    and the caller's refit -- which :func:`check_fit_status` warns about in turn -- runs
    on it. Any other all-failed study raises.
    """
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials = [records[t.number] for t in study.trials if t.number in records]
    numbers = {t.number for t in study.trials}
    if not complete and numbers and numbers <= set(unconverged):
        first = records[min(numbers)]
        logger.warning(
            "Every one of the %d tuning trials for %r stopped at max_iter without "
            "converging (fit_status_ != 0), so none has a validation score; falling "
            "back to trial %d's params %r, with tuning_score NaN.",
            len(numbers), model, first.number, dict(first.params),
        )
        return TunedParams(
            first.params, metric=metric, score=float("nan"), n_trials=len(study.trials),
            trials=trials, best_trial=first.number, val_idx=validation.val_idx,
            y_val=validation.y_val, fixed=fixed,
        )
    if not complete:
        raise ValueError(
            f"Every tuning trial for {model!r} failed on the validation split, so there "
            f"is no best configuration to report. The trial log (level INFO) names the "
            f"params and error of each; check the 'gridsearch_{model}_args' block."
        )
    failed = len(study.trials) - len(complete)
    if failed:
        logger.warning(
            "%d of %d tuning trials for %r failed and were recorded as FAIL; the best "
            "of the %d that ran was used.", failed, len(study.trials), model,
            len(complete),
        )
    return TunedParams(
        study.best_params, metric=metric, score=study.best_value,
        n_trials=len(study.trials), trials=trials, best_trial=study.best_trial.number,
        val_idx=validation.val_idx, y_val=validation.y_val, fixed=fixed,
    )


def run_study(
    estimator_cls, space, X, y, *, cv, n_trials, model=None, seed=None, fixed=None,
    scoring=None, validation=None, default_params=None,
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
        validation (ValidationSplit or None): ``split_mode: manifest``. Each trial is
            then ONE fit on ``validation.X_fit`` scored on ``validation.X_val`` -- no
            cross-validation, and ``X``, ``y`` and ``cv`` are not used by the search
            (the caller refits the winner on its full ``X_train``). A trial that raises,
            or whose fit reports ``fit_status_ != 0`` (libsvm stopped at ``max_iter``),
            is recorded as FAIL with a NaN value and the study goes on. Every trial is
            returned as a :class:`~qbiocode.evaluation.protocol.TrialRecord` with its
            validation predictions. ``None`` (the default) is the cross-validated search
            above, unchanged.
        default_params (Mapping or None): The arm's default config. Its searched names
            are enqueued as trial 0, which counts inside ``n_trials``; a default the
            space cannot hold widens the space by that value (logged). ``None``
            enqueues nothing.

    Returns:
        TunedParams: The best trial's hyperparameters, in the same shape
        ``GridSearchCV.best_params_`` returned, so callers refit unchanged -- with the
        metric, its best mean CV score (the validation score with ``validation``) and
        the number of trials run as attributes, and with ``validation`` every trial.

    Raises:
        ValueError: With ``validation``, if every trial failed -- unless every one
            failed only by stopping at ``max_iter``, when trial 0's params come back
            with a NaN score and a warning.
    """
    fixed = dict(fixed or {})
    metric, scoring = _scoring_parts(scoring)
    # The config key where the caller supplied one, not `estimator_cls.__name__`: every
    # other message in this module names the model as the config spells it ('rf'), which
    # is what makes "gridsearch_rf_args" findable. 'RandomForestClassifier' appears in no
    # config, so it left the one message that rejects a budget unable to point anywhere.
    _validate_budget(model or estimator_cls.__name__, n_trials)

    if validation is not None:
        return _validation_study(
            estimator_cls, space, validation, n_trials=n_trials,
            model=model or estimator_cls.__name__, seed=seed, fixed=fixed,
            scoring=scoring, metric=metric, default_params=default_params,
        )
    enqueue = {}
    if default_params is not None:
        enqueue, space, _ = _default_trial(model or estimator_cls.__name__, space,
                                        default_params)

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
    if enqueue:
        study.enqueue_trial(enqueue)
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
    validation=None,
    default_params=None,
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
        validation (ValidationSplit or None): Score every trial on this fixed split
            instead of by cross-validation (``split_mode: manifest``); forwarded to
            :func:`run_study`. Optuna engine only.
        default_params (Mapping or None): The arm's default config, enqueued as trial 0;
            forwarded to :func:`run_study`. Optuna engine only.

    Returns:
        TunedParams: The best hyperparameters found, in the shape
        ``GridSearchCV.best_params_`` returned, so callers refit unchanged, carrying
        the metric and the best mean cross-validated score as attributes.

    Raises:
        ValueError: If ``tuner='grid'`` is combined with ``validation``: the grid fits
            every point whatever ``n_trials`` says, so the arms could not be given one
            equal budget.
    """
    fixed = dict(fixed or {})
    metric, scoring = _scoring_parts(scoring)
    if tuner == "grid" and validation is not None:
        raise ValueError(
            f"tuner: 'grid' cannot tune {model!r} on a validation split: the exhaustive "
            f"grid ignores n_trials, so the equal per-arm trial budget of split_mode: "
            f"manifest cannot hold. Use tuner: optuna."
        )
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
        validation=validation,
        default_params=default_params,
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
    validation=None,
    default_params=None,
    reseed=None,
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
        validation (ValidationSplit or None): ``split_mode: manifest``. Every trial
            is then ``compute_fn(X_fit, X_val, y_fit, y_val, ...)`` on this split: no
            inner ``train_test_split`` (``validation_split`` is ignored), no frozen
            parameters loaded or saved, and no ``kernel_dump_dir`` in the trial's args,
            so trials cannot overwrite the refit's kernel dumps. A trial that raises is
            recorded as FAIL with a NaN value and the study goes on. Every trial is
            returned as a :class:`~qbiocode.evaluation.protocol.TrialRecord`, with the
            validation predictions read from the frame's single ``y_predicted_*`` /
            ``y_score_*`` pair (None when the frame has several, as QPL's heads do).
        default_params (Mapping or None): The arm's default config; its searched names
            are enqueued as trial 0 (see :func:`run_study`).
        reseed (callable or None): Called with no arguments before every trial and once
            more just before returning, i.e. before the caller's refit. model_run passes
            one that resets the global RNGs to the run's seeds, so a model whose initial
            point comes from a global stream (qnn, vqc) starts every trial -- and the
            refit -- from the same state, whatever ran before it.

    Returns:
        TunedParams: The best trial's hyperparameters -- from a fresh search, or from
        the frozen file written by the first resample when freezing is on
        (``.reused`` is then True and ``.score`` is the score recorded when the file
        was written, NaN for a file from before scores were recorded). With
        ``validation`` it also carries every trial.

    Raises:
        ValueError: If ``tuning_metric`` is unknown, if tuning would run against real
            hardware without ``allow_hardware_tuning``, if the data cannot be split so
            that both sides carry every class, or if every trial failed.
    """
    if validation is not None:
        return _validation_function_study(
            compute_fn, space, args, validation, model=model, n_trials=n_trials,
            seed=seed, fixed=fixed, default_params=default_params, reseed=reseed,
        )
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

    enqueue = {}
    if default_params is not None:
        enqueue, space, _ = _default_trial(model, space, default_params)
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
    if enqueue:
        study.enqueue_trial(enqueue)
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


def _frame_predictions(frame):
    """``(y_pred, y_score)`` of a one-model ``modeleval`` frame, or ``(None, None)``.

    Read from the single ``y_predicted_<label>`` column; a frame carrying several (QPL,
    one per classical head) has no one prediction per row to report.
    """
    import numpy as np

    columns = [c for c in frame.columns if c.startswith("y_predicted_")]
    if len(columns) != 1:
        return None, None
    label = columns[0][len("y_predicted_"):]
    values = [v for v in frame[columns[0]] if v is not None]
    if not values:
        return None, None
    y_pred = np.asarray(values[0])
    y_score = None
    score_column = "y_score_" + label
    if score_column in frame.columns:
        scores = [v for v in frame[score_column] if v is not None]
        if scores and np.ndim(scores[0]) == 1:
            y_score = np.asarray(scores[0])
    return y_pred, y_score


#: The number of the validation-study trial whose ``compute_fn`` call is running, else
#: None. A wrapper that keeps something of each trial's call (QPL's per-head frames)
#: keys it by this, so a trial that never reached the call costs only its own entry.
_CURRENT_TRIAL = contextvars.ContextVar("qbiocode_tuning_trial", default=None)


def current_trial_number():
    """The number of the validation-study trial now calling ``compute_fn``, or None."""
    return _CURRENT_TRIAL.get()


def _validation_function_study(compute_fn, space, args, validation, *, model, n_trials,
                               seed, fixed, default_params, reseed):
    """:func:`run_function_study` on a fixed validation split, every trial kept."""
    from qbiocode.learning._param_cache import freezing_enabled

    metric = tuning_metric(args)
    # Params frozen on one fold and reused on the next would be tuned on rows that are
    # the next fold's test rows; the protocol forbids it, so no freeze load or save.
    if freezing_enabled(args):
        raise ValueError(
            f"freeze_quantum_params cannot be combined with a validation split "
            f"(split_mode: manifest) when tuning {model!r}: every fold is tuned on its "
            f"own validation rows. Set freeze_quantum_params: False."
        )
    ensure_tuning_is_affordable(args, model)
    _validate_budget(model, n_trials)
    fixed = dict(fixed or {})

    scoring_args = {**args, "grid_search": False}
    # The kernel dumps belong to the refit on the outer split; a trial writing them
    # would overwrite (or pre-empt) gram_/proj_ files named for the same data_key.
    scoring_args.pop("kernel_dump_dir", None)

    enqueue, space, whole = _default_trial(model, space, default_params)
    # Trial 0 is the default config only when every searched default was enqueued.
    is_default = bool(enqueue) and whole
    size = _finite_size(space)
    if size is not None:
        n_trials = min(n_trials, size)
    records = {}

    def objective(trial):
        params = {name: spec.suggest(trial, name) for name, spec in space.items()}
        record = TrialRecord(
            number=trial.number, params=params, value=math.nan, state="FAIL",
            is_default=is_default and trial.number == 0,
        )
        records[trial.number] = record
        start = time.perf_counter()
        try:
            if reseed is not None:
                reseed()
            token = _CURRENT_TRIAL.set(trial.number)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()):
                        frame = compute_fn(
                            validation.X_fit, validation.X_val,
                            validation.y_fit, validation.y_val, scoring_args,
                            **params, **fixed,
                        )
            finally:
                _CURRENT_TRIAL.reset(token)
            score = _metric_of(frame, model, metric)
            y_pred, y_score = _frame_predictions(frame)
        except Exception as error:  # noqa: BLE001 -- an unbuildable corner costs a trial
            logger.info("tuning %r: trial %d with params %r failed: %s: %s",
                        model, trial.number, params, type(error).__name__, error)
            return float("nan")
        finally:
            record.duration_s = time.perf_counter() - start
        value = score if math.isfinite(score) else float("-inf")
        record.value, record.state = value, "COMPLETE"
        record.y_pred, record.y_score = y_pred, y_score
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if enqueue:
        study.enqueue_trial(enqueue)
    # The scratch projection directory: see run_function_study.
    with tempfile.TemporaryDirectory(prefix="qbiocode_tuning_") as scratch:
        scoring_args["pqk_projection_dir"] = scratch
        scoring_args["qpl_projection_dir"] = scratch
        with _suppress_unstorable_choice_warning():
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = _tuned_from_study(study, records, model, metric, validation, fixed=fixed)
    if reseed is not None:
        # The caller refits next; start it from the same global state as every trial.
        reseed()
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

    When it also carries trials (a search on a validation split, ``split_mode:
    manifest``), each ``results_<label>`` column gets a ``trials_<label>`` sibling
    holding :meth:`TunedParams.trial_log`, on the row that column is populated on --
    unless the frame already has that column, which a wrapper writing a per-head log
    (QPL) owns. Without trials the frame gains no column.
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
    log = best_params.trial_log() if isinstance(best_params, TunedParams) else None
    if log is not None:
        _attach_trial_log(frame, log)
    return frame


def _object_column(cells, index):
    """``cells`` as an object Series on ``index``: a dict cell must stay one cell."""
    import pandas as pd

    series = pd.Series([None] * len(cells), index=index, dtype=object)
    for position, cell in enumerate(cells):
        series.iat[position] = cell
    return series


def _attach_trial_log(frame, log):
    """Add ``trials_<label>`` next to every populated ``results_<label>`` column."""
    for column in [c for c in frame.columns if c.startswith("results_")]:
        target = TRIALS_PREFIX + column[len("results_"):]
        if target in frame.columns:
            continue
        cells = [log if isinstance(value, dict) else None for value in frame[column]]
        frame[target] = _object_column(cells, frame.index)
