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

"""Turning a partly-filled config block into a grid ``GridSearchCV`` accepts.

Every ``compute_*_opt`` function takes one keyword per tunable hyperparameter and
used to hand all of them to ``GridSearchCV`` unconditionally. Each defaulted to
``[]``, so any config that did not enumerate *every* hyperparameter of the chosen
model died inside sklearn::

    ValueError: Parameter grid for parameter 'colsample_bytree' need to be a
    non-empty sequence, got: []

The message names a parameter the user never mentioned and says nothing about the
config, which is the opposite of the useful direction. It also made a deliberately
small grid impossible to express: trimming a demo config down to two parameters
was indistinguishable from corrupting it.

A hyperparameter nobody asked to tune should simply be left at the estimator's own
default, which is what dropping it from the grid does.
"""

import warnings
from collections.abc import Mapping, Sequence


def to_plain(value):
    """Strip config-library wrappers out of a hyperparameter value, recursively.

    OmegaConf hands a YAML list to these builders as a ``ListConfig`` and a mapping as a
    ``DictConfig``. Both behave like the ``Sequence`` and ``Mapping`` the builders test
    for, and a *scalar* leaf reads back as a plain ``int`` or ``str``, which is why this
    went unnoticed for so long. What does not survive is a *container* leaf.
    ``gridsearch_mlp_args`` writes ``hidden_layer_sizes: [[20], [50], [100]]``, and
    ``list(values)`` unwraps only the outer ``ListConfig`` -- each choice is still a
    ``ListConfig``. Optuna eventually puts that choice through ``json.dumps`` and raises
    ``TypeError: Object of type ListConfig is not JSON serializable`` inside a joblib
    worker, which takes all 13 models of the pass down with it. All 12 pilot jobs died
    exactly this way, having fit nothing.

    Two things about the trigger are worth recording, because both are reasons a test
    can be written and still miss it. It is version-dependent: the ``json.dumps`` is
    Optuna's constant-liar bookkeeping, and ``TPESampler`` only began defaulting
    ``constant_liar=True`` in Optuna 5, so the same configs ran on Optuna 4. And it is
    late: the sampler draws its first ``n_startup_trials`` (10) independently at random
    and only then consults the relative path, so the failure lands on trial 10 of a
    50-trial search rather than trial 0.

    Done structurally rather than through ``OmegaConf.to_object`` so the learning layer
    stays independent of the config library -- a direct caller passing plain dicts and
    lists is the ordinary case, and the defect is "a ``Sequence`` that is not a ``list``",
    which is not unique to OmegaConf. Nothing is lost by not using OmegaConf's own
    converter: iterating a ``ListConfig`` resolves ``${...}`` interpolations on element
    access, so the values seen here are already resolved.

    Args:
        value: Whatever the config supplied, at any nesting depth.

    Returns:
        The same value with every ``Mapping`` rebuilt as a ``dict`` and every non-string
        ``Sequence`` or set rebuilt as a ``list``. Scalars, ``None`` and objects that are
        neither (a numpy array, say) are returned unchanged.
    """
    if isinstance(value, Mapping):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (Sequence, set, frozenset)) and not isinstance(value, (str, bytes)):
        return [to_plain(item) for item in value]
    return value


def build_param_grid(model, candidates):
    """Build a ``GridSearchCV`` ``param_grid`` from the values actually supplied.

    Args:
        model (str): Model name, used only to make the error message specific.
        candidates (dict): Maps hyperparameter name to the values to search. A
            value of ``None`` or an empty sequence means "not tuned" and is
            dropped, leaving the estimator's own default in force. A bare scalar
            or string is wrapped into a one-element list.

    Returns:
        dict: Only the entries worth searching, each a non-empty list.

    Raises:
        ValueError: If nothing at all was supplied. Grid search over an empty grid
            is a config mistake, not a one-point search, so it is worth saying so
            here rather than letting sklearn report it against an arbitrary
            parameter name.
    """
    grid = {}
    for name, values in candidates.items():
        values = to_plain(values)
        if values is None:
            continue
        # A `{low, high}` range is meaningful to the Optuna tuner but not to a grid,
        # which can only enumerate. A dict is not a Sequence, so it used to be wrapped
        # into a one-element list and handed to the estimator as a *value*, surfacing
        # as `InvalidParameterError: The 'C' parameter of SVC must be a float ... Got
        # {'low': 0.001, ...}` -- an error about the estimator, naming neither the
        # config entry nor the tuner that would accept it.
        if isinstance(values, Mapping):
            raise ValueError(
                f"{model!r} hyperparameter {name!r} is written as a range "
                f"({dict(values)!r}), which only the Optuna tuner can sample. Either "
                f"set tuner: optuna, or write {name!r} as a list of values for the "
                f"grid to enumerate."
            )
        # A string is a sequence, so `max_features: sqrt` would otherwise be
        # searched as ['s', 'q', 'r', 't'] -- four invalid values, no error, and a
        # best_params_ that means nothing.
        if isinstance(values, str) or not isinstance(values, (Sequence, set, frozenset)):
            values = [values]
        values = list(values)
        if not values:
            continue
        grid[name] = values

    if not grid:
        raise ValueError(
            f"Grid search was requested for {model!r} but no hyperparameter values "
            f"were given, so there is nothing to search. Either add a "
            f"'gridsearch_{model}_args' block to the config naming at least one "
            f"hyperparameter and the values to try, or set grid_search: False to "
            f"run {model!r} at its default hyperparameters. "
            f"Recognised hyperparameters for this model: "
            f"{', '.join(sorted(candidates))}."
        )
    return grid


def warn_ignored_hyperparameter(model, name, reason):
    """Flag a hyperparameter the estimator will accept and then disregard.

    XGBoost's sklearn wrapper takes unknown keyword arguments without complaint,
    so a grid entry it does not implement is not an error -- it just multiplies the
    number of fits while every duplicate returns the same model.
    """
    warnings.warn(
        f"{model!r} was given a grid for {name!r}, which {reason} Every value will "
        f"be searched and will produce the same model, multiplying the run time for "
        f"nothing. Remove {name!r} from the grid.",
        UserWarning,
        stacklevel=3,
    )


def one_value(name, value, why, block):
    """Resolve a parameter that ``_opt`` passes to every trial rather than searching.

    Every *searched* key in a ``gridsearch_*`` block is written as a list, so a
    one-element list is the natural way to spell a constant there, and it is taken
    as that constant. The parameters routed through here are not searched, so
    without this unwrapping the list reaches the estimator verbatim -- and a list
    where the estimator wants a scalar does not fail in the estimator. It fails
    later, in sklearn's ``clone`` during cross-validation, inside a joblib worker::

        RuntimeError: Cannot clone object CatBoostClassifier(..., thread_count=[1],
        ...), as the constructor either does not set or modifies parameter
        thread_count

    That names the parameter but not the block it came from, surfaces three layers
    from the config key that caused it, and breaks the *tuned* path -- which
    ``grid_search: True`` makes the only path that runs, since it replaces the
    result label rather than adding to it. Every pilot config paired
    ``'thread_count': [1]`` with ``grid_search: True``, so catboost fit nothing.

    Several values are the opposite mistake: the author believed the parameter was
    searched when it is not, and quietly keeping the first would hide that. So that
    is refused, naming the block, the key and ``why``. An empty list is the shipped
    default for a key the config never set (see :func:`build_param_grid`) and means
    the same as ``None``.

    ``block`` is required rather than defaulted: the whole value of the message is
    that it names the config block to edit, and a default would silently name the
    wrong one for every caller that forgot it.

    Args:
        name (str): The config key, as written in ``block``.
        value: Whatever the config supplied -- a scalar, a one-element list, an
            empty list, or None.
        why (str): Why this parameter is not searched, in a clause that reads after
            "but it is not searched: ".
        block (str): The config block ``value`` came from, e.g.
            ``'gridsearch_xgb_args'``.

    Returns:
        The single value, or None if the config did not set one.

    Raises:
        ValueError: If ``value`` holds more than one value.
    """
    value = to_plain(value)
    if value is None:
        return None
    if isinstance(value, (Sequence, set, frozenset, Mapping)) and not isinstance(value, str):
        values = list(value)
        if not values:
            return None
        if len(values) > 1:
            raise ValueError(
                f"{block!r} gives {name!r} several values ({value!r}), but it is not "
                f"searched: {why}. Give it a single value instead."
            )
        return values[0]
    return value
