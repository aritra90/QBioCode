"""Shared pieces of the ``compute_*_opt`` wrappers under ``split_mode: manifest``.

Not a model. Every tuned wrapper does the same three things when it is given a
:class:`~qbiocode.evaluation.protocol.ValidationSplit` (``validation``), and nothing
at all without one, so internal-mode runs stay unchanged:

* the configured defaults of the keys it does *not* search go into the ``fixed``
  params, so every trial and the refit use the arm's configured default config and
  not the estimator library's own defaults (:func:`fold_fixed`);
* libsvm fits are capped at :data:`FOLD_SVC_MAX_ITER` iterations unless the config
  sets a cap of its own (:func:`fold_svc_max_iter`). An uncapped poly-kernel ``SVC``
  fit once ran for 24 hours in a pilot job;
* the hidden ``RandomizedSearchCV`` inside a PQK or QPL head is scored with the run's
  tuning metric rather than accuracy (:func:`head_scorer`).
"""

import inspect
import logging

from qbiocode.learning._tuning import build_search_space, tuning_metric, tuning_scorer

logger = logging.getLogger(__name__)

#: The libsvm iteration cap of a fold-based run when the config sets none. libsvm's own
#: default is -1 (no cap); a trial that hits this cap reports ``fit_status_ = 1`` and
#: the tuner records it as FAIL (see :func:`qbiocode.learning._tuning.check_fit_status`).
FOLD_SVC_MAX_ITER = 10_000_000


def searched_names(model, candidates):
    """The hyperparameters a search over ``candidates`` actually varies.

    Worked out by :func:`~qbiocode.learning._tuning.build_search_space` itself, so it
    agrees with the search on what "not tuned" means (``None`` or an empty list). An
    empty search is left for the search itself to report, with its own message.
    """
    try:
        return set(build_search_space(model, candidates))
    except ValueError:
        return set()


def estimator_param_names(estimator_cls):
    """The constructor keywords ``estimator_cls`` accepts.

    The union of the ``__init__`` signature and ``get_params`` of a default instance:
    XGBoost's ``__init__`` takes ``**kwargs``, so its signature names almost nothing,
    while CatBoost's ``get_params`` reports only the params that were set.
    """
    try:
        parameters = inspect.signature(estimator_cls.__init__).parameters.values()
        names = {
            p.name for p in parameters
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
    except (TypeError, ValueError):  # pragma: no cover - C callables
        names = set()
    try:
        names |= set(estimator_cls().get_params(deep=False))
    except Exception:  # noqa: BLE001 -- the signature is all there is
        pass
    return names


#: What a ``compute_<m>_opt`` wrapper passes its base function itself: the data, the
#: config and the refit's label. Never a fixed param.
_WRAPPER_ARGS = frozenset(
    {"X_train", "X_test", "y_train", "y_test", "args", "model", "data_key", "verbose"}
)


def function_param_names(compute_fn):
    """The hyperparameter keywords a ``compute_<m>`` function accepts."""
    return set(inspect.signature(compute_fn).parameters) - _WRAPPER_ARGS


def fold_fixed(model, candidates, default_params, valid, fixed=None):
    """The ``fixed`` params of a fold-based search: unsearched defaults, then ``fixed``.

    Trial 0 enqueues the searched defaults only (see
    :func:`qbiocode.learning._tuning.run_study`), so without this a key the search does
    not vary would sit at the estimator library's default in every trial and in the
    refit -- ``LogisticRegression``'s ``lbfgs`` at 100 iterations rather than the
    ``saga`` at 10000 that ``compute_lr`` runs untuned. A default that is ``None`` means
    "the library's default" and is left out, as is any key ``valid`` does not name.

    Args:
        model (str): Model name, as the config spells it ('rf').
        candidates (dict): What the wrapper hands the search, name -> values.
        default_params (Mapping or None): The arm's default config from model_run.
        valid (set): Keywords the estimator (or compute function) accepts.
        fixed (Mapping or None): The wrapper's own fixed params; they win.

    Returns:
        dict: The params every trial and the refit are built with.
    """
    searched = searched_names(model, candidates)
    merged = {
        name: value
        for name, value in dict(default_params or {}).items()
        if name not in searched and name in valid and value is not None
    }
    merged.update(dict(fixed or {}))
    if merged:
        logger.info("tuning %r on a validation split with fixed params %r.", model, merged)
    return merged


def fold_svc_max_iter(configured):
    """The libsvm ``max_iter`` of a fold-based fit: the configured cap, else the default.

    Args:
        configured: ``max_iter`` as the config gives it, or None. A value below 1 is
            libsvm's "no cap" and is replaced by :data:`FOLD_SVC_MAX_ITER`.

    Returns:
        int: A positive iteration cap.
    """
    try:
        value = int(configured)
    except (TypeError, ValueError):
        return FOLD_SVC_MAX_ITER
    return value if value > 0 else FOLD_SVC_MAX_ITER


def head_scorer(head_scoring, args):
    """The ``scoring`` of a PQK/QPL head's ``RandomizedSearchCV``.

    Args:
        head_scoring (str or None): ``None`` keeps sklearn's default (the estimator's
            ``score``, accuracy), which is what every internal-mode run uses. A tuning
            metric name ('balanced_accuracy', ...) selects that metric's
            :class:`~qbiocode.learning._tuning.TuningScorer`, with ``f1_score``'s
            averaging read from ``args`` as usual.
        args (Mapping): The run's config.

    Returns:
        The ``scoring`` argument for ``RandomizedSearchCV``.
    """
    if head_scoring is None:
        return None
    # Validated through tuning_metric, so a typo fails as the config key would.
    tuning_metric({"tuning_metric": head_scoring})
    return tuning_scorer({**dict(args or {}), "tuning_metric": head_scoring})
