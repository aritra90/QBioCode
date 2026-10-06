"""Freeze a quantum model's tuned hyperparameters after the first resample.

The fairness problem
--------------------
With ``grid_search: True`` and ``tune_quantum: False`` -- the configuration the sweep was
sized for -- every classical learner gets an Optuna search and no quantum learner gets
one. The benchmark then measures "tuned classical vs default quantum" while reporting it
as "classical vs quantum", and every quantum loss is confounded with the fact that nobody
searched its space.

Turning ``tune_quantum: True`` on fixes the fairness and costs too much: each trial is a
full quantum fit, so the quantum side is multiplied by ``n_trials_quantum`` **per
resample**. At 5 resamples and 32 trials that is 160 quantum fits per (dataset,
embedding, model) where the untuned run does 5.

This module implements the middle option: **search once on the first resample, reuse
that configuration for the rest.** Cost falls from ``iter * n_trials`` to
``n_trials + (iter - 1)`` per arm -- at iter=5, n_trials_quantum=32, from 160 fits to 36
-- while every arm still runs at a searched configuration rather than a default one.

What it costs scientifically
----------------------------
Reuse is **optimistic** for the frozen arms, not conservative. An earlier version of this
docstring called it a handicap that made a quantum win a lower bound; that is wrong, and
the reason is how the resamples are drawn. They are repeated random 80/20 holdouts of one
dataset, so they overlap heavily: in the pilot, 73-87% of a later iteration's TEST rows
were TRAINING rows of iteration 0 -- the rows the frozen configuration was searched on,
and partly chosen by fitting. Scoring that configuration on iteration k's test split
therefore scores it largely on data that helped select it. A classical arm is re-searched
every iteration on that iteration's training rows only, so its test rows never
influenced its own selection. The asymmetry leaks test information toward the frozen
(quantum) side.

The effect measured on the pilot is small -- a difference-in-differences of about
-0.02 balanced accuracy, not significant -- but it points the wrong way for a claim of
quantum advantage, so a frozen-arm result must be read as an upper-leaning estimate and
never described as a symmetric protocol or as a lower bound. Confirmatory runs should set
``freeze_quantum_params: False`` (per-iteration tuning, the symmetric protocol) or use
disjoint-fold splits, where no test row of one resample was a training row of another.
:func:`load_frozen_params` logs a warning saying so, once per process, the first time
frozen parameters are reused.

The freeze also *reduces* the quantum side's variance across resamples, since the
configuration no longer moves. Nested selection in :mod:`qbiocode.utils.fair_selection`
does not compare per-arm variances, so this does not reintroduce the fragmentation bias
that a shared post-hoc argmax suffers from -- but it is a further reason not to compare
the two sides' raw score spreads directly.

Why the cache key omits the iteration
-------------------------------------
Reuse across resamples is the entire point, so ``iteration`` cannot be part of the key.
Everything else that changes what a good configuration *is* must be: the dataset, the
embedding, the component count, and the model. The projection caches in
``compute_pqk``/``compute_qpl`` get this wrong in the opposite direction -- they include
``data_key`` whole, iteration and all, which is correct for them because a projection is
split-specific. A hyperparameter choice is not.

The key is derived from ``data_key`` by dropping its trailing iteration token rather than
by re-deriving the parts, so that a change to how ``qprofiler`` builds ``data_key``
cannot silently desynchronise the two.

Failure policy
--------------
Every operation degrades to "search again". A corrupt file, an unreadable directory, a
concurrent writer, a parameter set that no longer matches the search space: all return
``None`` and cost one redundant search. Raising instead would abort a sweep hours in over
a cache, and silently *using* a stale entry would put an unsearched configuration into a
fairness-critical comparison.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Config key naming the directory that holds the frozen parameter files.
CACHE_DIR_KEY = "quantum_param_dir"
#: Default directory, alongside the projection caches' own defaults.
DEFAULT_CACHE_DIR = "quantum_tuned_params"
#: Config key that switches freezing on. Off when the key is absent: with it off,
#: behaviour is exactly what it was before this module existed. Note that the shipped
#: ``apps/qprofiler/configs/config.yaml`` and every pilot10 config set it to True.
FREEZE_KEY = "freeze_quantum_params"

#: Whether the "frozen results are optimistic" warning has been logged in this process.
_REUSE_WARNED = False


def _warn_reuse_once(model: str, path: str) -> None:
    """Log, once per process, that frozen parameters make results optimistic."""
    global _REUSE_WARNED
    if _REUSE_WARNED:
        return
    _REUSE_WARNED = True
    logger.warning(
        "Reusing frozen tuned parameters for %r from %s (freeze_quantum_params: True). "
        "Results for frozen models are OPTIMISTIC: the parameters were tuned on the first "
        "resample's training rows, most of which reappear in later resamples' test "
        "splits, while re-tuned models never see their own test rows. For confirmatory "
        "runs set freeze_quantum_params: False (per-iteration tuning) or use "
        "disjoint-fold splits. This warning is logged once per process.",
        model, path,
    )


def freeze_key(data_key: str) -> str:
    """``data_key`` with its trailing iteration token removed.

    ``qprofiler`` builds ``data_key`` as
    ``'_'.join([dataset_stem, embed, str(n_components), str(iter)])``, so dropping the
    last underscore-separated token yields a key that is stable across resamples and
    still distinguishes dataset, embedding and component count. A ``data_key`` with no
    separator is returned unchanged rather than emptied.
    """
    stem, sep, _last = str(data_key).rpartition("_")
    return stem if sep else str(data_key)


def backend_signature(args: Mapping[str, Any]) -> str:
    """Where a frozen parameter set was searched, as one comparable string.

    Part of the payload rather than of the filename. A frozen set is only reusable on
    the target it was searched on: ``entanglement: 'full'`` at 20 qubits is the best
    candidate on an exact statevector and a bond-dimension blow-up under MPS, and
    ``maxiter``'s optimum moves with the primitive's noise. Before this, the key was
    ``(dataset, embedding, n_components, model)`` only, so pointing two runs with
    different backends at one ``quantum_param_dir`` made the second silently reuse the
    first's search -- and comparing ``mps_simulator`` against
    ``statevector_simulator`` on one dataset is precisely the validation this design
    invites, so the collision was reachable by doing the right thing.

    Kept out of the filename so an existing cache directory stays readable: a file
    written before this key existed simply reports no signature and is retuned once.

    Args:
        args (Mapping): the config dict. Reads ``backend`` and ``sim_method`` only.

    Returns:
        str: e.g. ``'simulator_aer+matrix_product_state'``.
    """
    backend = str(args.get("backend", "?"))
    method = args.get("sim_method")
    return f"{backend}+{method}" if method else backend


def _cache_path(args: Mapping[str, Any], data_key: str, model: str) -> str:
    directory = os.path.expanduser(str(args.get(CACHE_DIR_KEY, DEFAULT_CACHE_DIR)))
    return os.path.join(directory, f"{freeze_key(data_key)}__{model}.json")


def freezing_enabled(args: Mapping[str, Any]) -> bool:
    """True when the config asked for tuned quantum parameters to be reused."""
    return bool(args.get(FREEZE_KEY, False))


#: What a frozen payload without a ``tuning_metric`` field was tuned on. Such files
#: predate the key, when every tuner selected on plain accuracy.
LEGACY_TUNING_METRIC = "accuracy"


def load_frozen_params(
    args: Mapping[str, Any],
    data_key: str,
    model: str,
    space: Mapping[str, Any] | None = None,
    *,
    metric: str | None = None,
) -> dict | None:
    """The parameters frozen for this ``(dataset, embedding, n_components, model)``.

    Returns ``None`` -- meaning "search this resample" -- when freezing is off, nothing
    is cached yet, the file is unreadable or malformed, or the cached names no longer
    match ``space``. That last check matters: editing ``gridsearch_<model>_args`` between
    runs would otherwise keep feeding the old configuration into a search space the user
    has since changed, and the run would look tuned without being tuned to the current
    space. For the same reason a file tuned on a ``tuning_metric`` other than
    ``metric`` is discarded. A file written before that field existed was tuned on
    accuracy (see :data:`LEGACY_TUNING_METRIC`), so it is read as such: still used by an
    ``accuracy`` run, retuned by any other.

    A hit is returned as a :class:`~qbiocode.learning._tuning.TunedParams` with
    ``reused=True`` and the ``score``/``metric`` recorded when the file was written
    (NaN and ``'accuracy'`` for an older file). They are attributes, never keys, so they cannot reach
    the estimator. The first hit in a process logs that frozen results are optimistic;
    see "What it costs scientifically" above.

    Args:
        args (Mapping): the config dict.
        data_key (str): the per-pass key, iteration token included.
        model (str): the model label, e.g. ``'qsvc'``.
        space (Mapping): optional current search space, for the staleness check.
        metric (str or None): the run's ``tuning_metric``, for the same check. ``None``
            skips it.

    Returns:
        TunedParams | None: the frozen parameters, or ``None`` to search.
    """
    if not freezing_enabled(args):
        return None
    path = _cache_path(args, data_key, model)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        # A truncated file is the expected shape of a crash mid-write. One redundant
        # search is a far better outcome than aborting a sweep over a cache artefact.
        logger.warning(
            "Could not read frozen quantum parameters from %s (%s: %s); tuning this "
            "resample instead.", path, type(error).__name__, error,
        )
        return None

    params = payload.get("params") if isinstance(payload, dict) else None
    if not isinstance(params, dict) or not params:
        logger.warning("Frozen parameter file %s holds no 'params' mapping; retuning.", path)
        return None

    # Retune when the cached set was searched on a different execution target. A file
    # from before this field existed carries None and is treated as a mismatch, which
    # costs one search per (dataset, model) once and then self-heals.
    cached_signature = payload.get("backend_signature")
    current_signature = backend_signature(args)
    if cached_signature != current_signature:
        logger.warning(
            "Frozen parameters for %r in %s were searched on %s but this run targets "
            "%s. A configuration is only tuned for the backend it was searched on, so "
            "they are discarded and this resample is retuned.",
            model, path, cached_signature or "an unrecorded backend", current_signature,
        )
        return None

    if space is not None and set(params) != set(space):
        logger.warning(
            "Frozen parameters for %r in %s name %s but the current "
            "gridsearch_%s_args names %s. The search space changed since these were "
            "written, so they are discarded and this resample is retuned.",
            model, path, sorted(params), model, sorted(space),
        )
        return None

    # Every payload written before ``tuning_metric`` existed was searched on plain
    # accuracy (the only objective the tuners had), so a missing field means that and
    # not "unknown": reusing such a file under balanced accuracy would silently report
    # accuracy-tuned parameters as balanced-accuracy-tuned ones.
    cached_metric = payload.get("tuning_metric") or LEGACY_TUNING_METRIC
    if metric is not None and cached_metric != metric:
        logger.warning(
            "Frozen parameters for %r in %s were selected on %s but this run tunes on "
            "%s, so they are discarded and this resample is retuned.",
            model, path, cached_metric, metric,
        )
        return None

    from qbiocode.learning._tuning import TunedParams

    score = payload.get("tuning_score")
    try:
        score = float("nan") if score is None else float(score)
    except (TypeError, ValueError):
        score = float("nan")
    _warn_reuse_once(model, path)
    return TunedParams(params, metric=cached_metric, score=score, reused=True)


def save_frozen_params(
    args: Mapping[str, Any],
    data_key: str,
    model: str,
    params: Mapping[str, Any],
    *,
    metric: str | None = None,
    score: float | None = None,
) -> str | None:
    """Persist ``params`` as the frozen configuration for later resamples.

    Written atomically, via a temporary file in the destination directory followed by
    ``os.replace``. ``model_run`` parallelises over models with joblib, so two workers
    can reach this for different models at once and a reader can arrive mid-write; a
    partial JSON file would be read back as a cache miss, which is survivable, but
    ``os.replace`` removes the window entirely at no cost.

    ``metric`` and ``score`` -- the tuning metric and the chosen configuration's
    validation score -- are stored beside ``params`` as optional fields, so a later
    resample can report them and a file without them still loads. A file written
    without ``metric`` is read back as tuned on :data:`LEGACY_TUNING_METRIC`, so callers
    that tuned on anything else must pass it.

    Returns the path written, or ``None`` if freezing is off, ``params`` is empty, or the
    write failed. A failed write is logged and ignored: the run continues and later
    resamples simply search again.
    """
    if not freezing_enabled(args) or not params:
        return None
    path = _cache_path(args, data_key, model)
    payload = {
        "model": model,
        "freeze_key": freeze_key(data_key),
        "data_key_written_from": str(data_key),
        "backend_signature": backend_signature(args),
        "params": dict(params),
    }
    if metric is not None:
        payload["tuning_metric"] = str(metric)
    if score is not None:
        # json writes NaN/-inf as bare tokens that its own reader accepts; kept as a
        # float so the round trip is exact.
        payload["tuning_score"] = float(score)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False,
            dir=os.path.dirname(path) or ".", prefix=".frozen_", suffix=".tmp",
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            os.replace(handle.name, path)
        except BaseException:
            # Do not leave the scratch file behind on any failure path, including
            # KeyboardInterrupt -- a sweep is long enough that it is a realistic exit.
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError) as error:
        logger.warning(
            "Could not persist frozen quantum parameters for %r to %s (%s: %s); later "
            "resamples will tune again.", model, path, type(error).__name__, error,
        )
        return None
    return path


__all__ = [
    "CACHE_DIR_KEY",
    "DEFAULT_CACHE_DIR",
    "FREEZE_KEY",
    "freeze_key",
    "freezing_enabled",
    "LEGACY_TUNING_METRIC",
    "load_frozen_params",
    "save_frozen_params",
]
