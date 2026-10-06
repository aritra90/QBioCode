"""Shared vocabulary of the fold-based evaluation protocol (``split_mode: manifest``).

Under ``split_mode: manifest`` QProfiler takes its outer splits from a frozen manifest
(:mod:`qbiocode.apps.qprofiler.split_manifest`): stratified K-fold, repeated, with a
validation set carved out of every training fold. Tuning then runs *inside* the fold --
each trial is fit on the fit rows and scored on the validation rows -- and the arm that
goes forward is the one with the best validation score, never the best test score.

This module is the contract between the layers that implement that. It is a leaf (numpy
only), so :mod:`qbiocode.learning._tuning`, :mod:`qbiocode.evaluation.model_run`, the
QProfiler app and the analysis utilities can all import it without a cycle.

- :class:`ValidationSplit` -- the fit/validation arrays a tuner scores its trials on.
- :class:`TrialRecord` -- one tuning trial, kept so every trial config (not only the
  winner) has its validation score and, optionally, its validation predictions.
- :data:`PROTOCOL_COLUMNS` -- the per-row provenance columns (split, seeds, host) a
  ModelResults row carries in manifest mode. Analysis code must treat them as
  identifiers, never as meta-features.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# model_run returns one key per model per artifact; the trial log of model ``m`` travels
# as ``TRIALS_PREFIX + m`` next to ``results_m`` and ``y_pred_m`` style keys.
TRIALS_PREFIX = "trials_"

# Split modes QProfiler accepts. ``internal`` is the historical repeated train_test_split
# (``iter`` x ``test_size``); ``manifest`` reads the frozen outer folds.
SPLIT_MODES = ("internal", "manifest")

# Validation schemes a manifest may declare. ``next_fold``: the validation rows of outer
# fold f are the test rows of fold (f + 1) mod k of the same repeat, so validation is
# carved from the training fold and every row is a validation row exactly once per repeat.
VALIDATION_SCHEMES = ("next_fold",)

# Provenance columns of a ModelResults row under split_mode=manifest. Names are the
# contract between the writer (qprofiler) and the readers (fair_selection,
# qc_winner_finder, meta_regression, collate/status). Internal mode writes none of them,
# so its outputs are unchanged. ``iteration`` is not listed: it predates this protocol
# and keeps its meaning, the 1-based global split id (repeat * k + fold + 1).
SPLIT_COLUMNS = (
    "split_mode",          # 'manifest'
    "repeat",              # 0-based repeat index
    "fold",                # 0-based outer fold index within the repeat
    "split_k",             # outer folds per repeat
    "split_repeats",       # repeats
    "split_validation",    # validation scheme ('next_fold')
    "split_seed",          # seed of the StratifiedKFold that produced this repeat
    "manifest_sha256",     # sha256 of the manifest file
    "dataset_sha256",      # sha256 of the dataset CSV bytes
    "split_generator",     # manifest generator_version
    "n_fit",               # rows the tuning trials were fit on
    "n_val",               # rows the tuning trials were scored on
    "n_test",              # outer test rows
)
SEED_COLUMNS = (
    "seed",                # model seed (args['seed'])
    "q_seed",              # quantum simulator seed
    "embed_seed",          # seed handed to the embedding (seed + iteration)
)
HOST_COLUMNS = (
    "host",                # socket.gethostname()
    "cpu_model",           # /proc/cpuinfo 'model name' (UMAP, tabpfn and vqc vary by it)
    "lsf_jobid",           # $LSB_JOBID, '' outside LSF
)
PROTOCOL_COLUMNS = SPLIT_COLUMNS + SEED_COLUMNS + HOST_COLUMNS

# Sidecar files written per pass (data_key) under the run's output directory.
OOF_DIR = "oof"                        # outer-test predictions of every model
TRIALS_DIR = "trials"                  # every tuning trial of every model
VAL_PREDICTIONS_DIR = "val_predictions"  # validation predictions of every trial

OOF_COLUMNS = ("data_key", "model", "row_id", "y_true", "y_pred", "y_score")
TRIAL_COLUMNS = (
    "data_key", "model", "trial", "params", "value", "state",
    "duration_s", "is_default", "is_best", "metric", "fixed",
)
VAL_PREDICTION_COLUMNS = ("data_key", "model", "trial", "row_id", "y_true", "y_pred", "y_score")

# Continuous validation scores of the trial a model was refit with, written on its
# ModelResults row in manifest mode. Balanced accuracy on a validation fold of ten-odd rows
# takes only a few values, so arms tie often; fair_selection breaks such ties on
# ``val_auc`` (any ranking score works, decision functions included). ``val_log_loss`` is
# finite only for arms whose scores are probabilities, so it cannot rank every arm.
TIEBREAK_COLUMNS = ("val_auc", "val_log_loss")


def _as_1d(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


@dataclass(frozen=True, eq=False)
class ValidationSplit:
    """The rows a tuner fits each trial on and the rows it scores the trial on.

    Built by QProfiler from one outer training fold: ``fit`` is the training fold minus
    the validation rows the manifest names. Features are transformed separately for this
    stage -- scaler and embedding fit on the fit rows only -- so a trial never sees a
    validation row during fitting, not even through the preprocessing.

    Attributes:
        X_fit: Features of the fit rows, shape (n_fit, p).
        X_val: Features of the validation rows, shape (n_val, p), same p.
        y_fit: Labels of the fit rows.
        y_val: Labels of the validation rows.
        fit_idx: Row ids (positions in the dataset CSV) of the fit rows, or None.
        val_idx: Row ids of the validation rows, or None.
    """

    X_fit: Any
    X_val: Any
    y_fit: Any
    y_val: Any
    fit_idx: Any = None
    val_idx: Any = None

    def __post_init__(self) -> None:
        n_fit, n_val = len(self.X_fit), len(self.X_val)
        if n_fit == 0 or n_val == 0:
            raise ValueError(f"empty validation split (n_fit={n_fit}, n_val={n_val})")
        if len(self.y_fit) != n_fit:
            raise ValueError(f"X_fit has {n_fit} rows but y_fit has {len(self.y_fit)}")
        if len(self.y_val) != n_val:
            raise ValueError(f"X_val has {n_val} rows but y_val has {len(self.y_val)}")
        p_fit, p_val = np.shape(self.X_fit)[1:], np.shape(self.X_val)[1:]
        if p_fit != p_val:
            raise ValueError(f"X_fit and X_val disagree on feature shape: {p_fit} vs {p_val}")
        for name, idx, n in (("fit_idx", self.fit_idx, n_fit), ("val_idx", self.val_idx, n_val)):
            if idx is not None and len(_as_1d(name, idx)) != n:
                raise ValueError(f"{name} has {len(idx)} ids for {n} rows")
        if self.fit_idx is not None and self.val_idx is not None:
            shared = np.intersect1d(np.asarray(self.fit_idx), np.asarray(self.val_idx))
            if shared.size:
                raise ValueError(f"{shared.size} row ids are both fit and validation rows")
        if len(np.unique(np.asarray(self.y_fit))) < 2:
            raise ValueError("the fit rows hold a single class; no classifier can be tuned on them")

    @property
    def n_fit(self) -> int:
        return len(self.X_fit)

    @property
    def n_val(self) -> int:
        return len(self.X_val)


@dataclass
class TrialRecord:
    """One tuning trial: the config tried, its validation score, and how it went.

    Attributes:
        number: Trial index within the study; trial 0 is the default config.
        params: The config, as plain JSON-serialisable values.
        value: Validation score under the tuning metric; NaN when the trial failed.
        state: 'COMPLETE', 'FAIL' or 'PRUNED' (Optuna's names).
        duration_s: Wall time of the trial in seconds.
        is_default: True for the trial that evaluated the arm's default config.
        y_pred: Validation-row predictions, aligned with ValidationSplit.val_idx, or None.
        y_score: Validation-row positive-class scores, or None.
    """

    number: int
    params: Mapping[str, Any]
    value: float
    state: str = "COMPLETE"
    duration_s: float = math.nan
    is_default: bool = False
    y_pred: Any = field(default=None, repr=False)
    y_score: Any = field(default=None, repr=False)

    def row(self) -> dict:
        """The scalar part, as one row of a TRIAL_COLUMNS table (params JSON-encoded)."""
        return {
            "trial": int(self.number),
            "params": json.dumps(_plain(self.params), sort_keys=True, default=str),
            "value": float(self.value) if self.value is not None else math.nan,
            "state": str(self.state),
            "duration_s": float(self.duration_s),
            "is_default": bool(self.is_default),
        }


def _plain(value: Any) -> Any:
    """numpy scalars/arrays and tuples to JSON-native values, recursively."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------------------
# The trial log: how one model's trials travel from the tuner to the results files.
#
# The tuner attaches TrialRecords to its TunedParams (``.trials``); modeleval /
# record_tuned_params turn them into ONE plain-dict cell, ``trials_<model>``, of the
# model's one-row frame; model_run passes it through like ``y_predicted_<model>``; and
# QProfiler writes it to the pass's sidecar files and keeps it in results.pkl. Plain
# dicts and numpy arrays only, so unpickling results.pkl never needs these classes.
# ---------------------------------------------------------------------------------------


def trial_log(
    trials: Sequence[TrialRecord],
    *,
    metric: str | None,
    best: int | None,
    val_idx: Any = None,
    y_val: Any = None,
    fixed: Mapping[str, Any] | None = None,
) -> dict:
    """The ``trials_<model>`` cell for one model.

    Args:
        trials: Every trial the study ran, in trial order.
        metric: The tuning metric the values are in.
        best: ``number`` of the trial whose params were refit, or None.
        val_idx: Row ids of the validation rows the predictions are aligned with.
        y_val: Their labels.
        fixed: The non-searched params every trial was fitted with, or None. A trial's
            ``params`` holds the searched names only; this completes its config.

    Returns:
        ``{"metric", "best", "val_idx", "y_val", "trials": [...]}``, plus ``"fixed"``
        when ``fixed`` is given; each trial is its :meth:`TrialRecord.row` fields with
        ``params`` left as a dict, plus ``is_best``, ``y_pred`` and ``y_score``.
    """
    rows = []
    for record in trials:
        entry = record.row()
        entry["params"] = _plain(dict(record.params))
        entry["is_best"] = best is not None and int(record.number) == int(best)
        entry["y_pred"] = None if record.y_pred is None else np.asarray(record.y_pred)
        entry["y_score"] = None if record.y_score is None else np.asarray(record.y_score)
        rows.append(entry)
    log = {
        "metric": metric,
        "best": None if best is None else int(best),
        "val_idx": None if val_idx is None else np.asarray(val_idx),
        "y_val": None if y_val is None else np.asarray(y_val),
        "trials": rows,
    }
    if fixed is not None:
        log["fixed"] = _plain(dict(fixed))
    return log


def _cell(value: Any) -> Any:
    """model_run hands columns back as ``{0: value}``; accept either form."""
    if isinstance(value, Mapping) and set(value) == {0}:
        return value[0]
    return value


def binary_validation_scores(y_true: Any, y_score: Any) -> dict:
    """``val_auc`` and ``val_log_loss`` of one score vector on its validation rows.

    Args:
        y_true: Validation labels, exactly two classes; the larger one is positive.
        y_score: Positive-class scores aligned with ``y_true``: probabilities, or any
            ranking score such as a ``decision_function``.

    Returns:
        ``{"val_auc", "val_log_loss"}``. The AUC is the Mann-Whitney statistic with tied
        scores at their average rank, so it needs only the ranking. The log loss is
        finite only when every score lies in [0, 1], which a probability always does
        and a decision function only by chance (qsvc does on ~1% of folds). Compare it
        only among models that output probabilities. Both are NaN without two classes
        or with a non-finite score.
    """
    out = {"val_auc": math.nan, "val_log_loss": math.nan}
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    classes = np.unique(y)
    if len(classes) != 2 or s.shape != y.shape or not np.isfinite(s).all():
        return out
    pos = y == classes[1]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = ((end - counts + 1 + end) / 2.0)[inverse]          # 1-based, ties averaged
    out["val_auc"] = float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    if ((s >= 0.0) & (s <= 1.0)).all():
        p = np.clip(s, 1e-15, 1.0 - 1e-15)
        out["val_log_loss"] = float(-np.mean(np.where(pos, np.log(p), np.log1p(-p))))
    return out


def validation_tiebreak(log: Any) -> dict:
    """:data:`TIEBREAK_COLUMNS` for one model, from its ``trials_<model>`` cell.

    Scores the trial the model was refit with (``log["best"]``) on the validation rows
    it was tuned against. NaN when the cell has no chosen trial or no validation scores
    (an untuned model, or a model that exposes no ranking score).
    """
    log = _cell(log)
    nan = {name: math.nan for name in TIEBREAK_COLUMNS}
    if not log or log.get("best") is None or log.get("y_val") is None:
        return nan
    best = int(log["best"])
    entry = next((e for e in log.get("trials", []) if int(e["trial"]) == best), None)
    if entry is None or entry.get("y_score") is None:
        return nan
    return binary_validation_scores(log["y_val"], entry["y_score"])


def tiebreak_from_sidecars(trials, val_predictions):
    """:data:`TIEBREAK_COLUMNS` per ``(data_key, model)`` from a pass's sidecar files.

    For runs written before ModelResults carried these columns: the ``is_best`` trial of
    each model in ``trials`` (TRIAL_COLUMNS) is looked up in ``val_predictions``
    (VAL_PREDICTION_COLUMNS) and scored with :func:`binary_validation_scores`.

    Returns:
        A DataFrame with ``data_key``, ``model`` and the TIEBREAK_COLUMNS.
    """
    import pandas as pd

    cols = ["data_key", "model", *TIEBREAK_COLUMNS]
    best = trials[trials["is_best"].astype(str).str.lower().isin(("true", "1"))]
    if best.empty or val_predictions.empty:
        return pd.DataFrame(columns=cols)
    chosen = val_predictions.merge(best[["data_key", "model", "trial"]],
                                   on=["data_key", "model", "trial"])
    rows = []
    for (key, model), g in chosen.groupby(["data_key", "model"], sort=True):
        rows.append({"data_key": key, "model": model,
                     **binary_validation_scores(g["y_true"], g["y_score"])})
    return pd.DataFrame(rows, columns=cols)


def _column_or_nan(values: Any, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan)
    array = np.asarray(values)
    if array.ndim != 1 or array.size != n:
        raise ValueError(f"expected {n} per-row values, got shape {array.shape}")
    return array


def oof_frame(summary: Mapping[str, Any], test_idx: Any, data_key: str):
    """Outer-test predictions of every model in one pass, one row per (model, test row).

    Args:
        summary: The pass's model_run output (``y_test_<m>``, ``y_predicted_<m>``,
            ``y_score_<m>`` keys; ``{0: value}`` or bare values).
        test_idx: Row ids of the test rows, in the order X_test was built.
        data_key: The pass's data_key.

    Returns:
        A DataFrame with OOF_COLUMNS.
    """
    import pandas as pd

    test_idx = np.asarray(test_idx)
    frames = []
    for key in sorted(summary):
        if not key.startswith("y_predicted_"):
            continue
        model = key[len("y_predicted_"):]
        y_pred = _column_or_nan(_cell(summary[key]), test_idx.size)
        y_true = _column_or_nan(_cell(summary.get(f"y_test_{model}")), test_idx.size)
        y_score = _column_or_nan(_cell(summary.get(f"y_score_{model}")), test_idx.size)
        frames.append(pd.DataFrame({
            "data_key": data_key, "model": model, "row_id": test_idx,
            "y_true": y_true, "y_pred": y_pred, "y_score": y_score,
        }))
    if not frames:
        return pd.DataFrame(columns=list(OOF_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(OOF_COLUMNS)]


def trial_frames(summary: Mapping[str, Any], data_key: str):
    """Every tuning trial of every model in one pass, and their validation predictions.

    Args:
        summary: The pass's model_run output (``trials_<m>`` keys holding trial_log dicts).
        data_key: The pass's data_key.

    Returns:
        (trials, val_predictions): DataFrames with TRIAL_COLUMNS and
        VAL_PREDICTION_COLUMNS. ``params`` (the searched params) and ``fixed`` (the
        non-searched params every trial of the model was fitted with, empty when the log
        has none) are JSON with sorted keys; together they are the trial's configuration.
    """
    import pandas as pd

    trial_rows, prediction_frames = [], []
    for key in sorted(summary):
        if not key.startswith(TRIALS_PREFIX):
            continue
        model = key[len(TRIALS_PREFIX):]
        log = _cell(summary[key])
        if not log:
            continue
        val_idx = log.get("val_idx")
        y_val = log.get("y_val")
        # The same for every trial of the model; repeated so a row alone rebuilds the fit.
        fixed = log.get("fixed")
        fixed = None if fixed is None else json.dumps(_plain(fixed), sort_keys=True,
                                                       default=str)
        for entry in log.get("trials", []):
            trial_rows.append({
                "data_key": data_key, "model": model, "trial": entry["trial"],
                "params": json.dumps(_plain(entry["params"]), sort_keys=True, default=str),
                "value": entry["value"], "state": entry["state"],
                "duration_s": entry["duration_s"], "is_default": entry["is_default"],
                "is_best": entry["is_best"], "metric": log.get("metric"),
                "fixed": fixed,
            })
            if entry.get("y_pred") is None or val_idx is None:
                continue
            n_val = len(val_idx)
            prediction_frames.append(pd.DataFrame({
                "data_key": data_key, "model": model, "trial": entry["trial"],
                "row_id": np.asarray(val_idx),
                "y_true": _column_or_nan(y_val, n_val),
                "y_pred": _column_or_nan(entry["y_pred"], n_val),
                "y_score": _column_or_nan(entry.get("y_score"), n_val),
            }))
    trials = pd.DataFrame(trial_rows, columns=list(TRIAL_COLUMNS))
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)[list(VAL_PREDICTION_COLUMNS)]
    else:
        predictions = pd.DataFrame(columns=list(VAL_PREDICTION_COLUMNS))
    return trials, predictions
