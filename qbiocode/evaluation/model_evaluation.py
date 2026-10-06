# ====== Base class imports ======

import time
from typing import Literal

import numpy as np
import pandas as pd

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from qbiocode.evaluation.protocol import TRIALS_PREFIX
from qbiocode.utils.helper_fn import print_results


def _positive_class_column(scores, estimator):
    """Which column of a two-column score matrix holds the positive class.

    ``roc_auc_score`` treats the *larger* of the two labels in ``y_true`` as positive,
    so the score vector handed to it has to be the column for that same label. Both
    scikit-learn and qiskit-machine-learning order their output columns by sorted
    class label, which makes ``scores[:, 1]`` right almost always -- and silently
    inverted (AUC ``1 - a``) on the estimator that does not, which is exactly the kind
    of defect a plausible-looking number in [0, 1] hides. So the column is looked up
    through ``classes_`` rather than hardcoded.

    Args:
        scores (numpy.ndarray): A ``(n_samples, 2)`` score or probability matrix.
        estimator: The fitted estimator that produced ``scores``.

    Returns:
        int or None: The index of the positive-class column, or ``None`` if
        ``classes_`` is present but disagrees with the width of ``scores`` -- a
        mismatch means the mapping cannot be established and no guess should be made.
        Estimators with no ``classes_`` at all (``PegasosQSVC``,
        ``NeuralNetworkClassifier``) fall back to the last column, which is the
        sorted-label convention both of them follow.
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return scores.shape[1] - 1
    classes = np.asarray(classes)
    if classes.shape[0] != scores.shape[1]:
        return None
    # The largest label. ndarray.max() has no ufunc loop for a '<U' string array
    # (UFuncTypeError), while np.argmax compares any sortable dtype, strings included.
    return int(np.argmax(classes))


def _score_vector(raw, estimator):
    """Reduce whatever a scoring method returned to one ranking value per sample.

    Args:
        raw (numpy.ndarray): The output of ``predict_proba`` or ``decision_function``.
        estimator: The fitted estimator that produced ``raw``.

    Returns:
        numpy.ndarray or None: A one-dimensional score array, or ``None`` when ``raw``
        carries no single ranking -- three or more classes, or a two-column matrix
        whose columns cannot be mapped to labels.
    """
    if raw.ndim == 1:
        # ``OneVsOneClassifier.decision_function`` and ``SVC.decision_function`` both
        # collapse to this shape on a binary problem, already oriented so that larger
        # means the positive class.
        return raw
    if raw.ndim != 2:
        return None
    if raw.shape[1] == 1:
        # ``EstimatorQNN.forward`` maps to a single value in [-1, +1] and
        # ``NeuralNetworkClassifier`` classifies by its sign, so it is a ranking even
        # though it is not a probability.
        return raw.ravel()
    if raw.shape[1] != 2:
        # Three or more classes. A multiclass AUC needs the whole matrix and an
        # explicit averaging choice, which is a different metric from the binary one
        # this column has always reported; see ``evaluation_metrics`` for that.
        return None
    column = _positive_class_column(raw, estimator)
    if column is None:
        return None
    return raw[:, column]


def extract_binary_scores(estimator, X):
    """Best continuous score a fitted estimator can give for a binary problem.

    ``roc_auc_score`` needs a *ranking* -- probabilities or decision values -- not
    predicted labels. Which method supplies one differs across the learners in this
    package, and the differences are not incidental:

    * The seven older classical learners (``dt``, ``lr``, ``mlp``, ``nb``, ``rf``,
      ``svc``, ``xgb``) are fitted inside a :class:`~sklearn.multiclass.OneVsOneClassifier`.
      That wrapper has **no** ``predict_proba``, which is why the AUC here could not
      simply be read off probabilities -- but it does have ``decision_function``. On a
      binary target the wrapper holds exactly one pairwise estimator, and its
      ``decision_function`` is that estimator's own ranking, returned as a
      ``(n_samples,)`` vector oriented towards the positive class. That is a genuine
      AUC input. ``compute_svc`` asks ``SVC`` for ``probability=True``; the wrapper
      does not expose the result, which costs nothing now -- ``decision_function``
      makes Platt scaling unnecessary here rather than the passing of it broken.
    * A fully grown ``DecisionTreeClassifier`` produces only two distinct
      ``decision_function`` values, so its AUC from scores equals its AUC from labels.
      That is honest: a tree of pure leaves has no ranking to offer. It is not a
      reason to fall back to labels for everything else.
    * ``catboost`` and ``tabpfn``, every ``_opt`` twin, and the searched heads inside
      ``compute_pqk``/``compute_qpl`` are fitted unwrapped, so ``predict_proba`` is
      available and preferred.
    * On the quantum side ``QSVC`` inherits ``SVC.decision_function``; ``PegasosQSVC``,
      ``VQC`` and ``NeuralNetworkClassifier`` all publish ``predict_proba``.

    ``predict_proba`` is tried first and ``decision_function`` second. The two are
    monotonically related wherever both exist, so the order does not change an AUC --
    it only prefers the more interpretable of the two.

    Args:
        estimator: A fitted classifier.
        X (numpy.ndarray): The samples to score, in the same space the estimator was
            fitted in (for ``compute_pqk``/``compute_qpl`` that is the quantum
            projection, not the raw features).

    Returns:
        numpy.ndarray or None: One score per row of ``X``, larger meaning the positive
        class; or ``None`` when the estimator offers no ranking at all. ``None`` is a
        real answer and callers must pass it on -- ``modeleval`` records NaN for it
        rather than substituting a label-based number, which is the bug this helper
        exists to fix.
    """
    for name in ("predict_proba", "decision_function"):
        # getattr-with-default rather than hasattr-then-call: scikit-learn guards
        # ``SVC.predict_proba`` with ``available_if``, which raises AttributeError on
        # *attribute access* when ``probability=False``, and getattr swallows that.
        method = getattr(estimator, name, None)
        if not callable(method):
            continue
        try:
            raw = np.asarray(method(X), dtype=float)
        except (AttributeError, NotImplementedError):
            # An estimator that advertises the method and refuses to run it. Treated
            # as "no score from this route" so the next one still gets a turn.
            continue
        scores = _score_vector(raw, estimator)
        if scores is not None:
            return scores
    return None


#: The quality metrics ``modeleval`` writes into every ``results_<model>`` row, in the
#: order they appear there. ``time`` is deliberately excluded: it is in the row, but it
#: is a cost, not a score, so a consumer that ranks or averages over the block must not
#: fold it in with the rest. Use ``METRIC_COLUMNS`` where cost belongs alongside quality.
SCORE_COLUMNS = (
    "accuracy",
    "f1_score",
    "balanced_accuracy",
    "mcc",
    "auc",
    "pr_auc",
)

#: The prefix of the per-model metrics column ``modeleval`` builds (``results_<model>``).
#: Code that READS those columns elsewhere names it through this constant, so that
#: modeleval stays the only place a ``results_`` column is composed.
RESULTS_PREFIX = "results_"

#: Every numeric column of the metrics row: the scores above plus wall-clock cost.
METRIC_COLUMNS = SCORE_COLUMNS + ("time",)

#: The hyperparameter-search evidence a TUNED row carries (``BestParams_Tuned``;
#: untuned rows lack them, so they read NaN in ModelResults.csv): the metric searched
#: on, the chosen configuration's validation score, and whether the parameters were
#: reused from the frozen quantum cache. ``tuning_score`` is numeric but it is a score
#: OF THE MODEL, label-dependent like the metrics above, so any consumer that treats
#: numeric columns as dataset covariates (``utils.meta_regression``) must exclude these.
TUNING_EVIDENCE_COLUMNS = ("tuning_metric", "tuning_score", "tuning_reused")


def available_metric_columns(columns, candidates=METRIC_COLUMNS):
    """Those of ``candidates`` that ``columns`` actually holds, in ``candidates`` order.

    Selecting the metric block by hardcoded name raises ``KeyError`` on every
    ModelResults.csv written before that name existed. That is not hypothetical: it is
    the same failure the ``BestParams_GridSearch`` comment in ``apps/sage/sage.py``
    records, where demanding a column older tables lack made QuantumSage
    unconstructible from its own documented input in either configuration.
    ``balanced_accuracy``, ``mcc`` and ``pr_auc`` were added after the committed
    benchmark table was produced, and that table cannot be regenerated from this
    repository, so every consumer of the block has to tolerate their absence rather
    than require them.

    Returns a list rather than a tuple, so the result can be used directly as a
    DataFrame key.
    """
    present = set(columns)
    return [name for name in candidates if name in present]


def _positive_label(y_true):
    """The label ``average_precision_score`` must be told to treat as positive.

    This exists because two sklearn metrics disagree about what "positive" means, and
    the scores handed to them here are oriented for only one of the two.

    ``roc_auc_score`` infers the positive class as the *larger* of the two labels in
    ``y_true``, and :func:`_positive_class_column` orients ``y_score`` to match that --
    it selects the largest class. ``average_precision_score`` does not infer anything:
    it defaults to ``pos_label=1``. Those coincide for a ``{0, 1}`` target and diverge
    for every other encoding, and PMLB/OpenML/libsvm targets are not all ``{0, 1}`` --
    ``{1, 2}`` and ``{-1, 1}`` both occur. On a ``{1, 2}`` target the default would
    score class ``1`` against a ranking built for class ``2``, returning roughly
    ``1 - AP`` rather than ``AP``: a silently *inverted* precision-recall curve, worst
    on exactly the imbalanced datasets PR-AUC was added to describe.

    Returning the largest class keeps PR-AUC on the same orientation as ``auc`` and as
    ``y_score`` itself, so the three are comparable.

    Args:
        y_true (array-like): The true labels.

    Returns:
        The label to pass as ``pos_label``, or ``None`` when ``y_true`` does not hold
        exactly two classes -- in which case a binary PR-AUC is undefined and the
        caller records NaN.
    """
    classes = np.unique(np.asarray(y_true))
    if classes.size != 2:
        return None
    # np.unique sorts, so the last entry is the larger label -- without classes.max(),
    # which raises on a string array.
    return classes[-1]


def _was_tuned(model, tuned):
    """Whether this row's parameters came from a hyperparameter search.

    Args:
        model (str): The label this row is filed under.
        tuned (bool or None): An explicit answer, or None to infer one.

    Returns:
        bool: ``tuned`` when given, else whether ``model`` ends in ``_opt``.

    Note:
        The inference works because ``model_run`` labels a tuned run ``<name>_opt`` --
        the dispatch key and the column name are the same string. It is deliberately not
        a substring test: ``qpl_opt_rf`` contains ``_opt`` and so does a hypothetical
        head called ``adam_optimizer``, so ``compute_qpl`` states the answer instead of
        relying on a match that would be right by luck.
    """
    if tuned is not None:
        return bool(tuned)
    return str(model).endswith("_opt")


def modeleval(
    y_test,
    y_predicted,
    beg_time,
    params,
    args,
    model: str,
    verbose=True,
    average="weighted",
    y_score=None,
    tuned=None,
):
    """
    Evaluates the model performance using accuracy, F1 score, balanced accuracy,
    Matthews correlation, ROC AUC and PR AUC.

    ``accuracy`` and ``f1_score`` are computed from ``y_predicted``. ``auc`` is
    computed from ``y_score`` and from nothing else.

    **The ``auc`` column changed meaning here.** It used to be
    ``roc_auc_score(y_test, y_predicted)`` -- ``roc_auc_score`` applied to *hard
    predicted labels*, which on a binary target is identically
    ``balanced_accuracy_score(y_test, y_predicted)`` and not any ranking AUC. Every
    ``auc`` QBioCode wrote before this change is that statistic, including the
    committed ``tutorial/QSage/data/qprofiler_benchmarks.csv`` that QuantumSage trains
    on, so old and new numbers are not comparable.

    ``auc`` is ``float('nan')`` when, and only when, a real AUC cannot be computed:

    * ``y_score`` is ``None`` -- the caller's estimator exposes neither
      ``predict_proba`` nor ``decision_function`` (see :func:`extract_binary_scores`,
      which returns ``None`` for exactly that case);
    * ``y_score`` carries no single ranking, because the target has three or more
      classes;
    * ``y_test`` holds one class only, so no ROC curve is defined.

    NaN is deliberate. Falling back to the label-based number would put a different
    statistic under the same column name, which is what went wrong before, and every
    reader of this column -- ``qc_winner_finder``, QuantumSage, the correlation
    analysis -- would carry it into a published figure believing it to be an AUC. A
    missing value is visible; a mislabelled one is not.

    Args:
        y_test (array-like): True labels for the test set.
        y_predicted (array-like): Predicted labels by the model.
        beg_time (float): Start time for measuring execution time.
        params (dict): Model parameters used during training.
        args (dict): Read for one key only, ``'average'``, and never for model
            configuration. ``args['grid_search']`` used to be the only key this function
            touched, and it was the wrong signal -- a run-wide flag deciding a per-row
            column (see the comment at the parameter-column branch below). ``tuned``
            replaced it, and nothing here reads it any more. ``'average'`` is the opposite
            case: a genuinely run-wide choice about how a multiclass metric is averaged,
            which is exactly what a run-wide dict should carry. It is read with ``.get``
            and falls back to the ``average`` parameter's own default, so an ``args`` that
            omits it behaves as before.

            The parameter stays because all 24 call sites pass it positionally, and
            because dropping it would be a breaking change to a public function for no
            gain. One incidental benefit of the ``grid_search`` removal: a direct
            ``compute_<model>(...)`` call with an ``args`` dict that has no
            ``'grid_search'`` key used to raise ``KeyError`` here, *after* the fit had
            completed. It no longer can.
        model (str): Name of the model being evaluated.
        verbose (bool): If True, prints the evaluation results.
        average (str): Type of averaging to use for F1 score calculation.
            Default is 'weighted'.
        y_score (array-like or None): Continuous scores for the positive class, one
            per test sample -- probabilities or decision values, never labels. This is
            the only input to ``auc``. Default None, which records NaN.
        tuned (bool or None): Whether a hyperparameter search produced ``params``, which
            decides between the ``BestParams_Tuned`` and ``Model_Parameters`` column.
            Default None means infer it from ``model``: the dispatcher labels a tuned run
            ``<name>_opt``, so the label already carries the answer for 13 of the 14
            learners. ``compute_qpl`` passes it explicitly, because its label is
            ``qpl_opt_<head>`` and the marker is not a suffix.

            When ``params`` is a :class:`~qbiocode.learning._tuning.TunedParams` -- what
            every classical tuner returns -- a tuned row also carries its selection
            evidence as ``tuning_metric``, ``tuning_score`` (the chosen configuration's
            best mean cross-validated score) and ``tuning_reused``, and the parameter
            column holds a plain ``dict`` copy. The quantum wrappers add the same three
            keys through ``record_tuned_params``. Untuned rows carry none of them.

            When that ``TunedParams`` also carries ``trials`` (a search scored on a
            validation split, ``split_mode: manifest``), ``tuning_score`` is the chosen
            trial's validation score and the frame gains a fourth column,
            ``trials_<model>``, holding its ``trial_log()`` dict. Otherwise no such
            column is added.

    Returns:
        pd.DataFrame: A ONE-ROW frame with three columns per model, named for it:
        ``y_test_<model>``, ``y_predicted_<model>`` and ``y_score_<model>`` each hold one
        array in a single cell (``y_score_<model>`` holds ``None`` when the estimator
        publishes no ranking), and ``results_<model>`` holds a dict of the six metrics --
        ``accuracy``, ``f1_score``, ``balanced_accuracy``, ``mcc``, ``auc``, ``pr_auc`` --
        plus ``time`` and exactly one parameter column, ``BestParams_Tuned`` if this model
        was tuned and ``Model_Parameters`` if it was not.

        ``qprofiler`` flattens ``results_<model>`` into one CSV row per model with
        ``{**row_base, **outervalue[0]}``, so a key added to that dict reaches
        ModelResults.csv with no change to the writer, and ``_append_model_row`` widens an
        existing header rather than misaligning it.
    """
    # `average` is a documented config key, so honour it. No call site passes it -- all 24
    # pass `args` positionally and stop before this parameter -- so the signature default
    # silently won and a config asking for 'macro' got 'weighted' without a word. `args`
    # is already here, so read it from there and keep the parameter as the fallback for
    # direct calls whose `args` omits the key.
    average = (args or {}).get("average", average) or average

    # Calculate evaluation metrics
    if y_score is None:
        auc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_test, np.asarray(y_score, dtype=float))
        except ValueError:
            # A single-class y_test, or a multiclass one that slipped past
            # ``_score_vector``. Reported as missing for the reason in the docstring;
            # ``evaluation_metrics`` below answers a malformed AUC request the same way.
            auc = float("nan")
    accuracy = accuracy_score(y_test, y_predicted, normalize=True)
    # zero_division=0 is explicit rather than inherited: a model that predicts a single
    # class leaves the other label with no predicted samples, and the default emits an
    # UndefinedMetricWarning per fold while returning 0.0 anyway. 0.0 is the reading we
    # want (see the mcc note below on why 0.0 beats NaN here), and every other f1_score
    # call in this repo already passes zero_division=0 -- this was the one that did not.
    f1 = f1_score(y_test, y_predicted, average=average, zero_division=0)

    # Three further metrics, added because `accuracy` and a *weighted* F1 cannot carry an
    # imbalance result on their own. On this corpus a majority-class DummyClassifier
    # reaches a weighted F1 of 0.906 (openml__ozone-level-8hr, minority fraction 0.063),
    # so a headline weighted F1 near 0.9 there says nothing at all about whether a model
    # learned anything. Each of these three fails differently on that dataset, which is
    # the point of carrying all of them:
    #
    #   balanced_accuracy  mean of per-class recall; 0.5 for the majority-class dummy on
    #                      ANY imbalance, so it exposes the dummy that weighted F1 hides.
    #   mcc                Matthews correlation; 0.0 for the dummy, and unlike balanced
    #                      accuracy it also penalises a model that buys minority recall
    #                      with a flood of false positives. Symmetric in the two classes,
    #                      so it needs no pos_label.
    #   pr_auc             average precision: the threshold-free companion to `auc` that
    #                      does not credit true negatives, which is the whole difficulty
    #                      with ROC-AUC under heavy imbalance.
    #
    # `mcc` and `balanced_accuracy` read `y_predicted`, so they are available for every
    # model. `pr_auc` needs the ranking, so it is NaN for exactly the estimators whose
    # `auc` is NaN -- see the docstring.
    try:
        balanced_accuracy = balanced_accuracy_score(y_test, y_predicted)
    except ValueError:
        balanced_accuracy = float("nan")
    # Returns 0.0, not NaN, when a denominator vanishes -- a model predicting one class
    # everywhere. That is the correct reading (no correlation with the truth), and it is
    # deliberately not converted to NaN: 0.0 is an informative score here, whereas NaN
    # would be dropped from a mean over datasets and quietly flatter the model.
    try:
        mcc = matthews_corrcoef(y_test, y_predicted)
    except ValueError:
        mcc = float("nan")
    pos_label = _positive_label(y_test)
    if y_score is None or pos_label is None:
        pr_auc = float("nan")
    else:
        try:
            pr_auc = average_precision_score(
                y_test, np.asarray(y_score, dtype=float), pos_label=pos_label
            )
        except ValueError:
            pr_auc = float("nan")

    compile_time = time.time() - beg_time
    if verbose == True:
        print_results(model, accuracy, f1, compile_time, params)

    # The tuned-parameter column is named for the branch that produced it. It used to
    # be 'BestParams_GridSearch' back when an exhaustive grid was the only search;
    # Optuna is now the default, so the name no longer claims an engine. Readers
    # (qc_winner_finder, QuantumSage) accept the old name too, because every
    # ModelResults.csv written before this change carries it.
    #
    # Decided per ROW, not per run. This used to read `args["grid_search"] == True`, a
    # run-wide flag -- so in a `grid_search: True` run every model reported under
    # 'BestParams_Tuned' whether or not it had been tuned. That mattered because tuning
    # is not run-wide: quantum models stay untuned unless `tune_quantum` is also set (the
    # documented default), so their feature-map and kernel *defaults* were filed under a
    # column claiming a search that never ran -- and both qc_winner_finder and
    # QuantumSage prefer that column when reading parameters back.
    #
    # A consequence worth knowing: one run can now carry BOTH columns -- tuned classical
    # models and untuned quantum ones in the same table. Every reader already accepts
    # either name; qc_winner_finder additionally coalesces them per row.
    # One frame, built once. The two branches this replaces were identical but for the
    # name of the parameter key, so every other column was written out twice and a change
    # to any of them had to be made in both places to take effect.
    parameter_column = "BestParams_Tuned" if _was_tuned(model, tuned) else "Model_Parameters"
    # The validation score of the chosen configuration, which is otherwise lost once the
    # search returns. Read off the attributes rather than imported by type: this module
    # sits below qbiocode.learning, and importing _tuning here would pull optuna into
    # every evaluation. Only tuned rows get the keys, so an untuned row reads NaN in
    # the CSV exactly as it did before they existed.
    tuning_evidence = {}
    trial_log = None
    if parameter_column == "BestParams_Tuned" and callable(getattr(params, "evidence", None)):
        tuning_evidence = params.evidence()
        # Every trial of a validation-split search, as one plain-dict cell; see
        # qbiocode.evaluation.protocol.trial_log. None for any other search.
        if callable(getattr(params, "trial_log", None)):
            trial_log = params.trial_log()
        # Stored as a plain dict so results.pkl stays readable without the tuning
        # module's class; str() -- the BestParams_Tuned CSV text -- is the same either way.
        params = dict(params)
    frame = pd.DataFrame(
        {
            "y_test_" + model: [y_test],
            "y_predicted_" + model: [y_predicted],
            # Persisted so that every threshold-free metric stays recomputable from
            # results.pkl without re-fitting anything. Before this, `y_score` was built by
            # `extract_binary_scores`, consumed once for `auc`, and dropped -- which made
            # `auc` and `pr_auc` the only ranking statistics this benchmark could ever
            # report. Adding a metric later (a calibration curve, a Brier score, an AUC at
            # a different positive class) would have meant re-running the whole sweep,
            # ~3500 quantum fits included. `y_predicted` alone cannot substitute: hard
            # labels carry no ranking.
            #
            # Stored as `None` when the estimator offers no ranking at all, which is a real
            # answer and not a failure -- see `extract_binary_scores`. Readers must expect
            # the column to hold None for those rows.
            "y_score_" + model: [y_score],
            "results_"
            + model: [
                {
                    "model": model,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "balanced_accuracy": balanced_accuracy,
                    "mcc": mcc,
                    "time": compile_time,
                    "auc": auc,
                    "pr_auc": pr_auc,
                    parameter_column: params,
                    **tuning_evidence,
                }
            ],
        }
    )
    if trial_log is not None:
        # Assigned after construction: a dict handed to the constructor inside a list
        # is one cell either way, but this keeps the column absent, not NaN, when
        # there is no log.
        frame[TRIALS_PREFIX + model] = pd.Series([trial_log], dtype=object)
    return frame


def evaluation_metrics(predictions, y_test, metrics=["accuracy", "brier"], save=False):
    """
    Calculate evaluation metrics for classification predictions.

    Computes specified metrics for model predictions. Supports accuracy, Brier score,
    F1 score, precision, recall, and AUC-ROC. The Brier score measures the mean
    squared difference between predicted probabilities and actual outcomes, providing
    a measure of calibration quality.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes)
    y_test : np.ndarray
        True labels, shape (n_samples,)
    metrics : list of str, optional
        List of metrics to compute. Options: 'accuracy', 'brier', 'f1',
        'precision', 'recall', 'auc' (default: ['accuracy', 'brier'])
    save : bool, optional
        Whether to save results (reserved for future use, default: False)

    Returns
    -------
    tuple or dict
        If metrics=['accuracy', 'brier'] (default): returns (accuracy, brier_score)
        Otherwise: returns dict with requested metrics as keys

    Examples
    --------
    >>> import numpy as np
    >>> from qbiocode.evaluation import evaluation_metrics
    >>>
    >>> # Binary classification example - default metrics
    >>> predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    >>> y_test = np.array([0, 1, 0])
    >>> accuracy, brier = evaluation_metrics(predictions, y_test)
    >>> print(f"Accuracy: {accuracy:.2f}, Brier Score: {brier:.3f}")
    Accuracy: 1.00, Brier Score: 0.060

    >>> # Multiple metrics
    >>> results = evaluation_metrics(predictions, y_test,
    ...                              metrics=['accuracy', 'brier', 'f1', 'auc'])
    >>> print(results)
    {'accuracy': 1.0, 'brier': 0.06, 'f1': 1.0, 'auc': 1.0}

    Notes
    -----
    - For binary classification, Brier score is computed using the probability
      of the positive class
    - For multi-class classification, the average Brier score across all classes
      is returned
    - F1, precision, and recall use weighted averaging for multi-class
    - AUC uses one-vs-rest for multi-class
    - Lower Brier scores indicate better calibrated probability predictions

    References
    ----------
    Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability".
    Monthly Weather Review, 78(1), 1-3.
    """
    import numpy as np
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Get predicted classes
    y_pred = np.argmax(predictions, axis=1)

    results = {}

    # Calculate requested metrics
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_test, y_pred)

    if "brier" in metrics:
        if predictions.shape[1] == 2:
            # Binary classification: use probability of positive class
            results["brier"] = brier_score_loss(y_test, predictions[:, 1])
        else:
            # Multi-class: use average Brier score across all classes
            results["brier"] = np.mean(
                [
                    brier_score_loss(y_test == i, predictions[:, i])
                    for i in range(predictions.shape[1])
                ]
            )

    if "f1" in metrics:
        results["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if "precision" in metrics:
        results["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    if "recall" in metrics:
        results["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    if "auc" in metrics:
        try:
            if predictions.shape[1] == 2:
                # Binary classification
                results["auc"] = roc_auc_score(y_test, predictions[:, 1])
            else:
                # Multi-class: one-vs-rest
                results["auc"] = roc_auc_score(
                    y_test, predictions, multi_class="ovr", average="weighted"
                )
        except ValueError:
            # Handle cases where AUC cannot be computed (e.g., single class in y_test)
            results["auc"] = np.nan

    # For backward compatibility: return tuple if default metrics
    if metrics == ["accuracy", "brier"]:
        return results["accuracy"], results["brier"]

    return results
