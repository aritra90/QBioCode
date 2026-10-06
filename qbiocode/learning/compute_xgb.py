# ====== Base class imports ======

import time

import numpy as np

# ====== Scikit-learn imports ======

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = None
except Exception as e:
    # Catch all exceptions including XGBoostError, ImportError, OSError
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(e)
    XGBClassifier = None  # type: ignore

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qbiocode.learning._grid import one_value, warn_ignored_hyperparameter
from qbiocode.learning._tuning import search_hyperparameters, tuning_scorer
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval

# ====== Begin functions ======


#: Why the thread cap is passed to every fit rather than searched. Quoted into the
#: message :func:`qbiocode.learning._grid.one_value` raises.
_NTHREAD_WHY = (
    "it caps the threads each fit may use, which is a resource decision rather than a "
    "model one -- `model_run` already fans the models out over joblib, so anything above "
    "1 oversubscribes the cores the job asked for"
)


def _thread_kwargs(n_jobs, nthread, block):
    """Settle XGBoost's thread cap from whichever of its two spellings the config used.

    XGBoost's sklearn wrapper documents ``n_jobs``; the native API calls the same knob
    ``nthread`` and the wrapper still accepts it. A config author reaches for either, and
    until this existed the other one was not a silent no-op but a crash: the block's keys
    arrive as keyword arguments to this module's functions, so ``'nthread': 1`` raised
    ``TypeError: compute_xgb_opt() got an unexpected keyword argument 'nthread'`` from
    inside a joblib worker. Both spellings are taken, and disagreement is refused rather
    than resolved by precedence -- there is no reading of two different caps that is what
    the author meant.

    Unset means unset. XGBoost then takes ``omp_get_max_threads()``, which is the right
    default for a single interactive fit and the wrong one under ``model_run``'s joblib
    fan-out; the pilot configs pin it to 1 and ``submit_pilot.sh`` exports
    ``OMP_NUM_THREADS=1`` as the outer belt. Defaulting it to 1 here would instead impose
    a permanent single-thread ceiling on every caller, which is not this function's
    decision to make -- the same reason ``compute_catboost`` leaves ``thread_count`` unset.

    Args:
        n_jobs: Value from the ``n_jobs`` key, if the block set one.
        nthread: Value from the ``nthread`` key, if the block set one.
        block (str): The config block both came from, for the error message.

    Returns:
        dict: ``{'n_jobs': cap}``, or empty to leave XGBoost at its own default. Returning
        the keyword rather than the value keeps an unset cap out of ``get_params()`` and
        out of the estimator's identity under sklearn's ``clone``, so a config that sets
        nothing builds exactly the estimator it built before this argument existed.

    Raises:
        ValueError: If the block sets both spellings to different values.
    """
    resolved_n_jobs = one_value("n_jobs", n_jobs, _NTHREAD_WHY, block)
    resolved_nthread = one_value("nthread", nthread, _NTHREAD_WHY, block)
    if (
        resolved_n_jobs is not None
        and resolved_nthread is not None
        and resolved_n_jobs != resolved_nthread
    ):
        raise ValueError(
            f"{block!r} sets both 'n_jobs' ({resolved_n_jobs!r}) and 'nthread' "
            f"({resolved_nthread!r}), which are two spellings of one XGBoost setting, to "
            f"different values. Give one of them, or give both the same value."
        )
    cap = resolved_n_jobs if resolved_n_jobs is not None else resolved_nthread
    return {} if cap is None else {"n_jobs": cap}


def compute_xgb(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="xgb",
    data_key="",
    n_estimators=100,
    *,
    criterion="gini",
    max_depth=None,
    subsample=0.5,
    learning_rate=0.5,
    colsample_bytree=1,
    min_child_weight=1,
    random_state=None,
    n_jobs=None,
    nthread=None,
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost. It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'XGBoost'.
        data_key (str): Key for identifying the dataset, default is an empty string.
        n_estimators (int): Number of trees in the forest, default is 100.
        max_depth (int or None): Maximum depth of the tree, default is None.
        subsample (float) : Subsample ratio of the training instances. Default 0.5
        learning_rate (float): Step size shrinkage used in update to prevent overfitting. Default is 0.5
        colsample_bytree  (float): subsample ratio of columns when constructing each tree. Default is 1
        min_child_weight (int) : Minimum sum of instance weight (hessian) needed in a child. Default is 1
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator drawing from the global RNG.
        n_jobs (int or None): Threads each XGBoost fit may use. ``nthread`` is XGBoost's
            own name for the same setting and is accepted as an alias; giving both
            different values is an error. Left unset XGBoost takes every core
            ``omp_get_max_threads()`` reports, which oversubscribes badly underneath
            ``model_run``'s joblib fan-out -- measured on a 128-core node, one 42-row fit
            went from over 280 s to 0.06 s once threads were capped. Distinct from the
            top-level ``n_jobs`` in a config, which sizes that fan-out rather than a
            single fit.
        nthread (int or None): Alias for ``n_jobs``; see above.
     Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.

    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    xgb = OneVsOneClassifier(
        XGBClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,  # type: ignore
            subsample=subsample,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            **_thread_kwargs(n_jobs, nthread, "xgb_args"),
        )
    )
    # Fit the training datset
    model_fit = xgb.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = xgb.predict(X_test)
    # `auc` is a ranking metric and is computed from these scores alone -- passing
    # y_predicted, as this used to, silently reported balanced accuracy instead.
    # OneVsOneClassifier publishes no predict_proba, so what comes back here is its
    # decision_function; see extract_binary_scores for why that is a real ranking on a
    # binary target, and None (recorded as NaN) when it is not.
    y_score = extract_binary_scores(xgb, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        model_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
    )


def compute_xgb_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    cv=5,
    model="xgb",
    bootstrap=None,
    max_depth=None,
    max_features=None,
    learning_rate=None,
    subsample=None,
    colsample_bytree=None,
    n_estimators=None,
    min_child_weight=None,
    random_state=None,
    n_jobs=None,
    nthread=None,
    *,
    tuner="optuna",
    n_trials=50,
):
    """
    This function generates a model using an Extreme Gradient Boositing (xgb) Classifier method as implemented in xgboost.
    The difference here is that this function tunes the model's hyperparameters.
    The values or ranges searched for each parameter are specified in the config.yaml file,
    and ``tuner`` selects the search engine (Optuna by default). The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to repeat the search.
    The model is trained on the training dataset and validated on the test dataset. The function returns the evaluation of the model
    on the test dataset, including accuracy, AUC, F1 score, and the time taken to train and validate the model across the search.
    This function is designed to be used in a supervised learning context, where the goal is to classify data points.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Additional arguments, typically from a configuration file.
        verbose (bool): If True, prints additional information during execution.
        cv (int): Number of cross-validation folds, default is 5.
        model (str): Name of the model being used, default is 'Random Forest'.
        bootstrap (list): List of bootstrap options for the search.
        max_depth (list): List of maximum depth options for the search.
        subsample (list): List of subsample ratio of the training instances options for the search.
        learning_rate (list): List of step size shrinkage used in update to prevent overfitting options for the search.
        colsample_bytree (list): List of subsample ratio of columns when constructing each tree options for the search.
        n_estimators (list): List of number of estimators options for the search.
        min_child_weight (list): List of minimum sum of instance weight (hessian) needed in a childoptions for the search.
        random_state (int or None): Seed for the estimator's own randomness. QProfiler fills this in from the run's ``seed`` so two runs at one seed agree; None leaves the estimator drawing from the global RNG.
        n_jobs (int or None): Threads each XGBoost fit may use. ``nthread`` is XGBoost's
            own name for the same setting and is accepted as an alias; giving both
            different values is an error. Left unset XGBoost takes every core
            ``omp_get_max_threads()`` reports, which oversubscribes badly underneath
            ``model_run``'s joblib fan-out -- measured on a 128-core node, one 42-row fit
            went from over 280 s to 0.06 s once threads were capped. Distinct from the
            top-level ``n_jobs`` in a config, which sizes that fan-out rather than a
            single fit.
        nthread (int or None): Alias for ``n_jobs``; see above.

        tuner (str): Which search to run. ``'optuna'`` (default) spends ``n_trials`` on
            Optuna's TPE sampler, which also allows a hyperparameter to be given as a
            ``{low, high}`` range rather than a list. ``'grid'`` restores the exhaustive
            ``GridSearchCV`` sweep over every combination.
        n_trials (int): Trial budget when ``tuner='optuna'``, default is 50. Lowered
            automatically when the configured values describe fewer distinct
            combinations than that, so a small block does not re-evaluate the same
            models.
    Returns:
        modeleval (dict): A dictionary containing the evaluation metrics of the model, including accuracy, AUC, F1 score, and the time taken for training and validation.

    Raises:
        ImportError: If XGBoost is not properly installed or configured.
    """

    if not XGBOOST_AVAILABLE:
        error_msg = (
            "XGBoost is not properly installed or configured.\n"
            f"Error: {_XGBOOST_ERROR}\n\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n\n"
            "See installation documentation for more details."
        )
        raise ImportError(error_msg)

    beg_time = time.time()
    # XGBoost has no bootstrap parameter, but its sklearn wrapper accepts unknown
    # keyword arguments without complaint, so this was never an error -- just a
    # silently doubled search returning identical models.
    if bootstrap:
        warn_ignored_hyperparameter(
            "xgb", "bootstrap", "XGBoost does not implement -- it samples rows via 'subsample'."
        )

    # Only the hyperparameters actually supplied. Passing all of them meant a
    # config that named a subset died in sklearn on the first one it left at its
    # `[]` default; see qbiocode.learning._grid.
    candidates = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "bootstrap": bootstrap,
    }

    # Resolved before the search so a contradictory block fails now, with a message naming
    # the config key, rather than on whichever trial first samples it.
    fixed = {
        "random_state": random_state,
        **_thread_kwargs(n_jobs, nthread, "gridsearch_xgb_args"),
    }

    best_params = search_hyperparameters(
        "xgb",
        XGBClassifier,
        candidates,
        X_train,
        y_train,
        cv=cv,
        tuner=tuner,
        scoring=tuning_scorer(args),
        n_trials=n_trials,
        seed=random_state,
        fixed=fixed,
    )
    # `**fixed` rather than `random_state=random_state`: `search_hyperparameters` applies
    # `fixed` to every trial's estimator but returns only the searched parameters, so a
    # refit that names one of them by hand silently drops the rest -- here, the thread cap,
    # on the one fit whose cost is not amortised over a cross-validation. This is what
    # `compute_catboost` already does with its own `fixed`.
    best_xgb = XGBClassifier(**best_params, **fixed)  # type: ignore
    best_xgb.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_xgb.predict(X_test)
    # Fitted unwrapped, so predict_proba is available. `auc` is computed from these
    # scores alone; see extract_binary_scores.
    y_score = extract_binary_scores(best_xgb, X_test)
    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        best_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
        # This function IS the tuned branch, so it states so rather than letting
        # modeleval infer it from the label: a DIRECT call leaves `model` at its
        # display-name default ('Decision Tree'), which carries no _opt marker.
        tuned=True,
    )
