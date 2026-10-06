# ====== Base class imports ======
import inspect
import json
import logging
import os
import warnings

import pandas as pd

# ======= Parallelization =====
from joblib import Parallel, delayed

current_dir = os.getcwd()

logger = logging.getLogger(__name__)

#: Search engines ``args['tuner']`` may name. Both are driven by the same
#: ``gridsearch_<model>_args`` blocks; see :mod:`qbiocode.learning._tuning`.
_TUNERS = frozenset({"optuna", "grid"})

#: Keys a TUNED quantum model still reads from ``<model>_args`` rather than from
#: ``gridsearch_<model>_args``, per model.
#:
#: The two blocks answer different questions. ``gridsearch_<model>_args`` says what to
#: SEARCH; ``<model>_args`` says how to run the model. Almost every setting is one or the
#: other, but ``classical_models`` is unambiguously the second -- it selects which
#: classical heads ``compute_qpl`` fits on the quantum projection, so its value is a list
#: of heads and not a list of candidates to choose among.
#:
#: The tuned branch below used to build its kwargs from the gridsearch block alone, so
#: ``qpl_args: {classical_models: ['lr']}`` was honoured with tuning off and silently
#: dropped with it on -- a tuned run searched and reported all six default heads. That is
#: worse than a slow run: ``_tuning._metric_of`` scores a QPL candidate by the MEAN
#: tuning metric across heads, so the projection was chosen to suit heads the config had
#: excluded. Naming the key in the gridsearch block instead was not a workaround either;
#: it reached ``compute_qpl_opt`` as an unexpected keyword argument.
_QUANTUM_PASSTHROUGH = {"qpl": ("classical_models",)}

#: The models that run on a quantum backend, as ``args['model']`` spells them.
#:
#: The dispatcher branches on this to decide what ``tune_quantum`` covers, and
#: :mod:`qbiocode.visualization.visualize_correlation` reads it to colour and order the
#: quantum models in the figures. That second consumer is why it is a module constant
#: rather than the local it used to be: the plotting code had its own hardcoded
#: ``("QNN", "PQK", "VQC", "QSVC")`` -- and a THIRD copy inline in the scatter arm -- both
#: missing ``qpl``, so every QPL row was coloured and ordered as a classical model in
#: every published figure. One list, one place to add the next quantum learner to.
QUANTUM_MODELS = frozenset({"qsvc", "qnn", "vqc", "pqk", "qpl"})


def _call_with_global_seeds(compute_fn, seed, q_seed, *fn_args, **fn_kwargs):
    """Re-establish the global RNG seeds inside the worker, then run ``compute_fn``.

    ``qprofiler`` sets ``np.random.seed`` and ``algorithm_globals.random_seed`` in
    the parent process, but the models run under joblib's loky backend, which
    starts fresh interpreters. Neither seed crosses that boundary, so anything
    reading a global RNG -- ``compute_qnn``'s initial weights come from
    ``algorithm_globals.random`` -- started from OS entropy and produced a
    different answer on every run.

    This is a floor, not the mechanism: ``_seeded_kwargs`` below sets
    ``random_state`` on each estimator explicitly, because joblib batches tasks
    and how far an earlier task advanced a shared global stream depends on
    timing. Seeding here covers the randomness that has no ``random_state`` to
    set.

    The caller's numpy global stream is restored on return. Under loky this runs in a
    throwaway worker and that is moot, but with ``n_jobs: 1`` joblib runs it in the
    calling process, and the re-seed plus the model's own draws then moved the CALLER's
    stream -- so whatever qprofiler drew next depended on which model had just run.
    """
    import numpy as np

    caller_state = np.random.get_state()
    try:
        return _seed_and_call(compute_fn, seed, q_seed, fn_args, fn_kwargs)
    finally:
        np.random.set_state(caller_state)


def _seed_and_call(compute_fn, seed, q_seed, fn_args, fn_kwargs):
    _set_global_seeds(seed, q_seed)
    return compute_fn(*fn_args, **fn_kwargs)


def _set_global_seeds(seed, q_seed):
    """Seed numpy's global stream and both qiskit ``algorithm_globals`` singletons."""
    import numpy as np

    if seed is not None:
        np.random.seed(seed)
    if q_seed is not None:
        # TWO distinct singletons, not one. qiskit-machine-learning 0.9 ships its own
        # `algorithm_globals` (qiskit_machine_learning.utils) alongside the
        # qiskit-algorithms one, and they are separate objects with separate state --
        # setting `random_seed` on the qiskit-algorithms singleton leaves the
        # qiskit-machine-learning one at None. VQC and QNN draw their initial point
        # through `TrainableModel`, which reads the qiskit-machine-learning one, so
        # seeding only the first left both models starting from OS entropy: two runs
        # at the same `q_seed` disagreed, and nothing said why. Seed both.
        for module_path in (
            "qiskit_algorithms.utils",
            "qiskit_machine_learning.utils",
        ):
            try:
                module = __import__(module_path, fromlist=["algorithm_globals"])
                algorithm_globals = module.algorithm_globals
            except (ImportError, AttributeError):
                # Classical-only install, or a version that does not ship this
                # singleton: nothing in this worker reads it, so there is nothing
                # to set.
                continue
            algorithm_globals.random_seed = q_seed


class _Reseed:
    """The ``reseed`` callable a quantum ``_opt`` gets under ``split_mode: manifest``.

    Resets numpy's global stream, Python's ``random`` and both qiskit
    ``algorithm_globals`` to the state :func:`_seed_and_call` starts a model
    from. The tuner calls it before every trial and before the refit, so qnn and vqc --
    whose initial point is drawn from ``algorithm_globals`` -- start each trial from the
    same state whatever ran before it, and trial order cannot move a score. A class
    rather than a closure so it pickles into a loky worker by reference.
    """

    def __init__(self, seed, q_seed):
        self.seed = seed
        self.q_seed = q_seed

    def __call__(self):
        import random

        if self.seed is not None:
            random.seed(self.seed)
        _set_global_seeds(self.seed, self.q_seed)

    def __repr__(self):
        return f"_Reseed(seed={self.seed!r}, q_seed={self.q_seed!r})"


#: Keyword arguments model_run passes to a model function itself; a config block naming
#: one is dropped (with a warning) rather than colliding with it. See _seeded_kwargs.
_RESERVED_KWARGS = (
    "model", "data_key", "n_trials", "validation_split", "cv", "tuner", "verbose",
    "validation", "default_params", "reseed",
)


def _default_params(compute_fn, model_args):
    """The default config of one arm: ``compute_<m>``'s defaults under ``<m>_args``.

    What the arm runs at untuned, and so what a fold-based search enqueues as trial 0.
    Reserved keyword arguments (see :data:`_RESERVED_KWARGS`) and ``random_state`` --
    which the ``_opt`` function is given explicitly from ``args['seed']`` -- are left out.
    """
    from qbiocode.learning._grid import to_plain

    try:
        parameters = inspect.signature(compute_fn).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover - C callables
        parameters = ()
    defaults = {
        p.name: p.default for p in parameters
        if p.default is not inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    defaults.update(to_plain(dict(model_args or {})))
    for key in _RESERVED_KWARGS + ("random_state",):
        defaults.pop(key, None)
    return defaults


def _check_fold_tuning(requested, compute_ml_dict, grid_search, tune_quantum, args):
    """What ``split_mode: manifest`` needs of a model_run config, checked before any fit.

    The protocol tunes every arm inside the fold on the same trial budget and selects on
    validation, so an arm that would run untuned, or be searched by an engine that
    ignores the budget, or reuse parameters frozen on another fold, is a config error.
    """
    from qbiocode.learning._param_cache import freezing_enabled

    if not grid_search:
        raise ValueError(
            "A validation split was given (split_mode: manifest), which tunes every "
            "model inside the fold, but grid_search is off. Set grid_search: True."
        )
    no_twin = [m for m in requested if (m + "_opt") not in compute_ml_dict]
    if no_twin:
        raise ValueError(
            f"split_mode: manifest tunes every model on the validation split, but "
            f"{no_twin} have no '_opt' implementation to tune. Drop them from "
            f"args['model']."
        )
    untuned = [m for m in requested if m in QUANTUM_MODELS and not tune_quantum]
    if untuned:
        raise ValueError(
            f"split_mode: manifest gives every arm the same tuning budget, but "
            f"tune_quantum is off, so {untuned} would run untuned. Set tune_quantum: True."
        )
    if args.get("tuner", "optuna") == "grid":
        raise ValueError(
            "tuner: 'grid' cannot be used with split_mode: manifest: the exhaustive grid "
            "ignores n_trials, so the arms could not be given one equal budget. Use "
            "tuner: optuna."
        )
    if freezing_enabled(args):
        raise ValueError(
            "freeze_quantum_params cannot be used with split_mode: manifest: each fold "
            "is tuned on its own validation rows, and parameters frozen on one fold were "
            "chosen on rows that are another fold's test rows. Set "
            "freeze_quantum_params: False."
        )
    n_trials = args.get("n_trials", 50)
    n_trials_quantum = args.get("n_trials_quantum")
    if (
        n_trials_quantum is not None
        and n_trials_quantum != n_trials
        and any(m in QUANTUM_MODELS for m in requested)
    ):
        logger.warning(
            "n_trials_quantum=%r is ignored under split_mode: manifest; every arm, "
            "quantum included, is tuned with n_trials=%r.", n_trials_quantum, n_trials,
        )


def model_run(X_train, X_test, y_train, y_test, data_key, args, validation=None):
    """This function runs the ML methods, with or without a grid search, as specified in the config.yaml file.
    It returns a python dictionary contatining these results, which can then be parsed out. It is designed to run
    each of the ML methods in parallel, for each data set (this is done by calling the Parallel module in results below).
    The arguments X_train, X_test, y_train, y_test are all passed in from the main script (qmlbench.py) as the input
    datasets are processed, while the remaining arguments are passed from the config.yaml file.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        data_key (str): Key for the dataset being processed.
        args (dict): Dictionary containing configuration parameters, including:

            - model: List of models to run.
            - n_jobs: Number of parallel jobs to run.
            - grid_search: Boolean indicating whether to tune hyperparameters.
            - tuner: 'optuna' (default) or 'grid' -- which search to run when
              grid_search is on.
            - n_trials: Trial budget for the Optuna tuner, default 50.
            - tuning_metric: What every tuner selects on, classical and quantum --
              'balanced_accuracy' (default), 'accuracy' (the objective before this key
              existed), 'mcc' or 'f1_score' (averaged by args['average']). See
              qbiocode.learning._tuning.tuning_metric.
            - cross_validation: Number of cross-validation folds, default 5.
            - gridsearch_<model>_args: Values or ranges to search for each model.
              'catboost' and 'tabpfn' use the same blocks as the other classical
              models; see qbiocode.learning.compute_catboost and .compute_tabpfn for
              the hyperparameters each accepts.
            - <model>_args: Additional arguments for each model.
        validation (ValidationSplit or None): ``split_mode: manifest``. The fit and
            validation rows of this outer training fold, with features transformed for
            the tuning stage (see :class:`qbiocode.evaluation.protocol.ValidationSplit`).
            Every model is then tuned -- classical and quantum alike, so grid_search and
            (for quantum models) tune_quantum must be on, and every model needs an
            '_opt' twin -- on ONE budget, args['n_trials'] (n_trials_quantum is ignored,
            with a warning when it differs). Each trial is one fit on the fit rows
            scored on the validation rows; trial 0 is the arm's default config
            (``compute_<m>``'s defaults under ``<m>_args``); the winner is refit on
            X_train and scored once on X_test. Each tuned model also contributes a
            'trials_<label>' key holding its trial log
            (:func:`qbiocode.evaluation.protocol.trial_log`). None (the default) runs
            exactly as before.

    Returns:
        model_total_result (dict): The results of every model run, ready to be turned
        into a Pandas DataFrame -- that is how the 'ModelResults.csv' files in the
        results directory are written when the main profiler runs
        (qbiocode-profiler.py).

        The keys are NOT the model names. Each model contributes four of them, each
        prefixed, where <label> is the name from args['model'] with '_opt' appended
        when that model was tuned:

            'results_<label>'      the metrics row
            'y_test_<label>'       the true labels it was scored against
            'y_predicted_<label>'  the labels it predicted
            'y_score_<label>'      the ranking score behind those labels, or None

        'y_score_<label>' is what the threshold-free metrics are computed from, and it
        is persisted so a reader can recompute a PR or ROC curve, or re-threshold, from
        the results table instead of re-fitting. It is None for a model that exposes
        neither predict_proba nor decision_function, which is also when 'auc' and
        'pr_auc' are NaN.

        So args['model'] = ['dt'] yields 'results_dt', not 'dt', and turning
        grid_search on yields 'results_dt_opt'. Every value is a one-entry dict keyed
        by the integer 0, because the frame is pivoted onto a single row index before
        to_dict() -- read a metrics row as result['results_dt'][0], not
        result['results_dt'].

        That row holds 'model', the METRIC_COLUMNS of
        qbiocode.evaluation.model_evaluation -- 'accuracy', 'f1_score',
        'balanced_accuracy', 'mcc', 'auc', 'pr_auc', 'time' -- and one parameter key: 'Model_Parameters' when the model ran at its configured
        hyperparameters, 'BestParams_Tuned' when it was tuned. See
        qbiocode.evaluation.model_evaluation.modeleval, which builds it -- in
        particular for 'auc' and 'pr_auc', which are ranking metrics and NaN where no
        score exists. 'balanced_accuracy' and 'mcc' are always finite; they are
        reported because 'accuracy' is misleading on the imbalanced datasets in the
        corpus and 'f1_score' ignores the true negatives.

    Raises:
        ValueError: If args['model'] is empty, names a model that is not in the
            dispatch table, or names one twice; if grid_search is on for a model with
            no '_opt' twin; if tune_quantum is on without grid_search, or without a
            gridsearch_<model>_args block per quantum model; or if args['tuner'] is
            not one of 'optuna' or 'grid', or args['tuning_metric'] is not a known
            metric. All of these are raised before any model is fitted. With
            ``validation``, also if grid_search is off, a model has no '_opt' twin, a
            quantum model is requested without tune_quantum, args['tuner'] is 'grid',
            or freeze_quantum_params is on.

    """

    # Lazy imports to avoid circular dependency
    # These imports happen inside the function, not at module level
    from qbiocode.learning.compute_catboost import compute_catboost, compute_catboost_opt
    from qbiocode.learning.compute_dt import compute_dt, compute_dt_opt
    from qbiocode.learning.compute_lr import compute_lr, compute_lr_opt
    from qbiocode.learning.compute_rf import compute_rf, compute_rf_opt
    from qbiocode.learning.compute_mlp import compute_mlp, compute_mlp_opt
    from qbiocode.learning.compute_xgb import compute_xgb, compute_xgb_opt
    from qbiocode.learning.compute_pqk import compute_pqk, compute_pqk_opt
    from qbiocode.learning.compute_qpl import compute_qpl, compute_qpl_opt
    from qbiocode.learning.compute_qnn import compute_qnn, compute_qnn_opt
    from qbiocode.learning.compute_qsvc import compute_qsvc, compute_qsvc_opt
    from qbiocode.learning.compute_nb import compute_nb, compute_nb_opt
    from qbiocode.learning.compute_svc import compute_svc, compute_svc_opt
    from qbiocode.learning.compute_vqc import compute_vqc, compute_vqc_opt
    # TabPFN imports its own dependency lazily, so naming it here does not require
    # the optional [tabpfn] extra to be installed -- only *selecting* it does.
    from qbiocode.learning.compute_tabpfn import compute_tabpfn, compute_tabpfn_opt
    
    # Build model dictionary
    compute_ml_dict = {
        "svc_opt": compute_svc_opt,
        "svc": compute_svc,
        "dt_opt": compute_dt_opt,
        "dt": compute_dt,
        "lr_opt": compute_lr_opt,
        "lr": compute_lr,
        "nb_opt": compute_nb_opt,
        "nb": compute_nb,
        "rf_opt": compute_rf_opt,
        "rf": compute_rf,
        "xgb_opt": compute_xgb_opt,
        "xgb": compute_xgb,
        "catboost_opt": compute_catboost_opt,
        "catboost": compute_catboost,
        "tabpfn_opt": compute_tabpfn_opt,
        "tabpfn": compute_tabpfn,
        "mlp_opt": compute_mlp_opt,
        "mlp": compute_mlp,
        "qsvc": compute_qsvc,
        "qsvc_opt": compute_qsvc_opt,
        "vqc": compute_vqc,
        "vqc_opt": compute_vqc_opt,
        "qnn": compute_qnn,
        "qnn_opt": compute_qnn_opt,
        "pqk": compute_pqk,
        "pqk_opt": compute_pqk_opt,
        "qpl": compute_qpl,
        "qpl_opt": compute_qpl_opt,
    }

    quantum_models = QUANTUM_MODELS

    # Quantum models now have `_opt` twins, but they stay off unless asked for twice:
    # `grid_search: True` alone tunes only the classical models, exactly as before. A
    # quantum fit builds an n-by-n fidelity kernel by circuit simulation, so turning
    # tuning on for a quantum model multiplies its cost by the trial budget -- which
    # would have made every existing config that names one dramatically slower on
    # upgrade, with no change on the user's part.
    tune_quantum = bool(args.get("tune_quantum", False))

    # Validate the requested models before dispatching. An unknown name otherwise
    # reached `compute_ml_dict[method]` inside a joblib worker and came back as a
    # bare KeyError with no indication of what the valid names are.
    requested = list(args["model"])
    if not requested:
        raise ValueError(
            "args['model'] is empty; there is nothing to run. Choose at least one "
            f"of {sorted(compute_ml_dict)}."
        )
    unknown = [m for m in requested if m not in compute_ml_dict]
    if unknown:
        raise ValueError(
            f"Unknown model(s) {unknown} in args['model']. Available models: "
            f"{sorted(compute_ml_dict)} (quantum: {sorted(quantum_models)}). "
            f"Note the '_opt' variants are selected with args['grid_search'], not "
            f"by naming them here."
        )
    # A repeated name is rejected here rather than at the end. Two entries write the
    # same three columns ('results_<model>', 'y_test_<model>', 'y_predicted_<model>'),
    # so the `pd.melt(...).pivot(...)` fold-up at the bottom of this function had two
    # values for one index and died with "Index contains duplicate entries, cannot
    # reshape" -- AFTER every fit had run, naming neither the model nor args['model'].
    # There is no arrangement of the results in which a repeated name means anything,
    # so the config is simply wrong and can say so immediately.
    duplicated = sorted({name for name in requested if requested.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"Duplicate model(s) {duplicated} in args['model']; each model may be "
            f"named at most once. Every model writes one 'results_<model>' column, so "
            f"a repeat has nowhere to put its second result. Requested: {requested}."
        )
    grid_search = bool(args.get("grid_search", False))
    if grid_search:
        # Unreachable as the table stands -- every classical entry has an `_opt` twin --
        # and kept as a guard for the next learner added without one, which would
        # otherwise fail on `compute_ml_dict[method + "_opt"]` inside a joblib worker.
        missing_opt = [
            m for m in requested
            if m not in quantum_models and (m + "_opt") not in compute_ml_dict
        ]
        if missing_opt:
            raise ValueError(
                f"grid_search is enabled but {missing_opt} have no '_opt' "
                f"implementation. Disable grid_search or drop those models."
            )
        # A misspelt tuner would otherwise fall through to the `else` branch inside
        # every `_opt` function and run Optuna, so a config asking for the
        # exhaustive grid would silently not get it.
        missing_blocks = [
            m for m in requested
            if m in quantum_models
            and tune_quantum
            and not args.get("gridsearch_" + m + "_args")
        ]
        if missing_blocks:
            raise ValueError(
                f"tune_quantum is enabled but {missing_blocks} have no "
                f"'gridsearch_<model>_args' block naming what to search, so there is "
                f"nothing to tune. Add one per model, or drop tune_quantum to run them "
                f"at their configured hyperparameters."
            )
        # There is no exhaustive-grid engine for the quantum models: their `_opt`
        # wrappers score a whole compute function, not an estimator GridSearchCV could
        # drive. Asking for `tuner: grid` and getting Optuna anyway is the kind of
        # silent substitution that makes a result impossible to interpret later.
        if tune_quantum and args.get("tuner", "optuna") == "grid":
            quantum_requested = [m for m in requested if m in quantum_models]
            if quantum_requested:
                warnings.warn(
                    f"tuner: 'grid' applies to the classical models only. "
                    f"{quantum_requested} will still be tuned with Optuna -- a quantum "
                    f"candidate is scored by running the whole model, so there is no "
                    f"exhaustive-grid engine for them. Set tune_quantum: False to run "
                    f"them at their configured hyperparameters instead.",
                    UserWarning,
                    stacklevel=2,
                )
        tuner = args.get("tuner", "optuna")
        if tuner not in _TUNERS:
            raise ValueError(
                f"Unknown tuner {tuner!r} in args['tuner']. Choose one of "
                f"{sorted(_TUNERS)}: 'optuna' samples args['n_trials'] "
                f"configurations with Optuna, 'grid' fits every combination."
            )
        # Same reasoning as the tuner check: a misspelt metric would otherwise surface
        # inside every `_opt` call in a joblib worker, after the untuned models had run.
        from qbiocode.learning._tuning import tuning_metric

        tuning_metric(args)
    elif tune_quantum:
        raise ValueError(
            "tune_quantum is enabled but grid_search is not, so no tuning would run. "
            "Set grid_search: True as well, or drop tune_quantum."
        )
    if validation is not None:
        _check_fold_tuning(requested, compute_ml_dict, grid_search, tune_quantum, args)

    # Run classical and quantum models
    n_jobs = len(args["model"])
    if "n_jobs" in args.keys():
        n_jobs = min(args["n_jobs"], len(args["model"]))

    # Check if any quantum models are in the model list when grid_search is enabled
    if grid_search:
        quantum_in_models = [m for m in args["model"] if m in quantum_models]
        if quantum_in_models and not tune_quantum:
            print("\n" + "=" * 80)
            print("NOTE: Hyperparameter tuning is enabled, but not for these quantum",
                  "models:", quantum_in_models)
            print("=" * 80)
            print("They will run at their configured hyperparameters. Quantum tuning is")
            print("off by default because each trial is a quantum fit: on the simulator a")
            print("single QSVC fit builds an n-by-n fidelity kernel, so a 10-trial study")
            print("costs roughly ten ordinary runs of that model.")
            print("\nTo tune them with Optuna, set both keys and give each model a")
            print("gridsearch_<model>_args block:")
            print("    grid_search: True")
            print("    tune_quantum: True")
            print("    n_trials_quantum: 10")
            print("\nTo sweep them exhaustively instead, generate one config per")
            print("combination and compare across runs:")
            print("  from qbiocode.utils import generate_qml_experiment_configs")
            print("  num_configs, _ = generate_qml_experiment_configs(")
            print("      template_config_path='configs/config.yaml',")
            print("      output_dir='configs/qml_gridsearch',")
            print("      data_dirs=['data/your_data_dir']")
            print("  )")
            print("\nSee documentation: qbiocode.utils.generate_qml_experiment_configs")
            print("=" * 80 + "\n")

    def _model_args(method):
        """Per-model hyperparameters from the config, or the estimator defaults.

        `args[method + "_args"]` raised KeyError for any model whose config block
        was absent -- which includes ``xgb`` and ``qpl`` in the shipped
        config.yaml, so naming either in ``model`` failed before the estimator was
        ever constructed. The grid-search branch below already used ``.get(...,
        {})``; this makes the two agree, and says so in the log rather than
        substituting silently.
        """
        key = method + "_args"
        if key in args:
            return args[key]
        logger.info(
            "No %r block in the config; running %r with its default "
            "hyperparameters.", key, method,
        )
        return {}

    def _tuned_quantum_kwargs(method):
        """What to search, plus the run settings that are not candidates.

        ``gridsearch_<method>_args`` supplies the search space. Anything in
        :data:`_QUANTUM_PASSTHROUGH` is copied over from ``<method>_args`` as well, so a
        setting that selects *which models run* is not lost by turning tuning on; see
        that constant for why ``classical_models`` is the one key that needs it.
        """
        kwargs = dict(args.get("gridsearch_" + method + "_args", {}))
        model_args = args.get(method + "_args") or {}
        for key in _QUANTUM_PASSTHROUGH.get(method, ()):
            # `not in` rather than unconditional: a value already in the gridsearch block
            # was written there deliberately and is what the user is looking at.
            if key not in kwargs and key in model_args:
                kwargs[key] = model_args[key]
        return kwargs

    def _seeded_kwargs(compute_fn, model_kwargs):
        """Fill in ``random_state`` from ``args['seed']`` wherever an estimator takes one.

        Two runs at the same seed used to disagree on the decision-tree rows.
        ``DecisionTreeClassifier`` at ``random_state=None`` permutes the features
        before choosing a split, so a tie between two equally-good splits broke
        one way or the other at random; on a 60-sample dataset that moved
        accuracy by a whole test sample (0.889 vs 0.944). The same applies to
        every other estimator here that draws from a global RNG: random forests,
        the MLP's weight init, XGBoost's row subsampling, SVC's probability
        calibration.

        A ``random_state`` already present in the config wins -- this only fills
        the gap. Functions that take no ``random_state`` (naive Bayes) are left
        alone.
        """
        # Drop keys that model_run itself passes explicitly at every call site below.
        # A config block is splatted as **kwargs into the same namespace as those fixed
        # arguments, so any overlap is a TypeError raised while the delayed() list is
        # being BUILT -- before Parallel runs, so it takes down all 13 models having fit
        # nothing, not just the one model that owns the key. `verbose` was live: every
        # pilot config carries 'verbose' in catboost_args and gridsearch_catboost_args
        # (it is a real CatBoost parameter), which produced
        #   TypeError: _call_with_global_seeds() got multiple values for keyword
        #              argument 'verbose'
        # CatBoost's own training chatter is already silenced by `_QUIET` in
        # compute_catboost, and QBioCode's `verbose` selects the result summary -- a
        # different thing, as that module's comment says -- so the config key is
        # redundant here and the explicit value must win. Warn rather than drop
        # silently, so a key that was meant to do something is not simply ignored.
        model_kwargs = dict(model_kwargs)
        for reserved in _RESERVED_KWARGS:
            if reserved in model_kwargs:
                logger.warning(
                    "ignoring %r from this model's config block: model_run passes it "
                    "explicitly, and duplicating it raises TypeError before any model "
                    "runs. Remove it from the config to silence this.", reserved
                )
                del model_kwargs[reserved]

        seed = args.get("seed")
        if seed is None or "random_state" in model_kwargs:
            return model_kwargs
        try:
            takes_random_state = "random_state" in inspect.signature(compute_fn).parameters
        except (TypeError, ValueError):  # pragma: no cover - C callables
            return model_kwargs
        if not takes_random_state:
            return model_kwargs
        return {**model_kwargs, "random_state": seed}

    seed = args.get("seed")
    q_seed = args.get("q_seed")

    def _fold_kwargs(method, quantum):
        """The extra ``_opt`` keywords of a fold-based run; none without a validation split."""
        if validation is None:
            return {}
        kwargs = {
            "validation": validation,
            "default_params": _default_params(
                compute_ml_dict[method], args.get(method + "_args")
            ),
        }
        if quantum:
            kwargs["reseed"] = _Reseed(seed, q_seed)
        return kwargs

    if grid_search:
        results = []
        for method in args["model"]:
            if method in quantum_models and tune_quantum:
                # Tuned like the classical models, from the same
                # `gridsearch_<model>_args` block, but on its own budget and without
                # `cv`/`tuner`: a quantum candidate is scored on one stratified holdout
                # rather than k folds, and there is no exhaustive-grid engine to select.
                compute_fn = compute_ml_dict[method + "_opt"]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    data_key=data_key,
                    # One budget for every arm in a fold-based run; see
                    # _check_fold_tuning for the warning when the two keys differ.
                    n_trials=(
                        args.get("n_trials_quantum", 10) if validation is None
                        else args.get("n_trials", 50)
                    ),
                    validation_split=args.get("validation_split", 0.25),
                    **_seeded_kwargs(
                        compute_fn, _tuned_quantum_kwargs(method)
                    ),
                    **_fold_kwargs(method, quantum=True),
                    verbose=False,
                )
            elif method in quantum_models:
                # Untuned: run at the configured hyperparameters, as before.
                compute_fn = compute_ml_dict[method]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method,
                    data_key=data_key,
                    **_seeded_kwargs(compute_fn, args.get(method + "_args", {})),
                    verbose=False,
                )
            else:
                # Classical models have _opt versions with grid search
                compute_fn = compute_ml_dict[method + "_opt"]
                result = delayed(_call_with_global_seeds)(
                    compute_fn,
                    seed,
                    q_seed,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    args,
                    model=method + "_opt",
                    # `args["cross_validation"]` was an unguarded lookup on this
                    # branch only, so tuning a model from a config that omitted the
                    # key died here rather than at validation.
                    cv=args.get("cross_validation", 5),
                    tuner=args.get("tuner", "optuna"),
                    n_trials=args.get("n_trials", 50),
                    **_seeded_kwargs(
                        compute_fn, args.get("gridsearch_" + method + "_args", {})
                    ),
                    **_fold_kwargs(method, quantum=False),
                    verbose=False,
                )
            results.append(result)
        results = Parallel(n_jobs=n_jobs)(results)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_call_with_global_seeds)(
                compute_ml_dict[method],
                seed,
                q_seed,
                X_train,
                X_test,
                y_train,
                y_test,
                args,
                model=method,
                data_key=data_key,
                **_seeded_kwargs(compute_ml_dict[method], _model_args(method)),
                verbose=False,
            )
            for method in args["model"]
        )

    model_total_result = pd.melt(pd.concat(results)).dropna()  # type: ignore
    model_total_result["i"] = 0
    model_total_result = model_total_result.pivot(columns="variable", values="value", index="i")
    return model_total_result.to_dict()
