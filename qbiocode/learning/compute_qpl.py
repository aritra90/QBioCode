# ====== Base class imports ======
import hashlib
import os
import logging
import math
import time
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Deliberately broad: a missing xgboost raises ImportError, but an xgboost whose
# native library cannot load raises OSError (no libomp on macOS) or
# xgboost.core.XGBoostError, which subclasses ValueError -- narrowing to
# ImportError here would let those escape as an unhandled error at import time.
# The reason is kept so the messages below can quote the actual failure instead
# of guessing at it.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGBOOST_ERROR = None
except Exception as exc:
    XGBOOST_AVAILABLE = False
    _XGBOOST_ERROR = str(exc)
    XGBClassifier = None  # type: ignore

# Same broad guard, same reason: catboost's failure mode when its native extension
# cannot load is an OSError rather than an ImportError.
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    _CATBOOST_ERROR = None
except Exception as exc:  # noqa: BLE001 -- see above
    CATBOOST_AVAILABLE = False
    _CATBOOST_ERROR = str(exc)
    CatBoostClassifier = None  # type: ignore

# Cap on the outer parallelism of the RandomizedSearchCV heads below. `n_jobs=-1` asks
# joblib for one worker per core, and every worker is a fresh process that re-imports
# xgboost and catboost. CatBoost's extension module is 264 MB; read from a network
# filesystem that import costs ~17 s on its own, so past a handful of workers startup
# dominates and the workers contend for the same file. Measured here on a 128-core host,
# the catboost head's 200-candidate search (n_iter=40, cv=5) over 30x9 data took 12.4 s
# at n_jobs=1, 11.1 s at 4 and 13.4 s at 8, but did not finish in 8 minutes at -1 -- and
# with the per-fit thread pins below removed it was killed outright by the OOM reaper.
# The searches are small and the work per candidate is milliseconds, so the parallelism
# beyond a few workers buys nothing that the startup cost does not take back.
_SEARCH_N_JOBS = min(os.cpu_count() or 1, 8)

logger = logging.getLogger(__name__)

#: How many candidates each searched head's ``RandomizedSearchCV`` draws, and over how
#: many folds. Named rather than repeated per factory: the seven factories below used to
#: spell ``n_iter=40, cv=5`` individually, so a head could silently drift from its
#: siblings and a QPL run would be comparing heads on unequal budgets.
HEAD_N_ITER = 40
HEAD_CV = 5

#: The downstream classical models QPL can fit on a quantum projection, in the order the
#: default head list runs them. ``'tabpfn'`` is selectable but deliberately not a
#: default; see :func:`compute_qpl`.
#:
#: This is the one place the set is written down. The dispatch used to be a seven-branch
#: ``if/elif`` chain inside the head loop, with the valid names repeated as prose in two
#: error messages and the default list repeated again -- four copies that had already
#: drifted (the "Valid options" text named every head, including ones a given install
#: cannot provide).
QPL_HEADS = ("rf", "mlp", "svc", "lr", "xgb", "catboost", "tabpfn")

#: What ``classical_models`` means when it is not given. Every head but TabPFN, which
#: needs the optional ``[tabpfn]`` extra -- defaulting it on would make an ordinary
#: install warn on every QPL run. Unchanged from the list this function has always used.
QPL_DEFAULT_HEADS = ("rf", "mlp", "svc", "lr", "xgb", "catboost")

# from qiskit.primitives import Sampler
from functools import reduce

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import (
    RESULTS_PREFIX,
    extract_binary_scores,
    modeleval,
)
from qbiocode.learning.compute_pqk import _resolve_data_map
from qbiocode.learning._grid import build_param_grid, to_plain
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
    seed_from,
    tuning_metric,
)
from qbiocode.learning.compute_fold import (
    fold_fixed,
    fold_svc_max_iter,
    function_param_names,
    head_scorer,
)

# Imported for its availability probe and lazy loader rather than for the estimator
# itself: `tabpfn_is_available` uses importlib.util.find_spec, so asking whether the
# optional extra is present costs neither the tabpfn import nor torch's OpenMP
# runtime. See qbiocode.learning.compute_tabpfn.
from qbiocode.learning.compute_tabpfn import (
    TABPFN_DEFAULT_VERSION,
    TABPFN_MAX_CLASSES,
    _explain_weight_access_failure,
    _load_tabpfn_classifier,
    resolve_model_path,
    tabpfn_is_available,
)


def _resolve_heads(classical_models):
    """Validate and normalise the requested downstream models.

    Accepts one name or several, so a config may write ``classical_models: svc`` as
    well as ``['svc']``. Checked up front, before the quantum projection is computed:
    an unknown name used to be warned about and skipped *inside* the head loop, which
    meant a typo in the only requested head paid for the whole projection and then
    raised "none of the requested classical models could be fitted", naming the typo
    only in passing.

    Args:
        classical_models (list, tuple, str or None): What the caller asked for. None
            means :data:`QPL_DEFAULT_HEADS`.

    Returns:
        list: The head names, in the order given.

    Raises:
        ValueError: If the list is empty, names an unknown head, or repeats one.
    """
    if classical_models is None:
        return list(QPL_DEFAULT_HEADS)
    # to_plain first: these come off OmegaConf, where a YAML list is a ListConfig that
    # is not a `list`. A bare string is a one-head selection, not a sequence of letters.
    classical_models = to_plain(classical_models)
    if isinstance(classical_models, str):
        classical_models = [classical_models]
    if not isinstance(classical_models, (list, tuple, set, frozenset)):
        raise ValueError(
            f"classical_models must be a head name or a list of them, got "
            f"{classical_models!r}. Valid heads: {list(QPL_HEADS)}."
        )
    # A set is accepted (the head loop has always iterated whatever it was handed) but
    # sorted, because set iteration order is not the insertion order and the heads decide
    # the order of the result rows. An unordered selection should at least be stable.
    if isinstance(classical_models, (set, frozenset)):
        classical_models = sorted(classical_models)
    heads = [str(head) for head in classical_models]
    if not heads:
        raise ValueError(
            f"classical_models is empty, so QPL has no downstream model to fit on the "
            f"quantum projection. Name at least one of {list(QPL_HEADS)}, or leave it "
            f"unset for the default {list(QPL_DEFAULT_HEADS)}."
        )
    unknown = [head for head in heads if head not in QPL_HEADS]
    if unknown:
        raise ValueError(
            f"Unknown QPL downstream model(s) {unknown} in classical_models. Valid "
            f"heads: {list(QPL_HEADS)}. The default is {list(QPL_DEFAULT_HEADS)}; "
            f"'tabpfn' is selectable but needs the optional [tabpfn] extra."
        )
    # A repeat has nowhere to put its second result: every head writes one
    # `results_<model>_<head>` column, and model_run's melt/pivot fold-up raises
    # "Index contains duplicate entries, cannot reshape" on a duplicate -- after the
    # projection and every head search had run. Same reasoning as model_run applies to
    # args['model'].
    duplicated = sorted({head for head in heads if heads.count(head) > 1})
    if duplicated:
        raise ValueError(
            f"Duplicate QPL downstream model(s) {duplicated} in classical_models; each "
            f"may be named at most once, because each writes one results column. "
            f"Requested: {heads}."
        )
    return heads


def _resolve_head_args(classical_models_args, heads):
    """Validate the per-head override blocks and drop the ones no head will read.

    Args:
        classical_models_args (Mapping or None): ``{head: {param: values}}``.
        heads (list): The selected heads, for the unused-block warning.

    Returns:
        dict: head -> its override mapping, for the selected heads only.

    Raises:
        ValueError: If a key is not a known head, or a block is not a mapping.
    """
    if not classical_models_args:
        return {}
    blocks = to_plain(classical_models_args)
    if not isinstance(blocks, Mapping):
        raise ValueError(
            f"classical_models_args must map a head name to its settings, as in "
            f"{{'svc': {{'C': [1, 10]}}}}, got {blocks!r}."
        )
    unknown = sorted(set(blocks) - set(QPL_HEADS))
    if unknown:
        raise ValueError(
            f"classical_models_args names unknown QPL downstream model(s) {unknown}. "
            f"Valid heads: {list(QPL_HEADS)}."
        )
    for head, block in blocks.items():
        if not isinstance(block, Mapping):
            raise ValueError(
                f"classical_models_args[{head!r}] must be a mapping of settings, got "
                f"{block!r}. For a searched head it replaces that head's "
                f"hyperparameter grid, as in {{'C': [1, 10], 'kernel': ['rbf']}}."
            )
        # Build each searched head's grid now and throw the result away, so a malformed
        # one -- a `{low, high}` range, which a RandomizedSearchCV cannot sample -- is
        # reported before the quantum projection is computed rather than after. Without
        # this the same ValueError still arrived, but from inside the head loop, having
        # paid for the projection first. TabPFN's block is constructor settings rather
        # than a grid, so there is nothing to build for it.
        if head != "tabpfn" and block:
            build_param_grid(f"qpl head {head!r}", block)
    # A block for a head that is not running is silently dead otherwise, and the usual
    # cause is a head missing from `classical_models` -- worth a line, not an error.
    unused = sorted(set(blocks) - set(heads))
    if unused:
        logger.info(
            "classical_models_args has blocks for %s, which are not in classical_models "
            "%s, so they are not used.", unused, heads,
        )
    return {head: dict(blocks[head]) for head in heads if head in blocks}


def compute_qpl(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="qpl",
    data_key="",
    verbose=False,
    encoding="Z",
    primitive="estimator",
    entanglement="linear",
    reps=2,
    classical_models=None,
    data_map="unit",
    bandwidth=1.0,
    *,
    classical_models_args=None,
    head_scoring=None,
    head_max_iter=None,
):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits,
    and evaluates the performance of classical machine learning models on the projected data.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features. The classical models are trained on the projected training data and
    evaluated on the projected test data. The function returns evaluation metrics and model parameters.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. The results are saved to files for training and test projections, which are reused
    if they already exist to avoid redundant computations.
    This function is part of the main quantum machine learning pipeline (QProfiler.py) and is intended for use in supervised learning tasks.
    It leverages quantum computing to enhance feature extraction and classification performance on complex datasets.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    QPL is the generalised PQK: both project the rows through the same quantum feature
    map, and where :func:`qbiocode.learning.compute_pqk.compute_pqk` then fits one SVC,
    this fits a *configurable* set of downstream classical models and reports one row per
    model, labelled ``<model>_<head>`` (``qpl_svc``, ``qpl_rf``, ``qpl_tabpfn``, ...).

    **Two distinct optimisations, which should not be confused.** The projection itself
    -- ``encoding``, ``reps``, ``entanglement``, ``data_map``, ``bandwidth`` -- is *not*
    optimised here; this function runs it at whatever it is given.
    :func:`compute_qpl_opt` is what searches those, with Optuna, scoring a candidate
    projection by the mean tuning metric across the heads. Independently of that, each
    downstream model *is* hyperparameter-optimised here, inside this one fit: every head
    but TabPFN is a ``RandomizedSearchCV`` over its own model-specific grid (see
    :func:`build_head`). So an untuned QPL run still tunes its downstream models; what
    ``compute_qpl_opt`` adds is a search over the projection.

    **The projection is computed once and reused by every head.** The expensive part --
    one circuit simulation per row -- happens before the head loop is entered, and the
    loop reads the resulting ``projections_train``/``projections_test`` arrays. Adding a
    head costs one classical search, never another projection::

                            quantum projection (once)
                                      |
                       ---------------+---------------
                       |              |              |
                      SVC            RF            TabPFN
                       |              |              |
                      tune           tune        (no search)
                       |              |              |
                    evaluate       evaluate       evaluate

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        y_train (np.ndarray): Training data labels.
        y_test (np.ndarray): Test data labels.
        args (dict): Arguments containing backend and other configurations.
        model (str): Model type, default is 'QPL'.
        data_key (str): Key for the dataset, default is ''.
        verbose (bool): If True, print additional information, default is False.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.
        classical_models (list or str): Which downstream classical models to fit on the
            quantum projection -- one name or several, from :data:`QPL_HEADS`
            (``'rf'``, ``'mlp'``, ``'svc'``, ``'lr'``, ``'xgb'``, ``'catboost'``,
            ``'tabpfn'``). A bare string is accepted as a one-model list, so
            ``classical_models: svc`` and ``['svc']`` mean the same thing. Each selected
            model gets its own ``results_<model>_<head>`` row, and the projection is
            computed once for all of them. Defaults to :data:`QPL_DEFAULT_HEADS`,
            ``['rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost']``. ``'tabpfn'`` is accepted
            but deliberately absent from the default: it needs the optional ``[tabpfn]``
            extra, so defaulting it on would make every QPL run warn in an ordinary
            install. Name it explicitly to use it. It needs no API token -- QBioCode
            pins the ungated ``v2`` weights.
        classical_models_args (dict or None): Per-head overrides, ``{head: {...}}``.
            Keyword-only. For a searched head the mapping *replaces* that head's default
            hyperparameter grid, validated exactly as a ``gridsearch_<model>_args`` block
            is (:func:`qbiocode.learning._grid.build_param_grid`), so a list is a list of
            values to try, a bare scalar is a one-value list, and ``None``/``[]`` leaves
            the estimator's own default in force. For the TabPFN head, which has no
            grid, the mapping is its constructor settings (``model_version``,
            ``n_estimators``, ...). A head not named here keeps its default grid, so
            omitting this argument reproduces the shipped behaviour exactly.
        data_map (str or bool): How features become gate angles, as in
            :func:`qbiocode.learning.compute_pqk.compute_pqk`. ``'unit'`` (the default,
            and QPL's historical map) keeps every multiplicative factor of a feature at
            1.0; ``'qiskit'`` uses qiskit's default ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)``,
            the map the ``eng_zz``/``qlab_zz`` generators use. ``True``/``False`` are
            accepted as ``'unit'``/``'qiskit'``.
        bandwidth (float): Scale applied to the features before the feature map
            (:func:`qbiocode.utils.qutils.apply_bandwidth`). Default 1.0, the unscaled
            map; any other value is part of the projection cache key.
        head_scoring (str or None): Metric every searched head's ``RandomizedSearchCV``
            (:data:`HEAD_N_ITER` candidates x :data:`HEAD_CV` folds on the projections)
            picks its config by. None (the
            default) is sklearn's, each estimator's ``score`` (accuracy).
            ``compute_qpl_opt`` sets the run's tuning metric under ``split_mode:
            manifest``. The head searches are part of this one fit -- one trial of an
            outer study -- not extra tuning budget. The bare TabPFN head has no search.
        head_max_iter (int or None): libsvm iteration cap of the SVC head's fits. None
            (the default) is libsvm's -1, no cap. No other head reads it.

    Returns:
        modeleval (pd.DataFrame): A DataFrame containing evaluation metrics and model
        parameters for all models -- one ``results_<model>_<head>`` column per selected
        downstream model.

    Raises:
        ValueError: If ``data_map`` is not ``'unit'``, ``'qiskit'`` or a bool; if
            ``classical_models`` names an unknown model, is empty, or repeats a name; if
            ``classical_models_args`` names an unknown model or is malformed; or if no
            selected model could be fitted.
    """

    # Set default classical models if not provided
    classical_models = _resolve_heads(classical_models)
    # Validated here, with the head list, so a typo in a per-head block is reported
    # before the projection is computed rather than after -- the projection is the
    # expensive half, and an unknown head name is not worth paying for it to discover.
    classical_models_args = _resolve_head_args(classical_models_args, classical_models)

    # Checked before the projection directory is created, so a typo leaves no trace.
    data_map = _resolve_data_map(data_map)
    # Before the cache key and the feature map read the rows. 1.0 returns them unchanged.
    X_train, X_test = qutils.apply_bandwidth(bandwidth, X_train, X_test)

    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    # The projection cache used to be keyed on `data_key` alone, in a hardcoded
    # "qpl_projections" directory. Two consequences, both silent:
    #
    #   * Changing `encoding`, `entanglement`, `reps` or `primitive` and rerunning reused
    #     the projection computed for the *previous* settings, so the new circuit was
    #     never run and the reported result described the old one. Tuning made this acute
    #     -- every trial after the first would have scored the same cached projection, so
    #     the search would have compared a hyperparameter against itself.
    #   * A different split of the same dataset (an inner validation split, say) matched
    #     the same file name and was loaded at the wrong length, surfacing downstream as
    #     `ValueError: Found input variables with inconsistent numbers of samples`, which
    #     names neither the cache nor the file.
    #
    # Both are fixed the way compute_pqk already handles it: fingerprint the settings
    # that change the circuit into the file name, validate the row count on load, and let
    # the directory be redirected so throwaway projections stay out of the real cache.
    projection_dir = os.path.expanduser(args.get("qpl_projection_dir", "qpl_projections"))
    os.makedirs(projection_dir, exist_ok=True)

    # `projection_backend` joins the fingerprint ONLY when it is set. It has to be in
    # there for a head-to-head: without it the second and third backends load the first
    # one's cached projection, which makes their timings meaningless and their agreement
    # tautological. But appending a bare `None` would change every legacy hash too, and
    # that silently orphans the projection caches already on disk (88 .npy files ship in
    # this repo alone) -- a slow surprise, not a wrong answer, but avoidable.
    fingerprint_parts = (encoding, entanglement, reps, primitive, feat_dimension)
    _projection_backend = args.get("projection_backend")
    if _projection_backend:
        fingerprint_parts = fingerprint_parts + (_projection_backend,)
    # `data_map` joins only when it is not the default, for the same reason: 'unit' keys
    # stay byte-identical, and a 'qiskit' projection can never load a 'unit' file.
    if data_map != "unit":
        fingerprint_parts = fingerprint_parts + (f"data_map={data_map}",)
    if float(bandwidth) != 1.0:
        fingerprint_parts = fingerprint_parts + (f"bandwidth={float(bandwidth)!r}",)
    # The data itself, plus the two settings that change the numbers a projection holds
    # without changing the circuit. Without them the key names the feature map and the
    # dataset but never the rows, so a different fold of the same dataset, a regenerated
    # CSV under an unchanged name, or a different shot count all reach the same file -- and
    # the row-count/width validation below cannot reject any of them, because those cases
    # share a shape. This does orphan projections cached under the older key, which costs a
    # recompute rather than correctness; only the tutorial trees hold any.
    fingerprint_parts = fingerprint_parts + (
        qutils.dataset_fingerprint(X_train, X_test),
        args.get("shots"),
        args.get("backend"),
    )
    feature_map_fingerprint = hashlib.sha256(
        repr(fingerprint_parts).encode()
    ).hexdigest()[:10]

    file_projection_train = os.path.join(
        projection_dir,
        "qpl_projection_" + data_key + "_" + feature_map_fingerprint + "_train.npy",
    )
    file_projection_test = os.path.join(
        projection_dir,
        "qpl_projection_" + data_key + "_" + feature_map_fingerprint + "_test.npy",
    )

    # Each file paired with the rows it holds, so the loops below never have to work out
    # which split a path belongs to. They used to do that with `"train" in f_tr` -- on
    # the WHOLE path, directory included -- so any projection directory whose path
    # contained "train" (`~/runs/trained/`, a pytest tmp_path for a test whose name says
    # "train", a dataset directory called "training_data") made the *test* file match the
    # train branch and be filled with the TRAINING rows. The result is not a crash where
    # the two splits differ in size -- it surfaces much later and somewhere else, as
    # `ValueError: Found input variables with inconsistent numbers of samples` from
    # inside a head's fit, naming neither the cache nor the path that caused it. Where
    # they happen to be the same size it is worse: no error at all, and every head
    # silently scored on its own training rows.
    projection_targets = (
        (file_projection_train, X_train),
        (file_projection_test, X_test),
    )

    def _validate_projection_file(path, expected_len):
        """Refuse a cached projection whose row count does not match the current data."""
        if not os.path.exists(path):
            return
        cached = np.load(path, allow_pickle=False)
        if len(cached) != expected_len:
            raise ValueError(
                f"Projection file {path} has {len(cached)} rows, but the current dataset "
                f"expects {expected_len} rows. Remove this projection file or use a "
                f"different qpl_projection_dir."
            )

    _validate_projection_file(file_projection_train, len(X_train))
    _validate_projection_file(file_projection_test, len(X_test))

    #  This function ensures that all multiplicative factors of data features inside single qubit gates are 1.0
    def data_map_func(x: np.ndarray):
        """
        Define a function map from R^n to R.

        Args:
            x: data

        Returns:
            the mapped value (float or Parameter expression)
        """
        coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
        # Check if coeff is a numeric type before converting to float
        # If it's a Parameter expression, return it as-is for Qiskit to handle
        try:
            return float(coeff)
        except (TypeError, ValueError):
            # If conversion fails, it's likely a Parameter expression
            return coeff

    # 'qiskit' passes no data map, so the feature map falls back to qiskit's default.
    data_map_func = data_map_func if data_map == "unit" else None

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding,
        feat_dimension=X_train.shape[1],
        reps=reps,
        entanglement=entanglement,
        data_map_func=data_map_func,
    )

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    if (not os.path.exists(file_projection_train)) | (not os.path.exists(file_projection_test)):

        projection_backend = args.get("projection_backend")
        if projection_backend:
            # Local simulator path: one state preparation per row, all 3n expectation
            # values read off it. The branch below instead calls StatevectorEstimator,
            # which re-simulates the circuit once PER OBSERVABLE -- a ~3n-fold overhead
            # unrelated to simulation method. Kept as the default only for hardware and
            # for backwards compatibility.
            if args["backend"] != "simulator":
                raise ValueError(
                    f"projection_backend={projection_backend!r} runs on a local "
                    f"simulator, but backend={args['backend']!r} selects remote or "
                    f"noisy execution. Set backend: 'simulator', or remove "
                    f"projection_backend to use the runtime primitive."
                )
            from qbiocode.utils.projection import make_projector

            projector = make_projector(
                feat_dimension, encoding=encoding, reps=reps,
                entanglement=entanglement, backend=projection_backend,
                data_map_func=data_map_func,
            )
            for f_tr, split in projection_targets:
                if os.path.exists(f_tr):
                    continue
                dat = split.copy()
                projections = projector.project(
                    dat, progress_every=100, n_jobs=args.get("projection_n_jobs", 1)
                )
                # Reverse the qubit axis. The legacy path builds its observables as
                # Pauli(id[:i] + "X" + id[i+1:]), indexing by STRING POSITION -- and in a
                # qiskit Pauli label the rightmost character is qubit 0, so its
                # observables_x[i] is X on qubit n-1-i. projection.py indexes by qubit
                # number. Without this flip the projected columns come out mirrored:
                # numerically self-consistent, identical downstream accuracy (a fixed
                # permutation of features), but NOT byte-identical to previously cached
                # projections or to a legacy-path run -- so `projection_backend` would
                # silently stop being a drop-in. Verified to give exactly 0.0 difference.
                np.save(f_tr, projections[:, :, ::-1])
            # Truncation is silent, so surface it rather than leaving it to be noticed
            # in the accuracy numbers. None means the backend does not report it.
            fidelity = projector.fidelity_estimate()
            if fidelity is not None and fidelity < 0.999:
                warnings.warn(
                    f"projection_backend={projection_backend!r} truncated the state: "
                    f"fidelity estimate {fidelity:.4g} (1.0 = exact). These projected "
                    f"features are approximate and any metric computed from them "
                    f"should be reported as such.",
                    RuntimeWarning,
                )
            session = None
        else:
            #  Generate the backend, session and primitive
            backend, session, prim = qutils.get_backend_session(
                args, "estimator", num_qubits=num_qubits
            )

            # Transpile
            if args["backend"] != "simulator":
                circuit = qutils.transpile_circuit(
                    circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
                )


            # Set the global phase to 0 to avoid header size issues
            circuit.global_phase = 0
        
            for f_tr, split in projection_targets:
                if not os.path.exists(f_tr):
                    projections = []
                    dat = split.copy()

                    # Identity operator on all qubits
                    id = "I" * feat_dimension

                    # We group all commuting observables
                    # These groups are the Pauli X, Y and Z operators on individual qubits
                    # Apply the circuit layout to the observable if mapped to device
                    if args["backend"] != "simulator":
                        # num_qubits comes from the CIRCUIT, not the backend. On a device the two agree:
                        # transpiling against a 127-qubit backend returns a 127-qubit circuit, and the
                        # observables built either way are byte-identical (checked on FakeManilaV2 and
                        # FakeSherbrooke). On Aer they do not agree -- AerSimulator reports num_qubits=63,
                        # a memory-derived capacity rather than a device width, while transpiling leaves
                        # the circuit at its own width and sets circuit.layout to None. Laying the Pauli
                        # out onto 63 qubits raised, for every estimator-primitive model reached through
                        # backend: 'mps_simulator':
                        #   ValueError: The number of qubits of the circuit (10) does not match the
                        #              number of qubits of the (0,)-th observable (63).
                        # With layout None and num_qubits == circuit.num_qubits the call is the identity,
                        # so this branch now agrees with the else branch on Aer and is unchanged on hardware.
                        observables_x = []
                        observables_y = []
                        observables_z = []
                        for i in range(feat_dimension):
                            observables_x.append(
                                Pauli(id[:i] + "X" + id[(i + 1) :]).apply_layout(
                                    circuit.layout, num_qubits=circuit.num_qubits
                                )
                            )
                            observables_y.append(
                                Pauli(id[:i] + "Y" + id[(i + 1) :]).apply_layout(
                                    circuit.layout, num_qubits=circuit.num_qubits
                                )
                            )
                            observables_z.append(
                                Pauli(id[:i] + "Z" + id[(i + 1) :]).apply_layout(
                                    circuit.layout, num_qubits=circuit.num_qubits
                                )
                            )
                    else:
                        observables_x = [
                            Pauli(id[:i] + "X" + id[(i + 1) :]) for i in range(feat_dimension)
                        ]
                        observables_y = [
                            Pauli(id[:i] + "Y" + id[(i + 1) :]) for i in range(feat_dimension)
                        ]
                        observables_z = [
                            Pauli(id[:i] + "Z" + id[(i + 1) :]) for i in range(feat_dimension)
                        ]

                    # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
                    # of datapoint i on qubit k
                    projections = []

                    for i in range(len(dat)):
                        if i % 100 == 0:
                            print(f"at datapoint {str(i)}")

                        # Get training sample
                        parameters = dat[i]

                        # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                        # set of observables and the circuit parameters
                        pub_x = (circuit, observables_x, parameters)
                        pub_y = (circuit, observables_y, parameters)
                        pub_z = (circuit, observables_z, parameters)

                        job = prim.run([pub_x, pub_y, pub_z])
                        job_result_x = job.result()[0].data.evs
                        job_result_y = job.result()[1].data.evs
                        job_result_z = job.result()[2].data.evs

                        # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                        projections.append([job_result_x, job_result_y, job_result_z])
                    np.save(f_tr, projections)

        if not isinstance(session, type(None)):
            session.close()

    # Load computed projections
    projections_train = np.load(file_projection_train)
    projections_train = np.array(projections_train).reshape(len(projections_train), -1)
    projections_test = np.load(file_projection_test)
    projections_test = np.array(projections_test).reshape(len(projections_test), -1)

    # Check if XGBoost is requested but not available
    if "xgb" in classical_models and not XGBOOST_AVAILABLE:
        warnings.warn(
            "XGBoost is not properly installed or configured and will be skipped.\n"
            f"Error: {_XGBOOST_ERROR}\n"
            "On macOS, you may need to install OpenMP:\n"
            "  brew install libomp\n"
            "Then reinstall XGBoost:\n"
            "  pip install --force-reinstall xgboost\n"
            "See installation documentation for more details.\n"
            f"Continuing with other models: {[m for m in classical_models if m != 'xgb']}",
            UserWarning,
        )
        # Remove xgb from the list
        classical_models = [m for m in classical_models if m != "xgb"]

    # Same warn-and-drop treatment for catboost: one unusable head should cost that
    # head, not the whole quantum projection that has already been computed.
    if "catboost" in classical_models and not CATBOOST_AVAILABLE:
        warnings.warn(
            "CatBoost is not properly installed or configured and will be skipped.\n"
            f"Error: {_CATBOOST_ERROR}\n"
            "CatBoost is a core dependency, so this is a broken install; reinstall with:\n"
            "  pip install --force-reinstall catboost\n"
            f"Continuing with other models: {[m for m in classical_models if m != 'catboost']}",
            UserWarning,
        )
        classical_models = [m for m in classical_models if m != "catboost"]

    # TabPFN is an optional extra, so its absence is an ordinary configuration state
    # rather than a broken install -- the message says how to add it and moves on.
    if "tabpfn" in classical_models and not tabpfn_is_available():
        warnings.warn(
            "TabPFN is not installed and will be skipped as a QPL head.\n"
            'Install it with: pip install "qbiocode[tabpfn]"\n'
            f"Continuing with other models: {[m for m in classical_models if m != 'tabpfn']}",
            UserWarning,
        )
        classical_models = [m for m in classical_models if m != "tabpfn"]

    # TabPFN's pretrained head cannot represent more than ten classes, and unlike the
    # row and feature limits that one is not waivable. Checked here so an unsuitable
    # dataset drops the head with an explanation instead of failing the run.
    if "tabpfn" in classical_models:
        n_classes = len(np.unique(np.asarray(y_train)))
        if n_classes > TABPFN_MAX_CLASSES:
            warnings.warn(
                f"TabPFN supports at most {TABPFN_MAX_CLASSES} classes but this target has "
                f"{n_classes}, so it will be skipped as a QPL head.\n"
                f"Continuing with other models: "
                f"{[m for m in classical_models if m != 'tabpfn']}",
                UserWarning,
            )
            classical_models = [m for m in classical_models if m != "tabpfn"]

    # If no models remain after filtering, raise an error. Unlike the up-front
    # validation in `_resolve_heads`, reaching this means every NAMED head was dropped
    # as unavailable, so the list is the thing to report rather than the valid names.
    if not classical_models:
        raise ValueError(
            f"Every requested QPL downstream model was dropped as unavailable, so there "
            f"is nothing to fit on the quantum projection. See the warnings above for "
            f"why each was skipped. Valid heads: {list(QPL_HEADS)}."
        )

    # `estimator`, not `model`, inside this loop. It used to rebind `model` -- the label
    # parameter -- to each head's estimator, so the label was destroyed on the first
    # iteration. That is why the results label below was hardcoded to "qpl_" + head:
    # there was nothing left to read it from. Same shadowing bug as compute_pqk had.
    model_res = []
    # None unless compute_qpl_opt asked for the tuning metric: RandomizedSearchCV's own
    # default, so an internal-mode head search is unchanged.
    scoring = head_scorer(head_scoring, args)
    # Every head reads the SAME `projections_train`/`projections_test` computed above.
    # The loop builds and fits one downstream model per iteration and never touches the
    # projection path, which is what makes a multi-head run cost one projection.
    for method in classical_models:
        # One dispatch for every head, in place of the seven-branch if/elif this used to
        # be: `build_head` owns which estimator and which model-specific grid each name
        # means, so a head cannot be added to QPL_HEADS and then silently fall through
        # to the unknown-name branch. `max_iter` is read by the svc head alone and
        # `search_args` is empty unless the config overrode that head's grid.
        estimator = build_head(
            method,
            args["seed"],
            scoring=scoring,
            max_iter=head_max_iter,
            search_args=classical_models_args.get(method),
        )

        # Built from the `model` label rather than hardcoded to "qpl_". The hardcoded
        # form threw away the argument, so `compute_qpl_opt` passing model="qpl_opt" had
        # no effect and a TUNED run produced exactly the columns an untuned one did --
        # `results_qpl_<head>` with model='qpl_<head>' -- so ModelResults.csv could not
        # say whether a search had run. The head name stays the suffix (a QPL run fans
        # out to one column per classical head), so a tuned run reads 'qpl_opt_<head>'.
        method_qpl = f"{model}_{method}"
        print(method_qpl)
        try:
            estimator.fit(projections_train, y_train)
            y_predicted = estimator.predict(projections_test)
            # `auc` is computed from these scores alone, never from y_predicted. Every
            # head here is fitted unwrapped, so predict_proba is reachable: the six
            # searched heads are RandomizedSearchCV objects that delegate to
            # `best_estimator_`, and the bare TabPFN head answers directly. The one
            # exception is the SVC head -- `probability` is not in its grid, so
            # extract_binary_scores falls through to decision_function, which ranks
            # just as well. Scored on the *projections*: that is the space these heads
            # were fitted in, and the raw features would be the wrong width.
            y_score = extract_binary_scores(estimator, projections_test)
        except Exception as error:  # noqa: BLE001 -- narrowed immediately below
            # Only a weights-unavailable failure is survivable here, and only TabPFN can
            # raise one: its checkpoint sits behind a license acceptance that cannot be
            # detected in advance, so unlike a missing extra it is not caught by the
            # availability filtering above. Dropping the head matches what this function
            # already does for an unusable xgboost or catboost -- and matters more here,
            # because by this point the quantum projection has been computed and paid
            # for. Anything else is a real failure and propagates.
            explained = _explain_weight_access_failure(error, method_qpl)
            if explained is None:
                raise
            warnings.warn(
                f"{method_qpl} could not run and was skipped.\n{explained}",
                UserWarning,
            )
            continue

        hyperparameters = {
            "feature_map": feature_map.__class__.__name__,
            "feature_map_reps": reps,
            "entanglement": entanglement,
            # Every other head is a RandomizedSearchCV and carries best_params_.
            # TabPFN is fitted bare -- see create_tabpfn_model for why -- so there is
            # no search result to report and its own settings are the honest answer.
            "best_params": getattr(estimator, "best_params_", None) or estimator.get_params(),
            # Add other hyperparameters as needed
        }
        # Only when not the default, so 'unit' rows stay identical to earlier results.
        if data_map != "unit":
            hyperparameters["data_map"] = data_map
        if float(bandwidth) != 1.0:
            hyperparameters["bandwidth"] = bandwidth
        # Likewise only when set, so internal-mode rows are unchanged.
        # The bare TabPFN head has no search, so no head metric is recorded for it.
        if head_scoring is not None and method != "tabpfn":
            hyperparameters["head_scoring"] = head_scoring
        if head_max_iter is not None and method == "svc":
            hyperparameters["head_max_iter"] = head_max_iter
        # Only for a head whose grid (or TabPFN's settings) the config replaced, so a
        # default run's rows are byte-identical to earlier ones. Recorded because
        # `best_params` alone cannot say what was searched to find it: a narrowed grid
        # and the default grid can report the same winner for very different reasons.
        if classical_models_args.get(method):
            hyperparameters["head_search_args"] = classical_models_args[method]
        model_params = hyperparameters

        model_res.append(
            modeleval(
                y_test,
                y_predicted,
                beg_time,
                model_params,
                args,
                model=method_qpl,
                # Explicit because the label is 'qpl_opt_<head>': the marker is not a
                # suffix, so modeleval's endswith("_opt") inference cannot see it.
                tuned=str(model).endswith("_opt"),
                verbose=verbose,
                y_score=y_score,
            )
        )

    # Every head having been dropped leaves nothing to concatenate, and `pd.concat([])`
    # raises "No objects to concatenate" -- which says nothing about the heads or the
    # projection that produced them. Reachable now that a gated TabPFN is skipped rather
    # than fatal, and already reachable before via the unknown-model-name branch.
    if not model_res:
        raise ValueError(
            f"None of the requested classical models {classical_models} could be fitted "
            f"on the quantum projection, so there are no results to report. See the "
            f"warnings above for why each was skipped."
        )

    model_res = pd.concat(model_res)
    return model_res


def _head_search_spaces():
    """Each head's default hyperparameter grid, name -> {param: values}.

    The grids every ``create_<head>_model`` below has always searched, lifted out so
    that :func:`build_head` can hand them to the library's shared candidate validation
    (:func:`qbiocode.learning._grid.build_param_grid`) and so a config can override one
    per head. A function rather than a module constant because the random-forest entry
    builds numpy ranges, and a mutable default shared across calls is a trap.

    TabPFN is absent on purpose and has no grid: its weights are pretrained and frozen,
    so it exposes inference settings rather than training hyperparameters. See
    :func:`create_tabpfn_model`.
    """
    return {
        "rf": {
            "n_estimators": np.arange(100, 1000, 100),
            "max_depth": np.arange(5, 20),
            "min_samples_split": np.arange(2, 10),
            "min_samples_leaf": np.arange(1, 5),
            "bootstrap": [True, False],
        },
        "mlp": {
            "hidden_layer_sizes": [(128, 64, 32, 10), (64, 32, 10), (128, 64, 32)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.00005, 0.0005],
        },
        "svc": {
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
        },
        "lr": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
        "xgb": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        },
        "catboost": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [3, 5, 7],
            "l2_leaf_reg": [1.0, 3.0, 9.0],
            "subsample": [0.7, 0.8, 1.0],
        },
    }


def _head_estimator(head, seed, max_iter=None):
    """The unfitted estimator a searched head wraps, with its non-searched settings.

    The per-head constructor settings that are deliberately NOT hyperparameters: the
    seed, and the thread and quiet pins whose reasons are given at each branch. Split
    out from the grid so :func:`build_head` can pair any grid with the right estimator.
    """
    if head == "rf":
        return RandomForestClassifier(random_state=seed)
    if head == "mlp":
        return MLPClassifier(random_state=seed)
    if head == "svc":
        # max_iter only when given, so the default head is unchanged.
        return (
            SVC(random_state=seed) if max_iter is None
            else SVC(random_state=seed, max_iter=max_iter)
        )
    if head == "lr":
        return LogisticRegression(random_state=seed, max_iter=1000)
    if head == "xgb":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not properly installed or configured.\n"
                f"Error: {_XGBOOST_ERROR}\n\n"
                "On macOS, you may need to install OpenMP:\n"
                "  brew install libomp\n\n"
                "Then reinstall XGBoost:\n"
                "  pip install --force-reinstall xgboost\n\n"
                "See installation documentation for more details."
            )
        # random_state=seed, like every sibling head: the grid varies `subsample` and
        # `colsample_bytree`, both of which sample rows and columns at random, so an
        # unseeded estimator made this model irreproducible even though the search
        # itself was seeded.
        # n_jobs=1 because the RandomizedSearchCV below runs several fits at once.
        # sklearn's own estimators get their thread pools limited inside a joblib worker
        # by threadpoolctl; XGBoost drives its own OpenMP pool and does not, so every
        # worker would otherwise claim every core at once. Unpinned on a 128-core host
        # that oversubscribes by the worker count and the search is either unusably slow
        # or killed outright by the OOM reaper. One thread per fit keeps the outer
        # parallelism, which is the useful one -- see _SEARCH_N_JOBS for its cap.
        return XGBClassifier(  # type: ignore
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=1,
        )
    if head == "catboost":
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not properly installed or configured.\n"
                f"Error: {_CATBOOST_ERROR}\n\n"
                "CatBoost is a core QBioCode dependency, so this is a broken install. "
                "Reinstall it with:\n"
                "  pip install --force-reinstall catboost"
            )
        # `bootstrap_type` is pinned to 'Bernoulli' rather than left unset. The grid
        # varies `subsample`, which CatBoost accepts under Bernoulli/MVS/Poisson but
        # rejects under the Bayesian bootstrap -- and Bayesian is exactly what it
        # defaults to once the target has more than two classes. Unpinned, this head
        # would work on a binary projection and raise CatBoostError on a multiclass one.
        #
        # `allow_writing_files=False` keeps every fit from dropping a `catboost_info/`
        # directory into the working directory; `verbose=False` silences the
        # per-iteration training log. The search fits several of these at once, so both
        # matter more here than in a single fit.
        #
        # `thread_count=1` for the same reason xgb pins `n_jobs=1`: CatBoost manages its
        # own thread pool, which joblib cannot see and threadpoolctl does not limit.
        #
        # random_state=seed because the grid varies `subsample`, which samples at random.
        return CatBoostClassifier(  # type: ignore
            random_state=seed,
            bootstrap_type="Bernoulli",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    raise ValueError(
        f"Unknown QPL head {head!r}. Valid heads: {list(QPL_HEADS)}."
    )


def build_head(head, seed, scoring=None, max_iter=None, search_args=None):
    """The downstream model for one QPL head, searched over its own hyperparameters.

    Every head but TabPFN is a ``RandomizedSearchCV`` over that head's own grid --
    :data:`HEAD_N_ITER` candidates over :data:`HEAD_CV` folds of the quantum
    projection. Which hyperparameters those are is model-specific and comes from
    :func:`_head_search_spaces`, unless ``search_args`` overrides it.

    An override goes through :func:`qbiocode.learning._grid.build_param_grid`, the same
    validation every classical ``compute_*_opt`` uses, so a QPL head and a standalone
    model reject the same malformed block with the same message: a ``{low, high}``
    range (which only the Optuna engine can sample, and a ``RandomizedSearchCV`` cannot)
    is refused rather than passed to the estimator as a literal value, a bare scalar is
    accepted as a one-value list, and a key set to ``None`` or ``[]`` means "not
    searched" and leaves the estimator's default in force.

    Args:
        head (str): One of :data:`QPL_HEADS`.
        seed (int or None): ``random_state`` of both the estimator and the search.
        scoring (str, callable or None): What the search selects on. ``None`` (the
            default) is sklearn's, each estimator's ``score`` (accuracy).
        max_iter (int or None): libsvm iteration cap, read by the ``svc`` head only.
        search_args (Mapping or None): Replaces that head's default grid, as
            ``{param: values}``. ``None`` or empty keeps the default grid, so the
            shipped behaviour is unchanged.

    Returns:
        A ``RandomizedSearchCV``, or for ``'tabpfn'`` a bare classifier (see
        :func:`create_tabpfn_model`).

    Raises:
        ImportError: If the head's library is unusable -- ``compute_qpl`` filters those
            heads out beforehand, so reaching this means a direct call.
        ValueError: If ``head`` is not a known head, or ``search_args`` is malformed.
    """
    if head == "tabpfn":
        # No search: the weights are frozen, so there are no training hyperparameters
        # to vary. `search_args` for this head is its own constructor settings.
        return create_tabpfn_model(seed, **dict(search_args or {}))

    estimator = _head_estimator(head, seed, max_iter=max_iter)
    search_args = dict(search_args or {})
    if search_args:
        # `gridsearch_qpl_heads_args.<head>`, so the message names the block to edit.
        # `build_param_grid` and not `build_search_space`: the head searches are
        # RandomizedSearchCV, which enumerates values and cannot sample a range.
        distributions = build_param_grid(f"qpl head {head!r}", search_args)
    else:
        distributions = _head_search_spaces()[head]

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=HEAD_N_ITER,
        cv=HEAD_CV,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )


def create_xgb_model(seed, scoring=None):
    """The searched XGBoost head. See :func:`build_head`, which this delegates to."""
    return build_head("xgb", seed, scoring=scoring)


def create_catboost_model(seed, scoring=None):
    """The searched CatBoost head. See :func:`build_head`, which this delegates to.

    Three CatBoost-specific points, all applied by :func:`_head_estimator`:

    * ``bootstrap_type`` is pinned to ``'Bernoulli'`` rather than left unset. The grid
      varies ``subsample``, which CatBoost accepts under Bernoulli/MVS/Poisson but
      rejects under the Bayesian bootstrap -- and Bayesian is exactly what it
      defaults to once the target has more than two classes. Unpinned, this head would
      work on a binary projection and raise ``CatBoostError`` on a multiclass one.

    * ``allow_writing_files=False`` keeps every fit from dropping a ``catboost_info/``
      directory into the working directory; ``verbose=False`` silences the
      per-iteration training log. The search fits several of these at once, so
      both matter more here than in a single fit.

    * ``thread_count=1`` for the same reason ``create_xgb_model`` pins ``n_jobs=1``:
      CatBoost manages its own thread pool, which joblib cannot see and threadpoolctl
      does not limit, so unpinned every parallel worker would claim every core.
    """
    return build_head("catboost", seed, scoring=scoring)


def create_tabpfn_model(seed, model_version=TABPFN_DEFAULT_VERSION, model_path="auto",
                        **settings):
    """A bare TabPFN head -- the only one here that is not wrapped in a search.

    Every sibling head is a ``RandomizedSearchCV`` because its estimator has training
    hyperparameters worth searching. TabPFN has none: the weights are pretrained and
    frozen, ``fit`` only memorises the training rows, and what it exposes are
    inference settings that move accuracy very little. Wrapping it as the others are
    would cost ``HEAD_N_ITER * HEAD_CV`` transformer forward passes -- 200 at the
    settings used above -- per projection, per embedding, per split, to choose between
    near-identical candidates. Running it once at its defaults is both the honest
    configuration and the affordable one, which is rather the point of the model.

    ``model_path`` is resolved through
    :func:`qbiocode.learning.compute_tabpfn.resolve_model_path` rather than left at
    upstream's ``'auto'``. That matters for more than tidiness: upstream's ``'auto'``
    resolves to its own default version, which is ``v3`` -- whose weights are
    non-commercial, non-production, and reachable only after an interactive licence
    acceptance. So this head used to ignore QBioCode's pinned ``v2`` and reach for a
    gated checkpoint, which either failed with a licence error after the quantum
    projection had already been paid for, or silently ran a differently-licensed model
    than ``compute_tabpfn`` did on the same data. The two now agree.

    The caller reads ``best_params_`` off the returned object; ``compute_qpl`` falls
    back to ``get_params()`` for exactly this head.

    Args:
        seed (int or None): ``random_state`` of the classifier.
        model_version (str): Which pretrained checkpoint to pin. Defaults to
            QBioCode's :data:`~qbiocode.learning.compute_tabpfn.TABPFN_DEFAULT_VERSION`,
            the only one whose weights permit commercial use; selecting another warns.
        model_path (str): ``'auto'`` resolves from ``model_version``; an explicit path
            wins and is the way to run fully offline.
        **settings: Any other ``TabPFNClassifier`` inference setting, passed through --
            how ``classical_models_args: {tabpfn: {...}}`` reaches this head.

    Raises:
        ImportError: If the optional ``[tabpfn]`` extra is absent. ``compute_qpl``
            checks availability first and drops the head with a warning, so reaching
            this means the factory was called directly.
    """
    classifier_cls = _load_tabpfn_classifier()
    return classifier_cls(
        random_state=seed,
        model_path=resolve_model_path(model_path, model_version),
        **settings,
    )


def create_lr_model(seed, scoring=None):
    """The searched logistic-regression head. See :func:`build_head`."""
    return build_head("lr", seed, scoring=scoring)


def create_rf_model(seed, scoring=None):
    """The searched random-forest head. See :func:`build_head`."""
    return build_head("rf", seed, scoring=scoring)


def create_mlp_model(seed, scoring=None):
    """The searched MLP head. See :func:`build_head`."""
    return build_head("mlp", seed, scoring=scoring)


def create_svc_model(seed, scoring=None, max_iter=None):
    """The searched SVC head. See :func:`build_head`.

    ``max_iter`` caps libsvm's iterations; None (the default) is libsvm's -1, no cap.
    """
    return build_head("svc", seed, scoring=scoring, max_iter=max_iter)

def compute_qpl_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='qpl_opt' explicitly, but a caller using the default would otherwise
    # produce a row labelled as untuned -- and modeleval infers `tuned` from this
    # very string, so the label and the parameter column would BOTH be wrong.
    model="qpl_opt",
    data_key="",
    encoding=None,
    primitive=None,
    entanglement=None,
    reps=None,
    classical_models=None,
    data_map=None,
    bandwidth=None,
    *,
    classical_models_args=None,
    n_trials=10,
    validation_split=0.25,
    validation=None,
    default_params=None,
    reseed=None,
):
    """Tune QPL's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_qpl_args`` config block -- a list is a choice, a
    ``{low, high}`` mapping is a range. It differs in how a candidate is scored: a
    quantum fit computes a projection of every row by circuit simulation, so scoring by
    k-fold cross-validation would multiply an already expensive search by k. Each trial
    is scored once, on a stratified holdout carved out of ``X_train``; the caller's test
    set is never touched by the search.

    Only reachable when the config sets both ``grid_search: True`` and
    ``tune_quantum: True``. Tuning against a real device is refused unless
    ``allow_hardware_tuning: True`` -- every trial would be a queued job.

    **Across splits.** With ``freeze_quantum_params: True`` the projection search runs on
    the first resample only and its winner is reused by the rest, cutting the cost from
    ``iter * n_trials`` to ``n_trials + (iter - 1)``; ``False`` (the key's own default)
    searches the projection on every split, each on its own training rows. Only the
    *projection* is frozen: each downstream model's own ``RandomizedSearchCV`` still runs
    on every split, and a reusing split projects its own rows through the frozen feature
    map rather than reusing the first split's projected features. See
    :mod:`qbiocode.learning._param_cache`, which also explains why a frozen arm's later
    scores are optimistic rather than conservative.

    Args:
        X_train (array-like): Training data features. Split again internally to score
            candidates; the final model is refitted on all of it.
        X_test (array-like): Test data features, used only for the final evaluation.
        y_train (array-like): Training data labels.
        y_test (array-like): Test data labels.
        args (dict): Run configuration. ``backend``, ``shots`` and ``seed`` are read
            from it by the underlying quantum function.
        verbose (bool): If True, prints additional information during execution.
        model (str): Name of the model being used, default is 'QPL'.
        data_key (str): Key for identifying the dataset.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        data_map (list): Data maps to search ('unit', 'qiskit'; see :func:`compute_qpl`).
        bandwidth (list or dict): Input scales to search (see :func:`compute_qpl`). None
            leaves it at 1.0.
            None leaves it at the default.
        classical_models (list, optional): Which downstream classical models to fit on
            the quantum projection. **Not** a hyperparameter to search -- it selects
            which models run, so it is a list of heads rather than a list of candidate
            values, and every trial fits all of them. Forwarded to ``compute_qpl``
            unchanged, so None means its default six heads. ``model_run`` fills this in
            from ``qpl_args``, the same block the untuned path reads, so turning
            ``tune_quantum`` on does not change which heads run.
        classical_models_args (dict, optional): Per-head hyperparameter grids, as in
            :func:`compute_qpl`. Also **not** searched, and passed to every trial for
            the same reason ``classical_models`` is: the objective is the mean tuning
            metric across the heads, so a trial whose heads were searched over different
            grids than the refit's would be scoring a different pipeline. Note the two
            levels this creates -- the outer Optuna study searches the *projection*,
            while each head's own ``RandomizedSearchCV`` searches that *downstream
            model*, once per trial.
        n_trials (int): Trial budget, default 10 -- an order of magnitude below the
            classical default because each trial is a quantum fit. Lowered
            automatically when the configured values describe fewer combinations.
        validation_split (float): Fraction of the training data held out to score
            candidates on, default 0.25.
        validation (ValidationSplit or None): ``split_mode: manifest``. Every trial is
            then one fit on ``validation.X_fit`` scored on ``validation.X_val``
            (``validation_split`` is ignored), trials write no kernel dumps, and the
            unsearched keys of ``default_params`` are fixed for the trials and the refit
            (see :func:`qbiocode.learning.compute_fold.fold_fixed`). None (the default)
            is the inner-holdout search, unchanged. The heads' hidden searches are then
            scored with the tuning metric and the SVC head is capped at
            ``FOLD_SVC_MAX_ITER`` libsvm iterations (``head_scoring``/``head_max_iter``
            of :func:`compute_qpl`); they are part of one trial's fit, not extra budget.
            Trials are selected on the mean tuning metric over the heads, but every
            head's row gets its own ``trials_<model>_<head>`` log and its own
            ``tuning_score`` (its validation score at the refit trial).
        default_params (dict or None): The arm's default config, enqueued as trial 0.
        reseed (callable or None): Resets the global RNGs; called by the tuner before
            every trial and before the refit.

    Returns:
        modeleval (dict): The evaluation of the model at the best hyperparameters found,
        with the tuned values recorded in the results frame and the reported time
        covering the whole search rather than only the final fit.
    """
    beg_time = time.time()

    candidates = {
        "encoding": encoding,
        "primitive": primitive,
        "entanglement": entanglement,
        "reps": reps,
        "data_map": data_map,
        "bandwidth": bandwidth,
    }

    # `classical_models` selects which heads run; it is not a candidate value, so it goes
    # to every trial via `fixed` rather than into the search space. Handing it to the
    # trials as well as to the final fit is what keeps the two consistent: the objective
    # is the MEAN tuning_metric across heads (see _tuning._metric_of), so scoring candidates
    # on the default six while the final fit ran a different set would have chosen the
    # projection that suited heads the config excluded. `classical_models_args` rides
    # along for the same reason: it changes what each head's own search covers, so a
    # trial without it would be scoring a different downstream pipeline than the refit.
    fixed = {"classical_models": classical_models}
    if classical_models_args is not None:
        fixed["classical_models_args"] = classical_models_args
    compute_fn = compute_qpl
    trial_frames = None
    if validation is not None:
        # An explicit head list wins; None falls back to the configured one (qpl_args),
        # which default_params carries, rather than to compute_qpl's six.
        explicit = {} if classical_models is None else {"classical_models": classical_models}
        if classical_models_args is not None:
            explicit["classical_models_args"] = classical_models_args
        fixed = fold_fixed("qpl", candidates, default_params,
                           function_param_names(compute_qpl), explicit)
        # Every searched head's hidden 40 x 5 RandomizedSearchCV picks by the tuning
        # metric, and the SVC head's libsvm fits are capped. Both are part of one
        # trial's fit, not extra budget. The plain metric name keeps the log plain.
        fixed.setdefault("head_scoring", tuning_metric(args))
        # Assigned outright: a configured head_max_iter below 1 is libsvm's "no cap".
        fixed["head_max_iter"] = fold_svc_max_iter(fixed.get("head_max_iter"))
        # Each trial's frame, by trial number, for the per-head trial logs below.
        trial_frames = {}
        compute_fn = _recording(compute_qpl, trial_frames)

    best_params = run_function_study(
        compute_fn,
        build_search_space("qpl", candidates),
        X_train,
        y_train,
        args,
        model="qpl",
        n_trials=n_trials,
        seed=seed_from(args),
        validation_split=validation_split,
        data_key=data_key,
        fixed=fixed,
        validation=validation,
        default_params=default_params,
        reseed=reseed,
    )

    frame = compute_qpl(
        X_train,
        X_test,
        y_train,
        y_test,
        args,
        model=model,
        data_key=data_key,
        verbose=verbose,
        **best_params,
        **fixed,
    )
    head_scores = None
    if trial_frames is not None:
        head_scores = _attach_head_trial_logs(frame, best_params, trial_frames, model,
                                              tuning_metric(args))
    frame = record_tuned_params(frame, best_params, beg_time)
    if head_scores:
        _set_head_tuning_scores(frame, model, head_scores)
    return frame


def _recording(compute_fn, frames):
    """``compute_fn``, storing each trial's frame in ``frames`` under its trial number.

    The number is the running validation-study trial's
    (:func:`~qbiocode.learning._tuning.current_trial_number`); a call outside a study
    is not stored. A call that raised stores nothing, so its trial has no frame.
    """
    from qbiocode.learning._tuning import current_trial_number

    def recorded(*args, **kwargs):
        frame = compute_fn(*args, **kwargs)
        number = current_trial_number()
        if number is not None:
            frames[number] = frame
        return frame

    return recorded


def _head_of(frame, head):
    """``(metrics, y_pred, y_score)`` of the ``qpl_<head>`` rows of one trial's frame."""
    label = f"qpl_{head}"

    def first(column):
        if frame is None or column not in frame.columns:
            return None
        for value in frame[column]:
            if isinstance(value, (dict, np.ndarray, list)) or (
                value is not None and not (isinstance(value, float) and math.isnan(value))
            ):
                return value
        return None

    metrics = first(RESULTS_PREFIX + label)
    y_pred = first("y_predicted_" + label)
    y_score = first("y_score_" + label)
    if y_score is not None and np.ndim(y_score) != 1:
        y_score = None
    return (
        metrics if isinstance(metrics, dict) else None,
        None if y_pred is None else np.asarray(y_pred),
        None if y_score is None else np.asarray(y_score),
    )


def _attach_head_trial_logs(frame, best_params, trial_frames, model, metric):
    """Give every head of the refit its own ``trials_<model>_<head>`` log.

    The study scores a trial by the mean of ``metric`` over the heads (see
    ``_tuning._metric_of``) and has no single prediction to keep for a multi-head frame.
    Each head's log copies the study's trials -- params, state, duration, default flag
    -- with that head's own validation score and predictions. Written before
    :func:`record_tuned_params`, which then leaves these columns alone.

    ``trial_frames`` maps a trial number to that trial's frame. A trial with none (it
    failed before or inside the call) keeps its params and state, with a NaN score and
    no predictions; the other trials are unaffected.

    Returns:
        dict: head -> its validation score at the refit trial, or None when the study
        carries no trials (logged; the study-level log is then attached to every head
        instead, as for any other model).
    """
    from qbiocode.evaluation.protocol import TRIALS_PREFIX, TrialRecord
    from qbiocode.evaluation.protocol import trial_log as protocol_trial_log

    trials = getattr(best_params, "trials", None)
    if not trials:
        logger.warning("qpl per-head trial logs skipped: the study recorded no trials.")
        return None
    missing = [t.number for t in trials if t.number not in trial_frames]
    if missing:
        logger.info("qpl trials %s have no frame (they failed before scoring); their "
                    "per-head entries carry no score or predictions.", missing)
    prefix = f"{RESULTS_PREFIX}{model}_"
    heads = [c[len(prefix):] for c in frame.columns if c.startswith(prefix)]
    scores = {}
    for head in heads:
        records = []
        for trial in trials:
            metrics, y_pred, y_score = _head_of(trial_frames.get(trial.number), head)
            value = math.nan
            if trial.state == "COMPLETE" and metrics is not None:
                value = float(metrics.get(metric, math.nan))
            records.append(TrialRecord(
                number=trial.number, params=dict(trial.params), value=value,
                state=trial.state, duration_s=trial.duration_s,
                is_default=trial.is_default, y_pred=y_pred, y_score=y_score,
            ))
            if trial.number == best_params.best_trial:
                scores[head] = value
        log = protocol_trial_log(
            records, metric=best_params.metric, best=best_params.best_trial,
            val_idx=best_params.val_idx, y_val=best_params.y_val, fixed=best_params.fixed,
        )
        column = f"{RESULTS_PREFIX}{model}_{head}"
        cells = [log if isinstance(v, dict) else None for v in frame[column]]
        series = pd.Series([None] * len(cells), index=frame.index, dtype=object)
        for position, cell in enumerate(cells):
            series.iat[position] = cell
        frame[TRIALS_PREFIX + f"{model}_{head}"] = series
    return scores


def _set_head_tuning_scores(frame, model, head_scores):
    """Overwrite each head's ``tuning_score`` (the study's mean) with its own."""
    for head, score in head_scores.items():
        column = f"{RESULTS_PREFIX}{model}_{head}"
        for value in frame[column]:
            if isinstance(value, dict):
                value["tuning_score"] = score
