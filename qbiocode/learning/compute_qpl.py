# ====== Base class imports ======
import hashlib
import os
import logging
import math
import time
import warnings

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
    TABPFN_MAX_CLASSES,
    _explain_weight_access_failure,
    _load_tabpfn_classifier,
    tabpfn_is_available,
)


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
        classical_models (list): List of classical models to train on quantum projections.
            Defaults to ``['rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost']``. ``'tabpfn'`` is
            also accepted but is deliberately absent from the default: it needs the
            optional ``[tabpfn]`` extra, so defaulting it on would make every QPL run warn
            in an ordinary install. Name it explicitly to use it. It needs no API token:
            QBioCode pins the ungated ``v2`` weights.
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
            (40 candidates x 5 folds on the projections) picks its config by. None (the
            default) is sklearn's, each estimator's ``score`` (accuracy).
            ``compute_qpl_opt`` sets the run's tuning metric under ``split_mode:
            manifest``. The head searches are part of this one fit -- one trial of an
            outer study -- not extra tuning budget. The bare TabPFN head has no search.
        head_max_iter (int or None): libsvm iteration cap of the SVC head's fits. None
            (the default) is libsvm's -1, no cap. No other head reads it.

    Returns:
        modeleval (pd.DataFrame): A DataFrame containing evaluation metrics and model parameters for all models.

    Raises:
        ValueError: If ``data_map`` is not ``'unit'``, ``'qiskit'`` or a bool.
    """

    # Set default classical models if not provided
    if classical_models is None:
        classical_models = ["rf", "mlp", "svc", "lr", "xgb", "catboost"]

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
            for f_tr in [file_projection_train, file_projection_test]:
                if os.path.exists(f_tr):
                    continue
                dat = X_train.copy() if "train" in f_tr else X_test.copy()
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
        
            for f_tr in [file_projection_train, file_projection_test]:
                if not os.path.exists(f_tr):
                    projections = []
                    if "train" in f_tr:
                        dat = X_train.copy()
                    else:
                        dat = X_test.copy()

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

    # If no models remain after filtering, raise an error
    if not classical_models:
        raise ValueError(
            "No valid classical models specified. Please provide at least one model "
            "from: 'rf', 'mlp', 'svc', 'lr', 'xgb', 'catboost', 'tabpfn'"
        )

    # `estimator`, not `model`, inside this loop. It used to rebind `model` -- the label
    # parameter -- to each head's estimator, so the label was destroyed on the first
    # iteration. That is why the results label below was hardcoded to "qpl_" + head:
    # there was nothing left to read it from. Same shadowing bug as compute_pqk had.
    model_res = []
    # None unless compute_qpl_opt asked for the tuning metric: RandomizedSearchCV's own
    # default, so an internal-mode head search is unchanged.
    scoring = head_scorer(head_scoring, args)
    for method in classical_models:
        if method == "rf":
            estimator = create_rf_model(args["seed"], scoring=scoring)
        elif method == "svc":
            estimator = create_svc_model(args["seed"], scoring=scoring, max_iter=head_max_iter)
        elif method == "mlp":
            estimator = create_mlp_model(args["seed"], scoring=scoring)
        elif method == "lr":
            estimator = create_lr_model(args["seed"], scoring=scoring)
        elif method == "xgb":
            estimator = create_xgb_model(args["seed"], scoring=scoring)
        elif method == "catboost":
            estimator = create_catboost_model(args["seed"], scoring=scoring)
        elif method == "tabpfn":
            estimator = create_tabpfn_model(args["seed"])
        else:
            warnings.warn(
                f"Unknown model type '{method}' skipped. Valid options: 'rf', 'mlp', "
                f"'svc', 'lr', 'xgb', 'catboost', 'tabpfn'",
                UserWarning,
            )
            continue

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


def create_xgb_model(seed, scoring=None):
    # Initialize the XGBoost Classifier
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
    # random_state=seed, like every sibling create_*_model here: the search grid
    # below varies `subsample` and `colsample_bytree`, both of which sample rows and
    # columns at random, so an unseeded estimator made this model irreproducible even
    # though the search itself was seeded.
    # n_jobs=1 because the RandomizedSearchCV below runs several fits at once. sklearn's
    # own estimators get their thread pools limited inside a joblib worker by
    # threadpoolctl; XGBoost drives its own OpenMP pool and does not, so every worker
    # would otherwise claim every core at once. Unpinned on a 128-core host that
    # oversubscribes by the worker count and the search is either unusably slow or killed
    # outright by the OOM reaper. One thread per fit keeps the outer parallelism, which is
    # the useful one -- see _SEARCH_N_JOBS for the cap on how many workers that is.
    xgb = XGBClassifier(  # type: ignore
        objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1
    )

    xgb_param_distributions = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    # Initialize RandomizedSearchCV
    xgb_model = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return xgb_model


def create_catboost_model(seed, scoring=None):
    """A searched CatBoost head, matching how the other tree-based heads are built.

    Two CatBoost-specific points:

    * ``bootstrap_type`` is pinned to ``'Bernoulli'`` rather than left unset. The grid
      below varies ``subsample``, which CatBoost accepts under Bernoulli/MVS/Poisson
      but rejects under the Bayesian bootstrap -- and Bayesian is exactly what it
      defaults to once the target has more than two classes. Unpinned, this head would
      work on a binary projection and raise ``CatBoostError`` on a multiclass one.

    * ``allow_writing_files=False`` keeps every fit from dropping a ``catboost_info/``
      directory into the working directory; ``verbose=False`` silences the
      per-iteration training log. The search below fits several of these at once, so
      both matter more here than in a single fit.

    * ``thread_count=1`` for the same reason ``create_xgb_model`` pins ``n_jobs=1``:
      CatBoost manages its own thread pool, which joblib cannot see and threadpoolctl
      does not limit, so unpinned every parallel worker would claim every core.
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError(
            "CatBoost is not properly installed or configured.\n"
            f"Error: {_CATBOOST_ERROR}\n\n"
            "CatBoost is a core QBioCode dependency, so this is a broken install. "
            "Reinstall it with:\n"
            "  pip install --force-reinstall catboost"
        )
    # random_state=seed for the same reason as create_xgb_model: the grid varies
    # `subsample` and `rsm`, both of which sample at random.
    catboost = CatBoostClassifier(  # type: ignore
        random_state=seed,
        bootstrap_type="Bernoulli",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )

    catboost_param_distributions = {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [3, 5, 7],
        "l2_leaf_reg": [1.0, 3.0, 9.0],
        "subsample": [0.7, 0.8, 1.0],
    }

    # Initialize RandomizedSearchCV
    catboost_model = RandomizedSearchCV(
        estimator=catboost,
        param_distributions=catboost_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return catboost_model


def create_tabpfn_model(seed):
    """A bare TabPFN head -- the only one here that is not wrapped in a search.

    Every sibling factory returns a ``RandomizedSearchCV`` because its estimator has
    training hyperparameters worth searching. TabPFN has none: the weights are
    pretrained and frozen, ``fit`` only memorises the training rows, and what it
    exposes are inference settings that move accuracy very little. Wrapping it as the
    others are would cost ``n_iter * cv`` transformer forward passes -- 200 at the
    settings used above -- per projection, per embedding, per split, to choose between
    near-identical candidates. Running it once at its defaults is both the honest
    configuration and the affordable one, which is rather the point of the model.

    The caller reads ``best_params_`` off the returned object; ``compute_qpl`` falls
    back to ``get_params()`` for exactly this head.

    Raises:
        ImportError: If the optional ``[tabpfn]`` extra is absent. ``compute_qpl``
            checks availability first and drops the head with a warning, so reaching
            this means the factory was called directly.
    """
    classifier_cls = _load_tabpfn_classifier()
    return classifier_cls(random_state=seed)


def create_lr_model(seed, scoring=None):
    # Initialize the Logistic Regression Classifier
    lr = LogisticRegression(random_state=seed, max_iter=1000)

    lr_param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }

    # Initialize RandomizedSearchCV
    lr_model = RandomizedSearchCV(
        estimator=lr,
        param_distributions=lr_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return lr_model


def create_rf_model(seed, scoring=None):
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=seed)

    rf_param_distributions = {
        "n_estimators": np.arange(100, 1000, 100),
        "max_depth": np.arange(5, 20),
        "min_samples_split": np.arange(2, 10),
        "min_samples_leaf": np.arange(1, 5),
        "bootstrap": [True, False],
    }

    # Initialize RandomizedSearchCV
    rf_model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return rf_model


def create_mlp_model(seed, scoring=None):
    mlp_param_distributions = {
        "hidden_layer_sizes": [(128, 64, 32, 10), (64, 32, 10), (128, 64, 32)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": [0.00005, 0.0005],
    }

    # Initialize the MLP Classifier
    mlp = MLPClassifier(random_state=seed)

    # Initialize RandomizedSearchCV
    mlp_model = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=mlp_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return mlp_model


def create_svc_model(seed, scoring=None, max_iter=None):
    svc_param_distributions = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    }

    # Initialize the SVC. max_iter only when given, so the default head is unchanged.
    svc = SVC(random_state=seed) if max_iter is None else SVC(random_state=seed, max_iter=max_iter)

    # Initialize RandomizedSearchCV
    svc_model = RandomizedSearchCV(
        estimator=svc,
        param_distributions=svc_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        # None (the default) is the estimator's own score, accuracy; see compute_qpl.
        scoring=scoring,
    )

    return svc_model

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
        classical_models (list, optional): Which classical heads to fit on the quantum
            projection. **Not** a hyperparameter to search -- it selects which models
            run, so it is a list of heads rather than a list of candidate values, and
            every trial fits all of them. Forwarded to ``compute_qpl`` unchanged, so
            None means its default six heads. ``model_run`` fills this in from
            ``qpl_args``, the same block the untuned path reads, so turning
            ``tune_quantum`` on does not change which heads run.
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
    # projection that suited heads the config excluded.
    fixed = {"classical_models": classical_models}
    compute_fn = compute_qpl
    trial_frames = None
    if validation is not None:
        # An explicit head list wins; None falls back to the configured one (qpl_args),
        # which default_params carries, rather than to compute_qpl's six.
        explicit = {} if classical_models is None else {"classical_models": classical_models}
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
