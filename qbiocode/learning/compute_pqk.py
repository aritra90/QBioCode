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

# ====== Base class imports ======
import hashlib
import json
import os
import time
import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# from qiskit.primitives import Sampler

# ====== Qiskit imports ======
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime.exceptions import IBMRuntimeError, RuntimeJobFailureError

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
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


#: The accepted spellings of PQK's ``data_map``, as ``qbiocode.data_generation`` spells
#: them (``quantum_core.DATA_MAPS``): ``'unit'`` is
#: :func:`qbiocode.utils.qutils.unit_coefficient_data_map`, the map PQK has always
#: used; ``'qiskit'`` passes no ``data_map_func``, so the feature map uses qiskit's
#: default -- ``phi(x_i) = x_i`` and ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)``, the map
#: the ``eng_*`` and ``qlab_zz`` synthetic families are generated with by default.
PQK_DATA_MAPS = ("unit", "qiskit")


def _resolve_data_map(data_map):
    """Canonicalise PQK's ``data_map`` to ``'unit'`` or ``'qiskit'``.

    Also accepts the boolean convention of :func:`qbiocode.embeddings.embed.pqk`,
    whose ``data_map=True`` selects the unit-coefficient map and ``False`` qiskit's
    default, so a value copied from an embedding config means the same thing here.

    Args:
        data_map (str or bool): ``'unit'``, ``'qiskit'``, ``True`` or ``False``.

    Returns:
        str: ``'unit'`` or ``'qiskit'``.

    Raises:
        ValueError: for any other value, naming the accepted options.
    """
    # bool (and numpy bool) first: True == 1 would otherwise be indistinguishable
    # from an int, and neither is a str.
    if isinstance(data_map, (bool, np.bool_)):
        return "unit" if data_map else "qiskit"
    if isinstance(data_map, str) and data_map in PQK_DATA_MAPS:
        return data_map
    raise ValueError(
        f"data_map must be one of {list(PQK_DATA_MAPS)} (or a bool: True for 'unit', "
        f"False for 'qiskit'); got {data_map!r}."
    )


def _dump_pqk_projections(
    args, data_key, model, Z_train, Z_test, X_train, y_train, y_test, best_params
):
    """Write the projected features PQK's classical head was fitted on.

    Unlike QSVC's fidelity Gram, PQK's kernel is cheap to *recompute* -- the head is an SVC
    over the projections, so ``pairwise_kernels(Z, metric=kernel, gamma=gamma)`` reproduces
    it exactly and costs no circuits. What blocks the diagnostic is bookkeeping, not cost:

      * the projection cache is keyed by a sha256 over the feature-map parameters **and**
        ``dataset_fingerprint(X_train, X_test)``, so locating the right ``.npy`` after the
        fact requires already holding the exact split that produced it;
      * the cache stores no labels, and kernel-target alignment needs ``y`` in the row order
        of ``Z``, which is a property of the split;
      * ``X_train`` is needed for the classical side of ``g(K_c || K_q)`` and is persisted
        nowhere -- ``qprofiler`` pickles only the results frame.

    The head searches ``kernel`` over ``['linear', 'rbf', 'poly', 'sigmoid']``, so the chosen
    kernel is **not** necessarily RBF. ``best_params`` is stored here so a reader reconstructs
    what was actually fitted instead of assuming a radial basis.

    Projections are ``n x feat_dimension``, so this file is smaller than a QSVC Gram by a
    factor of ``n / feat_dimension``.
    """
    dump_dir = args.get("kernel_dump_dir") if isinstance(args, Mapping) else None
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    stem = os.path.join(dump_dir, f"proj_{model}_{data_key}")
    np.savez_compressed(
        stem + ".npz",
        Z_train=np.asarray(Z_train),
        Z_test=np.asarray(Z_test),
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        # npz holds arrays, not mappings: the SVC choice travels as a JSON scalar and is
        # read back with json.loads(str(z["best_params"])).
        best_params=np.asarray(json.dumps(best_params, default=str)),
    )


def compute_pqk(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    # Lower case, matching the dispatch key. This default was "PQK", which was invisible
    # while the body hardcoded its label -- but now that the label is honoured, a direct
    # call with no model= would otherwise file results under a name no config can name.
    # qc_winner_finder's quantum list was also written against the upper-case spelling
    # while every real results table carries the lower-case one.
    model="pqk",
    data_key="",
    verbose=False,
    encoding="Z",
    primitive="estimator",
    entanglement="linear",
    reps=2,
    data_map="unit",
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
        model (str): Model type, default is 'PQK'.
        data_key (str): Key for the dataset, default is ''.
        verbose (bool): If True, print additional information, default is False.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.
        data_map (str or bool): How features become gate angles. ``'unit'`` (the
            default) is :func:`qbiocode.utils.qutils.unit_coefficient_data_map`;
            ``'qiskit'`` is qiskit's default map, ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)``.
            ``True``/``False`` are accepted as ``'unit'``/``'qiskit'``, matching
            :func:`qbiocode.embeddings.embed.pqk`.
        head_scoring (str or None): Metric the classical head's ``RandomizedSearchCV``
            (40 candidates x 5 folds on the projections) picks its SVC by. None (the
            default) is sklearn's, accuracy. ``compute_pqk_opt`` sets the run's tuning
            metric under ``split_mode: manifest``. The head search is part of this one
            fit -- one trial of an outer study -- not extra tuning budget.
        head_max_iter (int or None): libsvm iteration cap of the head's SVC fits. None
            (the default) is libsvm's -1, no cap.

    Returns:
        modeleval (pd.DataFrame): A DataFrame containing evaluation metrics and model parameters for all models.

    Raises:
        ValueError: if any argument is outside its accepted set or the train/test
            arrays are inconsistent. Every check runs before any directory is
            created or any cached projection is read, so a mistyped ``encoding``
            costs nothing and reports the parameter, the value received and the
            accepted set instead of failing later inside qiskit.
    """
    # --- boundary validation -------------------------------------------------
    # These used to surface far from their cause: a bad `encoding` reached
    # qutils.get_feature_map and came back as the input string, failing with
    # "'str' object has no attribute 'num_qubits'"; a bad `entanglement` reached
    # qiskit and came back as "Something went wrong in Rust space".
    if not isinstance(args, Mapping):
        raise ValueError(
            f"args must be a mapping of run configuration (it is read with "
            f"args['backend'] and args.get(...)); got {type(args).__name__}."
        )
    if "backend" not in args:
        raise ValueError(
            "args is missing the required 'backend' key (e.g. 'simulator', or an "
            f"IBM Quantum backend name). Keys present: {sorted(args)}."
        )
    if encoding not in qutils.SUPPORTED_FEATURE_MAPS:
        raise ValueError(
            f"encoding must be one of {qutils.SUPPORTED_FEATURE_MAPS} "
            f"(case-sensitive); got {encoding!r}."
        )
    if isinstance(entanglement, str) and entanglement not in qutils.SUPPORTED_ENTANGLEMENTS:
        raise ValueError(
            f"entanglement must be one of {qutils.SUPPORTED_ENTANGLEMENTS}; "
            f"got {entanglement!r}."
        )
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError(
            f"reps is the number of feature-map repetitions and must be a "
            f"positive integer; got {reps!r}."
        )
    data_map = _resolve_data_map(data_map)
    if primitive != "estimator":
        # PQK projects onto Pauli expectation values, which is an Estimator
        # measurement; the backend below is requested as "estimator"
        # unconditionally. Accepting 'sampler' therefore changed only the cache
        # fingerprint, not the computation -- two cache files holding identical
        # projections, and a caller who believed they had measured something else.
        raise ValueError(
            f"primitive must be 'estimator'; got {primitive!r}. Projected quantum "
            f"kernels are built from Pauli expectation values, which only the "
            f"Estimator primitive provides. For sampler-based models see "
            f"compute_vqc or compute_qnn."
        )
    if not isinstance(data_key, str):
        raise ValueError(
            f"data_key is interpolated into the projection cache filename and "
            f"must be a string; got {type(data_key).__name__} ({data_key!r})."
        )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"X_train and X_test must be 2-D (n_samples, n_features); got "
            f"{X_train.ndim}-D and {X_test.ndim}-D. Reshape a single sample with "
            f"X.reshape(1, -1)."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of features -- one "
            f"feature map is built for both -- got {X_train.shape[1]} and "
            f"{X_test.shape[1]}."
        )
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"X_train and X_test must both be non-empty; got "
            f"{X_train.shape[0]} training and {X_test.shape[0]} test samples."
        )
    if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
        raise ValueError(
            f"Labels and features must be aligned; got {X_train.shape[0]} train "
            f"samples vs {len(y_train)} train labels, and {X_test.shape[0]} test "
            f"samples vs {len(y_test)} test labels."
        )
    # ------------------------------------------------------------------------

    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    projection_dir = os.path.expanduser(args.get("pqk_projection_dir", "pqk_projections"))
    if not os.path.exists(projection_dir):
        os.makedirs(projection_dir)

    # The cached projections are only valid for the exact feature map that produced them, so the
    # feature-map parameters must be part of the cache key. Without this, changing `encoding`,
    # `entanglement`, `reps` or `primitive` and rerunning into the same pqk_projection_dir
    # silently reloads the previous run's projections and reports them as the new result.
    # A short digest keeps the filename bounded regardless of how many parameters are added.
    # `projection_backend` joins the fingerprint ONLY when it is set. It has to be in
    # there for a head-to-head: without it the second and third backends load the first
    # one's cached projection, which makes their timings meaningless and their agreement
    # tautological. But appending a bare `None` would change every legacy hash too, and
    # that silently orphans the projection caches already on disk (88 .npy files ship in
    # this repo alone) -- a slow surprise, not a wrong answer, but avoidable.
    fingerprint_parts = (encoding, entanglement, reps, primitive, feat_dimension)
    # `data_map` joins only when it is not the default, for the same reason as
    # `projection_backend` below: every cache on disk was written with the 'unit' map,
    # and keying 'unit' explicitly would orphan all of them. 'qiskit' is a different
    # circuit, so it must never reach a 'unit' file.
    if data_map != "unit":
        fingerprint_parts = fingerprint_parts + (f"data_map={data_map}",)
    _projection_backend = args.get("projection_backend")
    if _projection_backend:
        fingerprint_parts = fingerprint_parts + (_projection_backend,)
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
        "pqk_projection_" + data_key + "_" + feature_map_fingerprint + "_train.npy",
    )
    file_projection_test = os.path.join(
        projection_dir,
        "pqk_projection_" + data_key + "_" + feature_map_fingerprint + "_test.npy",
    )
    checkpoint_dir = os.path.join(projection_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_every = int(args.get("pqk_checkpoint_every", 1))
    session_chunk_size = int(args.get("pqk_session_chunk_size", 100))
    max_runtime_retries = int(args.get("pqk_runtime_max_retries", 3))

    def _checkpoint_path(final_path):
        base = os.path.basename(final_path)
        return os.path.join(checkpoint_dir, base.replace(".npy", ".partial.npy"))

    def _save_projection_array(path, projections):
        tmp_path = path + ".tmp.npy"
        np.save(tmp_path, np.asarray(projections))
        os.replace(tmp_path, path)

    def _load_checkpoint(path, expected_len):
        if not os.path.exists(path):
            return []
        projections = np.load(path, allow_pickle=False)
        if len(projections) > expected_len:
            raise ValueError(
                f"Checkpoint {path} has {len(projections)} rows, "
                f"but the dataset only has {expected_len} rows."
            )
        return list(projections)

    def _validate_projection_file(path, expected_len):
        if not os.path.exists(path):
            return
        projections = np.load(path, allow_pickle=False)
        if len(projections) != expected_len:
            raise ValueError(
                f"Projection file {path} has {len(projections)} rows, "
                f"but the current dataset expects {expected_len} rows. "
                "Remove this projection file or use a different pqk_projection_dir."
            )
        # Each row holds one expectation value per Pauli-X/Y/Z observable per qubit, so the
        # flattened width must be 3 * feat_dimension. A mismatch means the file was written by a
        # run with a different feature dimension and must not be silently reused.
        expected_width = 3 * feat_dimension
        actual_width = int(np.prod(np.asarray(projections).shape[1:])) if len(projections) else 0
        if len(projections) and actual_width != expected_width:
            raise ValueError(
                f"Projection file {path} has {actual_width} features per row, "
                f"but the current feature map produces {expected_width} "
                f"(3 observables x {feat_dimension} qubits). "
                "Remove this projection file or use a different pqk_projection_dir."
            )

    def _is_closed_session_error(exc):
        return isinstance(exc, IBMRuntimeError) and (
            "Session has been closed" in str(exc) or '"code":1217' in str(exc)
        )

    def _is_retryable_runtime_error(exc):
        return isinstance(exc, RuntimeJobFailureError) and (
            "Temporary Internal Error" in str(exc) or "Error code 9707" in str(exc)
        )

    def _close_session(session):
        if not isinstance(session, type(None)):
            session.close()

    def _refresh_runtime(session):
        _close_session(session)
        _, new_session, new_prim = qutils.get_backend_session(
            args, "estimator", num_qubits=num_qubits
        )
        return new_session, new_prim

    # 'unit' is shared with qbiocode.embeddings.embed.pqk -- see
    # qutils.unit_coefficient_data_map for why the symbolic case must not be
    # narrowed to a float. 'qiskit' passes None, which get_feature_map and the
    # projector both hand to qiskit as "use your default map".
    data_map_func = qutils.unit_coefficient_data_map if data_map == "unit" else None

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

    _validate_projection_file(file_projection_train, len(X_train))
    _validate_projection_file(file_projection_test, len(X_test))

    if (not os.path.exists(file_projection_train)) | (not os.path.exists(file_projection_test)):

        projection_backend = args.get("projection_backend")
        if projection_backend:
            # One state preparation per row, all 3n Pauli expectations read off it. The
            # else-branch uses StatevectorEstimator, which re-simulates the circuit once
            # PER OBSERVABLE -- a ~3n-fold overhead that is not simulation work. It stays
            # the default for hardware, session/checkpoint handling and compatibility.
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
            for f_tr, dat in [
                (file_projection_train, X_train.copy()),
                (file_projection_test, X_test.copy()),
            ]:
                if os.path.exists(f_tr):
                    continue
                projections = projector.project(
                    dat, progress_every=100,
                    n_jobs=args.get("projection_n_jobs", 1),
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
                _save_projection_array(f_tr, projections[:, :, ::-1])
            fidelity = projector.fidelity_estimate()
            if fidelity is not None and fidelity < 0.999:
                warnings.warn(
                    f"projection_backend={projection_backend!r} truncated the state: "
                    f"fidelity estimate {fidelity:.4g} (1.0 = exact). These projected "
                    f"features are approximate.",
                    RuntimeWarning,
                )
        else:
            #  Generate the backend, session and primitive
            backend, session, prim = qutils.get_backend_session(
                args, "estimator", num_qubits=num_qubits
            )
            try:

                # Transpile
                if args["backend"] != "simulator":
                    circuit = qutils.transpile_circuit(
                        circuit, opt_level=3, backend=backend, PT=True, initial_layout=None
                    )

                # Set the global phase to 0 to avoid header size issues
                circuit.global_phase = 0
        
                for f_tr, dat in [
                    (file_projection_train, X_train.copy()),
                    (file_projection_test, X_test.copy()),
                ]:
                    if not os.path.exists(f_tr):
                        projections = []

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

                        checkpoint_file = _checkpoint_path(f_tr)
                        projections = _load_checkpoint(checkpoint_file, len(dat))
                        if projections:
                            print(
                                f"Resuming {os.path.basename(f_tr)} from "
                                f"datapoint {len(projections)}"
                            )

                        datapoints_in_session = 0
                        for i in range(len(projections), len(dat)):
                            if i % 100 == 0:
                                print(f"at datapoint {str(i)}")
                            if (
                                session is not None
                                and session_chunk_size > 0
                                and datapoints_in_session >= session_chunk_size
                            ):
                                session, prim = _refresh_runtime(session)
                                datapoints_in_session = 0

                            # Get training sample
                            parameters = dat[i]

                            # We define the primitive unified blocs (PUBs) consisting of the embedding circuit,
                            # set of observables and the circuit parameters
                            pub_x = (circuit, observables_x, parameters)
                            pub_y = (circuit, observables_y, parameters)
                            pub_z = (circuit, observables_z, parameters)

                            retry_count = 0
                            while True:
                                try:
                                    job = prim.run([pub_x, pub_y, pub_z])
                                    job_result = job.result()
                                    job_result_x = job_result[0].data.evs
                                    job_result_y = job_result[1].data.evs
                                    job_result_z = job_result[2].data.evs
                                    break
                                except Exception as exc:
                                    _save_projection_array(checkpoint_file, projections)
                                    if session is not None and _is_closed_session_error(exc):
                                        session, prim = _refresh_runtime(session)
                                        datapoints_in_session = 0
                                        continue
                                    if (
                                        session is not None
                                        and _is_retryable_runtime_error(exc)
                                        and retry_count < max_runtime_retries
                                    ):
                                        retry_count += 1
                                        print(
                                            f"Retrying datapoint {i} after temporary runtime "
                                            f"failure ({retry_count}/{max_runtime_retries})"
                                        )
                                        session, prim = _refresh_runtime(session)
                                        datapoints_in_session = 0
                                        continue
                                    raise

                            # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                            projections.append([job_result_x, job_result_y, job_result_z])
                            datapoints_in_session += 1
                            if checkpoint_every > 0 and len(projections) % checkpoint_every == 0:
                                _save_projection_array(checkpoint_file, projections)

                        _save_projection_array(f_tr, projections)
                        if os.path.exists(checkpoint_file):
                            os.remove(checkpoint_file)

            finally:
                if not isinstance(session, type(None)):
                    session.close()

    # Load computed projections
    projections_train = np.load(file_projection_train)
    projections_train = np.array(projections_train).reshape(len(projections_train), -1)
    projections_test = np.load(file_projection_test)
    projections_test = np.array(projections_test).reshape(len(projections_test), -1)

    # `estimator`, not `model`. This assignment used to be `model = create_svc_model(...)`,
    # which overwrote the `model` PARAMETER -- the label this function was told to file its
    # results under -- with the fitted estimator object. The label was therefore gone
    # before it could be used, and `method_pqk = "pqk"` on the next line was the
    # workaround: a hardcoded label that ignored the argument. The visible consequence was
    # that `compute_pqk_opt` passing model="pqk_opt" had no effect, so a TUNED PQK run
    # produced `results_pqk` with model='pqk' -- byte-identical to an untuned one, leaving
    # no way to tell from ModelResults.csv whether a search had run.
    estimator = create_svc_model(
        args["seed"], scoring=head_scorer(head_scoring, args), max_iter=head_max_iter
    )

    estimator.fit(projections_train, y_train)
    y_predicted = estimator.predict(projections_test)
    # `auc` is computed from these scores alone, never from y_predicted. The head is a
    # RandomizedSearchCV over SVC, which delegates to `best_estimator_`; `probability`
    # is not searched, so there is no predict_proba and extract_binary_scores falls
    # through to decision_function. Scored on the *projections*, which is the space this
    # estimator was fitted in -- the raw features would silently be the wrong width.
    y_score = extract_binary_scores(estimator, projections_test)

    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "feature_map_reps": reps,
        "entanglement": entanglement,
        "best_params": estimator.best_params_,
        # Add other hyperparameters as needed
    }
    # Recorded only off the default, so a 'unit' run's parameter column stays
    # byte-identical to results written before `data_map` existed.
    if data_map != "unit":
        hyperparameters["data_map"] = data_map
    # Likewise only when set, so internal-mode rows are unchanged.
    if head_scoring is not None:
        hyperparameters["head_scoring"] = head_scoring
    if head_max_iter is not None:
        hyperparameters["head_max_iter"] = head_max_iter
    model_params = hyperparameters

    _dump_pqk_projections(
        args,
        data_key,
        model,
        projections_train,
        projections_test,
        X_train,
        y_train,
        y_test,
        estimator.best_params_,
    )

    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        params=model_params,
        args=args,
        model=model,
        verbose=verbose,
        y_score=y_score,
    )





# Parallelism for the classical SVC head below. It was ``n_jobs=-1``, which is wrong in
# the one place this function is actually called from: ``compute_pqk``/``compute_pqk_opt``
# run inside a joblib worker, because ``model_run`` fans the model list out over loky.
# joblib does not let a nested ``Parallel`` start new processes -- it swaps in the
# threading backend -- so this never deadlocked and never showed up as an error. What it
# did instead, measured on a 128-core host with only four outer workers, was take each
# worker from 3 OS threads to ~150. At the pilot's ``n_jobs: 13`` that is well over a
# thousand runnable threads inside a 16-slot cgroup: the job does not fail, it thrashes,
# which from the outside is indistinguishable from being stuck. ``OMP_NUM_THREADS`` does
# not help, since these are joblib's own threads rather than OpenMP's.
#
# 1, not a cap like ``compute_qpl._SEARCH_N_JOBS``: the search is 40 candidates x 5 folds
# of SVC on a projected kernel of a few hundred rows, so each fit is milliseconds and the
# outer loop already has every core busy. ``_tuning.py`` pins its own searches to 1 for
# the same reason.
_SEARCH_N_JOBS = 1


def create_svc_model(seed, scoring=None, max_iter=None):
    """The PQK head: a ``RandomizedSearchCV`` over ``SVC`` (40 candidates, 5 folds).

    Args:
        seed (int): Seeds the SVC and the candidate draw.
        scoring (callable or str or None): The search's ``scoring``; None is the SVC's
            own ``score`` (accuracy), as before.
        max_iter (int or None): The SVC's ``max_iter``; None leaves libsvm's -1.

    Returns:
        RandomizedSearchCV: Unfitted.
    """
    svc_param_distributions = {
        "C": [0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
    }

    # Initialize the SVC
    svc = SVC(random_state=seed) if max_iter is None else SVC(random_state=seed, max_iter=max_iter)

    # Initialize RandomizedSearchCV
    svc_model = RandomizedSearchCV(
        estimator=svc,
        param_distributions=svc_param_distributions,
        n_iter=40,
        cv=5,
        random_state=seed,
        n_jobs=_SEARCH_N_JOBS,
        scoring=scoring,
    )

    return svc_model


def compute_pqk_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='pqk_opt' explicitly, but a caller using the default would otherwise
    # produce a row labelled as untuned -- and modeleval infers `tuned` from this
    # very string, so the label and the parameter column would BOTH be wrong.
    model="pqk_opt",
    data_key="",
    encoding=None,
    primitive=None,
    entanglement=None,
    reps=None,
    data_map=None,
    *,
    n_trials=10,
    validation_split=0.25,
    validation=None,
    default_params=None,
    reseed=None,
):
    """Tune PQK's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_pqk_args`` config block -- a list is a choice, a
    ``{low, high}`` mapping is a range. It differs in how a candidate is scored: a
    quantum fit builds an n-by-n fidelity kernel by circuit simulation, so scoring by
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
        model (str): Name of the model being used, default is 'PQK'.
        data_key (str): Key for identifying the dataset.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        data_map (list): Data maps to search ('unit', 'qiskit'; see :func:`compute_pqk`).
            None leaves it at the default, 'unit'.
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
            is the inner-holdout search, unchanged. The head's hidden 40 x 5
            ``RandomizedSearchCV`` is then scored with the tuning metric and its SVC is
            capped at ``FOLD_SVC_MAX_ITER`` libsvm iterations (``head_scoring``/
            ``head_max_iter`` of :func:`compute_pqk`); it is part of one trial's fit, not
            extra budget.
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
    }

    fixed = {}
    if validation is not None:
        fixed = fold_fixed("pqk", candidates, default_params,
                           function_param_names(compute_pqk))
        # The head's hidden 40 x 5 search picks by the tuning metric, not accuracy, and
        # its libsvm fits are capped. Both are part of one trial's fit, not budget.
        # tuning_metric's plain name, so the trial log and parameter column stay plain.
        fixed.setdefault("head_scoring", tuning_metric(args))
        # Assigned outright: a configured head_max_iter below 1 is libsvm's "no cap".
        fixed["head_max_iter"] = fold_svc_max_iter(fixed.get("head_max_iter"))

    best_params = run_function_study(
        compute_pqk,
        build_search_space("pqk", candidates),
        X_train,
        y_train,
        args,
        model="pqk",
        n_trials=n_trials,
        seed=seed_from(args),
        validation_split=validation_split,
        data_key=data_key,
        fixed=fixed,
        validation=validation,
        default_params=default_params,
        reseed=reseed,
    )

    frame = compute_pqk(
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
    return record_tuned_params(frame, best_params, beg_time)
