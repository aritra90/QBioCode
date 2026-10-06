# ====== Base class imports ======
import time
from typing import Literal

import numpy as np

# from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.utils import algorithm_globals

# ====== Qiskit imports ======
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.circuit.library import qnn_circuit as QNNCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
    seed_from,
)
from qbiocode.learning.compute_fold import fold_fixed, function_param_names


#: The values of compute_qnn's ``readout``.
_READOUTS = ("global", "local")


def _readout_label(readout, num_qubits):
    """The Pauli label of the estimator's observable: Z...Z, or Z on qubit 0 only."""
    if readout == "global":
        return "Z" * num_qubits
    return "I" * (num_qubits - 1) + "Z"


def _local_bit(qc):
    """The measured bit of virtual qubit 0 in a transpiled circuit (its physical index)."""
    if qc.layout is None:
        return 0
    return qc.layout.final_index_layout()[0]


def compute_qnn(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="qnn",
    data_key="",
    primitive: Literal["estimator", "sampler"] = "sampler",
    verbose=False,
    local_optimizer: Literal["COBYLA", "L_BFGS_B", "GradientDescent"] = "COBYLA",
    maxiter=100,
    encoding="Z",
    entanglement="linear",
    reps=2,
    ansatz_type="amp",
    readout: Literal["global", "local"] = "global",
):
    """
    This function computes a Quantum Neural Network (QNN) model on the provided training data and evaluates it on the test data.
    It constructs a QNN circuit with a specified feature map and ansatz, optimizes it using a chosen optimizer, and fits the model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    The classifier is binary. The two labels of ``y_train`` are encoded for the network
    -- as -1/+1 for the estimator's expectation value, as parity outcome 0/1 for the
    sampler -- and predictions are decoded back, so ``y_predicted`` holds the caller's
    own labels in their own dtype, and the AUC score is oriented towards the larger one.

    The primitive always comes from :func:`qbiocode.utils.qutils.get_backend_session`,
    so ``args['seed']`` (and, for the sampler, ``args['shots']``) apply on
    ``'simulator'`` as on every other backend. ``'simulator'`` needs no transpilation;
    ``'simulator_aer'`` and the IBM backends also get an optimization-level-3 preset
    pass manager. The estimator runs at ``default_precision=0.0``: no artificial noise
    is added to its expectation values, so it is exact and reproducible on both
    simulators.

    Args:
        X_train (array-like): Training feature set.
        X_test (array-like): Test feature set.
        y_train (array-like): Training labels. Exactly two distinct values, of any
            sortable type; the larger is the positive class.
        y_test (array-like): Test labels.
        args (dict): Dictionary containing configuration parameters for the QNN.
        model (str, optional): Model type. Defaults to 'QNN'.
        data_key (str, optional): Key for the dataset. Defaults to ''.
        primitive (Literal['estimator', 'sampler'], optional): Type of primitive to use.
            'estimator' is an exact expectation value of Z on every qubit, classified by
            its sign; 'sampler' is the parity of the measured bitstring, estimated from
            ``args['shots']`` shots. Defaults to 'sampler'.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        local_optimizer (Literal['COBYLA', 'L_BFGS_B', 'GradientDescent'], optional): Optimizer to use. Defaults to 'COBYLA'.
        maxiter (int, optional): Maximum number of iterations for the optimizer. Defaults to 100.
        encoding (str, optional): Feature encoding method. Defaults to 'Z'.
        entanglement (str, optional): Entanglement strategy for the circuit. Defaults to 'linear'.
        reps (int, optional): Number of repetitions for the feature map and ansatz. Defaults to 2.
        ansatz_type (str, optional): Type of ansatz to use. Defaults to 'amp'.
        readout (Literal['global', 'local'], optional): What the class is read from.
            'global' (the default) is Z on every qubit -- the estimator's Z...Z
            expectation, the sampler's parity of the whole bitstring. 'local' is Z on
            qubit 0 only -- the estimator's <Z_0>, the sampler's bit of qubit 0. Both are placed
            through the transpiled circuit's layout. Neither has a bias term: with the
            Z feature map and a one-rep RealAmplitudes, for instance, <Z_0> sees x_0
            only through cos(2 * x_0), so 'local' separates classes only across a sign
            change of it.

    Returns:
        modeleval (dict): A dictionary containing the evaluation results, including accuracy, runtime, model parameters, and other relevant metrics.

    Raises:
        ValueError: If ``y_train`` does not hold exactly two classes, or ``readout``
            is not 'global' or 'local'.
    """
    beg_time = time.time()
    if readout not in _READOUTS:
        raise ValueError(
            f"compute_qnn readout must be one of {list(_READOUTS)}, got {readout!r}: "
            f"'global' reads Z on every qubit, 'local' Z on qubit 0 only."
        )

    # NeuralNetworkClassifier does not encode integer labels: it trains against y as
    # given and returns the network's own output space from predict() -- sign(raw) in
    # {-1, +1} for the one-output EstimatorQNN, the argmax column index for SamplerQNN.
    # Fitting it on a {0, 1} target therefore trained the estimator network with a
    # squared loss against the wrong targets and made it unable to ever predict class 0
    # (balanced accuracy <= 0.5 by construction). Both are fitted here on an explicit
    # encoding and decoded back to the caller's labels below. Checked first, before
    # any backend or runtime session is opened, so a bad target leaves nothing open.
    classes = np.unique(np.asarray(y_train))
    if classes.shape[0] != 2:
        raise ValueError(
            f"compute_qnn is a binary classifier: y_train must hold exactly two classes, "
            f"got {classes.shape[0]} ({classes.tolist()!r}). Both primitives map the "
            f"circuit to a two-way decision -- the estimator's sign, or the sampler's "
            f"parity -- so a multiclass target cannot be represented."
        )
    # Index 1 is the larger label, the class roc_auc_score treats as positive, so the
    # scores extract_binary_scores reads off below stay oriented towards it.
    y_index = (np.asarray(y_train) == classes[1]).astype(int)

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    # get ansatz
    ansatz = qutils.get_ansatz(
        ansatz_type=ansatz_type,
        feat_dimension=feature_map.num_qubits,
        reps=reps,
        entanglement=entanglement,
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    # Get Optimizer
    optimizer = qutils.get_optimizer(local_optimizer, max_iter=maxiter)

    # qc, input_params, weight_params = QNNCircuit(num_qubits=X_train.shape[1], feature_map=feature_map, ansatz=ansatz)
    qc, _, _ = QNNCircuit(num_qubits=X_train.shape[1], feature_map=feature_map, ansatz=ansatz)

    print(f"Currently running a quantum neural network (QNN) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")
    print(f"The number of ansatz parameters in your circuit is: {ansatz.num_parameters}")

    # 'simulator' hands over a seeded Statevector primitive, which runs any circuit as
    # it stands, so no transpilation is needed. Every other backend ('simulator_aer' and
    # the IBM ones) is a real backend object whose primitive expects ISA circuits.
    # Decided on the normalised name, like get_backend_session itself, so an alias
    # takes the same branch as the name it resolves to.
    needs_transpile = qutils.normalize_backend(args)["backend"] != "simulator"
    pm = (
        generate_preset_pass_manager(backend=backend, optimization_level=3)
        if needs_transpile
        else None
    )

    neural_network: EstimatorQNN | SamplerQNN

    if primitive == "estimator":
        # Z on every qubit, which is EstimatorQNN's own default -- built here because the
        # default is placed on physical qubits 0..n-1 of the transpiled circuit without
        # applying its layout, so on a device with a non-trivial layout it would measure
        # the wrong qubits. 'local' is Z on virtual qubit 0 (the rightmost label
        # character), mapped through the same layout.
        observable = SparsePauliOp(_readout_label(readout, qc.num_qubits))
        if pm is not None:
            qc = pm.run(qc)
            observable = observable.apply_layout(qc.layout)
        # default_precision=0.0: EstimatorQNN's default (0.015625) is passed as the
        # target precision of every run, and a V2 estimator honours it by adding
        # Gaussian noise of that standard deviation to each expectation value. On Aer's
        # EstimatorV2 that noise is unseeded, which is what made qnn on 'simulator_aer'
        # irreproducible; on a statevector it is noise with no physical meaning.
        neural_network = EstimatorQNN(
            circuit=qc,
            observables=observable,
            estimator=prim,
            pass_manager=pm,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            default_precision=0.0,
        )

        # QNN maps inputs to [-1, +1]
        neural_network.forward(
            X_train[0, :], algorithm_globals.random.random(neural_network.num_weights)
        )
        # The network's range: classes[0] -> -1, classes[1] -> +1.
        y_fit = 2 * y_index - 1
    else:
        # parity maps bitstrings to 0 or 1; 'local' reads virtual qubit 0's bit alone,
        # the sampler's counterpart of <Z_0>. SamplerQNN transpiles first and only then
        # measures every physical qubit, so bit k is physical qubit k. The global parity
        # needs no layout (idle ancillas stay |0> and add nothing to it); the local bit
        # is found through the layout of a circuit transpiled here, which SamplerQNN
        # then keeps as it stands.
        if readout == "global":
            def parity(x):
                return "{:b}".format(x).count("1") % 2
        else:
            bit = 0
            if pm is not None:
                qc = pm.run(qc)
                bit = _local_bit(qc)

            def parity(x):
                return (x >> bit) & 1

        output_shape = (
            2  # corresponds to the number of classes, possible outcomes of the (parity) mapping
        )
        neural_network = SamplerQNN(
            circuit=qc,
            sampler=prim,
            interpret=parity,
            output_shape=output_shape,
            pass_manager=pm,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )
        # Parity outcome k is class classes[k].
        y_fit = y_index

    # construct classifier
    qnn = NeuralNetworkClassifier(neural_network=neural_network, optimizer=optimizer)

    # fit classifier to data
    model_fit = qnn.fit(X_train, y_fit)
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        "ansatz": ansatz.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "optimizer_params": optimizer.settings,
        # Add other hyperparameters as needed
    }
    # Only when not the default, so 'global' rows stay identical to earlier results.
    if readout != "global":
        hyperparameters["readout"] = readout
    model_params = hyperparameters
    # Decoded back to the caller's labels, in their own dtype: raw > 0 (estimator) or
    # parity 1 (sampler) is classes[1], anything else classes[0].
    raw_predicted = np.asarray(qnn.predict(X_test)).reshape(-1)
    y_predicted = classes[(raw_predicted > 0).astype(int)]
    # `auc` is computed from these scores alone, never from y_predicted.
    # NeuralNetworkClassifier.predict_proba returns the network's forward pass, and its
    # shape depends on which primitive was chosen above -- both are rankings, and
    # extract_binary_scores handles each:
    #   * sampler (the default): SamplerQNN with `interpret=parity` and
    #     `output_shape=2`, so (n, 2) probabilities over the two parity outcomes; column
    #     1 is parity 1, which the encoding above made classes[1];
    #   * estimator: EstimatorQNN, so a single (n, 1) expectation value in [-1, +1] --
    #     not a probability, but exactly the quantity `predict` takes the sign of, and
    #     +1 was classes[1] in training.
    # Either way higher means classes[1], the larger label, which is the class
    # roc_auc_score treats as positive.
    # Scored before the session is closed: the forward pass runs the circuit again.
    y_score = extract_binary_scores(qnn, X_test)

    if not isinstance(session, type(None)):
        session.close()

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


def compute_qnn_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='qnn_opt' explicitly, but a caller using the default would otherwise
    # produce a row labelled as untuned -- and modeleval infers `tuned` from this
    # very string, so the label and the parameter column would BOTH be wrong.
    model="qnn_opt",
    data_key="",
    primitive=None,
    local_optimizer=None,
    maxiter=None,
    encoding=None,
    entanglement=None,
    reps=None,
    ansatz_type=None,
    *,
    n_trials=10,
    validation_split=0.25,
    readout=None,
    validation=None,
    default_params=None,
    reseed=None,
):
    """Tune QNN's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_qnn_args`` config block -- a list is a choice, a
    ``{low, high}`` mapping is a range. It differs in how a candidate is scored: a
    quantum fit simulates the variational circuit at every optimizer step, so scoring
    by k-fold cross-validation would multiply an already expensive search by k. Each trial
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
        model (str): Name of the model being used, default is 'QNN'.
        data_key (str): Key for identifying the dataset.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        local_optimizer (list or dict): Optimizers to search ('COBYLA', 'L_BFGS_B', 'GradientDescent'). None leaves it at the default.
        maxiter (list or dict): Optimizer iteration budgets to search. None leaves it at the default.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        ansatz_type (list or dict): Ansatz types to search ('amp', ...). None leaves it at the default.
        n_trials (int): Trial budget, default 10 -- an order of magnitude below the
            classical default because each trial is a quantum fit. Lowered
            automatically when the configured values describe fewer combinations.
        validation_split (float): Fraction of the training data held out to score
            candidates on, default 0.25.
        readout (list or dict): Readouts to search ('global', 'local'; see
            :func:`compute_qnn`). None leaves it at the default.
        validation (ValidationSplit or None): ``split_mode: manifest``. Every trial is
            then one fit on ``validation.X_fit`` scored on ``validation.X_val``
            (``validation_split`` is ignored), trials write no kernel dumps, and the
            unsearched keys of ``default_params`` are fixed for the trials and the refit
            (see :func:`qbiocode.learning.compute_fold.fold_fixed`). None (the default)
            is the inner-holdout search, unchanged.
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
        "primitive": primitive,
        "local_optimizer": local_optimizer,
        "maxiter": maxiter,
        "encoding": encoding,
        "entanglement": entanglement,
        "reps": reps,
        "ansatz_type": ansatz_type,
        "readout": readout,
    }

    fixed = {}
    if validation is not None:
        fixed = fold_fixed("qnn", candidates, default_params,
                           function_param_names(compute_qnn))
    best_params = run_function_study(
        compute_qnn,
        build_search_space("qnn", candidates),
        X_train,
        y_train,
        args,
        model="qnn",
        n_trials=n_trials,
        seed=seed_from(args),
        validation_split=validation_split,
        data_key=data_key,
        fixed=fixed,
        validation=validation,
        default_params=default_params,
        reseed=reseed,
    )

    frame = compute_qnn(
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
