import logging
import os
import time
from typing import Literal

import numpy as np
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.model_selection import GridSearchCV

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)
from qbiocode.learning._grid import warn_ignored_hyperparameter

# ====== Scikit-learn imports ======


# ====== Qiskit imports ======

logger = logging.getLogger(__name__)


class StatevectorFidelityKernel(FidelityQuantumKernel):
    """``FidelityQuantumKernel`` over ``ComputeUncompute(StatevectorSampler(seed=s))``, the
    same matrix entry for entry, from ``n`` statevectors instead of ``O(n^2)`` circuits.

    The circuit path simulates one compute-uncompute circuit per kernel entry and samples
    it: ``n_tr(n_tr-1)/2 + n_te*n_tr`` circuits for a fit, ``n_te*n_tr`` more for each of
    predict and decision_function -- 46,548 at heart's 216/54 split, ~41 min per
    fit+predict+score at 13 qubits. Yet every entry is a function of two statevectors, and so is its sampling
    noise: ``StatevectorSampler`` reseeds every pub with the same ``seed``
    (``final_state.seed(self._seed)`` in ``_run_pub``), and numpy's ``Generator.choice``
    picks outcome ``|0...0>`` exactly when a uniform falls below its probability. So each
    entry is ``#{k : u_k < F_ij} / shots`` for one fixed set of ``shots`` uniforms
    ``u = default_rng(seed).random(shots)``, where ``F_ij = |<psi_i|psi_j>|^2``. This
    computes that from the statevectors directly, with the circuit path's diagonal (1,
    not evaluated), symmetry (upper triangle, mirrored) and PSD projection (the base
    class's own ``_make_psd``).

    Bit-identical (max |dK| = 0) to ``FidelityQuantumKernel`` on all 36 encoding x reps x
    entanglement settings the pilot's ``gridsearch_qsvc_args`` searches
    (tests/test_statevector_kernel.py), and 147-370x faster at heart's size. Because that identity rests on two library internals -- per-pub
    reseeding, and how ``choice`` consumes its uniforms -- it is not taken on trust: the
    first ``evaluate`` also runs a few of its entries as real circuits through the
    configured fidelity and compares them exactly. On any mismatch this instance logs a
    warning and falls back to the circuit path for good, and ``evaluation`` (recorded in
    the results as ``kernel_evaluation``) says which path produced the matrix.

    The noise is worth knowing about in its own right: because every entry reuses the
    same uniforms, "1024 shots" on this backend is not independent binomial noise per
    entry but one fixed monotone quantisation of the exact kernel to a 1/shots grid.
    """

    #: Held in memory at once: ``n_x + n_y`` statevectors of ``2**q`` complex128. Past
    #: this, the circuit path's one statevector at a time is the one that fits.
    MAX_STATE_BYTES = 2 * 1024 ** 3

    def __init__(self, *, feature_map, fidelity, shots, seed, n_check=4):
        super().__init__(feature_map=feature_map, fidelity=fidelity)
        # What Generator.choice draws for one pub: random(size) from a fresh default_rng.
        self._uniforms = np.sort(np.random.default_rng(seed).random(shots))
        self._shots = shots
        self._n_check = n_check
        self._verified = False
        self.evaluation = "statevector"

    @staticmethod
    def applies(fidelity, sampler):
        """Whether the circuit path's noise is the fixed-uniforms function above.

        Only a global-fidelity ``ComputeUncompute`` with no run options of its own, on a
        ``StatevectorSampler`` seeded with an integer: ``seed=None`` draws fresh entropy
        per pub and a ``Generator`` seed is consumed pub after pub, so neither is a
        function of the entry alone.
        """
        seed = getattr(sampler, "seed", None)
        # What ComputeUncompute._run forwards to sampler.run. (Not `opts or ...`: an
        # empty Options is falsy.)
        opts = getattr(fidelity, "_default_options", None)
        return (
            isinstance(fidelity, ComputeUncompute)
            and not getattr(fidelity, "_local", True)
            and not (vars(opts) if opts is not None else {})
            and isinstance(sampler, StatevectorSampler)
            and isinstance(seed, (int, np.integer))
            and not isinstance(seed, bool)
        )

    def _states(self, X):
        return np.array([Statevector(self._feature_map.assign_parameters(x)).data for x in X])

    def evaluate(self, x_vec, y_vec=None):
        if self.evaluation == "circuits":
            return super().evaluate(x_vec, y_vec)
        x_vec, y_vec = self._validate_input(x_vec, y_vec)
        # The circuit path's own test, so both paths take the same branch.
        is_symmetric = y_vec is None or np.array_equal(x_vec, y_vec)
        if y_vec is None:
            y_vec = x_vec
        n_states = len(x_vec) + (0 if is_symmetric else len(y_vec))
        if n_states * 2 ** self._feature_map.num_qubits * 16 > self.MAX_STATE_BYTES:
            return super().evaluate(x_vec, y_vec)

        Sx = self._states(x_vec)
        Sy = Sx if is_symmetric else self._states(y_vec)
        F = np.abs(Sx.conj() @ Sy.T) ** 2
        # side='left': the number of uniforms strictly below F, as choice's searchsorted
        # (side='right' on the cdf) selects outcome 0 exactly when u < p_0.
        K = np.searchsorted(self._uniforms, F, side="left") / self._shots
        if not self._verified and not self._agrees_with_circuits(x_vec, y_vec, F, K, is_symmetric):
            return super().evaluate(x_vec, y_vec)
        if is_symmetric:
            iu = np.triu_indices(len(x_vec), 1)
            upper = K[iu]
            K = np.ones_like(K)
            K[iu], K[iu[::-1]] = upper, upper
            if self._enforce_psd:
                K = self._make_psd(K)
        return K

    def _agrees_with_circuits(self, x_vec, y_vec, F, K, is_symmetric):
        """Run a few entries as real circuits; True when they match to the last bit."""
        if is_symmetric:
            rows, cols = np.triu_indices(len(x_vec), 1)
        else:
            rows, cols = (a.ravel() for a in np.indices(F.shape))
        if not len(rows):
            return True  # nothing off-diagonal to check yet; the next call checks
        # Interior quantiles of F: an entry at 0 or 1 samples to 0 or 1 whatever the
        # uniforms are, so it would check nothing.
        order = np.argsort(F[rows, cols], kind="stable")
        at = ((np.arange(self._n_check) + 1) / (self._n_check + 1) * (len(order) - 1)).round()
        pick = np.unique(order[at.astype(int)])
        r, c = rows[pick], cols[pick]
        fm = [self._feature_map] * len(pick)
        circuit = np.asarray(self._fidelity.run(fm, fm, x_vec[r], y_vec[c]).result().fidelities)
        self._verified = True
        if np.array_equal(circuit, K[r, c]):
            return True
        logger.warning(
            "StatevectorFidelityKernel: %d of %d checked entries differ from the circuit "
            "path (max |dK| %.3g) -- a qiskit or numpy change has broken the identity it "
            "relies on. Falling back to circuits for this kernel: slower, same results.",
            int((circuit != K[r, c]).sum()), len(pick), float(np.abs(circuit - K[r, c]).max()),
        )
        self.evaluation = "circuits"
        return False


def _fidelity_kernel(feature_map, fidelity, sampler, args):
    """The circuit path's kernel, evaluated from statevectors wherever that is exact.

    ``qsvc_kernel_evaluation: 'circuits'`` in the config forces the circuit path.
    """
    if (
        args.get("qsvc_kernel_evaluation", "statevector") != "circuits"
        and StatevectorFidelityKernel.applies(fidelity, sampler)
    ):
        return StatevectorFidelityKernel(
            feature_map=feature_map, fidelity=fidelity, shots=sampler.default_shots,
            seed=sampler.seed,
        )
    return FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)


def _dump_gram_matrices(qkernel, args, data_key, model, y_train, X_train, y_test):
    """Patch ``qkernel.evaluate`` to keep the Gram matrices the fit already computes.

    The fidelity kernel is the single most expensive object in the benchmark --
    ``n_tr(n_tr-1)/2 + n_te*n_tr`` circuits, which is the arm that sets a job's wall clock.
    ``QSVC`` hands ``quantum_kernel.evaluate`` to libsvm as a callable kernel, and sklearn
    does not retain what a callable returns. So the matrix is built, used once, and freed:
    every kernel-level diagnostic (alignment, geometric separation against the classical
    kernel, RKHS margin) becomes unrecoverable, and recovering it later means paying those
    circuits a second time.

    Recording is therefore free and dropping it is not. This adds no circuit: it wraps the
    bound method and keeps a reference to what already came back. Cost is one O(n^2) float64
    array on disk -- 2.6 MB at n=569, the widest split in the pilot.

    Wrapping must happen *before* ``QSVC.__init__``, which captures ``evaluate`` at
    construction; patching afterwards records nothing while appearing to work.

    Returns the dict the caller should fill after fit, or None when dumping is off.
    """
    dump_dir = args.get("kernel_dump_dir")
    if not dump_dir:
        return None
    os.makedirs(dump_dir, exist_ok=True)
    seen = []
    evaluate = qkernel.evaluate

    def recording_evaluate(x_vec, y_vec=None):
        K = evaluate(x_vec, y_vec)
        seen.append(np.asarray(K))
        return K

    qkernel.evaluate = recording_evaluate
    return {"dir": dump_dir, "seen": seen, "data_key": data_key, "model": model,
            "y_train": np.asarray(y_train), "X_train": np.asarray(X_train),
            "y_test": np.asarray(y_test)}


def _write_gram_matrices(handle):
    """Write the recorded train Gram and the label vector it must be read against.

    The train kernel is the square matrix; ``QSVC`` also evaluates a rectangular
    ``(n_test, n_train)`` block during predict, and telling them apart by shape is more
    robust than assuming call order, which differs between QSVC and PegasosQSVC.

    ``y_train`` travels with the matrix because alignment is meaningless without the exact
    label order the rows are in, and that order is a property of the split, not the dataset.

    ``X_train`` travels with it for the same reason and one more: every kernel-level
    comparison is *against a classical kernel*, and the classical kernel is a function of
    the embedded features. Nothing else in the run persists them -- ``qprofiler`` pickles
    only the results frame -- so without this array the quantum Gram is recoverable and
    ``g(K_c || K_q)`` still is not. It costs ``n_tr x d`` floats next to an ``n_tr^2``
    matrix, i.e. nothing at these widths.
    """
    if not handle:
        return
    square = [K for K in handle["seen"] if K.ndim == 2 and K.shape[0] == K.shape[1]]
    if not square:
        return
    K_train = max(square, key=lambda K: K.shape[0])
    payload = {
        "K_train": K_train,
        "y_train": handle["y_train"],
        "X_train": handle["X_train"],
        "y_test": handle["y_test"],
    }
    # The (n_test, n_train) block predict() evaluated, when there is one. Kept because it
    # is the only way to score the recovered kernel on held-out rows without re-running the
    # circuits, and it is already paid for. Identified by shape, not by call order: a square
    # test block would be ambiguous, so require shape[0] != shape[1] and a matching n_train.
    rect = [
        K for K in handle["seen"]
        if K.ndim == 2 and K.shape[0] != K.shape[1] and K.shape[1] == K_train.shape[0]
    ]
    if rect:
        payload["K_test"] = max(rect, key=lambda K: K.shape[0])
    stem = os.path.join(handle["dir"], f"gram_{handle['model']}_{handle['data_key']}")
    np.savez_compressed(stem + ".npz", **payload)


def compute_qsvc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    model="qsvc",
    data_key="",
    C=1,
    gamma="scale",
    pegasos=False,
    encoding: Literal["ZZ", "Z", "P"] = "ZZ",
    entanglement="linear",
    primitive="sampler",
    reps=2,
    verbose=False,
    local_optimizer="",
):
    """
    This function computes a quantum support vector classifier (QSVC) using the Qiskit Machine Learning library.
    It takes training and testing datasets, along with various parameters to configure the QSVC model.
    It initializes the quantum feature map, sets up the backend and session, and fits the QSVC model to the training data.
    It then predicts the labels for the test data and evaluates the model's performance.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.

    Args:
        X_train (np.ndarray): Training feature set.
        X_test (np.ndarray): Testing feature set.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        args (dict): Dictionary containing arguments for the quantum backend and other settings.
        model (str): Model type, default is 'QSVC'.
        data_key (str): Key for the dataset, default is an empty string.
        C (float): Regularization parameter for the SVM, default is 1.
        gamma (str or float): Kernel coefficient, default is 'scale'.
        pegasos (bool): Whether to use Pegasos QSVC, default is False.
        encoding (str): Feature map encoding type, options are 'ZZ', 'Z', or 'P', default is 'ZZ'.
        entanglement (str): Entanglement strategy for the feature map, default is 'linear'.
        primitive (str): Primitive type to use, default is 'sampler'.
        reps (int): Number of repetitions for the feature map, default is 2.
        verbose (bool): Whether to print additional information, default is False.

    Returns:
        modeleval (dict): A dictionary containing the evaluation results, including accuracy, runtime, model parameters, and other relevant metrics.
    """
    beg_time = time.time()

    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(
        feature_map=encoding, feat_dimension=X_train.shape[1], reps=reps, entanglement=entanglement
    )

    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(
        args, primitive, num_qubits=feature_map.num_qubits
    )

    print(f"Currently running a quantum support vector classifier (QSVC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")

    if "simulator" == args["backend"]:
        fidelity = ComputeUncompute(sampler=prim)
    else:
        # Need to instatiate a basic pass manager to store the chosen hardware backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        fidelity = ComputeUncompute(
            sampler=prim, pass_manager=pm
        )  # , num_virtual_qubits = feature_map.num_qubits )

    Qkernel = _fidelity_kernel(feature_map, fidelity, prim, args)
    # Before QSVC is constructed: it binds Qkernel.evaluate at __init__.
    _gram = _dump_gram_matrices(
        Qkernel, args, data_key, model, y_train, X_train, y_test
    )
    if pegasos == True:
        qsvc = PegasosQSVC(C=C, quantum_kernel=Qkernel)
    else:
        qsvc = QSVC(C=C, gamma=gamma, quantum_kernel=Qkernel)

    model_fit = qsvc.fit(X_train, y_train)
    # model_params = model_fit.get_params()
    hyperparameters = {
        "feature_map": feature_map.__class__.__name__,
        # The estimator, not how it was evaluated: StatevectorFidelityKernel is the same
        # FidelityQuantumKernel matrix, and says which path produced it separately.
        "quantum_kernel": (
            FidelityQuantumKernel.__name__
            if isinstance(Qkernel, FidelityQuantumKernel)
            else Qkernel.__class__.__name__
        ),
        "kernel_evaluation": getattr(Qkernel, "evaluation", "circuits"),
        "C": C,
        "gamma": gamma,
    }
    model_params = hyperparameters
    y_predicted = qsvc.predict(X_test)
    # `auc` is computed from these scores alone, never from y_predicted. Both kernel
    # classifiers offer a ranking: QSVC subclasses sklearn's SVC and inherits its
    # decision_function, and PegasosQSVC publishes predict_proba (a sigmoid of its own
    # decision_function, so the same ordering). Scored before the session is closed --
    # on hardware the primitive is what evaluates the kernel entries this needs.
    y_score = extract_binary_scores(qsvc, X_test)
    _write_gram_matrices(_gram)

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


def compute_qsvc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='qsvc_opt' explicitly, but a caller using the default would otherwise
    # produce a row labelled as untuned -- and modeleval infers `tuned` from this
    # very string, so the label and the parameter column would BOTH be wrong.
    model="qsvc_opt",
    data_key="",
    C=None,
    gamma=None,
    pegasos=None,
    encoding=None,
    entanglement=None,
    primitive=None,
    reps=None,
    local_optimizer=None,
    *,
    n_trials=10,
    validation_split=0.25,
):
    """Tune QSVC's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, and driven by
    the same ``gridsearch_qsvc_args`` config block -- a list is a choice, a
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
        model (str): Name of the model being used, default is 'QSVC'.
        data_key (str): Key for identifying the dataset.
        C (list or dict): Regularization strength values to search. None leaves it at the default.
        gamma (list or dict): Kernel coefficient values to search. None leaves it at the default.
        pegasos (list or dict): Whether to use the Pegasos QSVC solver. None leaves it at the default.
        encoding (list or dict): Feature-map values to search ('Z', 'ZZ', 'P'). None leaves it at the default.
        entanglement (list or dict): Entanglement patterns to search ('linear', 'full', ...). None leaves it at the default.
        primitive (list or dict): Qiskit primitives to search ('sampler', 'estimator'). None leaves it at the default.
        reps (list or dict): Feature-map repetition counts to search. None leaves it at the default.
        local_optimizer (list or dict): Accepted so a shared config block can name it,
            and warned about -- compute_qsvc takes the parameter and never reads it.
        n_trials (int): Trial budget, default 10 -- an order of magnitude below the
            classical default because each trial is a quantum fit. Lowered
            automatically when the configured values describe fewer combinations.
        validation_split (float): Fraction of the training data held out to score
            candidates on, default 0.25.

    Returns:
        modeleval (dict): The evaluation of the model at the best hyperparameters found,
        with the tuned values recorded in the results frame and the reported time
        covering the whole search rather than only the final fit.
    """
    beg_time = time.time()
    # Accepted by compute_qsvc and never read, so a grid over it would
    # multiply the trials while every one returned the same model.
    if local_optimizer:
        warn_ignored_hyperparameter("qsvc", "local_optimizer", "QSVC does not read -- the kernel is fitted by libsvm, not an optimizer.")

    candidates = {
        "C": C,
        "gamma": gamma,
        "pegasos": pegasos,
        "encoding": encoding,
        "entanglement": entanglement,
        "primitive": primitive,
        "reps": reps,
    }

    best_params = run_function_study(
        compute_qsvc,
        build_search_space("qsvc", candidates),
        X_train,
        y_train,
        args,
        model="qsvc",
        n_trials=n_trials,
        seed=args.get("seed") if isinstance(args, dict) else None,
        validation_split=validation_split,
        data_key=data_key,
    )

    frame = compute_qsvc(
        X_train,
        X_test,
        y_train,
        y_test,
        args,
        model=model,
        data_key=data_key,
        verbose=verbose,
        **best_params,
    )
    return record_tuned_params(frame, best_params, beg_time)
