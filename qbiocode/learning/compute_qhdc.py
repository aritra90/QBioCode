"""hdlib quantum hyperdimensional classification on QProfiler's supplied split.

hdlib is imported only when qHDC is selected. The optional PyPI dependency is
listed in requirements/requirements-qhdc.txt. Training and optional retraining use
the already-scaled split supplied by QProfiler.
"""

import sys
import time
from numbers import Integral, Real

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

from qbiocode.evaluation.model_evaluation import modeleval
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
)
from qbiocode.utils.ibm_account import instantiate_runtime_service


def _load_classifier():
    if sys.version_info < (3, 11):
        raise ImportError("qHDC requires Python 3.11 or 3.12; other models support Python 3.10.")
    try:
        from hdlib.model import QuantumClassificationModel
    except ImportError as exc:
        raise ImportError(
            "qHDC requires hdlib's quantum classifier. From the QBioCode checkout run "
            "'python -m pip install -r requirements/requirements-qhdc.txt'."
        ) from exc
    return QuantumClassificationModel


def _positive_integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"qHDC {name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _backend(args, num_qubits, device):
    """Use the same account resolver as the other learners, without a spare session.

    hdlib opens its own prediction session. Passing it a backend from QBioCode's
    service preserves named accounts, custom URLs and explicit config overrides.
    """
    from qiskit_aer import AerSimulator

    name = args.get("backend", "simulator")
    if name in ("simulator", "simulator_aer"):
        return AerSimulator(device=device)
    if not isinstance(name, str) or not (name.startswith("ibm_") or name.startswith("noisy_ibm_")):
        raise ValueError(
            f"Unsupported qHDC backend {name!r}; use 'simulator', 'simulator_aer', "
            "'ibm_least', an 'ibm_*' device, or 'noisy_ibm_*'."
        )
    service = instantiate_runtime_service(args)
    if name == "ibm_least":
        backend = service.least_busy(simulator=False, operational=True, min_num_qubits=num_qubits)
    else:
        backend = service.backend(name=name.removeprefix("noisy_"))
    if backend.num_qubits < num_qubits:
        raise ValueError(f"qHDC needs {num_qubits} qubits; {name!r} has {backend.num_qubits}.")
    if name.startswith("noisy_"):
        return AerSimulator.from_backend(backend, device=device)
    return backend


def compute_qhdc(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="qhdc",
    data_key="",
    *,
    size=64,
    levels=2,
    retrain_epochs=0,
    learning_rate=1.0,
    early_stop=True,
    device="CPU",
    seed=None,
):
    """Fit qHDC and return the standard QBioCode metrics/prediction DataFrame.

    ``X_train`` and ``X_test`` are the split (and optional embedding) supplied by
    QProfiler. QProfiler fits the configured scaler on the training side and applies
    it to the test side before dispatching every model. No rows are resampled,
    reordered, or scaled here.

    ``size`` is a power-of-two hypervector dimension, independent of feature
    count; the current hdlib classifier uses log2(size) qubits. ``levels`` is the
    number of quantization levels. Retraining defaults off and, when enabled,
    uses only the training set. AUC uses positive minus negative class similarity
    (a decision score, not a calibrated probability).

    ``args`` supplies backend, shots (default 1024), and the usual IBM account
    settings. The model seed defaults to q_seed, then seed, then 42. hdlib handles
    hardware sessions and its own mitigation; QBioCode's resil_level does not
    configure that mitigation. PQK checkpoint/resume is not available for qHDC.
    """
    started = time.time()
    size = _positive_integer("size", size, 2)
    if size & (size - 1):
        raise ValueError("qHDC size must be a power of two (for example 16, 32, 64).")
    levels = _positive_integer("levels", levels, 2)
    retrain_epochs = _positive_integer("retrain_epochs", retrain_epochs, 0)
    shots = _positive_integer("shots", args.get("shots", 1024))
    seed = _positive_integer(
        "seed", args.get("q_seed", args.get("seed", 42)) if seed is None else seed, 0
    )
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("qHDC learning_rate must be finite and > 0.")
    if not isinstance(early_stop, bool):
        raise ValueError("qHDC early_stop must be a boolean.")
    if device not in ("CPU", "GPU"):
        raise ValueError("qHDC device must be 'CPU' or 'GPU'.")
    X_train, y_train = check_X_y(X_train, y_train, dtype=float)
    X_test, y_test = check_X_y(X_test, y_test, dtype=float)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("qHDC train and test arrays must have the same feature count.")
    classes = np.unique(y_train)
    if len(classes) != 2 or not np.isin(y_test, classes).all():
        raise ValueError("qHDC requires two training classes and test labels from those classes.")
    train, test = X_train, X_test

    classifier = _load_classifier()
    learner = classifier(size=size, levels=levels, seed=seed, shots=shots)
    # hdlib's constructor attempts GPU Aer without checking whether a job can
    # actually run on a GPU. Replace it explicitly, also for CPU-only installs.
    learner.backend = _backend(args, size.bit_length() - 1, device)
    train_points, train_labels = train.tolist(), y_train.tolist()
    learner.fit(train_points, train_labels)
    if retrain_epochs:
        learner.retrain(
            train_points,
            train_labels,
            epochs=retrain_epochs,
            lr=float(learning_rate),
            early_stop=early_stop,
            verbose=verbose,
        )
    predictions, similarities = learner.predict(test.tolist())
    predictions = np.asarray(predictions)
    similarities = check_array(similarities, dtype=float)
    model_classes = np.asarray(learner.classes_)
    if predictions.shape != y_test.shape or similarities.shape != (len(y_test), 2):
        raise ValueError("hdlib returned predictions/similarities that do not match the test set.")
    if model_classes.shape != (2,) or set(model_classes) != set(classes):
        raise ValueError("hdlib's class order does not match the training labels.")
    positive = int(np.flatnonzero(model_classes == classes[1])[0])
    scores = similarities[:, positive] - similarities[:, 1 - positive]
    params = {
        "size": size,
        "levels": levels,
        "retrain_epochs": retrain_epochs,
        "learning_rate": float(learning_rate),
        "early_stop": early_stop,
        "device": device,
        "seed": seed,
        "shots": shots,
        "backend": args.get("backend", "simulator"),
        "hdlib_version": getattr(learner, "version", "unknown"),
        "auc_score": "positive_minus_negative_similarity",
    }
    return modeleval(
        y_test,
        predictions,
        started,
        params,
        args,
        model=model,
        verbose=verbose,
        y_score=scores,
    )


def compute_qhdc_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="qhdc_opt",
    data_key="",
    *,
    size=None,
    levels=None,
    retrain_epochs=None,
    learning_rate=None,
    early_stop=True,
    device="CPU",
    seed=None,
    n_trials=10,
    validation_split=0.25,
):
    """Tune on an inner training split, refit on all training rows, then test.

    Search size/levels/epochs/rate with the same Optuna syntax as the other
    quantum models. Backend and execution controls stay fixed; QProfiler scaling
    remains outside the learner and is never tuned.
    The shared tuner requires allow_hardware_tuning for real hardware trials.
    """
    started = time.time()
    candidates = dict(
        size=size, levels=levels, retrain_epochs=retrain_epochs, learning_rate=learning_rate
    )
    fixed = dict(early_stop=early_stop, device=device, seed=seed)
    best = run_function_study(
        compute_qhdc,
        build_search_space("qhdc", candidates),
        X_train,
        y_train,
        args,
        model="qhdc",
        n_trials=n_trials,
        seed=args.get("seed", 42),
        validation_split=validation_split,
        fixed=fixed,
    )
    frame = compute_qhdc(
        X_train,
        X_test,
        y_train,
        y_test,
        args,
        verbose=verbose,
        model=model,
        data_key=data_key,
        **fixed,
        **best,
    )
    return record_tuned_params(frame, best, started)
