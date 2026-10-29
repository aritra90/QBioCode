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

"""Quantum extreme learning machine: a random quantum reservoir and a ridge readout.

What it is
----------
One state preparation per row through *the data feature map followed by a fixed random
unitary*, then a configurable set of Pauli expectation values read off that state, then
a closed-form linear classifier on those features::

    x -> feature map -> reservoir (fixed, untrained) -> F observables -> ridge

The extreme-learning-machine contract is that the representation is random and
**untrained** and only the output layer is fitted -- here by least squares, which has a
closed form. That is the whole difference in kind from :mod:`qbiocode.learning.compute_vqc`
and :mod:`qbiocode.learning.compute_qnn`, which train their circuit parameters.

How it relates to PQK and QPL
-----------------------------
All three project the rows through a quantum circuit and fit a classical head on the
projection. They differ in what is fixed and what the head is:

=========  ==================================  ==========  ===============================
model      circuit                             features    head
=========  ==================================  ==========  ===============================
pqk        data feature map                     ``3n``     SVC, 40 x 5 ``RandomizedSearchCV``
qpl        data feature map                     ``3n``     up to 6 heads, each searched
qelm       feature map + random reservoir       ``F``      ridge, closed form
=========  ==================================  ==========  ===============================

The readout is deliberately the weakest of the three. A win for a strong head is hard to
attribute -- it may be the head extracting from an ordinary representation -- whereas a
ridge on random features can only do well if the *representation* carries the signal.
That is the question this arm exists to answer, and it is why the head is not a search
over model families.

**The ablation.** ``reservoir_reps: 0`` builds no reservoir at all, and with
``measurement: 'xyz'`` the features are then *exactly* PQK's projection (to the bit; see
:func:`qbiocode.utils.qelm_features.qelm_reservoir` for why qiskit's own ``reps=0`` would
not do). So the three cells

* ``reservoir_reps: 0`` -- PQK's representation, ridge readout
* ``pqk`` -- PQK's representation, SVC readout
* ``reservoir_reps: 1+`` -- reservoir representation, ridge readout

separate the effect of the readout from the effect of the reservoir. Comparing only the
outer two would confound them.

Where the data comes from
-------------------------
Exactly where every other model's does. ``model_run`` hands this function the same
post-embedding ``X_train``/``X_test`` it hands ``lr``, ``svc``, ``tabpfn``, ``qsvc``,
``pqk`` and ``qpl``; scaling, feature selection, the embedding and ``n_components`` all
happen above it in ``qprofiler`` and are not this module's business. Nothing here is a
QELM-specific feature path, which is what makes the arm comparable to the others at all.

The seed is configuration, not a hyperparameter
-----------------------------------------------
``reservoir_seed`` fixes the reservoir and the random observable pool, and it is read
from ``args`` -- never searched. Tuning it would pick whichever random draw happened to
suit the validation holdout, so the reported score would be a maximum over the budget
rather than the performance of one reservoir. See
:mod:`qbiocode.utils.qelm_features`, and :func:`compute_qelm_opt` for what *is* searched.
"""

# ====== Base class imports ======
import hashlib
import json
import logging
import os
import time
import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

import qbiocode.utils.qutils as qutils

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import extract_binary_scores, modeleval
from qbiocode.learning._grid import to_plain
from qbiocode.learning._tuning import (
    build_search_space,
    record_tuned_params,
    run_function_study,
    seed_from,
    tuning_metric,
)
from qbiocode.learning.compute_fold import fold_fixed, function_param_names
from qbiocode.learning.compute_pqk import _resolve_data_map
from qbiocode.utils.qelm_features import (
    MEASUREMENTS,
    observables_fingerprint,
    qelm_observables,
    qelm_reservoir,
)

logger = logging.getLogger(__name__)

#: Config key holding the seed of the reservoir and the random observable pool. Absent,
#: it falls back to ``q_seed`` and then ``seed``, so an ordinary config needs no new key
#: and a run is still reproducible. Deliberately read from ``args`` rather than taken as
#: a hyperparameter -- see the module docstring.
RESERVOIR_SEED_KEY = "qelm_reservoir_seed"

#: ``alpha`` values :func:`create_ridge_model` selects among by generalised
#: cross-validation. Six decades, because the useful ridge penalty moves with the number
#: of features and that is itself tuned here: at ``n_features`` well above the row count
#: -- routine on this corpus, where ``colon_cancer`` has 62 rows -- the penalty is most
#: of the model, and a grid that stopped at 1.0 would be pinned at its top end.
DEFAULT_RIDGE_ALPHAS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)

#: How ``n_features`` may be written. An int is that many observables; a float is a
#: multiple of the qubit count (``3.0`` on 8 features is 24), which is what makes one
#: config block mean the same thing across datasets of different widths -- an absolute
#: count cannot, since ``n_components: 3`` cannot yield 36 single-qubit observables.
_FEATURE_COUNT_TYPES = "an int (a count) or a float (a multiple of the qubit count)"


def resolve_reservoir_seed(args):
    """The reservoir seed of a run: ``qelm_reservoir_seed``, else ``q_seed``, else ``seed``.

    Falling back rather than defaulting to a constant keeps a QELM run reproducible from
    the seeds a config already sets, and keeps the reservoir tied to the run's own
    randomness rather than to a number baked into this module.

    Raises:
        ValueError: if none of the three is set. A reservoir drawn from OS entropy would
            make the run irreproducible, and because the features stay perfectly
            plausible the only symptom would be that two runs at one ``seed`` disagree.
    """
    for key in (RESERVOIR_SEED_KEY, "q_seed", "seed"):
        value = args.get(key) if isinstance(args, Mapping) else None
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"args[{key!r}] seeds the QELM reservoir and must be an integer; "
                f"got {value!r}."
            )
        return int(value)
    raise ValueError(
        f"QELM needs a seed for its random reservoir, but none of "
        f"{RESERVOIR_SEED_KEY!r}, 'q_seed' or 'seed' is set in args. An unseeded "
        f"reservoir makes the run irreproducible without failing, so it is refused. "
        f"Set 'seed' (or 'q_seed') as the other models do."
    )


def resolve_n_features(n_features, n_qubits, measurement):
    """How many observables to measure, from an int, a float multiple, or ``None``.

    Args:
        n_features (int, float or None): A positive int is a count; a positive float is a
            multiple of ``n_qubits`` (rounded down); ``None`` means the structured pool's
            own size, and is required to be ``None`` for those pools.
        n_qubits (int): Circuit width, i.e. the post-embedding feature count.
        measurement (str): One of :data:`~qbiocode.utils.qelm_features.MEASUREMENTS`.

    Returns:
        int or None: The count to pass to
        :func:`~qbiocode.utils.qelm_features.qelm_observables`; ``None`` for a
        structured pool, whose width is fixed by ``n_qubits``.

    Raises:
        ValueError: If a count is given for a structured pool (whose width is not free),
            if ``'random_pauli'`` is given none, or if the value is not a positive number.

    Note:
        A float of ``3.0`` is ``3 * n_qubits`` and not three observables. That is the
        point -- the structured pools are ``3n`` and ``4n - 1`` wide, so a multiplier is
        the only spelling under which one ``gridsearch_qelm_args`` block means the same
        thing on an 8-component and a 19-feature dataset.
    """
    if measurement != "random_pauli":
        if n_features is not None:
            raise ValueError(
                f"n_features={n_features!r} was given with measurement={measurement!r}, "
                f"whose width is fixed by the qubit count (it is a structured pool, not "
                f"a draw). Leave n_features unset for this measurement, or use "
                f"measurement='random_pauli', whose width is free."
            )
        return None
    if n_features is None:
        raise ValueError(
            f"measurement='random_pauli' needs n_features -- the number of observables "
            f"to draw. Write {_FEATURE_COUNT_TYPES}, e.g. 24 or 3.0 (which is "
            f"3 x {n_qubits} = {3 * n_qubits} here)."
        )
    if isinstance(n_features, (bool, np.bool_)):
        raise ValueError(f"n_features must be {_FEATURE_COUNT_TYPES}; got {n_features!r}.")
    if isinstance(n_features, (int, np.integer)):
        resolved = int(n_features)
    elif isinstance(n_features, (float, np.floating)):
        if not np.isfinite(n_features) or n_features <= 0:
            raise ValueError(
                f"n_features as a multiple of the qubit count must be a positive finite "
                f"number; got {n_features!r}."
            )
        resolved = int(np.floor(float(n_features) * n_qubits))
    else:
        raise ValueError(f"n_features must be {_FEATURE_COUNT_TYPES}; got {n_features!r}.")
    if resolved < 1:
        raise ValueError(
            f"n_features resolved to {resolved} observables (from {n_features!r} on "
            f"{n_qubits} qubits), which measures nothing. Raise it."
        )
    return resolved


def create_ridge_model(seed, alphas=DEFAULT_RIDGE_ALPHAS, scoring=None):
    """The QELM readout: a ridge classifier whose penalty is chosen in closed form.

    ``RidgeClassifierCV`` selects ``alpha`` by leave-one-out generalised
    cross-validation, which for ridge is computed from one matrix decomposition rather
    than by refitting per fold. So the whole head -- penalty selection included -- costs
    about one fit, against the ``HEAD_N_ITER * HEAD_CV`` = 200 SVC fits inside a PQK
    head. That difference is the arm's cost argument and is not incidental to it.

    Args:
        seed (int or None): Accepted for signature symmetry with the other head
            factories and deliberately unused: neither ridge nor its GCV draws random
            numbers, so there is no ``random_state`` to set. Taking it keeps the call
            sites uniform rather than making this head the one exception.
        alphas (sequence): Penalties to select among. A one-element sequence pins the
            penalty, which is how a search over ``alpha`` fixes it per trial.
        scoring (str, callable or None): How ``alpha`` is selected. ``None`` is ridge's
            own GCV, the cheap closed-form route. A scorer switches
            ``RidgeClassifierCV`` to real cross-validation, so it is passed only under
            ``split_mode: manifest``, where every arm must select on the run's tuning
            metric.

    Returns:
        RidgeClassifier or RidgeClassifierCV: Unfitted. A single ``alpha`` returns a
        plain ``RidgeClassifier`` -- ``RidgeClassifierCV`` over one value would run a
        selection with nothing to select, and would then report an ``alpha_`` that no
        search produced.
    """
    alphas = [float(a) for a in (alphas if np.ndim(alphas) else [alphas])]
    if not alphas:
        raise ValueError(
            "alphas is empty, so there is no ridge penalty to fit with. Pass at least "
            f"one value; the default is {list(DEFAULT_RIDGE_ALPHAS)}."
        )
    if any(a <= 0 for a in alphas):
        raise ValueError(
            f"every ridge alpha must be positive (it is an L2 penalty); got {alphas}."
        )
    if len(alphas) == 1:
        return RidgeClassifier(alpha=alphas[0])
    # cv=None is the generalised cross-validation path; a scorer needs real folds, and
    # sklearn refuses `scoring` with cv=None, so the two travel together.
    if scoring is None:
        return RidgeClassifierCV(alphas=alphas)
    return RidgeClassifierCV(alphas=alphas, scoring=scoring, cv=5)


def _dump_qelm_projections(
    args, data_key, model, Z_train, Z_test, X_train, y_train, y_test, params
):
    """Write the reservoir features the ridge readout was fitted on.

    The same diagnostic ``compute_pqk`` writes, for the same reason: the projection cache
    is keyed by a digest over the circuit *and* the data, holds no labels, and keeps no
    copy of ``X_train``, so none of kernel-target alignment, a geometric difference or an
    RKHS margin can be recomputed after the fact from what a run normally persists.

    ``params`` records the circuit and the pool, because the features alone do not say
    what produced them -- and for this arm that includes the reservoir seed, without
    which the representation cannot be rebuilt at all.
    """
    dump_dir = args.get("kernel_dump_dir") if isinstance(args, Mapping) else None
    if not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    stem = os.path.join(dump_dir, f"qelm_{model}_{data_key}")
    np.savez_compressed(
        stem + ".npz",
        Z_train=np.asarray(Z_train),
        Z_test=np.asarray(Z_test),
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        # npz holds arrays, so the mapping travels as a JSON scalar and is read back
        # with json.loads(str(z["params"])), as compute_pqk's best_params is.
        params=np.asarray(json.dumps(params, default=str)),
    )


def compute_qelm(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    # Lower case, matching the dispatch key, for the reason compute_pqk's default
    # carries: `modeleval` names its columns after whatever label arrives, so a direct
    # call with no model= would otherwise file results under a name no config can name.
    model="qelm",
    data_key="",
    verbose=False,
    encoding="Z",
    primitive="estimator",
    entanglement="linear",
    reps=2,
    data_map="unit",
    bandwidth=1.0,
    reservoir_reps=1,
    measurement="xyz",
    n_features=None,
    alpha=None,
    *,
    max_weight=None,
    head_scoring=None,
):
    """Fit a quantum extreme learning machine and evaluate it on the test split.

    Projects every row through the data feature map and a fixed random reservoir, reads
    ``n_features`` Pauli expectation values off each state, and fits a ridge classifier
    on them. The projection is cached under ``qelm_projection_dir`` and reused, keyed by
    a digest of everything that changes what the features hold.

    Args:
        X_train (np.ndarray): Training features -- the same post-embedding matrix every
            other model in the run receives.
        X_test (np.ndarray): Test features, the same width.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        args (dict): Run configuration. Reads ``backend``, ``seed``, ``shots``,
            ``qelm_projection_dir``, ``projection_backend``, ``projection_n_jobs``,
            ``kernel_dump_dir`` and the reservoir seed (see
            :func:`resolve_reservoir_seed`).
        model (str): Results label; the row is filed as ``results_<model>``.
        data_key (str): The pass's key, interpolated into the projection cache name.
        verbose (bool): Print the metrics summary.
        encoding (str): Feature map -- ``'Z'``, ``'ZZ'`` or ``'P'``.
        primitive (str): Must be ``'estimator'``; Pauli expectations are an Estimator
            measurement.
        entanglement (str): Feature-map entanglement pattern.
        reps (int): Feature-map repetitions.
        data_map (str or bool): How features become gate angles, as in
            :func:`qbiocode.learning.compute_pqk.compute_pqk` -- ``'unit'`` or
            ``'qiskit'``.
        bandwidth (float): Scale applied to the features before the feature map
            (:func:`qbiocode.utils.qutils.apply_bandwidth`).
        reservoir_reps (int): Reservoir depth. **0 builds no reservoir**, which with
            ``measurement='xyz'`` reproduces PQK's projection exactly -- the ablation
            cell described in the module docstring.
        measurement (str): Which observables to read;
            :data:`~qbiocode.utils.qelm_features.MEASUREMENTS`. ``'xyz'`` is ``3n``
            features, ``'xyz+zz'`` is ``4n - 1``, ``'random_pauli'`` is ``n_features``
            drawn from the seeded pool.
        n_features (int, float or None): How many observables, for ``'random_pauli'``
            only -- an int count or a float multiple of the qubit count. The structured
            pools fix their own width and require this to be unset. See
            :func:`resolve_n_features`.
        alpha (float, sequence or None): The ridge penalty. ``None`` selects among
            :data:`DEFAULT_RIDGE_ALPHAS` by GCV inside this one fit; a single value pins
            it, which is what a search over ``alpha`` does per trial.
        max_weight (int or None): Pauli-weight cap for ``'random_pauli'``. Keyword-only.
            ``None`` is
            :data:`~qbiocode.utils.qelm_features.DEFAULT_MAX_WEIGHT` (two-body).
        head_scoring (str or None): Metric the ridge penalty is selected by. ``None``
            (the default) is ridge's own GCV. ``compute_qelm_opt`` sets the run's tuning
            metric under ``split_mode: manifest``, which switches the head to real
            cross-validation; it is part of this one fit, not extra tuning budget.

    Returns:
        pd.DataFrame: One ``results_<model>`` row with the metrics, the predictions and
        the ranking scores, as ``modeleval`` builds it.

    Raises:
        ValueError: For any argument outside its accepted set, or inconsistent
            train/test arrays. Every check runs before a directory is created or a cache
            is read, so a typo costs nothing and names the parameter, the value and the
            accepted set.
    """
    # --- boundary validation -------------------------------------------------
    # Up front and in this order for the reason compute_pqk gives: these used to surface
    # far from their cause, a bad `encoding` coming back out of qutils as the input
    # string and failing with "'str' object has no attribute 'num_qubits'".
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
    if not isinstance(reps, (int, np.integer)) or isinstance(reps, bool) or reps < 1:
        raise ValueError(
            f"reps is the number of feature-map repetitions and must be a positive "
            f"integer; got {reps!r}."
        )
    if measurement not in MEASUREMENTS:
        raise ValueError(
            f"measurement must be one of {list(MEASUREMENTS)}; got {measurement!r}. "
            f"'xyz' is the 3n single-qubit pool (and, with reservoir_reps: 0, "
            f"reproduces PQK exactly); 'random_pauli' is the one whose width is free."
        )
    data_map = _resolve_data_map(data_map)
    if primitive != "estimator":
        # Same refusal, and the same reason, as compute_pqk: QELM reads Pauli
        # expectation values, which only the Estimator primitive provides. Accepting
        # 'sampler' would change the cache fingerprint without changing the
        # computation -- two cache files holding identical features, and a caller who
        # believed they had measured something else.
        raise ValueError(
            f"primitive must be 'estimator'; got {primitive!r}. A QELM reads Pauli "
            f"expectation values off the reservoir state, which only the Estimator "
            f"primitive provides. For sampler-based models see compute_vqc or "
            f"compute_qnn."
        )
    if not isinstance(data_key, str):
        raise ValueError(
            f"data_key is interpolated into the projection cache filename and must be "
            f"a string; got {type(data_key).__name__} ({data_key!r})."
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
            f"X_train and X_test must have the same number of features -- one feature "
            f"map and one reservoir are built for both -- got {X_train.shape[1]} and "
            f"{X_test.shape[1]}."
        )
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"X_train and X_test must both be non-empty; got {X_train.shape[0]} "
            f"training and {X_test.shape[0]} test samples."
        )
    if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
        raise ValueError(
            f"Labels and features must be aligned; got {X_train.shape[0]} train "
            f"samples vs {len(y_train)} train labels, and {X_test.shape[0]} test "
            f"samples vs {len(y_test)} test labels."
        )

    reservoir_seed = resolve_reservoir_seed(args)
    feat_dimension = X_train.shape[1]
    # Built before the cache key, because both of them are part of it and both can fail:
    # an impossible pool (more distinct Paulis than exist) should be reported before any
    # directory is created.
    resolved_features = resolve_n_features(n_features, feat_dimension, measurement)
    pool_kwargs = {} if max_weight is None else {"max_weight": int(max_weight)}
    observables = qelm_observables(
        feat_dimension, measurement, n_features=resolved_features,
        seed=reservoir_seed, **pool_kwargs,
    )
    reservoir = qelm_reservoir(feat_dimension, reservoir_reps, reservoir_seed)
    # Before the cache key, the reservoir and the feature map read the rows, so all of
    # them see the scaled inputs. 1.0 returns the arrays unchanged.
    X_train, X_test = qutils.apply_bandwidth(bandwidth, X_train, X_test)
    # ------------------------------------------------------------------------

    beg_time = time.time()

    projection_dir = os.path.expanduser(
        args.get("qelm_projection_dir", "qelm_projections")
    )
    os.makedirs(projection_dir, exist_ok=True)

    # Everything that changes what a cached projection HOLDS is in the key. Leave one
    # out and the failure is silent in the worst way: change that setting, rerun into
    # the same directory, and the previous run's features are reloaded and reported as
    # the new result -- the trap compute_qpl documents at its own fingerprint. Unlike
    # pqk/qpl there is no legacy cache to stay compatible with, so every parameter is
    # named unconditionally rather than only when it differs from a default.
    fingerprint_parts = (
        "qelm-v1",
        encoding,
        entanglement,
        reps,
        primitive,
        feat_dimension,
        f"data_map={data_map}",
        f"bandwidth={float(bandwidth)!r}",
        f"reservoir_reps={int(reservoir_reps)}",
        f"reservoir_seed={reservoir_seed}",
        f"measurement={measurement}",
        # The pool digest, not n_features: two pools of one width can hold different
        # operators (a different seed, a different weight cap), and it is the operators
        # that decide the numbers.
        f"observables={observables_fingerprint(observables)}",
        qutils.dataset_fingerprint(X_train, X_test),
        args.get("shots"),
        args.get("backend"),
    )
    _projection_backend = args.get("projection_backend")
    if _projection_backend:
        fingerprint_parts = fingerprint_parts + (_projection_backend,)
    fingerprint = hashlib.sha256(repr(fingerprint_parts).encode()).hexdigest()[:10]

    file_projection_train = os.path.join(
        projection_dir, f"qelm_projection_{data_key}_{fingerprint}_train.npy"
    )
    file_projection_test = os.path.join(
        projection_dir, f"qelm_projection_{data_key}_{fingerprint}_test.npy"
    )
    checkpoint_dir = os.path.join(projection_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_every = int(args.get("qelm_checkpoint_every", 1))

    def _save_projection_array(path, projections):
        """Written via a temporary file: a reader arriving mid-write would otherwise
        load a truncated array, and a kill mid-write would leave one on disk."""
        tmp_path = path + ".tmp.npy"
        np.save(tmp_path, np.asarray(projections))
        os.replace(tmp_path, path)

    def _checkpoint_path(final_path):
        base = os.path.basename(final_path)
        return os.path.join(checkpoint_dir, base.replace(".npy", ".partial.npy"))

    def _load_checkpoint(path, expected_len):
        if not os.path.exists(path):
            return []
        projections = np.load(path, allow_pickle=False)
        if len(projections) > expected_len:
            raise ValueError(
                f"Checkpoint {path} has {len(projections)} rows, but the dataset only "
                f"has {expected_len} rows. Remove it, or use a different "
                f"qelm_projection_dir."
            )
        return list(projections)

    def _validate_projection_file(path, expected_len):
        """Refuse a cached projection that does not describe the current data.

        The row count catches a different split of the same dataset; the width catches a
        file written by a run whose pool was a different size. Neither is caught by the
        fingerprint alone -- a cache file can only be checked once it is opened.
        """
        if not os.path.exists(path):
            return
        cached = np.load(path, allow_pickle=False)
        if len(cached) != expected_len:
            raise ValueError(
                f"Projection file {path} has {len(cached)} rows, but the current "
                f"dataset expects {expected_len} rows. Remove this projection file or "
                f"use a different qelm_projection_dir."
            )
        width = int(np.prod(np.asarray(cached).shape[1:])) if len(cached) else 0
        if len(cached) and width != len(observables):
            raise ValueError(
                f"Projection file {path} has {width} features per row, but the current "
                f"measurement produces {len(observables)} "
                f"(measurement={measurement!r} on {feat_dimension} qubits). Remove this "
                f"projection file or use a different qelm_projection_dir."
            )

    _validate_projection_file(file_projection_train, len(X_train))
    _validate_projection_file(file_projection_test, len(X_test))

    # 'unit' is shared with compute_pqk and qbiocode.embeddings.embed.pqk; 'qiskit'
    # passes None, which the feature map reads as "use your default map".
    data_map_func = qutils.unit_coefficient_data_map if data_map == "unit" else None

    if not (os.path.exists(file_projection_train) and os.path.exists(file_projection_test)):
        # One projector for both splits, built once: it transpiles the circuit, which at
        # 100+ qubits is the expensive part and does not depend on the rows.
        #
        # `projection_backend` is honoured as pqk/qpl honour it, but unlike them QELM has
        # no runtime-primitive fallback path: its observable pool can hold multi-qubit
        # Paulis, and the per-row Estimator loop those two use is written around 3n
        # single-qubit PUB groups. Rather than ship a second, differently-shaped
        # implementation of the same thing, this arm always goes through
        # qbiocode.utils.projection -- which is also the faster path (one state
        # preparation per row, every observable read off it, against
        # StatevectorEstimator's one re-simulation PER OBSERVABLE).
        if args["backend"] != "simulator":
            raise ValueError(
                f"backend={args['backend']!r} selects remote or noisy execution, which "
                f"this arm does not support: a QELM reads an arbitrary pool of Pauli "
                f"observables off one state, and the runtime-primitive path used by pqk "
                f"and qsvc is built around the 3n single-qubit groups those models "
                f"measure. Set backend: 'simulator' (optionally with "
                f"projection_backend: 'aer_mps' or 'quimb_mps' for width)."
            )
        from qbiocode.utils.projection import make_projector

        projector = make_projector(
            feat_dimension,
            encoding=encoding,
            reps=reps,
            entanglement=entanglement,
            # 'auto' routes on the FEATURE MAP's pattern; the reservoir is
            # nearest-neighbour by construction, so a bounded pattern stays bounded.
            backend=args.get("projection_backend") or "auto",
            data_map_func=data_map_func,
            observables=observables,
            extra_circuit=reservoir,
        )
        for path, split in (
            (file_projection_train, X_train),
            (file_projection_test, X_test),
        ):
            if os.path.exists(path):
                continue
            checkpoint_file = _checkpoint_path(path)
            done = _load_checkpoint(checkpoint_file, len(split))
            if done:
                print(
                    f"Resuming {os.path.basename(path)} from datapoint {len(done)}"
                )
            rows = list(done)
            # Row by row rather than one projector.project() call, so a killed job
            # resumes from its checkpoint instead of restarting the pass. The quantum
            # arms are the ones that hit a scheduler wall, and on them one pass is hours.
            #
            # n_jobs is deliberately not used here: it would project the whole split in
            # one parallel call and so could not checkpoint. The parallelism that pays
            # for this workload is model_run's fan-out over models, which is already
            # running.
            for index in range(len(rows), len(split)):
                if index % 100 == 0:
                    print(f"at datapoint {index}")
                rows.append(projector.project_row(split[index]))
                if checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
                    _save_projection_array(checkpoint_file, rows)
            _save_projection_array(path, rows)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        # Truncation is silent, so surface it rather than leaving it to be noticed in
        # the accuracy numbers. None means the backend does not report it.
        fidelity = projector.fidelity_estimate()
        if fidelity is not None and fidelity < 0.999:
            warnings.warn(
                f"the QELM projection truncated the state: fidelity estimate "
                f"{fidelity:.4g} (1.0 = exact). These features are approximate and any "
                f"metric computed from them should be reported as such.",
                RuntimeWarning,
            )

    projections_train = np.load(file_projection_train, allow_pickle=False)
    projections_train = projections_train.reshape(len(projections_train), -1)
    projections_test = np.load(file_projection_test, allow_pickle=False)
    projections_test = projections_test.reshape(len(projections_test), -1)

    # `estimator`, not `model`: the label parameter must survive this assignment. Both
    # compute_pqk and compute_qpl once rebound `model` to the fitted estimator here, and
    # the visible consequence was that a TUNED run produced rows byte-identical to an
    # untuned one -- there was no way to tell from ModelResults.csv whether a search had
    # run. Not repeated here.
    alphas = DEFAULT_RIDGE_ALPHAS if alpha is None else alpha
    estimator = create_ridge_model(
        args.get("seed"), alphas=alphas,
        scoring=_ridge_scorer(head_scoring, args, alphas),
    )
    estimator.fit(projections_train, y_train)
    y_predicted = estimator.predict(projections_test)
    # `auc` and `pr_auc` are computed from these scores alone, never from y_predicted.
    # Ridge publishes no predict_proba, so extract_binary_scores falls through to
    # decision_function -- a real ranking, which is what those metrics need. Scored on
    # the PROJECTIONS, the space this estimator was fitted in; the raw features would
    # silently be the wrong width.
    y_score = extract_binary_scores(estimator, projections_test)

    model_params = {
        "feature_map": encoding,
        "feature_map_reps": reps,
        "entanglement": entanglement,
        "data_map": data_map,
        "bandwidth": float(bandwidth),
        "reservoir_reps": int(reservoir_reps),
        # Recorded because the representation cannot be rebuilt without it, and because
        # it is NOT tuned -- a reader comparing two rows needs to see that the reservoir
        # was the same one.
        "reservoir_seed": reservoir_seed,
        "measurement": measurement,
        "n_features": len(observables),
        # What the head actually chose, which is the one thing about this model that was
        # fitted rather than fixed.
        "alpha": float(getattr(estimator, "alpha_", getattr(estimator, "alpha", np.nan))),
    }
    if resolved_features is not None:
        # Only for the pool whose width is free, so a structured-pool row does not carry
        # a request that did not apply.
        model_params["n_features_requested"] = n_features
    if max_weight is not None:
        model_params["max_weight"] = int(max_weight)
    if head_scoring is not None:
        model_params["head_scoring"] = head_scoring

    _dump_qelm_projections(
        args, data_key, model, projections_train, projections_test,
        X_train, y_train, y_test, model_params,
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


def _ridge_scorer(head_scoring, args, alphas):
    """The ridge head's ``scoring``, or ``None`` for its own GCV.

    ``None`` keeps the closed-form generalised cross-validation that makes this head
    cheap, which is what every internal-mode run uses. A tuning-metric name switches it
    to k-fold selection on that metric, which ``split_mode: manifest`` requires of every
    arm. Pinned to a single ``alpha`` there is nothing to select, so the scorer is
    dropped rather than handed to a plain ``RidgeClassifier`` that cannot use it.
    """
    if head_scoring is None:
        return None
    if np.ndim(alphas) and len(alphas) == 1:
        return None
    from qbiocode.learning.compute_fold import head_scorer

    return head_scorer(head_scoring, args)


def compute_qelm_opt(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    # '_opt', so a DIRECT call is self-describing. model_run always passes
    # model='qelm_opt' explicitly, but a caller using the default would otherwise produce
    # a row labelled as untuned -- and modeleval infers `tuned` from this very string, so
    # the label and the parameter column would BOTH be wrong.
    model="qelm_opt",
    data_key="",
    encoding=None,
    primitive=None,
    entanglement=None,
    reps=None,
    data_map=None,
    bandwidth=None,
    reservoir_reps=None,
    measurement=None,
    n_features=None,
    alpha=None,
    *,
    max_weight=None,
    n_trials=10,
    validation_split=0.25,
    validation=None,
    default_params=None,
    reseed=None,
):
    """Tune QELM's hyperparameters with Optuna, then run it at the best ones found.

    The quantum counterpart of the classical ``compute_*_opt`` functions, driven by the
    same ``gridsearch_qelm_args`` block -- a list is a choice, a ``{low, high}`` mapping
    is a range. A candidate is scored by running the whole model once on a stratified
    holdout carved out of ``X_train``, because a QELM fit projects every row by circuit
    simulation and k-fold would multiply an already expensive search by k. The caller's
    test set is never touched by the search.

    Only reachable when the config sets both ``grid_search: True`` and
    ``tune_quantum: True``. Tuning against a real device is refused unless
    ``allow_hardware_tuning: True``.

    **What is searched, and what is not.** The circuit and the pool are searched:
    ``encoding``, ``reps``, ``entanglement``, ``bandwidth``, ``data_map``,
    ``reservoir_reps``, ``measurement``, ``n_features``. The ridge penalty may be
    searched too, but need not be -- left out of the block it is selected inside every
    fit by GCV, which is cheaper and is the arm's default. ``reservoir_seed`` is **not**
    searchable and is read from ``args``: see the module docstring for why tuning it
    would turn the reported score into a maximum over random draws.

    **Across splits.** With ``freeze_quantum_params: True`` the search runs on the first
    resample only and its winner is reused by the rest, cutting the cost from
    ``iter * n_trials`` to ``n_trials + (iter - 1)``; ``False`` (the key's own default)
    searches on every split, each on its own training rows. Only the searched parameters
    are frozen -- a reusing split projects its own rows through the frozen circuit and
    refits its own ridge. See :mod:`qbiocode.learning._param_cache`, which also explains
    why a frozen arm's later scores are optimistic rather than conservative.

    Args:
        X_train (array-like): Training features. Split again internally to score
            candidates; the final model is refitted on all of it.
        X_test (array-like): Test features, used only for the final evaluation.
        y_train (array-like): Training labels.
        y_test (array-like): Test labels.
        args (dict): Run configuration, passed through to :func:`compute_qelm`.
        verbose (bool): Print the metrics summary of the final fit.
        model (str): Results label of the refit row.
        data_key (str): The pass's key; also what the frozen-parameter cache is keyed on.
        encoding (list or dict): Feature maps to search. ``None`` leaves the default.
        primitive (list or dict): Accepted for symmetry and effectively fixed --
            :func:`compute_qelm` takes ``'estimator'`` only.
        entanglement (list or dict): Entanglement patterns to search.
        reps (list or dict): Feature-map repetition counts to search.
        data_map (list): Data maps to search (``'unit'``, ``'qiskit'``).
        bandwidth (list or dict): Input scales to search.
        reservoir_reps (list or dict): Reservoir depths to search. Including ``0`` puts
            the PQK-equivalent representation in the search space, so the tuner can
            decide whether the reservoir earns its depth on this dataset.
        measurement (list): Observable pools to search.
        n_features (list or dict): Observable counts to search -- ints, or floats as
            multiples of the qubit count. Read only by ``'random_pauli'``; see
            :func:`resolve_n_features`.
        alpha (list or dict): Ridge penalties to search. ``None`` (the default) leaves
            the penalty to each fit's own GCV, which is both cheaper and the arm's
            defining behaviour.
        max_weight (int or None): Pauli-weight cap, fixed for every trial rather than
            searched -- it bounds the pool the draw comes from, so varying it varies what
            ``n_features`` means.
        n_trials (int): Trial budget, default 10 -- an order of magnitude below the
            classical default because each trial is a quantum fit. Lowered automatically
            when the configured values describe fewer combinations.
        validation_split (float): Fraction of the training data held out to score
            candidates on.
        validation (ValidationSplit or None): ``split_mode: manifest``. Every trial is
            then one fit on ``validation.X_fit`` scored on ``validation.X_val``,
            ``validation_split`` is ignored, trials write no kernel dumps, and the
            unsearched keys of ``default_params`` are fixed for the trials and the refit.
            The ridge head is then selected on the run's tuning metric rather than by
            GCV -- part of one trial's fit, not extra budget.
        default_params (dict or None): The arm's default config, enqueued as trial 0.
        reseed (callable or None): Resets the global RNGs before every trial and before
            the refit.

    Returns:
        pd.DataFrame: The evaluation at the best hyperparameters found, with the tuned
        values recorded in the row (and as ``selected_<name>`` columns) and the reported
        time covering the whole search rather than only the final fit.
    """
    beg_time = time.time()

    candidates = {
        "encoding": encoding,
        "primitive": primitive,
        "entanglement": entanglement,
        "reps": reps,
        "data_map": data_map,
        "bandwidth": bandwidth,
        "reservoir_reps": reservoir_reps,
        "measurement": measurement,
        "n_features": n_features,
        "alpha": alpha,
    }
    # Refused rather than ignored. `reservoir_seed` in a gridsearch block is the one
    # mistake that would invalidate a result while looking like ordinary tuning: the
    # study would report the best of n_trials random reservoirs as if it were the
    # performance of one. It cannot reach compute_qelm as a keyword (it is not a
    # parameter), so without this check it would surface as a TypeError inside a joblib
    # worker, after the untuned arms had run.
    _refuse_seed_tuning(args)

    fixed = {}
    if max_weight is not None:
        fixed["max_weight"] = int(max_weight)
    if validation is not None:
        fixed = fold_fixed("qelm", candidates, default_params,
                           function_param_names(compute_qelm), fixed)
        # The ridge penalty is then selected on the run's tuning metric rather than by
        # GCV, so every arm of a fold-based run selects on one statistic. Part of one
        # trial's fit, not budget. tuning_metric's plain name keeps the log plain.
        fixed.setdefault("head_scoring", tuning_metric(args))

    best_params = run_function_study(
        compute_qelm,
        build_search_space("qelm", candidates),
        X_train,
        y_train,
        args,
        model="qelm",
        n_trials=n_trials,
        seed=seed_from(args),
        validation_split=validation_split,
        data_key=data_key,
        fixed=fixed,
        validation=validation,
        default_params=default_params,
        reseed=reseed,
    )

    frame = compute_qelm(
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


def _refuse_seed_tuning(args):
    """Refuse a ``gridsearch_qelm_args`` block that tries to search the reservoir seed.

    Checked against the config rather than the keyword arguments because the seed is not
    a parameter of :func:`compute_qelm` at all -- so a block naming it reaches the tuner
    as an unexpected keyword and fails inside a joblib worker with a ``TypeError`` that
    names neither the config key nor why it is forbidden.
    """
    block = args.get("gridsearch_qelm_args") if isinstance(args, Mapping) else None
    if not block:
        return
    block = to_plain(block)
    if not isinstance(block, Mapping):
        return
    offenders = sorted(
        key for key in block
        if key in (RESERVOIR_SEED_KEY, "reservoir_seed", "seed", "q_seed")
    )
    if offenders:
        raise ValueError(
            f"gridsearch_qelm_args names {offenders}, which cannot be tuned. The "
            f"reservoir seed fixes the random representation; searching it would select "
            f"whichever draw best suited the validation holdout, so the reported score "
            f"would be the maximum over {len(block)} random reservoirs rather than the "
            f"performance of one. Set it once as "
            f"{RESERVOIR_SEED_KEY!r} (or leave it to 'q_seed'/'seed') outside the "
            f"gridsearch block."
        )
