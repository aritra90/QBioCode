"""Fold-internal feature selection: the stage before scaling and the embedding.

Wide data -- RNA-seq at ~20k genes is the case this was written for -- reaches the
quantum arms through an embedding, because ``generate_pilot_configs.backend_for()`` bands
a dataset by its width and anything past ``--embed-above`` is reduced to
``n_components``. Selecting features first is a different operation from embedding and
answers a different question: a variance filter keeps a subset of the ORIGINAL columns,
so the result is still gene expression, while PCA returns linear combinations that are
not.

Why this is a configured stage rather than a preprocessing script
-----------------------------------------------------------------
Selecting before the pipeline, once per dataset, uses every row of the file -- including
the rows each fold will be scored on. The leak is unsupervised (no labels are read) and
usually small, but it is a leak, and it is avoidable: the selection is a function of the
training side alone, exactly as the scaler and the embedding already are. Doing it here
also makes the choice appear in the embedding cache spec and in the result rows, so
``n_features`` becomes a measurable axis of a sweep instead of a fact buried in whoever
prepared the inputs.

The contract
------------
:func:`select_features` is given the two sides of one (split, stage) -- for
``stage='final'`` the outer training fold and the test fold, for ``stage='tune'`` the fit
rows and the validation rows -- and returns both sides restricted to columns chosen from
the FIRST side only. It is a pure function of ``(X_a, method, n_features)``: the second
side cannot influence which columns are kept, which is the property
``fit_scope: train_fold`` names and the one its test asserts.

Ordering: selection runs on the RAW values, before scaling. This is not a preference.
The default ``scaling`` resolves to ``MinMaxScaler``
(:func:`qbiocode.apps.qprofiler.qprofiler._resolve_scaling`), which maps every column
onto [0, 1] and so flattens the variance ranking into a statement about how far each
column's extremes sit from each other -- not about expression spread. Selecting after it
would silently rank genes by a different quantity than the one the method names.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Selection methods. ``'none'`` is the default everywhere and keeps every column, so a
#: config that does not mention feature selection behaves exactly as before -- which is
#: what lets an existing embedding cache stay valid (its spec is unchanged).
METHODS = ("none", "variance")

#: Scopes the selection may be fitted on. Deliberately one value: a ``'cohort'`` scope
#: would be the whole-file selection this module exists to avoid, and giving it a name in
#: an enum is how it would end up used by accident. If a transductive scope is ever
#: wanted it should arrive with its own justification and its own warning, the way
#: :func:`qbiocode.embeddings.embed.is_transductive` declares the spectral case.
FIT_SCOPES = ("train_fold",)


def normalise_config(value):
    """Validate a ``feature_selection`` config block and fill in its defaults.

    Args:
        value: the config's ``feature_selection``, or None when it sets none. A mapping
            with ``method``, and for a real method ``n_features`` and optionally
            ``fit_scope``. OmegaConf nodes are accepted: only ``.get`` and ``in`` are
            used, and the values are cast here.

    Returns:
        dict: ``{'method': str, 'n_features': int | None, 'fit_scope': str}``, canonical
        and JSON-safe, suitable for both the cache spec and
        :func:`select_features`. ``method='none'`` carries ``n_features=None``, so every
        no-selection run produces one spec whatever n_features a config happens to leave
        lying around.

    Raises:
        ValueError: on an unknown method or fit_scope, a non-positive ``n_features``, or
            a real method with no ``n_features``. Named here rather than defaulted,
            because a silent default would decide the width of every quantum circuit.
    """
    if value is None:
        return {"method": "none", "n_features": None, "fit_scope": FIT_SCOPES[0]}

    try:
        method = value.get("method", "none")
    except AttributeError as exc:
        raise ValueError(
            f"feature_selection must be a mapping with a 'method' key; got {value!r}"
        ) from exc

    method = str(method).strip().lower()
    if method not in METHODS:
        raise ValueError(
            f"feature_selection.method {method!r} is not one of {list(METHODS)}."
        )

    fit_scope = str(value.get("fit_scope", FIT_SCOPES[0])).strip().lower()
    if fit_scope not in FIT_SCOPES:
        raise ValueError(
            f"feature_selection.fit_scope {fit_scope!r} is not one of {list(FIT_SCOPES)}. "
            f"Selection is fitted on the training side of each split; a scope that reads "
            f"the held-out rows would leak them into the features every arm is scored on."
        )

    if method == "none":
        return {"method": "none", "n_features": None, "fit_scope": fit_scope}

    n_features = value.get("n_features", None)
    if n_features is None:
        raise ValueError(
            f"feature_selection.method {method!r} needs n_features; it decides how many "
            f"columns reach the embedding and, unembedded, how many qubits a circuit uses."
        )
    n_features = int(n_features)
    if n_features < 1:
        raise ValueError(f"feature_selection.n_features must be >= 1; got {n_features}.")

    return {"method": method, "n_features": n_features, "fit_scope": fit_scope}


def variance_ranking(X):
    """Column indices of ``X`` by descending variance, ties broken by column order.

    ``kind='stable'`` for the same reason as
    :func:`qbiocode.evaluation.task_spectrum._knn_mask`: duplicate values are common --
    a gene that is constant across a small fold has variance exactly 0, and there are
    usually many of them -- and an unstable sort would put a different column in the last
    kept slot on different runs. With a stable sort the tie-break is the column order,
    which is a property of the file.

    This makes the ranking reproducible from identical floats. It does NOT make it
    reproducible across CPU types: a variance is an accumulation, and this session
    measured byte-identical inputs giving different sums on different compute nodes. Two
    genes whose variances differ in the last bits can therefore swap across the cut on a
    different host, which is why the selected indices are RECORDED per split rather than
    recomputed from a hash (see :func:`selection_digest`).

    Args:
        X (numpy.ndarray): 2-D, rows are samples.

    Returns:
        numpy.ndarray: int64 column indices, highest variance first.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    # Descending by negating, so the stable tie-break stays "lower column index first".
    return np.argsort(-np.var(X, axis=0), kind="stable").astype(np.int64)


def selection_digest(indices):
    """16 hex digits over the selected column indices, for a cheap equality check.

    A short digest makes "did these two runs select the same columns?" a string
    comparison, and it goes in the result rows where the full index list would not fit.
    It is a check, not the record: the indices themselves are the artifact, because a
    digest that disagrees cannot distinguish a genuine mismatch from the last-bit
    cross-host difference described in :func:`variance_ranking`.

    Args:
        indices (array-like): the selected column indices, in the order selected.

    Returns:
        str: 16 lowercase hex digits.
    """
    array = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def select_features(X_a, X_b, config):
    """Restrict both sides of one split to columns chosen from ``X_a`` alone.

    Args:
        X_a (numpy.ndarray): the side the selection is fitted on -- the outer training
            fold at ``stage='final'``, the fit rows at ``stage='tune'``.
        X_b (numpy.ndarray): the side it is applied to (test, or validation).
        config (dict): a :func:`normalise_config` result.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray | None): the two sides, and the
        selected column indices in selection order -- None when nothing was selected, so
        a caller can tell "kept everything" from "selected all of them".

    Raises:
        ValueError: if the two sides disagree on width.
    """
    X_a = np.asarray(X_a)
    X_b = np.asarray(X_b)
    if X_a.ndim != 2 or X_b.ndim != 2 or X_a.shape[1] != X_b.shape[1]:
        raise ValueError(
            f"both sides must be 2-D with the same width; got {X_a.shape} and {X_b.shape}"
        )

    if config["method"] == "none":
        return X_a, X_b, None

    # Fewer columns than asked for is not an error: n_features is a cap, and a dataset
    # narrower than the cap is simply kept whole. Reporting the indices anyway keeps the
    # sidecar's shape the same whether or not the cap bit.
    keep = min(int(config["n_features"]), X_a.shape[1])
    indices = variance_ranking(X_a)[:keep]
    # Sorted, so the surviving columns stay in their original file order: a downstream
    # reader that joins these indices back to a gene list, and a human reading the
    # matrix, both expect column order to mean what it means in the file. The RANKING
    # decided membership; it does not have to decide layout.
    indices = np.sort(indices)
    return X_a[:, indices], X_b[:, indices], indices
