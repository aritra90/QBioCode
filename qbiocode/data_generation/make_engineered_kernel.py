"""
Generate datasets whose labels are engineered to favour a quantum kernel.

This family is the **positive control**: a dataset built so that a quantum kernel
should win, by construction rather than by luck. If a quantum method cannot beat
classical baselines here, the pipeline is misconfigured -- most likely the encoder
does not match the one that made the labels.

The construction follows the geometric-difference argument of Huang et al.
(Nat. Commun. 12, 2631, 2021). Given a quantum kernel :math:`K_Q` and a classical
kernel :math:`K_C`, their eq. (5) bounds how much better the quantum model can be
by the geometric difference :math:`g(K_C \\| K_Q)`. That bound is *saturable*: the
continuous target

.. math:: y = \\sqrt{K_Q}\\, v, \\qquad
          v = \\arg\\max \\; \\sqrt{K_Q}\\,(K_C + \\lambda I)^{-1}\\sqrt{K_Q}

makes the quantum model's complexity :math:`s_Q = 1` while the classical model's is
:math:`s_C = g^2`. Here :math:`K_Q` is an RBF kernel on the 1-local Bloch vectors of
the ``ZZFeatureMap`` state, and :math:`K_C` is an RBF kernel on the raw inputs with
the bandwidth chosen *adversarially* -- scanned over 26 values and the one giving
the **smallest** :math:`g` is kept, so the classical baseline is as strong as an RBF
can be, not a straw man. The test suite asserts :math:`s_Q = 1` and
:math:`s_C = g^2` numerically.

.. important::
   :math:`g` is computed for the **continuous** target. The label written to disk is
   its median binarisation, and thresholding is not guaranteed to preserve the
   separation. The metadata records :math:`g` under ``diagnostics`` with that
   caveat attached: the binarised label's separation has to be measured, not
   assumed. Do not quote :math:`g` as an advantage for the classification task.

As with :mod:`~qbiocode.data_generation.make_quantum_labels`, the inputs are already
in :math:`[0,1]` and must reach the encoder unscaled, so QProfiler needs
``scaling: false``. Only ``x_view`` is written.
"""

import itertools

import numpy as np

from .quantum_core import (
    DATA_MAPS,
    Pauli,
    as_float_list,
    as_list,
    blas_limit,
    engineered_labels,
    ensure_unique_names,
    expvals,
    rbf,
    reject_name_for_sweep,
    threshold,
    write_dataset,
    zz_feature_state,
)

#: Parameters varied across the default sweep.
N_QUBITS = [6]
N_SAMPLES = [300]
GAMMA_Q = [1.0]

#: Classical RBF bandwidths scanned to find the strongest classical adversary.
GAMMA_C_GRID = np.logspace(-3, 2, 26)


def _engineered_kernel_name(n, gamma_q, seed, data_map="qiskit"):
    """The default dataset name for one configuration.

    The ``data_map`` suffix is appended only for the non-default map, so every
    name generated before that option existed is unchanged.
    """
    suffix = "" if data_map == "qiskit" else f"_dm{data_map}"
    return f"eng_zz_n{n}_gq{gamma_q:g}_s{seed}{suffix}"


def _engineered_kernel_dataset(save_path, name, n, n_rows, gamma_q, reps, entanglement,
                               lam, margin, seed, blas_threads, data_map):
    """Build and write one engineered-kernel dataset."""
    rng = np.random.default_rng(seed)
    with blas_limit(n, blas_threads):
        X = rng.uniform(0, 1, size=(n_rows, n))
        pool_bloch = [Pauli(n, {i: s}) for i in range(n) for s in "XYZ"]
        B = np.array([expvals(zz_feature_state(x, reps, entanglement, data_map), pool_bloch)[0]
                      for x in X])
        K_Q = rbf(B, gamma_q)
        gamma_c, g, yc = None, None, None
        for gc in GAMMA_C_GRID:                       # best RBF adversary = smallest g
            y_gc, g_gc = engineered_labels(rbf(X, gc), K_Q, lam)[:2]
            if g is None or g_gc < g:
                gamma_c, g, yc = gc, g_gc, y_gc
        thr, keep = threshold(yc, margin)
        y = (yc > thr).astype(int)
        meta = dict(family="eng", n=n, reps=reps, entanglement=entanglement, data_map=data_map,
                    gamma_q=gamma_q,
                    gamma_c_adversary=gamma_c, lam=lam, threshold=thr, margin=margin, seed=seed,
                    diagnostics={"g_continuous": g, "g2_continuous": g * g,
                                 "note": "g is for the CONTINUOUS target; the binarized label's separation "
                                         "must be measured (e.g. boundary-complexity ratio), not assumed"})
        return write_dataset(save_path, name, X[keep], [f"x{i}" for i in range(n)],
                             y[keep], yc[keep], meta)


def generate_engineered_kernel_datasets(
    n_qubits=N_QUBITS,
    n_samples=N_SAMPLES,
    gamma_q=GAMMA_Q,
    reps=2,
    entanglement="linear",
    lam=1e-3,
    margin=0.0,
    save_path=None,
    name=None,
    random_state=0,
    blas_threads=1,
    data_map="qiskit",
):
    """
    Generate datasets whose labels saturate the quantum-advantage bound.

    Sweeps the Cartesian product of ``n_qubits``, ``n_samples``, ``gamma_q`` and
    ``random_state``, writing one dataset per combination.

    .. important::
       The labels are tuned to one specific quantum kernel, so the learner must
       use the *same* feature map -- including the same data map. QProfiler's
       ``qsvc`` uses Qiskit's default (``data_map='qiskit'``) and its ``pqk``
       uses the halving map (``data_map='unit'``); there is no value here that
       aligns with both. Generating for the wrong one does not merely weaken the
       advantage, it inverts it: projected-kernel accuracy on
       ``eng_zz_n4_gq1_s0`` is 0.797 under the matching map and 0.403 under the
       other. See :func:`~qbiocode.data_generation.quantum_core.zz_feature_state`.

    Parameters
    ----------
    n_qubits : int or list of int, default=[6]
        Qubit counts, which are also the feature counts.
    n_samples : int or list of int, default=[300]
        Rows per dataset, before ``margin`` filtering. This family's cost is driven
        by ``n_samples`` rather than ``n_qubits``: the adversary scan does 26 dense
        ``n_samples``-by-``n_samples`` eigendecompositions and solves.
    gamma_q : float or list of float, default=[1.0]
        Bandwidth of the RBF kernel on the Bloch vectors, i.e. of the quantum
        kernel the labels are built to favour.
    reps : int, default=2
        Feature-map repetitions used to build the quantum kernel. A learner must use
        the same depth; see :mod:`~qbiocode.data_generation.make_quantum_labels`.
        Not swept -- it does not appear in the dataset name.
    entanglement : {'linear', 'pairwise', 'full'}, default='linear'
        Entanglement pattern of the feature map. Not swept, as for ``reps``.
    lam : float, default=1e-3
        Ridge added to :math:`K_C` before inversion. It keeps the geometric
        difference finite; a smaller value makes the classical adversary weaker and
        ``g`` larger, so it is part of the construction rather than a free knob.
    margin : float, default=0.0
        Drop rows whose continuous target lies within ``margin`` of the threshold.
    save_path : str, optional
        Directory to write into; defaults to ``'quantum_data'``.
    name : str, optional
        Override the generated dataset name. Only valid when the sweep yields a
        single dataset.
    random_state : int or list of int, default=0
        Seed, or seeds to sweep over.
    blas_threads : int or None, default=1
        BLAS threads for the linear algebra. Pass ``None`` to leave threading
        untouched.
    data_map : {'qiskit', 'unit'}, default='qiskit'
        Data map of the feature map the labels are built against. ``'qiskit'``
        targets QProfiler's ``qsvc``, ``'unit'`` targets its ``pqk``. Not swept;
        the non-default value adds a ``_dmunit`` suffix to the dataset name.

    Returns
    -------
    list of dict
        The metadata written for each dataset, including the selected adversarial
        bandwidth under ``gamma_c_adversary`` and ``g`` under ``diagnostics``.

    Raises
    ------
    ValueError
        If ``name`` is given for a sweep of more than one configuration, if the
        sweep would write two datasets to the same name, or if ``data_map`` is
        not one of ``'qiskit'`` or ``'unit'``.

    Notes
    -----
    Only ``x_view`` is written, with columns ``x0 … x{n-1}``. ``g`` describes the
    continuous target, not the binarised label -- see the module docstring.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_engineered_kernel_datasets
    >>> generate_engineered_kernel_datasets(n_qubits=4, n_samples=32,
    ...                                    save_path='qdata')       # doctest: +SKIP
    Generating engineered-kernel datasets...
    """
    print("Generating engineered-kernel datasets...")
    if data_map not in DATA_MAPS:
        raise ValueError(
            f"data_map must be one of {list(DATA_MAPS)}, got {data_map!r}. "
            "'qiskit' targets QProfiler's qsvc, 'unit' targets its pqk."
        )
    if save_path is None:
        save_path = "quantum_data"
    configurations = list(itertools.product(
        as_list(n_qubits), as_list(n_samples), as_float_list(gamma_q), as_list(random_state)
    ))
    reject_name_for_sweep(
        name, len(configurations), ["n_qubits", "n_samples", "gamma_q", "random_state"]
    )
    names = [name or _engineered_kernel_name(n, gq, seed, data_map)
             for n, _, gq, seed in configurations]
    ensure_unique_names(names, unnamed_knobs=["n_samples"])
    metas = []
    for (n, n_rows, gq, seed), dataset_name in zip(configurations, names):
        metas.append(_engineered_kernel_dataset(
            save_path, dataset_name, n, n_rows, gq, reps, entanglement,
            float(lam), float(margin), seed, blas_threads, data_map,
        ))
    return metas
