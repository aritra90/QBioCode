"""
Generate ground-state observable-learning datasets.

Each row is one instance of a mixed-field Ising chain

.. math::

    H(x) = -\\sum_i J_i Z_i Z_{i+1} - \\sum_i h_i X_i
           - \\kappa \\sum_i X_i X_{i+1},

described classically by its couplings :math:`x = (J, h)`. The binary label is a
property of the corresponding ground state, so the task is "predict a ground-state
observable from the Hamiltonian's parameters" -- the regime studied by Huang et al.
(Science 376, 1182, 2022).

Two label rules are available. ``'sparse'`` uses a hidden sparse observable
:math:`O(\\alpha) = \\sum_{j \\in S} \\alpha_j P_j` drawn from a pool of local,
:math:`Z_2`-even Paulis, which is the learning-observables setting. ``'e2e'`` uses
the end-to-end correlator :math:`Z_0 Z_{n-1}`, which is deliberately *not* in that
pool; it has a known closed-form classical twin, and the generator measures how
well that twin does so a surprising result can be checked immediately.

This family is the **classically-easy control** of the suite. The states are
gapped and the label depends smoothly on the couplings, so classical models on
``x_view`` should match or beat the quantum arms. A quantum win here is a signal
to go looking for a bug, not a result.

Two feature views are written. ``x_view`` holds the couplings, which is what
classical baselines and QProfiler's quantum encoders alike see -- this family's
labels come from the physics rather than from a kernel, so unlike ``ql``/``eng`` it
is not tied to a particular feature map or data map. ``phi_view`` holds
the local Pauli expectation values of the ground state -- a learner that has
already been handed the right measurements.
"""

import itertools

import numpy as np

from .quantum_core import (
    Pauli,
    as_float_list,
    as_list,
    blas_limit,
    ensure_unique_names,
    even_sector_ground_state,
    expvals,
    ising_terms,
    pool_z2_even,
    reject_name_for_sweep,
    shot_noise,
    sparse_observable,
    threshold,
    write_dataset,
)

#: Parameters varied across the default sweep.
N_QUBITS = [8]
N_SAMPLES = [400]
LABEL = ["sparse", "e2e"]
KAPPA = [0.5]


def _ground_state_name(label, n, kappa, seed):
    """The default dataset name for one configuration."""
    return f"gs_{label}_n{n}_k{kappa}_s{seed}"


def _ground_state_dataset(save_path, name, n, n_rows, label, n_terms, kappa, J_range, h_range,
                          margin, shots, seed, blas_threads):
    """Build and write one ground-state dataset. See :func:`generate_ground_state_datasets`."""
    rng = np.random.default_rng(seed)
    with blas_limit(n, blas_threads):
        pool = pool_z2_even(n)
        if label == "sparse":
            S, alpha = sparse_observable(pool, n_terms, rng)
        zz_e2e = Pauli(n, {0: "Z", n - 1: "Z"})
        X, PHI, E2E, gaps, res = [], [], [], [], []
        for _ in range(n_rows):
            J = rng.uniform(J_range[0], J_range[1], n - 1)
            h = rng.uniform(h_range[0], h_range[1], n)
            psi, gap, r = even_sector_ground_state(n, ising_terms(n, J, h, kappa=kappa))
            X.append(np.r_[J, h])
            PHI.append(expvals(psi, pool)[0])
            E2E.append(expvals(psi, [zz_e2e])[0, 0])
            gaps.append(gap)
            res.append(r)
        X, PHI, E2E = np.array(X), np.array(PHI), np.array(E2E)
        if label == "sparse":
            F = PHI[:, S] @ alpha
            rule = {"observable": [pool[j].label for j in S], "alpha": alpha.tolist()}
        else:                                   # end-to-end correlator, deliberately NOT in the local pool
            F = E2E
            rule = {"observable": [zz_e2e.label], "alpha": [1.0]}
        thr, keep = threshold(F, margin)
        y = (F > thr).astype(int)
        phi_obs = shot_noise(PHI, shots, rng)
        # classical-twin diagnostic for the Fisher-type product criterion (e2e only):
        diag = {"even_sector_gap_min": float(np.min(gaps)), "even_sector_gap_median": float(np.median(gaps)),
                "max_eigen_residual": float(np.max(res))}
        if label == "e2e":
            # Scored on the rows actually written, so the number is comparable with a
            # model's accuracy on the shipped dataset, and so the guard below covers
            # every value np.log is handed.
            Xk, yk = X[keep], y[keep]
            # The criterion is a log-product of the couplings, so it exists only for
            # positive ones. Both ranges default to positive intervals; a caller who
            # widens either past zero gets the diagnostic omitted with a reason,
            # rather than a nan that quietly propagates into a plausible-looking
            # accuracy (comparisons against nan are simply False).
            if np.all(Xk > 0):
                z = np.log(Xk[:, : n - 1]).sum(1) - np.log(Xk[:, n - 1:]).sum(1)
                half = len(z) // 2
                cands = np.sort(z[:half])
                accs = [np.mean((z[:half] > c) == yk[:half]) for c in cands]
                c_best = cands[int(np.argmax(accs))]
                diag["product_criterion_holdout_acc"] = float(np.mean((z[half:] > c_best) == yk[half:]))
            else:
                diag["product_criterion_holdout_acc"] = None
                diag["product_criterion_note"] = (
                    f"omitted: the criterion is a log-product of the couplings and "
                    f"{int(np.sum(Xk <= 0))} of them are not positive, which needs "
                    f"J_range or h_range to span zero"
                )
        meta = dict(family="gs", n=n, kappa=kappa, J=[J_range[0], J_range[1]], h=[h_range[0], h_range[1]],
                    label_rule=rule, threshold=thr, margin=margin, shots=shots, seed=seed,
                    diagnostics=diag)
        xcols = [f"J{i}" for i in range(n - 1)] + [f"h{i}" for i in range(n)]
        return write_dataset(save_path, name, X[keep], xcols, y[keep], F[keep], meta,
                             phi_obs[keep], [p.label for p in pool])


def generate_ground_state_datasets(
    n_qubits=N_QUBITS,
    n_samples=N_SAMPLES,
    label=LABEL,
    kappa=KAPPA,
    n_terms=4,
    J_range=(0.5, 1.5),
    h_range=(0.2, 2.0),
    margin=0.0,
    shots=0,
    save_path=None,
    name=None,
    random_state=0,
    blas_threads=1,
):
    """
    Generate ground-state observable-learning datasets.

    Sweeps the Cartesian product of ``n_qubits``, ``n_samples``, ``label``,
    ``kappa`` and ``random_state``, writing one dataset per combination. Every
    configuration gets a fresh generator seeded from its own entry of
    ``random_state``, so a dataset does not depend on where it fell in the sweep
    and a one-element sweep reproduces a single-configuration call exactly.

    Parameters
    ----------
    n_qubits : int or list of int, default=[8]
        Chain lengths. Exact diagonalisation makes cost and memory grow as
        ``2 ** n``; 12 is a practical ceiling.
    n_samples : int or list of int, default=[400]
        Rows to draw per dataset, before ``margin`` filtering.
    label : str or list of str, default=['sparse', 'e2e']
        Label rule. ``'sparse'`` uses a hidden sparse observable over the local
        :math:`Z_2`-even pool; ``'e2e'`` uses the end-to-end correlator
        :math:`Z_0 Z_{n-1}`, which lies outside that pool and has a known
        classical twin.
    kappa : float or list of float, default=[0.5]
        Nearest-neighbour XX coupling. Zero leaves a free-fermion model; any
        non-zero value makes it interacting.
    n_terms : int, default=4
        Number of terms in the hidden sparse observable (``label='sparse'``).
    J_range : tuple of float, default=(0.5, 1.5)
        Range the ZZ couplings are drawn from, uniformly.
    h_range : tuple of float, default=(0.2, 2.0)
        Range the transverse fields are drawn from, uniformly.
    margin : float, default=0.0
        Drop rows whose continuous target lies within ``margin`` of the
        threshold, removing the ambiguous band around the boundary.
    shots : int, default=0
        Shots per Pauli for the ``phi_view`` features. ``0`` gives exact
        expectation values; a finite value adds the binomial estimation noise a
        real measurement would carry. Labels are always computed from the exact
        values, so shot noise degrades the features without moving the target.
    save_path : str, optional
        Directory to write into; defaults to ``'quantum_data'``.
    name : str, optional
        Override the generated dataset name. Only valid when the sweep yields a
        single dataset.
    random_state : int or list of int, default=0
        Seed, or seeds to sweep over. Also recorded in each dataset's metadata.
    blas_threads : int or None, default=1
        BLAS threads for the diagonalisation loop. One is much faster here than
        the default -- these matrices are small enough that threading costs more
        than it saves. Pass ``None`` to leave threading untouched.

    Returns
    -------
    list of dict
        The metadata written for each dataset.

    Raises
    ------
    ValueError
        If ``name`` is given for a sweep of more than one configuration, which
        would make every dataset overwrite the previous one.

    Notes
    -----
    Feature count in ``x_view`` is ``2 n - 1`` (``n - 1`` couplings and ``n``
    fields), which is the qubit count QProfiler will use with ``embeddings:
    ['none']``. ``phi_view`` has ``5 n - 5`` features, one per pooled Pauli.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_ground_state_datasets
    >>> generate_ground_state_datasets(n_qubits=4, n_samples=32, label='sparse',
    ...                                save_path='qdata')     # doctest: +SKIP
    Generating ground-state observable datasets...
    """
    print("Generating ground-state observable datasets...")
    if save_path is None:
        save_path = "quantum_data"
    configurations = list(itertools.product(
        as_list(n_qubits), as_list(n_samples), as_list(label), as_float_list(kappa), as_list(random_state)
    ))
    reject_name_for_sweep(
        name, len(configurations), ["n_qubits", "n_samples", "label", "kappa", "random_state"]
    )
    names = [name or _ground_state_name(lab, n, kap, seed)
             for n, _, lab, kap, seed in configurations]
    ensure_unique_names(names, unnamed_knobs=["n_samples"])
    metas = []
    for (n, n_rows, lab, kap, seed), dataset_name in zip(configurations, names):
        metas.append(_ground_state_dataset(
            save_path, dataset_name, n, n_rows, lab, n_terms, kap, J_range, h_range,
            float(margin), shots, seed, blas_threads,
        ))
    return metas
