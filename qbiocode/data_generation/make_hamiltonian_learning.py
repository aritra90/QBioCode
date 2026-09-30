"""
Generate Hamiltonian-learning-as-classification datasets.

Each row is one random mixed-field Ising Hamiltonian :math:`H(\\lambda)`. The
features are local expectation values measured at one or more times after a quench
from :math:`|0\\ldots0\\rangle`, and the label is a property of the unknown
parameters :math:`\\lambda` -- specifically
:math:`\\mathbb{1}[\\overline{J} - \\overline{h} > \\mathrm{median}]`. So the task
is the classification form of Hamiltonian learning: recover something about the
Hamiltonian from measurement records of its dynamics.

Unlike the other families, the features here *are* measurement records by
construction, which makes ``shots`` the natural knob: it controls how much
estimation noise stands between the learner and the underlying parameters.

This family is a second **classically-easy control**. The short-time expansion
:math:`\\langle Z_i(t) \\rangle = 1 - 2 h_i^2 t^2 + O(t^4)` hands a learner
:math:`h_i` almost directly, so classical models should do well; if they do not,
suspect the features or the label rather than the models. The test suite asserts
that expansion numerically, so the property this control depends on is checked
rather than assumed.

Only ``x_view`` is written: the features are already expectation values, so there
is no second "quantum feature" view to contrast against.
"""

import itertools

import numpy as np

from .quantum_core import (
    Pauli,
    as_float_list,
    as_list,
    blas_limit,
    ensure_unique_names,
    expvals,
    ising_terms,
    pauli_sum,
    reject_name_for_sweep,
    shot_noise,
    threshold,
    write_dataset,
)

#: Parameters varied across the default sweep.
N_QUBITS = [6]
N_SAMPLES = [400]
TIMES = [0.5]
SHOTS = [1000]


def _hamiltonian_learning_name(n, longitudinal_field, shots, seed):
    """The default dataset name for one configuration."""
    return f"hl_n{n}_g{longitudinal_field}_shots{shots}_s{seed}"


def _hamiltonian_learning_dataset(save_path, name, n, n_rows, times, longitudinal_field,
                                  J_range, h_range, margin, shots, seed, blas_threads):
    """Build and write one Hamiltonian-learning dataset."""
    rng = np.random.default_rng(seed)
    with blas_limit(n, blas_threads):
        D = 1 << n
        obs = [Pauli(n, {i: s}) for s in "ZX" for i in range(n)]
        psi0 = np.zeros(D, complex)
        psi0[0] = 1.0
        feats, F = [], []
        for _ in range(n_rows):
            J = rng.uniform(J_range[0], J_range[1], n - 1)
            h = rng.uniform(h_range[0], h_range[1], n)
            H = pauli_sum(n, ising_terms(n, J, h, g=np.full(n, longitudinal_field), sign=1.0)).toarray()
            E, V = np.linalg.eigh(H)
            c0 = V.conj().T @ psi0
            row = []
            for t in times:
                row.append(expvals(V @ (np.exp(-1j * E * t) * c0), obs)[0])
            feats.append(np.concatenate(row))
            F.append(J.mean() - h.mean())
        feats, F = np.array(feats), np.array(F)
        thr, keep = threshold(F, margin)
        y = (F > thr).astype(int)
        cols = [f"{p.label}_t{t:g}" for t in times for p in obs]
        meta = dict(family="hl", n=n, g=longitudinal_field, times=times,
                    label_rule="1[mean(J) - mean(h) > median]",
                    threshold=thr, margin=margin, shots=shots, seed=seed)
        return write_dataset(save_path, name, shot_noise(feats, shots, rng)[keep], cols,
                             y[keep], F[keep], meta)


def generate_hamiltonian_learning_datasets(
    n_qubits=N_QUBITS,
    n_samples=N_SAMPLES,
    times=TIMES,
    shots=SHOTS,
    longitudinal_field=0.5,
    J_range=(0.5, 1.5),
    h_range=(0.5, 1.5),
    margin=0.0,
    save_path=None,
    name=None,
    random_state=0,
    blas_threads=1,
):
    """
    Generate Hamiltonian-learning-as-classification datasets.

    Sweeps the Cartesian product of ``n_qubits``, ``n_samples``, ``shots`` and
    ``random_state``, writing one dataset per combination. ``times`` is *not*
    swept: all requested times become feature blocks of the same dataset, since a
    measurement record at several times is one record.

    Parameters
    ----------
    n_qubits : int or list of int, default=[6]
        Chain lengths. One dense diagonalisation is done per row, so this is the
        most expensive family per sample; cost grows as ``2 ** n``.
    n_samples : int or list of int, default=[400]
        Rows (Hamiltonians) per dataset, before ``margin`` filtering.
    times : float or list of float, default=[0.5]
        Quench times at which the local observables are measured. Each time adds
        ``2 n`` feature columns, so the feature count is ``2 n`` times
        ``len(times)``.
    shots : int or list of int, default=[1000]
        Shots per observable, swept. ``0`` gives exact expectation values. Labels
        are always computed from the exact parameters, so this adds noise to the
        features only.
    longitudinal_field : float, default=0.5
        Uniform longitudinal field, recorded as ``g`` in the metadata and in the
        dataset name. It makes the model non-integrable.
    J_range : tuple of float, default=(0.5, 1.5)
        Range the ZZ couplings are drawn from, uniformly.
    h_range : tuple of float, default=(0.5, 1.5)
        Range the transverse fields are drawn from, uniformly.
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
        BLAS threads for the diagonalisation loop. Pass ``None`` to leave
        threading untouched.

    Returns
    -------
    list of dict
        The metadata written for each dataset.

    Raises
    ------
    ValueError
        If ``name`` is given for a sweep of more than one configuration.

    Notes
    -----
    Only ``x_view`` is written. Feature columns are named ``<pauli>_t<time>``,
    e.g. ``Z0_t0.5``, with all Z observables before all X observables.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_hamiltonian_learning_datasets
    >>> generate_hamiltonian_learning_datasets(n_qubits=4, n_samples=32, shots=1000,
    ...                                        save_path='qdata')   # doctest: +SKIP
    Generating Hamiltonian-learning datasets...
    """
    print("Generating Hamiltonian-learning datasets...")
    if save_path is None:
        save_path = "quantum_data"
    configurations = list(itertools.product(
        as_list(n_qubits), as_list(n_samples), as_list(shots), as_list(random_state)
    ))
    reject_name_for_sweep(
        name, len(configurations), ["n_qubits", "n_samples", "shots", "random_state"]
    )
    names = [name or _hamiltonian_learning_name(n, float(longitudinal_field), sh, seed)
             for n, _, sh, seed in configurations]
    ensure_unique_names(names, unnamed_knobs=["n_samples"])
    metas = []
    for (n, n_rows, sh, seed), dataset_name in zip(configurations, names):
        metas.append(_hamiltonian_learning_dataset(
            save_path, dataset_name, n, n_rows, as_float_list(times), float(longitudinal_field),
            J_range, h_range, float(margin), sh, seed, blas_threads,
        ))
    return metas
