"""
Generate time-evolution datasets with a tunable difficulty ladder.

Inputs are computational-basis states :math:`|x\\rangle`, :math:`x \\in \\{0,1\\}^n`,
evolved for a time :math:`\\tau` under one *fixed* mixed-field Ising chain, and
the label comes from a hidden sparse observable on the evolved state. This is a
small-``n`` instance of the time-evolution concept class of Molteni, Gyurik and
Dunjko (npj Quantum Information 12, 19, 2026).

The point of the family is the :math:`\\tau` ladder. Because the input space is
the full Boolean cube, the continuous target can be evaluated on *all*
:math:`2^n` inputs and decomposed exactly into Walsh (Fourier) degrees. Short
evolution keeps the target low-degree and easy; longer evolution spreads weight
to higher degrees, and every learner reading ``x_view`` should degrade as it does.
The effective degree is recorded per dataset as
``diagnostics.walsh_effective_degree``, so the difficulty is measured rather than
assumed.

``phi_view`` holds the local Pauli expectation values of the evolved state -- the
"measure first, with the right measurements" learner. Its advantage over
``x_view`` is expected to *grow* with :math:`\\tau`, and that gap is the
quantity this family exists to expose.

The chain carries weak disorder in all three couplings. That breaks the
reflection symmetry which would otherwise make many observables degenerate, while
keeping the spectrum chaotic; ``diagnostics.level_spacing_ratio`` reports the mean
adjacent-gap ratio, which should sit near the GOE value of 0.53 rather than the
Poisson value of 0.39.
"""

import itertools

import numpy as np

from .quantum_core import (
    as_float_list,
    as_list,
    blas_limit,
    ensure_unique_names,
    expvals,
    ising_terms,
    level_spacing_ratio,
    pauli_sum,
    pool_local,
    reject_name_for_sweep,
    shot_noise,
    sparse_observable,
    threshold,
    walsh_degree_profile,
    write_dataset,
)

#: Parameters varied across the default sweep.
N_QUBITS = [10]
N_SAMPLES = [400]
TAUS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _time_evolution_name(n, n_terms, seed):
    """The default base name for one configuration; ``_tau<tau>`` is appended per tau."""
    return f"te_n{n}_s{n_terms}_seed{seed}"


def _time_evolution_datasets(save_path, name, n, n_rows, n_terms, taus, disorder,
                             margin, shots, seed, blas_threads):
    """Build and write one dataset per entry of ``taus``, sharing one Hamiltonian."""
    rng = np.random.default_rng(seed)
    with blas_limit(n, blas_threads):
        D = 1 << n
        J = rng.uniform(1 - disorder, 1 + disorder, n - 1)   # weak disorder breaks reflection symmetry
        h = rng.uniform(1 - disorder, 1 + disorder, n)       # while keeping level statistics GOE-like
        g = rng.uniform(0.5 - disorder, 0.5 + disorder, n)   # (check diagnostics.level_spacing_ratio)
        H = pauli_sum(n, ising_terms(n, J, h, g=g, sign=1.0)).toarray()
        E, V = np.linalg.eigh(H)
        pool = pool_local(n)
        S, alpha = sparse_observable(pool, n_terms, rng)
        xs = np.sort(rng.choice(D, size=n_rows, replace=False))
        bits = (xs[:, None] >> np.arange(n)) & 1
        lsr = level_spacing_ratio(E)
        metas = []
        for tau in taus:
            U = (V * np.exp(-1j * E * tau)) @ V.conj().T     # column x = U|x>
            PHI_all = expvals(U, pool)                         # (2^n, |pool|)
            F_all = PHI_all[:, S] @ alpha
            prof, eff_deg = walsh_degree_profile(F_all, n)
            F = F_all[xs]
            thr, keep = threshold(F, margin)
            y = (F > thr).astype(int)
            dataset = f"{name}_tau{tau:g}"
            meta = dict(family="te", n=n, tau=tau, H={"J": J.tolist(), "h": h.tolist(), "g": g.tolist()},
                        label_rule={"observable": [pool[j].label for j in S], "alpha": alpha.tolist()},
                        threshold=thr, margin=margin, shots=shots, seed=seed,
                        diagnostics={"level_spacing_ratio": lsr,
                                     "walsh_variance_by_degree": prof.tolist(),
                                     "walsh_effective_degree": eff_deg})
            metas.append(write_dataset(save_path, dataset, bits[keep], [f"b{i}" for i in range(n)],
                                       y[keep], F[keep], meta,
                                       shot_noise(PHI_all[xs][keep], shots, rng),
                                       [p.label for p in pool]))
        return metas


def generate_time_evolution_datasets(
    n_qubits=N_QUBITS,
    n_samples=N_SAMPLES,
    taus=TAUS,
    n_terms=4,
    disorder=0.1,
    margin=0.0,
    shots=0,
    save_path=None,
    name=None,
    random_state=0,
    blas_threads=1,
):
    """
    Generate time-evolution datasets forming a difficulty ladder in evolution time.

    Sweeps the Cartesian product of ``n_qubits``, ``n_samples`` and
    ``random_state``; each configuration then writes one dataset per entry of
    ``taus``, all sharing the same Hamiltonian, the same hidden observable and the
    same inputs. Holding those fixed is what makes the ladder interpretable: only
    the evolution time changes between the datasets of one configuration.

    Parameters
    ----------
    n_qubits : int or list of int, default=[10]
        Chain lengths. The target is evaluated on all ``2 ** n`` inputs to get the
        exact Walsh spectrum, so cost grows as ``2 ** n``.
    n_samples : int or list of int, default=[400]
        Rows to draw per dataset, sampled without replacement from the
        ``2 ** n`` possible inputs. Must not exceed ``2 ** n``.
    taus : float or list of float, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Evolution times. Each produces its own dataset, named with a ``_tau``
        suffix.
    n_terms : int, default=4
        Number of terms in the hidden sparse observable.
    disorder : float, default=0.1
        Half-width of the uniform disorder on the couplings. Enough to break
        reflection symmetry; large values push the model towards localisation and
        away from the chaotic regime the ladder assumes.
    margin : float, default=0.0
        Drop rows whose continuous target lies within ``margin`` of the threshold.
    shots : int, default=0
        Shots per Pauli for the ``phi_view`` features; ``0`` gives exact values. Note
        that a non-zero value draws from the configuration's generator inside the
        ``taus`` loop, so the dataset for one tau depends on which taus preceded it:
        ``taus=[1.0]`` does not reproduce the ``_tau1`` dataset of a
        ``taus=[0.25, 0.5, 1.0]`` sweep. At the default ``0`` nothing is drawn and the
        datasets are independent of the sweep.
    save_path : str, optional
        Directory to write into; defaults to ``'quantum_data'``.
    name : str, optional
        Base dataset name, to which the ``_tau`` suffix is still appended. Only
        valid when the sweep yields a single configuration.
    random_state : int or list of int, default=0
        Seed, or seeds to sweep over.
    blas_threads : int or None, default=1
        BLAS threads for the diagonalisation. Pass ``None`` to leave threading
        untouched.

    Returns
    -------
    list of dict
        The metadata written for each dataset, across all configurations and
        times.

    Raises
    ------
    ValueError
        If ``name`` is given for a sweep of more than one configuration.

    Notes
    -----
    ``x_view`` has ``n`` binary features, one per bit of the input bitstring;
    ``phi_view`` has ``6 n - 3``. Both views keep the same rows, so a comparison
    between them is paired.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_time_evolution_datasets
    >>> generate_time_evolution_datasets(n_qubits=4, n_samples=12, taus=[0.25, 1.0],
    ...                                  save_path='qdata')   # doctest: +SKIP
    Generating time-evolution datasets...
    """
    print("Generating time-evolution datasets...")
    if save_path is None:
        save_path = "quantum_data"
    configurations = list(itertools.product(
        as_list(n_qubits), as_list(n_samples), as_list(random_state)
    ))
    reject_name_for_sweep(name, len(configurations), ["n_qubits", "n_samples", "random_state"])
    tau_list = as_float_list(taus)
    bases = [name or _time_evolution_name(n, n_terms, seed) for n, _, seed in configurations]
    ensure_unique_names([f"{base}_tau{tau:g}" for base in bases for tau in tau_list],
                        unnamed_knobs=["n_samples"])
    # Checked for the whole sweep before anything is written, like ensure_unique_names
    # above: inside the loop below, n_qubits=[10, 4] wrote the five tau datasets of the
    # first configuration and only then raised on the second.
    for n, n_rows, _ in configurations:
        if n_rows > (1 << n):
            raise ValueError(
                f"n_samples={n_rows} exceeds the {1 << n} distinct inputs available at "
                f"n_qubits={n}; inputs are drawn without replacement."
            )
    metas = []
    for (n, n_rows, seed), base in zip(configurations, bases):
        metas += _time_evolution_datasets(
            save_path, base, n, n_rows, n_terms, tau_list, disorder,
            float(margin), shots, seed, blas_threads,
        )
    return metas
