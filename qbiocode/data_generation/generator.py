"""
Main data generation interface for QBioCode.

This module provides a unified interface to generate various types of synthetic
datasets for machine learning benchmarking and evaluation.
"""

### Imports ###

import qbiocode.data_generation.make_circles as circles
import qbiocode.data_generation.make_class as make_class
import qbiocode.data_generation.make_engineered_kernel as engineered_kernel
import qbiocode.data_generation.make_ground_state as ground_state
import qbiocode.data_generation.make_hamiltonian_learning as hamiltonian_learning
import qbiocode.data_generation.make_moons as moons
import qbiocode.data_generation.make_quantum_labels as quantum_labels
import qbiocode.data_generation.make_s_curve as s_curve
import qbiocode.data_generation.make_spheres as spheres
import qbiocode.data_generation.make_spirals as spirals
import qbiocode.data_generation.make_swiss_roll as swiss_roll
import qbiocode.data_generation.make_time_evolution as time_evolution

### Main Function ###

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLE = [True, False]
N_CLASSES = [2]
DIM = [3, 6, 9, 12]
RAD = [3, 6, 9, 12]
N_FEATURES = list(range(10, 60, 20))
N_INFORMATIVE = list(range(2, 8, 4))
N_REDUNDANT = list(range(2, 8, 4))
N_CLUSTERS_PER_CLASS = list(range(1, 2, 3))
WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]

#: The quantum families, mapped to the module holding each one's generator and
#: runsheet defaults. Kept as data so the dispatch and the error message that lists
#: the valid names cannot disagree.
QUANTUM_MODULES = {
    "ground_state": (ground_state, "generate_ground_state_datasets"),
    "time_evolution": (time_evolution, "generate_time_evolution_datasets"),
    "hamiltonian_learning": (hamiltonian_learning, "generate_hamiltonian_learning_datasets"),
    "quantum_labels": (quantum_labels, "generate_quantum_label_datasets"),
    "engineered_kernel": (engineered_kernel, "generate_engineered_kernel_datasets"),
}


def _quantum_kwargs(module, quantum_args, save_path, n_samples, dim, random_state):
    """Assemble the keyword arguments for one quantum generator.

    ``dim`` carries the qubit count and ``n_samples`` the row count, so the quantum
    families need no parameters of their own on
    :func:`generate_data`'s already wide signature. But their *defaults* cannot be
    shared: :data:`DIM` reaches 12 qubits, a 4096-dimensional Hilbert space
    diagonalised once per row, and :data:`N_SAMPLES` holds ten row counts that no
    dataset name distinguishes. So a caller who leaves them alone gets the family's
    documented runsheet configuration instead, and a caller who sets either one gets
    exactly what they asked for.

    Anything else -- ``label``, ``kappa``, ``taus``, ``shots``, ``encoding``,
    ``reps``, ``blas_threads`` -- goes in ``quantum_args`` and is passed straight
    through, so an unknown key raises :class:`TypeError` naming it rather than being
    silently dropped. Keys there win over the positional arguments, which is how
    ``random_state`` can be set back to the runsheet's 0.

    Parameters
    ----------
    module : module
        The ``make_*`` module, read for its ``N_QUBITS`` and ``N_SAMPLES`` defaults.
    quantum_args : dict or None
        Family-specific keyword arguments.
    save_path : str
        Directory to write into.
    n_samples : list of int
        Row counts, honoured only if not :data:`N_SAMPLES`.
    dim : list of int
        Qubit counts, honoured only if not :data:`DIM`.
    random_state : int
        Seed.

    Returns
    -------
    dict
        Keyword arguments for the family's ``generate_*_datasets`` function.
    """
    kwargs = dict(
        n_qubits=module.N_QUBITS if dim is DIM else dim,
        n_samples=module.N_SAMPLES if n_samples is N_SAMPLES else n_samples,
        save_path=save_path,
        random_state=random_state,
    )
    kwargs.update(quantum_args or {})
    return kwargs


def generate_data(
    type_of_data=None,
    save_path=None,
    n_samples=N_SAMPLES,
    noise=NOISE,
    hole=HOLE,
    n_classes=N_CLASSES,
    dim=DIM,
    rad=RAD,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_clusters_per_class=N_CLUSTERS_PER_CLASS,
    weights=WEIGHTS,
    quantum_args=None,
    random_state=42,
):
    """
    Generate synthetic datasets for machine learning benchmarking.

    Unified interface to generate various types of synthetic datasets with
    configurable parameters. Each dataset type creates multiple configurations
    by varying the specified parameters.

    Parameters
    ----------
    type_of_data : str
        Type of dataset to generate. Classical options: 'circles', 'moons',
        'classes', 's_curve', 'spheres', 'spirals', 'swiss_roll'. Simulated-quantum
        options: 'ground_state', 'time_evolution', 'hamiltonian_learning',
        'quantum_labels', 'engineered_kernel'.
    save_path : str
        Directory path where datasets will be saved.
    n_samples : list of int, default=range(100, 300, 20)
        Sample sizes for dataset configurations.
    noise : list of float, default=[0.1, 0.2, ..., 0.9]
        Noise levels to apply.
    hole : list of bool, default=[True, False]
        Whether to include hole (for swiss_roll only).
    n_classes : list of int, default=[2]
        Number of classes (for spirals and classes).
    dim : list of int, default=[3, 6, 9, 12]
        Dimensionalities (for spheres and spirals).
    rad : list of float, default=[3, 6, 9, 12]
        Radii (for spheres only).
    n_features : list of int, default=range(10, 60, 20)
        Feature counts (for classes only).
    n_informative : list of int, default=range(2, 8, 4)
        Informative feature counts (for classes only).
    n_redundant : list of int, default=range(2, 8, 4)
        Redundant feature counts (for classes only).
    n_clusters_per_class : list of int, default=range(1, 2, 3)
        Clusters per class (for classes only).
    weights : list of list of float, default=[[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
        Class weight distributions (for classes only).
    quantum_args : dict, optional
        Extra keyword arguments for the quantum families, passed straight to the
        family's generator -- ``label`` and ``kappa`` for 'ground_state', ``taus``
        for 'time_evolution', ``times`` and ``shots`` for 'hamiltonian_learning',
        ``encoding`` and ``reps`` for 'quantum_labels', ``gamma_q`` for
        'engineered_kernel', and ``margin``, ``name`` or ``blas_threads`` for any of
        them. For these families ``dim`` is the qubit count and ``n_samples`` the row
        count; left at their defaults, each family's documented runsheet
        configuration is used instead, since ``dim``'s default reaches 12 qubits.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    None
        Saves generated datasets to the specified path.

    Raises
    ------
    ValueError
        If type_of_data is not one of the supported types.

    Examples
    --------
    >>> from qbiocode.data_generation import generate_data
    >>> generate_data(type_of_data='circles', save_path='data/circles')
    Generating circles dataset...
    Dataset generation complete.

    >>> generate_data(type_of_data='ground_state', save_path='data/quantum',
    ...               dim=[4], n_samples=[16])                      # doctest: +SKIP
    Generating ground-state observable datasets...
    Dataset generation complete.
    """

    if type_of_data == "circles":
        # Generate circles dataset
        circles.generate_circles_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "moons":
        # Generate moons dataset
        moons.generate_moons_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "classes":
        # Generate higher-dimensional classification dataset
        make_class.generate_classification_datasets(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights,
            save_path=save_path,
            random_state=random_state,
        )
    elif type_of_data == "s_curve":
        # Generate S-curve dataset
        s_curve.generate_s_curve_datasets(
            n_samples=n_samples, noise=noise, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "spheres":
        # Generate spheres dataset
        spheres.generate_spheres_datasets(
            n_s=n_samples, dim=dim, radius=rad, save_path=save_path, random_state=random_state
        )
    elif type_of_data == "spirals":
        # Generate spirals dataset
        spirals.generate_spirals_datasets(
            n_s=n_samples,
            n_c=n_classes,
            n_n=noise,
            n_d=dim,
            save_path=save_path,
            random_state=random_state,
        )
    elif type_of_data == "swiss_roll":
        # Generate Swiss roll dataset
        swiss_roll.generate_swiss_roll_datasets(
            n_samples=n_samples,
            noise=noise,
            hole=hole,
            save_path=save_path,
            random_state=random_state,
        )
    elif type_of_data in QUANTUM_MODULES:
        # Simulated quantum datasets: binary by construction, with the label rule
        # recorded per dataset in <save_path>/meta/<name>.json.
        module, function = QUANTUM_MODULES[type_of_data]
        getattr(module, function)(
            **_quantum_kwargs(module, quantum_args, save_path, n_samples, dim, random_state)
        )
    else:
        valid = ["circles", "moons", "classes", "s_curve", "spheres", "spirals",
                 "swiss_roll"] + list(QUANTUM_MODULES)
        raise ValueError(
            f"Invalid type_of_data {type_of_data!r}. Choose from "
            + ", ".join(repr(name) for name in valid[:-1])
            + f", or {valid[-1]!r}."
        )

    print("Dataset generation complete.")
    return
