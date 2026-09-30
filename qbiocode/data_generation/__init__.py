"""
Data Generation Module for QBioCode.

This module provides functions to generate synthetic datasets for testing
machine learning algorithms. Each function creates multiple dataset configurations
with varying parameters, useful for benchmarking and evaluation.

Available dataset generators:
- generate_blobs_datasets: Isotropic Gaussian blobs (clusters)
- generate_circles_datasets: 2D concentric circles
- generate_moons_datasets: 2D interleaving half-circles
- generate_classification_datasets: High-dimensional multi-class data
- generate_s_curve_datasets: 3D S-shaped manifold
- generate_spheres_datasets: N-dimensional concentric spheres
- generate_spirals_datasets: N-dimensional intertwined spirals
- generate_swiss_roll_datasets: 3D Swiss roll manifold

Simulated quantum datasets, from exact statevector simulation. These are binary
by construction and carry per-dataset metadata recording the label rule, so what
a model is being asked to learn stays auditable:
- generate_ground_state_datasets: Ground-state local Pauli expectations (Hamiltonian learning)
- generate_time_evolution_datasets: Sparse observables after time evolution, a tau-indexed difficulty ladder
- generate_hamiltonian_learning_datasets: Hamiltonian parameters from quench measurement records
- generate_quantum_label_datasets: Classical inputs with circuit-generated labels
- generate_engineered_kernel_datasets: Labels engineered to favour a quantum kernel (positive control)
- run_selftest: verify the physics behind all five on the current install
"""

from .make_blobs import generate_blobs_datasets, generate_default_blobs_datasets
from .make_circles import generate_circles_datasets
from .make_class import generate_classification_datasets
from .make_engineered_kernel import generate_engineered_kernel_datasets
from .make_ground_state import generate_ground_state_datasets
from .make_hamiltonian_learning import generate_hamiltonian_learning_datasets
from .make_moons import generate_moons_datasets
from .make_quantum_labels import generate_quantum_label_datasets
from .make_s_curve import generate_s_curve_datasets
from .make_spheres import generate_spheres_datasets
from .make_spirals import generate_spirals_datasets
from .make_swiss_roll import generate_swiss_roll_datasets
from .make_time_evolution import generate_time_evolution_datasets
from .quantum_selftest import run_selftest

__all__ = [
    "generate_blobs_datasets",
    "generate_default_blobs_datasets",
    "generate_circles_datasets",
    "generate_moons_datasets",
    "generate_classification_datasets",
    "generate_s_curve_datasets",
    "generate_spheres_datasets",
    "generate_spirals_datasets",
    "generate_swiss_roll_datasets",
    "generate_ground_state_datasets",
    "generate_time_evolution_datasets",
    "generate_hamiltonian_learning_datasets",
    "generate_quantum_label_datasets",
    "generate_engineered_kernel_datasets",
    "run_selftest",
]

