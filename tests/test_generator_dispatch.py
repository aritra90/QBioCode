import pytest

from conftest import ensure_package, load_module


def load_generator_module():
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.data_generation", "qbiocode/data_generation")

    # quantum_core first: the five make_* quantum modules import from it by relative
    # name, and these modules are loaded by path rather than imported as a package.
    load_module(
        "qbiocode.data_generation.quantum_core",
        "qbiocode/data_generation/quantum_core.py",
    )

    for module_name in [
        "make_circles",
        "make_moons",
        "make_class",
        "make_s_curve",
        "make_spheres",
        "make_spirals",
        "make_swiss_roll",
        "make_engineered_kernel",
        "make_ground_state",
        "make_hamiltonian_learning",
        "make_quantum_labels",
        "make_time_evolution",
    ]:
        load_module(
            f"qbiocode.data_generation.{module_name}",
            f"qbiocode/data_generation/{module_name}.py",
        )

    return load_module(
        "qbiocode.data_generation.generator",
        "qbiocode/data_generation/generator.py",
    )


@pytest.mark.parametrize(
    ("dataset_type", "module_attr", "function_name", "expected_kwargs"),
    [
        (
            "circles",
            "circles",
            "generate_circles_datasets",
            {"n_samples": [9], "noise": [0.2], "save_path": "out", "random_state": 5},
        ),
        (
            "classes",
            "make_class",
            "generate_classification_datasets",
            {
                "n_samples": [9],
                "n_features": [6],
                "n_informative": [2],
                "n_redundant": [1],
                "n_classes": [2],
                "n_clusters_per_class": [1],
                "weights": [[0.5, 0.5]],
                "save_path": "out",
                "random_state": 5,
            },
        ),
        (
            "spheres",
            "spheres",
            "generate_spheres_datasets",
            {"n_s": [9], "dim": [6], "radius": [4], "save_path": "out", "random_state": 5},
        ),
        (
            "swiss_roll",
            "swiss_roll",
            "generate_swiss_roll_datasets",
            {
                "n_samples": [9],
                "noise": [0.2],
                "hole": [True],
                "save_path": "out",
                "random_state": 5,
            },
        ),
        # The quantum families take the qubit count through `dim` and the row count
        # through `n_samples`, and nothing else: every remaining knob travels in the
        # single `quantum_args` dict, so `generate_data` does not grow a parameter per
        # family. `dim=[6]` below is therefore the qubit count, not a feature count.
        (
            "ground_state",
            "ground_state",
            "generate_ground_state_datasets",
            {"n_qubits": [6], "n_samples": [9], "save_path": "out", "random_state": 5},
        ),
        (
            "time_evolution",
            "time_evolution",
            "generate_time_evolution_datasets",
            {"n_qubits": [6], "n_samples": [9], "save_path": "out", "random_state": 5},
        ),
        (
            "hamiltonian_learning",
            "hamiltonian_learning",
            "generate_hamiltonian_learning_datasets",
            {"n_qubits": [6], "n_samples": [9], "save_path": "out", "random_state": 5},
        ),
        (
            "quantum_labels",
            "quantum_labels",
            "generate_quantum_label_datasets",
            {"n_qubits": [6], "n_samples": [9], "save_path": "out", "random_state": 5},
        ),
        (
            "engineered_kernel",
            "engineered_kernel",
            "generate_engineered_kernel_datasets",
            {"n_qubits": [6], "n_samples": [9], "save_path": "out", "random_state": 5},
        ),
    ],
)
def test_generate_data_dispatches_to_expected_backend(
    monkeypatch,
    dataset_type,
    module_attr,
    function_name,
    expected_kwargs,
):
    generator = load_generator_module()
    captured = {}

    def fake_backend(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(getattr(generator, module_attr), function_name, fake_backend)

    generator.generate_data(
        type_of_data=dataset_type,
        save_path="out",
        n_samples=[9],
        noise=[0.2],
        hole=[True],
        n_classes=[2],
        dim=[6],
        rad=[4],
        n_features=[6],
        n_informative=[2],
        n_redundant=[1],
        n_clusters_per_class=[1],
        weights=[[0.5, 0.5]],
        random_state=5,
    )

    assert captured == expected_kwargs


def test_generate_data_rejects_unknown_dataset_type():
    generator = load_generator_module()

    with pytest.raises(ValueError, match="Invalid type_of_data"):
        generator.generate_data(type_of_data="unknown", save_path="out")


@pytest.mark.parametrize(
    ("dataset_type", "module_attr", "function_name"),
    [
        ("ground_state", "ground_state", "generate_ground_state_datasets"),
        ("time_evolution", "time_evolution", "generate_time_evolution_datasets"),
        ("hamiltonian_learning", "hamiltonian_learning", "generate_hamiltonian_learning_datasets"),
        ("quantum_labels", "quantum_labels", "generate_quantum_label_datasets"),
        ("engineered_kernel", "engineered_kernel", "generate_engineered_kernel_datasets"),
    ],
)
def test_a_quantum_family_left_at_the_defaults_gets_its_own_runsheet_size(
    monkeypatch, dataset_type, module_attr, function_name
):
    """The shared ``dim``/``n_samples`` defaults are wrong for exact simulation.

    ``DIM`` reaches 12 qubits -- a 4096-dimensional Hilbert space diagonalised once per
    row -- and ``N_SAMPLES`` holds ten row counts that no dataset name distinguishes, so
    a sweep over it would write ten datasets to one filename. A caller who leaves both
    alone therefore gets the family's documented runsheet size instead of the classical
    generators' defaults.
    """
    generator = load_generator_module()
    module = getattr(generator, module_attr)
    captured = {}

    monkeypatch.setattr(module, function_name, lambda **kwargs: captured.update(kwargs))
    generator.generate_data(type_of_data=dataset_type, save_path="out")

    assert captured["n_qubits"] == module.N_QUBITS
    assert captured["n_samples"] == module.N_SAMPLES
    assert captured["n_qubits"] is not generator.DIM
    assert captured["n_samples"] is not generator.N_SAMPLES


def test_quantum_args_reach_the_generator(monkeypatch):
    """Everything past qubit and row count travels in one dict, including overrides."""
    generator = load_generator_module()
    captured = {}

    monkeypatch.setattr(
        generator.ground_state, "generate_ground_state_datasets",
        lambda **kwargs: captured.update(kwargs),
    )
    generator.generate_data(
        type_of_data="ground_state",
        save_path="out",
        dim=[4],
        n_samples=[16],
        quantum_args={"label": "e2e", "shots": 1000, "margin": 0.05, "n_samples": [8]},
    )

    assert captured["label"] == "e2e"
    assert captured["shots"] == 1000
    assert captured["margin"] == 0.05
    assert captured["n_qubits"] == [4]
    # quantum_args is applied last, so it can override a value dim/n_samples supplied.
    assert captured["n_samples"] == [8]


def test_the_unknown_type_error_lists_every_valid_name():
    """The message is the only discovery path for the type names, so it must be complete."""
    generator = load_generator_module()

    with pytest.raises(ValueError) as excinfo:
        generator.generate_data(type_of_data="unknown", save_path="out")

    message = str(excinfo.value)
    classical = ["circles", "moons", "classes", "s_curve", "spheres", "spirals", "swiss_roll"]
    missing = [name for name in classical + list(generator.QUANTUM_MODULES) if repr(name) not in message]
    assert not missing, f"valid type names absent from the ValueError: {missing}"
