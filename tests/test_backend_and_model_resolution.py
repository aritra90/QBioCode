"""Tests for the two config resolvers that run before any validation.

Both were added with the ``classical_model``/``quantum_model`` split and the
``statevector_simulator``/``mps_simulator`` backend rule, and both rewrite ``args`` in
place before the rest of the run reads it -- so a mistake here misconfigures every
model silently rather than raising. Neither had coverage.
"""

import pytest
from omegaconf import OmegaConf

from qbiocode.apps.qprofiler.qprofiler import _resolve_backend_alias, _resolve_model_lists
from qbiocode.evaluation.model_run import QUANTUM_MODELS
from qbiocode.utils.qutils import BACKEND_ALIASES, normalize_backend


class _Log:
    """Minimal logger stand-in; the resolvers only ever call .info()."""

    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(msg % a if a else msg)


# --------------------------------------------------------------------------- backend


@pytest.mark.parametrize(
    "alias,expected_backend,expected_sim_method",
    [
        ("statevector_simulator", "simulator", None),
        ("mps_simulator", "simulator_aer", "matrix_product_state"),
        ("simulator", "simulator", None),
        ("simulator_aer", "simulator_aer", None),
    ],
)
def test_normalize_backend_maps_each_alias(alias, expected_backend, expected_sim_method):
    out = normalize_backend({"backend": alias})
    assert out["backend"] == expected_backend
    assert out.get("sim_method") == expected_sim_method


def test_every_alias_in_the_table_is_covered_by_the_test_above():
    """Guards against a new alias being added without a mapping assertion."""
    covered = {"statevector_simulator", "mps_simulator", "simulator", "simulator_aer"}
    assert set(BACKEND_ALIASES) == covered


def test_normalize_backend_passes_ibm_devices_through_untouched():
    args = {"backend": "ibm_torino", "sim_method": "matrix_product_state"}
    out = normalize_backend(args)
    assert out["backend"] == "ibm_torino"
    assert out["sim_method"] == "matrix_product_state"


def test_normalize_backend_does_not_mutate_its_argument():
    """Several callers hold the run config across models; mutation would leak."""
    args = {"backend": "mps_simulator"}
    normalize_backend(args)
    assert args == {"backend": "mps_simulator"}


def test_normalize_backend_is_idempotent():
    """qprofiler resolves once, but a direct API caller may resolve again."""
    once = normalize_backend({"backend": "mps_simulator"})
    twice = normalize_backend(once)
    assert twice == once


def test_mps_alias_accepts_an_agreeing_explicit_sim_method():
    out = normalize_backend(
        {"backend": "mps_simulator", "sim_method": "matrix_product_state"}
    )
    assert out["sim_method"] == "matrix_product_state"


def test_mps_alias_rejects_a_contradicting_sim_method():
    with pytest.raises(ValueError, match="pins sim_method"):
        normalize_backend({"backend": "mps_simulator", "sim_method": "statevector"})


@pytest.mark.parametrize("bad", ["aer", "statevector", "mps", "", "MPS_SIMULATOR"])
def test_normalize_backend_rejects_unknown_names(bad):
    with pytest.raises(ValueError, match="Unknown backend"):
        normalize_backend({"backend": bad})


@pytest.mark.parametrize("bad", [None, 42, ["mps_simulator"]])
def test_normalize_backend_rejects_non_strings(bad):
    with pytest.raises(ValueError, match="must be a string"):
        normalize_backend({"backend": bad})


def test_resolve_backend_alias_writes_both_keys_into_a_dictconfig():
    """Struct mode is on for a hydra config; the resolver has to unlock it."""
    args = OmegaConf.create({"backend": "mps_simulator"})
    OmegaConf.set_struct(args, True)
    _resolve_backend_alias(args, _Log())
    assert args["backend"] == "simulator_aer"
    assert args["sim_method"] == "matrix_product_state"


# ----------------------------------------------------------------------- model lists


def test_split_lists_are_joined_classical_first():
    args = {"classical_model": ["lr", "rf"], "quantum_model": ["qsvc", "pqk"]}
    _resolve_model_lists(args, _Log())
    assert args["model"] == ["lr", "rf", "qsvc", "pqk"]


def test_a_legacy_model_list_alone_is_left_untouched():
    args = {"model": ["lr", "qsvc"]}
    _resolve_model_lists(args, _Log())
    assert args["model"] == ["lr", "qsvc"]


def test_only_one_of_the_two_new_keys_is_enough():
    args = {"classical_model": ["lr"]}
    _resolve_model_lists(args, _Log())
    assert args["model"] == ["lr"]


def test_writing_model_alongside_the_split_keys_is_an_error():
    args = {"model": ["lr"], "classical_model": ["rf"], "quantum_model": []}
    with pytest.raises(ValueError, match="as well as"):
        _resolve_model_lists(args, _Log())


def test_naming_no_model_key_at_all_is_an_error():
    with pytest.raises(ValueError, match="none of 'model'"):
        _resolve_model_lists({}, _Log())


def test_a_quantum_model_filed_under_classical_is_an_error():
    """The mistake that corrupts the comparison without failing anything."""
    args = {"classical_model": ["lr", "qsvc"], "quantum_model": ["pqk"]}
    with pytest.raises(ValueError, match="classical_model names quantum model"):
        _resolve_model_lists(args, _Log())


def test_a_classical_model_filed_under_quantum_is_an_error():
    args = {"classical_model": ["lr"], "quantum_model": ["pqk", "rf"]}
    with pytest.raises(ValueError, match="quantum_model names"):
        _resolve_model_lists(args, _Log())


def test_a_duplicate_across_the_two_lists_is_an_error():
    args = {"classical_model": ["lr", "lr"], "quantum_model": ["qsvc"]}
    with pytest.raises(ValueError, match="Duplicate model"):
        _resolve_model_lists(args, _Log())


def test_two_empty_lists_are_an_error():
    args = {"classical_model": [], "quantum_model": []}
    with pytest.raises(ValueError, match="both empty"):
        _resolve_model_lists(args, _Log())


def test_every_quantum_model_is_accepted_under_quantum_model():
    """Pins the split against QUANTUM_MODELS rather than a hand-copied list."""
    args = {"classical_model": ["lr"], "quantum_model": sorted(QUANTUM_MODELS)}
    _resolve_model_lists(args, _Log())
    assert args["model"] == ["lr"] + sorted(QUANTUM_MODELS)


def test_resolve_model_lists_writes_into_a_struct_dictconfig():
    args = OmegaConf.create({"classical_model": ["lr"], "quantum_model": ["qsvc"]})
    OmegaConf.set_struct(args, True)
    _resolve_model_lists(args, _Log())
    assert list(args["model"]) == ["lr", "qsvc"]
