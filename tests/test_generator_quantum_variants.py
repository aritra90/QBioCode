"""generate_pilot_configs.py: qpl, --quantum-space, --measurement, --qpl-heads, noclassical.

Each test pins a failure that occurred while building these options, not a line of code:

* A fixed measurement given only as a one-value search list was NOT fixed: the tuner
  enqueues the default config as trial 0 and, finding the default outside the choices,
  adds it back -- so an "rdm2" run scored rdm1 trials and on a short budget reported
  rdm1 as the winner. It must also be the arm's default.
* Nothing configured must mean nothing written, so existing configs (and the embedding
  caches keyed on them) stay byte-identical.
* The new flags must be refused outside manifest mode rather than silently ignored.
"""

import importlib.util
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = ROOT / "experiments" / "pilot10" / "generate_pilot_configs.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gpc_qv", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(block):
    return yaml.safe_load(block) or {}


class TestNothingConfiguredWritesNothing:
    def test_no_options_and_no_qpl_is_empty(self, gen):
        assert gen.quantum_variant_block(["pqk"]) == ""
        assert gen.quantum_variant_block(["qsvc"]) == ""
        assert gen.quantum_variant_block([]) == ""

    def test_a_non_projection_arm_gets_nothing_whatever_the_options(self, gen):
        assert gen.quantum_variant_block(["qsvc"], quantum_space="wide",
                                         measurement="rdm2") == ""


class TestFixedMeasurementIsTheDefaultToo:
    @pytest.mark.parametrize("arm", ["pqk", "qpl"])
    def test_it_is_in_both_the_search_and_the_arm_default(self, gen, arm):
        """The regression this suite exists for: trial 0 reads <arm>_args, so a fixed
        measurement absent from it is re-added to the search as the default."""
        y = _parse(gen.quantum_variant_block([arm], measurement="rdm2"))
        assert y[f"gridsearch_{arm}_args"]["measurement"] == ["rdm2"]
        assert y[f"{arm}_args"]["measurement"] == "rdm2"

    def test_rdm2_selects_the_statevector_projector(self, gen):
        y = _parse(gen.quantum_variant_block(["qpl"], measurement="rdm2_nn"))
        assert y["projection_backend"] == "statevector"

    def test_rdm1_leaves_the_projector_alone(self, gen):
        y = _parse(gen.quantum_variant_block(["qpl"], measurement="rdm1"))
        assert "projection_backend" not in y


class TestWideSpace:
    def test_the_wide_space_is_written_for_each_projection_arm(self, gen):
        y = _parse(gen.quantum_variant_block(["pqk", "qpl"], quantum_space="wide"))
        for arm in ("pqk", "qpl"):
            assert y[f"gridsearch_{arm}_args"] == gen.WIDE_QUANTUM_SPACE

    def test_bandwidth_never_passes_pi_and_keeps_the_legacy_value(self, gen):
        """Past pi the feature-map angles wrap, so a wider range only adds collisions;
        1.0 is what every earlier run used and must stay searchable."""
        bw = gen.WIDE_QUANTUM_SPACE["bandwidth"]
        assert max(bw) <= 3.1416 and 1.0 in bw

    def test_size(self, gen):
        size = 1
        for values in gen.WIDE_QUANTUM_SPACE.values():
            size *= len(values)
        assert size == 480


class TestQplHeads:
    def test_default_heads(self, gen):
        y = _parse(gen.quantum_variant_block(["qpl"]))
        assert y["qpl_args"]["classical_models"] == ["catboost", "tabpfn"]

    def test_custom_heads(self, gen):
        y = _parse(gen.quantum_variant_block(["qpl"], qpl_heads=["xgb"]))
        assert y["qpl_args"]["classical_models"] == ["xgb"]


class TestModelGroups:
    def test_noclassical_drops_the_classical_job(self, gen):
        assert [g for g, _, _ in gen.model_groups(["pqk", "qpl", "noclassical"])] \
            == ["pqk", "qpl"]

    def test_named_classical_arms_form_a_matched_baseline(self, gen):
        groups = {g: c for g, c, _ in gen.model_groups(["qpl", "catboost", "tabpfn"])}
        assert groups["classical"] == ["catboost", "tabpfn"]

    def test_noclassical_with_classical_arms_is_a_contradiction(self, gen):
        with pytest.raises(ValueError, match="contradicts"):
            gen.model_groups(["qpl", "catboost", "noclassical"])

    def test_without_noclassical_the_group_is_kept_whole(self, gen):
        """Unchanged behaviour: a run naming only quantum arms still gets all nine."""
        groups = {g: c for g, c, _ in gen.model_groups(["pqk"])}
        assert len(groups["classical"]) == 9


class TestRefusedOutsideManifestMode:
    @pytest.mark.parametrize("flag", [["--quantum-space", "wide"],
                                      ["--measurement", "rdm2"],
                                      ["--qpl-heads", "catboost"]])
    def test_flag(self, flag, tmp_path):
        import subprocess
        import sys
        out = subprocess.run([sys.executable, str(GEN), *flag, "--config-dir",
                              str(tmp_path)], capture_output=True, text=True)
        assert out.returncode != 0
        assert "only apply with --split-mode manifest" in out.stderr


class TestCostModelPricesQpl:
    """--wall auto priced jobs with a cost model that had no qpl entry and raised
    KeyError: "no calibration for arm 'qpl'" on the first qpl job -- missed locally
    because the end-to-end run used a fixed --wall."""

    @pytest.fixture(scope="class")
    def cm(self):
        spec = importlib.util.spec_from_file_location(
            "cm_qv", ROOT / "experiments" / "pilot10" / "cost_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_qpl_has_a_price_above_pqk(self, cm):
        pqk = cm.manifest_job_hours("pqk", "sv", 8, 250, n_splits=15)
        qpl = cm.manifest_job_hours("qpl", "sv", 8, 250, n_splits=15)
        assert qpl > pqk > 0

    def test_qpl_is_always_reported_as_extrapolated(self, cm):
        """No qpl job has been timed, so its wall must not look calibrated."""
        assert cm.manifest_extrapolated("sv", 8, 150, group="qpl")
        assert not cm.manifest_extrapolated("sv", 8, 150, group="pqk")

    def test_wide_space_and_rdm2_raise_the_projection_price(self, cm):
        base = cm.manifest_job_hours("pqk", "sv", 8, 250)
        wide = cm.manifest_job_hours("pqk", "sv", 8, 250, quantum_space="wide")
        rdm2 = cm.manifest_job_hours("pqk", "sv", 8, 250, quantum_space="wide",
                                     measurement="rdm2")
        assert base < wide < rdm2

    def test_the_factor_leaves_other_arms_alone(self, cm):
        for group in ("qsvc", "classical"):
            assert (cm.manifest_job_hours(group, "sv", 8, 250)
                    == cm.manifest_job_hours(group, "sv", 8, 250, quantum_space="wide",
                                             measurement="rdm2"))
