"""The ``qdata-gen`` console script's parsing and dispatch, in process.

Added after a coverage run named the gap. ``quantum_cli.py`` reported **0%** under
``pytest --cov=qbiocode`` while still being "tested": ``tests/integration/test_cli_smoke.py``
runs every console script through ``subprocess.run([sys.executable, ...])``, so the CLI
really does execute, but in a child process that ``--cov`` does not measure. The effect
was that ``--help`` exiting 0 was the only thing anyone checked, and the part most likely
to rot -- the translation from flag name to generator keyword -- was asserted nowhere.

That translation is not mechanical. Six flags are renamed on the way through
(``--s`` becomes ``n_terms`` for two different families, ``--te-w`` becomes ``disorder``,
``--g`` becomes ``longitudinal_field``, ``--n``/``--N`` become ``n_qubits``/``n_samples``)
and one is transformed (``--blas-threads 0`` means "leave BLAS threading alone", which the
generators spell as ``None``). A typo in any of them would still parse, still dispatch and
still write a file -- with the wrong physics in it.

The generators are replaced with recorders here, so these tests assert the wiring and
cost milliseconds. One test at the end does run the real thing at ``n=4``, because a
recorder cannot show that the wiring produces output.
"""

import sys

import pytest

from conftest import ensure_package, load_module


def _load_cli():
    """Load ``quantum_cli`` under its real dotted name.

    It reaches its five generators through ``from .make_x import ...``, so the module
    must be registered as ``qbiocode.data_generation.quantum_cli`` for those relative
    imports to resolve -- the same constraint ``test_quantum_data_generation.py``
    documents.
    """
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.data_generation", "qbiocode/data_generation")
    return load_module(
        "qbiocode.data_generation.quantum_cli",
        "qbiocode/data_generation/quantum_cli.py",
    )


quantum_cli = _load_cli()

#: The generator each family must reach, by the name it carries in ``quantum_cli``.
DISPATCH = {
    "gs": "generate_ground_state_datasets",
    "te": "generate_time_evolution_datasets",
    "hl": "generate_hamiltonian_learning_datasets",
    "ql": "generate_quantum_label_datasets",
    "eng": "generate_engineered_kernel_datasets",
}


@pytest.fixture
def recorded(monkeypatch):
    """Replace all five generators with recorders and return the call log."""
    calls = {}

    def recorder(name):
        def record(**kwargs):
            calls[name] = kwargs

        return record

    for attribute in DISPATCH.values():
        monkeypatch.setattr(quantum_cli, attribute, recorder(attribute))
    return calls


class TestTheParserAcceptsWhatTheDocsPromise:
    def test_every_family_in_the_module_constant_parses(self):
        """``FAMILIES`` and the ``family`` positional cannot drift apart."""
        parser = quantum_cli.build_parser()
        for family in quantum_cli.FAMILIES:
            assert parser.parse_args([family]).family == family

    def test_an_unknown_family_is_rejected_rather_than_ignored(self):
        with pytest.raises(SystemExit):
            quantum_cli.build_parser().parse_args(["nonesuch"])

    def test_the_five_generating_families_are_exactly_the_dispatch_table(self):
        assert set(quantum_cli.FAMILIES) == set(DISPATCH) | {"selftest"}


class TestEachFamilyReachesItsOwnGenerator:
    @pytest.mark.parametrize("family", sorted(DISPATCH))
    def test_the_family_calls_that_generator_and_no_other(self, family, recorded):
        assert quantum_cli.main([family, "--n", "4", "--N", "16"]) == 0
        assert list(recorded) == [DISPATCH[family]], (
            f"{family} dispatched to {list(recorded)}, expected [{DISPATCH[family]!r}]"
        )

    @pytest.mark.parametrize("family", sorted(DISPATCH))
    def test_the_shared_flags_arrive_under_their_generator_names(self, family, recorded):
        """``--n``/``--N`` are ``n_qubits``/``n_samples`` on the other side."""
        quantum_cli.main(
            [family, "--n", "5", "--N", "17", "--seed", "3", "--margin", "0.25",
             "--out", "somewhere", "--name", "fixed"]
        )
        kwargs = recorded[DISPATCH[family]]
        assert kwargs["n_qubits"] == 5
        assert kwargs["n_samples"] == 17
        assert kwargs["random_state"] == 3
        assert kwargs["margin"] == 0.25
        assert kwargs["save_path"] == "somewhere"
        assert kwargs["name"] == "fixed"


class TestTheRenamedFlags:
    """The six flags whose CLI spelling differs from the keyword they become."""

    def test_s_becomes_n_terms_for_gs(self, recorded):
        quantum_cli.main(["gs", "--n", "4", "--N", "16", "--s", "3"])
        assert recorded["generate_ground_state_datasets"]["n_terms"] == 3

    def test_s_becomes_n_terms_for_te_too(self, recorded):
        quantum_cli.main(["te", "--n", "4", "--N", "16", "--s", "3"])
        assert recorded["generate_time_evolution_datasets"]["n_terms"] == 3

    def test_te_w_becomes_disorder(self, recorded):
        quantum_cli.main(["te", "--n", "4", "--N", "16", "--te_w", "0.3"])
        assert recorded["generate_time_evolution_datasets"]["disorder"] == 0.3

    def test_g_becomes_longitudinal_field(self, recorded):
        quantum_cli.main(["hl", "--n", "4", "--N", "16", "--g", "0.75"])
        assert recorded["generate_hamiltonian_learning_datasets"]["longitudinal_field"] == 0.75

    def test_the_J_and_h_bounds_are_paired_into_ranges(self, recorded):
        """Four scalar flags become two tuples, in the order the generator expects."""
        quantum_cli.main(
            ["gs", "--n", "4", "--N", "16",
             "--J_lo", "0.1", "--J_hi", "0.9", "--h_lo", "0.2", "--h_hi", "0.8"]
        )
        kwargs = recorded["generate_ground_state_datasets"]
        assert kwargs["J_range"] == (0.1, 0.9)
        assert kwargs["h_range"] == (0.2, 0.8)

    def test_data_map_survives_its_dashed_spelling(self, recorded):
        """``--data-map`` reaches the generator as ``data_map``.

        Worth its own test: this is the flag the audit found decides whether the ``eng``
        positive control separates at all.
        """
        quantum_cli.main(["eng", "--n", "4", "--N", "16", "--data-map", "unit"])
        assert recorded["generate_engineered_kernel_datasets"]["data_map"] == "unit"


class TestBothFlagSpellingsWork:
    """Every multi-word flag accepts a dash and an underscore.

    The CLI grew two conventions: six flags inherited from the standalone generator use
    underscores (``--J_lo``, ``--te_w``, ``--gamma_q``) and the two added during the port
    use dashes (``--data-map``, ``--blas-threads``). Nothing was broken -- every spelling
    that appears in the README, the docs page and the notebook is the one its flag
    actually accepts -- but a user who guesses the other convention got
    ``unrecognized arguments`` for a flag that plainly exists in ``--help``. Both
    spellings are now aliases of one destination, which is additive: no documented
    command changes meaning.
    """

    #: ``(flag_a, flag_b, value)`` for every multi-word flag.
    SPELLINGS = [
        ("--J_lo", "--J-lo", "0.3"),
        ("--J_hi", "--J-hi", "0.7"),
        ("--h_lo", "--h-lo", "0.3"),
        ("--h_hi", "--h-hi", "0.7"),
        ("--te_w", "--te-w", "0.3"),
        ("--gamma_q", "--gamma-q", "2.0"),
        ("--data-map", "--data_map", "unit"),
        ("--blas-threads", "--blas_threads", "2"),
    ]

    @pytest.mark.parametrize("underscored,dashed,value", SPELLINGS)
    def test_the_two_spellings_parse_to_the_same_namespace(self, underscored, dashed, value):
        parser = quantum_cli.build_parser()
        assert parser.parse_args(["gs", underscored, value]) == parser.parse_args(
            ["gs", dashed, value]
        )

    @pytest.mark.parametrize("underscored,dashed,value", SPELLINGS)
    def test_help_still_lists_the_flag(self, underscored, dashed, value):
        """An alias must not hide the primary spelling from ``--help``."""
        assert underscored in quantum_cli.build_parser().format_help()


class TestTheBlasThreadSentinel:
    """``--blas-threads 0`` is a sentinel, not a thread count."""

    def test_zero_becomes_none_meaning_do_not_touch_threading(self, recorded):
        quantum_cli.main(["gs", "--n", "4", "--N", "16", "--blas-threads", "0"])
        assert recorded["generate_ground_state_datasets"]["blas_threads"] is None

    def test_a_positive_count_passes_through_unchanged(self, recorded):
        quantum_cli.main(["gs", "--n", "4", "--N", "16", "--blas-threads", "4"])
        assert recorded["generate_ground_state_datasets"]["blas_threads"] == 4

    def test_the_default_pins_one_thread(self, recorded):
        """Small-matrix work thrashes on a many-core box; the default protects it."""
        quantum_cli.main(["gs", "--n", "4", "--N", "16"])
        assert recorded["generate_ground_state_datasets"]["blas_threads"] == 1


class TestSelftestRouting:
    """``selftest`` is the one family whose exit code carries information."""

    def test_a_passing_selftest_exits_zero(self, monkeypatch):
        monkeypatch.setattr(quantum_cli, "run_selftest", lambda: (True, {}))
        assert quantum_cli.main(["selftest"]) == 0

    def test_a_failing_selftest_exits_one(self, monkeypatch):
        """The branch nothing else reaches: a real failure must not exit 0.

        ``qdata-gen selftest`` is what the docs tell a user to run before generating
        anything, and CI's console-script step only checks ``--help``. If this returned
        0 on failure the check would be decorative.
        """
        monkeypatch.setattr(quantum_cli, "run_selftest", lambda: (False, {}))
        assert quantum_cli.main(["selftest"]) == 1

    def test_selftest_ignores_the_generation_flags_rather_than_erroring(self, monkeypatch):
        monkeypatch.setattr(quantum_cli, "run_selftest", lambda: (True, {}))
        assert quantum_cli.main(["selftest", "--n", "4", "--out", "unused"]) == 0


class TestTheWiringActuallyProducesOutput:
    """One real run, because a recorder cannot show that the CLI writes anything."""

    def test_a_real_invocation_writes_the_four_artefacts(self, tmp_path):
        out = tmp_path / "qdata"
        assert quantum_cli.main(
            ["gs", "--n", "4", "--N", "16", "--seed", "0",
             "--name", "cli_smoke", "--out", str(out)]
        ) == 0
        assert (out / "x_view" / "cli_smoke.csv").is_file()
        assert (out / "meta" / "cli_smoke.json").is_file()
        assert (out / "meta" / "cli_smoke_F.npy").is_file()

    def test_argv_defaults_to_sys_argv_when_none_is_passed(self, monkeypatch, recorded):
        """``main()`` with no argument must read ``sys.argv``, as ``main`` promises."""
        monkeypatch.setattr(sys, "argv", ["qdata-gen", "gs", "--n", "4", "--N", "16"])
        assert quantum_cli.main() == 0
        assert "generate_ground_state_datasets" in recorded
