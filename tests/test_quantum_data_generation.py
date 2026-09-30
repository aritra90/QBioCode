"""
Tests for the simulated quantum data generators.

Two layers, and they fail for different reasons.

The :class:`TestThePhysicsIsRight` layer runs the seven checks of
:mod:`qbiocode.data_generation.quantum_selftest`, each of which compares an
implementation against an independent computation of the same quantity: the
hand-written Pauli action against a dense Kronecker product, the sparse Hamiltonian
against a dense one, the ground state against the residual of its own eigenvalue
equation, the fast Walsh-Hadamard transform against its inverse and Parseval's
identity, the quench dynamics against the analytic short-time expansion, the
engineered labels against the bound they are built to saturate, and the native
``ZZFeatureMap`` against Qiskit's. A failure here means a *number* is wrong, which
would silently corrupt every dataset the family writes.

The :class:`TestTheOutputContract` layer checks what QProfiler needs to be true of
the files: a binary label in the last column, the documented feature count, matching
rows across the two views, and the metadata that records how the label was made. A
failure here means the files are unreadable or mislabelled, not that the physics is
wrong.

Sizes are kept at ``n_qubits=4`` and 16 rows throughout. Exact simulation costs
``2 ** n``, and none of these assertions get truer at n=8.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from conftest import ensure_package, load_module


def _load_quantum_modules():
    """Load the generators by path, as the rest of the suite does.

    The suite has to run against a source checkout with no installed ``qbiocode``,
    so nothing here may be a plain ``import qbiocode...``. The five family modules
    reach ``quantum_core`` through a relative import, which fixes the names they
    must be registered under: loading them as ``tests._make_ground_state`` would
    make ``from .quantum_core import ...`` resolve against ``tests``.
    """
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.data_generation", "qbiocode/data_generation")
    names = [
        "quantum_core",
        "quantum_selftest",
        "make_ground_state",
        "make_time_evolution",
        "make_hamiltonian_learning",
        "make_quantum_labels",
        "make_engineered_kernel",
    ]
    return {
        name: load_module(
            f"qbiocode.data_generation.{name}",
            f"qbiocode/data_generation/{name}.py",
        )
        for name in names
    }


_MODULES = _load_quantum_modules()
quantum_core = _MODULES["quantum_core"]
quantum_selftest = _MODULES["quantum_selftest"]
ground_state = _MODULES["make_ground_state"]
time_evolution = _MODULES["make_time_evolution"]
hamiltonian_learning = _MODULES["make_hamiltonian_learning"]
quantum_labels = _MODULES["make_quantum_labels"]
engineered_kernel = _MODULES["make_engineered_kernel"]

#: One small configuration per family, with the feature count its docstring promises.
#: ``n`` is 4 and the row count 16 everywhere, so the whole module stays seconds-fast.
FAMILIES = {
    "gs": (
        ground_state.generate_ground_state_datasets,
        {"n_qubits": 4, "n_samples": 16, "label": "sparse", "kappa": 0.5},
        7,        # 2n - 1: n-1 couplings + n fields
        True,     # writes phi_view
    ),
    "te": (
        time_evolution.generate_time_evolution_datasets,
        {"n_qubits": 4, "n_samples": 16, "taus": [0.5]},
        4,        # n: one column per input bit
        True,
    ),
    "hl": (
        hamiltonian_learning.generate_hamiltonian_learning_datasets,
        {"n_qubits": 4, "n_samples": 16, "times": [0.5], "shots": 0},
        8,        # 2n * len(times)
        False,
    ),
    "ql": (
        quantum_labels.generate_quantum_label_datasets,
        {"n_qubits": 4, "n_samples": 16, "encoding": "zz", "tau": 1.0},
        4,        # n
        False,
    ),
    "eng": (
        engineered_kernel.generate_engineered_kernel_datasets,
        {"n_qubits": 4, "n_samples": 16, "gamma_q": 1.0},
        4,        # n
        False,
    ),
}


def generate(family, tmp_path, **overrides):
    """Run one family into ``tmp_path`` and return ``(metadata list, save_path)``."""
    function, kwargs, _, _ = FAMILIES[family]
    return function(save_path=str(tmp_path), **{**kwargs, **overrides}), tmp_path


def read_view(save_path, view, name):
    """Read one written CSV the way QProfiler reads it -- no index column."""
    return pd.read_csv(save_path / view / f"{name}.csv")


def _names(save_path):
    """Every dataset name written under ``save_path``, in sorted order."""
    return sorted(path.stem for path in (save_path / "x_view").glob("*.csv"))


def _only_name(save_path):
    """The single dataset name written under ``save_path``."""
    names = _names(save_path)
    assert len(names) == 1, f"expected one dataset, found {names}"
    return names[0]


class TestThePhysicsIsRight:
    """Each implementation against an independent computation of the same quantity.

    These are the regression guard on the numerics. They are also exactly what
    ``qdata-gen selftest`` runs, called through the same functions rather than
    reimplemented here, so the CLI's report and the suite's verdict cannot disagree.
    """

    def test_the_pauli_action_matches_a_dense_kronecker_product(self):
        measured = quantum_selftest.check_pauli_action()
        assert measured["n_strings"] == 4 ** 4        # every Pauli string on 4 qubits

    def test_the_sparse_hamiltonian_matches_a_dense_one(self):
        quantum_selftest.check_sparse_pauli_sum()

    def test_the_even_sector_ground_state_solves_its_own_eigenproblem(self):
        """Residual, parity and global minimality: a wrong state fails one of the three.

        The parity check is the one specific to this construction. The penalty term
        that lifts the odd sector is what removes the finite-size near-degeneracy
        between the two symmetry sectors, and if it were mis-signed the solver would
        return an odd-sector state that still has a small residual.
        """
        measured = quantum_selftest.check_ground_state()
        assert measured["parity"] == pytest.approx(1.0, abs=1e-9)

    def test_the_walsh_transform_inverts_and_preserves_energy(self):
        quantum_selftest.check_walsh_transform()

    def test_the_quench_dynamics_match_the_analytic_short_time_expansion(self):
        """``<Z_i(t)> = 1 - 2 h_i^2 t^2 + O(t^4)``, the one closed form available here."""
        quantum_selftest.check_short_time_limit()

    def test_the_engineered_labels_saturate_the_bound_they_target(self):
        """``s_Q = 1`` and ``s_C = g^2`` -- the construction's whole claim, as numbers."""
        quantum_selftest.check_engineered_labels()

    def test_the_native_zz_feature_map_matches_qiskit(self):
        """The encoder is reimplemented in NumPy for speed; it must be the same circuit.

        This is the check with the widest blast radius. The ``ql`` labels are defined
        by this state, so a divergence from Qiskit would not corrupt the data -- it
        would make the *label rule* differ from the encoder QProfiler then uses to
        learn it, and the family would silently measure nothing.
        """
        measured = quantum_selftest.check_zz_feature_map()
        assert measured["n_comparisons"] >= 3 * len(quantum_selftest.ZZ_CROSSCHECK_CASES)

    def test_run_selftest_reports_every_check(self):
        """The aggregate the CLI prints, so a check cannot be silently dropped from it."""
        ok, results = quantum_selftest.run_selftest(verbose=False)
        assert ok, results
        assert set(results) == {label for label, _ in quantum_selftest.CHECKS}
        assert len(quantum_selftest.CHECKS) == 7


class TestTheAdjointIsAConjugateTranspose:
    """D14: every propagator must survive a complex Hamiltonian.

    All five time-evolution operators are built the same way -- diagonalise, scale the
    eigenvalues, reassemble: ``U = (V * exp(-i E t)) @ V.conj().T``. Four of them used to
    spell that adjoint ``V.T``, which is *bitwise identical* as long as ``H`` is real, and
    ``pauli_sum`` makes it real on purpose: it downcasts whenever ``abs(H.imag).max() <
    1e-14``, which holds for every Hamiltonian the package builds (the Heisenberg term
    included -- ``Y (x) Y`` is real, the two factors of ``i`` cancel).

    So there was nothing to see at runtime, and that is the problem this class exists to
    prevent. The invariant is conditional on the terms: one genuinely complex term -- a
    single-site ``Y`` field, a flux phase, a DMI coupling -- and ``V`` comes back complex,
    at which point ``V.T`` computes a *different operator*. Not an invalid one. It is still
    exactly unitary, so a unitarity check, a norm check and a "is this still a quantum
    evolution" assertion all pass while the dynamics are wrong by O(1).

    These tests therefore compare against ``scipy.linalg.expm``, computed independently of
    the eigendecomposition, which is the only check that distinguishes the two spellings.
    """

    @staticmethod
    def _ising(n, rng):
        return quantum_core.ising_terms(
            n, rng.uniform(0.9, 1.1, n - 1), rng.uniform(0.9, 1.1, n),
            g=np.full(n, 0.5), sign=1.0,
        )

    def test_pauli_sum_keeps_a_complex_hamiltonian_complex(self):
        """The downcast is conditional, so the hazard is reachable, not hypothetical."""
        rng = np.random.default_rng(0)
        n = 4
        real_H = quantum_core.pauli_sum(n, self._ising(n, rng)).toarray()
        complex_H = quantum_core.pauli_sum(
            n, self._ising(n, rng) + [(0.3, quantum_core.Pauli(n, {0: "Y"}))]
        ).toarray()
        assert not np.iscomplexobj(real_H), "every Hamiltonian the package builds is real"
        assert np.iscomplexobj(complex_H), "one Y term is enough to defeat the downcast"

    @pytest.mark.parametrize("add_a_complex_term", [False, True])
    def test_the_reassembled_propagator_matches_expm(self, add_a_complex_term):
        """``(V * exp(-iEt)) @ V.conj().T`` is ``expm(-iHt)`` for real *and* complex H."""
        rng = np.random.default_rng(0)
        n, tau = 4, 0.7
        terms = self._ising(n, rng)
        if add_a_complex_term:
            terms = terms + [(0.3, quantum_core.Pauli(n, {0: "Y"}))]
        H = quantum_core.pauli_sum(n, terms).toarray()
        E, V = np.linalg.eigh(H)
        assert np.abs(expm(-1j * H * tau) - (V * np.exp(-1j * E * tau)) @ V.conj().T).max() < 1e-12

    def test_a_plain_transpose_would_be_wrong_and_still_unitary(self):
        """Why no cheaper guard would have caught this.

        Pinning the failure mode: with complex ``V`` the wrong spelling is off by O(1)
        from the propagator while remaining unitary to machine precision. If this ever
        stops being true the comment above is stale and should be rewritten.
        """
        rng = np.random.default_rng(0)
        n, tau = 4, 0.7
        H = quantum_core.pauli_sum(
            n, self._ising(n, rng) + [(0.3, quantum_core.Pauli(n, {0: "Y"}))]
        ).toarray()
        E, V = np.linalg.eigh(H)
        wrong = (V * np.exp(-1j * E * tau)) @ V.T
        identity = np.eye(1 << n)
        assert np.abs(wrong - expm(-1j * H * tau)).max() > 1e-3     # wrong operator
        assert np.abs(wrong.conj().T @ wrong - identity).max() < 1e-10   # yet still unitary


class TestTheOutputContract:
    """What QProfiler needs to be true of the written files."""

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_it_writes_the_documented_layout(self, family, tmp_path):
        """One CSV per view, plus the metadata and the continuous target beside them."""
        _, _, n_features, has_phi = FAMILIES[family]
        metas, save_path = generate(family, tmp_path)

        assert len(metas) == 1, f"{family} reported {len(metas)} datasets for one configuration"
        names = _names(save_path)
        assert names, "no x_view CSV was written"
        for name in names:
            assert (save_path / "meta" / f"{name}.json").exists()
            assert (save_path / "meta" / f"{name}_F.npy").exists()
            # phi_view is written only by the families that have a second view; the
            # directory itself always exists, so this has to test the file.
            assert (save_path / "phi_view" / f"{name}.csv").exists() is has_phi

        frame = read_view(save_path, "x_view", names[0])
        assert frame.shape[1] == n_features + 1, (
            f"{family} wrote {frame.shape[1] - 1} features, expected {n_features}"
        )

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_the_label_is_the_last_column_and_binary(self, family, tmp_path):
        """QProfiler splits positionally (``iloc[:, :-1]`` / ``iloc[:, -1:]``).

        So the label's *position* is the contract and its name is not, and a label
        that is not exactly two-valued fails QProfiler's binary-only check later,
        at model-fitting time, where the cause is much harder to see.
        """
        _, save_path = generate(family, tmp_path)

        for view in ("x_view", "phi_view"):
            for path in sorted((save_path / view).glob("*.csv")):
                frame = pd.read_csv(path)
                assert frame.columns[-1] == "label", list(frame.columns)
                labels = frame.iloc[:, -1]
                assert set(np.unique(labels)) == {0, 1}, f"{path.name}: {np.unique(labels)}"
                features = frame.iloc[:, :-1].to_numpy()
                # 'i' as well as 'f': te's inputs are computational-basis bitstrings,
                # so its x_view is genuinely integral and round-trips as int64.
                assert features.dtype.kind in "fi", features.dtype
                assert np.isfinite(features).all()

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_the_label_is_a_median_split_so_the_classes_are_balanced(self, family, tmp_path):
        """A median threshold makes balance a *property*, not luck -- so it can be asserted.

        Balance is what makes accuracy a meaningful score on these datasets, and the
        tolerance is one row out of 16 rather than a loose band, because anything
        looser would also pass a threshold that had drifted off the median.
        """
        metas, save_path = generate(family, tmp_path)

        for meta in metas:
            assert meta["class_balance"] == pytest.approx(0.5, abs=1.5 / meta["n_rows"])
        for path in sorted((save_path / "x_view").glob("*.csv")):
            labels = pd.read_csv(path).iloc[:, -1].to_numpy()
            assert labels.mean() == pytest.approx(0.5, abs=1.5 / len(labels))

    #: Keys every family records, and then the ones that identify its label rule.
    #: ``eng`` has no ``label_rule`` string because its target is not an observable:
    #: it is the vector saturating the geometric-difference bound, and what pins it
    #: down is the kernel bandwidths and the ridge, which are recorded instead.
    REQUIRED_METADATA = {
        "gs": ("label_rule",),
        "te": ("label_rule", "tau", "H"),
        "hl": ("label_rule", "times", "shots"),
        "ql": ("label_rule", "encoding", "reps", "entanglement"),
        "eng": ("gamma_q", "gamma_c_adversary", "lam", "reps", "entanglement"),
    }

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_the_metadata_records_how_the_label_was_made(self, family, tmp_path):
        """Without the rule, a dataset is an unfalsifiable claim about its own difficulty.

        Checked both in the returned metadata and in the JSON on disk, because they are
        written by two different steps and only the file survives the process.
        """
        metas, save_path = generate(family, tmp_path)
        universal = ("family", "n", "threshold", "margin", "seed", "n_rows", "class_balance")

        for meta in metas:
            on_disk = json.loads(
                (save_path / "meta" / f"{_only_name(save_path)}.json").read_text(encoding="utf-8")
            )
            for key in universal + self.REQUIRED_METADATA[family]:
                assert key in meta, f"{family} metadata is missing {key!r}"
                assert key in on_disk, f"{family} metadata on disk is missing {key!r}"
            assert meta["n"] == 4
            assert meta["family"] == family

    @pytest.mark.parametrize("family", ["gs", "te"])
    def test_the_two_views_describe_the_same_rows(self, family, tmp_path):
        """A paired comparison between the views is the point; unpaired rows void it."""
        _, save_path = generate(family, tmp_path)

        for path in sorted((save_path / "x_view").glob("*.csv")):
            x_view = pd.read_csv(path)
            phi_view = read_view(save_path, "phi_view", path.stem)
            assert len(x_view) == len(phi_view)
            assert x_view["label"].equals(phi_view["label"])
            assert phi_view.shape[1] > x_view.shape[1]      # more measurements than knobs

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_the_continuous_target_agrees_with_the_written_label(self, family, tmp_path):
        """``_F.npy`` is the pre-threshold target; ``label`` must be its own threshold."""
        metas, save_path = generate(family, tmp_path)

        for meta in metas:
            name = _only_name(save_path)
            F = np.load(save_path / "meta" / f"{name}_F.npy")
            labels = read_view(save_path, "x_view", name).iloc[:, -1].to_numpy()
            assert len(F) == len(labels)
            assert np.array_equal((F > meta["threshold"]).astype(int), labels)

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_the_same_seed_reproduces_the_same_dataset(self, family, tmp_path):
        """Every generator seeds a fresh ``default_rng`` per configuration, so this holds
        for a single call and for one configuration drawn out of a wider sweep alike.

        Compared by *content*, not by filename: the seed appears in the name, so a
        name-keyed comparison would find nothing to compare between two seeds and pass
        for the wrong reason.
        """
        _, first = generate(family, tmp_path / "first", random_state=3)
        _, second = generate(family, tmp_path / "second", random_state=3)
        _, other = generate(family, tmp_path / "other", random_state=4)

        assert _names(first) == _names(second)
        for name in _names(first):
            # f-string, not Path.with_suffix: the names carry dots of their own
            # ("k0.5", "tau0.5"), and with_suffix would cut the name at the last one.
            assert (first / "x_view" / f"{name}.csv").read_bytes() == (
                (second / "x_view" / f"{name}.csv").read_bytes()
            )
        # Compared on the continuous target rather than on the features, because the
        # features are not what every seed drives: te at n_samples == 2**n enumerates
        # the whole Boolean cube, so its inputs are the same for every seed by
        # construction and only the Hamiltonian, the observable and hence F move.
        targets = [
            np.load(path / "meta" / f"{_only_name(path)}_F.npy") for path in (first, other)
        ]
        assert not np.array_equal(*targets), (
            f"{family}: a different seed produced an identical target, so random_state "
            f"is ignored"
        )

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_a_margin_drops_the_rows_nearest_the_boundary(self, family, tmp_path):
        """The ambiguous band is what a near-chance score comes from, so it is removable.

        The margin is derived from each family's own target spread rather than fixed,
        because the five targets are not on a common scale: at n=4 the whole ``gs``
        target spans about 0.13 while ``te``'s interquartile range alone is 0.9. A
        single constant is therefore either a no-op for one family or removes every
        row of another -- and removing every row is now an error, so it would fail
        here for the wrong reason.
        """
        _, plain_path = generate(family, tmp_path / "plain")
        plain_F = np.load(plain_path / "meta" / f"{_only_name(plain_path)}_F.npy")
        margin = 0.25 * float(np.subtract(*np.percentile(plain_F, [75, 25])))
        wide, wide_path = generate(family, tmp_path / "margin", margin=margin)

        plain_rows = [len(pd.read_csv(p)) for p in sorted((plain_path / "x_view").glob("*.csv"))]
        margin_rows = [len(pd.read_csv(p)) for p in sorted((wide_path / "x_view").glob("*.csv"))]  # noqa: E501
        assert sum(margin_rows) < sum(plain_rows), (
            f"{family}: margin={margin:g} dropped no rows ({margin_rows} vs {plain_rows})"
        )
        for meta in wide:
            F = np.load(wide_path / "meta" / f"{_only_name(wide_path)}_F.npy")
            assert (np.abs(F - meta["threshold"]) >= margin).all()

    @pytest.mark.parametrize("family", ["gs", "te"])
    def test_shots_perturb_the_features_but_not_the_labels(self, family, tmp_path):
        """Shot noise is estimation noise on the *features*.

        The label is always computed from the exact expectation values, so a finite
        ``shots`` degrades what the learner sees without moving the target it is
        learning -- which is the only way the knob measures anything. If it moved the
        labels too, a drop in accuracy could not be attributed.
        """
        _, exact = generate(family, tmp_path / "exact", shots=0)
        _, noisy = generate(family, tmp_path / "noisy", shots=64)

        for path in sorted((exact / "phi_view").glob("*.csv")):
            exact_view = pd.read_csv(path)
            noisy_view = read_view(noisy, "phi_view", path.stem)
            assert exact_view["label"].equals(noisy_view["label"])
            assert not np.allclose(
                exact_view.iloc[:, :-1].to_numpy(), noisy_view.iloc[:, :-1].to_numpy()
            )
            # Binomial estimation noise on a +-1-valued Pauli: the estimate stays in range.
            assert np.abs(noisy_view.iloc[:, :-1].to_numpy()).max() <= 1.0 + 1e-12
        for path in sorted((exact / "x_view").glob("*.csv")):
            assert pd.read_csv(path).equals(read_view(noisy, "x_view", path.stem))


class TestASweepCannotSilentlyOverwriteItself:
    """The dataset names encode physics, not every knob -- so collisions are possible.

    No family's name encodes the row count, and ``ql``'s encodes neither ``reps`` nor
    ``entanglement``. A sweep over one of those would write several datasets to one
    path, report all of them as generated, and leave only the last on disk. The
    generators refuse before computing anything instead.
    """

    def test_sweeping_a_knob_absent_from_the_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="overwrite"):
            ground_state.generate_ground_state_datasets(
                n_qubits=4, n_samples=[8, 16], label="sparse", save_path=str(tmp_path)
            )

    def test_the_refusal_names_the_knob_that_is_not_in_the_name(self, tmp_path):
        with pytest.raises(ValueError, match="n_samples"):
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=[8, 16], save_path=str(tmp_path)
            )

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_an_explicit_name_is_refused_for_a_multi_configuration_sweep(self, family, tmp_path):
        """``name`` overrides the autoname, so it can only mean one dataset."""
        function, kwargs, _, _ = FAMILIES[family]
        with pytest.raises(ValueError, match="name="):
            function(
                save_path=str(tmp_path),
                name="collide",
                **{**kwargs, "random_state": [0, 1]},
            )

    def test_a_single_configuration_accepts_an_explicit_name(self, tmp_path):
        ground_state.generate_ground_state_datasets(
            n_qubits=4, n_samples=16, label="sparse", kappa=0.5,
            save_path=str(tmp_path), name="chosen",
        )
        assert (tmp_path / "x_view" / "chosen.csv").exists()
        assert (tmp_path / "meta" / "chosen.json").exists()

    def test_a_sweep_writes_one_dataset_per_configuration(self, tmp_path):
        metas = ground_state.generate_ground_state_datasets(
            n_qubits=4, n_samples=16, label=["sparse", "e2e"], kappa=0.5,
            save_path=str(tmp_path), random_state=[0, 1],
        )
        written = _names(tmp_path)
        assert len(metas) == 4
        assert len(written) == 4, written


class TestTheKnobsThatChangeTheConcept:
    """Parameters a silent fallback would turn into a different learning problem."""

    def test_an_unknown_encoding_is_refused_rather_than_defaulted(self, tmp_path):
        """The ``ql`` label is *defined* by the encoder, so a fallback changes the task."""
        with pytest.raises(ValueError, match="encoding"):
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=16, encoding="not-an-encoding", save_path=str(tmp_path)
            )

    @pytest.mark.parametrize("encoding", sorted(quantum_labels.ENCODINGS))
    def test_every_advertised_encoding_runs(self, encoding, tmp_path):
        metas = quantum_labels.generate_quantum_label_datasets(
            n_qubits=4, n_samples=16, encoding=encoding, tau=1.0, save_path=str(tmp_path)
        )
        assert len(metas) == 1
        assert encoding in _only_name(tmp_path)

    def test_more_rows_than_inputs_is_refused_for_time_evolution(self, tmp_path):
        """``te`` samples the Boolean cube without replacement, so N <= 2**n is a hard cap.

        Left to NumPy this surfaces as ``ValueError: Cannot take a larger sample than
        population``, from inside ``rng.choice``, which does not say which knob to change.
        """
        with pytest.raises(ValueError, match="without replacement"):
            time_evolution.generate_time_evolution_datasets(
                n_qubits=4, n_samples=32, taus=[0.5], save_path=str(tmp_path)
            )

    def test_the_tau_ladder_shares_one_hamiltonian_and_one_input_set(self, tmp_path):
        """Only the evolution time may differ between the rungs, or the ladder is not one."""
        metas = time_evolution.generate_time_evolution_datasets(
            n_qubits=4, n_samples=16, taus=[0.25, 1.0], save_path=str(tmp_path)
        )
        assert len(metas) == 2
        assert metas[0]["H"] == metas[1]["H"]
        assert metas[0]["label_rule"] == metas[1]["label_rule"]
        assert metas[0]["tau"] != metas[1]["tau"]

        names = _names(tmp_path)
        assert names == ["te_n4_s4_seed0_tau0.25", "te_n4_s4_seed0_tau1"], names
        first, second = (read_view(tmp_path, "x_view", name) for name in names)
        assert first.iloc[:, :-1].equals(second.iloc[:, :-1])   # same inputs, different labels

    def test_the_hamiltonian_learning_features_are_one_block_per_time(self, tmp_path):
        """Several times are one measurement record, not several datasets."""
        metas = hamiltonian_learning.generate_hamiltonian_learning_datasets(
            n_qubits=4, n_samples=16, times=[0.25, 0.5, 1.0], shots=0, save_path=str(tmp_path)
        )
        assert len(metas) == 1
        assert read_view(tmp_path, "x_view", _only_name(tmp_path)).shape[1] - 1 == 2 * 4 * 3

    def test_the_engineered_kernel_records_its_geometric_difference(self, tmp_path):
        """``g`` is the construction's own diagnostic, and it comes with a caveat.

        It is computed for the continuous target, and the label on disk is that
        target's median binarisation. The caveat travels in the metadata rather than
        only in the docs, because that is where someone reading the number will be.
        """
        metas = engineered_kernel.generate_engineered_kernel_datasets(
            n_qubits=4, n_samples=16, gamma_q=1.0, save_path=str(tmp_path)
        )
        diagnostics = metas[0]["diagnostics"]
        assert diagnostics["g_continuous"] > 0
        assert diagnostics["g2_continuous"] == pytest.approx(diagnostics["g_continuous"] ** 2)
        assert "CONTINUOUS" in diagnostics["note"]
        assert "must be measured" in diagnostics["note"]


class TestTheGuardsOnDegenerateInput:
    """Each of these wrote something misleading before it raised or returned.

    They are grouped because they share one failure mode: a knob that is wrong in a
    way NumPy is happy with. A margin that removes every row, a qubit count of one,
    an entanglement pattern the chosen encoding never consults -- none of them is a
    Python error, so each used to produce a plausible-looking artifact instead of a
    complaint. The metadata is what makes that dangerous: it records the knob as
    though it had been honoured.
    """

    def test_a_margin_wider_than_the_target_is_refused(self, tmp_path):
        """It used to write a 0-row CSV whose recorded class balance was ``nan``."""
        with pytest.raises(ValueError, match="removed all"):
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=16, encoding="zz", tau=1.0, margin=10.0,
                save_path=str(tmp_path),
            )

    def test_the_refusal_says_what_the_target_actually_spans(self, tmp_path):
        """The fix is either a smaller margin or more rows, and which one depends on it."""
        with pytest.raises(ValueError) as excinfo:
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=16, encoding="zz", tau=1.0, margin=10.0,
                save_path=str(tmp_path),
            )
        message = str(excinfo.value)
        assert "targets span" in message
        assert "margin=10" in message

    def test_a_single_class_survivor_set_is_refused(self, tmp_path):
        """One row cannot be a median split, and a single-class CSV trains nothing.

        QProfiler's loader asserts binary labels, so this used to surface several
        steps downstream as a failure of the dataset rather than of the request.
        """
        with pytest.raises(ValueError, match="all carry label"):
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=1, encoding="zz", tau=1.0, save_path=str(tmp_path)
            )

    def test_an_unknown_data_map_is_refused_by_both_kernel_aligned_families(self, tmp_path):
        """A typo here silently produces a *negative* control that looks like a bug.

        The data map decides which QProfiler arm the labels are aligned with, so a
        fallback would not merely mislabel the metadata -- it would hand back a dataset
        on which the intended arm scores below chance.
        """
        with pytest.raises(ValueError, match="data_map"):
            quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=16, encoding="zz", tau=1.0,
                data_map="not-a-map", save_path=str(tmp_path),
            )
        with pytest.raises(ValueError, match="data_map"):
            engineered_kernel.generate_engineered_kernel_datasets(
                n_qubits=4, n_samples=16, data_map="not-a-map", save_path=str(tmp_path)
            )

    def test_the_default_data_map_is_qiskits_so_old_datasets_still_reproduce(self, tmp_path):
        """Adding the option must not have moved any previously generated dataset.

        The default is the stock ``ZZFeatureMap`` map, and the name gains no suffix, so a
        runsheet written before the option existed reproduces bit-for-bit.
        """
        explicit = tmp_path / "explicit"
        default = tmp_path / "default"
        common = dict(n_qubits=4, n_samples=16, encoding="zz", tau=1.0, random_state=0)
        quantum_labels.generate_quantum_label_datasets(save_path=str(default), **common)
        quantum_labels.generate_quantum_label_datasets(
            save_path=str(explicit), data_map="qiskit", **common
        )
        name = _only_name(default)
        assert name == _only_name(explicit)
        assert "dm" not in name
        assert ((default / "x_view" / f"{name}.csv").read_bytes()
                == (explicit / "x_view" / f"{name}.csv").read_bytes())

    @pytest.mark.parametrize("data_map", sorted(quantum_core.DATA_MAPS))
    def test_each_data_map_is_recorded_and_only_the_non_default_renames(
        self, data_map, tmp_path
    ):
        """The metadata has to say which arm a dataset is aligned with.

        Without it there is no way to tell a correctly-specified negative control from a
        misconfigured positive one, which is the whole failure this option exists to stop.
        """
        for family, call in (
            ("ql", lambda p: quantum_labels.generate_quantum_label_datasets(
                n_qubits=4, n_samples=16, encoding="zz", tau=1.0,
                data_map=data_map, save_path=str(p))),
            ("eng", lambda p: engineered_kernel.generate_engineered_kernel_datasets(
                n_qubits=4, n_samples=16, data_map=data_map, save_path=str(p))),
        ):
            out = tmp_path / f"{family}_{data_map}"
            metas = call(out)
            assert metas[0]["data_map"] == data_map
            name = _only_name(out)
            assert name.endswith("_dmunit") is (data_map == "unit"), name

    def test_the_two_data_maps_are_different_unitaries_not_a_reparameterisation(self):
        """The reason a mismatch cannot be rescued downstream.

        If ``'unit'`` were a rescaling of ``'qiskit'`` the kernel would be the same up to
        a bandwidth, and tuning ``gamma`` would recover the alignment. It is not: the
        states differ, so ``K_Q`` is a different kernel and no downstream search fixes it.
        """
        x = np.array([0.2, 0.7, 0.4, 0.9])
        qiskit_state = quantum_core.zz_feature_state(x, 2, "linear", "qiskit")
        unit_state = quantum_core.zz_feature_state(x, 2, "linear", "unit")
        assert abs(np.vdot(qiskit_state, unit_state)) < 0.99

    def test_evo_records_no_data_map_because_it_has_none(self, tmp_path):
        """``evo`` never reaches the feature map, so a recorded map would be a fiction."""
        metas = quantum_labels.generate_quantum_label_datasets(
            n_qubits=4, n_samples=16, encoding="evo", tau=1.0,
            data_map="unit", save_path=str(tmp_path),
        )
        assert metas[0]["data_map"] is None
        assert "dm" not in _only_name(tmp_path)

    def test_an_unknown_entanglement_is_refused_even_where_it_is_unused(self, tmp_path):
        """``evo`` never reaches the feature map, so nothing else would notice the typo.

        It would still be recorded in the metadata as the pattern used, which is the
        part that misleads: the file says ``entanglement: 'pairwse'`` and the data was
        generated as though a pattern had been chosen.
        """
        for encoding in sorted(quantum_labels.ENCODINGS):
            with pytest.raises(ValueError, match="entanglement"):
                quantum_labels.generate_quantum_label_datasets(
                    n_qubits=4, n_samples=16, encoding=encoding, tau=1.0,
                    entanglement="pairwse", save_path=str(tmp_path),
                )

    def test_the_product_criterion_is_omitted_rather_than_nan(self, tmp_path):
        """The ``e2e`` diagnostic is a log-product, so it needs positive couplings.

        Both ranges default to positive intervals and the runsheet never leaves them,
        but a caller who widens one past zero used to get a ``nan`` accuracy from
        ``np.log`` -- and a nan compares False against everything, so it arrived as a
        plausible number rather than as missing data.
        """
        metas = ground_state.generate_ground_state_datasets(
            n_qubits=4, n_samples=16, label="e2e", kappa=0.5, J_range=(-1.5, 1.5),
            save_path=str(tmp_path),
        )
        diagnostics = metas[0]["diagnostics"]
        assert diagnostics["product_criterion_holdout_acc"] is None
        assert "omitted" in diagnostics["product_criterion_note"]

    def test_the_level_spacing_ratio_is_nan_rather_than_an_error(self):
        """Fewer than two gaps is not enough to have a ratio, and te at n=1 has none.

        It is a reported diagnostic rather than a computed input, so the honest value
        is nan. Returning it explicitly also stops the mean-of-empty-slice warning
        that made a legitimate small run look broken.
        """
        assert np.isnan(quantum_core.level_spacing_ratio(np.array([0.0, 1.0])))
        assert not np.isnan(quantum_core.level_spacing_ratio(np.arange(8.0) ** 1.5))


class TestTheSweepArgumentsAcceptWhatCallersActuallyPass:
    """``as_list`` decides what one configuration is, so a wrong answer sweeps wrongly."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (8, [8]),
            ([6, 8], [6, 8]),
            ((6, 8), [6, 8]),
            (range(4, 7), [4, 5, 6]),
            (np.array([6, 8]), [6, 8]),
            ("sparse", ["sparse"]),
            (np.int64(6), [6]),
            (np.array(6), [6]),
        ],
    )
    def test_as_list_separates_sequences_from_scalars(self, value, expected):
        assert quantum_core.as_list(value) == expected

    def test_as_list_returns_json_serialisable_scalars(self):
        """Every knob it returns is written verbatim into ``meta/<name>.json``.

        A NumPy integer is not JSON-serialisable, and the writer's ``default=float``
        fallback would record a qubit count of ``6`` as ``6.0``.
        """
        assert json.dumps({"n": quantum_core.as_list(np.array([6, 8]))}) == '{"n": [6, 8]}'

    def test_as_float_list_takes_an_array_of_times(self):
        """``np.linspace`` is the obvious way to build a tau ladder or a time grid."""
        assert quantum_core.as_float_list(np.linspace(0.5, 1.5, 3)) == [0.5, 1.0, 1.5]
        assert quantum_core.as_float_list(1) == [1.0]

    def test_a_qubit_count_given_as_an_array_sweeps_rather_than_crashing(self, tmp_path):
        """The end-to-end consequence: an ndarray used to become one bogus configuration."""
        metas = quantum_labels.generate_quantum_label_datasets(
            n_qubits=np.array([3, 4]), n_samples=16, encoding="zz", tau=1.0,
            save_path=str(tmp_path),
        )
        assert [meta["n"] for meta in metas] == [3, 4]
        assert len(_names(tmp_path)) == 2
