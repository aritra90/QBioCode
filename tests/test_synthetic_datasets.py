"""benchmark/create_synthetic_datasets.py writes datasets QProfiler's protocol can run.

The contract:

  1. **The label is what the family says it is.** Re-derived here from the stored latent
     coordinates with an independent formula, it matches every row.
  2. **No column carries the label.** The reference sphere and swiss roll put cos(k theta)
     in a feature, which made their labels an XOR of two columns.
  3. **One manifold per configuration.** The embedding map is fixed per (family, d, k), so
     two seeds sample the same manifold; the rows differ.
  4. **Byte-deterministic, balanced, in [0, 1], in curate.py's format**, and make_splits.py
     and the generator's dataset resolution accept the output.
  5. **The meta-analysis hold-out (holdout.py) draws whole clusters**: every variant of a
     synthetic family falls on one side, the draw ignores listing order, and it is
     written once.
  6. **A positive control is gated on its generator and matched kernel alone**
     (control_gates.py): no classical learner, no benchmark row. The dry run's
     8-qubit ql_zz is rejected before any run.

benchmark/ is not a package: the scripts are loaded by path.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

BENCHMARK = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
pytestmark = pytest.mark.skipif(not (BENCHMARK / "create_synthetic_datasets.py").is_file(),
                                reason="benchmark/create_synthetic_datasets.py not in this checkout")


def _load(name):
    sys.path.insert(0, str(BENCHMARK))
    spec = importlib.util.spec_from_file_location(name, BENCHMARK / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def csd():
    return _load("create_synthetic_datasets")


@pytest.fixture(scope="module")
def ss():
    return _load("synthetic_shapes")


@pytest.fixture(scope="module")
def ho():
    return _load("holdout")


@pytest.fixture(scope="module")
def cg():
    return _load("control_gates")


SHAPE_CASES = [("torus", 2, 8), ("torus", 3, 4), ("sphere", 2, 8), ("concentric_circles", 3, 8),
               ("concentric_spheres", 2, 6), ("checkerboard", 3, 8), ("random_manifold", 2, 8),
               ("swiss_roll", 3, 8), ("half_moons", 3, 8), ("parity", 3, 8),
               ("perm_parity", 3, 8), ("simple_linear", 0, 8)]


def _independent_F(name, k, d, latent):
    """The label function from the latent coordinates, written out again by hand."""
    if name == "torus":
        t, f = latent.T
        return np.cos(k * t) * np.cos(k * f)
    if name == "sphere":
        theta, phi = latent.T
        return np.cos(k * theta) * np.cos(phi)
    if name in ("concentric_circles", "concentric_spheres"):
        return np.sin((k + 1) * np.pi * latent[:, 0])
    if name == "checkerboard":
        return np.sin(k * np.pi * latent[:, 0]) * np.sin(k * np.pi * latent[:, 1])
    if name == "random_manifold":
        return np.cos(k * np.pi * latent[:, 0]) * np.cos(k * np.pi * latent[:, 1])
    if name == "swiss_roll":
        t, h = latent.T
        return np.sin(k * np.pi * (t - 1.5 * np.pi) / (3 * np.pi)) * h
    if name == "half_moons":
        return np.where(latent[:, 0] % 2 == 0, 1.0, -1.0)
    if name == "parity":
        return np.prod(latent - 0.5, axis=1)
    if name == "perm_parity":
        sign = np.ones(len(latent))
        for a in range(k):
            for b in range(a + 1, k):
                sign *= np.sign(latent[:, b] - latent[:, a])
        return sign
    if name == "simple_linear":
        return latent[:, 0]
    raise AssertionError(name)


class TestShapes:
    @pytest.mark.parametrize("name, k, d", SHAPE_CASES)
    def test_label_shape_range_and_balance(self, csd, name, k, d):
        frame, y, F, latent, _ = csd.make_shape(name, 200, d, k, seed=3, noise=0.0)
        assert frame.shape == (200, d)
        assert frame.to_numpy().min() >= 0.0 and frame.to_numpy().max() <= 1.0
        assert (y == 1).sum() == (y == 0).sum() == 100
        np.testing.assert_array_equal(y, (F > 0).astype(int))
        np.testing.assert_allclose(_independent_F(name, k, d, latent), F, atol=1e-12)

    @pytest.mark.parametrize("name, k", [("sphere", 3), ("swiss_roll", 3), ("torus", 2)])
    def test_no_column_carries_the_band_function(self, csd, name, k):
        frame, y, F, latent, _ = csd.make_shape(name, 400, 8, k, seed=0, noise=0.0)
        if name == "sphere":
            band = np.cos(k * latent[:, 0])
        elif name == "swiss_roll":
            band = np.sin(k * np.pi * (latent[:, 0] - 1.5 * np.pi) / (3 * np.pi))
        else:
            band = np.cos(k * latent[:, 0])
        corr = [abs(np.corrcoef(frame[c], band)[0, 1]) for c in frame]
        assert max(corr) < 0.95, corr

    def test_two_seeds_share_one_embedding(self, csd, ss):
        # torus d=8: columns 4-7 are fixed harmonics of (t, f), the same map for every seed.
        for seed in (0, 1):
            frame, y, F, latent, _ = csd.make_shape("torus", 100, 8, 2, seed=seed, noise=0.0)
            extra = ss._harmonics(latent, csd.structure_rng("torus", 8, 2), 4)
            np.testing.assert_allclose(frame.iloc[:, 4:].to_numpy(), extra, atol=1e-12)
        a = csd.make_shape("torus", 100, 8, 2, seed=0, noise=0.0)[0]
        b = csd.make_shape("torus", 100, 8, 2, seed=1, noise=0.0)[0]
        assert not np.allclose(a.to_numpy(), b.to_numpy())

    def test_noise_comes_after_the_label(self, csd):
        clean = csd.make_shape("torus", 100, 4, 2, seed=5, noise=0.0)
        noisy = csd.make_shape("torus", 100, 4, 2, seed=5, noise=0.05)
        np.testing.assert_array_equal(clean[1], noisy[1])          # labels unchanged
        assert not np.allclose(clean[0].to_numpy(), noisy[0].to_numpy())

    @pytest.mark.parametrize("name", ["torus", "checkerboard", "sphere"])
    def test_noise_leaves_no_atoms_at_the_cube_faces(self, csd, name):
        # Clipped noise put ~5% of a torus's entries at exactly 0 or 1, in every column.
        frame = csd.make_shape(name, 400, 8, 2, seed=0, noise=0.01)[0].to_numpy()
        assert frame.min() > 0.0 and frame.max() < 1.0
        assert len(np.unique(frame)) == frame.size

    def test_reflection_folds_and_keeps_the_rest(self, csd):
        x = np.array([-0.02, 0.0, 0.3, 1.0, 1.03])
        np.testing.assert_allclose(csd.reflect_into_unit(x), [0.02, 0.0, 0.3, 1.0, 0.97])

    def test_cells_follow_k(self, ss):
        assert ss.SHAPES["torus"].cells(2, 4) == 16
        assert ss.SHAPES["checkerboard"].cells(5, 8) == 25
        assert ss.SHAPES["perm_parity"].cells(4, 8) == 24


class TestQuantum:
    def test_angle_encoding_label_is_the_fidelity_product(self, csd):
        frame, y, F, latent, _ = csd.make_quantum("angle_encoding", 200, 6, 2, seed=1)
        X = frame.to_numpy()
        sq = _load("synthetic_quantum")
        ang = csd.structure_rng("angle_encoding", 6, 2).uniform(0.0, np.pi / 2, 2)
        fid = np.ones(len(X))
        for i in range(2):
            amp0, amp1 = np.cos(np.pi * X[:, 2 * i]), np.sin(np.pi * X[:, 2 * i]) * np.exp(
                2j * np.pi * X[:, 2 * i + 1])
            fid *= np.abs(np.cos(ang[i]) * amp0 + np.sin(ang[i]) * amp1) ** 2
        np.testing.assert_array_equal(y, (fid > np.median(fid)).astype(int))
        # F is fid minus its median, row for row.
        assert np.all(np.argsort(F, kind="stable") == np.argsort(fid - np.median(fid), kind="stable"))
        assert (y == 1).sum() == 100
        # A product state: its kernel has an exact classical twin, so it fails gate G1.
        assert sq.QUANTUM["angle_encoding"].role == "product_kernel_control"
        assert sq.QUANTUM["angle_encoding"].entangling is False

    @pytest.mark.parametrize("name, d, p", [("ql_zz", 4, 4), ("gs_sparse", 5, 5), ("hl", 4, 4)])
    def test_qbiocode_families_round_trip(self, csd, name, d, p):
        frame, y, F, latent, params = csd.make_quantum(name, 40, d, 2, seed=0)
        assert frame.shape == (40, p) and set(np.unique(y)) == {0, 1}
        assert abs(int(y.sum()) - 20) <= 1
        assert "qbiocode_meta" in params and len(F) == 40


class TestTheTool:
    def test_plan_skips_what_a_family_cannot_take(self, csd):
        jobs, skips = csd.plan(["parity", "torus"], ["te", "angle_encoding"], [2, 9], [3, 8],
                               [300], [0])
        text = "\n".join(skips)
        assert "parity k=9 d=3" in text and "torus d=3" in text
        assert "te d=8 n=300: at most 256 distinct rows" in text
        assert "angle_encoding k=9 d=8" in text
        assert ("shapes", "parity", 2, 8, 300, 0, 1.0) in jobs

    def test_a_family_without_k_is_built_once(self, csd):
        jobs, _ = csd.plan(["simple_linear"], [], [2, 5, 8], [8], [100], [0])
        assert jobs == [("shapes", "simple_linear", None, 8, 100, 0, 1.0)]

    def test_bandwidth_reaches_only_the_families_that_take_one(self, csd):
        jobs, _ = csd.plan(["torus"], ["ql_zz"], [2], [4], [100], [0], [1.0, 0.5])
        assert ("shapes", "torus", 2, 4, 100, 0, 1.0) in jobs
        assert not any(j[1] == "torus" and j[6] == 0.5 for j in jobs)
        assert {j[6] for j in jobs if j[1] == "ql_zz"} == {1.0, 0.5}
        assert csd.dataset_id("quantum", "ql_zz", 2, 4, 100, 0, 0.5) == "quantum__ql_zz_k2_d4_bw0.5_n100_s0"

    def test_writes_the_curated_format_deterministically(self, csd, tmp_path):
        args = ["--shapes", "torus,half_moons", "--quantum", "angle_encoding", "--k", "2",
                "--d", "4", "--n", "60"]
        assert csd.main(["--out-root", str(tmp_path / "a" / "datasets"), *args]) == 0
        assert csd.main(["--out-root", str(tmp_path / "b" / "datasets"), *args]) == 0
        ids = sorted(p.name for p in (tmp_path / "a" / "datasets").iterdir())
        assert ids == ["quantum__angle_encoding_k2_d4_n60_s0", "shapes__half_moons_k2_d4_n60_s0",
                       "shapes__torus_k2_d4_n60_s0"]
        for ds in ids:
            a = (tmp_path / "a" / "datasets" / ds / f"{ds}.csv").read_bytes()
            b = (tmp_path / "b" / "datasets" / ds / f"{ds}.csv").read_bytes()
            assert a == b
            meta = yaml.safe_load((tmp_path / "a" / "datasets" / ds / "meta.yaml").read_text())
            assert meta["dataset_id"] == ds and meta["n"] == 60 and meta["p"] == 4
            assert meta["class_counts"] == {0: 30, 1: 30} and meta["group_col"] is None
            assert meta["role"] and meta["label_rule"] and meta["family"].startswith(("shape_", "quantum_"))
            df = pd.read_csv(tmp_path / "a" / "datasets" / ds / f"{ds}.csv")
            assert list(df.columns)[-1] == "label" and set(df["label"]) == {0, 1}
        inv = pd.read_csv(tmp_path / "a" / "inventory_synthetic.csv")
        assert sorted(inv["dataset_id"]) == ids

    def test_dry_run_writes_nothing(self, csd, tmp_path):
        assert csd.main(["--out-root", str(tmp_path / "datasets"), "--shapes", "all",
                         "--dry-run"]) == 0
        assert not (tmp_path / "datasets").exists()

    def test_make_splits_and_the_generator_take_the_output(self, csd, tmp_path):
        out = tmp_path / "datasets"
        assert csd.main(["--out-root", str(out), "--shapes", "torus", "--quantum",
                         "angle_encoding", "--k", "2", "--d", "4", "--n", "60"]) == 0
        ms = _load("make_splits")
        assert ms.main(["--datasets", str(out), "--out", str(tmp_path / "splits"), "--env-path",
                        str(tmp_path)]) == 0
        from qbiocode.apps.qprofiler import split_manifest
        for ds in ("shapes__torus_k2_d4_n60_s0", "quantum__angle_encoding_k2_d4_n60_s0"):
            manifest = split_manifest.load_manifest(tmp_path / "splits" / f"{ds}.json")
            frame = pd.read_csv(out / ds / f"{ds}.csv")
            split_manifest.verify_dataset(manifest, dataset_file=out / ds / f"{ds}.csv",
                                          y=frame["label"].to_numpy())
        spec = importlib.util.spec_from_file_location(
            "gpc", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "pilot10"
            / "generate_pilot_configs.py")
        gpc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpc)
        assert gpc.resolve_dataset("torus_k2_d4_n60_s0", str(out)) == "shapes__torus_k2_d4_n60_s0"
        assert gpc.curated_dataset("shapes__torus_k2_d4_n60_s0", str(out))["feats"] == 4


class TestHoldout:
    """benchmark/holdout.py draws whole clusters, reproducibly, and only once."""

    @staticmethod
    def _tree(root, families):
        for ds, family in families.items():
            (root / ds).mkdir(parents=True)
            (root / ds / "meta.yaml").write_text(yaml.safe_dump({"dataset_id": ds, "family": family}))
        return root

    FAMILIES = {
        **{f"shapes__torus_k{k}_d8_n400_s{s}": "shape_torus" for k in (2, 5) for s in (0, 1)},
        **{f"quantum__ql_zz_k{k}_d8_n400_s0": "quantum_ql_zz" for k in (2, 5, 8)},
        "shapes__sphere_k2_d8_n400_s0": "shape_sphere",
        **{f"pmlb__real{i}": "pmlb_binary" for i in range(6)},
    }

    def test_a_synthetic_family_is_one_cluster_and_a_real_dataset_its_own(self, ho, tmp_path):
        frame = ho.dataset_clusters(self._tree(tmp_path / "datasets", self.FAMILIES))
        cl = dict(zip(frame["dataset_id"], frame["cluster"]))
        assert {cl[d] for d in self.FAMILIES if d.startswith("shapes__torus")} == {"shape_torus"}
        assert {cl[d] for d in self.FAMILIES if d.startswith("quantum__ql_zz")} == {"quantum_ql_zz"}
        assert all(cl[f"pmlb__real{i}"] == f"pmlb__real{i}" for i in range(6))
        assert frame["cluster"].nunique() == 3 + 6

    def test_no_cluster_is_split_and_the_draw_is_order_free(self, ho, tmp_path):
        frame = ho.dataset_clusters(self._tree(tmp_path / "datasets", self.FAMILIES))
        for seed in range(20):
            out = ho.draw_holdout(frame, 0.3, seed)
            assert (out.groupby("cluster")["holdout"].nunique() == 1).all()
            assert out.loc[out["holdout"], "cluster"].nunique() == round(0.3 * 9)
            shuffled = ho.draw_holdout(frame.sample(frac=1.0, random_state=seed), 0.3, seed)
            pd.testing.assert_frame_equal(out, shuffled)

    def test_a_cluster_map_merges_real_variants(self, ho, tmp_path):
        root = self._tree(tmp_path / "datasets", self.FAMILIES)
        pd.DataFrame({"dataset_id": ["pmlb__real0", "pmlb__real1"], "cluster": ["realA"] * 2}).to_csv(
            tmp_path / "map.csv", index=False)
        frame = ho.dataset_clusters(root, tmp_path / "map.csv")
        assert frame.set_index("dataset_id").loc[["pmlb__real0", "pmlb__real1"], "cluster"].tolist() == ["realA"] * 2
        assert frame["cluster"].nunique() == 8

    def test_written_once_and_reproducible(self, ho, tmp_path, capsys):
        root = self._tree(tmp_path / "datasets", self.FAMILIES)
        out = tmp_path / "holdout.csv"
        args = ["--datasets", str(root), "--fraction", "0.3", "--seed", "7", "--out", str(out)]
        assert ho.main(args) == 0
        first = out.read_bytes()
        assert ho.main(args) == 1, "a second draw must be refused"
        assert out.read_bytes() == first
        assert ho.main([*args, "--force"]) == 0 and out.read_bytes() == first
        with pytest.raises(ValueError, match="empty side"):
            ho.draw_holdout(ho.dataset_clusters(root), 0.01, 0)


class TestControlGates:
    """A positive control is accepted on its generator and matched kernel alone."""

    def test_the_gates_need_no_classical_model(self, cg):
        import inspect
        source = inspect.getsource(cg)
        for name in ("RandomForest", "XGB", "CatBoost", "TabPFN", "LogisticRegression", "MLP"):
            assert name not in source, f"control_gates fits a classical learner ({name})"
        assert "kernel=\"precomputed\"" in source   # only the matched kernel is ever fitted

    def test_an_identity_kernel_is_concentrated_and_a_product_map_fails_g1(self, cg):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (80, 3))
        y = (X[:, 0] > 0.5).astype(int)
        result = cg.evaluate(X, y, np.eye(80), n_train=64, entangling=False)
        assert result["passed"]["G1"] is False and result["passed"]["G2"] is False
        assert result["accepted"] is False and "G1" in cg.reasons(result)

    def test_a_fidelity_kernel_is_judged_against_the_random_state_overlap(self, cg):
        """2^-d is what random states overlap by, so a fixed floor would ignore the width."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (60, 4))
        y = (X[:, 0] > 0.5).astype(int)
        near_random = np.full((60, 60), 1.5 / 16) + np.eye(60) * (1 - 1.5 / 16)
        twice = np.full((60, 60), 3.0 / 16) + np.eye(60) * (1 - 3.0 / 16)
        g_near = cg.evaluate(X, y, near_random, 48, True, n_qubits=4)
        g_twice = cg.evaluate(X, y, twice, 48, True, n_qubits=4)
        assert g_near["values"]["overlap_over_random"] == pytest.approx(1.5)
        assert "random" in cg.reasons(g_near)
        assert g_twice["values"]["overlap_over_random"] == pytest.approx(3.0)
        assert g_near["passed"]["G2"] is False

    def test_the_dry_run_ql_zz_is_rejected_before_any_run(self, csd):
        """The 2026-10-06 dry run's control (8 qubits, bandwidth 1) lost to TabPFN.

        Its matched kernel was near-identity there (off-diagonal 0.0074); the gates see
        that from a pilot draw, with no benchmark row and no classical model.
        """
        gate = csd.control_gate("ql_zz", 2, 8, 400, 1.0)
        assert gate["accepted"] is False
        assert gate["passed"]["G2"] is False and gate["passed"]["G3"] is False
        assert gate["values"]["offdiag_mean"] < 0.02

    def test_a_failed_control_is_skipped_or_kept_as_gate_rejected(self, csd, tmp_path):
        args = ["--quantum", "ql_zz", "--k", "2", "--d", "8", "--n", "400"]
        assert csd.main(["--out-root", str(tmp_path / "a" / "datasets"), *args]) == 0
        assert not (tmp_path / "a" / "datasets" / "quantum__ql_zz_k2_d8_n400_s0").exists()
        assert csd.main(["--out-root", str(tmp_path / "b" / "datasets"), *args,
                         "--keep-failed-controls"]) == 0
        meta = yaml.safe_load((tmp_path / "b" / "datasets" / "quantum__ql_zz_k2_d8_n400_s0"
                               / "meta.yaml").read_text())
        assert meta["role"] == "gate_rejected" and meta["control_accepted"] is False
        assert meta["control_gates"]["version"].startswith("control_gates/")
        assert meta["control_gates"]["pilot_seed"] not in (0, 1, 2)

    def test_the_ql_seeds_sample_one_concept(self, csd, tmp_path):
        out = tmp_path / "datasets"
        assert csd.main(["--out-root", str(out), "--quantum", "ql_evo", "--k", "2", "--d", "4",
                         "--n", "40", "--seeds", "0,1"]) == 0
        metas = [yaml.safe_load((out / f"quantum__ql_evo_k2_d4_n40_s{s}" / "meta.yaml").read_text())
                 for s in (0, 1)]
        assert metas[0]["concept_seed"] == metas[1]["concept_seed"] is not None
        assert metas[0]["sha256"] != metas[1]["sha256"]
