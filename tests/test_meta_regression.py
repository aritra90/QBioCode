"""Tests for :mod:`qbiocode.utils.meta_regression`.

The module exists because the obvious analyses return clean-looking p-values that are
wrong: rows treated as independent, or a pairs bootstrap over twelve datasets. A test
that checked shapes would pass against either. So the load-bearing tests here are
statistical. The fixtures have few clusters, strong within-cluster dependence and a known
answer, and the tests assert level and power. The analytic pieces are pinned against
statsmodels, the reference a reviewer would reach for.

``TestFewClusterLevel`` is the one to keep if the others are ever dropped. It encodes
why the primary test is the wild cluster restricted bootstrap: at G = 12 with correlated
errors, iid OLS rejects a true null about a third of the time, and the wild bootstrap
holds 5%.

Every fixture is synthetic and seeded; nothing reads the corpus.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from qbiocode.utils import meta_regression as mr
from qbiocode.utils.fair_selection import select_winners


def clustered_design(G=12, per=5, seed=0):
    """A cluster-level covariate on G clusters of ``per`` rows, plus an intercept."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(G), per)
    x = rng.standard_normal(G)[codes]
    return np.column_stack([np.ones(codes.size), x]), codes


def clustered_noise(codes, R, rho=0.8, seed=1):
    """``N x R`` null responses with intra-cluster correlation ``rho``."""
    rng = np.random.default_rng(seed)
    G = codes.max() + 1
    return (np.sqrt(rho) * rng.standard_normal((G, R))[codes]
            + np.sqrt(1 - rho) * rng.standard_normal((codes.size, R)))


def results_frame(datasets=("a", "b", "c"), embeddings=("pca", "umap"), iters=5,
                  quantum_shift=0.0, seed=0):
    """A ModelResults-shaped frame: two classical and two quantum arms per pass."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in datasets:
        for emb in embeddings:
            for it in range(1, iters + 1):
                for m in ("svc", "lr"):
                    rows.append(dict(Dataset=d, embeddings=emb, iteration=it, model=m,
                                     balanced_accuracy=rng.normal(0.75, 0.03)))
                for m in ("qsvc", "pqk"):
                    rows.append(dict(Dataset=d, embeddings=emb, iteration=it, model=m,
                                     balanced_accuracy=rng.normal(0.75 + quantum_shift, 0.03)))
    return pd.DataFrame(rows)


def design_rows(n_none=3, n_embedded=2, iters=5):
    """Instance rows shaped like the pilot: some datasets raw, some with two embeddings."""
    rows = []
    for d in range(n_none):
        rows += [dict(Dataset=f"n{d}", embeddings="none", iteration=i) for i in range(1, iters + 1)]
    for d in range(n_embedded):
        for emb in ("pca", "umap"):
            rows += [dict(Dataset=f"e{d}", embeddings=emb, iteration=i)
                     for i in range(1, iters + 1)]
    rows = pd.DataFrame(rows).sort_values(list(mr.INSTANCE)).reset_index(drop=True)
    rows["cluster"] = rows["Dataset"]
    return rows


class TestAgainstStatsmodels:
    def test_cr1_standard_error_matches(self):
        X, codes = clustered_design()
        y = clustered_noise(codes, 1)[:, 0] + 0.3 * X[:, 1]
        ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
        t, se = mr.ClusteredOLS(X, codes, 12).t_cr1(y, 1)
        assert se[0] == pytest.approx(ref.bse[1], rel=1e-10)
        assert t[0] == pytest.approx(ref.tvalues[1], rel=1e-10)

    def test_cv3_is_the_cluster_jackknife(self):
        X, codes = clustered_design()
        y = clustered_noise(codes, 1)[:, 0]
        full = np.linalg.lstsq(X, y, rcond=None)[0][1]
        loo = [np.linalg.lstsq(X[codes != g], y[codes != g], rcond=None)[0][1] for g in range(12)]
        expected = np.sqrt(11 / 12 * np.sum((np.array(loo) - full) ** 2))
        _, se = mr.ClusteredOLS(X, codes, 12).t_cv3(y, 1)
        assert se[0] == pytest.approx(expected, rel=1e-10)

    def test_vif_matches(self):
        rng = np.random.default_rng(3)
        a, b = rng.standard_normal(40), rng.standard_normal(40)
        frame = pd.DataFrame({"a": a, "b": b, "c": a + 0.5 * b + 0.3 * rng.standard_normal(40)})
        exog = sm.add_constant(frame).to_numpy()
        ours = mr.variance_inflation(frame)
        for i, name in enumerate(frame.columns, start=1):
            assert ours[name] == pytest.approx(variance_inflation_factor(exog, i), rel=1e-8)

    def test_vif_is_infinite_for_an_exact_combination(self):
        rng = np.random.default_rng(4)
        a, b = rng.standard_normal(20), rng.standard_normal(20)
        vif = mr.variance_inflation(pd.DataFrame({"a": a, "b": b, "c": a - 2 * b}))
        assert np.isinf(vif).all()

    @pytest.mark.parametrize("ours, theirs", [("bh", "fdr_bh"), ("by", "fdr_by")])
    def test_fdr_matches(self, ours, theirs):
        p = np.random.default_rng(5).uniform(size=30) ** 2
        assert np.allclose(mr.fdr_adjust(p, ours), multipletests(p, method=theirs)[1])

    def test_fdr_leaves_nans_out_of_the_family(self):
        q = mr.fdr_adjust([0.01, np.nan, 0.04])
        assert np.isnan(q[1])
        assert np.allclose(q[[0, 2]], multipletests([0.01, 0.04], method="fdr_bh")[1])


@pytest.fixture(scope="module")
def null():
    """400 null responses on the G = 12 design, intra-cluster correlation 0.8."""
    X, codes = clustered_design()
    return mr.ClusteredOLS(X, codes, 12), clustered_noise(codes, 400)


class TestFewClusterLevel:
    """At G = 12 with correlated errors: who holds 5% on a true null, and who does not."""

    def test_iid_ols_over_rejects(self, null):
        from scipy import stats
        ols, Y = null
        t, _ = ols.t_iid(Y, 1)
        rate = np.mean(2 * stats.t.sf(np.abs(t), ols.N - ols.K) < 0.05)
        assert rate > 0.2

    def test_wild_cluster_restricted_holds_its_level(self, null):
        ols, Y = null
        _, p = ols.wild_restricted(Y, 1, 199, np.random.default_rng(6))
        assert 0.02 <= np.mean(p < 0.05) <= 0.09

    def test_wild_cluster_restricted_still_detects_a_real_effect(self):
        X, codes = clustered_design()
        y = clustered_noise(codes, 1, seed=7)[:, 0] + 1.5 * X[:, 1]
        _, p = mr.ClusteredOLS(X, codes, 12).wild_restricted(y, 1, 999, np.random.default_rng(8))
        assert p[0] < 0.01

    def test_webb_weights_have_mean_zero_and_unit_variance(self):
        assert mr.WEBB_WEIGHTS.mean() == pytest.approx(0.0, abs=1e-15)
        assert np.mean(mr.WEBB_WEIGHTS ** 2) == pytest.approx(1.0, rel=1e-12)

    def test_p_values_are_reproducible_under_a_seed(self, null):
        ols, Y = null
        a = ols.wild_restricted(Y[:, :5], 1, 99, np.random.default_rng(9))[1]
        b = ols.wild_restricted(Y[:, :5], 1, 99, np.random.default_rng(9))[1]
        assert np.array_equal(a, b)


class TestBlockPermutations:
    def test_whole_clusters_move_only_onto_clusters_of_the_same_shape(self):
        rows = design_rows()
        codes = pd.factorize(rows["cluster"], sort=True)[0]
        idx = mr.block_permutations(rows, codes, 50, np.random.default_rng(10))
        moved = False
        for perm in idx:
            assert sorted(perm) == list(range(len(rows)))
            for g in np.unique(codes):
                dst = np.flatnonzero(codes == g)
                src = perm[dst]
                assert len(set(codes[src])) == 1, "a cluster's block was split"
                assert (rows.loc[src, ["embeddings", "iteration"]].to_numpy()
                        == rows.loc[dst, ["embeddings", "iteration"]].to_numpy()).all()
                moved |= codes[src[0]] != g
        assert moved

    def test_a_shape_held_by_one_cluster_never_moves(self):
        rows = design_rows(n_none=3, n_embedded=1)
        codes = pd.factorize(rows["cluster"], sort=True)[0]
        idx = mr.block_permutations(rows, codes, 20, np.random.default_rng(11))
        lone = np.flatnonzero(rows["Dataset"].to_numpy() == "e0")
        assert (idx[:, lone] == lone).all()


@pytest.fixture(scope="module")
def unit_frame():
    """16 passes: a twin pair, a near-exact combination, a constant, a gap, a wide range."""
    rng = np.random.default_rng(12)
    idx = pd.MultiIndex.from_tuples([(f"d{i}", "none") for i in range(16)],
                                    names=list(mr.UNIT))
    a, b, c = (rng.standard_normal(16) for _ in range(3))
    return pd.DataFrame({
        "a": a, "a_twin": 3 * a + 1, "b": b, "c": c,
        "abc": a + b + c + 0.05 * rng.standard_normal(16),
        "const": np.ones(16), "gappy": np.r_[np.nan, rng.standard_normal(15)],
        "wide": np.exp(3 * rng.standard_normal(16)),
    }, index=idx)


class TestScreen:
    def test_stages_are_the_ones_the_columns_deserve(self, unit_frame):
        screen = mr.screen_features(unit_frame)
        stage = screen.ledger["stage"]
        assert stage["gappy"] == "nonfinite"
        assert stage["const"] == "near_constant"
        assert {stage["a"], stage["a_twin"]} == {"selected", "redundant"}
        assert screen.ledger.loc["wide", "transform"] == "log10"
        funnel = screen.funnel()
        assert list(funnel["stage"]) == list(mr.STAGES)
        assert funnel["n_features"].sum() == unit_frame.shape[1]

    def test_every_admitted_feature_is_under_the_vif_ceiling(self, unit_frame):
        screen = mr.screen_features(unit_frame, vif_max=5.0)
        assert (screen.vif <= 5.0).all()
        assert not {"a", "b", "c", "abc"} <= set(screen.selected)

    def test_selection_does_not_depend_on_column_order(self, unit_frame):
        a = mr.screen_features(unit_frame).selected
        b = mr.screen_features(unit_frame[unit_frame.columns[::-1]]).selected
        assert sorted(a) == sorted(b)

    def test_the_screen_cannot_see_the_response(self):
        assert not {"y", "response", "qadv"} & set(inspect.signature(mr.screen_features).parameters)

    def test_unreliable_and_host_dependent_features_leave_early(self, unit_frame):
        rel = pd.DataFrame({"reliability": 1.0, "host_noise": 0.0}, index=unit_frame.columns)
        rel.loc["b", "reliability"] = 0.3
        rel.loc["c", "host_noise"] = 0.9
        stage = mr.screen_features(unit_frame, rel).ledger["stage"]
        assert stage["b"] == "unreliable"
        assert stage["c"] == "host_dependent"


class TestMetaFeatures:
    def test_median_over_model_rows_and_nonfinite_become_nan(self):
        rows = pd.DataFrame({
            "Dataset": "d", "embeddings": "none", "iteration": 1,
            "model": ["svc", "lr", "qsvc"], "balanced_accuracy": [0.7, 0.8, 0.6],
            "mfe.x": [1.0, 1.0, 10.0], "mfe.y": [np.inf, 2.0, 2.0],
        })
        X, quality = mr.canonical_meta_features(rows)
        assert list(X.columns) == ["mfe.x", "mfe.y"]
        assert X.iloc[0]["mfe.x"] == 1.0
        assert X.iloc[0]["mfe.y"] == 2.0
        assert quality.loc["mfe.y", "nonfinite_rows"] == pytest.approx(1 / 3)
        assert quality.loc["mfe.x", "host_sd"] > 0

    def test_reliability_separates_stable_from_noisy_features(self):
        rng = np.random.default_rng(13)
        idx = pd.MultiIndex.from_product([[f"d{i}" for i in range(10)], ["none"], range(5)],
                                         names=list(mr.INSTANCE))
        stable = np.repeat(rng.standard_normal(10), 5)
        X = pd.DataFrame({"stable": stable, "noise": rng.standard_normal(50)}, index=idx)
        rel = mr.feature_reliability(X)
        assert rel.loc["stable", "reliability"] == pytest.approx(1.0)
        assert rel.loc["noise", "reliability"] < 0.5

    def test_tuning_evidence_is_not_a_meta_feature(self):
        # tuning_score is a validation score of the model (label-dependent), and
        # tuning_reused reads as bool once every row is tuned: neither is a covariate.
        from qbiocode.evaluation.model_evaluation import TUNING_EVIDENCE_COLUMNS

        rows = pd.DataFrame({
            "Dataset": "d", "embeddings": "none", "iteration": 1,
            "model": ["svc", "qsvc"], "balanced_accuracy": [0.7, 0.6],
            "tuning_metric": "balanced_accuracy", "tuning_score": [0.71, 0.64],
            "tuning_reused": [False, True], "mfe.x": [1.0, 1.0],
        })
        assert set(TUNING_EVIDENCE_COLUMNS) <= set(rows.columns)
        assert mr.meta_feature_columns(rows) == ["mfe.x"]


class TestResponses:
    def test_unit_report_is_select_winners_run_per_pass(self):
        res = results_frame()
        ours = mr.unit_report(res, test_size=0.2, seed=3)
        relabelled = res.assign(Dataset=res["Dataset"] + "|" + res["embeddings"])
        ref = select_winners(relabelled, metric="balanced_accuracy", epsilon=0.027,
                             test_size=0.2, seed=3)
        assert np.allclose(ours.selection["quantum_score"], ref.selection["quantum_score"])
        assert set(zip(ours.selection["Dataset"], ours.selection["embeddings"])) == {
            (d, e) for d in "abc" for e in ("pca", "umap")}

    def test_qadv_is_positive_when_quantum_is_better(self):
        contrast = mr.loio_contrast(mr.unit_report(results_frame(quantum_shift=0.1),
                                                   test_size=0.2))
        assert contrast["qadv"].mean() == pytest.approx(0.1, abs=0.03)

    def test_unit_report_requires_test_size(self):
        with pytest.raises(ValueError, match="test_size"):
            mr.unit_report(results_frame())

    def test_unit_report_forwards_margin_fdr_and_controls(self):
        res = results_frame(quantum_shift=0.1)
        rep = mr.unit_report(res, test_size=0.2, margin=0.01, fdr=0.2, controls="c")
        assert (rep.margin, rep.fdr) == (0.01, 0.2)
        fam = rep.per_dataset.set_index(["Dataset", "embeddings"])["family"]
        assert fam.loc[("c", "pca")] == fam.loc[("c", "umap")] == "control"
        assert set(fam.drop("c", level=0)) == {"discovery"}
        # Returned as dataset names, so they match the split-back Dataset column.
        assert rep.controls == ("c",)
        with pytest.raises(ValueError, match="absent"):
            mr.unit_report(res, test_size=0.2, controls=["nope"])

    def test_matched_contrast_is_the_plain_difference(self):
        res = results_frame()
        m = mr.matched_contrast(res, "qsvc", "svc")
        wide = res.pivot_table(index=list(mr.INSTANCE), columns="model", values="balanced_accuracy")
        assert np.allclose(m["qadv"], (wide["qsvc"] - wide["svc"]).to_numpy())

    def test_family_map_merges_variants_and_leaves_the_rest(self):
        fam = mr.family_map(["te_a_tau1", "te_a_tau4", "wdbc"], {r"^te_a_tau": "te_a"})
        assert fam == {"te_a_tau1": "te_a", "te_a_tau4": "te_a", "wdbc": "wdbc"}

    def test_embedding_indicators_use_none_as_reference(self):
        idx = pd.MultiIndex.from_tuples([("a", "none"), ("b", "pca"), ("b", "umap")],
                                        names=list(mr.UNIT))
        assert list(mr.embedding_indicators(idx).columns) == ["emb[pca]", "emb[umap]"]


def synthetic_design(beta=0.0, G=12, seed=14):
    """Pilot-shaped design: G datasets, two features, response clustered by dataset."""
    rng = np.random.default_rng(seed)
    passes = [(f"d{g:02d}", "none") for g in range(G)]
    Z = pd.DataFrame(rng.standard_normal((G, 2)), columns=["signal", "noise"],
                     index=pd.MultiIndex.from_tuples(passes, names=list(mr.UNIT)))
    resp = pd.DataFrame([dict(Dataset=d, embeddings=e, iteration=i) for d, e in passes
                         for i in range(1, 6)])
    level = dict(zip(Z.index.get_level_values(0), beta * Z["signal"] + 0.5 * rng.standard_normal(G)))
    resp["qadv"] = resp["Dataset"].map(level) + 0.1 * rng.standard_normal(len(resp))
    return mr.build_design(resp, Z)


class TestEndToEnd:
    def test_a_real_signal_is_found_and_the_noise_feature_is_not(self):
        d = synthetic_design(beta=1.0)
        table = mr.marginal_tests(d, n_boot=999, n_perm=999, seed=15).set_index("feature")
        assert table.loc["signal", "q_bh"] < 0.05
        assert table.loc["noise", "p_wcr"] > 0.05
        assert table.loc["signal", "ci_lo"] > 0

    def test_nested_ridge_never_sees_the_held_out_cluster(self):
        d = synthetic_design(beta=1.0)
        X, pen = d.matrix(), d.penalty_mask()
        before, _, _ = mr.ridge_nested_cv(X, d.y, d.codes, d.G, pen)
        y = d.y.copy()
        y[d.codes == 0] += 100.0
        after, _, _ = mr.ridge_nested_cv(X, y, d.codes, d.G, pen)
        held = d.codes == 0
        assert np.allclose(before[held], after[held])
        assert not np.allclose(before[~held], after[~held])

    def test_ridge_omnibus_detects_a_strong_signal(self):
        out = mr.ridge_omnibus(synthetic_design(beta=1.0), n_perm=99, seed=16)
        assert out.delta_r2 > 0
        assert out.p_value <= 0.05

    def test_calibration_reports_every_method_and_ranks_iid_worst(self):
        d = synthetic_design()
        table = mr.null_calibration(d, n_null=40, n_boot=99, seed=17).rejection().set_index("key")
        assert list(table.index) == list(mr.CALIBRATION_METHODS)
        assert table.loc["ols_iid", "type1"] == table["type1"].max()

    def test_joint_model_refuses_when_clusters_are_too_few(self):
        assert mr.joint_tests(synthetic_design(), n_boot=99) is None

    def test_too_few_clusters_give_nan_not_zero_p_values(self, caplog):
        # Two clusters for an intercept and a feature: the CR1 standard error is zero and,
        # before the guard, t was ~1e15 with p_cr1 = 0. Now the cluster-robust columns are
        # NaN, the iid column (wrong by construction) is still reported, and it is logged.
        d = synthetic_design(beta=1.0, G=2)
        with caplog.at_level("WARNING", logger="qbiocode.utils.meta_regression"):
            table = mr.marginal_tests(d, n_boot=99, n_perm=0, seed=19)
        for col in ("se_cr1", "se_cv3", "ci_lo", "ci_hi", "t", "p_cr1", "p_cv3", "p_wcr", "q_bh"):
            assert table[col].isna().all(), col
        assert table["p_ols_iid"].notna().all()
        assert "not identified" in caplog.text

    def test_identified_designs_keep_their_p_values(self):
        table = mr.marginal_tests(synthetic_design(), n_boot=99, n_perm=0, seed=20)
        assert table[["se_cr1", "t", "p_cr1", "p_wcr", "q_bh"]].notna().all().all()

    def test_lodo_influence_keeps_its_columns_when_no_fold_is_estimable(self):
        cols = ["feature", "beta", "beta_min", "beta_max", "sign_flips", "most_influential",
                "beta_without_it"]
        lodo = mr.lodo_influence(synthetic_design(G=2))
        assert list(lodo.columns) == cols
        assert lodo["sign_flips"].isna().all() and lodo["most_influential"].isna().all()
        estimable = mr.lodo_influence(synthetic_design())
        assert list(estimable.columns) == cols and estimable["sign_flips"].notna().all()
        assert list(mr.lodo_influence(synthetic_design(), features=[]).columns) == cols

    def test_within_unit_test_finds_a_shift_on_one_iteration(self):
        d = synthetic_design()
        first = (d.rows["iteration"] == 1).to_numpy()
        shifted = dataclasses.replace(d, y=d.y + 0.3 * first)
        out = mr.within_unit_test(shifted, first.astype(float), n_boot=999, seed=18)
        # The row noise is 0.1, so the SE of the shift is about 0.03.
        assert out["beta"] == pytest.approx(0.3, abs=0.1)
        assert out["p_wcr"] < 0.01


CLUSTER_MAP = Path(__file__).resolve().parents[1] / "experiments" / "cluster_map_draft.csv"


class TestDatasetFamily:
    SYNTHETIC = {
        "eng_zz_n6_gq1_s0.csv": "syn_eng", "eng_zz_n6_gq1_s0_dmunit.csv": "syn_eng",
        "gs_e2e_n8_k0.5_s0.csv": "syn_gs", "gs_sparse_n8_k0.5_s0.csv": "syn_gs",
        "hl_n6_g0.5_shots1000_s0.csv": "syn_hl", "ql_evo_n8_tau1_s0.csv": "syn_ql",
        "ql_zz_n8_tau1_s0.csv": "syn_ql", "ql_zz_pqkdefaults.csv": "syn_ql",
    }

    def test_synthetic_generators_are_one_cluster_each(self):
        assert {n: mr.dataset_family(n) for n in self.SYNTHETIC} == self.SYNTHETIC

    def test_fractional_taus_join_their_generator(self):
        names = [f"te_n10_s4_seed0_tau{t}.csv" for t in ("0.25", "0.5", "1", "2", "4")]
        assert {mr.dataset_family(n) for n in names} == {"syn_te"}

    def test_gametes_variants_are_one_cluster(self):
        names = ["GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1.csv",
                 "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001"]
        assert {mr.dataset_family(n) for n in names} == {"GAMETES"}

    def test_real_datasets_are_their_own_stem(self):
        assert mr.dataset_family("wdbc.csv") == "wdbc"
        assert mr.dataset_family("wdbc.csv|umap") == "wdbc"
        assert mr.dataset_family("analcatdata_lawsuit") == "analcatdata_lawsuit"

    def test_mapping_wins_and_falls_back_to_the_rules(self):
        mapping = {"kc1.csv": "nasa", "pc1": "nasa"}
        assert mr.dataset_family("kc1.csv", mapping) == "nasa"
        assert mr.dataset_family("pc1.csv|pca", mapping) == "nasa"
        assert mr.dataset_family("te_a_tau1.csv", mapping) == "syn_te"

    def test_a_fallback_from_an_explicit_mapping_is_logged(self, caplog, monkeypatch):
        monkeypatch.setattr(mr, "_UNMAPPED_LOGGED", set())
        with caplog.at_level("WARNING", logger=mr.logger.name):
            assert mr.dataset_family("wdbc.csv", {"kc1": "nasa"}) == "wdbc"
            assert mr.dataset_family("wdbc.csv|umap", {"kc1": "nasa"}) == "wdbc"
        hits = [r for r in caplog.records if "not in the explicit mapping" in r.message]
        assert len(hits) == 1

    def test_a_blank_cluster_raises(self):
        with pytest.raises(ValueError, match="no cluster"):
            mr.dataset_family("x", {"x": float("nan")})

    def test_a_stem_with_two_clusters_raises(self):
        with pytest.raises(ValueError, match="two clusters"):
            mr.dataset_family("kc1", {"kc1": "a", "kc1.csv": "b"})

    def test_draft_csv_stem_column_is_ambiguous(self):
        # breast_cancer is both libsvm (Wisconsin) and PMLB (Ljubljana).
        with pytest.raises(ValueError, match="breast_cancer"):
            mr.dataset_family("wdbc", str(CLUSTER_MAP))

    def test_draft_csv_maps_every_dataset_to_59_clusters(self):
        table = pd.read_csv(CLUSTER_MAP)
        assert len(table) == 91
        got = [mr.dataset_family(d, str(CLUSTER_MAP), key="dataset_id")
               for d in table["dataset_id"]]
        assert got == table["cluster_cons"].tolist()
        assert len(set(got)) == 59
        lib = {mr.dataset_family(d, str(CLUSTER_MAP), key="dataset_id", column="cluster_lib")
               for d in table["dataset_id"]}
        assert len(lib) == 67

    def test_rules_agree_with_the_draft_on_synthetic_and_gametes(self):
        table = pd.read_csv(CLUSTER_MAP)
        rule = table[table["stem"].str.match(r"(eng|gs|hl|ql|te)_|GAMETES")]
        assert len(rule) == 19
        assert [mr.dataset_family(s) for s in rule["stem"]] == rule["cluster_cons"].tolist()


class TestManifestMode:
    """split_mode: manifest -- protocol columns reserved, validation selection per pass."""

    @staticmethod
    def fold_frame(k=3, repeats=2, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for d in ("a", "b"):
            for emb in ("pca", "umap"):
                for g in range(k * repeats):
                    for m, val in (("svc", 0.6), ("lr", 0.9), ("qsvc", 0.7), ("pqk", 0.5)):
                        rows.append(dict(Dataset=d, embeddings=emb, iteration=g + 1,
                                         repeat=g // k, fold=g % k, model=m,
                                         split_mode="manifest", split_k=k,
                                         split_repeats=repeats, n_fit=30, n_val=10,
                                         n_test=10, seed=42, q_seed=42, embed_seed=43 + g,
                                         balanced_accuracy=rng.normal(0.75, 0.03),
                                         tuning_score=val,
                                         tuning_metric="balanced_accuracy",
                                         **{"mfe.x": {"a": 1.0, "b": 2.0}[d]}))
        return pd.DataFrame(rows)

    def test_protocol_columns_are_not_meta_features(self):
        from qbiocode.evaluation.protocol import PROTOCOL_COLUMNS
        res = self.fold_frame()
        assert {"repeat", "fold", "n_val", "embed_seed", "split_k"} <= set(PROTOCOL_COLUMNS)
        assert mr.meta_feature_columns(res) == ["mfe.x"]

    def test_validation_tiebreak_scores_are_not_meta_features(self):
        # val_auc / val_log_loss are a model's validation performance, never a property
        # of the dataset; before they were reserved they entered the screen as features.
        res = self.fold_frame().assign(val_auc=0.8, val_log_loss=0.4)
        assert mr.meta_feature_columns(res) == ["mfe.x"]

    def test_unit_report_forwards_the_selection_mode(self):
        res = self.fold_frame()
        rep = mr.unit_report(res, selection="validation")
        assert rep.selection_mode == "validation" and rep.k == 3
        sel = rep.selection
        assert (sel["classical_arm"].str.endswith("|lr")).all()
        assert (sel["quantum_arm"].str.endswith("|qsvc")).all()
        assert set(zip(sel["Dataset"], sel["embeddings"])) == {
            (d, e) for d in "ab" for e in ("pca", "umap")}
        contrast = mr.loio_contrast(rep)
        assert len(contrast) == 2 * 2 * 6
        assert {"repeat", "fold", "classical_val", "quantum_val"} <= set(contrast.columns)
        assert np.allclose(contrast["qadv"],
                           contrast["quantum_score"] - contrast["classical_score"])

    def test_unit_report_under_validation_matches_select_winners_per_pass(self):
        res = self.fold_frame()
        ours = mr.unit_report(res, selection="validation", validation_col="tuning_score",
                              k=3)
        relabelled = res.assign(Dataset=res["Dataset"] + "|" + res["embeddings"])
        ref = select_winners(relabelled, metric="balanced_accuracy", epsilon=0.027,
                             selection="validation")
        assert np.allclose(ours.per_dataset["se"], ref.per_dataset["se"])
