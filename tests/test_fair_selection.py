"""Tests for :mod:`qbiocode.utils.fair_selection`.

The suite is organised around the defect the module exists to fix rather than around
its public surface, because the defect is not a crash -- ``qml_winner`` runs cleanly,
returns plausible numbers, and is wrong. A test that only checks shapes and dtypes
would have passed against the biased selector too. So the load-bearing tests here are
statistical: they feed in data with a *known* answer and assert the answer comes back.

``test_the_null_does_not_hand_classical_a_sweep`` is the one to keep if the others are
ever dropped. It encodes the measurement that motivated the rewrite: on a pure null the
old rule reported mean delta_f1 = +0.0741 and gave classical the win on 84/84 datasets.

Nothing here touches the corpus. Every fixture is synthetic and seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbiocode.utils.fair_selection import (
    PARAMETER_COLUMNS,
    QUANTUM_STEMS,
    WinnerReport,
    arm_iteration_table,
    corpus_inference,
    iteration_floor_half_width,
    model_side,
    select_winners,
)

CLASSICAL = ["lr", "mlp", "svc", "nb", "rf", "dt", "xgb", "catboost"]
QUANTUM = ["qsvc", "pqk", "qnn"]
EMBEDDINGS = ["PCA", "UMAP"]


def synth(
    n_datasets: int = 84,
    iters: int = 5,
    mu: float = 0.75,
    sd: float = 0.04,
    quantum_shift: float = 0.0,
    classical_tuned: bool = True,
    seed: int = 0,
    metric: str = "f1_score",
) -> pd.DataFrame:
    """A ModelResults.csv-shaped frame with a known ground truth.

    ``quantum_shift`` is the *true* effect: 0.0 is a pure null, in which neither side is
    better by construction. ``classical_tuned`` reproduces the live sweep config
    (``grid_search: True``, ``tune_quantum: False``) by giving every classical row a
    distinct parameter string, as Optuna does when it re-searches per resample, while
    quantum rows share a constant one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_datasets):
        for it in range(iters):
            for emb in EMBEDDINGS:
                for m in CLASSICAL:
                    rows.append({
                        "Dataset": f"ds{d:03d}", "embeddings": emb, "model": m,
                        "iteration": it, metric: rng.normal(mu, sd),
                        "Model_Parameters": (
                            f"C={rng.random():.8f}" if classical_tuned else "default"
                        ),
                    })
                for m in QUANTUM:
                    rows.append({
                        "Dataset": f"ds{d:03d}", "embeddings": emb, "model": m,
                        "iteration": it, metric: rng.normal(mu + quantum_shift, sd),
                        "Model_Parameters": "default",
                    })
    return pd.DataFrame(rows)


class TestSideClassification:
    """Which arm counts as quantum -- an error here moves an arm to the wrong side."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("qsvc", "quantum"), ("qnn", "quantum"), ("vqc", "quantum"),
            ("pqk", "quantum"), ("qpl", "quantum"),
            ("qsvc_opt", "quantum"),
            ("qpl_rf", "quantum"), ("qpl_catboost", "quantum"), ("qpl_opt_rf", "quantum"),
            ("qensemble", "quantum"), ("QEnsemble", "quantum"),
            ("lr", "classical"), ("rf", "classical"), ("catboost", "classical"),
            ("tabpfn", "classical"), ("nb", "classical"), ("xgb_opt", "classical"),
        ],
    )
    def test_labels_qprofiler_actually_writes(self, label, expected):
        assert model_side(label) == expected

    def test_qpl_heads_are_quantum_not_classical(self):
        """``qpl_rf`` is a quantum arm even though ``rf`` appears in its name.

        ``compute_qpl`` fans out one column per classical head, so a rule that searched
        for a classical name *anywhere* in the label would move all six QPL arms onto
        the classical side -- inflating that side's arm count while deleting the quantum
        method with the most arms. Matching the leading ``_``-token is what prevents it.
        """
        for head in ["lr", "rf", "catboost", "xgb", "mlp", "svc"]:
            assert model_side(f"qpl_{head}") == "quantum"

    def test_qensemble_is_registered(self):
        """The hardcoded allowlist omitted it, so a quantum learner read as classical.

        That single omission pushes twice in the same direction: it removes an arm from
        the quantum side and adds one to the classical side, so it both suppresses a
        quantum win and inflates the classical maximum.
        """
        assert "qensemble" in QUANTUM_STEMS
        assert model_side("qensemble") == "quantum"


class TestNoParameterFragmentation:
    """Parameters are payload, never a grouping key. This is bug #1, the dominant one."""

    def test_a_tuned_arm_is_not_shattered_into_singletons(self):
        """One row per (arm, iteration) whether or not parameters vary.

        ``qml_winner`` grouped on the parameter string, so a tuned learner -- whose
        string differs every resample -- produced ``iters`` singleton groups instead of
        one averaged group. Its "mean across splits" silently became "no averaging at
        all", leaving the tuned side at ``sqrt(iters)`` times the noise of the untuned
        side entering the same ``max()``.
        """
        df = synth(n_datasets=3, iters=5, classical_tuned=True, seed=1)
        table = arm_iteration_table(df)
        expected = 3 * (len(CLASSICAL) + len(QUANTUM)) * len(EMBEDDINGS) * 5
        assert len(table) == expected
        per_arm = table.groupby(["Dataset", "arm"]).size()
        assert per_arm.eq(5).all(), "an arm should appear exactly once per iteration"

    def test_tuned_and_untuned_tables_have_identical_shape(self):
        """Whether a side was tuned must not change the shape of its evidence."""
        shapes = {
            tuned: arm_iteration_table(
                synth(n_datasets=3, classical_tuned=tuned, seed=2)
            ).shape
            for tuned in (True, False)
        }
        assert shapes[True] == shapes[False]

    @pytest.mark.parametrize("column", PARAMETER_COLUMNS)
    def test_a_null_parameter_value_does_not_delete_the_arm(self, column):
        """``groupby`` drops null keys by default; a model reporting no parameters wrote
        ``None``, which round-trips through CSV as ``NaN``. Every such arm vanished."""
        df = synth(n_datasets=2, iters=3, seed=3).drop(columns=["Model_Parameters"])
        df[column] = np.where(df["model"].isin(QUANTUM), None, "C=1.0")
        table = arm_iteration_table(df)
        assert set(table.loc[table["side"] == "quantum", "model"]) == set(QUANTUM), (
            "quantum arms with a null parameter value were dropped from the table"
        )

    def test_duplicate_rows_are_averaged_with_a_warning(self):
        """A resumed run re-appends to ModelResults.csv. Keeping the first row silently
        would make the verdict depend on row order, so they are reduced and announced."""
        df = synth(n_datasets=1, iters=3, seed=4)
        doubled = pd.concat([df, df], ignore_index=True)
        with pytest.warns(UserWarning, match="duplicate"):
            table = arm_iteration_table(doubled)
        assert len(table) == len(arm_iteration_table(df))


class TestUnbiasedUnderTheNull:
    """The regression that motivated the module."""

    def test_the_null_does_not_hand_classical_a_sweep(self):
        """Pure null, live config: neither side may sweep the corpus.

        Measured against the old rule on this exact fixture: mean delta_f1 = +0.0741
        (2.7x epsilon) and classical "won" 84/84. Two stacked one-directional biases
        caused it -- tuned-vs-untuned fragmentation (+0.0741) and 16-vs-6 arm count
        (+0.0093) -- and leave-one-iteration-out selection removes both, because the
        iteration that picks an arm is never the iteration that scores it.
        """
        report = select_winners(synth(seed=7), epsilon=0.027, test_size=0.3, seed=0)
        delta = report.per_dataset["delta"]

        assert abs(delta.mean()) < 0.027, (
            f"mean delta {delta.mean():+.4f} exceeds epsilon under a pure null; "
            f"the old pooled-argmax rule scored +0.0741 here"
        )
        n = report.per_dataset["Dataset"].nunique()
        assert len(report.classical_datasets) < n, "classical swept a null corpus"
        assert len(report.quantum_datasets) < n, "quantum swept a null corpus"
        # A two-sided 5% rule may fire on a few of 84 datasets by chance; a *sweep* is
        # the failure mode being guarded, so the bound is deliberately loose.
        claimed = len(report.classical_datasets) + len(report.quantum_datasets)
        assert claimed <= 0.15 * n, f"{claimed}/{n} datasets claimed a win under a null"

    def test_the_corpus_verdict_on_a_null_is_no_difference(self):
        report = select_winners(synth(seed=7), epsilon=0.027, seed=0)
        assert report.corpus["across_datasets"]["direction"] == "no_detectable_difference"

    def test_neither_side_is_favoured_by_having_more_arms(self):
        """Selection must not reward arm count under the null.

        With 8 classical against 3 quantum learners, ``E[max of 16] > E[max of 6]``
        biases any pooled argmax toward classical. Nested selection is what frees the
        method list to be chosen on scientific grounds instead of padded for balance.
        """
        deltas = [
            select_winners(synth(n_datasets=20, seed=s), seed=s).per_dataset["delta"].mean()
            for s in range(6)
        ]
        assert abs(float(np.mean(deltas))) < 0.015, (
            f"mean delta across seeds {np.mean(deltas):+.4f} shows a residual arm-count bias"
        )

    def test_the_naive_column_still_exhibits_the_bias_it_diagnoses(self):
        """``naive_delta`` is kept so a paper can *show* the winner's curse.

        It reproduces the old pooled-argmax rule with the fragmentation bug removed, so
        it must remain measurably more classical-leaning than the LOIO estimate -- that
        residual gap is the arm-count term alone.
        """
        report = select_winners(synth(seed=7), seed=0)
        assert report.corpus["mean_naive_delta"] > report.per_dataset["delta"].mean()


class TestItStillDetectsRealEffects:
    """Unbiased is worthless if it is also blind."""

    def test_a_real_quantum_advantage_is_reported_as_a_quantum_win(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027, seed=0
        )
        assert report.per_dataset["delta"].mean() < -0.10
        assert len(report.quantum_datasets) >= 15, (
            "a +0.15 true quantum effect was certified on fewer than half the datasets"
        )
        assert not report.classical_datasets

    def test_a_real_classical_advantage_reports_the_other_sign(self):
        """Sign convention, per the benchmark's own definition: delta = classical -
        quantum, so delta > 0 is classical ahead and delta < 0 is quantum ahead."""
        report = select_winners(
            synth(n_datasets=30, quantum_shift=-0.15, seed=12), epsilon=0.027, seed=0
        )
        assert report.per_dataset["delta"].mean() > +0.10
        assert len(report.classical_datasets) >= 15
        assert not report.quantum_datasets

    def test_the_corpus_test_agrees_with_the_per_dataset_verdicts(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027, seed=0
        )
        across = report.corpus["across_datasets"]
        assert across["direction"] == "quantum"
        assert across["sign"]["datasets_favoring_quantum"] > \
            across["sign"]["datasets_favoring_classical"]
        assert across["wilcoxon"]["p_two_sided"] < 0.05


class TestEpsilonSemantics:
    """``abs(delta) > epsilon``, applied to the interval rather than the point."""

    def test_a_win_needs_the_whole_interval_to_clear_the_margin(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027, seed=0
        )
        won = report.per_dataset[report.per_dataset["verdict_raw"] == "quantum_wins"]
        assert not won.empty
        assert (won["ci_hi"] < -0.027).all(), (
            "a dataset was called a quantum win although its interval reached above "
            "-epsilon; that is a point-estimate claim, not an unequivocal one"
        )

    def test_raising_epsilon_can_only_remove_wins(self):
        """Monotonicity. A larger margin is a strictly stronger demand."""
        df = synth(n_datasets=30, quantum_shift=+0.15, seed=11)
        strict = select_winners(df, epsilon=0.20, seed=0).quantum_datasets
        loose = select_winners(df, epsilon=0.027, seed=0).quantum_datasets
        assert set(strict).issubset(set(loose))
        assert len(strict) <= len(loose)

    def test_verdicts_come_from_the_documented_vocabulary(self):
        report = select_winners(synth(n_datasets=10, seed=13), seed=0)
        from qbiocode.utils.fair_selection import VERDICTS
        assert set(report.per_dataset["verdict_raw"]) <= set(VERDICTS)
        assert set(report.per_dataset["verdict_adjusted"]) <= set(VERDICTS)

    def test_an_indistinguishable_pair_is_labelled_equivalent_not_a_win(self):
        """Tight, genuinely-equal arms should read as equivalence.

        With sd shrunk far below epsilon the interval fits inside the margin, which is
        the TOST-shaped conclusion "the same to within epsilon" -- a positive finding,
        and distinct from "inconclusive".
        """
        report = select_winners(
            synth(n_datasets=20, iters=5, sd=0.001, seed=14), epsilon=0.027, seed=0
        )
        counts = report.per_dataset["verdict_raw"].value_counts()
        assert counts.get("equivalent", 0) >= 10, dict(counts)


class TestMultiplicity:
    def test_fdr_demotes_but_never_promotes(self):
        """BH may only weaken a raw claim. It cannot manufacture one."""
        report = select_winners(
            synth(n_datasets=40, quantum_shift=+0.08, seed=15), epsilon=0.027, seed=0
        )
        per = report.per_dataset
        promoted = per[
            per["verdict_raw"].eq("inconclusive")
            & per["verdict_adjusted"].isin(["quantum_wins", "classical_wins"])
        ]
        assert promoted.empty, "FDR adjustment invented a win"

    def test_adjusted_p_is_never_below_raw_p(self):
        report = select_winners(
            synth(n_datasets=40, quantum_shift=+0.08, seed=15), epsilon=0.027, seed=0
        )
        per = report.per_dataset.dropna(subset=["p_adjusted"])
        assert not per.empty
        assert (per["p_adjusted"] >= per["p_margin"] - 1e-12).all()


class TestNadeauBengioFloor:
    """The protocol, not the analysis, sets the smallest certifiable margin."""

    def test_more_iterations_cannot_cross_the_floor(self):
        """``SE = s*sqrt(1/I + n_test/n_train)`` tends to ``s*sqrt(r)``, not to 0.

        This is why epsilon cannot be bought with compute: at ``test_size=0.3`` the
        half-width floors at ``1.28*s``, so at the measured ``s=0.0267`` no ``iter``
        reaches ``epsilon=0.027``.
        """
        floor = iteration_floor_half_width(0.0267, 0.3)
        assert floor == pytest.approx(0.0343, abs=5e-4)
        assert floor > 0.027, "the floor must exceed epsilon at test_size=0.3"

    def test_lowering_test_size_is_what_makes_epsilon_reachable(self):
        assert iteration_floor_half_width(0.0267, 0.21) <= 0.027
        assert iteration_floor_half_width(0.0267, 0.15) < \
            iteration_floor_half_width(0.0267, 0.30)

    def test_the_report_says_when_epsilon_is_unreachable(self):
        report = select_winners(synth(n_datasets=20, seed=16), epsilon=1e-6, seed=0)
        assert report.corpus["epsilon_is_reachable"] is False


class TestDegenerateInput:
    def test_a_missing_iteration_column_is_refused_by_name(self):
        """Without a resample axis there is nothing to hold out and no interval."""
        df = synth(n_datasets=2, seed=17).drop(columns=["iteration"])
        with pytest.raises(ValueError, match="iteration"):
            select_winners(df)

    def test_a_single_iteration_is_named_not_silently_inconclusive(self):
        """One iteration cannot give a spread, and saying so keeps such datasets out of
        the corpus counts instead of hiding them among genuinely ambiguous ones."""
        report = select_winners(synth(n_datasets=3, iters=1, seed=18), seed=0)
        assert (report.per_dataset["verdict_raw"] == "insufficient_iterations").all()
        assert not report.quantum_datasets and not report.classical_datasets

    def test_a_dataset_with_only_one_side_yields_no_win(self):
        df = synth(n_datasets=3, seed=19)
        classical_only = df[df["model"].isin(CLASSICAL)]
        report = select_winners(classical_only, seed=0)
        assert not report.quantum_datasets and not report.classical_datasets
        assert report.per_dataset["delta"].isna().all()

    def test_non_numeric_metric_cells_are_dropped_not_crashed_on(self):
        df = synth(n_datasets=3, iters=4, seed=20)
        # cast first: assigning a string into a float64 column is a pandas FutureWarning,
        # and the point of the test is the selector's tolerance, not pandas' dtype rules
        df["f1_score"] = df["f1_score"].astype(object)
        df.loc[df.index[:20], "f1_score"] = "failed"
        report = select_winners(df, seed=0)
        assert len(report.per_dataset) == 3

    def test_the_report_is_always_truthy(self):
        """``qml_winner`` returned a bare ``None`` when no quantum dataset was found, so
        "no winner" and "the function changed shape" were the same observation and no
        test could distinguish them."""
        report = select_winners(synth(n_datasets=3, iters=1, seed=21), seed=0)
        assert bool(report) is True
        assert isinstance(report, WinnerReport)
        assert report.quantum_datasets == []


class TestDeterminismAndTieBreaking:
    def test_the_same_seed_gives_the_same_verdicts(self):
        df = synth(n_datasets=10, seed=22)
        a = select_winners(df, seed=5).per_dataset
        b = select_winners(df, seed=5).per_dataset
        pd.testing.assert_frame_equal(a, b)

    def test_exact_ties_do_not_resolve_alphabetically(self):
        """With every arm identical, ``argmax`` picks the first row -- which after a
        ``groupby`` sort means the alphabetically first model name, so ``catboost``
        beats ``qsvc`` and ``nb`` beats ``qnn`` on every tie. Weighted F1 on ``m`` test
        rows moves in steps of about ``1/m``, so exact ties are common, not exotic.
        """
        rows = []
        for d in range(4):
            for it in range(4):
                for m in CLASSICAL + QUANTUM:
                    rows.append({
                        "Dataset": f"ds{d}", "embeddings": "PCA", "model": m,
                        "iteration": it, "f1_score": 0.8, "Model_Parameters": "x",
                    })
        df = pd.DataFrame(rows)
        picked = {
            seed: set(
                select_winners(df, seed=seed).selection["classical_arm"].unique()
            )
            for seed in range(8)
        }
        chosen = set().union(*picked.values())
        assert len(chosen) > 1, (
            f"every seed chose the same arm on an exact tie: {chosen}"
        )

    def test_a_perfect_tie_is_never_a_win_for_either_side(self):
        rows = []
        for d in range(4):
            for it in range(4):
                for m in CLASSICAL + QUANTUM:
                    rows.append({
                        "Dataset": f"ds{d}", "embeddings": "PCA", "model": m,
                        "iteration": it, "f1_score": 0.8, "Model_Parameters": "x",
                    })
        report = select_winners(pd.DataFrame(rows), seed=0)
        assert not report.quantum_datasets and not report.classical_datasets


class TestMetricIsParameterised:
    """The comparison must not be welded to weighted F1."""

    @pytest.mark.parametrize("metric", ["balanced_accuracy", "mcc", "pr_auc", "accuracy"])
    def test_any_higher_is_better_metric_works(self, metric):
        df = synth(n_datasets=8, quantum_shift=+0.15, seed=23, metric=metric)
        report = select_winners(df, metric=metric, epsilon=0.027, seed=0)
        assert report.metric == metric
        assert report.per_dataset["delta"].mean() < -0.10

    def test_mcc_can_go_negative_without_breaking_the_interval(self):
        """MCC spans [-1, 1] unlike F1, so nothing may assume a non-negative metric."""
        df = synth(n_datasets=6, mu=-0.1, sd=0.05, seed=24, metric="mcc")
        report = select_winners(df, metric="mcc", seed=0)
        assert report.per_dataset["ci_lo"].notna().all()


class TestDummyFloor:
    def test_a_corpus_that_beats_nothing_is_marked_below_baseline(self):
        """An arm that cannot beat a majority-class dummy is not evidence.

        Weighted F1 hides this: on ``openml__ozone-level-8hr`` (minority fraction
        0.063) a majority-class dummy already reaches 0.906, so two arms can look
        excellent and be indistinguishable from predicting one class.
        """
        df = synth(n_datasets=6, seed=25)
        baseline = pd.DataFrame({
            "Dataset": sorted(df["Dataset"].unique()), "f1_score": 0.99,
        })
        report = select_winners(df, baseline=baseline, seed=0)
        assert (report.per_dataset["verdict_raw"] == "below_baseline").all()
        assert not report.quantum_datasets and not report.classical_datasets

    def test_an_easily_beaten_baseline_changes_nothing(self):
        df = synth(n_datasets=6, seed=25)
        baseline = pd.DataFrame({
            "Dataset": sorted(df["Dataset"].unique()), "f1_score": 0.1,
        })
        with_base = select_winners(df, baseline=baseline, seed=0).per_dataset
        assert "below_baseline" not in set(with_base["verdict_raw"])

    def test_a_baseline_missing_the_metric_column_is_refused(self):
        df = synth(n_datasets=3, seed=26)
        bad = pd.DataFrame({"Dataset": sorted(df["Dataset"].unique()), "score": 0.5})
        with pytest.raises(ValueError, match="baseline"):
            select_winners(df, baseline=bad, seed=0)

    def test_the_baseline_is_joined_on_dataset_not_on_index(self):
        """Index-aligned joins are what caused cross-dataset misattribution in
        ``qml_winner``; a shuffled baseline must still land on the right rows."""
        df = synth(n_datasets=6, seed=27)
        names = sorted(df["Dataset"].unique())
        ordered = pd.DataFrame({"Dataset": names, "f1_score": np.linspace(0.1, 0.99, 6)})
        shuffled = ordered.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = select_winners(df, baseline=ordered, seed=0).per_dataset
        b = select_winners(df, baseline=shuffled, seed=0).per_dataset
        pd.testing.assert_series_equal(
            a.set_index("Dataset")["verdict_raw"].sort_index(),
            b.set_index("Dataset")["verdict_raw"].sort_index(),
        )


class TestCorpusInference:
    def test_all_three_tests_are_reported(self):
        report = select_winners(synth(n_datasets=30, seed=28), seed=0)
        across = report.corpus["across_datasets"]
        for key in ("t", "wilcoxon", "sign"):
            assert key in across, f"{key} missing from corpus inference"
        assert "p_two_sided" in across["t"]

    def test_the_sign_test_counts_match_epsilon(self):
        per = pd.DataFrame({"delta": [0.5, 0.4, -0.5, -0.4, 0.001, -0.001, 0.0]})
        out = corpus_inference(per, epsilon=0.027)
        assert out["sign"]["datasets_favoring_classical"] == 2
        assert out["sign"]["datasets_favoring_quantum"] == 2
        assert out["sign"]["datasets_within_epsilon_0.027"] == 3

    def test_no_direction_is_claimed_when_the_mean_is_inside_epsilon(self):
        """A tiny but consistent difference is statistically detectable and practically
        irrelevant. With N datasets the t-test will find it; epsilon is what stops it
        being reported as a win."""
        per = pd.DataFrame({"delta": [0.005] * 60})
        out = corpus_inference(per, epsilon=0.027)
        assert out["direction"] == "no_detectable_difference"

    def test_a_large_consistent_effect_is_given_a_direction(self):
        per = pd.DataFrame({"delta": list(np.linspace(-0.20, -0.10, 40))})
        out = corpus_inference(per, epsilon=0.027)
        assert out["direction"] == "quantum"

    def test_too_few_datasets_yields_no_inference_rather_than_a_fake_one(self):
        assert corpus_inference(pd.DataFrame({"delta": [0.1]}))["n_datasets_used"] == 1
        assert "ci" not in corpus_inference(pd.DataFrame({"delta": [0.1]}))
        assert corpus_inference(pd.DataFrame()) == {}


class TestOutputTables:
    def test_the_selection_trace_names_the_arm_used_for_each_held_out_iteration(self):
        """Auditability: the reader must be able to see *which* arm produced each score,
        because "the best arm" under nested selection is not one arm."""
        report = select_winners(synth(n_datasets=4, iters=5, seed=29), seed=0)
        sel = report.selection
        assert len(sel) == 4 * 5
        assert {"Dataset", "iteration", "classical_arm", "quantum_arm"} <= set(sel.columns)
        assert sel["quantum_arm"].str.contains("|").any()

    def test_per_arm_reports_how_many_distinct_parameter_strings_an_arm_had(self):
        """The fragmentation diagnostic: >1 means that arm was re-tuned per resample,
        which is exactly the condition under which the old rule broke."""
        report = select_winners(synth(n_datasets=3, iters=5, classical_tuned=True, seed=30),
                                seed=0)
        per_arm = report.per_arm
        tuned = per_arm[per_arm["side"] == "classical"]["n_distinct_parameters"]
        untuned = per_arm[per_arm["side"] == "quantum"]["n_distinct_parameters"]
        assert (tuned == 5).all(), "tuned arms should show one parameter string per resample"
        assert (untuned == 1).all()

    def test_every_dataset_appears_exactly_once_in_the_verdict_table(self):
        df = synth(n_datasets=12, seed=31)
        report = select_winners(df, seed=0)
        assert report.per_dataset["Dataset"].is_unique
        assert set(report.per_dataset["Dataset"]) == set(df["Dataset"])
