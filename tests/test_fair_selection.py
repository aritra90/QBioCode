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
from scipy import stats

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
from qbiocode.utils.fair_selection import TIE_SEPARATOR, _loio_side_scores

CLASSICAL = ["lr", "mlp", "svc", "nb", "rf", "dt", "xgb", "catboost"]
QUANTUM = ["qsvc", "pqk", "qnn"]
EMBEDDINGS = ["PCA", "UMAP"]
#: The split fraction every call passes explicitly. select_winners has no default for it
#: (the NB correction depends on it); 0.3 is what these fixtures were calibrated at.
TS = 0.3


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


def paired_frame(mus, spread: float = 0.01, iters: int = 5) -> pd.DataFrame:
    """One classical and one quantum arm per dataset, with the deltas set exactly.

    With a single arm per side LOIO has nothing to choose, so ``delta_i`` is
    ``mu + spread * z_i`` for a fixed centred ``z`` and every p-value is known in
    advance. Dataset ``k`` is named ``ds{k}``.
    """
    z = np.arange(iters) - (iters - 1) / 2.0
    rows = []
    for k, mu in enumerate(mus):
        for it in range(iters):
            for model, score in (("lr", 0.7 + mu + spread * z[it]), ("qsvc", 0.7)):
                rows.append({"Dataset": f"ds{k}", "embeddings": "PCA", "model": model,
                             "iteration": it, "f1_score": score,
                             "Model_Parameters": "x"})
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

    def test_duplicate_rows_are_averaged_with_a_warning(self, caplog):
        """A resumed run re-appends to ModelResults.csv. Keeping the first row silently
        would make the verdict depend on row order, so they are reduced and announced.

        Announced by logging, not warnings.warn: warnings are silenced in any process
        that has imported matplotlib.pyplot, which qc_winner_finder does."""
        df = synth(n_datasets=1, iters=3, seed=4)
        doubled = pd.concat([df, df], ignore_index=True)
        with caplog.at_level("WARNING", logger="qbiocode.utils.fair_selection"):
            table = arm_iteration_table(doubled)
        assert "duplicate" in caplog.text
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
        report = select_winners(synth(seed=7), epsilon=0.027, test_size=TS, seed=0)
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
        report = select_winners(synth(seed=7), epsilon=0.027, test_size=TS, seed=0)
        assert report.corpus["across_datasets"]["direction"] == "no_detectable_difference"

    def test_neither_side_is_favoured_by_having_more_arms(self):
        """Selection must not reward arm count under the null.

        With 8 classical against 3 quantum learners, ``E[max of 16] > E[max of 6]``
        biases any pooled argmax toward classical. Nested selection is what frees the
        method list to be chosen on scientific grounds instead of padded for balance.
        """
        deltas = [
            select_winners(synth(n_datasets=20, seed=s), test_size=TS, seed=s)
            .per_dataset["delta"].mean()
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
        report = select_winners(synth(seed=7), test_size=TS, seed=0)
        assert report.corpus["mean_naive_delta"] > report.per_dataset["delta"].mean()


class TestItStillDetectsRealEffects:
    """Unbiased is worthless if it is also blind."""

    def test_a_real_quantum_advantage_is_reported_as_a_quantum_win(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027,
            test_size=TS, seed=0,
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
            synth(n_datasets=30, quantum_shift=-0.15, seed=12), epsilon=0.027,
            test_size=TS, seed=0,
        )
        assert report.per_dataset["delta"].mean() > +0.10
        assert len(report.classical_datasets) >= 15
        assert not report.quantum_datasets

    def test_the_corpus_test_agrees_with_the_per_dataset_verdicts(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027,
            test_size=TS, seed=0,
        )
        across = report.corpus["across_datasets"]
        assert across["direction"] == "quantum"
        assert across["sign"]["datasets_favoring_quantum"] > \
            across["sign"]["datasets_favoring_classical"]
        assert across["wilcoxon"]["p_two_sided"] < 0.05


class TestEpsilonSemantics:
    """``margin`` decides wins, ``epsilon`` decides equivalence -- no longer one number.

    Semantics changed: epsilon used to be both the win margin and the equivalence
    bound, so a real effect had to clear the resolution floor twice. A win is now judged
    against the pre-registered ``margin`` (default 0), and epsilon is only the TOST bound.
    """

    def test_a_win_needs_the_whole_interval_to_clear_the_margin(self):
        report = select_winners(
            synth(n_datasets=30, quantum_shift=+0.15, seed=11), epsilon=0.027,
            margin=0.027, test_size=TS, seed=0,
        )
        won = report.per_dataset[report.per_dataset["verdict_raw"] == "quantum_wins"]
        assert not won.empty
        assert (won["ci_hi"] < -0.027).all(), (
            "a dataset was called a quantum win although its interval reached above "
            "-margin; that is a point-estimate claim, not an unequivocal one"
        )

    def test_raising_the_margin_can_only_remove_wins(self):
        """Monotonicity. A larger margin is a strictly stronger demand."""
        df = synth(n_datasets=30, quantum_shift=+0.15, seed=11)
        strict = select_winners(df, margin=0.20, test_size=TS).quantum_datasets
        loose = select_winners(df, margin=0.027, test_size=TS).quantum_datasets
        zero = select_winners(df, margin=0.0, test_size=TS).quantum_datasets
        assert set(strict) <= set(loose) <= set(zero)
        assert len(strict) < len(zero)

    def test_epsilon_no_longer_moves_the_win_verdicts(self):
        """Changed semantics: epsilon only relabels non-wins as equivalent or not."""
        df = synth(n_datasets=30, quantum_shift=+0.15, seed=11)
        big = select_winners(df, epsilon=0.20, test_size=TS).per_dataset
        small = select_winners(df, epsilon=0.027, test_size=TS).per_dataset
        wins = ["quantum_wins", "classical_wins"]
        pd.testing.assert_series_equal(big["p_value"], small["p_value"])
        assert (big["verdict_raw"].isin(wins) == small["verdict_raw"].isin(wins)).all()
        assert big["within_equivalence"].sum() >= small["within_equivalence"].sum()

    def test_verdicts_come_from_the_documented_vocabulary(self):
        report = select_winners(synth(n_datasets=10, seed=13), test_size=TS, seed=0)
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
            synth(n_datasets=20, iters=5, sd=0.001, seed=14), epsilon=0.027,
            test_size=TS, seed=0,
        )
        counts = report.per_dataset["verdict_raw"].value_counts()
        assert counts.get("equivalent", 0) >= 10, dict(counts)


class TestMultiplicity:
    def test_fdr_at_or_below_alpha_demotes_but_never_promotes(self):
        """With ``fdr <= alpha`` BH may only weaken a raw claim.

        Semantics changed: the BH family is now every dataset, so with ``fdr > alpha``
        a raw-inconclusive dataset may legitimately be confirmed (next test).
        """
        report = select_winners(
            synth(n_datasets=40, quantum_shift=+0.08, seed=15), epsilon=0.027,
            fdr=0.05, alpha=0.05, test_size=TS,
        )
        per = report.per_dataset
        promoted = per[
            ~per["verdict_raw"].isin(["quantum_wins", "classical_wins"])
            & per["verdict_adjusted"].isin(["quantum_wins", "classical_wins"])
        ]
        assert promoted.empty, "FDR adjustment invented a win"

    def test_fdr_above_alpha_may_confirm_a_raw_non_win(self):
        """BH at 0.25 over many true effects confirms datasets whose raw p is in
        (alpha, BH threshold] -- the FDR procedure working, not a promotion bug."""
        report = select_winners(
            synth(n_datasets=40, quantum_shift=+0.08, seed=15), fdr=0.25, alpha=0.05,
            test_size=TS,
        )
        per = report.per_dataset
        confirmed = per[~per["verdict_raw"].isin(["quantum_wins", "classical_wins"])
                        & per["verdict_adjusted"].eq("quantum_wins")]
        assert not confirmed.empty
        assert (confirmed["p_value"] > 0.05).all()
        assert (confirmed["p_adjusted"] <= 0.25).all()

    def test_adjusted_p_is_never_below_raw_p(self):
        report = select_winners(
            synth(n_datasets=40, quantum_shift=+0.08, seed=15), epsilon=0.027,
            test_size=TS,
        )
        per = report.per_dataset.dropna(subset=["p_adjusted"])
        assert not per.empty
        assert (per["p_adjusted"] >= per["p_value"] - 1e-12).all()

    def test_every_dataset_with_a_p_value_enters_the_bh_family(self):
        """Not only the raw wins: the family is fixed before looking at verdicts."""
        per = select_winners(synth(n_datasets=12, seed=13), test_size=TS).per_dataset
        assert per["p_adjusted"].notna().sum() == per["p_value"].notna().sum() == 12
        assert (per["family"] == "discovery").all()

    def test_all_dataset_bh_demotes_what_claimed_only_bh_kept(self):
        """The old scope adjusted only raw wins, so with one claim m = 1 and BH was
        inert. Over all 10 datasets the same claim no longer survives fdr=0.10."""
        mus = [0.042] + [0.004] * 9
        df = paired_frame(mus)
        per = select_winners(df, test_size=0.2, fdr=0.10).per_dataset.set_index("Dataset")
        assert per.loc["ds0", "verdict_raw"] == "classical_wins"
        assert (per.drop(index="ds0")["verdict_raw"] != "classical_wins").all()
        p0 = per.loc["ds0", "p_value"]
        assert p0 < 0.05
        # claimed-only scope: m = 1, q = p0 <= 0.10 -> the win would have stood
        assert p0 <= 0.10
        # all-dataset scope: q = min_k p_(k) * 10 / k > 0.10 -> demoted
        assert per.loc["ds0", "p_adjusted"] > 0.10
        assert per.loc["ds0", "verdict_adjusted"] == "inconclusive"

    def test_bh_q_values_match_a_reference_implementation(self):
        per = select_winners(synth(n_datasets=15, quantum_shift=0.05, seed=3),
                             test_size=TS).per_dataset
        p = per["p_value"].to_numpy()
        m = p.size
        ref = np.array([min(1.0, min(np.sort(p)[k:] * m / np.arange(k + 1, m + 1)))
                        for k in range(m)])
        np.testing.assert_allclose(np.sort(per["p_adjusted"].to_numpy()), ref)


class TestMargin:
    def test_margin_zero_is_the_ordinary_two_sided_t_test(self):
        from scipy import stats
        per = select_winners(paired_frame([0.03, -0.03, 0.0]), test_size=0.2).per_dataset
        df_ = per["n_iterations"] - 1
        ref = 2 * stats.t.sf(per["delta"].abs() / per["se"], df_)
        np.testing.assert_allclose(per["p_value"], np.minimum(ref, 1.0))
        # p < alpha exactly when the interval excludes zero
        excludes = (per["ci_lo"] > 0) | (per["ci_hi"] < 0)
        assert ((per["p_value"] < 0.05) == excludes).all()

    def test_a_positive_margin_raises_p_and_can_remove_a_win(self):
        df = paired_frame([0.04])
        zero = select_winners(df, test_size=0.2, margin=0.0).per_dataset.iloc[0]
        wide = select_winners(df, test_size=0.2, margin=0.03).per_dataset.iloc[0]
        assert zero["verdict_raw"] == "classical_wins"
        assert wide["verdict_raw"] != "classical_wins"
        assert wide["p_value"] > zero["p_value"]
        assert wide["ci_lo"] == zero["ci_lo"]  # the interval itself does not move

    def test_a_negative_margin_is_refused(self):
        with pytest.raises(ValueError, match="margin"):
            select_winners(paired_frame([0.04]), test_size=0.2, margin=-0.01)

    @pytest.mark.parametrize("fdr", [0.0, 1.0, 5.0, -0.1])
    def test_an_fdr_outside_the_unit_interval_is_refused(self, fdr):
        # fdr >= 1 would turn every finite p (even 0.7) into a directional win
        with pytest.raises(ValueError, match="fdr"):
            select_winners(paired_frame([0.04]), test_size=0.2, fdr=fdr)

    def test_within_equivalence_is_reported_alongside_a_win(self):
        """A significant but negligible win stays a win, flagged as inside +/-epsilon."""
        per = select_winners(paired_frame([0.01], spread=0.0005), test_size=0.2,
                             epsilon=0.027).per_dataset.iloc[0]
        assert per["verdict_raw"] == "classical_wins"
        assert bool(per["within_equivalence"]) is True
        assert per["ci_lo"] > 0 and per["ci_hi"] < 0.027


class TestControls:
    def test_controls_form_a_separate_holm_family(self):
        mus = [0.042] + [0.004] * 9 + [0.06, 0.0]
        df = paired_frame(mus)
        controls = ["ds10", "ds11"]
        rep_ = select_winners(df, test_size=0.2, controls=controls)
        per = rep_.per_dataset.set_index("Dataset")
        assert set(per.index[per["family"] == "control"]) == set(controls)
        assert (per.drop(index=controls)["family"] == "discovery").all()
        # Holm over the two controls only
        pc = per.loc[controls, "p_value"].to_numpy()
        lo, hi = np.argsort(pc)
        holm = np.empty(2)
        holm[lo] = min(1.0, 2 * pc[lo])
        holm[hi] = min(1.0, max(holm[lo], pc[hi]))
        np.testing.assert_allclose(per.loc[controls, "p_adjusted"], holm)
        # the discovery family's BH is untouched by the controls
        alone = select_winners(paired_frame(mus[:10]), test_size=0.2).per_dataset
        np.testing.assert_allclose(per.drop(index=controls)["p_adjusted"],
                                   alone["p_adjusted"])
        # the planted effect is judged at alpha and kept out of the headline lists
        assert per.loc["ds10", "verdict_adjusted"] == "classical_wins"
        assert "ds10" not in rep_.classical_datasets
        assert rep_.controls == ("ds10", "ds11")
        assert rep_.corpus["across_datasets"]["n_datasets_used"] == 10

    def test_an_unknown_control_name_is_refused(self):
        with pytest.raises(ValueError, match="nope"):
            select_winners(paired_frame([0.01, 0.02]), test_size=0.2,
                           controls=["ds0", "nope"])

    def test_a_control_with_no_finite_metric_is_not_called_absent(self):
        df = paired_frame([0.01, 0.02])
        df.loc[df["Dataset"] == "ds1", "f1_score"] = np.nan
        with pytest.raises(ValueError, match="no finite 'f1_score'") as err:
            select_winners(df, test_size=0.2, controls=["ds1"])
        assert "absent" not in str(err.value)


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

    def test_the_report_says_when_epsilon_is_unreachable(self, caplog):
        """epsilon is now only the equivalence bound, and the message says so."""
        with caplog.at_level("WARNING", logger="qbiocode.utils.fair_selection"):
            report = select_winners(synth(n_datasets=20, seed=16), epsilon=1e-6,
                                    test_size=TS)
        assert report.corpus["epsilon_is_reachable"] is False
        assert "equivalence cannot be certified" in caplog.text


class TestDegenerateInput:
    def test_test_size_has_no_default(self):
        """The NB correction r = test/(1-test) depends on the split, which
        ModelResults.csv does not record; the old silent 0.3 default mis-corrected the
        pilot, which used 0.2."""
        df = synth(n_datasets=2, iters=3, seed=17)
        with pytest.raises(ValueError, match="split fraction"):
            select_winners(df)
        with pytest.raises(ValueError, match="test_size"):
            select_winners(df, test_size=None)
        with pytest.raises(ValueError, match="test_size"):
            select_winners(df, test_size=1.0)

    def test_a_missing_iteration_column_is_refused_by_name(self):
        """Without a resample axis there is nothing to hold out and no interval."""
        df = synth(n_datasets=2, seed=17).drop(columns=["iteration"])
        with pytest.raises(ValueError, match="iteration"):
            select_winners(df, test_size=TS)

    def test_a_single_iteration_is_named_not_silently_inconclusive(self):
        """One iteration cannot give a spread, and saying so keeps such datasets out of
        the corpus counts instead of hiding them among genuinely ambiguous ones."""
        report = select_winners(synth(n_datasets=3, iters=1, seed=18), test_size=TS, seed=0)
        assert (report.per_dataset["verdict_raw"] == "insufficient_iterations").all()
        assert not report.quantum_datasets and not report.classical_datasets

    def test_a_dataset_with_only_one_side_yields_no_win(self):
        df = synth(n_datasets=3, seed=19)
        classical_only = df[df["model"].isin(CLASSICAL)]
        report = select_winners(classical_only, test_size=TS, seed=0)
        assert not report.quantum_datasets and not report.classical_datasets
        assert report.per_dataset["delta"].isna().all()

    def test_non_numeric_metric_cells_are_dropped_not_crashed_on(self):
        df = synth(n_datasets=3, iters=4, seed=20)
        # cast first: assigning a string into a float64 column is a pandas FutureWarning,
        # and the point of the test is the selector's tolerance, not pandas' dtype rules
        df["f1_score"] = df["f1_score"].astype(object)
        df.loc[df.index[:20], "f1_score"] = "failed"
        report = select_winners(df, test_size=TS, seed=0)
        assert len(report.per_dataset) == 3

    def test_the_report_is_always_truthy(self):
        """``qml_winner`` returned a bare ``None`` when no quantum dataset was found, so
        "no winner" and "the function changed shape" were the same observation and no
        test could distinguish them."""
        report = select_winners(synth(n_datasets=3, iters=1, seed=21), test_size=TS, seed=0)
        assert bool(report) is True
        assert isinstance(report, WinnerReport)
        assert report.quantum_datasets == []


def _all_tied_frame() -> pd.DataFrame:
    rows = []
    for d in range(4):
        for it in range(4):
            for m in CLASSICAL + QUANTUM:
                rows.append({
                    "Dataset": f"ds{d}", "embeddings": "PCA", "model": m,
                    "iteration": it, "f1_score": 0.8, "Model_Parameters": "x",
                })
    return pd.DataFrame(rows)


class TestDeterminismAndTieBreaking:
    def test_results_do_not_depend_on_the_seed(self):
        """Semantics changed: ties used to be broken by a seeded permutation, so only
        the *same* seed reproduced a verdict. ``seed`` is now unused and any two seeds
        must agree exactly."""
        df = synth(n_datasets=10, seed=22)
        a = select_winners(df, test_size=TS, seed=5)
        b = select_winners(df, test_size=TS, seed=123)
        pd.testing.assert_frame_equal(a.per_dataset, b.per_dataset)
        pd.testing.assert_frame_equal(a.selection, b.selection)

    def test_exact_ties_do_not_resolve_alphabetically(self):
        """With every arm identical, ``argmax`` picks the first row -- which after a
        ``groupby`` sort means the alphabetically first model name, so ``catboost``
        beats ``qsvc`` and ``nb`` beats ``qnn`` on every tie. Weighted F1 on ``m`` test
        rows moves in steps of about ``1/m``, so exact ties are common, not exotic.

        Semantics changed: the tie is no longer broken at random. Every tied arm is
        named in the trace and their held-out scores are averaged.
        """
        report = select_winners(_all_tied_frame(), test_size=TS)
        want_c = TIE_SEPARATOR.join(sorted(f"PCA|{m}" for m in CLASSICAL))
        want_q = TIE_SEPARATOR.join(sorted(f"PCA|{m}" for m in QUANTUM))
        assert set(report.selection["classical_arm"]) == {want_c}
        assert set(report.selection["quantum_arm"]) == {want_q}

    def test_a_perfect_tie_is_never_a_win_for_either_side(self):
        """Also guards the rounding trap: averaging eight identical 0.8s must not leave
        a 1e-16 delta with zero spread that reads as a certain win."""
        report = select_winners(_all_tied_frame(), test_size=TS)
        assert not report.quantum_datasets and not report.classical_datasets
        assert (report.per_dataset["delta"] == 0.0).all()
        assert (report.per_dataset["verdict_raw"] != "quantum_wins").all()
        assert (report.per_dataset["verdict_raw"] != "classical_wins").all()

    def test_tied_arms_score_the_mean_of_their_held_out_scores(self):
        """Holding out iteration 2, both arms average 0.8 on iterations 0-1, so the
        held-out score is the expectation over a fair coin: (0.8 + 0.5) / 2."""
        mat = np.array([[0.9, 0.7, 0.8],
                        [0.7, 0.9, 0.5]])
        scores, chosen = _loio_side_scores(mat, ["PCA|a", "PCA|b"])
        assert scores[2] == pytest.approx(0.65)
        assert chosen[2] == "PCA|a" + TIE_SEPARATOR + "PCA|b"
        # untied folds pick the single best arm as before (a: 0.75 vs 0.70, 0.85 vs 0.60)
        assert chosen[0] == "PCA|a" and scores[0] == pytest.approx(0.9)
        assert chosen[1] == "PCA|a" and scores[1] == pytest.approx(0.7)

    def test_tie_averaging_is_seed_invariant_end_to_end(self):
        mat = {"lr": [0.9, 0.7, 0.8], "rf": [0.7, 0.9, 0.5], "qsvc": [0.75, 0.75, 0.75]}
        df = pd.DataFrame([
            {"Dataset": "ds0", "embeddings": "PCA", "model": m, "iteration": it,
             "f1_score": v[it], "Model_Parameters": "x"}
            for m, v in mat.items() for it in range(3)
        ])
        runs = [select_winners(df, test_size=0.2, seed=s) for s in range(4)]
        for r in runs[1:]:
            pd.testing.assert_frame_equal(runs[0].selection, r.selection)
        sel = runs[0].selection.set_index("iteration")
        assert sel.loc[2, "classical_arm"] == "PCA|lr+PCA|rf"
        assert sel.loc[2, "classical_score"] == pytest.approx(0.65)

    def test_the_tie_separator_cannot_collide_with_an_arm_label(self):
        assert TIE_SEPARATOR != "|"
        table = arm_iteration_table(synth(n_datasets=1, iters=2, seed=1))
        assert not table["arm"].str.contains(TIE_SEPARATOR, regex=False).any()


class TestMetricIsParameterised:
    """The comparison must not be welded to weighted F1."""

    @pytest.mark.parametrize("metric", ["balanced_accuracy", "mcc", "pr_auc", "accuracy"])
    def test_any_higher_is_better_metric_works(self, metric):
        df = synth(n_datasets=8, quantum_shift=+0.15, seed=23, metric=metric)
        report = select_winners(df, metric=metric, epsilon=0.027, test_size=TS, seed=0)
        assert report.metric == metric
        assert report.per_dataset["delta"].mean() < -0.10

    def test_mcc_can_go_negative_without_breaking_the_interval(self):
        """MCC spans [-1, 1] unlike F1, so nothing may assume a non-negative metric."""
        df = synth(n_datasets=6, mu=-0.1, sd=0.05, seed=24, metric="mcc")
        report = select_winners(df, metric="mcc", test_size=TS, seed=0)
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
        report = select_winners(df, baseline=baseline, test_size=TS, seed=0)
        assert (report.per_dataset["verdict_raw"] == "below_baseline").all()
        assert not report.quantum_datasets and not report.classical_datasets
        # Changed: every finite p-value now enters the BH family, below_baseline rows
        # included, but a special verdict is carried through the adjustment unchanged.
        assert (report.per_dataset["verdict_adjusted"] == "below_baseline").all()
        assert report.per_dataset["p_adjusted"].notna().all()

    def test_an_easily_beaten_baseline_changes_nothing(self):
        df = synth(n_datasets=6, seed=25)
        baseline = pd.DataFrame({
            "Dataset": sorted(df["Dataset"].unique()), "f1_score": 0.1,
        })
        with_base = select_winners(df, baseline=baseline, test_size=TS, seed=0).per_dataset
        assert "below_baseline" not in set(with_base["verdict_raw"])

    def test_a_baseline_missing_the_metric_column_is_refused(self):
        df = synth(n_datasets=3, seed=26)
        bad = pd.DataFrame({"Dataset": sorted(df["Dataset"].unique()), "score": 0.5})
        with pytest.raises(ValueError, match="baseline"):
            select_winners(df, baseline=bad, test_size=TS, seed=0)

    def test_the_baseline_is_joined_on_dataset_not_on_index(self):
        """Index-aligned joins are what caused cross-dataset misattribution in
        ``qml_winner``; a shuffled baseline must still land on the right rows."""
        df = synth(n_datasets=6, seed=27)
        names = sorted(df["Dataset"].unique())
        ordered = pd.DataFrame({"Dataset": names, "f1_score": np.linspace(0.1, 0.99, 6)})
        shuffled = ordered.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = select_winners(df, baseline=ordered, test_size=TS, seed=0).per_dataset
        b = select_winners(df, baseline=shuffled, test_size=TS, seed=0).per_dataset
        pd.testing.assert_series_equal(
            a.set_index("Dataset")["verdict_raw"].sort_index(),
            b.set_index("Dataset")["verdict_raw"].sort_index(),
        )


class TestCorpusInference:
    def test_all_three_tests_are_reported(self):
        report = select_winners(synth(n_datasets=30, seed=28), test_size=TS, seed=0)
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

    def test_the_margin_splits_from_epsilon(self):
        """margin=None keeps the legacy rule (epsilon is also the margin); an explicit
        margin drives the sign counts and the direction, epsilon stays descriptive."""
        per = pd.DataFrame({"delta": [0.005] * 60})
        out = corpus_inference(per, epsilon=0.027, margin=0.0)
        assert out["direction"] == "classical"
        assert out["margin"] == 0.0
        assert out["sign"]["datasets_favoring_classical"] == 60
        assert out["sign"]["datasets_within_epsilon_0.027"] == 60
        assert corpus_inference(per, epsilon=0.027)["margin"] == 0.027

    def test_select_winners_records_margin_and_fdr(self):
        report = select_winners(synth(n_datasets=4, seed=28), test_size=TS, margin=0.01,
                                fdr=0.2)
        assert report.corpus["margin"] == 0.01 and report.corpus["fdr"] == 0.2
        assert report.margin == 0.01 and report.fdr == 0.2 and report.controls == ()
        assert report.corpus["across_datasets"]["margin"] == 0.01

    def test_winner_report_still_constructs_without_the_new_fields(self):
        r = WinnerReport(per_dataset=pd.DataFrame(), per_arm=pd.DataFrame(),
                         selection=pd.DataFrame(), metric="f1_score", epsilon=0.027,
                         alpha=0.05, test_size=0.2)
        assert (r.margin, r.fdr, r.controls) == (0.0, 0.10, ())

    def test_too_few_datasets_yields_no_inference_rather_than_a_fake_one(self):
        assert corpus_inference(pd.DataFrame({"delta": [0.1]}))["n_datasets_used"] == 1
        assert "ci" not in corpus_inference(pd.DataFrame({"delta": [0.1]}))
        assert corpus_inference(pd.DataFrame()) == {}


class TestOutputTables:
    def test_the_selection_trace_names_the_arm_used_for_each_held_out_iteration(self):
        """Auditability: the reader must be able to see *which* arm produced each score,
        because "the best arm" under nested selection is not one arm."""
        report = select_winners(synth(n_datasets=4, iters=5, seed=29), test_size=TS, seed=0)
        sel = report.selection
        assert len(sel) == 4 * 5
        assert {"Dataset", "iteration", "classical_arm", "quantum_arm"} <= set(sel.columns)
        assert sel["quantum_arm"].str.contains("|").any()

    def test_per_arm_reports_how_many_distinct_parameter_strings_an_arm_had(self):
        """The fragmentation diagnostic: >1 means that arm was re-tuned per resample,
        which is exactly the condition under which the old rule broke."""
        report = select_winners(synth(n_datasets=3, iters=5, classical_tuned=True, seed=30),
                                test_size=TS, seed=0)
        per_arm = report.per_arm
        tuned = per_arm[per_arm["side"] == "classical"]["n_distinct_parameters"]
        untuned = per_arm[per_arm["side"] == "quantum"]["n_distinct_parameters"]
        assert (tuned == 5).all(), "tuned arms should show one parameter string per resample"
        assert (untuned == 1).all()

    def test_every_dataset_appears_exactly_once_in_the_verdict_table(self):
        df = synth(n_datasets=12, seed=31)
        report = select_winners(df, test_size=TS, seed=0)
        assert report.per_dataset["Dataset"].is_unique
        assert set(report.per_dataset["Dataset"]) == set(df["Dataset"])


# ---- validation selection (split_mode: manifest) --------------------------------------

def fold_frame(val, test, k=3, repeats=2, datasets=("ds0",), tuning_metric="balanced_accuracy"):
    """A manifest-mode ModelResults frame: ``val``/``test`` map model -> per-fold scores.

    Scores are lists of length ``k * repeats``, indexed by the 0-based global fold
    ``repeat * k + fold``; ``iteration`` is that plus one, as qprofiler writes it.
    """
    rows = []
    for ds in datasets:
        for model in test:
            for g in range(k * repeats):
                rows.append({
                    "Dataset": ds, "embeddings": "none", "model": model,
                    "iteration": g + 1, "repeat": g // k, "fold": g % k,
                    "split_mode": "manifest", "split_k": k, "split_repeats": repeats,
                    "balanced_accuracy": test[model][g],
                    "tuning_score": val[model][g],
                    "tuning_metric": tuning_metric,
                })
    return pd.DataFrame(rows)


class TestValidationSelection:
    """``selection='validation'``: argmax of each fold's own validation score."""

    def _two_classical(self):
        # 'lr' validates best on every fold but tests worse than 'rf'; the test-argmax
        # rule would pick rf, validation selection must pick lr.
        n = 6
        val = {"lr": [0.9] * n, "rf": [0.6] * n, "qsvc": [0.7] * n}
        test = {"lr": [0.70, 0.72, 0.71, 0.69, 0.73, 0.70],
                "rf": [0.90] * n,
                "qsvc": [0.60, 0.61, 0.66, 0.62, 0.64, 0.59]}
        return fold_frame(val, test), test

    def test_the_validation_winner_is_used_even_when_another_arm_tests_better(self):
        df, test = self._two_classical()
        rep = select_winners(df, metric="balanced_accuracy", selection="validation")
        sel = rep.selection
        assert (sel["classical_arm"] == "none|lr").all()
        assert np.allclose(sel["classical_score"], test["lr"])
        assert np.allclose(sel["classical_val"], 0.9)
        assert np.allclose(sel["quantum_val"], 0.7)
        # The LOIO rule on the same frame chooses rf, the test-argmax arm.
        loio = select_winners(df, metric="balanced_accuracy", test_size=1 / 3)
        assert (loio.selection["classical_arm"] == "none|rf").all()

    def test_trace_carries_repeat_and_fold_and_keeps_the_score_columns(self):
        df, _ = self._two_classical()
        sel = select_winners(df, metric="balanced_accuracy", selection="validation").selection
        assert {"Dataset", "iteration", "repeat", "fold", "classical_arm", "quantum_arm",
                "classical_score", "quantum_score", "classical_val",
                "quantum_val"} <= set(sel.columns)
        assert sel["repeat"].tolist() == [0, 0, 0, 1, 1, 1]
        assert sel["fold"].tolist() == [0, 1, 2, 0, 1, 2]

    def test_the_loio_trace_is_unchanged(self):
        df, _ = self._two_classical()
        sel = select_winners(df, metric="balanced_accuracy", test_size=1 / 3).selection
        assert list(sel.columns) == ["Dataset", "iteration", "classical_arm", "quantum_arm",
                                     "classical_score", "quantum_score"]

    def test_corrected_repeated_cv_t_by_hand(self):
        """r = 1/(k-1), n = kR folds, df = kR - 1 (Bouckaert-Frank / Nadeau-Bengio)."""
        df, test = self._two_classical()
        k, R = 3, 2
        rep = select_winners(df, metric="balanced_accuracy", selection="validation",
                             alpha=0.05)
        d = np.array(test["lr"]) - np.array(test["qsvc"])
        n = k * R
        se = d.std(ddof=1) * np.sqrt(1.0 / n + 1.0 / (k - 1))
        tcrit = stats.t.ppf(0.975, df=n - 1)
        p = 2 * stats.t.sf(abs(d.mean()) / se, df=n - 1)
        row = rep.per_dataset.iloc[0]
        assert row["n_iterations"] == n
        assert row["delta"] == pytest.approx(d.mean())
        assert row["se"] == pytest.approx(se)
        assert row["ci_hi"] - row["delta"] == pytest.approx(tcrit * se)
        assert row["p_value"] == pytest.approx(p)
        assert rep.test_size == pytest.approx(1 / k)
        assert rep.k == k and rep.selection_mode == "validation"

    def test_k_comes_from_the_argument_or_split_k(self):
        df, _ = self._two_classical()
        a = select_winners(df, metric="balanced_accuracy", selection="validation")
        b = select_winners(df.drop(columns="split_k"), metric="balanced_accuracy",
                           selection="validation", k=3)
        assert a.per_dataset["se"].iloc[0] == pytest.approx(b.per_dataset["se"].iloc[0])
        with pytest.raises(ValueError, match="needs k"):
            select_winners(df.drop(columns="split_k"), metric="balanced_accuracy",
                           selection="validation")
        with pytest.raises(ValueError, match="disagrees"):
            select_winners(df, metric="balanced_accuracy", selection="validation", k=5)
        with pytest.raises(ValueError, match="contradicts"):
            select_winners(df, metric="balanced_accuracy", selection="validation",
                           test_size=0.3)

    def test_validation_ties_average_the_tied_test_scores(self):
        n = 6
        val = {"lr": [0.8] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        test = {"lr": [0.6] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        sel = select_winners(fold_frame(val, test), metric="balanced_accuracy",
                             selection="validation").selection
        assert (sel["classical_arm"] == TIE_SEPARATOR.join(["none|lr", "none|rf"])).all()
        assert np.allclose(sel["classical_score"], 0.7)

    def test_nan_validation_rows_are_excluded_and_logged(self, caplog):
        n = 6
        val = {"lr": [np.nan] * n, "rf": [0.6] * n, "qsvc": [0.7] * n}
        test = {"lr": [0.99] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        with caplog.at_level("WARNING", logger="qbiocode.utils.fair_selection"):
            rep = select_winners(fold_frame(val, test), metric="balanced_accuracy",
                                 selection="validation")
        assert "excluding 6 row(s)" in caplog.text and "none|lr" in caplog.text
        assert (rep.selection["classical_arm"] == "none|rf").all()
        assert "lr" not in set(rep.per_arm["model"])

    def test_mixed_tuning_metrics_raise(self):
        df, _ = self._two_classical()
        df.loc[df["model"] == "qsvc", "tuning_metric"] = "f1_score"
        with pytest.raises(ValueError, match="different metrics"):
            select_winners(df, metric="balanced_accuracy", selection="validation")

    def test_a_missing_validation_column_or_mode_raises(self):
        df, _ = self._two_classical()
        with pytest.raises(ValueError, match="validation column"):
            select_winners(df.drop(columns="tuning_score"), metric="balanced_accuracy",
                           selection="validation")
        with pytest.raises(ValueError, match="selection must be"):
            select_winners(df, metric="balanced_accuracy", selection="argmax", test_size=0.2)

    def test_the_loio_default_ignores_the_validation_column(self):
        df, _ = self._two_classical()
        a = select_winners(df, metric="balanced_accuracy", test_size=1 / 3)
        b = select_winners(df.drop(columns=["tuning_score", "tuning_metric"]),
                           metric="balanced_accuracy", test_size=1 / 3)
        pd.testing.assert_frame_equal(a.per_dataset, b.per_dataset)
        assert a.selection_mode == "loio" and a.k is None
        assert "selection" not in a.corpus


class TestValidationTieBreak:
    """Ties on the validation score are broken on ``val_auc`` (manifest-mode rows)."""

    def _tied(self, auc):
        n = 6
        val = {"lr": [0.8] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        test = {"lr": [0.6] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        df = fold_frame(val, test)
        df["val_auc"] = df["model"].map(auc)
        return df

    def test_the_higher_validation_auc_wins_a_tie(self, caplog):
        df = self._tied({"lr": 0.9, "rf": 0.7, "qsvc": 0.6})
        with caplog.at_level("INFO", logger="qbiocode.utils.fair_selection"):
            sel = select_winners(df, metric="balanced_accuracy", selection="validation").selection
        assert (sel["classical_arm"] == "none|lr").all()
        assert np.allclose(sel["classical_score"], 0.6)
        assert "6 side-folds tied" in caplog.text and "6 broken by val_auc" in caplog.text

    def test_a_loss_tie_break_takes_the_lower_value(self):
        df = self._tied({"lr": 0.9, "rf": 0.7, "qsvc": 0.6}).rename(columns={"val_auc": "val_log_loss"})
        sel = select_winners(df, metric="balanced_accuracy", selection="validation",
                             tiebreak_col="val_log_loss", tiebreak_higher_is_better=False).selection
        assert (sel["classical_arm"] == "none|rf").all()

    def test_a_missing_tie_break_value_keeps_the_tie(self):
        df = self._tied({"lr": 0.9, "rf": np.nan, "qsvc": 0.6})
        sel = select_winners(df, metric="balanced_accuracy", selection="validation").selection
        assert (sel["classical_arm"] == TIE_SEPARATOR.join(["none|lr", "none|rf"])).all()
        assert np.allclose(sel["classical_score"], 0.7)

    def test_equal_tie_break_values_keep_the_tie(self):
        df = self._tied({"lr": 0.8, "rf": 0.8, "qsvc": 0.6})
        sel = select_winners(df, metric="balanced_accuracy", selection="validation").selection
        assert (sel["classical_arm"] == TIE_SEPARATOR.join(["none|lr", "none|rf"])).all()

    def test_the_tie_break_never_overrides_the_validation_score(self):
        # rf validates higher; lr's better AUC must not matter when there is no tie.
        n = 6
        val = {"lr": [0.7] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        test = {"lr": [0.6] * n, "rf": [0.8] * n, "qsvc": [0.5] * n}
        df = fold_frame(val, test)
        df["val_auc"] = df["model"].map({"lr": 0.99, "rf": 0.5, "qsvc": 0.6})
        sel = select_winners(df, metric="balanced_accuracy", selection="validation").selection
        assert (sel["classical_arm"] == "none|rf").all()

    def test_none_or_an_absent_column_averages_as_before(self):
        df = self._tied({"lr": 0.9, "rf": 0.7, "qsvc": 0.6})
        a = select_winners(df, metric="balanced_accuracy", selection="validation",
                           tiebreak_col=None).selection
        b = select_winners(df.drop(columns="val_auc"), metric="balanced_accuracy",
                           selection="validation").selection
        for sel in (a, b):
            assert np.allclose(sel["classical_score"], 0.7)

    def test_loio_ignores_the_tie_break(self):
        df = self._tied({"lr": 0.9, "rf": 0.7, "qsvc": 0.6})
        a = select_winners(df, metric="balanced_accuracy", test_size=1 / 3)
        b = select_winners(df.drop(columns="val_auc"), metric="balanced_accuracy", test_size=1 / 3)
        pd.testing.assert_frame_equal(a.per_dataset, b.per_dataset)


class TestValidationSelectionPerEmbedding:
    """Validation selection picks per (dataset, embedding, fold): never across embeddings."""

    @staticmethod
    def _two_embeddings(n=6):
        g = np.arange(n)
        none = fold_frame(val={"lr": [0.9] * n, "vqc": [0.6] * n},
                          test={"lr": list(0.7 + 0.01 * g), "vqc": [0.6] * n})
        pca = fold_frame(val={"rf": [0.95] * n, "qsvc": [0.5] * n},
                         test={"rf": [0.8] * n, "qsvc": [0.75] * n})
        pca["embeddings"] = "pca"
        return pd.concat([none, pca], ignore_index=True), g

    def test_the_contrast_never_pairs_two_embeddings(self):
        df, g = self._two_embeddings()
        rep = select_winners(df, metric="balanced_accuracy", selection="validation")
        sel = rep.selection
        # One trace row per (embedding, fold), each pairing arms of its own embedding.
        assert len(sel) == 12
        assert (sel["classical_arm"].str.split("|").str[0] == sel["embeddings"]).all()
        assert (sel["quantum_arm"].str.split("|").str[0] == sel["embeddings"]).all()
        # A pooled argmax would pair pca|rf with none|vqc: delta 0.8 - 0.6 = 0.2.
        # Per embedding: none 0.1 + 0.01 g, pca 0.05; the dataset contrast is their mean.
        expected = ((0.1 + 0.01 * g) + 0.05) / 2
        row = rep.per_dataset.iloc[0]
        assert row["delta"] == pytest.approx(expected.mean())
        assert row["n_iterations"] == 6
        n, s = 6, expected.std(ddof=1)
        assert row["se"] == pytest.approx(s * np.sqrt(1 / n + 1 / 2))

    def test_a_fold_with_one_unpaired_embedding_uses_the_paired_one(self):
        df, g = self._two_embeddings()
        df = df[~((df["embeddings"] == "pca") & (df["model"] == "qsvc")
                  & (df["iteration"] == 1))]
        rep = select_winners(df, metric="balanced_accuracy", selection="validation")
        first = rep.selection[(rep.selection["iteration"] == 1)
                              & (rep.selection["embeddings"] == "pca")].iloc[0]
        assert first["quantum_arm"] == "" and np.isnan(first["quantum_score"])
        expected = ((0.1 + 0.01 * g) + 0.05) / 2
        expected[0] = 0.1                        # fold 1: only 'none' is paired
        assert rep.per_dataset.iloc[0]["delta"] == pytest.approx(expected.mean())

    def test_a_fold_with_no_paired_embedding_is_dropped(self):
        df, _ = self._two_embeddings()
        # Fold 1: 'none' has no quantum arm, 'pca' no classical one.
        df = df[~((df["iteration"] == 1)
                  & (((df["embeddings"] == "none") & (df["model"] == "vqc"))
                     | ((df["embeddings"] == "pca") & (df["model"] == "rf"))))]
        rep = select_winners(df, metric="balanced_accuracy", selection="validation")
        assert rep.per_dataset.iloc[0]["n_iterations"] == 5

    def test_internal_rows_are_refused_under_validation(self):
        df, _ = self._two_embeddings()
        df.loc[df["embeddings"] == "pca", "split_mode"] = "internal"
        with pytest.raises(ValueError, match="split_mode"):
            select_winners(df, metric="balanced_accuracy", selection="validation")
        df["split_mode"] = np.nan                # NaN reads as internal, too
        with pytest.raises(ValueError, match="split_mode"):
            select_winners(df, metric="balanced_accuracy", selection="validation")

    def test_loio_over_manifest_rows_is_logged(self, caplog):
        df, _ = self._two_embeddings()
        with caplog.at_level("WARNING", logger="qbiocode.utils.fair_selection"):
            select_winners(df, metric="balanced_accuracy", test_size=1 / 3)
        assert "designed for selection='validation'" in caplog.text

    def test_the_resolution_warning_speaks_of_k_under_validation(self, caplog):
        df, _ = self._two_embeddings()
        with caplog.at_level("WARNING", logger="qbiocode.utils.fair_selection"):
            select_winners(df, metric="balanced_accuracy", selection="validation",
                           epsilon=1e-6)
        assert "at k=3" in caplog.text and "test_size" not in caplog.text
