"""Fair classical-vs-quantum selection for the QBioCode benchmark.

Why this module exists
----------------------
``qc_winner_finder.qml_winner`` picks a winner with a pooled post-hoc argmax. On a
**pure null** -- both sides drawn from the same distribution, mean F1 0.75, sigma 0.04,
8 classical and 3 quantum learners over 2 embeddings and 5 resamples -- that rule
reports

    mean delta_f1 = classical - quantum = +0.0741        (epsilon = 0.027)
    classical "wins" on 84/84 datasets

so under the planned sweep config (``grid_search: True``, ``tune_quantum: False``) it
cannot report a quantum win *whatever the data says*. Decomposing that +0.0741:

    arm-count asymmetry alone (16 vs 6 arms, both averaged)   +0.0093
    + fragmentation applied to BOTH sides symmetrically       +0.0158
    + fragmentation on the classical side only (live config)  +0.0741
    leave-one-iteration-out nested selection (this module)    +0.0014

Two separate, one-directional biases stack, and the larger is not the obvious one:

1. **Fragmentation asymmetry (+0.074, dominant).** ``qc_winner_finder`` line 61 puts the
   *parameter column* in the groupby key. Optuna re-searches per resample
   (``model_run.py`` seeds ``random_state = seed + iter``), so a tuned learner's
   parameter string differs every iteration and each ``(Dataset, embedding, model)``
   shatters into ``iter`` singleton groups -- the "mean across splits" never happens for
   the tuned side. An untuned learner's parameter string is constant, so it *is*
   averaged. The subsequent ``max`` then compares classical **per-fold draws** against
   quantum **means of ``iter`` draws**: two different sampling distributions, the first
   with ``sqrt(iter)`` times the standard deviation. Nothing about the learners causes
   this; it is an artefact of which side was tuned.

2. **Arm-count asymmetry (+0.009).** ``E[max of 16] > E[max of 6]`` under the null.
   Real, but eight times smaller -- so balancing the *method lists* to equalise arm
   counts is treating the lesser problem. Nested selection removes this term too, which
   is what frees the method list to be chosen on scientific grounds.

The fix is not a better argmax. It is to stop letting the same data both choose an arm
and score it.

The algorithm
-------------
``metric`` is any column of ModelResults.csv (``f1_score``, ``balanced_accuracy``,
``mcc``, ``pr_auc``, ...), not just weighted F1.

An **arm** is one ``(embedding, model)`` pair. A **side** is ``classical`` or
``quantum``. Within one ``(dataset, iteration)`` every arm was scored on a
byte-identical test set -- ``qprofiler.py`` draws the ``train_test_split`` *above* the
embedding loop -- so arms are paired on both the model and the embedding axis, and a
paired analysis is valid.

1. **Per-arm, per-iteration table.** Parameters are carried as a payload, never as a
   grouping key. Aggregating over a parameter string is what caused bias (1), and a
   ``groupby`` on it also silently deletes rows whose parameter value is null
   (``groupby(dropna=True)``, and ``None`` round-trips through CSV as ``NaN``).

2. **Leave-one-iteration-out (LOIO) nested selection.** For each side and each held-out
   iteration ``i``, choose the arm with the best mean on the *other* ``I-1``
   iterations, then score that arm on ``i`` alone. Selection and scoring never share an
   iteration, so the score is unbiased for the selected arm's true performance. Arm
   count enters only through how stable selection is, not as an inflated maximum -- this
   is what kills biases (1) and (2) together. Ties are broken by a seeded permutation of
   the arm order rather than ``argmax``'s first-wins, which would otherwise resolve ties
   alphabetically by model name (``catboost`` over ``qsvc``, ``nb`` over ``qnn``) -- and
   ties are common, since a weighted F1 on ``m`` test rows moves in steps of about
   ``1/m``.

3. **Paired difference per iteration.** ``delta_i = classical_i - quantum_i``, keeping
   the user's sign convention: ``delta > 0`` classical ahead, ``delta < 0`` quantum
   ahead.

4. **Nadeau-Bengio corrected interval.** Resamples are independent
   ``train_test_split`` draws, so their test sets overlap and the ``delta_i`` are
   positively correlated; the naive ``s/sqrt(I)`` is anti-conservative. Use
   ``SE = s * sqrt(1/I + n_test/n_train)`` (Nadeau & Bengio 2003) with
   ``t_{1-alpha/2, I-1}``.

   A consequence worth stating plainly: that ``SE`` is bounded below by
   ``s * sqrt(n_test/n_train)`` *no matter how large ``I`` grows*. At
   ``test_size=0.3`` that floor is ``0.655 * s``, giving an asymptotic 95% half-width of
   ``1.28 * s``; at the measured ``s = 0.0267`` that is **0.034 > epsilon = 0.027**. So a
   per-dataset interval can never certify a margin as small as 0.027 at this
   ``test_size``, at any ``iter``. Buying more resamples does not help. Either accept
   ``epsilon ~ 0.04`` per dataset, or lower ``test_size`` to <= 0.21 -- a protocol
   change, not an analysis change. ``iteration_floor_half_width`` reports this bound so
   a run cannot silently chase an unreachable threshold.

5. **Four-bucket verdict.** The user's rule is ``abs(delta) > epsilon``; "unequivocal"
   requires the whole interval to clear the margin, not just the point estimate:

       ci_hi < -epsilon                      quantum_wins
       ci_lo > +epsilon                      classical_wins
       -epsilon < ci_lo and ci_hi < +epsilon equivalent      (both-sided, TOST-shaped)
       otherwise                             inconclusive

   ``inconclusive`` is not a failure. It is the honest label for a dataset where the
   data cannot separate a real margin from noise, and collapsing it into either win is
   how a benchmark overstates its result.

6. **Multiplicity.** The paper's claim is "these S of N datasets", a set-selection
   claim, so the per-dataset margin p-values are Benjamini-Hochberg adjusted. The
   p-value tests against the margin itself, matching the epsilon semantics: for a
   classical claim ``H0: delta <= +epsilon``, for a quantum claim
   ``H0: delta >= -epsilon``. Both raw and adjusted verdicts are returned; the adjusted
   one is what a paper should quote.

7. **Dummy floor (optional).** An arm that cannot beat a majority-class baseline is not
   evidence of anything, and on an imbalanced dataset weighted F1 hides it -- a
   majority-class dummy reaches weighted F1 0.906 on ``openml__ozone-level-8hr``
   (minority fraction 0.063). When a baseline is supplied, datasets where *neither*
   side clears it are marked ``below_baseline`` and excluded from the corpus counts
   rather than being reported as an equivalence.

References
----------
Nadeau & Bengio (2003), *Inference for the Generalization Error*, Machine Learning 52.
Benjamini & Hochberg (1995), JRSS-B 57. Demsar (2006), JMLR 7. Schuirmann (1987).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

#: Model-name stems that count as quantum. ``QUANTUM_MODELS`` is the dispatch registry
#: and is reused so this cannot drift from the learners that actually exist, but it
#: omits ``qensemble``: ``compute_qensemble`` defaults ``model='QEnsemble'``, whose
#: lowercased first ``_``-token is ``qensemble``, and a hardcoded quantum allowlist that
#: misses it classifies a quantum learner as *classical* -- suppressing a quantum win
#: and simultaneously adding an arm to the classical side. It is added here rather than
#: to ``QUANTUM_MODELS`` itself because ``model_run`` reads that set for seeding, and
#: widening it there would change fit behaviour rather than just bookkeeping.
try:  # pragma: no cover - the package always provides this
    from qbiocode.evaluation.model_run import QUANTUM_MODELS as _REGISTRY
except ImportError:  # pragma: no cover
    _REGISTRY = frozenset({"qsvc", "qnn", "vqc", "pqk", "qpl"})

QUANTUM_STEMS = frozenset(_REGISTRY) | {"qensemble"}

#: Candidate columns holding how a model was parameterised. ``model_evaluation`` writes
#: exactly one of these per run; older tables carry the third.
PARAMETER_COLUMNS = ("Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch")

VERDICTS = ("quantum_wins", "classical_wins", "equivalent", "inconclusive",
            "below_baseline", "insufficient_iterations")


def model_side(model: str) -> str:
    """``'quantum'`` or ``'classical'`` for a ModelResults.csv ``model`` label.

    Labels are matched on the first ``_``-separated token, lowercased, because the
    tuned twins append ``_opt`` (``qsvc_opt``) and ``compute_qpl`` fans out to
    ``qpl_<classical head>`` (``qpl_rf``, ``qpl_catboost``). A ``qpl_rf`` arm is a
    *quantum* arm -- the projection is quantum and the classical head only reads the
    projected features -- so matching the stem rather than searching for a classical
    name anywhere in the label is what gets those six arms onto the right side.
    """
    return "quantum" if str(model).split("_")[0].lower() in QUANTUM_STEMS else "classical"


@dataclass
class WinnerReport:
    """The result of :func:`select_winners`.

    Always truthy, deliberately. ``qml_winner`` returned a bare ``None`` when it found
    no quantum winner, and every caller guarded with ``assert found is not None`` before
    unpacking -- so "no quantum dataset" and "the function changed shape" were the same
    observation, and a shape change could not be caught by any test. An empty
    ``quantum_datasets`` on a populated ``per_dataset`` says "no winner" unambiguously.
    """

    per_dataset: pd.DataFrame
    per_arm: pd.DataFrame
    selection: pd.DataFrame
    metric: str
    epsilon: float
    alpha: float
    test_size: float
    corpus: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return True

    @property
    def quantum_datasets(self) -> list[str]:
        """Datasets with an unequivocal quantum win after FDR adjustment."""
        col = "verdict_adjusted"
        if self.per_dataset.empty or col not in self.per_dataset:
            return []
        hit = self.per_dataset[self.per_dataset[col] == "quantum_wins"]
        return sorted(hit["Dataset"].tolist())

    @property
    def classical_datasets(self) -> list[str]:
        col = "verdict_adjusted"
        if self.per_dataset.empty or col not in self.per_dataset:
            return []
        hit = self.per_dataset[self.per_dataset[col] == "classical_wins"]
        return sorted(hit["Dataset"].tolist())


def iteration_floor_half_width(sigma: float, test_size: float, alpha: float = 0.05) -> float:
    """Smallest 95% half-width a per-dataset interval can reach, at any ``iter``.

    The Nadeau-Bengio standard error ``s * sqrt(1/I + r)`` with ``r = test_size /
    (1 - test_size)`` tends to ``s * sqrt(r)`` as ``I`` grows, so the interval has a
    floor that more resamples cannot cross. Compare it against ``epsilon`` before a
    sweep: if the floor exceeds ``epsilon``, no amount of compute will certify a margin
    that small and ``test_size`` has to come down instead.
    """
    r = test_size / (1.0 - test_size)
    return float(stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(r) * sigma)


def corpus_inference(per_dataset: pd.DataFrame, epsilon: float = 0.027,
                     alpha: float = 0.05) -> dict:
    """Corpus-level inference over the per-dataset LOIO deltas.

    This is the level at which the benchmark's claim is actually testable, and the
    reason is a measured one. LOIO removes the selection bias but pays variance for it:
    each held-out score is a *single* iteration of a *possibly different* arm, so the
    per-iteration deltas scatter far wider than one arm's own resample noise. On a
    synthetic corpus the median per-dataset 95% half-width is about 0.11 -- four times
    the ``0.034`` floor that one fixed arm would give -- and a true quantum advantage of
    +0.15 is certified on only 20 of 30 datasets. Per-dataset certification at
    ``epsilon = 0.027`` is therefore underpowered by construction, and no epsilon choice
    repairs it.

    Across datasets the arithmetic reverses. Datasets are genuinely independent -- no
    shared test rows, so no Nadeau-Bengio correction applies and the ordinary
    ``s/sqrt(N)`` holds -- and ``N`` is 84 rather than ``I`` of 5. Three statements are
    returned, weakest assumption last:

    ``t``
        one-sample t on the per-dataset deltas: "the mean advantage across the corpus".
        Parametric, and the deltas are near-symmetric, but it is the least robust.
    ``wilcoxon``
        signed-rank on the same deltas. Demsar (2006) recommends exactly this for
        comparing two methods across datasets, since it assumes neither normality nor
        commensurability of F1 across datasets -- which a corpus spanning 2-class and
        many-class problems does not have.
    ``sign``
        an exact binomial test on the *count* of datasets each side leads by more than
        ``epsilon``. Assumes almost nothing and maps directly onto the sentence a
        benchmark paper wants to write ("quantum led on S of N datasets"), so it is the
        headline number even though it is the least efficient.

    Reporting all three, and reporting them together with the per-dataset counts, is the
    honest presentation: agreement among them is the evidence, and disagreement is
    itself a finding about the corpus.
    """
    out: dict = {}
    if per_dataset.empty or "delta" not in per_dataset:
        return out
    d = pd.to_numeric(per_dataset["delta"], errors="coerce")
    d = d[np.isfinite(d)]
    out["n_datasets_used"] = int(d.size)
    if d.size < 2:
        return out

    out["mean_delta"] = float(d.mean())
    out["sd_delta"] = float(d.std(ddof=1))
    se = float(d.std(ddof=1) / np.sqrt(d.size))  # independent datasets: no NB inflation
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=d.size - 1))
    out["ci"] = (out["mean_delta"] - tcrit * se, out["mean_delta"] + tcrit * se)
    out["t"] = {
        "statistic": float(out["mean_delta"] / se) if se > 0 else np.nan,
        "p_two_sided": float(2 * stats.t.sf(abs(out["mean_delta"] / se), df=d.size - 1))
        if se > 0 else np.nan,
    }
    try:
        with warnings.catch_warnings():
            # An all-zero delta vector makes scipy's normal approximation divide by a
            # zero scale. That is a perfect tie, whose honest p-value is 1.0, so the
            # NaN is normalised below rather than propagated into the direction rule.
            warnings.simplefilter("ignore", RuntimeWarning)
            w = stats.wilcoxon(d.to_numpy(), zero_method="wilcox", alternative="two-sided")
        p_w = float(w.pvalue)
        out["wilcoxon"] = {
            "statistic": float(w.statistic),
            "p_two_sided": p_w if np.isfinite(p_w) else 1.0,
        }
    except ValueError:
        # every delta identically zero -- perfect tie, no rank information
        out["wilcoxon"] = {"statistic": np.nan, "p_two_sided": 1.0}

    n_cls = int((d > epsilon).sum())
    n_qnt = int((d < -epsilon).sum())
    n_tied = int(d.size - n_cls - n_qnt)
    binom_p = (float(stats.binomtest(n_qnt, n_qnt + n_cls, 0.5).pvalue)
               if (n_qnt + n_cls) else 1.0)
    out["sign"] = {
        "datasets_favoring_classical": n_cls,
        "datasets_favoring_quantum": n_qnt,
        f"datasets_within_epsilon_{epsilon:g}": n_tied,
        "p_two_sided": binom_p,
    }
    # Direction is stated only when at least one test clears alpha; otherwise the
    # corpus verdict is 'no detectable difference', which is a result, not a gap.
    # nan-safe: a zero-variance delta vector leaves the t p-value undefined, and
    # min() with a NaN operand can return the NaN and make the comparison below silently
    # false. Undefined p-values are dropped rather than treated as significant.
    candidates = [
        p for p in (out["t"]["p_two_sided"], out["wilcoxon"]["p_two_sided"], binom_p)
        if p is not None and np.isfinite(p)
    ]
    signif = min(candidates) if candidates else 1.0
    if signif > alpha or abs(out["mean_delta"]) <= epsilon:
        out["direction"] = "no_detectable_difference"
    else:
        out["direction"] = "classical" if out["mean_delta"] > 0 else "quantum"
    return out


def _parameter_payload(df: pd.DataFrame) -> pd.Series:
    """First non-null parameter string per row, as a *payload* -- never a groupby key."""
    present = [c for c in PARAMETER_COLUMNS if c in df.columns]
    if not present:
        return pd.Series(["" for _ in range(len(df))], index=df.index, dtype=object)
    return df[present].bfill(axis=1).iloc[:, 0]


def arm_iteration_table(df: pd.DataFrame, metric: str = "f1_score") -> pd.DataFrame:
    """One row per ``(Dataset, side, arm, iteration)``: the atomic unit of comparison.

    Averaging happens over nothing here. Any duplicate ``(Dataset, embeddings, model,
    iteration)`` -- which a resumed or double-appended run produces -- is reduced by
    ``mean`` with a warning, because silently keeping the first would make the result
    depend on row order.
    """
    required = {"Dataset", "embeddings", "model", "iteration", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ModelResults frame lacks {sorted(missing)}. Present: {sorted(df.columns)}. "
            f"'iteration' is required -- without it there is no resample axis to "
            f"aggregate over, and every interval below would be undefined."
        )

    work = df.assign(
        _params=_parameter_payload(df),
        side=df["model"].map(model_side),
        arm=df["embeddings"].astype(str) + "|" + df["model"].astype(str),
    )
    work = work[pd.to_numeric(work[metric], errors="coerce").notna()].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")

    keys = ["Dataset", "side", "arm", "embeddings", "model", "iteration"]
    dup = work.duplicated(subset=keys).sum()
    if dup:
        # warnings.warn, not logger.warning: this one changes the numbers. A library log
        # line is easy to miss under a configured root logger, and a duplicated arm
        # silently reweights the iteration it lands on.
        warnings.warn(
            f"{dup} duplicate (Dataset, arm, iteration) rows in the results frame; "
            f"reducing each by mean. A resumed run that re-appended to "
            f"ModelResults.csv is the usual cause.",
            UserWarning,
            stacklevel=2,
        )
    out = (
        work.groupby(keys, dropna=False)
        .agg(**{metric: (metric, "mean"), "parameters": ("_params", "first")})
        .reset_index()
    )
    return out


def _loio_side_scores(
    mat: np.ndarray, arms: Sequence[str], rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    """Leave-one-iteration-out nested selection over one side's ``[arm, iteration]``.

    Returns the held-out score per iteration and the arm chosen for each. ``mat`` may
    contain NaN where an arm was not run on an iteration; selection uses ``nanmean`` and
    an arm with no usable training iteration is skipped.
    """
    n_arms, n_iter = mat.shape
    order = rng.permutation(n_arms)  # seeded tie-break; argmax alone is alphabetical
    scores = np.full(n_iter, np.nan)
    chosen: list[str] = []
    for i in range(n_iter):
        train = [j for j in range(n_iter) if j != i]
        if train:
            # Hand-rolled rather than np.nanmean: an arm absent from every training
            # iteration is an all-NaN slice, which nanmean answers correctly but with a
            # RuntimeWarning per arm per dataset -- thousands of lines of noise for a
            # case that is expected here.
            sub = mat[:, train]
            ok = np.isfinite(sub)
            counts = ok.sum(axis=1)
            sums = np.where(ok, sub, 0.0).sum(axis=1)
            means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        else:
            means = np.full(n_arms, np.nan)
        cand = [a for a in order if np.isfinite(means[a]) and np.isfinite(mat[a, i])]
        if not cand:
            chosen.append("")
            continue
        best = max(cand, key=lambda a: means[a])
        scores[i] = mat[best, i]
        chosen.append(arms[best])
    return scores, chosen


def select_winners(
    results: pd.DataFrame,
    metric: str = "f1_score",
    epsilon: float = 0.027,
    alpha: float = 0.05,
    test_size: float = 0.3,
    seed: int = 0,
    baseline: pd.DataFrame | None = None,
) -> WinnerReport:
    """Decide, per dataset, whether either side wins by more than ``epsilon``.

    Parameters
    ----------
    results
        A ModelResults.csv-shaped frame. Needs ``Dataset``, ``embeddings``, ``model``,
        ``iteration`` and ``metric``.
    metric
        Any metric column. Higher is better for all of ``f1_score``, ``accuracy``,
        ``balanced_accuracy``, ``mcc``, ``auc`` and ``pr_auc``, which is what the
        comparison assumes; do not pass ``time``.
    epsilon
        The margin. A win is declared only when the whole interval clears it. Check it
        against :func:`iteration_floor_half_width` first.
    baseline
        Optional ``Dataset`` -> baseline-metric frame (columns ``Dataset`` and
        ``metric``). Datasets where neither side beats it are marked
        ``below_baseline``.
    """
    table = arm_iteration_table(results, metric=metric)
    rng = np.random.default_rng(seed)

    per_arm = (
        table.groupby(["Dataset", "side", "arm", "embeddings", "model"], dropna=False)
        .agg(
            mean=(metric, "mean"),
            sd=(metric, lambda s: s.std(ddof=1)),
            n_iter=(metric, "size"),
            parameters=("parameters", "nunique"),
        )
        .reset_index()
        .rename(columns={"parameters": "n_distinct_parameters"})
    )

    rows, sel_rows = [], []
    for dataset, block in table.groupby("Dataset", sort=True):
        iters = sorted(block["iteration"].unique())
        side_scores, side_arms = {}, {}
        for side in ("classical", "quantum"):
            sub = block[block["side"] == side]
            if sub.empty:
                side_scores[side] = np.full(len(iters), np.nan)
                side_arms[side] = [""] * len(iters)
                continue
            wide = sub.pivot_table(index="arm", columns="iteration", values=metric,
                                   aggfunc="mean").reindex(columns=iters)
            s, chosen = _loio_side_scores(wide.to_numpy(float), list(wide.index), rng)
            side_scores[side], side_arms[side] = s, chosen

        for k, it in enumerate(iters):
            sel_rows.append(dict(
                Dataset=dataset, iteration=it,
                classical_arm=side_arms["classical"][k],
                quantum_arm=side_arms["quantum"][k],
                classical_score=side_scores["classical"][k],
                quantum_score=side_scores["quantum"][k],
            ))

        # Diagnostic only: what the old pooled-argmax rule would have said, with the
        # fragmentation bug removed (both sides averaged over iterations). Reporting it
        # beside `delta` is what lets a paper *exhibit* the selection bias rather than
        # merely assert it -- the gap between the two columns IS the winner's curse.
        naive = {}
        for side in ("classical", "quantum"):
            sub = block[block["side"] == side]
            naive[side] = (
                sub.groupby("arm")[metric].mean().max() if not sub.empty else np.nan
            )
        row_naive = naive["classical"] - naive["quantum"]

        delta = side_scores["classical"] - side_scores["quantum"]
        usable = delta[np.isfinite(delta)]
        row = dict(
            Dataset=dataset, n_iterations=int(usable.size), naive_delta=float(row_naive),
            classical_mean=float(np.nanmean(side_scores["classical"]))
            if np.isfinite(side_scores["classical"]).any() else np.nan,
            quantum_mean=float(np.nanmean(side_scores["quantum"]))
            if np.isfinite(side_scores["quantum"]).any() else np.nan,
        )

        # An interval needs a spread, so two usable iterations is the hard floor. Naming
        # this rather than emitting a NaN interval keeps such datasets out of the corpus
        # counts instead of silently landing in 'inconclusive' beside genuinely
        # ambiguous ones.
        if usable.size < 2:
            row.update(delta=float(usable[0]) if usable.size else np.nan,
                       se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_margin=np.nan,
                       verdict_raw="insufficient_iterations")
            rows.append(row)
            continue

        n = usable.size
        mean_d = float(usable.mean())
        s = float(usable.std(ddof=1))
        r = test_size / (1.0 - test_size)
        se = s * np.sqrt(1.0 / n + r)          # Nadeau & Bengio (2003)
        tcrit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        ci_lo, ci_hi = mean_d - tcrit * se, mean_d + tcrit * se

        # One-sided test against the MARGIN, in whichever direction the data points.
        # Testing against zero would answer a different question than epsilon asks.
        if se > 0:
            if mean_d >= 0:
                p = float(stats.t.sf((mean_d - epsilon) / se, df=n - 1))
            else:
                p = float(stats.t.cdf((mean_d + epsilon) / se, df=n - 1))
        else:
            p = 0.0 if abs(mean_d) > epsilon else 1.0

        if ci_hi < -epsilon:
            verdict = "quantum_wins"
        elif ci_lo > epsilon:
            verdict = "classical_wins"
        elif ci_lo > -epsilon and ci_hi < epsilon:
            verdict = "equivalent"
        else:
            verdict = "inconclusive"

        row.update(delta=mean_d, se=float(se), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                   p_margin=p, verdict_raw=verdict)
        rows.append(row)

    per_dataset = pd.DataFrame(rows)
    selection = pd.DataFrame(sel_rows)

    # ---- optional dummy floor ------------------------------------------------------
    if baseline is not None and not per_dataset.empty:
        if {"Dataset", metric} - set(baseline.columns):
            raise ValueError(
                f"baseline needs columns ['Dataset', {metric!r}]; got "
                f"{sorted(baseline.columns)}"
            )
        base = baseline.rename(columns={metric: "baseline"})[["Dataset", "baseline"]]
        per_dataset = per_dataset.merge(base, on="Dataset", how="left")  # on Dataset, never index
        beaten = (
            (per_dataset["classical_mean"] > per_dataset["baseline"])
            | (per_dataset["quantum_mean"] > per_dataset["baseline"])
        )
        per_dataset.loc[per_dataset["baseline"].notna() & ~beaten, "verdict_raw"] = "below_baseline"

    # ---- Benjamini-Hochberg across datasets ---------------------------------------
    per_dataset["verdict_adjusted"] = per_dataset.get("verdict_raw", pd.Series(dtype=object))
    per_dataset["p_adjusted"] = np.nan
    if not per_dataset.empty:
        claimed = per_dataset["verdict_raw"].isin(["quantum_wins", "classical_wins"])
        pv = per_dataset.loc[claimed, "p_margin"]
        if len(pv):
            order = np.argsort(pv.to_numpy())
            m = len(pv)
            ranked = pv.to_numpy()[order]
            adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)
            out = np.empty(m)
            out[order] = adj
            per_dataset.loc[claimed, "p_adjusted"] = out
            demoted = claimed & (per_dataset["p_adjusted"] > alpha)
            per_dataset.loc[demoted, "verdict_adjusted"] = "inconclusive"

    sigma = float(per_dataset["se"].dropna().median()) if "se" in per_dataset else np.nan
    corpus = {
        "n_datasets": int(per_dataset["Dataset"].nunique()) if not per_dataset.empty else 0,
        "epsilon": epsilon,
        "alpha": alpha,
        "metric": metric,
        "counts_raw": per_dataset.get("verdict_raw", pd.Series(dtype=object))
        .value_counts().to_dict(),
        "counts_adjusted": per_dataset.get("verdict_adjusted", pd.Series(dtype=object))
        .value_counts().to_dict(),
        "median_interval_half_width": float(
            ((per_dataset["ci_hi"] - per_dataset["ci_lo"]) / 2).median()
        ) if "ci_hi" in per_dataset and per_dataset["ci_hi"].notna().any() else np.nan,
        "epsilon_is_reachable": None,
        "mean_naive_delta": float(per_dataset["naive_delta"].mean())
        if "naive_delta" in per_dataset and per_dataset["naive_delta"].notna().any()
        else np.nan,
    }
    corpus["across_datasets"] = corpus_inference(per_dataset, epsilon=epsilon, alpha=alpha)
    if np.isfinite(sigma):
        # se already carries the NB inflation, so compare epsilon against the half-width
        # the corpus actually achieved rather than against a nominal s/sqrt(I).
        corpus["epsilon_is_reachable"] = bool(
            corpus["median_interval_half_width"] <= epsilon
        )
        if corpus["epsilon_is_reachable"] is False:
            logger.warning(
                "epsilon=%.4f is below the median achieved interval half-width %.4f, so "
                "most datasets cannot certify a margin that small at test_size=%.2f. "
                "Raising 'iter' will not close this -- the Nadeau-Bengio standard error "
                "floors at s*sqrt(test_size/(1-test_size)). Lower test_size or widen "
                "epsilon.",
                epsilon, corpus["median_interval_half_width"], test_size,
            )

    return WinnerReport(
        per_dataset=per_dataset, per_arm=per_arm, selection=selection,
        metric=metric, epsilon=epsilon, alpha=alpha, test_size=test_size, corpus=corpus,
    )


__all__ = [
    "QUANTUM_STEMS",
    "PARAMETER_COLUMNS",
    "VERDICTS",
    "WinnerReport",
    "arm_iteration_table",
    "corpus_inference",
    "iteration_floor_half_width",
    "model_side",
    "select_winners",
]
