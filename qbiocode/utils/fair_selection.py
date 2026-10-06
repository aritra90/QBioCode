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
    leave-one-iteration-out nested selection (this module)    +0.0014 (see caveat)

That last figure holds for *independent* iterations. The real protocol draws repeated
random holdouts, whose test sets share rows -- 73-87% of a later iteration's test rows
were training rows of an earlier one -- so iterations are positively correlated and the
held-out iteration is not fully independent of the ones that chose the arm. In
simulation LOIO then keeps a small asymmetry toward the side with more arms: about
+0.009 for 9 against 4 arms. It is far smaller than the bias it replaces, but it is not
zero, and it leans the way the arm counts lean.

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
   counts is treating the lesser problem. Nested selection largely removes this term
   too (up to the overlapping-holdout caveat above), which is what frees the method
   list to be chosen on scientific grounds.

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
   is what kills biases (1) and (2) together (up to the overlap caveat above).

   Ties are common, since a weighted F1 on ``m`` test rows moves in steps of about
   ``1/m``. ``argmax``'s first-wins would resolve them alphabetically by model name
   (``catboost`` over ``qsvc``, ``nb`` over ``qnn``), and a seeded random tie-break
   would make the verdict depend on the seed. Instead, when several arms tie for best on
   the selection mean (to ``1e-12``), the held-out score is the *mean* of the tied arms'
   held-out scores -- the expectation over a uniform random tie-break, computed exactly
   -- and the selection trace records the tied arms joined by ``TIE_SEPARATOR``
   (``"+"``, which cannot be confused with the ``"|"`` inside an
   ``"embedding|model"`` arm label).

   Under ``split_mode: manifest`` (stratified k-fold repeated R times, each outer fold
   with its own validation rows carved from its training rows) the choice needs no other
   iteration: ``selection='validation'`` takes, per fold and side, the arm with the best
   validation score (``tuning_score``) and scores it on that fold's test rows. Ties are
   averaged as above. The interval is then the corrected repeated-CV t of Bouckaert &
   Frank (2004), the same formula as step 4 with ``r = 1/(k-1)``, ``n = kR`` folds and
   ``kR - 1`` degrees of freedom.

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
   per-dataset interval can never certify equivalence within +/-0.027 at this
   ``test_size``, at any ``iter`` (nor a win whose true size is below the half-width).
   Buying more resamples does not help. Either accept ``epsilon ~ 0.04`` per dataset,
   or lower ``test_size`` to <= 0.21 -- a protocol
   change, not an analysis change. ``iteration_floor_half_width`` reports this bound so
   a run cannot silently chase an unreachable threshold. The correction depends on the
   split fraction, which ModelResults.csv does not record, so ``test_size`` has no
   default and must be passed (the pilot used 0.2).

5. **Two thresholds, two questions.** ``margin`` is the pre-registered *practical
   superiority* margin (default 0: "is there a difference at all"); ``epsilon`` is only
   the TOST *equivalence* bound. They used to be one number, set to the resolution
   floor, so a real effect had to clear noise twice: once as the margin and again as
   the interval's own width. The raw verdict, with the interval at level ``alpha``:

       ci_hi < -margin                       quantum_wins
       ci_lo > +margin                       classical_wins
       -epsilon < ci_lo and ci_hi < +epsilon equivalent      (both-sided, TOST-shaped)
       otherwise                             inconclusive

   Wins take precedence over equivalence; ``within_equivalence`` is reported separately
   so a significant but practically negligible win stays visible as such.
   ``inconclusive`` is not a failure. It is the honest label for a dataset where the
   data cannot separate an effect from noise, and collapsing it into either win is how
   a benchmark overstates its result.

6. **Multiplicity.** The paper's claim is "these S of N datasets", a set-selection
   claim. Per dataset ``p_value = min(1, 2 * t.sf((|delta| - margin) / se, I-1))``, the
   ordinary two-sided t-test when ``margin = 0``, for which ``p_value < alpha`` exactly
   when the interval excludes zero. Benjamini-Hochberg at level ``fdr`` is applied over
   *every* discovery dataset with a finite p-value, in both directions -- not only over
   the datasets already called a win, which picks the family after looking and never
   demotes a lone claim. ``verdict_adjusted`` is the direction of ``delta`` when
   ``p_adjusted <= fdr``, else ``equivalent`` or ``inconclusive`` from
   ``within_equivalence``. With ``fdr <= alpha`` the adjustment can only demote; with
   ``fdr > alpha`` BH may also confirm a dataset whose raw p lies in
   ``(alpha, BH threshold]``, which is the intended behaviour of an FDR procedure, not
   a promotion bug. Datasets named in ``controls`` (synthetic positive/negative
   controls) form a separate family with Holm-adjusted p-values judged at ``alpha``, so
   that planted effects neither dilute nor inflate the discovery family. Both raw and
   adjusted verdicts are returned; the adjusted one is what a paper should quote.

7. **Dummy floor (optional).** An arm that cannot beat a majority-class baseline is not
   evidence of anything, and on an imbalanced dataset weighted F1 hides it -- a
   majority-class dummy reaches weighted F1 0.906 on ``openml__ozone-level-8hr``
   (minority fraction 0.063). When a baseline is supplied, datasets where *neither*
   side clears it are marked ``below_baseline`` and excluded from the corpus counts
   rather than being reported as an equivalence.

References
----------
Nadeau & Bengio (2003), *Inference for the Generalization Error*, Machine Learning 52.
Bouckaert & Frank (2004), *Evaluating the Replicability of Significance Tests for
Comparing Learning Algorithms*, PAKDD.
Benjamini & Hochberg (1995), JRSS-B 57. Holm (1979), Scand. J. Statist. 6.
Demsar (2006), JMLR 7. Schuirmann (1987).
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

#: Joins the names of arms that tie for best in one LOIO selection step, sorted, in the
#: ``classical_arm`` / ``quantum_arm`` columns of the selection trace. Arm labels are
#: ``"embedding|model"`` and no embedding or model name contains ``"+"``, so a tied
#: label splits back into its arms unambiguously.
TIE_SEPARATOR = "+"

#: How the arm scored on each resample is chosen: ``'loio'`` (leave one iteration out,
#: for repeated random holdouts) or ``'validation'`` (argmax of each fold's own
#: validation score, for ``split_mode: manifest``). See :func:`select_winners`.
SELECTION_MODES = ("loio", "validation")

#: Split coordinates of a global ``iteration`` under ``split_mode: manifest``; carried
#: into the selection trace when present.
_FOLD_COLUMNS = ("repeat", "fold")

#: Verdicts that are not the outcome of a test and are carried through the multiplicity
#: adjustment unchanged.
_SPECIAL_VERDICTS = ("below_baseline", "insufficient_iterations")


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
    margin: float = 0.0
    fdr: float = 0.10
    controls: tuple = ()
    #: ``'loio'`` or ``'validation'``: how :attr:`selection` was built.
    selection_mode: str = "loio"
    #: Outer folds per repeat under ``selection_mode='validation'``, else ``None``.
    k: int | None = None

    def __bool__(self) -> bool:
        return True

    def _hits(self, verdict: str) -> list[str]:
        col = "verdict_adjusted"
        if self.per_dataset.empty or col not in self.per_dataset:
            return []
        hit = self.per_dataset[self.per_dataset[col] == verdict]
        if "family" in hit:
            hit = hit[hit["family"] != "control"]
        return sorted(hit["Dataset"].tolist())

    @property
    def quantum_datasets(self) -> list[str]:
        """Discovery datasets with a quantum win after FDR adjustment.

        Rows in the ``control`` family are excluded.
        """
        return self._hits("quantum_wins")

    @property
    def classical_datasets(self) -> list[str]:
        """Discovery datasets with a classical win after FDR adjustment.

        Rows in the ``control`` family are excluded.
        """
        return self._hits("classical_wins")


def iteration_floor_half_width(sigma: float, test_size: float, alpha: float = 0.05) -> float:
    """Smallest 95% half-width a per-dataset interval can reach, at any ``iter``.

    The Nadeau-Bengio standard error ``s * sqrt(1/I + r)`` with ``r = test_size /
    (1 - test_size)`` tends to ``s * sqrt(r)`` as ``I`` grows, so the interval has a
    floor that more resamples cannot cross. Compare it against ``epsilon`` before a
    sweep: if the floor exceeds ``epsilon``, no amount of compute will certify equivalence
    that small and ``test_size`` has to come down instead.
    """
    r = test_size / (1.0 - test_size)
    return float(stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(r) * sigma)


def corpus_inference(per_dataset: pd.DataFrame, epsilon: float = 0.027,
                     alpha: float = 0.05, *, margin: float | None = None) -> dict:
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
        ``margin``. Assumes almost nothing and maps directly onto the sentence a
        benchmark paper wants to write ("quantum led on S of N datasets"), so it is the
        headline number even though it is the least efficient.

    Reporting all three, and reporting them together with the per-dataset counts, is the
    honest presentation: agreement among them is the evidence, and disagreement is
    itself a finding about the corpus.

    Args:
        per_dataset: Frame with a ``delta`` column, one row per independent dataset.
        epsilon: Equivalence bound. Only descriptive here: the ``sign`` block reports
            how many deltas lie within ``+/-epsilon``.
        alpha: Level of the interval and of the direction rule.
        margin: Practical superiority margin. The sign test counts a dataset for a
            side only when its delta clears ``margin``, and a direction is stated only
            when ``|mean_delta| > margin``. ``None`` (the default) keeps the historical
            behaviour of using ``epsilon`` as that margin as well;
            :func:`select_winners` always passes its own ``margin`` explicitly.

    Returns:
        dict: ``n_datasets_used``, ``margin``, ``mean_delta``, ``sd_delta``, ``ci``,
        ``t``, ``wilcoxon``, ``sign`` and ``direction`` (empty when ``per_dataset`` has
        no ``delta``; only ``n_datasets_used`` and ``margin`` below two datasets).
    """
    out: dict = {}
    if per_dataset.empty or "delta" not in per_dataset:
        return out
    if margin is None:
        margin = epsilon
    d = pd.to_numeric(per_dataset["delta"], errors="coerce")
    d = d[np.isfinite(d)]
    out["n_datasets_used"] = int(d.size)
    out["margin"] = float(margin)
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

    n_cls = int((d > margin).sum())
    n_qnt = int((d < -margin).sum())
    n_tied = int(d.size - n_cls - n_qnt)
    binom_p = (float(stats.binomtest(n_qnt, n_qnt + n_cls, 0.5).pvalue)
               if (n_qnt + n_cls) else 1.0)
    out["sign"] = {
        "datasets_favoring_classical": n_cls,
        "datasets_favoring_quantum": n_qnt,
        f"datasets_within_margin_{margin:g}": n_tied,
        f"datasets_within_epsilon_{epsilon:g}": int((d.abs() <= epsilon).sum()),
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
    if signif > alpha or abs(out["mean_delta"]) <= margin:
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


def arm_iteration_table(df: pd.DataFrame, metric: str = "f1_score",
                        validation_col: str | None = None,
                        extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """One row per ``(Dataset, side, arm, iteration)``: the atomic unit of comparison.

    Averaging happens over nothing here. Any duplicate ``(Dataset, embeddings, model,
    iteration)`` -- which a resumed or double-appended run produces -- is reduced by
    ``mean`` with a logged warning, because silently keeping the first would make the
    result depend on row order.

    With ``validation_col`` (validation selection, see :func:`select_winners`) that
    column is carried as a second value beside ``metric``, reduced the same way; it may
    be NaN. ``repeat`` and ``fold`` are carried when the frame has them, and so is every
    numeric column in ``extra_cols`` (e.g. the validation tie-break), reduced by mean.
    """
    required = {"Dataset", "embeddings", "model", "iteration", metric}
    if validation_col is not None:
        required.add(validation_col)
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
    if validation_col is not None:
        work[validation_col] = pd.to_numeric(work[validation_col], errors="coerce")
    extra_cols = [c for c in extra_cols if c in work.columns]
    for col in extra_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    keys = ["Dataset", "side", "arm", "embeddings", "model", "iteration"]
    dup = work.duplicated(subset=keys).sum()
    if dup:
        # A log line at WARNING, not warnings.warn: this one changes the numbers, and
        # warnings are silenced in any process that has imported matplotlib.pyplot
        # (qc_winner_finder does), which would hide a duplicated arm silently
        # reweighting the iteration it lands on.
        logger.warning(
            "%d duplicate (Dataset, arm, iteration) rows in the results frame; "
            "reducing each by mean. A resumed run that re-appended to "
            "ModelResults.csv is the usual cause.",
            dup,
        )
    aggs = {metric: (metric, "mean"), "parameters": ("_params", "first")}
    if validation_col is not None:
        aggs[validation_col] = (validation_col, "mean")
    for col in extra_cols:
        aggs[col] = (col, "mean")
    for col in _FOLD_COLUMNS:
        # The split coordinates of the global iteration, carried for the selection trace.
        if col in work.columns:
            aggs[col] = (col, "first")
    out = work.groupby(keys, dropna=False).agg(**aggs).reset_index()
    return out


def _loio_side_scores(
    mat: np.ndarray, arms: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Leave-one-iteration-out nested selection over one side's ``[arm, iteration]``.

    Returns the held-out score per iteration and the arm chosen for each. ``mat`` may
    contain NaN where an arm was not run on an iteration; selection uses ``nanmean`` and
    an arm with no usable training iteration is skipped.

    Ties are resolved deterministically. When several arms share the best selection mean
    (``np.isclose`` with ``rtol=0, atol=1e-12``), the held-out score is the mean of their
    held-out scores -- the expected score under a uniform random tie-break, without the
    randomness -- and the chosen arm is the sorted tied labels joined by
    :data:`TIE_SEPARATOR`.
    """
    n_arms, n_iter = mat.shape
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
        cand = [a for a in range(n_arms) if np.isfinite(means[a]) and np.isfinite(mat[a, i])]
        if not cand:
            chosen.append("")
            continue
        top = max(means[a] for a in cand)
        tied = [a for a in cand if np.isclose(means[a], top, rtol=0.0, atol=1e-12)]
        held = mat[tied, i]
        # Identical held-out scores are returned as they are, not re-averaged: the
        # rounding in a mean of eight copies of 0.8 is enough to turn a perfect tie
        # between the sides into a 1e-16 "win" with zero standard error.
        scores[i] = float(held[0]) if np.all(held == held[0]) else float(held.mean())
        chosen.append(TIE_SEPARATOR.join(sorted(arms[a] for a in tied)))
    return scores, chosen


def _validation_side_scores(
    val: np.ndarray, test: np.ndarray, arms: Sequence[str],
    tiebreak: np.ndarray | None = None, ties: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validation-selected scores over one side's ``[arm, iteration]`` matrices.

    For each iteration (one outer fold) the arm with the best validation score on that
    fold is chosen and scored on the same fold's test rows. The validation rows are
    disjoint from the test rows, so the test score of the chosen arm carries no winner's
    curse. Candidates are arms finite in both ``val`` and ``test`` on that iteration.

    Arms whose validation scores are within ``atol=1e-12`` of the best are tied. With a
    ``tiebreak`` matrix (higher is better, same shape), a tie is narrowed to the tied
    arms with the best ``tiebreak`` value -- unless one of them has none, in which case
    the tie stands. A tie that remains follows :func:`_loio_side_scores`: the test score
    is the mean of the tied arms' test scores (or the shared value when identical), and
    the label joins the sorted tied arms with :data:`TIE_SEPARATOR`. ``ties``, when
    given, counts ``tied`` / ``broken`` / ``unbreakable`` folds.

    Returns:
        ``(test_scores, validation_scores, chosen)`` per iteration; NaN and ``""`` where
        no arm is a candidate.
    """
    n_arms, n_iter = test.shape
    scores = np.full(n_iter, np.nan)
    vals = np.full(n_iter, np.nan)
    chosen: list[str] = []
    for i in range(n_iter):
        cand = [a for a in range(n_arms) if np.isfinite(val[a, i]) and np.isfinite(test[a, i])]
        if not cand:
            chosen.append("")
            continue
        top = max(val[a, i] for a in cand)
        tied = [a for a in cand if np.isclose(val[a, i], top, rtol=0.0, atol=1e-12)]
        if len(tied) > 1 and ties is not None:
            ties["tied"] = ties.get("tied", 0) + 1
        if len(tied) > 1 and tiebreak is not None:
            tb = tiebreak[tied, i]
            if np.isfinite(tb).all():
                best = tb.max()
                tied = [a for a, v in zip(tied, tb) if np.isclose(v, best, rtol=0.0, atol=1e-12)]
                if ties is not None and len(tied) == 1:
                    ties["broken"] = ties.get("broken", 0) + 1
            elif ties is not None:
                ties["unbreakable"] = ties.get("unbreakable", 0) + 1
        held = test[tied, i]
        scores[i] = float(held[0]) if np.all(held == held[0]) else float(held.mean())
        vals[i] = float(top)
        chosen.append(TIE_SEPARATOR.join(sorted(arms[a] for a in tied)))
    return scores, vals, chosen


def _validation_dataset_scores(
    block: pd.DataFrame, iters: Sequence, metric: str, validation_col: str,
    tiebreak_col: str | None = None, tiebreak_higher_is_better: bool = True,
    ties: dict | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Validation selection for one dataset: per ``(embeddings, iteration)`` and side.

    Each embedding is its own selection unit: on every fold the classical and quantum
    winners are the validation argmax among that embedding's models
    (:func:`_validation_side_scores`), so a contrast never pairs a classical arm on one
    embedding with a quantum arm on another. The dataset's score per fold, which feeds
    the one-row-per-dataset inference, is the mean over the embeddings whose two winners
    are both finite on that fold (so ``delta`` is the mean paired contrast); the folds,
    not the embeddings, stay the replicates, since the embeddings of a dataset share
    their splits.

    Returns:
        ``(side_scores, trace)``: per-side score arrays aligned with ``iters``, and one
        trace row per ``(embeddings, iteration)`` without the ``Dataset`` key.
    """
    n_iter = len(iters)
    per_emb, trace = [], []
    coord_cols = [c for c in _FOLD_COLUMNS if c in block.columns]
    coords = (block.groupby("iteration")[coord_cols].first().to_dict("index")
              if coord_cols else {})
    for emb, eblock in block.groupby("embeddings", sort=True, dropna=False):
        scores, vals, arms = {}, {}, {}
        for side in ("classical", "quantum"):
            sub = eblock[eblock["side"] == side]
            if sub.empty:
                scores[side] = np.full(n_iter, np.nan)
                vals[side] = np.full(n_iter, np.nan)
                arms[side] = [""] * n_iter
                continue
            wide = sub.pivot_table(index="arm", columns="iteration", values=metric,
                                   aggfunc="mean").reindex(columns=iters)
            vwide = (sub.pivot_table(index="arm", columns="iteration",
                                     values=validation_col, aggfunc="mean")
                     .reindex(index=wide.index, columns=iters))
            tb = None
            if tiebreak_col is not None:
                tb = (sub.pivot_table(index="arm", columns="iteration", values=tiebreak_col,
                                      aggfunc="mean", dropna=False)
                      .reindex(index=wide.index, columns=iters).to_numpy(float))
                tb = tb if tiebreak_higher_is_better else -tb
            scores[side], vals[side], arms[side] = _validation_side_scores(
                vwide.to_numpy(float), wide.to_numpy(float), list(wide.index), tb, ties)
        per_emb.append(scores)
        for j, it in enumerate(iters):
            row = dict(embeddings=emb, iteration=it)
            row.update(coords.get(it, {}))
            row.update(
                classical_arm=arms["classical"][j], quantum_arm=arms["quantum"][j],
                classical_score=scores["classical"][j],
                quantum_score=scores["quantum"][j],
                classical_val=vals["classical"][j], quantum_val=vals["quantum"][j],
            )
            trace.append(row)

    c = np.vstack([e["classical"] for e in per_emb])
    q = np.vstack([e["quantum"] for e in per_emb])
    paired = np.isfinite(c) & np.isfinite(q)

    def _mean(m: np.ndarray, ok: np.ndarray) -> np.ndarray:
        n = ok.sum(axis=0)
        tot = np.where(ok, m, 0.0).sum(axis=0)
        return np.where(n > 0, tot / np.maximum(n, 1), np.nan)

    if len(per_emb) == 1:
        # One embedding: each side keeps its own folds, exactly as the per-side means of
        # the LOIO path do; the delta is NaN wherever one side is missing.
        return {"classical": c[0], "quantum": q[0]}, trace
    # Several embeddings: average only the embeddings paired on that fold, and leave a
    # fold with no paired embedding NaN on both sides -- mixing one embedding's classical
    # winner with another's quantum winner is the cross-embedding contrast this avoids.
    side_scores = {"classical": _mean(c, paired), "quantum": _mean(q, paired)}
    return side_scores, trace


def resolve_split_k(results: pd.DataFrame, k: int | None = None) -> int:
    """Outer folds per repeat for validation selection: ``k``, else the ``split_k`` column.

    The corrected repeated-CV standard error needs ``r = n_test / n_train = 1 / (k - 1)``.
    ``split_mode: manifest`` records ``k`` on every row as ``split_k``; an explicit ``k``
    must agree with it.

    Raises:
        ValueError: neither is available, the column holds several values, the two
            disagree, or ``k < 2``.
    """
    col = None
    if "split_k" in results.columns:
        vals = pd.to_numeric(results["split_k"], errors="coerce").dropna().unique()
        if len(vals) > 1:
            raise ValueError(
                f"split_k takes several values {sorted(vals.tolist())}; one correction "
                f"r = 1/(k-1) cannot cover them. Analyse each group separately."
            )
        if len(vals) == 1:
            col = int(vals[0])
    if k is None:
        if col is None:
            raise ValueError(
                "validation selection needs k (outer folds per repeat) for the corrected "
                "repeated-CV t, r = 1/(k-1): pass k= or supply a 'split_k' column "
                "(split_mode: manifest writes it)."
            )
        k = col
    elif col is not None and int(k) != col:
        raise ValueError(f"k={k} disagrees with the split_k column ({col})")
    if int(k) < 2:
        raise ValueError(f"k must be >= 2; got {k!r}")
    return int(k)


def _validation_rows(results: pd.DataFrame, metric: str, validation_col: str) -> pd.DataFrame:
    """The rows validation selection compares, with one consistent ``tuning_metric``.

    Rows whose ``validation_col`` is not finite cannot be selected and are dropped with a
    log message naming the arms. A validation score is comparable across arms only when
    every arm was scored by the same metric, so a ``tuning_metric`` column holding more
    than one value among the kept rows raises.
    """
    if validation_col not in results.columns:
        raise ValueError(
            f"selection='validation' needs the validation column {validation_col!r}; "
            f"present: {sorted(results.columns)}"
        )
    val = pd.to_numeric(results[validation_col], errors="coerce").to_numpy(float)
    # Rows without a test score are dropped by arm_iteration_table anyway; count only the
    # ones this rule removes.
    test_ok = (pd.to_numeric(results[metric], errors="coerce").notna().to_numpy()
               if metric in results.columns else np.ones(len(results), dtype=bool))
    bad = ~np.isfinite(val) & test_ok
    if bad.any():
        dropped = results.loc[bad]
        arms = sorted(set((dropped["embeddings"].astype(str) + "|"
                           + dropped["model"].astype(str)).tolist())) \
            if {"embeddings", "model"} <= set(dropped.columns) else []
        logger.warning(
            "excluding %d row(s) with no finite %r from validation selection (arms: %s); "
            "an arm without a validation score cannot be chosen.",
            int(bad.sum()), validation_col,
            ", ".join(arms[:20]) + (" ..." if len(arms) > 20 else ""),
        )
    kept = results.loc[~bad]
    if "tuning_metric" in kept.columns:
        metrics = sorted(set(kept["tuning_metric"].dropna().astype(str)))
        if len(metrics) > 1:
            raise ValueError(
                f"the compared rows were tuned on different metrics {metrics}; validation "
                f"scores on different metrics are not comparable across arms. Filter the "
                f"frame to one tuning_metric."
            )
        if kept["tuning_metric"].isna().any() and metrics:
            raise ValueError(
                f"some compared rows have no tuning_metric while others use {metrics}; "
                f"their validation scores cannot be assumed comparable."
            )
    else:
        logger.info("no 'tuning_metric' column; assuming every %r used the same metric",
                    validation_col)
    return kept


def _check_split_mode(results: pd.DataFrame, mode: str) -> None:
    """Refuse validation selection over rows that are not ``split_mode: manifest``.

    Internal-mode rows carry a ``tuning_score`` too (from their inner CV or holdout),
    but their iterations are random holdouts, not the folds ``r = 1/(k-1)`` assumes, so
    pooling them into a validation-mode report would mis-scale every interval. LOIO
    over manifest rows is valid but not what the protocol calls for, so it is logged.
    Frames without a ``split_mode`` column (older results) are not checked.

    Raises:
        ValueError: ``mode == 'validation'`` and some row's ``split_mode`` is not
            ``'manifest'``.
    """
    if "split_mode" not in results.columns:
        return
    modes = results["split_mode"].fillna("internal").astype(str)
    if mode == "validation":
        other = sorted(set(modes[modes != "manifest"]))
        if other:
            raise ValueError(
                f"selection='validation' needs split_mode 'manifest' rows; "
                f"{int((modes != 'manifest').sum())} row(s) have split_mode {other}. "
                f"Analyse internal-mode rows with selection='loio'."
            )
    elif (modes == "manifest").any():
        logger.warning(
            "selection=%r over %d split_mode 'manifest' row(s); those runs were designed "
            "for selection='validation'.", mode, int((modes == "manifest").sum()))


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, returned in the input order."""
    m = p.size
    if m == 0:
        return p.astype(float)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def _holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error), in the input order."""
    m = p.size
    if m == 0:
        return p.astype(float)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * (m - np.arange(m))
    adj = np.clip(np.maximum.accumulate(ranked), 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def select_winners(
    results: pd.DataFrame,
    metric: str = "f1_score",
    epsilon: float = 0.027,
    alpha: float = 0.05,
    test_size: float | None = None,
    seed: int = 0,
    baseline: pd.DataFrame | None = None,
    *,
    margin: float = 0.0,
    fdr: float = 0.10,
    controls: Iterable[str] | None = None,
    selection: str = "loio",
    validation_col: str = "tuning_score",
    k: int | None = None,
    tiebreak_col: str | None = "val_auc",
    tiebreak_higher_is_better: bool = True,
) -> WinnerReport:
    """Decide, per dataset, whether either side wins, and by how much.

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
        The TOST equivalence bound, and nothing else: a dataset is ``equivalent`` when
        the whole interval lies inside ``(-epsilon, +epsilon)``. Equivalence cannot be
        certified when ``epsilon`` is below the interval half-width, so check it against
        :func:`iteration_floor_half_width` first.
    alpha
        Level of the per-dataset intervals (``1 - alpha`` coverage), and of the Holm
        family of ``controls``.
    test_size
        Test fraction of the ``train_test_split`` that produced every iteration.
        Required with ``selection='loio'``: the Nadeau-Bengio correction
        ``r = test_size / (1 - test_size)`` depends on it and ModelResults.csv does not
        record it. The pilot used 0.2. With ``selection='validation'`` it is ``1/k`` and
        may be omitted; a value other than ``1/k`` raises.
    seed
        Unused. Kept so existing calls still work; ties are now resolved by averaging
        (see :func:`_loio_side_scores`), so results do not depend on any seed.
    baseline
        Optional ``Dataset`` -> baseline-metric frame (columns ``Dataset`` and
        ``metric``). Datasets where neither side beats it are marked
        ``below_baseline``.
    margin
        Pre-registered practical superiority margin. A raw win needs the whole interval
        beyond ``-margin`` (quantum) or ``+margin`` (classical), and ``p_value`` tests
        ``|delta| <= margin``. The default 0 tests against zero.
    fdr
        Benjamini-Hochberg level for the discovery family, in ``(0, 1)``. With ``fdr <= alpha`` the
        adjusted verdict can only demote a raw win; with ``fdr > alpha`` BH may also
        confirm a dataset whose raw ``p_value`` lies in ``(alpha, BH threshold]``.
    controls
        Dataset names forming a separate family, e.g. synthetic positive and negative
        controls. They are left out of the BH family, get Holm-adjusted p-values among
        themselves with ``verdict_adjusted`` judged at ``alpha``, and are excluded from
        :attr:`WinnerReport.quantum_datasets` / ``classical_datasets`` and from the
        across-dataset inference. A name absent from ``results``, or with no finite
        ``metric`` values, raises ``ValueError``.
    selection
        ``'loio'`` (default): leave-one-iteration-out nested selection, for repeated
        random holdouts (``split_mode: internal``). ``'validation'``: for
        ``split_mode: manifest``, where every outer fold has its own validation rows.
        Per ``(Dataset, embeddings, iteration)`` and side the chosen model is the
        argmax of ``validation_col`` over that embedding's models finite in both
        ``validation_col`` and ``metric``, and it is scored by ``metric`` on that fold's
        test rows; a contrast never pairs two embeddings. A dataset's fold contrast is
        the mean paired contrast over its embeddings with both winners finite on that
        fold (see :func:`_validation_dataset_scores`). Ties within
        ``atol=1e-12`` average the tied arms' test scores and join their labels with
        :data:`TIE_SEPARATOR`. The interval is the corrected repeated-CV t of
        Bouckaert & Frank (2004) / Nadeau & Bengio (2003): ``r = 1/(k-1)``, ``n = kR``
        folds, ``df = n - 1``. Rows with a non-finite ``validation_col`` are excluded
        (logged), the compared rows must share one ``tuning_metric``, and a
        ``split_mode`` column, when present, must read ``'manifest'`` on every row.
    validation_col
        Validation-score column for ``selection='validation'``; higher is better.
    k
        Outer folds per repeat for ``selection='validation'``. Defaults to the
        ``split_k`` column; one of the two is required.
    tiebreak_col
        ``selection='validation'`` only: a continuous validation score that breaks ties
        on ``validation_col``. A validation fold of ten-odd rows gives balanced accuracy
        only a few values, so arms tie often; the default ``'val_auc'`` (the refit
        trial's validation AUC, written on manifest-mode rows) ranks any arm, decision
        functions included. A tie stands -- and the tied arms' test scores are averaged --
        when a tied arm has no ``tiebreak_col`` value or the values tie too. ``None``, or
        a frame without the column, keeps the plain averaging rule.
    tiebreak_higher_is_better
        Direction of ``tiebreak_col``; pass ``False`` for a loss such as
        ``'val_log_loss'`` (finite only for arms that output probabilities).

    Returns
    -------
    WinnerReport
        ``per_dataset`` has one row per dataset with ``delta``, ``se``, ``ci_lo``,
        ``ci_hi``, ``p_value``, ``within_equivalence``, ``verdict_raw``, ``family``
        (``'discovery'`` or ``'control'``), ``p_adjusted`` (BH q-value, or Holm for
        controls) and ``verdict_adjusted``. ``selection`` has one row per
        ``(Dataset, iteration)`` with the chosen arms and their ``classical_score`` /
        ``quantum_score``; with ``selection='validation'`` it has one row per
        ``(Dataset, embeddings, iteration)`` instead, with ``repeat`` and ``fold`` (when
        the frame has them) and the chosen arms' validation scores ``classical_val`` /
        ``quantum_val``.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"selection must be one of {SELECTION_MODES}; got {selection!r}")
    mode = selection  # the name `selection` is reused for the trace frame below
    by_validation = mode == "validation"
    _check_split_mode(results, mode)
    if by_validation:
        k = resolve_split_k(results, k)
        if test_size is not None and not np.isclose(float(test_size), 1.0 / k,
                                                    rtol=0.0, atol=1e-9):
            raise ValueError(
                f"test_size={test_size!r} contradicts k={k}: with k-fold outer splits "
                f"each test fold is 1/k = {1.0 / k:.4g} of the rows. Omit test_size."
            )
        test_size = 1.0 / k
        results = _validation_rows(results, metric, validation_col)
    else:
        k = None
    if test_size is None:
        raise ValueError(
            "test_size is required: the Nadeau-Bengio correction r = test_size / "
            "(1 - test_size) depends on the train/test split fraction, which "
            "ModelResults.csv does not record, so it cannot be inferred. Pass the value "
            "the sweep used (the pilot used test_size=0.2)."
        )
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError(f"test_size must lie in (0, 1); got {test_size!r}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0; got {margin!r}")
    if not 0.0 < float(fdr) < 1.0:
        raise ValueError(f"fdr must lie in (0, 1); got {fdr!r}")
    if isinstance(controls, str):
        controls = [controls]
    control_set = frozenset(controls) if controls is not None else frozenset()
    del seed  # accepted for backward compatibility only; nothing below is random

    unknown = sorted(control_set - set(results["Dataset"].unique()))
    if unknown:
        raise ValueError(
            f"controls name datasets absent from results: {unknown}. Control names must "
            f"match the Dataset column exactly."
        )
    tb_col = None
    if by_validation and tiebreak_col is not None:
        if tiebreak_col in results.columns:
            tb_col = tiebreak_col
        else:
            logger.info("no %r column: validation ties are averaged, not broken", tiebreak_col)
    table = arm_iteration_table(results, metric=metric,
                                validation_col=validation_col if by_validation else None,
                                extra_cols=(tb_col,) if tb_col else ())
    unscorable = sorted(control_set - set(table["Dataset"].unique()))
    if unscorable:
        raise ValueError(
            f"controls name datasets with no finite {metric!r} values: {unscorable}. "
            f"A control that cannot be scored cannot serve as a control."
        )

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

    # r = n_test / n_train. For k-fold outer splits (test_size = 1/k) this is 1/(k-1),
    # the Bouckaert-Frank correction for repeated k-fold CV.
    r = test_size / (1.0 - test_size)
    rows, sel_rows = [], []
    ties: dict = {}
    for dataset, block in table.groupby("Dataset", sort=True):
        iters = sorted(block["iteration"].unique())
        if by_validation:
            side_scores, trace = _validation_dataset_scores(
                block, iters, metric, validation_col, tb_col, tiebreak_higher_is_better, ties)
            sel_rows.extend({"Dataset": dataset, **t} for t in trace)
        else:
            side_scores, side_arms = {}, {}
            for side in ("classical", "quantum"):
                sub = block[block["side"] == side]
                if sub.empty:
                    side_scores[side] = np.full(len(iters), np.nan)
                    side_arms[side] = [""] * len(iters)
                    continue
                wide = sub.pivot_table(index="arm", columns="iteration", values=metric,
                                       aggfunc="mean").reindex(columns=iters)
                side_scores[side], side_arms[side] = _loio_side_scores(
                    wide.to_numpy(float), list(wide.index))
            for j, it in enumerate(iters):
                sel_rows.append(dict(
                    Dataset=dataset, iteration=it,
                    classical_arm=side_arms["classical"][j],
                    quantum_arm=side_arms["quantum"][j],
                    classical_score=side_scores["classical"][j],
                    quantum_score=side_scores["quantum"][j],
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
                       se=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_value=np.nan,
                       within_equivalence=False, verdict_raw="insufficient_iterations")
            rows.append(row)
            continue

        n = usable.size
        mean_d = float(usable.mean())
        s = float(usable.std(ddof=1))
        se = s * np.sqrt(1.0 / n + r)          # Nadeau & Bengio (2003)
        tcrit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        ci_lo, ci_hi = mean_d - tcrit * se, mean_d + tcrit * se

        # Two-sided test of H0: |delta| <= margin. With margin = 0 this is the ordinary
        # paired t-test, and p < alpha exactly when the interval excludes zero.
        if se > 0:
            p = float(min(1.0, 2.0 * stats.t.sf((abs(mean_d) - margin) / se, df=n - 1)))
        else:
            p = 0.0 if abs(mean_d) > margin else 1.0

        within = bool(ci_lo > -epsilon and ci_hi < epsilon)
        if ci_hi < -margin:
            verdict = "quantum_wins"
        elif ci_lo > margin:
            verdict = "classical_wins"
        elif within:
            verdict = "equivalent"
        else:
            verdict = "inconclusive"

        row.update(delta=mean_d, se=float(se), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                   p_value=p, within_equivalence=within, verdict_raw=verdict)
        rows.append(row)

    if by_validation and ties.get("tied"):
        tied = ties["tied"]
        if tb_col:
            how = (f"{ties.get('broken', 0)} broken by {tb_col}, "
                   f"{ties.get('unbreakable', 0)} averaged because a tied arm has no {tb_col}, "
                   f"the rest averaged because {tb_col} tied as well")
        else:
            how = "all averaged (no tie-break column)"
        logger.info("validation selection: %d side-folds tied on %s; %s", tied, validation_col, how)
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

    # ---- multiplicity: BH over the discovery family, Holm over the controls --------
    # The family is every dataset with a finite p-value, in both directions, fixed
    # before any verdict is looked at. Adjusting only the raw wins chose the family
    # post hoc and, with m = number of claims, could never demote a lone claim.
    # below_baseline rows keep their verdict but their p-values stay in the family,
    # which only makes the adjustment more conservative.
    per_dataset["family"] = pd.Series(dtype=object)
    per_dataset["verdict_adjusted"] = per_dataset.get("verdict_raw", pd.Series(dtype=object))
    per_dataset["p_adjusted"] = np.nan
    if not per_dataset.empty:
        is_control = per_dataset["Dataset"].isin(control_set)
        per_dataset["family"] = np.where(is_control, "control", "discovery")
        finite = np.isfinite(pd.to_numeric(per_dataset["p_value"], errors="coerce"))
        special = per_dataset["verdict_raw"].isin(_SPECIAL_VERDICTS)
        direction = np.where(per_dataset["delta"] < 0, "quantum_wins", "classical_wins")
        fallback = np.where(per_dataset["within_equivalence"].astype(bool),
                            "equivalent", "inconclusive")
        for mask, adjust, level in ((~is_control & finite, _bh_adjust, fdr),
                                    (is_control & finite, _holm_adjust, alpha)):
            if not mask.any():
                continue
            per_dataset.loc[mask, "p_adjusted"] = adjust(
                per_dataset.loc[mask, "p_value"].to_numpy(float))
            judged = mask & ~special
            hit = judged & (per_dataset["p_adjusted"] <= level)
            per_dataset.loc[hit, "verdict_adjusted"] = direction[hit.to_numpy()]
            miss = judged & ~hit
            per_dataset.loc[miss, "verdict_adjusted"] = fallback[miss.to_numpy()]

    sigma = float(per_dataset["se"].dropna().median()) if "se" in per_dataset else np.nan
    corpus = {
        "n_datasets": int(per_dataset["Dataset"].nunique()) if not per_dataset.empty else 0,
        "n_controls": len(control_set),
        "epsilon": epsilon,
        "margin": margin,
        "alpha": alpha,
        "fdr": fdr,
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
    if by_validation:
        corpus.update(selection=mode, validation_col=validation_col, k=k)
    # Controls are planted effects, not part of the corpus the claim is about.
    discovery = (per_dataset[per_dataset["family"] == "discovery"]
                 if "family" in per_dataset else per_dataset)
    corpus["across_datasets"] = corpus_inference(discovery, epsilon=epsilon, alpha=alpha,
                                                 margin=margin)
    if np.isfinite(sigma):
        # se already carries the NB inflation, so compare epsilon against the half-width
        # the corpus actually achieved rather than against a nominal s/sqrt(I).
        corpus["epsilon_is_reachable"] = bool(
            corpus["median_interval_half_width"] <= epsilon
        )
        if corpus["epsilon_is_reachable"] is False and by_validation:
            logger.warning(
                "epsilon=%.4f (the equivalence bound) is below the median achieved "
                "interval half-width %.4f, so equivalence cannot be certified on most "
                "datasets at k=%d: an interval wider than +/-epsilon can never fit inside "
                "it. More repeats will not close this -- the corrected repeated-CV "
                "standard error floors at s*sqrt(1/(k-1)). Use more outer folds k or "
                "widen epsilon.",
                epsilon, corpus["median_interval_half_width"], k,
            )
        elif corpus["epsilon_is_reachable"] is False:
            logger.warning(
                "epsilon=%.4f (the equivalence bound) is below the median achieved "
                "interval half-width %.4f, so equivalence cannot be certified on most "
                "datasets at test_size=%.2f: an interval wider than +/-epsilon can never "
                "fit inside it. Raising 'iter' will not close this -- the Nadeau-Bengio "
                "standard error floors at s*sqrt(test_size/(1-test_size)). Lower "
                "test_size or widen epsilon.",
                epsilon, corpus["median_interval_half_width"], test_size,
            )

    return WinnerReport(
        per_dataset=per_dataset, per_arm=per_arm, selection=selection,
        metric=metric, epsilon=epsilon, alpha=alpha, test_size=test_size, corpus=corpus,
        margin=margin, fdr=fdr, controls=tuple(sorted(control_set)),
        selection_mode=mode, k=k,
    )


__all__ = [
    "QUANTUM_STEMS",
    "PARAMETER_COLUMNS",
    "SELECTION_MODES",
    "TIE_SEPARATOR",
    "VERDICTS",
    "WinnerReport",
    "arm_iteration_table",
    "corpus_inference",
    "iteration_floor_half_width",
    "model_side",
    "resolve_split_k",
    "select_winners",
]
