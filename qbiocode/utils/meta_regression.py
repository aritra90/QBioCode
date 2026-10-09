"""Meta-regression of the quantum-minus-classical contrast on dataset meta-features.

Why this module exists
----------------------
:mod:`qbiocode.utils.fair_selection` decides *who wins on each dataset*. The question a
paper is asked next is *on what kind of dataset*: which measurable properties of the data
move the contrast toward quantum. QProfiler writes ~140 meta-features beside every score,
so the tempting design is one regression of the per-iteration contrast on all of them,
with a bootstrap over datasets for p-values. This module is that regression, restricted
to the parts of it that survive review.

The unit of independence is the dataset
---------------------------------------
A ModelResults frame holds one row per ``(Dataset, embeddings, iteration, model)``. The
iterations are resamples of the same rows, and the ``pca`` and ``umap`` passes of one
dataset share their train/test splits, so no row is independent of its siblings. On the
pilot about 96% of a typical meta-feature's variance lies between ``(Dataset,
embeddings)`` passes and the rest is resampling noise: 80 iteration rows carry about 16
distinct values of every covariate, drawn from 12 independent datasets. Keeping the rows
at iteration level is still right, because it carries the response's resampling noise
into the inference without a separate variance model. But every standard error and every
reference distribution here clusters on the dataset, and the degrees of freedom are
``G - 1`` for ``G`` datasets, not ``N - K`` for ``N`` rows. More iterations tighten each
dataset's contrast; only more datasets tighten a meta-feature's coefficient.

This is robust variance estimation in the meta-analysis sense (Hedges, Tipton & Johnson
2010): dependent effect sizes, clustered by study, with small-sample corrections because
the number of studies is small (Tipton 2015).

Screening is outcome-blind
--------------------------
With ``p`` meta-features against ``G`` datasets and ``p >> G``, no joint model is
estimable and VIF is infinite for every column. :func:`screen_features` reduces the
features *without looking at the response*. It drops non-finite, near-constant,
unreliable and host-dependent columns, collapses correlation clusters to their medoid,
and admits the medoids one at a time under a VIF ceiling. Because the response never
enters, the screen cannot inflate the tests that follow, and a reader can audit it from
the ledger it returns.

Why not a pairs cluster bootstrap on ridge coefficients
-------------------------------------------------------
Resampling datasets with replacement and reading a p-value off the ridge coefficients'
bootstrap distribution fails three ways at this ``G``:

- With 12 clusters a replicate holds about 7.8 distinct datasets, so a dataset-level
  coefficient rests on 7-8 support points and is unidentified in some replicates.
- The pairs cluster bootstrap over-rejects when clusters are few (Cameron, Gelbach & Miller
  2008; MacKinnon & Webb 2017).
- A ridge coefficient is shrunk toward zero and shared across correlated columns. Its
  bootstrap distribution is centred on a biased estimand, so a percentile "p-value" does
  not test ``beta = 0``.

Ridge is kept for what it is good at, which is prediction, and tested there.
:func:`ridge_omnibus` asks whether the meta-features predict the contrast on *held-out
datasets* better than the embedding alone, with a permutation p-value for the whole
nested cross-validation. :func:`null_calibration` runs the pairs bootstrap beside the
tests used here on permuted responses, so its level on the actual design is measured
rather than asserted.

What the per-feature tests are
------------------------------
Each feature is tested in a marginal model: the feature, the embedding indicators and an
intercept. A joint model needs about ten datasets per covariate (Cochrane Handbook,
sec. 10.11.4), and the pilot has 12 datasets in all.

- Primary p-value: the wild cluster restricted bootstrap-t with Webb six-point weights
  (Webb 2023; Roodman et al. 2019). It holds its level with about ten clusters, where
  neither the pairs bootstrap nor the analytic CR1 t-test does.
- Two analytic references sit beside it, both on ``G - 1`` degrees of freedom: CR1 and
  the jackknife CV3 (MacKinnon, Nielsen & Webb 2023).
- A studentized block-permutation p-value, exact when datasets of the same shape are
  exchangeable.

Multiplicity is controlled across the screened family by Benjamini-Hochberg. The
Benjamini-Yekutieli adjustment is reported alongside, because correlated meta-features are
not guaranteed to satisfy BH's positive-dependence condition.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from qbiocode.evaluation.model_evaluation import TUNING_EVIDENCE_COLUMNS
from qbiocode.evaluation.protocol import PROTOCOL_COLUMNS, TIEBREAK_COLUMNS
from qbiocode.utils.fair_selection import PARAMETER_COLUMNS, WinnerReport, select_winners

logger = logging.getLogger(__name__)

#: One response row: a resample of one embedding pass of one dataset.
INSTANCE = ("Dataset", "embeddings", "iteration")
#: One embedding pass of one dataset, the level at which meta-features vary.
UNIT = ("Dataset", "embeddings")
#: Score and cost columns QProfiler writes beside the meta-features.
METRIC_COLUMNS = ("accuracy", "f1_score", "balanced_accuracy", "mcc", "time", "auc", "pr_auc")
#: Joins a dataset and an embedding into one label, so select_winners can run per pass.
UNIT_SEP = "|"
#: Where each feature leaves the screen, in order.
STAGES = ("nonfinite", "near_constant", "unreliable", "host_dependent", "redundant", "vif",
          "cap", "selected")

#: Webb's six-point distribution has mean 0 and variance 1. At G = 12 it gives 6**12
#: distinct weight vectors, where Rademacher gives 2**12 = 4096. That is few enough to
#: make a Rademacher p-value visibly discrete.
WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

#: Ridge penalties are ``grid * n_train``. Features are standardised, so this spans
#: near-OLS to near-total shrinkage at any sample size.
RIDGE_GRID = np.logspace(-4, 3, 29)


# ---- meta-features ------------------------------------------------------------------

def meta_feature_columns(results: pd.DataFrame) -> list[str]:
    """Numeric columns of a ModelResults frame that describe the data, not the model.

    The tuning evidence of tuned rows (``tuning_score``, ``tuning_reused``) is numeric
    but label-dependent -- a validation score of the model -- so it is reserved too, as
    are the validation tie-break scores (``TIEBREAK_COLUMNS``: val_auc, val_log_loss). So
    are the ``split_mode: manifest`` provenance columns (``PROTOCOL_COLUMNS``: repeat,
    fold, split sizes, seeds, host): they describe how a row was produced, and a numeric
    ``fold`` or ``n_val`` would otherwise enter the screen as a dataset property.
    """
    reserved = (set(INSTANCE) | {"model"} | set(METRIC_COLUMNS) | set(PARAMETER_COLUMNS)
                | set(TUNING_EVIDENCE_COLUMNS) | set(PROTOCOL_COLUMNS) | set(TIEBREAK_COLUMNS))
    return [c for c in results.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(results[c])]


def canonical_meta_features(
    results: pd.DataFrame, features: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One value per instance and feature: the median over that instance's model rows.

    Every model row of an instance repeats the instance's meta-features, but the copies
    are not identical. Each row was computed in its own job, and the kNN- and graph-based
    measures move with the host's CPU type. Taking the first row would make a covariate
    depend on which model happened to sort first. The median does not depend on row order
    and resists the odd host.

    Returns
    -------
    X : DataFrame indexed by ``INSTANCE``. Non-finite values become NaN before the median.
    quality : DataFrame indexed by feature, with two columns:
        ``nonfinite_rows`` is the share of raw rows that were NaN or inf.
        ``host_sd`` is the pooled standard deviation across one instance's model rows (the
        root of the mean within-instance variance). A median over instances would read 0
        for a feature that disagrees on only a tenth of them, which is the usual pattern.
    """
    missing = [c for c in INSTANCE if c not in results.columns]
    if missing:
        raise ValueError(f"results is missing {missing}")
    features = list(features) if features is not None else meta_feature_columns(results)
    values = results[features].apply(pd.to_numeric, errors="coerce").astype(float)
    finite = np.isfinite(values.to_numpy())
    keyed = pd.concat([results[list(INSTANCE)].reset_index(drop=True),
                       values.where(finite).reset_index(drop=True)], axis=1)
    grouped = keyed.groupby(list(INSTANCE), sort=True)[features]
    X = grouped.median()
    quality = pd.DataFrame(
        {"nonfinite_rows": 1.0 - finite.mean(axis=0),
         "host_sd": np.sqrt(grouped.var(ddof=1).mean()).reindex(features).to_numpy()},
        index=pd.Index(features, name="feature"),
    )
    return X, quality


def unit_means(X: pd.DataFrame) -> pd.DataFrame:
    """Per-pass means of instance-level values: the between (Mundlak) component."""
    return X.groupby(level=list(UNIT), sort=True).mean()


def feature_reliability(X: pd.DataFrame, quality: pd.DataFrame | None = None) -> pd.DataFrame:
    """How well a pass's mean measures a stable property of that pass.

    ``reliability`` is ICC(1,k) = 1 - MSW/MSB (Bliese 2000): the reliability of a unit
    mean over its ``k`` resamples. A coefficient on an unreliable covariate is attenuated
    toward zero by about that factor. ``icc1`` is the single-resample ICC(1).
    ``host_noise`` is the spread across one instance's model rows (see
    :func:`canonical_meta_features`) as a share of the between-pass SD. A feature above
    about 0.25 differs by host about as much as by dataset.
    """
    grouped = X.groupby(level=list(UNIT), sort=True)
    n_u = grouped.count()
    mean_u = grouped.mean()
    N = n_u.sum()
    U = (n_u > 0).sum()
    grand = (mean_u * n_u).sum() / N
    msb = (n_u * (mean_u - grand) ** 2).sum() / (U - 1)
    ssw = ((X - grouped.transform("mean")) ** 2).sum()
    dfw = N - U
    msw = ssw / dfw.where(dfw > 0)
    n0 = (N - (n_u ** 2).sum() / N) / (U - 1)
    positive = msb > 0
    out = pd.DataFrame({
        "n_units": U,
        "between_sd": mean_u.std(ddof=1),
        "icc1": ((msb - msw) / (msb + (n0 - 1.0) * msw)).where(positive),
        "reliability": (1.0 - msw / msb).where(positive).clip(lower=0.0, upper=1.0),
    })
    out.index.name = "feature"
    if quality is not None:
        out = out.join(quality)
        out["host_noise"] = (out["host_sd"] / out["between_sd"]).where(out["between_sd"] > 0)
    return out


def variance_decomposition(X: pd.DataFrame) -> pd.DataFrame:
    """Share of each column's instance-level variance between datasets and between passes."""
    centred = X - X.mean()
    total = (centred ** 2).sum()
    out = {}
    for name, levels in (("between_datasets", ["Dataset"]), ("between_passes", list(UNIT))):
        fitted = X.groupby(level=levels).transform("mean") - X.mean()
        out[name] = (fitted ** 2).sum() / total
    return pd.DataFrame(out).where(total > 0)


def variance_inflation(frame: pd.DataFrame) -> pd.Series:
    """VIF of every column, each regressed on all the others plus an intercept.

    ``inf`` where a column is constant or an exact combination of the others. That is the
    state of the unscreened pilot, whose 16 x 139 matrix has rank 15.
    """
    A = frame.to_numpy(float)
    A = A - A.mean(axis=0)
    out = {}
    for j, name in enumerate(frame.columns):
        target = A[:, j]
        tss = float(target @ target)
        others = np.delete(A, j, axis=1)
        if tss <= 0:
            out[name] = np.inf
            continue
        if others.shape[1] == 0:
            out[name] = 1.0
            continue
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        rss = float(resid @ resid)
        out[name] = np.inf if rss <= 1e-12 * tss else tss / rss
    return pd.Series(out, name="vif", dtype=float)


# ---- outcome-blind screening --------------------------------------------------------

@dataclass
class Screen:
    """The result of :func:`screen_features`.

    ``Z`` holds every feature that reached the transform stage, standardised across
    passes, so a sensitivity analysis can reuse it with other admission rules.
    ``ledger`` says where each input feature left the screen and why.
    """

    selected: list
    Z: pd.DataFrame
    ledger: pd.DataFrame
    clusters: pd.DataFrame
    vif: pd.Series
    rules: dict = field(default_factory=dict)

    @property
    def features(self) -> pd.DataFrame:
        return self.Z[self.selected]

    def funnel(self) -> pd.DataFrame:
        """How many features left at each stage, in stage order."""
        counts = self.ledger["stage"].value_counts()
        return pd.DataFrame({"stage": STAGES,
                             "n_features": [int(counts.get(s, 0)) for s in STAGES]})


def _support(col: pd.Series) -> tuple[int, float]:
    """Distinct values, and the share of passes at the commonest one."""
    c = col.dropna().to_numpy(float)
    if c.size == 0:
        return 0, np.nan
    scale = float(np.max(np.abs(c))) or 1.0
    _, counts = np.unique(np.round(c / scale, 9), return_counts=True)
    return int(counts.size), float(counts.max() / c.size)


def screen_features(
    X_unit: pd.DataFrame,
    reliability: pd.DataFrame | None = None,
    nuisance: pd.DataFrame | None = None,
    *,
    min_reliability: float = 0.7,
    max_host_noise: float = 0.25,
    max_top_share: float = 0.8,
    log_range: float = 100.0,
    redundancy: float = 0.8,
    vif_max: float = 5.0,
    max_features: int | None = None,
) -> Screen:
    """Reduce the meta-features to a set a regression can hold, without seeing the response.

    The stages run in order, and each feature exits at the first one it fails:

    1. ``nonfinite``: missing on any pass.
    2. ``near_constant``: a single value, or more than ``max_top_share`` of the passes
       sharing one value. Such a column is a dummy for a few datasets, not a gradient.
    3. ``unreliable``: ICC(1,k) below ``min_reliability``. Features whose reliability
       cannot be estimated are kept.
    4. ``host_dependent``: ``host_noise`` above ``max_host_noise``.
    5. Transform. A strictly positive feature spanning a ratio of at least ``log_range``
       is log10-transformed, so that one large dataset does not set the slope, and every
       feature is z-scored across passes.
    6. ``redundant``: average-linkage clustering on ``1 - |Spearman rho|``, cut at
       ``1 - redundancy``. Each cluster keeps its medoid, the member with the highest mean
       |rho| to the others; ties go to higher reliability, then to the name.
    7. ``vif`` and ``cap``: medoids are offered in a fixed order (larger cluster first,
       then higher reliability, then name). Each one is admitted if every admitted
       feature's VIF stays at or below ``vif_max``, with the nuisance columns in the
       design. Admission stops at ``max_features``. It also stops, in any case, while the
       pass-level design keeps two degrees of freedom.

    Nothing here reads the response, so the selected set is fixed before any test is run.
    """
    rules = dict(min_reliability=min_reliability, max_host_noise=max_host_noise,
                 max_top_share=max_top_share, log_range=log_range, redundancy=redundancy,
                 vif_max=vif_max, max_features=max_features)
    X_unit = X_unit.astype(float)
    cols = list(X_unit.columns)
    rel = reliability.reindex(cols) if reliability is not None else pd.DataFrame(index=cols)
    r = (rel["reliability"] if "reliability" in rel else pd.Series(np.nan, index=cols)).astype(float)
    h = (rel["host_noise"] if "host_noise" in rel else pd.Series(np.nan, index=cols)).astype(float)
    support = {c: _support(X_unit[c]) for c in cols}

    ledger = pd.DataFrame(index=pd.Index(cols, name="feature"))
    ledger["stage"] = ""
    ledger["detail"] = ""
    ledger["reliability"] = r.to_numpy()
    ledger["host_noise"] = h.to_numpy()
    ledger["n_distinct"] = [support[c][0] for c in cols]
    ledger["top_share"] = [support[c][1] for c in cols]
    for name in ("transform", "representative"):
        ledger[name] = ""
    ledger["cluster"] = np.nan
    ledger["cluster_size"] = np.nan
    ledger["vif"] = np.nan

    alive = list(cols)

    def retire(names, stage, detail: Callable[[str], str]):
        names = [c for c in alive if c in set(names)]
        for c in names:
            ledger.loc[c, "stage"] = stage
            ledger.loc[c, "detail"] = detail(c)
        return [c for c in alive if c not in set(names)]

    n_pass = len(X_unit)
    alive = retire([c for c in alive if X_unit[c].isna().any()], "nonfinite",
                   lambda c: f"missing on {int(X_unit[c].isna().sum())} of {n_pass} passes")
    alive = retire([c for c in alive if support[c][0] < 2 or support[c][1] > max_top_share],
                   "near_constant",
                   lambda c: f"{support[c][0]} distinct value(s); {support[c][1]:.0%} of "
                             "passes share the commonest")
    alive = retire([c for c in alive if np.isfinite(r[c]) and r[c] < min_reliability],
                   "unreliable",
                   lambda c: f"reliability of a pass mean {r[c]:.2f} < {min_reliability:g}")
    alive = retire([c for c in alive if np.isfinite(h[c]) and h[c] > max_host_noise],
                   "host_dependent",
                   lambda c: f"spread across one instance's model rows is {h[c]:.2f} of "
                             f"the between-pass SD (> {max_host_noise:g})")
    unassessed = [c for c in alive if not np.isfinite(r[c])]
    if unassessed:
        logger.info("reliability could not be estimated for %d feature(s); kept: %s",
                    len(unassessed), ", ".join(unassessed[:10]))

    transformed = {}
    for c in alive:
        v = X_unit[c]
        if v.min() > 0 and v.max() / v.min() >= log_range:
            ledger.loc[c, "transform"] = "log10"
            transformed[c] = np.log10(v)
        else:
            ledger.loc[c, "transform"] = "identity"
            transformed[c] = v
    T = pd.DataFrame(transformed, index=X_unit.index, columns=alive)
    Z = (T - T.mean()) / T.std(ddof=1)

    if len(alive) >= 2:
        rho = T.rank().corr().abs().fillna(0.0)
        D = 1.0 - rho.to_numpy()
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2.0, 0.0, None)
        raw = fcluster(linkage(squareform(D, checks=False), method="average"),
                       t=1.0 - redundancy, criterion="distance")
    else:
        rho = pd.DataFrame(1.0, index=alive, columns=alive)
        raw = np.ones(len(alive), int)

    def rel_key(c):
        return -(r[c] if np.isfinite(r[c]) else -1.0)

    members_of = defaultdict(list)
    for c, k in zip(alive, raw):
        members_of[k].append(c)
    reps = {}
    for k, members in members_of.items():
        if len(members) == 1:
            reps[k] = members[0]
            continue
        sub = rho.loc[members, members].to_numpy()
        centrality = dict(zip(members, (sub.sum(axis=1) - 1.0) / (len(members) - 1)))
        reps[k] = sorted(members, key=lambda c: (-round(centrality[c], 12), rel_key(c), c))[0]
    order = sorted(members_of, key=lambda k: (-len(members_of[k]), rel_key(reps[k]), reps[k]))
    relabel = {k: i + 1 for i, k in enumerate(order)}
    for k, members in members_of.items():
        for c in members:
            ledger.loc[c, "cluster"] = relabel[k]
            ledger.loc[c, "cluster_size"] = len(members)
            ledger.loc[c, "representative"] = reps[k]
    alive = retire([c for k, m in members_of.items() for c in m if c != reps[k]], "redundant",
                   lambda c: f"cluster {int(ledger.at[c, 'cluster'])}: represented by "
                             f"{ledger.at[c, 'representative']} (|rho| = "
                             f"{rho.at[c, ledger.at[c, 'representative']]:.2f})")

    if nuisance is not None and nuisance.shape[1]:
        nuis = nuisance.reindex(X_unit.index).astype(float)
    else:
        nuis = pd.DataFrame(index=X_unit.index)
    ceiling = n_pass - nuis.shape[1] - 2
    cap = ceiling if max_features is None else min(int(max_features), ceiling)
    selected = []
    for k in order:
        c = reps[k]
        if len(selected) >= cap:
            alive = retire([c], "cap", lambda c: f"{cap} features already admitted")
            continue
        trial = selected + [c]
        v = variance_inflation(pd.concat([nuis, Z[trial]], axis=1))[trial]
        worst = v.idxmax()
        if v[worst] <= vif_max:
            selected.append(c)
            continue
        if worst == c:
            msg = f"VIF {v[c]:.1f} against the {len(selected)} admitted feature(s)"
        else:
            msg = f"admitting it would raise the VIF of {worst} to {v[worst]:.1f}"
        alive = retire([c], "vif", lambda _c, msg=msg: msg)

    final = (variance_inflation(pd.concat([nuis, Z[selected]], axis=1))[selected]
             if selected else pd.Series(dtype=float, name="vif"))
    for c in selected:
        ledger.loc[c, "stage"] = "selected"
        ledger.loc[c, "vif"] = final[c]
        ledger.loc[c, "detail"] = f"VIF {final[c]:.2f}"

    clusters = pd.DataFrame([
        dict(cluster=relabel[k], representative=reps[k], size=len(m),
             outcome=ledger.at[reps[k], "stage"], members=", ".join(sorted(m)))
        for k, m in members_of.items()
    ], columns=["cluster", "representative", "size", "outcome", "members"])
    clusters = clusters.sort_values("cluster").reset_index(drop=True)
    return Screen(selected=selected, Z=Z, ledger=ledger, clusters=clusters, vif=final,
                  rules=rules)


def embedding_indicators(index: pd.MultiIndex, reference: str = "none") -> pd.DataFrame:
    """One-hot embedding columns for a pass-indexed frame, ``reference`` left out.

    When ``reference`` is absent, the first embedding in sorted order is the reference.
    Columns that are constant on ``index`` are dropped.
    """
    emb = pd.Series(index.get_level_values("embeddings").astype(str), index=index)
    levels = sorted(set(emb))
    base = reference if reference in levels else levels[0]
    out = pd.DataFrame({f"emb[{lv}]": (emb == lv).astype(float)
                        for lv in levels if lv != base}, index=index)
    return out.loc[:, out.nunique() > 1]


# ---- responses ----------------------------------------------------------------------

def _as_units(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["Dataset"] = (results["Dataset"].astype(str) + UNIT_SEP
                      + results["embeddings"].astype(str))
    return out


def _split_units(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "Dataset" not in frame.columns:
        return frame
    parts = frame["Dataset"].astype(str).str.rsplit(UNIT_SEP, n=1, expand=True)
    out = frame.drop(columns=["embeddings"], errors="ignore").copy()
    out["Dataset"] = parts[0].to_numpy()
    out.insert(out.columns.get_loc("Dataset") + 1, "embeddings", parts[1].to_numpy())
    return out


def unit_report(results: pd.DataFrame, metric: str = "balanced_accuracy",
                epsilon: float = 0.027, alpha: float = 0.05, test_size: float | None = None,
                seed: int = 0, *, margin: float = 0.0, fdr: float = 0.10,
                controls: str | Iterable[str] | None = None, selection: str = "loio",
                validation_col: str = "tuning_score",
                k: int | None = None) -> WinnerReport:
    """:func:`select_winners` run once per ``(Dataset, embeddings)`` pass.

    Each arm is then a model within one embedding, and the selected contrast exists per
    pass: LOIO by default, or with ``selection='validation'`` (``split_mode: manifest``)
    the arm with the best ``validation_col`` on each ``(Dataset, embeddings, iteration)``.
    That is the response a pass-level meta-feature can explain. The per-pass verdicts
    are descriptive. A dataset's ``pca`` and ``umap`` passes share their splits, so the
    corpus-level inference that :func:`select_winners` computes assumes independence it
    does not have here, and is replaced by a note. Take corpus claims from the
    per-dataset report.

    The arguments are those of :func:`select_winners`. ``test_size`` is required with
    ``selection='loio'`` (``None`` raises there): it is the run's split fraction, 0.2 on
    the pilot, and ModelResults.csv does not record it. With ``selection='validation'``
    it is ``1/k``, ``k`` from the argument or the ``split_k`` column. ``controls`` name
    datasets, not passes; every pass of a control dataset joins the control family, and
    the returned report's ``controls`` holds those dataset names, matching its split-back
    ``Dataset`` column.
    """
    units = _as_units(results)
    if controls is not None:
        wanted = {controls} if isinstance(controls, str) else set(controls)
        present = set(results["Dataset"].astype(str))
        # Unknown names are passed through unchanged so select_winners reports them.
        controls = sorted({u for u, d in zip(units["Dataset"], results["Dataset"].astype(str))
                           if d in wanted} | (wanted - present))
    report = select_winners(units, metric=metric, epsilon=epsilon, alpha=alpha,
                            test_size=test_size, seed=seed, margin=margin, fdr=fdr,
                            controls=controls, selection=selection,
                            validation_col=validation_col, k=k)
    note = ("per-pass report: passes of one dataset share their splits, so corpus-level "
            "inference must come from select_winners on the per-dataset frame")
    return dataclasses.replace(report, per_dataset=_split_units(report.per_dataset),
                               per_arm=_split_units(report.per_arm),
                               selection=_split_units(report.selection),
                               controls=tuple(sorted({c.rsplit(UNIT_SEP, 1)[0]
                                                      for c in report.controls})),
                               corpus={"note": note})


def loio_contrast(report: WinnerReport) -> pd.DataFrame:
    """Per-instance ``qadv = quantum_score - classical_score`` from a :func:`unit_report`.

    Positive means quantum ahead. That is the opposite sign to ``delta`` in
    :func:`select_winners`, chosen so that a positive coefficient reads as "this property
    favours quantum". Both scores are of arms chosen on the *other* iterations (LOIO) or
    on the fold's own validation rows (``unit_report(..., selection='validation')``), so
    the contrast carries no winner's curse. The selection mode is the report's; the
    trace columns ``repeat``, ``fold``, ``classical_val`` and ``quantum_val`` pass through
    when present.
    """
    sel = report.selection
    need = set(INSTANCE) | {"classical_score", "quantum_score"}
    if need - set(sel.columns):
        raise ValueError(f"selection lacks {sorted(need - set(sel.columns))}; pass the "
                         "report returned by unit_report")
    out = sel.copy()
    out["qadv"] = out["quantum_score"] - out["classical_score"]
    return out.sort_values(list(INSTANCE)).reset_index(drop=True)


def matched_contrast(results: pd.DataFrame, quantum_model: str, classical_model: str,
                     metric: str = "balanced_accuracy") -> pd.DataFrame:
    """Per-instance ``qadv`` for one fixed pair of models, with no selection step.

    A sensitivity analysis for the LOIO contrast. A pre-specified pair (e.g. qsvc vs
    svc, which share a kernel machine) has no selection step to go wrong, at the cost of
    answering a narrower question.
    """
    sub = results[results["model"].isin([quantum_model, classical_model])]
    wide = sub.pivot_table(index=list(INSTANCE), columns="model", values=metric,
                           aggfunc="mean")
    missing = {quantum_model, classical_model} - set(wide.columns)
    if missing:
        raise ValueError(f"no rows for {sorted(missing)}")
    out = (wide[quantum_model] - wide[classical_model]).rename("qadv").dropna().reset_index()
    out["quantum_arm"], out["classical_arm"] = quantum_model, classical_model
    return out


def degenerate_arms(results: pd.DataFrame, metric: str = "balanced_accuracy",
                    chance: float = 0.5, tol: float = 1e-9) -> pd.DataFrame:
    """Per ``(Dataset, embeddings, model)``: whether the arm never beat chance."""
    out = (results.groupby(list(UNIT) + ["model"], sort=True)[metric]
           .agg(n="size", mean="mean", best="max").reset_index())
    out["at_chance"] = out["best"] <= chance + tol
    return out


def family_map(datasets: Iterable[str], patterns: Mapping[str, str] | None = None) -> dict:
    """Cluster label per dataset: the family of the first regex it matches, else itself.

    Synthetic variants generated from one base (``te_*_tau*`` or ``eng_zz`` with its
    ``_dmunit`` twin) share a generator, so they are one cluster, not several.
    """
    out = {}
    for d in datasets:
        out[d] = next((fam for pat, fam in (patterns or {}).items() if re.search(pat, d)), d)
    return out


#: Name prefixes of the synthetic generators. Every variant of one generator (taus,
#: seeds, ``_dmunit`` twins, parameter defaults) is one cluster, ``syn_<prefix>``.
SYNTHETIC_PREFIXES = ("eng_", "gs_", "hl_", "ql_", "te_")
#: Default cluster column of a mapping CSV: the conservative clustering.
CLUSTER_COLUMN = "cluster_cons"

#: Sources whose ids ``benchmark/create_synthetic_datasets.py`` writes as
#: ``<source>__<family>[_k<k>]_d<d>[_bw<bw>]_n<n>_s<seed>``. Every k, d, n and seed of one
#: family is a draw from one generator, so the family is the cluster: the three seeds of
#: ``checkerboard_k2_d8_n400`` are replicates, not three independent datasets.
GENERATED_SOURCES = ("shapes", "quantum")
#: Family of such an id: everything before the first ``_k<digit>`` or ``_d<digit>``. Matched
#: non-greedily so ``simple_linear_d8_n400_s0``, which has no ``k``, keeps both words.
_GENERATED_ID = re.compile(r"^(?P<family>.*?)_(?:k\d|d\d)")


def _split_source(stem: str) -> tuple[str, str]:
    """``'pmlb__GAMETES_x'`` -> ``('pmlb', 'GAMETES_x')``; no ``__`` -> ``('', stem)``.

    ``curate.py`` prefixes every curated id with its source, which the naming rules below
    have to look past: ``'pmlb__GAMETES_...'`` does not start with ``'GAMETES'``.
    """
    source, separator, rest = stem.partition("__")
    return (source, rest) if separator else ("", stem)


def _family_by_name(stem: str) -> str:
    """The cluster the naming rules give ``stem``, or ``stem`` itself if none applies.

    The fallback is the *full* id, not the source-stripped name: ``breast_cancer`` is both
    the libsvm Wisconsin set and the PMLB Ljubljana set, which are different datasets and
    must not be merged into one cluster.
    """
    source, bare = _split_source(stem)
    if source in GENERATED_SOURCES:
        match = _GENERATED_ID.match(bare)
        if match:
            return f"syn_{source}_{match.group('family')}"
    for prefix in SYNTHETIC_PREFIXES:
        if bare.startswith(prefix):
            return "syn_" + prefix.rstrip("_")
    if bare.startswith("GAMETES"):
        return "GAMETES"
    return stem


def _dataset_stem(name: str) -> str:
    """``'te_n10_tau0.25.csv|pca'`` -> ``'te_n10_tau0.25'``: drop the pass and ``.csv``."""
    stem = str(name).split(UNIT_SEP, 1)[0]
    return stem[:-len(".csv")] if stem.endswith(".csv") else stem


@functools.lru_cache(maxsize=8)
def _read_cluster_csv(path: str, mtime: float, key: str | None, column: str) -> dict:
    # ``mtime`` is only part of the cache key, so an edited CSV is re-read.
    table = pd.read_csv(path)
    if key is None:
        key = next((c for c in ("stem", "dataset", "Dataset") if c in table.columns), None)
    if key not in table.columns or column not in table.columns:
        raise ValueError(f"{path} needs a key column ({key or 'stem/dataset'}) and a "
                         f"{column!r} column; it has {list(table.columns)}")
    return _cluster_dict(zip(table[key], table[column]), source=f"{path}:{key}")


def _cluster_dict(pairs, source: str) -> dict:
    out, clash = {}, defaultdict(set)
    for name, cluster in pairs:
        # A blank cell would otherwise become the cluster 'nan' and silently join every
        # other unmapped stem into one fake cluster.
        if cluster is None or (not isinstance(cluster, str) and pd.isna(cluster)) \
                or str(cluster).strip() == "":
            raise ValueError(f"mapping {source} gives {name!r} no cluster")
        stem, cluster = _dataset_stem(name), str(cluster)
        if out.setdefault(stem, cluster) != cluster:
            clash[stem] |= {out[stem], cluster}
    if clash:
        detail = "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(clash.items()))
        raise ValueError(f"mapping {source} gives one stem two clusters ({detail})")
    return out


#: Stems already reported as missing from an explicit mapping (logged once each).
_UNMAPPED_LOGGED: set = set()


def dataset_family(name: str, mapping: Mapping[str, str] | str | None = None, *,
                   key: str | None = None, column: str = CLUSTER_COLUMN) -> str:
    """The cluster, i.e. the unit of independence, that a dataset belongs to.

    Variants of one generator, or copies of one source, are not independent datasets.
    Counting them separately inflates ``G`` and so every cluster-robust test in this
    module. ``name`` may carry a trailing ``.csv`` and a :data:`UNIT_SEP` pass suffix;
    both are dropped first. Then, in order:

    1. ``mapping``, if given and it lists the stem. A dict maps stem (or ``<stem>.csv``)
       to cluster. A string is the path of a CSV with a key column (``key``, else the
       first of ``stem``, ``dataset``, ``Dataset``) and the cluster column ``column``.
       A stem the mapping does not list falls through to the rules below, with a
       warning logged once per stem.
    Rules 2-4 look past the ``<source>__`` prefix that ``curate.py`` puts on every curated
    id, so they apply to ``pmlb__GAMETES_x`` as they do to a bare ``GAMETES_x``:

    2. A :data:`GENERATED_SOURCES` id (``shapes__``, ``quantum__``): the generator family,
       ``'syn_shapes_checkerboard'``. Every k, d, n and seed of one family is one cluster.
    3. A synthetic-generator prefix (:data:`SYNTHETIC_PREFIXES`): ``eng_*`` ->
       ``'syn_eng'``, and likewise for ``gs_``, ``hl_``, ``ql_`` and ``te_``.
    4. A name starting ``GAMETES``: ``'GAMETES'``, one generator.
    5. Otherwise the whole id -- the id, not the source-stripped name, because
       ``breast_cancer`` is both the libsvm Wisconsin set and the PMLB Ljubljana set.

    ``experiments/cluster_map_draft.csv`` (91 datasets; 59 conservative clusters in
    ``cluster_cons``, 67 liberal ones in ``cluster_lib``) covers the 78 real curated
    datasets exactly, the dedup of 6 cross-source copies having been executed in
    ``curate.py``'s ``EXCLUDE``. Its remaining 13 rows are ``qdata_x_view__*`` sets outside
    that corpus, and it lists no ``shapes__``/``quantum__`` dataset, so a synthetic corpus
    relies on rule 2. Its ``stem`` column is also not unique (``breast_cancer``, above), so
    that file raises unless keyed by ``key='dataset_id'``.

    Args:
        name: a ``Dataset`` value, with or without ``.csv`` and a pass suffix.
        mapping: optional explicit ``stem -> cluster`` dict, or a CSV path.
        key: key column of a CSV mapping; ``None`` picks ``stem``/``dataset``/``Dataset``.
        column: cluster column of a CSV mapping.

    Returns:
        str: the cluster label.

    Raises:
        ValueError: if ``mapping`` gives one stem two different clusters or a blank
            (NaN) cluster, or a CSV lacks the key or cluster column.
    """
    stem = _dataset_stem(name)
    unmapped = False
    if mapping is not None:
        if isinstance(mapping, (str, os.PathLike)):
            path = os.fspath(mapping)
            table = _read_cluster_csv(path, os.path.getmtime(path), key, column)
        else:
            table = _cluster_dict(mapping.items(), source="dict")
        if stem in table:
            return table[stem]
        unmapped = True
    cluster = _family_by_name(stem)
    # Warn only where the fallback leaves the dataset alone in its own cluster, which is
    # the case that inflates G. A rule that groups it (a generator family, GAMETES) has
    # done the job the mapping would have, and warning there buried the real hazard under
    # one line per synthetic dataset -- 93 of them in the shapes corpus.
    if unmapped and cluster == stem and stem not in _UNMAPPED_LOGGED:
        _UNMAPPED_LOGGED.add(stem)
        logger.warning("dataset_family: %r is in neither the explicit mapping nor any "
                       "naming rule, so it counts as its own cluster", stem)
    return cluster


# ---- the regression design ----------------------------------------------------------

@dataclass
class MetaDesign:
    """Instance-level rows with pass-level covariates broadcast onto them.

    ``features`` are standardised across passes, so a coefficient is the change in the
    response per pass-level SD. ``codes`` index ``clusters``, the units of independence.
    """

    rows: pd.DataFrame
    y: np.ndarray
    features: pd.DataFrame
    nuisance: pd.DataFrame
    codes: np.ndarray
    clusters: list

    @property
    def G(self) -> int:
        return len(self.clusters)

    @property
    def N(self) -> int:
        return int(self.y.size)

    def matrix(self, features: Sequence[str] | None = None) -> np.ndarray:
        cols = list(self.features.columns) if features is None else list(features)
        return np.hstack([np.ones((self.N, 1)), self.nuisance.to_numpy(float),
                          self.features[cols].to_numpy(float)])

    def penalty_mask(self, features: Sequence[str] | None = None) -> np.ndarray:
        k = self.features.shape[1] if features is None else len(features)
        return np.r_[np.zeros(1 + self.nuisance.shape[1]), np.ones(k)]

    def subset(self, keep) -> "MetaDesign":
        """The rows in ``keep``. Features keep their full-design scale, so coefficients
        stay comparable across sensitivity analyses."""
        keep = np.asarray(keep, bool)
        rows = self.rows.loc[keep].reset_index(drop=True)
        nuis = self.nuisance.loc[keep].reset_index(drop=True)
        codes, names = pd.factorize(rows["cluster"], sort=True)
        return MetaDesign(rows, self.y[keep], self.features.loc[keep].reset_index(drop=True),
                          nuis.loc[:, nuis.nunique() > 1], codes.astype(int), list(names))

    def with_nuisance(self, extra: pd.DataFrame) -> "MetaDesign":
        """Add pass-indexed nuisance columns (e.g. the tuning-budget ratio)."""
        key = pd.MultiIndex.from_frame(self.rows[list(UNIT)])
        add = extra.reindex(key).reset_index(drop=True).astype(float)
        if add.isna().any().any():
            raise ValueError("extra nuisance is missing for some passes")
        return dataclasses.replace(self, nuisance=pd.concat([self.nuisance, add], axis=1))

    def with_clusters(self, cluster_of: Mapping[str, str] | Callable[[str], str]) -> "MetaDesign":
        rows = self.rows.copy()
        rows["cluster"] = _cluster_labels(rows["Dataset"], cluster_of).to_numpy()
        codes, names = pd.factorize(rows["cluster"], sort=True)
        return dataclasses.replace(self, rows=rows, codes=codes.astype(int), clusters=list(names))


def _cluster_labels(datasets: pd.Series, cluster_of) -> pd.Series:
    datasets = datasets.astype(str)
    if cluster_of is None:
        return datasets
    if callable(cluster_of):
        return datasets.map(cluster_of)
    return datasets.map(lambda d: cluster_of.get(d, d))


def build_design(response: pd.DataFrame, features: pd.DataFrame,
                 nuisance: pd.DataFrame | None = None, *, y: str = "qadv",
                 cluster_of: Mapping[str, str] | Callable[[str], str] | None = None) -> MetaDesign:
    """Join an instance-level response to pass-level features and nuisance columns.

    Rows are sorted by ``INSTANCE``. Passes with a response but no features are dropped
    and logged.
    """
    resp = response.dropna(subset=[y])
    feats = features.copy()
    feats.index = feats.index.set_names(list(UNIT))
    merged = resp.merge(feats.reset_index(), on=list(UNIT), how="inner",
                        validate="many_to_one")
    if len(merged) < len(resp):
        lost = (resp[list(UNIT)].drop_duplicates()
                .merge(feats.reset_index()[list(UNIT)], how="left", indicator=True))
        lost = lost[lost["_merge"] == "left_only"]
        logger.info("%d response rows from %d pass(es) have no screened features and were "
                    "dropped: %s", len(resp) - len(merged), len(lost),
                    ", ".join(f"{a}/{b}" for a, b in lost[list(UNIT)].to_numpy()))
    merged = merged.sort_values(list(INSTANCE)).reset_index(drop=True)
    rows = merged[list(INSTANCE)].copy()
    rows["cluster"] = _cluster_labels(rows["Dataset"], cluster_of).to_numpy()
    codes, names = pd.factorize(rows["cluster"], sort=True)
    if nuisance is not None and nuisance.shape[1]:
        key = pd.MultiIndex.from_frame(merged[list(UNIT)])
        nuis = nuisance.reindex(key).reset_index(drop=True).astype(float)
        if nuis.isna().any().any():
            raise ValueError("nuisance is missing for some passes")
        nuis = nuis.loc[:, nuis.nunique() > 1]
    else:
        nuis = pd.DataFrame(index=range(len(merged)))
    return MetaDesign(rows=rows, y=merged[y].to_numpy(float),
                      features=merged[list(features.columns)].astype(float).reset_index(drop=True),
                      nuisance=nuis, codes=codes.astype(int), clusters=list(names))


# ---- inference primitives -----------------------------------------------------------

def _draw_weights(kind: str, G: int, B: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "webb":
        return WEBB_WEIGHTS[rng.integers(0, 6, size=(G, B))]
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(G, B))
    raise ValueError(f"unknown wild bootstrap weights {kind!r}")


def _as_2d(Y) -> np.ndarray:
    Y = np.asarray(Y, float)
    return Y[:, None] if Y.ndim == 1 else Y


class ClusteredOLS:
    """OLS on a fixed design, with every y-independent quantity precomputed.

    One matrix product then yields coefficients and cluster-robust t-statistics for
    thousands of responses at once. That is what makes bootstrap and permutation p-values
    cheap enough to calibrate by simulation.
    """

    def __init__(self, X, codes, G: int):
        X = np.asarray(X, float)
        N, K = X.shape
        if G < 2:
            raise ValueError("cluster-robust inference needs at least two clusters")
        if N <= K or np.linalg.matrix_rank(X) < K:
            raise np.linalg.LinAlgError("design is rank-deficient")
        self.X, self.N, self.K, self.G = X, N, K, int(G)
        self.codes = np.asarray(codes, int)
        self.A = np.linalg.inv(X.T @ X)
        self.H = self.A @ X.T
        self.C = np.zeros((self.G, N))
        self.C[self.codes, np.arange(N)] = 1.0
        # CR1, as Stata and statsmodels' cov_type="cluster" apply it.
        self.c1 = self.G / (self.G - 1) * (N - 1) / (N - K)
        self._jack = None

    def fit(self, Y):
        Y = _as_2d(Y)
        B = self.H @ Y
        return B, Y - self.X @ B

    def t_iid(self, Y, j: int):
        """The t-statistic that treats rows as independent. Reported to show the damage."""
        B, U = self.fit(Y)
        se = np.sqrt((U ** 2).sum(axis=0) / (self.N - self.K) * self.A[j, j])
        return B[j] / se, se

    def t_cr1(self, Y, j: int):
        B, U = self.fit(Y)
        S = self.C @ (self.H[j][:, None] * U)
        se = np.sqrt(self.c1 * (S ** 2).sum(axis=0))
        return B[j] / se, se

    def _jackknife_ops(self):
        if self._jack is None:
            ops = []
            for g in range(self.G):
                keep = self.codes != g
                Xg = self.X[keep]
                if np.linalg.matrix_rank(Xg) < self.K:
                    self._jack = False
                    break
                ops.append((keep, np.linalg.solve(Xg.T @ Xg, Xg.T)))
            else:
                self._jack = ops
        return self._jack or None

    def jackknife(self, Y, j: int):
        """Leave-one-cluster-out coefficients on column ``j``, shape ``G x R``.

        ``None`` if some cluster carries a column alone.
        """
        ops = self._jackknife_ops()
        if ops is None:
            return None
        Y = _as_2d(Y)
        return np.stack([Hg[j] @ Y[keep] for keep, Hg in ops])

    def t_cv3(self, Y, j: int):
        """CV3 of MacKinnon, Nielsen & Webb (2023): the cluster jackknife variance."""
        B, _ = self.fit(Y)
        jk = self.jackknife(Y, j)
        if jk is None:
            nan = np.full(B.shape[1], np.nan)
            return nan, nan
        se = np.sqrt((self.G - 1) / self.G * ((jk - B[j]) ** 2).sum(axis=0))
        return B[j] / se, se

    def wild_restricted(self, Y, j: int, n_boot: int, rng: np.random.Generator,
                        weights: str = "webb", chunk_cells: int = 4_000_000):
        """Wild cluster restricted bootstrap-t p-values for ``beta_j = 0``, per column of Y.

        The null is imposed: residuals come from the fit without column ``j``, and whole
        clusters' residuals are flipped by one weight each. The statistic is the CR1
        t, so the bootstrap refines an asymptotically pivotal quantity. That is why this
        test holds its level at G near 10, where the pairs bootstrap does not
        (Cameron, Gelbach & Miller 2008; Roodman et al. 2019). Each column gets its own
        weights, so p-values of different columns are independent draws.
        """
        Y = _as_2d(Y)
        N, R = Y.shape
        t_obs, _ = self.t_cr1(Y, j)
        Xr = np.delete(self.X, j, axis=1)
        fitted = Xr @ (np.linalg.pinv(Xr) @ Y) if Xr.shape[1] else np.zeros_like(Y)
        resid = Y - fitted
        thr = np.abs(t_obs) * (1.0 - 1e-12)
        exceed = np.zeros(R, int)
        cols = max(1, min(R, chunk_cells // (N * n_boot)))
        for start in range(0, R, cols):
            sl = slice(start, min(R, start + cols))
            r = sl.stop - start
            per = max(1, chunk_cells // (N * r))
            done = 0
            while done < n_boot:
                b = min(per, n_boot - done)
                V = _draw_weights(weights, self.G, r * b, rng).reshape(self.G, r, b)
                Ystar = fitted[:, sl, None] + resid[:, sl, None] * V[self.codes]
                tstar = self.t_cr1(Ystar.reshape(N, r * b), j)[0].reshape(r, b)
                exceed[sl] += (np.abs(tstar) >= thr[sl, None]).sum(axis=1)
                done += b
        p = (1.0 + exceed) / (1.0 + n_boot)
        return t_obs, np.where(np.isfinite(t_obs), p, np.nan)

    def permutation_p(self, y, j: int, perm_index: np.ndarray, chunk: int = 2000):
        """Studentized permutation p-value, permuting rows by ``perm_index`` (R x N)."""
        y = np.asarray(y, float)
        t_obs = float(self.t_cr1(y, j)[0][0])
        exceed = 0
        for start in range(0, perm_index.shape[0], chunk):
            tp = self.t_cr1(y[perm_index[start:start + chunk]].T, j)[0]
            exceed += int(np.count_nonzero(np.abs(tp) >= abs(t_obs) * (1.0 - 1e-12)))
        return (1.0 + exceed) / (1.0 + perm_index.shape[0])


def _cluster_shapes(rows: pd.DataFrame, codes) -> tuple[dict, dict]:
    """Per cluster: its row positions in canonical order, and the shape those rows make."""
    codes = np.asarray(codes, int)
    positions, signature = {}, {}
    for g in np.unique(codes):
        pos = np.flatnonzero(codes == g)
        sub = rows.iloc[pos]
        rank = pd.factorize(sub["Dataset"], sort=True)[0]
        emb = sub["embeddings"].astype(str).to_numpy()
        it = sub["iteration"].to_numpy()
        order = np.lexsort((it, emb, rank))
        positions[g] = pos[order]
        signature[g] = tuple(zip(rank[order], emb[order], it[order]))
    return positions, signature


def block_permutations(rows: pd.DataFrame, codes, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    """``n x N`` row indices that swap whole clusters' responses among same-shape clusters.

    Two clusters have the same shape when their rows line up one to one, matching on the
    dataset's rank within the cluster, the embedding and the iteration. The permutation
    moves a cluster's entire response block onto another cluster of that shape, so all
    within-cluster dependence travels with it. Under "the meta-features carry no
    information" the datasets are exchangeable within a shape, and this is the exact
    reference distribution for any statistic. Shapes held by one cluster alone never move.
    """
    positions, signature = _cluster_shapes(rows, codes)
    strata = defaultdict(list)
    for g, s in signature.items():
        strata[s].append(g)
    idx = np.tile(np.arange(len(rows)), (n, 1))
    for members in strata.values():
        if len(members) < 2:
            continue
        src = np.stack([positions[g] for g in members])
        flat = src.reshape(-1)
        for i in range(n):
            idx[i, flat] = src[rng.permutation(len(members))].reshape(-1)
    return idx


def permutation_strata(design: "MetaDesign") -> pd.DataFrame:
    """The strata :func:`block_permutations` permutes within, and how many orders each has.

    A stratum of one cluster never moves, so it adds nothing to the reference
    distribution.
    """
    _, signature = _cluster_shapes(design.rows, design.codes)
    strata = defaultdict(list)
    for g, s in signature.items():
        strata[s].append(design.clusters[g])
    rows = []
    for s, members in strata.items():
        rows.append(dict(embeddings=", ".join(sorted({e for _, e, _ in s})),
                         rows_per_cluster=len(s), n_clusters=len(members),
                         orderings=float(np.prod(np.arange(1, len(members) + 1, dtype=float))),
                         members=", ".join(sorted(members))))
    return pd.DataFrame(rows).sort_values("n_clusters", ascending=False).reset_index(drop=True)


def _pairs_bootstrap_betas(X, codes, G, Y, n_boot, rng, penalty=None, counts=None,
                           cond_max=1e10):
    """``B x K x R`` coefficients with clusters resampled with replacement.

    NaN where a replicate's design is singular, which happens when a replicate draws too
    few distinct clusters to identify an unpenalised column. Returns the coefficients,
    the validity mask and the per-replicate cluster counts.
    """
    Y = _as_2d(Y)
    if counts is None:
        counts = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)
    W = counts[:, np.asarray(codes, int)].astype(float)
    XtWX = np.einsum("bn,nj,nk->bjk", W, X, X, optimize=True)
    if penalty is not None:
        XtWX = XtWX + np.diag(penalty)[None]
    XtWY = np.einsum("bn,nj,nr->bjr", W, X, Y, optimize=True)
    ok = np.linalg.cond(XtWX) < cond_max
    out = np.full((counts.shape[0], X.shape[1], Y.shape[1]), np.nan)
    if ok.any():
        out[ok] = np.linalg.solve(XtWX[ok], XtWY[ok])
    return out, ok, counts


def _percentile_p(boot: np.ndarray) -> np.ndarray:
    """Twice the bootstrap mass on the far side of zero, over valid replicates (axis 0)."""
    valid = np.isfinite(boot)
    n = valid.sum(axis=0)
    lo = np.where(valid, boot <= 0, False).sum(axis=0) / np.maximum(n, 1)
    hi = np.where(valid, boot >= 0, False).sum(axis=0) / np.maximum(n, 1)
    return np.where(n > 0, np.minimum(1.0, 2.0 * np.minimum(lo, hi)), np.nan)


def fdr_adjust(p, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg (``"bh"``) or Benjamini-Yekutieli (``"by"``) adjusted p-values.

    NaNs are left in place and do not count toward the family size.
    """
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    if method not in ("bh", "by"):
        raise ValueError(f"unknown FDR method {method!r}")
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if method == "by" else 1.0
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * m * c / np.arange(1, m + 1)
    adj = np.clip(np.minimum.accumulate(scaled[::-1])[::-1], 0.0, 1.0)
    back = np.empty(m)
    back[order] = adj
    out[ok] = back
    return out


# ---- tests on the real design -------------------------------------------------------

def marginal_tests(design: MetaDesign, features: Sequence[str] | None = None, *,
                   n_boot: int = 9999, n_perm: int = 9999, weights: str = "webb",
                   alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """One model per feature, ``y ~ 1 + nuisance + feature``, clustered on ``design.clusters``.

    ``p_wcr`` is the primary p-value, and ``q_bh``/``q_by`` adjust it across the family
    passed in. The interval is the CV3 t-interval on ``G - 1`` df. ``p_ols_iid`` treats
    rows as independent. It is wrong by construction, and is kept only to show the size
    of the error.

    Cluster-robust inference needs more clusters than coefficients. When ``G`` does not
    exceed them (intercept, nuisance and the feature), or the fit is exact so the CR1
    standard error is zero, the cluster-robust columns (``se_*``, interval, ``t``,
    ``p_cr1``, ``p_cv3``, ``p_wcr`` and the q-values) are NaN and a warning is logged,
    instead of a near-zero standard error turning into ``p = 0``.
    """
    feats = list(design.features.columns) if features is None else list(features)
    boot_seed, perm_seed = np.random.SeedSequence(seed).spawn(2)
    boot_rng = np.random.default_rng(boot_seed)
    perm = (block_permutations(design.rows, design.codes, n_perm, np.random.default_rng(perm_seed))
            if n_perm else None)
    df = design.G - 1
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    rows, unidentified = [], []
    for f in feats:
        ols = ClusteredOLS(design.matrix([f]), design.codes, design.G)
        j = ols.K - 1
        beta = float(ols.H[j] @ design.y)
        t_i, _ = ols.t_iid(design.y, j)
        t_1, se_1 = ols.t_cr1(design.y, j)
        if design.G <= ols.K or not float(se_1[0]) > 1e-12 * (abs(beta) + 1.0):
            unidentified.append(f)
            rows.append(dict(
                feature=f, beta=beta, se_cr1=np.nan, se_cv3=np.nan, ci_lo=np.nan,
                ci_hi=np.nan, t=np.nan,
                p_ols_iid=float(2 * stats.t.sf(abs(t_i[0]), ols.N - ols.K)),
                p_cr1=np.nan, p_cv3=np.nan, p_wcr=np.nan,
                p_perm=ols.permutation_p(design.y, j, perm) if perm is not None else np.nan,
            ))
            continue
        t_3, se_3 = ols.t_cv3(design.y, j)
        _, p_wcr = ols.wild_restricted(design.y, j, n_boot, boot_rng, weights)
        rows.append(dict(
            feature=f, beta=beta, se_cr1=float(se_1[0]), se_cv3=float(se_3[0]),
            ci_lo=beta - tcrit * float(se_3[0]), ci_hi=beta + tcrit * float(se_3[0]),
            t=float(t_1[0]),
            p_ols_iid=float(2 * stats.t.sf(abs(t_i[0]), ols.N - ols.K)),
            p_cr1=float(2 * stats.t.sf(abs(t_1[0]), df)),
            p_cv3=float(2 * stats.t.sf(abs(t_3[0]), df)),
            p_wcr=float(p_wcr[0]),
            p_perm=ols.permutation_p(design.y, j, perm) if perm is not None else np.nan,
        ))
    if unidentified:
        logger.warning("cluster-robust inference is not identified for %d of %d features with "
                       "G = %d clusters (an exact fit, or no more clusters than the intercept, "
                       "nuisance and feature coefficients); their cluster-robust p-values are NaN: %s",
                       len(unidentified), len(feats), design.G, ", ".join(unidentified))
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = fdr_adjust(out["p_wcr"], "bh")
        out["q_by"] = fdr_adjust(out["p_wcr"], "by")
    return out


def joint_tests(design: MetaDesign, features: Sequence[str] | None = None, *,
                n_boot: int = 9999, weights: str = "webb", alpha: float = 0.05,
                seed: int = 0, min_clusters_per_feature: int = 10) -> pd.DataFrame | None:
    """All features in one model, if there are enough clusters to hold them.

    Returns ``None``, and logs why, when ``G < min_clusters_per_feature * k``. The rule
    is the meta-regression one (about ten studies per covariate). Below it, a joint
    coefficient is identified mostly by which few datasets happen to sit at the extremes.
    """
    feats = list(design.features.columns) if features is None else list(features)
    need = min_clusters_per_feature * len(feats)
    if design.G < need:
        logger.info("joint model skipped: %d clusters for %d features (needs %d)",
                    design.G, len(feats), need)
        return None
    rng = np.random.default_rng(seed)
    ols = ClusteredOLS(design.matrix(feats), design.codes, design.G)
    first = ols.K - len(feats)
    df = design.G - 1
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    rows = []
    for k, f in enumerate(feats):
        j = first + k
        beta = float(ols.H[j] @ design.y)
        t_1, se_1 = ols.t_cr1(design.y, j)
        t_3, se_3 = ols.t_cv3(design.y, j)
        _, p_wcr = ols.wild_restricted(design.y, j, n_boot, rng, weights)
        rows.append(dict(feature=f, beta=beta, se_cv3=float(se_3[0]),
                         ci_lo=beta - tcrit * float(se_3[0]), ci_hi=beta + tcrit * float(se_3[0]),
                         p_cr1=float(2 * stats.t.sf(abs(t_1[0]), df)),
                         p_cv3=float(2 * stats.t.sf(abs(t_3[0]), df)), p_wcr=float(p_wcr[0])))
    out = pd.DataFrame(rows)
    out["q_bh"] = fdr_adjust(out["p_wcr"], "bh")
    return out


def lodo_influence(design: MetaDesign, features: Sequence[str] | None = None) -> pd.DataFrame:
    """How far each marginal coefficient moves when one cluster is left out.

    Every row has the same columns. When dropping some cluster leaves the model
    unidentified (too few clusters for the nuisance and the feature), the
    leave-one-out columns are NaN (``most_influential`` None) rather than missing.
    """
    feats = list(design.features.columns) if features is None else list(features)
    rows = []
    for f in feats:
        ols = ClusteredOLS(design.matrix([f]), design.codes, design.G)
        j = ols.K - 1
        beta = float(ols.H[j] @ design.y)
        jk = ols.jackknife(design.y, j)
        if jk is None:
            rows.append(dict(feature=f, beta=beta, beta_min=np.nan, beta_max=np.nan,
                             sign_flips=np.nan, most_influential=None,
                             beta_without_it=np.nan))
            continue
        jk = jk[:, 0]
        worst = int(np.argmax(np.abs(jk - beta)))
        rows.append(dict(feature=f, beta=beta, beta_min=float(jk.min()), beta_max=float(jk.max()),
                         sign_flips=int(np.sum(np.sign(jk) != np.sign(beta))),
                         most_influential=design.clusters[worst],
                         beta_without_it=float(jk[worst])))
    return pd.DataFrame(rows, columns=["feature", "beta", "beta_min", "beta_max", "sign_flips",
                                       "most_influential", "beta_without_it"])


def within_unit_test(design: MetaDesign, x, *, n_boot: int = 9999, weights: str = "webb",
                     seed: int = 0) -> dict:
    """Test a within-pass contrast by demeaning ``y`` and ``x`` inside each pass.

    The pass fixed effects are nested in the clusters, so they are absorbed rather than
    counted in the CR1 correction (Cameron & Miller 2015). Used for the freeze diagnostic:
    ``x`` = 1 on the resample the quantum search ran on.
    """
    frame = design.rows[list(UNIT)].copy()
    frame["y"], frame["x"] = design.y, np.asarray(x, float)
    means = frame.groupby(list(UNIT))[["y", "x"]].transform("mean")
    yd = (frame["y"] - means["y"]).to_numpy()
    xd = (frame["x"] - means["x"]).to_numpy()
    ols = ClusteredOLS(xd[:, None], design.codes, design.G)
    beta = float(ols.H[0] @ yd)
    t_1, se_1 = ols.t_cr1(yd, 0)
    _, p_wcr = ols.wild_restricted(yd, 0, n_boot, np.random.default_rng(seed), weights)
    return dict(beta=beta, se_cr1=float(se_1[0]),
                p_cr1=float(2 * stats.t.sf(abs(t_1[0]), design.G - 1)), p_wcr=float(p_wcr[0]))


# ---- ridge: prediction, and the pairs bootstrap as proposed -------------------------

def _ridge_path_predictions(X, Y, train, test, lams, penalty):
    Xt = X[train]
    XtX, XtY = Xt.T @ Xt, Xt.T @ Y[train]
    P = np.diag(penalty)
    jitter = 1e-10 * np.eye(X.shape[1])
    out = np.empty((len(lams), int(test.sum()), Y.shape[1]))
    for i, lam in enumerate(lams):
        out[i] = X[test] @ np.linalg.solve(XtX + lam * P + jitter, XtY)
    return out


def _lodo_ridge_errors(X, Y, codes, G, grid, penalty):
    err = np.zeros((len(grid), Y.shape[1]))
    for g in range(G):
        test = codes == g
        train = ~test
        P = _ridge_path_predictions(X, Y, train, test, grid * train.sum(), penalty)
        err += ((P - Y[test][None]) ** 2).sum(axis=1)
    return err


def ridge_nested_cv(X, Y, codes, G, penalty, grid=RIDGE_GRID, max_inner_folds: int = 10):
    """Leave-one-cluster-out predictions with the penalty chosen inside each training set.

    Per outer cluster, the penalty is chosen by grouped CV over the *remaining* clusters
    alone (LODO while they number at most ``max_inner_folds``). So no prediction has seen
    its own cluster's response, through either the fit or the penalty. Each column of
    ``Y`` is handled independently. Returns predictions, the training-mean baseline
    predictions, and the chosen grid value per ``(outer cluster, column)``.
    """
    Y = _as_2d(Y)
    codes = np.asarray(codes, int)
    N, R = Y.shape
    grid = np.asarray(grid, float)
    pred, base, chosen = np.empty((N, R)), np.empty((N, R)), np.empty((G, R))
    for g in range(G):
        test = codes == g
        train = ~test
        ids = np.unique(codes[train])
        k = min(max_inner_folds, ids.size)
        fold_of = np.full(G, -1)
        fold_of[ids] = np.arange(ids.size) % k
        row_fold = fold_of[codes]
        err = np.zeros((grid.size, R))
        if grid.size > 1:
            for f in range(k):
                itest = train & (row_fold == f)
                itrain = train & ~itest
                P = _ridge_path_predictions(X, Y, itrain, itest, grid * itrain.sum(), penalty)
                err += ((P - Y[itest][None]) ** 2).sum(axis=1)
        best = err.argmin(axis=0)
        P = _ridge_path_predictions(X, Y, train, test, grid * train.sum(), penalty)
        pick = np.broadcast_to(best[None, None, :], (1, int(test.sum()), R))
        pred[test] = np.take_along_axis(P, pick, axis=0)[0]
        base[test] = Y[train].mean(axis=0)
        chosen[g] = grid[best]
    return pred, base, chosen


@dataclass
class RidgeOmnibus:
    """Out-of-cluster predictive R^2 of ridge on the features, over the nuisance-only model."""

    r2_full: float
    r2_nuisance: float
    delta_r2: float
    p_value: float
    n_perm: int
    null_delta_r2: np.ndarray
    chosen_grid: np.ndarray
    per_cluster: pd.DataFrame


def ridge_omnibus(design: MetaDesign, features: Sequence[str] | None = None, *,
                  n_perm: int = 999, seed: int = 0, grid=RIDGE_GRID,
                  max_inner_folds: int = 10, chunk: int = 250) -> RidgeOmnibus:
    """Do the features, jointly, predict the contrast on datasets the model never saw?

    The statistic is ``delta_r2``: the gain in leave-one-cluster-out R^2 from adding the
    features to the nuisance-only model. Both models are refitted from scratch on every
    block-permuted response, penalty search included. So the p-value accounts for the
    whole pipeline, and one test replaces the per-coefficient p-values that ridge cannot
    provide. R^2 is against the training-fold mean and can be negative.
    """
    rng = np.random.default_rng(seed)
    perm = block_permutations(design.rows, design.codes, n_perm, rng)
    Y = np.column_stack([design.y, design.y[perm].T])
    X_f, pen_f = design.matrix(features), design.penalty_mask(features)
    X_n, pen_n = design.matrix([]), design.penalty_mask([])
    delta = np.empty(Y.shape[1])
    keep = {}
    for start in range(0, Y.shape[1], chunk):
        sl = slice(start, min(Y.shape[1], start + chunk))
        pf, base, chosen = ridge_nested_cv(X_f, Y[:, sl], design.codes, design.G, pen_f,
                                           grid, max_inner_folds)
        pn, _, _ = ridge_nested_cv(X_n, Y[:, sl], design.codes, design.G, pen_n,
                                   np.array([0.0]), max_inner_folds)
        ss0 = ((Y[:, sl] - base) ** 2).sum(axis=0)
        delta[sl] = (((Y[:, sl] - pn) ** 2).sum(axis=0) - ((Y[:, sl] - pf) ** 2).sum(axis=0)) / ss0
        if start == 0:
            keep = dict(pf=pf[:, 0], pn=pn[:, 0], base=base[:, 0], chosen=chosen[:, 0],
                        ss0=ss0[0])
    y = design.y
    r2_f = 1.0 - float(((y - keep["pf"]) ** 2).sum()) / keep["ss0"]
    r2_n = 1.0 - float(((y - keep["pn"]) ** 2).sum()) / keep["ss0"]
    per = pd.DataFrame({"cluster": np.asarray(design.clusters)[design.codes],
                        "se_full": (y - keep["pf"]) ** 2, "se_nuisance": (y - keep["pn"]) ** 2})
    per = per.groupby("cluster", sort=True).mean()
    per["gain"] = per["se_nuisance"] - per["se_full"]
    null = delta[1:]
    p = (1.0 + np.count_nonzero(null >= delta[0] - 1e-12)) / (1.0 + null.size)
    return RidgeOmnibus(r2_full=r2_f, r2_nuisance=r2_n, delta_r2=float(delta[0]), p_value=p,
                        n_perm=int(null.size), null_delta_r2=null, chosen_grid=keep["chosen"],
                        per_cluster=per.reset_index())


def ridge_pairs_bootstrap(design: MetaDesign, features: Sequence[str] | None = None, *,
                          n_boot: int = 1999, seed: int = 0, grid=RIDGE_GRID) -> pd.DataFrame:
    """The proposal as posed: joint ridge, clusters resampled, percentile p-values.

    The penalty is chosen by leave-one-cluster-out CV on the observed response and then
    held fixed. Reported beside the primary table so the two can be compared, and
    calibrated in :func:`null_calibration`. ``frac_singular`` is the share of replicates
    whose design could not be solved. ``distinct_clusters`` is the mean number of
    distinct clusters a replicate holds.
    """
    feats = list(design.features.columns) if features is None else list(features)
    X, pen = design.matrix(feats), design.penalty_mask(feats)
    y = design.y[:, None]
    g_best = grid[_lodo_ridge_errors(X, y, design.codes, design.G, grid, pen)[:, 0].argmin()]
    lam = g_best * design.N
    beta = np.linalg.solve(X.T @ X + lam * np.diag(pen), X.T @ y)[:, 0]
    boot, ok, counts = _pairs_bootstrap_betas(X, design.codes, design.G, y, n_boot,
                                              np.random.default_rng(seed), penalty=lam * pen)
    first = X.shape[1] - len(feats)
    b = boot[:, first:, 0]
    out = pd.DataFrame({
        "feature": feats, "beta_ridge": beta[first:],
        "boot_lo": np.nanpercentile(b, 2.5, axis=0), "boot_hi": np.nanpercentile(b, 97.5, axis=0),
        "p_boot": _percentile_p(b),
    })
    out["q_bh"] = fdr_adjust(out["p_boot"], "bh")
    out.attrs.update(penalty_grid_value=float(g_best), frac_singular=float(1 - ok.mean()),
                     distinct_clusters=float((counts > 0).sum(axis=1).mean()))
    return out


# ---- calibration and power ----------------------------------------------------------

#: Methods compared in :func:`null_calibration`, in display order.
CALIBRATION_METHODS = {
    "ols_iid": "OLS, rows treated as independent",
    "pairs_ols": "pairs cluster bootstrap, OLS",
    "pairs_ridge": "pairs cluster bootstrap, joint ridge (proposed)",
    "cr1": "CR1 cluster-robust t, G-1 df",
    "cv3": "CV3 cluster jackknife t, G-1 df",
    "wcr": "wild cluster restricted bootstrap-t, Webb (primary)",
}


@dataclass
class Calibration:
    """p-values of each method on responses with no signal, shape ``features x n_null``."""

    pvalues: dict
    features: list
    alpha: float
    q: float

    def rejection(self) -> pd.DataFrame:
        """Type-I error per method, with the familywise rate of BH across the features.

        Under the global null every discovery is false, so BH's FDR equals the chance of
        at least one discovery.
        """
        rows = []
        for key, label in CALIBRATION_METHODS.items():
            P = self.pvalues.get(key)
            if P is None:
                continue
            per_feature = np.nanmean(P < self.alpha, axis=1)
            q = np.apply_along_axis(fdr_adjust, 0, P)
            n = P.shape[1]
            rows.append(dict(method=label, key=key, type1=float(np.nanmean(P < self.alpha)),
                             type1_min=float(per_feature.min()), type1_max=float(per_feature.max()),
                             fdr_bh=float(np.mean(np.any(q <= self.q, axis=0))),
                             mc_se=float(np.sqrt(self.alpha * (1 - self.alpha) / n)),
                             n_null=n))
        return pd.DataFrame(rows)


def null_responses(design: MetaDesign, n: int, rng: np.random.Generator) -> np.ndarray:
    """``N x n`` responses with the real within-cluster structure and no link to the features."""
    return design.y[block_permutations(design.rows, design.codes, n, rng)].T


def null_calibration(design: MetaDesign, features: Sequence[str] | None = None, *,
                     n_null: int = 200, n_boot: int = 399, alpha: float = 0.05,
                     q: float = 0.05, seed: int = 0, grid=RIDGE_GRID,
                     weights: str = "webb") -> Calibration:
    """Every method's rejection rate on block-permuted responses, where no feature matters.

    The nulls keep the observed within-dataset dependence, the embedding structure and
    the real features. This measures each method on the actual design, not on a
    simulation that could have been tuned to flatter it. For the proposed ridge method,
    the penalty is re-chosen by leave-one-cluster-out CV on every null response, as the
    real analysis would do.
    """
    feats = list(design.features.columns) if features is None else list(features)
    rng = np.random.default_rng(seed)
    Y0 = null_responses(design, n_null, rng)
    P = {k: np.full((len(feats), n_null), np.nan) for k in CALIBRATION_METHODS}
    dfc = design.G - 1
    for i, f in enumerate(feats):
        ols = ClusteredOLS(design.matrix([f]), design.codes, design.G)
        j = ols.K - 1
        t, _ = ols.t_iid(Y0, j)
        P["ols_iid"][i] = 2 * stats.t.sf(np.abs(t), ols.N - ols.K)
        t, _ = ols.t_cr1(Y0, j)
        P["cr1"][i] = 2 * stats.t.sf(np.abs(t), dfc)
        t, _ = ols.t_cv3(Y0, j)
        P["cv3"][i] = 2 * stats.t.sf(np.abs(t), dfc)
        P["wcr"][i] = ols.wild_restricted(Y0, j, n_boot, rng, weights)[1]
        boot, _, _ = _pairs_bootstrap_betas(ols.X, design.codes, design.G, Y0, n_boot, rng)
        P["pairs_ols"][i] = _percentile_p(boot[:, j, :])

    X, pen = design.matrix(feats), design.penalty_mask(feats)
    best = _lodo_ridge_errors(X, Y0, design.codes, design.G, grid, pen).argmin(axis=0)
    counts = rng.multinomial(design.G, np.full(design.G, 1.0 / design.G), size=n_boot)
    first = X.shape[1] - len(feats)
    for b in np.unique(best):
        cols = best == b
        boot, _, _ = _pairs_bootstrap_betas(X, design.codes, design.G, Y0[:, cols], n_boot,
                                            rng, penalty=grid[b] * design.N * pen, counts=counts)
        P["pairs_ridge"][:, cols] = _percentile_p(boot[:, first:, :])
    return Calibration(pvalues=P, features=feats, alpha=alpha, q=q)


def power_curve(design: MetaDesign, features: Sequence[str] | None = None, *,
                effects: Sequence[float] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10),
                n_sim: int = 100, n_boot: int = 399, alpha: float = 0.05, q: float = 0.05,
                seed: int = 0, weights: str = "webb") -> pd.DataFrame:
    """Power of the primary test to detect ``effect`` per SD of one feature.

    The signal ``effect * z`` is added to block-permuted responses, where ``z`` is the
    target feature. Every feature is then tested and BH is applied across the family, as
    in the real analysis. Returns power before and after the FDR adjustment, per target
    feature and effect size.
    """
    feats = list(design.features.columns) if features is None else list(features)
    rng = np.random.default_rng(seed)
    Y0 = null_responses(design, n_sim, rng)
    models = [ClusteredOLS(design.matrix([f]), design.codes, design.G) for f in feats]
    Z = design.features[feats].to_numpy(float)
    rows = []
    for target, f in enumerate(feats):
        for effect in effects:
            Y = Y0 + effect * Z[:, [target]]
            pv = np.stack([m.wild_restricted(Y, m.K - 1, n_boot, rng, weights)[1] for m in models])
            qv = np.apply_along_axis(fdr_adjust, 0, pv)
            rows.append(dict(feature=f, effect=float(effect),
                             power=float(np.mean(pv[target] < alpha)),
                             power_bh=float(np.mean(qv[target] <= q))))
    return pd.DataFrame(rows)


__all__ = [
    "CALIBRATION_METHODS", "CLUSTER_COLUMN", "INSTANCE", "METRIC_COLUMNS", "RIDGE_GRID",
    "STAGES", "SYNTHETIC_PREFIXES", "UNIT", "WEBB_WEIGHTS", "Calibration", "ClusteredOLS",
    "MetaDesign", "RidgeOmnibus", "Screen", "block_permutations", "build_design",
    "canonical_meta_features", "dataset_family", "degenerate_arms",
    "embedding_indicators", "family_map", "fdr_adjust", "feature_reliability",
    "joint_tests", "lodo_influence", "loio_contrast", "marginal_tests", "matched_contrast",
    "meta_feature_columns", "null_calibration", "null_responses", "permutation_strata",
    "power_curve", "ridge_nested_cv", "ridge_omnibus", "ridge_pairs_bootstrap",
    "screen_features", "unit_means", "unit_report", "variance_decomposition",
    "variance_inflation", "within_unit_test",
]
