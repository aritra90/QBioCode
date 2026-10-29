"""Shared helpers for the ``collated_full1`` analysis notebook.

The notebook imports from here so that the heavy, easy-to-get-wrong pieces -- loading a
98 MB results table, importing ``fair_selection`` without dragging in qiskit, and
rebuilding the protocol's own validation-selection rule -- live in one tested place
rather than being retyped in several cells.

Sign convention throughout, matching ``qbiocode.utils.fair_selection``:

    delta = classical - quantum      delta > 0 classical ahead, delta < 0 quantum ahead
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

#: The two quantum arms and the nine classical arms of this sweep.
QUANTUM_ARMS = ("qsvc_opt", "pqk_opt")
CLASSICAL_ARMS = ("catboost_opt", "dt_opt", "lr_opt", "mlp_opt", "nb_opt",
                  "rf_opt", "svc_opt", "tabpfn_opt", "xgb_opt")

#: One cell of the design: the unit on which every arm was scored on identical rows.
CELL = ["Dataset", "embeddings", "iteration"]


def load_fair_selection(repo: Path | str):
    """Import ``qbiocode/utils/fair_selection.py`` without ``qbiocode/__init__.py``.

    The package ``__init__`` imports ``qiskit_ibm_runtime``, which an analysis
    environment need not have. ``fair_selection`` itself needs only one name from the
    package -- ``QUANTUM_MODELS``, the quantum-learner registry -- so that module is
    stubbed with a value copied verbatim from ``qbiocode/evaluation/model_run.py``, and
    :func:`check_registry_stub` asserts the copy still matches the repository.
    """
    repo = Path(repo)
    pkg = types.ModuleType("qbiocode"); pkg.__path__ = []
    ev = types.ModuleType("qbiocode.evaluation"); ev.__path__ = []
    mr = types.ModuleType("qbiocode.evaluation.model_run")
    mr.QUANTUM_MODELS = frozenset({"qsvc", "qnn", "vqc", "pqk", "qpl", "qelm"})
    sys.modules.setdefault("qbiocode", pkg)
    sys.modules.setdefault("qbiocode.evaluation", ev)
    sys.modules.setdefault("qbiocode.evaluation.model_run", mr)

    spec = importlib.util.spec_from_file_location(
        "fair_selection", repo / "qbiocode" / "utils" / "fair_selection.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fair_selection"] = mod      # dataclasses resolve __module__ here
    spec.loader.exec_module(mod)
    return mod


def check_registry_stub(repo: Path | str) -> frozenset:
    """Fail loudly if the stubbed ``QUANTUM_MODELS`` has drifted from the repository.

    A stub that silently fell behind the real registry would put a quantum arm on the
    classical side, which is exactly the bookkeeping error ``fair_selection``'s own
    docstring warns about, so it is checked rather than trusted.
    """
    src = (Path(repo) / "qbiocode" / "evaluation" / "model_run.py").read_text()
    hit = re.search(r"QUANTUM_MODELS\s*=\s*frozenset\(\{([^}]*)\}\)", src)
    if hit is None:
        raise AssertionError("could not find QUANTUM_MODELS in model_run.py")
    found = frozenset(re.findall(r'"([^"]+)"', hit.group(1)))
    expected = frozenset({"qsvc", "qnn", "vqc", "pqk", "qpl", "qelm"})
    if found != expected:
        raise AssertionError(f"registry drifted: repo has {sorted(found)}, "
                             f"the stub assumes {sorted(expected)}")
    return found


def load_results(collated: Path | str) -> pd.DataFrame:
    """``ModelResults.csv`` as a frame, with the family label the analysis groups on."""
    df = pd.read_csv(Path(collated) / "ModelResults.csv", low_memory=False)
    df["side"] = np.where(df["model"].isin(QUANTUM_ARMS), "quantum", "classical")
    df["family"] = df["Dataset"].map(dataset_family)
    return df


def dataset_family(name: str) -> str:
    """The group a dataset belongs to for leave-one-family-out checks.

    Real datasets group by source (``pmlb``, ``openml``, ``libsvm``). Synthetic ones group
    by *generator shape*, not by the ``shapes`` prefix: the three seeds and three
    complexity settings of ``torus`` share a label rule and a latent geometry, so they are
    not independent evidence about each other and must leave the design together.
    """
    stem = str(name).replace(".csv", "")
    if stem.startswith("shapes__"):
        base = stem[len("shapes__"):]
        base = re.sub(r"_k\d+_d\d+_n\d+_s\d+$", "", base)
        base = re.sub(r"_d\d+_n\d+_s\d+$", "", base)
        return f"shapes:{base}"
    return stem.split("__")[0]


def cell_matrix(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """``[cell x model]`` matrix of ``value``: the paired layout every contrast uses.

    Within one cell all eleven arms were scored on byte-identical test rows, so a paired
    analysis down the columns is valid.
    """
    return df.pivot_table(index=CELL, columns="model", values=value)


def validation_envelope(test: pd.DataFrame, val: pd.DataFrame,
                        arms) -> pd.Series:
    """Per cell, the test score of the arm with the best validation score among ``arms``.

    This is the protocol's own rule (``fair_selection`` ``selection='validation'``)
    reduced to one side: validation rows are disjoint from test rows, so the chosen arm's
    test score carries no winner's curse. Reimplemented here -- rather than read out of
    ``WinnerReport`` -- because the central question of this analysis is what happens when
    the *arm list* changes, which means calling it on arbitrary subsets.
    """
    arms = list(arms)
    v = val[arms].to_numpy(float)
    t = test[arms].to_numpy(float)
    pick = np.nanargmax(v, axis=1)
    return pd.Series(t[np.arange(len(t)), pick], index=test.index, name="+".join(arms))


def per_dataset_delta(test: pd.DataFrame, val: pd.DataFrame,
                      classical=CLASSICAL_ARMS, quantum=QUANTUM_ARMS) -> pd.Series:
    """One ``delta`` per dataset: the mean over its cells of classical - quantum.

    The dataset, not the fold, is the unit of independence (folds of one dataset share
    training rows), so every corpus-level statement in the notebook is computed on this
    series.
    """
    d = validation_envelope(test, val, classical) - validation_envelope(test, val, quantum)
    return d.groupby(level=0).mean()


def fisher_pooled_rho(groups) -> float:
    """Fisher-z average of per-group Spearman rho, weighted by ``n - 3``.

    Pooling ranks across groups of very unequal size instead -- as a single Spearman on
    within-group ranks does -- lets the largest group dominate and inflates the
    correlation; this keeps each group's rho on its own scale before averaging.
    """
    rs, ws = [], []
    for rho, n in groups:
        if not np.isfinite(rho) or n < 5:
            continue
        rs.append(np.clip(rho, -0.999, 0.999))
        ws.append(n - 3)
    if len(rs) < 2:
        return np.nan
    return float(np.tanh(np.average(np.arctanh(rs), weights=ws)))


# ---------------------------------------------------------------------------
# The merged corpus: full1 (tabular) + tcga1 (RNA-seq)
# ---------------------------------------------------------------------------

#: ``run`` values of the two sweeps, and the corpus label each maps to. The corpus, not
#: the run, is what the analysis conditions on: a later tabular sweep would join
#: ``tabular`` rather than become a third thing.
RUN_TO_CORPUS = {"full1": "tabular", "tcga1": "tcga"}

#: Rows expected per run, asserted on load. A merge that silently writes one corpus --
#: or a truncated ``to_csv`` -- produces a file that looks analysable and is not; this
#: project has already had both.
EXPECTED_ROWS = {"full1": 33990, "tcga1": 6270}


def load_merged(path: Path | str, *, strict: bool = True) -> pd.DataFrame:
    """The merged ``ModelResults`` of both sweeps, validated on the way in.

    Args:
        path: the merged CSV.
        strict: raise when a run's row count differs from :data:`EXPECTED_ROWS`, when a
            run is missing, or when any row lacks a model. Pass False only to inspect a
            file you already know is incomplete.

    Returns:
        The frame, plus ``side`` (quantum/classical), ``family`` (for
        leave-one-family-out) and ``corpus`` (tabular/tcga).

    Raises:
        ValueError: under ``strict``, naming what is wrong. Loudly, because every
            failure mode seen so far has produced a file that reads fine.
    """
    df = pd.read_csv(Path(path), low_memory=False)
    problems = []
    if "run" not in df.columns:
        problems.append("no 'run' column: the merge did not label its inputs, so the "
                        "two corpora cannot be told apart")
    else:
        counts = df["run"].value_counts(dropna=False).to_dict()
        for run, expected in EXPECTED_ROWS.items():
            got = int(counts.get(run, 0))
            if got != expected:
                problems.append(f"run {run!r}: {got} rows, expected {expected}")
        unknown = set(counts) - set(EXPECTED_ROWS)
        if unknown:
            problems.append(f"unexpected run value(s) {unknown}")
    for column in ("model", "iteration", "Dataset"):
        if column in df.columns and df[column].isna().any():
            problems.append(f"{int(df[column].isna().sum())} row(s) with no {column} "
                            f"-- a truncated write leaves exactly this")
    if problems and strict:
        raise ValueError("merged results are not usable:\n  - " + "\n  - ".join(problems))
    for message in problems:
        print(f"  !! {message}")

    df["side"] = np.where(df["model"].isin(QUANTUM_ARMS), "quantum", "classical")
    df["family"] = df["Dataset"].map(dataset_family)
    df["corpus"] = df["run"].map(RUN_TO_CORPUS)
    return df


def validation_reliability(df: pd.DataFrame, arms=None) -> pd.Series:
    """Per arm, Spearman rho between a cell's validation score and its test score.

    The precondition of the whole winner analysis, and not a given. Selection by
    validation is only meaningful where validation ranks predict test ranks; where rho
    is near zero the rule picks noise, and "arm X won" describes the draw rather than
    the arm. Reported per corpus rather than pooled, because this is exactly what
    differs between them.
    """
    arms = list(arms or (list(CLASSICAL_ARMS) + list(QUANTUM_ARMS)))
    test, val = cell_matrix(df, "balanced_accuracy"), cell_matrix(df, "val_auc")
    out = {}
    for arm in arms:
        if arm not in test.columns or arm not in val.columns:
            continue
        ok = test[arm].notna() & val[arm].notna()
        if ok.sum() >= 5:
            out[arm] = test.loc[ok, arm].corr(val.loc[ok, arm], method="spearman")
    return pd.Series(out, name="val_to_test_rho").sort_values(ascending=False)


def cluster_of(datasets, repo: Path | str = None, mapping=None) -> pd.Series:
    """The cluster-robust unit of each dataset, from ``meta_regression.dataset_family``.

    Distinct from :func:`dataset_family` above, which groups for leave-one-family-out:
    that one splits ``shapes`` by generator shape, this one is what the corpus-level
    tests resample. The 19 TCGA cohorts are ONE cluster here
    (``meta_regression.SINGLE_CLUSTER_SOURCES``) -- pass ``mapping={id: id}`` to get the
    per-cohort alternative for a sensitivity check.
    """
    from qbiocode.utils.meta_regression import dataset_family as cluster_fn

    stems = pd.Index(datasets).astype(str).str.replace(".csv", "", regex=False)
    return pd.Series([cluster_fn(s, mapping=mapping) for s in stems],
                     index=pd.Index(datasets), name="cluster")


def cluster_bootstrap_ci(values, clusters, *, n_boot: int = 10000, seed: int = 0,
                         alpha: float = 0.05):
    """``(mean, lo, hi)`` of ``values``, resampling CLUSTERS with replacement.

    Resampling datasets would treat 19 TCGA cohorts as 19 independent draws; they share
    an assay, a pipeline and a label definition, so their errors are correlated and the
    interval would be too narrow. Resampling clusters absorbs that.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    groups = [values[clusters == c] for c in pd.unique(clusters)]
    if len(groups) < 2:
        return float(np.mean(values)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[i] = np.mean(np.concatenate([groups[j] for j in pick]))
    return (float(np.mean(values)),
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))
