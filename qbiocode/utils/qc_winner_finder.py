## function to find datasets where QML methods did better than classical
import logging
import os
from collections.abc import Mapping

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qbiocode.utils.fair_selection import QUANTUM_STEMS, select_winners  # noqa: F401

# `warnings.warn` is useless in this module. Importing matplotlib -- which this file does at
# module scope, for the plots below -- installs a blanket ``('ignore', Warning)`` filter at
# position 0, so every warning raised from here is discarded before it reaches a terminal or
# an LSF .err file. That silence is worst for exactly the guard that matters most: the ragged
# -run notice, whose entire job is to say "a job died, these verdicts are void". `logging`
# is unaffected by that filter and is what `fair_selection` already uses, so the guards below
# go through this logger. Verified: with matplotlib imported, a warnings.warn from here
# produces no output at all.
logger = logging.getLogger(__name__)


def qml_winner(results_df, rawevals_df, output_dir, tag):
    """This function finds data sets where QML was beneficial (higher F1 scores than CML) and create new .csv files
    with the relevant evaluation and performance for these specific datasets, for further analysis.
    It also computes the best results per method across all splits and the best results per dataset.
    It returns two DataFrames: one with the datasets where QML methods outperformed CML methods, and another with the
    evaluation scores for the best QML method for each of these datasets.
    It also saves these DataFrames as .csv files in the specified output directory.

    Args:
        results_df (pandas.DataFrame): Dataset in pandas corresponding to 'ModelResults.csv'
        rawevals_df (pandas.DataFrame): Dataset in pandas corresponding to 'RawDataEvaluation.csv'
    Returns:
        qml_winners (pandas.DataFrame): contais the input datasets for which at least one QML method
                                        performed better than CML. DataFrame contains the scores of all
                                        the methods.
        winner_eval_score (pandas.DataFrame): contains the input datasets, their evaluation, and scores for the
                                            specific qml method that yielded the best score.
    """

    # This function is DESCRIPTIVE, not inferential, and the distinction is not
    # cosmetic. Its rule -- pool, take the max per side, compare -- is biased toward
    # whichever side has more arms or noisier per-arm estimates, and under the sweep
    # config it is *both*. Measured on a pure null (both sides mean F1 0.75, sigma 0.04,
    # 8 tuned classical vs 3 untuned quantum learners, 2 embeddings, 5 resamples,
    # 84 datasets) it reports mean delta_f1 = +0.0741 and hands classical the win on
    # 84/84 datasets -- i.e. it cannot report a quantum win whatever the data says.
    # Use it to browse and plot per-model scores; use `select_winners` for any claim.
    # logger, not warnings.warn, for the reason given at the top of this module: matplotlib
    # is imported here and silences every warning. A steer-away notice nobody can see is not
    # a steer-away notice, and this is the function that reports 84/84 classical wins on a
    # pure null.
    logger.warning(
        "qml_winner selects by pooled post-hoc argmax, which is biased toward the side "
        "with more arms and toward the side whose per-arm scores are noisier (a tuned "
        "learner re-searches per resample, so its rows do not average). On a pure null "
        "it reports classical wins on 84/84 datasets. It is retained for descriptive "
        "output and plotting; use qbiocode.utils.fair_selection.select_winners for any "
        "reported win, delta, or significance claim."
    )

    # pass in the ML results
    df = results_df.copy()
    # pull in the raw evaluations
    rawevals = rawevals_df.copy()
    # first, compute mean across all splits
    # model_evaluation.py writes exactly one parameter column, named for the branch
    # that produced it: 'Model_Parameters' with tuning off, 'BestParams_Tuned' with it
    # on -- or 'BestParams_GridSearch', the name that branch used before Optuna
    # replaced the exhaustive grid, which every older ModelResults.csv still carries.
    # The previous if/else assumed the absence of one meant the presence of the other,
    # so any third name became a KeyError raised from inside groupby.
    parameter_columns = [
        name for name in ("Model_Parameters", "BestParams_Tuned", "BestParams_GridSearch")
        if name in df.columns
    ]
    if not parameter_columns:
        raise ValueError(
            "None of the model-parameter columns ('Model_Parameters', "
            "'BestParams_Tuned', 'BestParams_GridSearch') are present in the results "
            f"table, which has {sorted(df.columns)}. Pass the ModelResults.csv written "
            "by QProfiler."
        )
    # Coalesced, not `parameter_columns[0]`. A single run can carry BOTH names now that
    # model_evaluation decides the column per row rather than per run (a tuned classical
    # model reports 'BestParams_Tuned' while an untuned quantum one in the same run
    # reports 'Model_Parameters'). Grouping on whichever name happened to sort first left
    # NaN in that key for every row belonging to the other column -- and pandas' groupby
    # drops NaN keys by default, so those rows would disappear from the winner table
    # silently. Taking the first non-null across the candidates recovers each row's real
    # parameters whichever column holds them.
    parameters = df[parameter_columns].bfill(axis=1).iloc[:, 0]
    df_across_split = (
        df.assign(_parameters=parameters)
        # dropna=False: '_parameters' is a groupby KEY here, and pandas drops rows
        # whose key is null by default. A model that reports no parameters writes
        # None, which round-trips through CSV as NaN -- so every such arm silently
        # vanished from the winner table. Observed with untuned quantum rows, which
        # disappeared entirely and left classical the winner by default.
        .groupby(["Dataset", "embeddings", "model", "_parameters"], dropna=False)["f1_score"]
        .mean()
        .reset_index()
        .rename(columns={"_parameters": parameter_columns[0]})
    )
    # now, extract the best results per method across embedding and iteration
    df_best = df_across_split.groupby(["Dataset", "model"])["f1_score"].max().reset_index()
    # df_best = df_across_split.groupby(['Dataset', 'model', 'Model_Parameters'])['f1_score'].max().reset_index()
    df_best.to_csv((os.path.join(output_dir, tag + "_best_across_split.csv")), index=False)
    # get summary accross all datasets
    df_best_model_mean = df_best.groupby("model")["f1_score"].mean()
    df_best_model_median = df_best.groupby("model")["f1_score"].median()
    df_best_model_max = df_best.groupby("model")["f1_score"].max()
    df_best_model_std = df_best.groupby("model")["f1_score"].std()
    df_best_permodel_summary = pd.concat(
        [df_best_model_mean, df_best_model_median, df_best_model_max, df_best_model_std], axis=1
    )
    df_best_permodel_summary.columns = [
        "Mean_F1_Score",
        "Median_F1_Score",
        "Max_F1_Score",
        "StandardDev_F1_Score",
    ]
    df_best_permodel_summary.to_csv((os.path.join(output_dir, tag + "_best_permodel_summary.csv")))
    # print(df_best_permodel_summary)

    # extract the best results per dataset
    best_per_dataset = df_best.loc[df_best.groupby("Dataset")["f1_score"].idxmax()]
    # best_per_dataset = df_across_split.loc[df_across_split.groupby('Dataset')['f1_score'].idxmax()]
    # create list of qml methods
    # Matched on the lower-cased first token of the label, not against a fixed list of
    # exact spellings. The list used to be ["QSVC", "QNN", "VQC", "PQK"] compared with
    # `.isin()`, which could never match: every label QProfiler writes is lower case, so
    # this branch was dead on real output and qml_winners.csv came back empty from every
    # genuine quantum run. Two further spellings it could not have matched either way --
    # 'qpl' was absent from the list entirely, and a tuned run is labelled '<name>_opt'
    # ('qpl_opt_<head>' for QPL, which fans out one column per classical head).
    # Derived from the dispatch registry instead of restated here, and widened by
    # 'qensemble': the literal set omitted it, so QEnsemble -- a quantum learner --
    # was classified as CLASSICAL. That error pushes in the expensive direction
    # twice over, suppressing a quantum win and inflating the classical side by an
    # extra arm at the same time.
    qml_families = set(QUANTUM_STEMS)

    def _is_quantum(label):
        return str(label).lower().split("_", 1)[0] in qml_families

    quantum_datasets = best_per_dataset.loc[
        best_per_dataset["model"].map(_is_quantum), "Dataset"
    ]
    qml_winner = df_across_split[df_across_split["Dataset"].isin(quantum_datasets)]
    if not qml_winner.empty:
        bestmethod = qml_winner.groupby("Dataset")["f1_score"].idxmax()
        qc_method_and_score = qml_winner.loc[bestmethod]
        qml_winner.to_csv((os.path.join(output_dir, tag + "_qml_winners.csv")), index=False)
        dataset = list(qml_winner["Dataset"].unique())

        #######
        # now let's find the raw data evaluations for the qml winner data sets
        # this wil produce another csv file that contains scores, evaluation, and qml method
        # for these "qml winners".
        winner_evals = []
        for file in dataset:
            eval = rawevals.loc[rawevals["Dataset"] == file]
            # print(eval)
            winner_evals.append(eval)
        winner_evals_df = pd.concat(winner_evals)
        winner_evals_df.to_csv((os.path.join(output_dir, tag + "_winner_evals.csv")), index=False)
        # 'Dataset' is carried explicitly rather than relying on iloc[:, -3:] landing
        # on it, so the join below has a key. The positional slice kept the last three
        # columns (model, parameters, f1_score) and dropped the dataset name -- which is
        # precisely what forced the index-aligned concat that follows.
        score_columns = [
            name for name in ("Dataset", "model", parameter_columns[0], "f1_score")
            if name in qc_method_and_score.columns
        ]
        winner_scores_df = qc_method_and_score[score_columns]
        winner_scores_df.to_csv((os.path.join(output_dir, tag + "_winner_score.csv")), index=False)

        print(winner_scores_df)
        # Merged on 'Dataset', never concatenated on the index. pd.concat(axis=1) aligns
        # on INDEX LABELS, and these two frames carry unrelated ones: winner_evals_df
        # keeps rawevals' original row numbers (a winning dataset can sit at row 57)
        # while winner_scores_df keeps df_across_split's. Almost no label appears in
        # both, so the union produced one row per label with NaN on whichever side was
        # missing -- of 84 winners, roughly 80 rows came back as phantom entries with a
        # NaN Dataset, and each surviving row risked pairing one dataset's evaluation
        # with another dataset's score. An in-memory caller hit it harder still: the
        # index out of `evaluate()` is [0, 0, 0, ...], whose duplicate labels make the
        # same concat raise InvalidIndexError outright.
        winner_eval_score = winner_evals_df.merge(
            winner_scores_df, on="Dataset", how="left", validate="many_to_one"
        )
        winner_eval_score.to_csv(
            (os.path.join(output_dir, tag + "_winner_eval_score.csv")), index=False
        )  # contains dataset, evaluation, qml method, and  average f1 score
        #######

        # optional print statements
        print("*** The number of qml winners is", len(dataset))
        print("*** The qml winners are:", dataset)

        return qml_winner, winner_eval_score, df_best

    else:
        print("*** QML methods were outperformed by CML methods in all datasets ***")

        return


def fair_winner(
    results_df,
    output_dir,
    tag,
    metric="f1_score",
    epsilon=0.027,
    alpha=0.05,
    test_size=None,
    seed=0,
    baseline=None,
    *,
    margin=0.0,
    fdr=0.10,
    controls=None,
    selection="loio",
    validation_col="tuning_score",
    k=None,
):
    """Unbiased counterpart to :func:`qml_winner`; writes the tables a paper can cite.

    Delegates to :func:`qbiocode.utils.fair_selection.select_winners` and persists its
    frames. Unlike ``qml_winner`` this returns a report object that is *always* truthy --
    ask ``report.quantum_datasets`` whether quantum won. ``qml_winner`` returned a bare
    ``None`` for "no quantum winner", which is indistinguishable from "the function
    changed shape" and forced every caller to guard before unpacking.

    ``rawevals_df`` is deliberately not a parameter. Joining dataset metafeatures onto a
    verdict is a separate concern, and doing it inside the selector is what produced the
    index-aligned concat bug; join ``per_dataset`` to RawDataEvaluation.csv on
    ``'Dataset'`` afterwards if you want them together.

    Args:
        results_df (pandas.DataFrame): a 'ModelResults.csv'-shaped frame. Must carry an
            ``iteration`` column -- without a resample axis there is nothing to hold out
            and no interval to compute.
        output_dir (str): directory for the output CSVs.
        tag (str): filename prefix.
        metric (str): column to compare; higher must be better. Not just ``f1_score``.
        epsilon (float): the TOST equivalence bound only: a dataset is ``equivalent``
            when its whole confidence interval lies inside ``(-epsilon, +epsilon)``. It
            plays no part in deciding a win; that is ``margin``.
        alpha (float): two-sided level of the per-dataset intervals, and of the Holm
            family of ``controls``.
        test_size (float): the split fraction the run used, for the Nadeau-Bengio term.
            Required with ``selection='loio'``: ModelResults.csv does not record it, so
            ``None`` raises ``ValueError`` (from ``select_winners``). The pilot used 0.2.
            With ``selection='validation'`` it is ``1/k`` and may be omitted.
        seed (int): unused; kept so existing calls still work. Ties are averaged, so
            the result does not depend on any seed.
        baseline (pandas.DataFrame): optional ``['Dataset', metric]`` dummy floor.
        margin (float): pre-registered superiority margin. A raw win needs the whole
            interval beyond ``-margin`` (quantum) or ``+margin`` (classical). Default 0.
        fdr (float): Benjamini-Hochberg level of the discovery family, in ``(0, 1)``.
        controls (str | Iterable[str] | None): datasets forming a separate
            Holm-adjusted family (e.g. synthetic controls), left out of the BH family and
            of ``report.quantum_datasets`` / ``classical_datasets``.
        selection (str): ``'loio'`` (default, repeated holdouts) or ``'validation'``
            (``split_mode: manifest``: per fold, the arm with the best validation score).
            The selection trace is written as ``<tag>_fair_loio_selection.csv`` or
            ``<tag>_fair_validation_selection.csv`` accordingly.
        validation_col (str): validation-score column for ``selection='validation'``.
        k (int | None): outer folds per repeat for ``selection='validation'``; defaults
            to the ``split_k`` column.

    Returns:
        qbiocode.utils.fair_selection.WinnerReport: verdicts, per-arm means, the arm
        chosen per held-out iteration (or per fold), and corpus-level inference.
    """
    report = select_winners(
        results_df,
        metric=metric,
        epsilon=epsilon,
        alpha=alpha,
        test_size=test_size,
        seed=seed,
        baseline=baseline,
        margin=margin,
        fdr=fdr,
        controls=controls,
        selection=selection,
        validation_col=validation_col,
        k=k,
    )
    os.makedirs(output_dir, exist_ok=True)
    report.per_dataset.to_csv(os.path.join(output_dir, tag + "_fair_verdicts.csv"), index=False)
    report.per_arm.to_csv(os.path.join(output_dir, tag + "_fair_per_arm.csv"), index=False)
    report.selection.to_csv(
        os.path.join(output_dir, f"{tag}_fair_{report.selection_mode}_selection.csv"),
        index=False,
    )
    pd.Series(report.corpus, dtype=object).to_frame("value").to_csv(
        os.path.join(output_dir, tag + "_fair_corpus.csv")
    )
    return report


# ======================================================================================
# Benchmark-scale aggregation: many one-dataset runs -> one delta table.
# ======================================================================================

#: Metrics compared by default. Higher is better for every one of these, which the
#: comparison assumes. ``time`` is deliberately absent: it is lower-is-better, so pooling
#: it with "take the better side" would report the slowest arm as the winner.
BENCHMARK_METRICS = (
    "balanced_accuracy",
    "f1_score",
    "mcc",
    "auc",
    "pr_auc",
    "accuracy",
)


def missing_folds(results, expected="observed"):
    """Arms of a ``split_mode: manifest`` frame that lack some of their dataset's folds.

    Validation selection compares arms fold by fold, so an arm whose job for one fold
    died is silently absent from that fold's choice -- the other side then wins that
    fold by default. Every arm must carry every global ``iteration`` its dataset is
    expected to have.

    Args:
        results (pandas.DataFrame): ModelResults-shaped frame with ``Dataset``,
            ``model`` and ``iteration`` (``embeddings``, ``split_k`` and
            ``split_repeats`` if present).
        expected (str | Iterable[int]): the iterations every arm should have.
            ``'observed'`` (default): every iteration any arm of the dataset has, which
            is right for a run over a subset of the splits (``--splits 1-5``) but cannot
            see a fold missing from every arm. ``'full'``: ``1 .. split_k *
            split_repeats`` (falling back to ``'observed'`` without those columns), for a
            run known to cover every split. An iterable of ints: exactly those.

    Returns:
        pandas.DataFrame: one row per incomplete ``(Dataset, embeddings, model)`` arm,
        with ``n_expected``, ``n_present`` and ``missing`` (the absent iterations,
        ``';'``-joined). Empty when every arm is complete.

    Raises:
        ValueError: ``expected`` is an unknown string.
    """
    cols = ["Dataset", "embeddings", "model", "n_expected", "n_present", "missing"]
    if isinstance(expected, str) and expected not in ("observed", "full"):
        raise ValueError(f"expected must be 'observed', 'full' or iterations; got {expected!r}")
    if results.empty:
        return pd.DataFrame(columns=cols)
    work = results.copy()
    if "embeddings" not in work:
        work["embeddings"] = ""
    work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
    rows = []
    for dataset, block in work.groupby("Dataset", sort=True):
        want = set(block["iteration"].dropna().astype(int))
        if not isinstance(expected, str):
            want = {int(i) for i in expected}
        elif expected == "full" and {"split_k", "split_repeats"} <= set(block.columns):
            k = pd.to_numeric(block["split_k"], errors="coerce").max()
            reps = pd.to_numeric(block["split_repeats"], errors="coerce").max()
            if np.isfinite(k) and np.isfinite(reps):
                want = set(range(1, int(k) * int(reps) + 1))
        for (emb, model), arm in block.groupby(["embeddings", "model"], dropna=False,
                                               sort=True):
            have = set(arm["iteration"].dropna().astype(int))
            gone = sorted(want - have)
            if gone:
                rows.append({"Dataset": dataset, "embeddings": emb, "model": model,
                             "n_expected": len(want), "n_present": len(have & want),
                             "missing": ";".join(map(str, gone))})
    return pd.DataFrame(rows, columns=cols)


def collect_model_results(results_root, pattern="**/ModelResults.csv"):
    """Concatenate every per-dataset ``ModelResults.csv`` under ``results_root``.

    The benchmark runs one LSF job per dataset, so the corpus arrives as N separate
    ``ModelResults.csv`` files rather than one -- each already carrying its own ``Dataset``
    column, which is what makes a plain concat correct here.

    Two failure modes are made loud rather than silent, because both produce a table that
    looks fine and means nothing:

    * A missing ``iteration`` column. Without a resample axis there is no variance to
      estimate, so every interval collapses and every dataset reports "insufficient
      iterations" -- a verdict-shaped answer to a question the data cannot address.
    * A job that died partway. Its CSV exists and parses, with fewer models than the
      others. Pooled max-per-side then compares a full classical arm against a truncated
      quantum one, and reports the truncation as a classical win. So the per-file model
      count is returned alongside the frame for the caller to check, not buried.

    * Under ``split_mode: manifest`` (a ``split_mode`` column holding ``'manifest'``),
      an arm missing a fold that other arms of its dataset have, e.g. one per-fold job
      that died. Each arm is checked with :func:`missing_folds` (its default
      ``'observed'`` rule, so a run over a subset of the splits is not flagged; a
      dataset covering fewer than its kR splits is logged at INFO); incomplete arms are
      logged, and the inventory
      gains ``n_missing_folds`` (incomplete arms of that file's datasets). The check
      runs on the pooled frame, because one fold's arms can arrive in separate files.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: the concatenated results, and a
        per-file inventory with columns ``['path', 'Dataset', 'n_rows', 'n_models',
        'n_iterations']`` (plus ``n_missing_folds`` in manifest mode).
    """
    import glob as _glob

    paths = sorted(_glob.glob(os.path.join(results_root, pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"no ModelResults.csv under {results_root!r} matching {pattern!r}. "
            "Point results_root at the directory holding the per-dataset run folders."
        )

    frames, inventory = [], []
    for path in paths:
        df = pd.read_csv(path)
        if "Dataset" not in df.columns:
            raise ValueError(f"{path} has no 'Dataset' column; it is not a ModelResults.csv")
        frames.append(df)
        inventory.append(
            {
                "path": path,
                "Dataset": "|".join(sorted(map(str, df["Dataset"].dropna().unique()))),
                "n_rows": len(df),
                "n_models": df["model"].nunique() if "model" in df else 0,
                "n_iterations": df["iteration"].nunique() if "iteration" in df else 0,
            }
        )

    results = pd.concat(frames, ignore_index=True)
    inv = pd.DataFrame(inventory)
    if "iteration" not in results.columns:
        raise ValueError(
            "the pooled results have no 'iteration' column, so no interval can be "
            "computed and no margin certified. Re-run with more than one resample."
        )
    if "split_mode" in results.columns and (results["split_mode"] == "manifest").any():
        manifest_rows = results[results["split_mode"] == "manifest"]
        gaps = missing_folds(manifest_rows)
        partial = missing_folds(manifest_rows, expected="full")
        partial = sorted(set(partial["Dataset"]) - set(gaps["Dataset"]))
        if partial:
            # Every arm has the same folds but not all kR of them: a run over a subset of
            # the splits (--splits), or a fold whose every job died. Not an arm gap.
            logger.info("%d dataset(s) cover fewer than their split_k*split_repeats "
                        "folds on every arm (a subset run?): %s",
                        len(partial), ", ".join(map(str, partial)))
        per_dataset = gaps.groupby("Dataset").size().to_dict()
        inv["n_missing_folds"] = [
            sum(per_dataset.get(d, 0) for d in str(ds).split("|")) for ds in inv["Dataset"]
        ]
        if not gaps.empty:
            logger.warning(
                "%d arm(s) lack some of their folds, so validation selection would drop "
                "them from those folds -- re-run the missing jobs before reading a "
                "verdict:\n%s",
                len(gaps),
                gaps.to_string(index=False),
            )
    return results, inv


def resolution_floor_epsilon(results_df, metric, test_size, alpha=0.05):
    """The smallest margin this design can certify for ``metric`` -- a defensible epsilon.

    An epsilon chosen by hand is the weakest link in a win/loss claim, and it is not
    fixable by choosing a smaller one: the Nadeau-Bengio standard error
    ``s * sqrt(1/I + r)`` tends to ``s * sqrt(r)`` as iterations grow, so the confidence
    interval has a floor that no amount of compute goes below. Any epsilon *under* that
    floor can never be cleared, which silently converts every dataset to "inconclusive";
    any epsilon far above it throws away real effects.

    So the floor itself is the natural choice: the margin at which this experiment stops
    being able to tell the two sides apart. It is derived per metric, which matters because
    the metrics are not on one scale -- MCC spans [-1, 1] where accuracy spans [0, 1], so a
    single shared epsilon is twice as strict on one as on the other.

    ``s`` is pooled *within* (dataset, embedding pass, arm, model) across iterations, never
    across datasets: between-dataset spread is the signal being studied, and folding it in
    would inflate the floor until nothing could ever be called. The same goes for the
    embedding: a dataset's PCA and UMAP passes differ in mean, and pooling them into one
    group adds that difference to ``s`` (pilot: 0.0688 pooled vs 0.0633 per pass). The
    ``embeddings`` column joins the key whenever the frame has one.

    Args:
        results_df (pandas.DataFrame): ModelResults-shaped frame with ``Dataset``,
            ``model``, ``iteration`` and ``metric``; ``embeddings`` if present.
        metric (str): the metric column.
        test_size (float): the run's split fraction, for the Nadeau-Bengio term.
        alpha (float): two-sided level of the interval.

    Returns:
        tuple[float, float]: ``(epsilon, sigma)``; ``epsilon`` is NaN when ``sigma`` is
        not a positive finite number.
    """
    from qbiocode.utils.fair_selection import (
        iteration_floor_half_width,
        model_side,
    )

    df = results_df.dropna(subset=[metric]).copy()
    df["arm"] = df["model"].map(model_side)
    key = ["Dataset", "embeddings", "arm", "model"] if "embeddings" in df else [
        "Dataset", "arm", "model"]
    # dropna=False: a missing embedding label is its own pass, not a reason to lose rows.
    per_group = df.groupby(key, dropna=False)[metric].std(ddof=1)
    sigma = float(np.nanmean(per_group.to_numpy())) if len(per_group) else np.nan
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan, sigma
    return float(iteration_floor_half_width(sigma, test_size, alpha)), sigma


def delta_metric_table(
    results_df,
    metrics=BENCHMARK_METRICS,
    epsilon=None,
    alpha=0.05,
    test_size=None,
    seed=0,
    baseline=None,
    *,
    margin=0.0,
    fdr=0.10,
    controls=None,
    selection="loio",
    validation_col="tuning_score",
    k=None,
):
    """One row per dataset, one column block per metric: delta, interval, verdict.

    Deliberately *not* named for any single metric. Which metric carries the headline is a
    decision to make after looking at the table -- balanced accuracy and MCC are the
    defensible choices on the imbalanced datasets in this corpus, where F1 moves with the
    positive-class convention -- and a function called ``delta_f1`` quietly makes that
    decision for the reader.

    ``delta`` follows :func:`~qbiocode.utils.fair_selection.select_winners`:
    ``classical - quantum``, so **negative means quantum won**. It is the held-out
    selection delta, not the pooled maximum: an arm is chosen on the other iterations and
    scored on the held-out one, because picking the best arm and reporting its own best
    score is the winner's curse and reports a quantum-vs-classical gap on pure noise (see
    the note on :func:`qml_winner`, which measures +0.0741 on a null).

    Args:
        results_df (pandas.DataFrame): pooled ModelResults-shaped frame.
        metrics (Sequence[str]): metric columns to compare. Higher must be better.
        epsilon (float | Mapping[str, float] | None): the TOST equivalence bound (it no
            longer decides wins; ``margin`` does). A float applies to every metric; a
            mapping gives one per metric; ``None`` derives each from
            :func:`resolution_floor_epsilon`, the smallest bound this design can certify.
        test_size (float): the run's split fraction. Required with
            ``selection='loio'``; ``None`` raises ``ValueError``. With
            ``selection='validation'`` it is ``1/k`` (``k`` from the argument or the
            ``split_k`` column) and may be omitted.
        alpha, seed, baseline, margin, fdr, controls, selection, validation_col, k:
            forwarded to ``select_winners`` (``seed`` is unused there). Under
            ``selection='validation'`` the arm scored on each fold is the one with the
            best validation score on that fold, not a held-out-iteration choice.

    Returns:
        tuple[pandas.DataFrame, dict]: the wide per-dataset table, and
        ``{metric: WinnerReport}`` for the per-arm and selection detail.
    """
    from qbiocode.utils.fair_selection import resolve_split_k, select_winners

    if selection == "validation":
        k = resolve_split_k(results_df, k)
        if test_size is None:
            # The resolution floor needs the same r = 1/(k-1) the selector uses.
            test_size = 1.0 / k
    if test_size is None:
        # Checked here, not left to select_winners, because resolution_floor_epsilon needs
        # it first.
        raise ValueError(
            "test_size is required: the Nadeau-Bengio correction depends on the run's "
            "split fraction, which ModelResults.csv does not record (the pilot used 0.2)."
        )
    present = [m for m in metrics if m in results_df.columns]
    missing = [m for m in metrics if m not in results_df.columns]
    if not present:
        raise ValueError(
            f"none of {list(metrics)} are columns of the results frame. "
            f"Available: {sorted(results_df.columns)[-12:]}"
        )
    if missing:
        logger.warning("skipping metrics absent from the results: %s", missing)

    reports, blocks = {}, []
    for metric in present:
        if epsilon is None:
            eps, sigma = resolution_floor_epsilon(results_df, metric, test_size, alpha)
            if not np.isfinite(eps):
                logger.warning(
                    "cannot derive an epsilon for %r (pooled sigma=%r); skipping it. "
                    "Pass epsilon= explicitly to force a comparison.",
                    metric,
                    sigma,
                )
                continue
        elif isinstance(epsilon, Mapping):
            if metric not in epsilon:
                continue
            eps = float(epsilon[metric])
        else:
            eps = float(epsilon)

        report = select_winners(
            results_df,
            metric=metric,
            epsilon=eps,
            alpha=alpha,
            test_size=test_size,
            seed=seed,
            baseline=baseline,
            margin=margin,
            fdr=fdr,
            controls=controls,
            selection=selection,
            validation_col=validation_col,
            k=k,
        )
        reports[metric] = report
        if report.per_dataset.empty:
            continue

        keep = [
            c
            for c in (
                "Dataset",
                "classical_mean",
                "quantum_mean",
                "delta",
                "se",
                "ci_lo",
                "ci_hi",
                "p_value",
                "within_equivalence",
                "verdict_raw",
                "family",
                "verdict_adjusted",
                "p_adjusted",
            )
            if c in report.per_dataset.columns
        ]
        block = report.per_dataset[keep].copy()
        block["epsilon"] = eps
        block["margin"] = float(margin)
        # Suffix rather than prefix so the frame sorts by dataset-level column then metric,
        # which keeps every column for one metric adjacent when the table is printed wide.
        block = block.rename(
            columns={c: f"{c}__{metric}" for c in block.columns if c != "Dataset"}
        )
        blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=["Dataset"]), reports

    wide = blocks[0]
    for block in blocks[1:]:
        # Merge on 'Dataset', never concat on index: the per-metric frames can carry
        # different dataset subsets (a metric is NaN wherever a model could not score it),
        # and an index-aligned concat then silently pairs row 3 of one metric with row 3 of
        # another -- the exact bug the fair_winner docstring records.
        wide = wide.merge(block, on="Dataset", how="outer")
    return wide.sort_values("Dataset").reset_index(drop=True), reports


def aggregate_benchmark(
    results_root,
    output_dir,
    tag="benchmark",
    metrics=BENCHMARK_METRICS,
    primary_metric="balanced_accuracy",
    epsilon=None,
    alpha=0.05,
    test_size=None,
    seed=0,
    baseline=None,
    kernels_root=None,
    *,
    margin=0.0,
    fdr=0.10,
    controls=None,
    selection="loio",
    validation_col="tuning_score",
    k=None,
):
    """Walk per-dataset run directories and write the corpus-level tables.

    This is the step between "12 LSF jobs finished" and "a table a paper can cite".

    Writes, all prefixed by ``tag``:

    ``<tag>_inventory.csv``
        One row per ``ModelResults.csv`` found, with its model and iteration counts. Read
        this first: a dataset missing models is a job that died, and it will otherwise show
        up as a confident classical win.
    ``<tag>_missing_folds.csv``
        Written only for ``split_mode: manifest`` results: the arms lacking a fold that
        other arms of their dataset have (see :func:`missing_folds`). Empty when every
        arm has the same folds.
    ``<tag>_kernel_diagnostics.csv``
        Written only when ``kernels_root`` is given: per-split kernel-target alignment,
        geometric separation ``g(K_c || K_q)`` over a lam sweep, and RKHS margins, read from
        the ``kernel_dump_dir`` npz files. This is the "why", and it exists only if the run
        was configured to keep the Gram matrices.
    ``<tag>_delta_metrics.csv``
        The wide per-dataset table, every metric.
    ``<tag>_per_arm_<primary_metric>.csv`` / ``<tag>_selection_<primary_metric>.csv``
        The arm-level means and the held-out choices behind the primary metric's verdicts.
    ``<tag>_verdict_summary.csv``
        Verdict counts per metric -- the agreement check. If balanced accuracy and MCC
        disagree about a dataset, that disagreement is a finding about the dataset's class
        balance, not a number to average away. Counts cover the discovery family only;
        ``n_controls`` counts the control datasets, and ``margin``/``fdr`` record the
        rule the verdicts were judged under.

    ``epsilon`` is the equivalence bound only (``None`` derives it per metric from
    :func:`resolution_floor_epsilon`); wins are judged against ``margin`` and the
    Benjamini-Hochberg level ``fdr``. ``margin``, ``fdr``, ``controls``, ``selection``,
    ``validation_col`` and ``k`` are forwarded to :func:`delta_metric_table`. With
    ``selection='validation'`` (``split_mode: manifest`` results) ``test_size`` may be
    omitted: it is ``1/k``.

    Returns:
        dict: ``{'results', 'inventory', 'delta_metrics', 'reports', 'summary',
        'primary_metric', 'epsilon'}``.
    """
    if test_size is None and selection != "validation":
        raise TypeError(
            "test_size is required: it is the 'test_size' of the run that produced these "
            "results, and the Nadeau-Bengio standard error s*sqrt(1/I + r) with "
            "r = test_size/(1-test_size) is a function of it. The pilot configs set 0.2, "
            "so aggregate_benchmark(..., test_size=0.2). There is deliberately no default "
            "-- a wrong one does not fail, it just reports intervals and verdicts for an "
            "experiment nobody ran."
        )
    os.makedirs(output_dir, exist_ok=True)
    results, inventory = collect_model_results(results_root)

    if primary_metric not in results.columns:
        raise ValueError(
            f"primary_metric={primary_metric!r} is not a column of the results. "
            f"Choose one of: {[m for m in metrics if m in results.columns]}"
        )

    ragged = inventory[inventory["n_models"] != inventory["n_models"].max()]
    if not ragged.empty:
        logger.warning(
            "these runs hold fewer models than the fullest one, so their datasets compare "
            "a complete arm against a truncated one -- treat their verdicts as void until "
            "the jobs are re-run:\n%s",
            ragged[["Dataset", "n_models", "n_rows"]].to_string(index=False),
        )

    wide, reports = delta_metric_table(
        results,
        metrics=metrics,
        epsilon=epsilon,
        alpha=alpha,
        test_size=test_size,
        seed=seed,
        baseline=baseline,
        margin=margin,
        fdr=fdr,
        controls=controls,
        selection=selection,
        validation_col=validation_col,
        k=k,
    )

    summary_rows = []
    for metric, report in reports.items():
        # Counted over the discovery family only, like report.quantum_datasets: control
        # datasets are judged in their own Holm family and reported as n_controls.
        pdr = report.per_dataset
        is_control = (
            pdr["family"] == "control" if "family" in pdr else pd.Series(False, index=pdr.index)
        )
        discovery = pdr[~is_control]
        counts = (
            discovery["verdict_adjusted"].value_counts().to_dict()
            if "verdict_adjusted" in discovery
            else {}
        )
        summary_rows.append(
            {
                "metric": metric,
                "epsilon": report.epsilon,
                "margin": report.margin,
                "fdr": report.fdr,
                "n_datasets": len(discovery),
                "n_controls": int(is_control.sum()),
                "quantum_wins": counts.get("quantum_wins", 0),
                "classical_wins": counts.get("classical_wins", 0),
                "equivalent": counts.get("equivalent", 0),
                "inconclusive": counts.get("inconclusive", 0),
                "quantum_datasets": ";".join(report.quantum_datasets),
            }
        )
    summary = pd.DataFrame(summary_rows)

    inventory.to_csv(os.path.join(output_dir, f"{tag}_inventory.csv"), index=False)
    if "split_mode" in results.columns and (results["split_mode"] == "manifest").any():
        missing_folds(results[results["split_mode"] == "manifest"]).to_csv(
            os.path.join(output_dir, f"{tag}_missing_folds.csv"), index=False
        )
    wide.to_csv(os.path.join(output_dir, f"{tag}_delta_metrics.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{tag}_verdict_summary.csv"), index=False)
    if primary_metric in reports:
        primary = reports[primary_metric]
        primary.per_arm.to_csv(
            os.path.join(output_dir, f"{tag}_per_arm_{primary_metric}.csv"), index=False
        )
        primary.selection.to_csv(
            os.path.join(output_dir, f"{tag}_selection_{primary_metric}.csv"), index=False
        )

    # Kernel diagnostics are opt-in because they depend on `kernel_dump_dir` having been set
    # on the run that produced these results. An empty frame here is a finding, not a bug:
    # it means the dumps were never written and no post-hoc analysis can recover them.
    diagnostics = None
    if kernels_root is not None:
        diagnostics = collect_kernel_diagnostics(
            kernels_root,
            output_csv=os.path.join(output_dir, f"{tag}_kernel_diagnostics.csv"),
        )
        if not len(diagnostics):
            logger.warning(
                "no kernel dumps found under %s: the run did not set 'kernel_dump_dir', so "
                "alignment, geometric separation and margins are not recoverable from these "
                "results.",
                kernels_root,
            )

    return {
        "results": results,
        "inventory": inventory,
        "delta_metrics": wide,
        "reports": reports,
        "summary": summary,
        "primary_metric": primary_metric,
        "epsilon": {m: r.epsilon for m, r in reports.items()},
        "kernel_diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------------------
# Kernel-level diagnostics over a finished run
# ---------------------------------------------------------------------------------------
#
# The accuracy table says which arm won. It cannot say *why*, and a quantum-advantage claim
# rests on the why: a kernel that aligns better with the labels, a geometric separation
# g(K_c || K_q) large enough that no classical kernel method could match it, a wider RKHS
# margin. Those are properties of the kernel, not of the score, and they are computable only
# from objects the run does not normally keep -- hence `kernel_dump_dir`.
#
# Two file kinds are read, written by the two quantum kernel arms:
#   gram_<model>_<data_key>.npz   QSVC   K_train, K_test, X_train, y_train, y_test
#   proj_<model>_<data_key>.npz   PQK    Z_train, Z_test, X_train, y_train, y_test,
#                                        best_params
# `data_key` is '<stem>_<embedding>_<n_components>_<iteration>' as built by qprofiler, so a
# row here joins the delta table on Dataset == '<stem>.csv'.

_KERNEL_MODELS = ("qsvc_opt", "qsvc", "pqk_opt", "pqk")


def _parse_dump_name(filename):
    """``gram_qsvc_opt_wdbc_pca_8_3.npz`` -> ``('qsvc_opt', 'wdbc_pca_8_3')``.

    Longest model label first: ``qsvc`` is a prefix of ``qsvc_opt``, and matching it first
    would file every tuned arm as untuned -- the one distinction the pilot depends on, since
    ``grid_search: True`` means only the ``_opt`` arm ever runs.
    """
    base = os.path.basename(filename)
    for prefix in ("gram_", "proj_"):
        if base.startswith(prefix):
            rest = base[len(prefix) : -len(".npz")]
            break
    else:
        return None, None
    for m in _KERNEL_MODELS:
        if rest.startswith(m + "_"):
            return m, rest[len(m) + 1 :]
    return None, rest


def _split_data_key(data_key):
    """``wdbc_pca_8_3`` -> ``('wdbc.csv', 'pca', 8, 3)``.

    The stem may itself contain underscores (``analcatdata_lawsuit``), so this splits from
    the right and only accepts the parse when the last two fields are integers. A dataset
    whose name ends in digits would otherwise be silently mis-split.
    """
    parts = data_key.rsplit("_", 3)
    if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
        return parts[0] + ".csv", parts[1], int(parts[2]), int(parts[3])
    return data_key, None, None, None


def _resolve_gamma(X, gamma):
    """sklearn's ``'scale'`` heuristic, made explicit so the value lands in the table."""
    X = np.asarray(X, dtype=float)
    if gamma == "scale":
        var = float(X.var())
        return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
    if gamma == "auto":
        return 1.0 / X.shape[1]
    return float(gamma)


def collect_kernel_diagnostics(
    kernels_root,
    classical_gamma="scale",
    lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    margin_C=1.0,
    form="symmetric",
    output_csv=None,
):
    """Walk a ``kernel_dump_dir`` tree and compute KTA, g(K_c||K_q) and margins per split.

    The classical comparison kernel is an RBF on the same embedded ``X_train`` the quantum
    arm saw, at ``classical_gamma`` (default sklearn's ``'scale'``). That is a *choice*, and
    the resolved value is written to the table as ``classical_gamma`` so it is never implicit:
    g(K_c||K_q) is a statement about one classical kernel, not about all of them, and a
    single gamma is a weaker claim than a sweep. Re-run with another gamma to bound it.

    Args:
        kernels_root (str): Directory holding the ``.npz`` dumps, searched recursively.
        classical_gamma (str|float): RBF bandwidth for the classical side, or ``'scale'``.
        lams (tuple): Regularisation values for the geometric separation sweep. A single
            lam is not a result -- g falls monotonically with it by an order of magnitude
            over this range.
        margin_C (float): ``C`` for the margin refits. Margins compare only at equal ``C``.
        form (str): ``'symmetric'`` or ``'standard'`` inner form, see
            :func:`~qbiocode.utils.kernel_diagnostics.geometric_separation`.
        output_csv (str|None): Written if given.

    Returns:
        pandas.DataFrame: one row per dumped split, with ``Dataset``, ``embeddings``,
        ``n_components``, ``iteration``, ``model``, ``kernel_source`` and every
        :func:`~qbiocode.utils.kernel_diagnostics.kernel_report` field. Empty (with the
        right columns absent) when nothing was dumped -- which is itself the finding, and
        means ``kernel_dump_dir`` was not set on the run.
    """
    import glob as _glob
    import json as _json

    from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel

    from qbiocode.utils import kernel_diagnostics as kd

    paths = sorted(
        _glob.glob(os.path.join(kernels_root, "**", "gram_*.npz"), recursive=True)
        + _glob.glob(os.path.join(kernels_root, "**", "proj_*.npz"), recursive=True)
    )
    rows = []
    for path in paths:
        model, data_key = _parse_dump_name(path)
        if model is None:
            continue
        dataset, embed, ncomp, it = _split_data_key(data_key)
        if ncomp is None:
            # qprofiler always names a pass '<stem>_<embed>_<n_components>_<iteration>'
            # (qprofiler.py:732), and the last two fields are always integers -- so a key
            # that does not split that way did not come from a profiling pass at all. It
            # came from a tuning trial or a direct compute_* call, both of which take the
            # `data_key=""` default, and every one of them writes to the SAME stem, so the
            # file is also whichever trial happened to finish last.
            #
            # Skipped rather than kept with blank identifiers, because the blank row is
            # worse than an absent one: `Dataset=''` joins against nothing in the
            # delta-metric table while still counting toward every per-dataset tally, so it
            # shifts means and n without ever being visibly wrong. The pilot run that died
            # at trial 10 left five of these ('proj_pqk_.npz') in kernels/.
            logger.warning(
                "skipping %s: data_key %r carries no dataset/embedding/iteration, so this "
                "dump cannot be attributed to a pass (tuning-trial or direct-call dumps "
                "take the empty default and overwrite one another)", path, data_key)
            continue
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as exc:  # a truncated dump from a killed job is not fatal here
            logger.warning("skipping unreadable dump %s: %s", path, exc)
            continue
        keys = set(z.files)
        if "y_train" not in keys or "X_train" not in keys:
            logger.warning("skipping %s: no y_train/X_train, diagnostics need both", path)
            continue
        y_train = z["y_train"]
        X_train = np.asarray(z["X_train"], dtype=float)
        svc_kernel = "rbf"
        if "K_train" in keys:  # QSVC: the fidelity Gram itself
            Kq = np.asarray(z["K_train"], dtype=float)
        elif "Z_train" in keys:  # PQK: rebuild exactly what the SVC head was fitted with
            best = {}
            if "best_params" in keys:
                try:
                    best = _json.loads(str(z["best_params"]))
                except Exception:
                    best = {}
            # The head searches kernel over linear/rbf/poly/sigmoid, so assuming RBF here
            # would silently report a different kernel than the one that produced the score.
            svc_kernel = best.get("kernel", "rbf")
            kw = {} if svc_kernel == "linear" else {"gamma": best.get("gamma", "scale")}
            if kw.get("gamma") == "scale":
                kw["gamma"] = _resolve_gamma(z["Z_train"], "scale")
            Kq = pairwise_kernels(
                np.asarray(z["Z_train"], dtype=float), metric=svc_kernel, **kw
            )
        else:
            logger.warning("skipping %s: neither K_train nor Z_train present", path)
            continue
        n = min(len(y_train), Kq.shape[0], X_train.shape[0])
        if n < 4 or len(np.unique(y_train[:n])) < 2:
            logger.warning("skipping %s: %d usable rows or a single class", path, n)
            continue
        gamma_c = _resolve_gamma(X_train[:n], classical_gamma)
        Kc = rbf_kernel(X_train[:n], gamma=gamma_c)
        report = kd.kernel_report(
            Kc, Kq[:n, :n], y_train[:n], lams=lams, form=form, margin_C=margin_C
        )
        rows.append(
            {
                "Dataset": dataset,
                "embeddings": embed,
                "n_components": ncomp,
                "iteration": it,
                "model": model,
                "quantum_kernel_form": "fidelity" if "K_train" in keys else svc_kernel,
                "classical_gamma": gamma_c,
                "n_features": int(X_train.shape[1]),
                "kernel_source": os.path.basename(path),
                # The run directory, relative to ``kernels_root``. Without it two rows
                # from different configs that happen to share a ``data_key`` -- a
                # re-run left in place, or one CSV reached by two configs -- are
                # byte-identical in every other identifying column, including
                # ``kernel_source``, so a per-dataset mean would double-count them
                # with nothing in the table to reveal it.
                "run_dir": os.path.relpath(os.path.dirname(path), kernels_root),
                **report,
            }
        )
    frame = pd.DataFrame(rows)
    # A split is identified by (Dataset, embeddings, n_components, iteration, model). If that
    # tuple repeats, two run directories produced diagnostics for the same split, and every
    # downstream per-dataset mean silently weights it twice. Say so rather than average it:
    # the fix is a decision about which run is authoritative, which this function cannot make.
    if len(frame):
        keys = ["Dataset", "embeddings", "n_components", "iteration", "model"]
        dup = frame.duplicated(subset=keys, keep=False)
        if dup.any():
            offenders = sorted(frame.loc[dup, "run_dir"].unique())
            logger.warning(
                "%d of %d kernel-diagnostic rows duplicate a (dataset, embedding, "
                "n_components, iteration, model) split across run directories %s. Per-dataset "
                "means over this table will double-count them. Keep one run per split, or "
                "filter on 'run_dir' before aggregating.",
                int(dup.sum()), len(frame), offenders,
            )
    if output_csv and len(frame):
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        frame.to_csv(output_csv, index=False)
    return frame
