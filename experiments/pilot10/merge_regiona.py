#!/usr/bin/env python3
"""Merge the Region A quantum-tuning runs into one table, keyed so the notebook can pair them.

    ./merge_regiona.py --collated-root $B --out $B/ModelResults_regionA.csv
    ./merge_regiona.py ... --baseline $B/ModelResults_merged.csv   also check the pairing

Reads <collated-root>/collated_<run>/ModelResults.csv for each run of the campaign
(full1-qt-rdm1 ... tcga1-qt-rdm2; collate_results.py writes them) and adds three columns:

    run           full1-qt-rdm2nn, ...
    corpus        tabular / tcga -- the labels load_merged gives the baseline
    measurement   rdm1 / rdm2_nn / rdm2

The model column gets the measurement as a suffix, ``pqk_opt@rdm2`` and
``qpl_opt_catboost@rdm2``: the same arm at three measurements is three arms, and a bare
``pqk_opt`` would also collide with the baseline's ``pqk_opt`` in the notebook.

Incomplete runs are fine: whatever has landed is merged, and the notebook restricts each
comparison to the cells every arm in it has. Four checks fail loudly instead:

  * a run with no collated file -- a merge that silently lacks a run reads as a result;
  * selected_measurement disagreeing with the run's measurement -- the measurement was
    meant to be fixed, and was not (the trial-0 bug this campaign already had once);
  * a duplicate (run, Dataset, embeddings, iteration, model) key;
  * with --baseline: a cell whose manifest_sha256, dataset_sha256 or n_test differs from
    the baseline's -- then the new rows are not on the baseline's split, and "improvement"
    would compare different test rows.
"""
import argparse
import os
import sys

import pandas as pd

RUNS = {f"{corpus_run}-qt-{tag}": (corpus, measurement)
        for corpus_run, corpus in (("full1", "tabular"), ("tcga1", "tcga"))
        for tag, measurement in (("rdm1", "rdm1"), ("rdm2nn", "rdm2_nn"), ("rdm2", "rdm2"))}
CELL = ["Dataset", "embeddings", "iteration"]
PAIRING = ["manifest_sha256", "dataset_sha256", "n_test"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collated-root", required=True,
                    help="directory holding collated_<run>/ for every run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", help="ModelResults_merged.csv, to check the pairing")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = ap.parse_args()

    frames, problems = [], []
    for run in args.runs:
        path = os.path.join(args.collated_root, f"collated_{run}", "ModelResults.csv")
        if not os.path.exists(path):
            problems.append(f"{run}: no {path} -- run collate_results.py for it first")
            continue
        corpus, measurement = RUNS[run]
        df = pd.read_csv(path, low_memory=False)
        if "selected_measurement" in df.columns:
            got = set(df["selected_measurement"].dropna().unique())
            if got - {measurement}:
                problems.append(f"{run}: selected_measurement {sorted(got)}, expected only "
                                f"{measurement!r} -- the measurement was not held fixed")
        df = pd.concat([pd.DataFrame({"run": run, "corpus": corpus,
                                      "measurement": measurement}, index=df.index), df],
                       axis=1)
        df["model"] = df["model"] + "@" + measurement
        frames.append(df)
        print(f"{run:18s} {len(df):6d} rows  {df.Dataset.nunique():3d} datasets  "
              f"{dict(df.model.str.split('@').str[0].value_counts())}")
    if not frames:
        sys.exit("\n".join(problems) or "nothing to merge")

    res = pd.concat(frames, ignore_index=True)
    dup = res.duplicated(["run"] + CELL + ["model"], keep=False)
    if dup.any():
        problems.append(f"{int(dup.sum())} rows share a (run, cell, model) key")

    if args.baseline:
        base = pd.read_csv(args.baseline, low_memory=False, usecols=CELL + PAIRING)
        base = base.drop_duplicates(CELL).set_index(CELL)
        new = res.drop_duplicates(CELL).set_index(CELL)[PAIRING]
        missing = new.index.difference(base.index)
        if len(missing):
            problems.append(f"{len(missing)} new cells have no baseline cell, e.g. "
                            f"{list(missing[:3])}")
        both = new.index.intersection(base.index)
        for col in PAIRING:
            bad = (new.loc[both, col].astype(str) != base.loc[both, col].astype(str)).sum()
            if bad:
                problems.append(f"{bad} cells differ from the baseline in {col}: not the "
                                f"same split, so not a paired comparison")
        if not problems:
            print(f"pairing: all {len(both)} cells match the baseline's split and data")

    if problems:
        sys.exit("!! " + "\n!! ".join(problems) + "\nnot writing " + args.out)

    tmp = args.out + ".tmp"
    res.to_csv(tmp, index=False)
    back = pd.read_csv(tmp, low_memory=False, usecols=["run", "model"])
    if len(back) != len(res) or back["model"].isna().any():
        sys.exit(f"!! read-back of {tmp} has {len(back)} rows, wrote {len(res)}; not replacing")
    os.replace(tmp, args.out)
    print(f"wrote {len(res)} rows, {res.Dataset.nunique()} datasets, to {args.out}")


if __name__ == "__main__":
    main()
