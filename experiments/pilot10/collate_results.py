#!/dccstor/boseukb/Q/envs/qbc/bin/python
"""Merge the split-layout runs into one ModelResults.csv, and check they belong together.

    ./collate_results.py                   everything that has landed, incomplete arms flagged
    ./collate_results.py --complete-only   only configs with every iteration
    ./analyze_pilot.py --results-root collated --kernels-root runs

One (dataset, embedding, model) job writes one ModelResults.csv under
runs/<dataset>/results/<config>/<backend>_<timestamp>/. A config resubmitted after a kill
has several such directories. Exactly one run is taken per config -- the latest complete
one, else the one with the most rows -- because a resubmit restarts at iteration 1, so
taking every directory would count those iterations twice. (That is what
qc_winner_finder.collect_model_results would do on runs/: it concatenates every
ModelResults.csv it finds, with no dedupe.)

Output goes to collated/, NOT under runs/, for the same reason: collect_model_results globs
**/ModelResults.csv, so a merged file inside runs/ would be read alongside the files it was
merged from.

    collated/ModelResults.csv       the per-job files, same columns, one row per
                                    (Dataset, embeddings, iteration, model)
    collated/RawDataEvaluation.csv  one row per dataset
    collated/results.pkl            the chosen runs' results.pkl lists, concatenated: one
                                    summary per (config, pass) instead of per pass
    collated/collate_manifest.csv   config -> run_dir, rows, expected, complete

Three checks fail loudly, because each would otherwise produce a table that looks fine:

  * a duplicate (Dataset, embeddings, iteration, model) key -- two configs claiming the
    same observation;
  * RawDataEvaluation disagreeing between the jobs of one dataset -- they did not read the
    same file;
  * the dataset-characteristic columns disagreeing between models of one (dataset,
    embedding, iteration). qprofiler computes them on the EMBEDDED training split, so a
    disagreement means the jobs trained on different features, and their scores are not a
    paired comparison. This is the production check of the seeded-UMAP fix: before it,
    this failed for every umap pass after the first.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from status import best_run, find_configs, read_config  # noqa: E402  (one definition of "the run that counts")

KEY = ["Dataset", "embeddings", "iteration", "model"]
#: Relative tolerance for "the same number". The measured cross-run noise on these columns
#: is ~1e-15 (a kernel-density sum in a different order); a real feature mismatch moves them
#: at the first or second significant figure.
RTOL, ATOL = 1e-6, 1e-9


def _disagreement(frame, cols):
    """Per column, the largest relative spread among ``frame``'s rows (inf: not even the
    same kind of value).

    Non-finite values are compared by kind before anything is subtracted: some
    characteristics are legitimately inf or nan ('Coefficient of Variation %' of a
    zero-mean column is inf), and inf - inf is nan, which compares False against any
    tolerance -- so an inf in one job and a number in another would read as agreement.
    """
    worst = {}
    for col in cols:
        raw = frame[col]
        v = pd.to_numeric(raw, errors="coerce").astype(float)
        if v.isna().sum() > raw.isna().sum():       # not a number column: compare as text
            if raw.astype(str).nunique() > 1:
                worst[col] = float("inf")
            continue
        kind = set(np.select([np.isnan(v), np.isposinf(v), np.isneginf(v)], [1, 2, 3], 0))
        if len(kind) > 1:
            worst[col] = float("inf")
            continue
        if kind != {0} or len(v) < 2:
            continue
        spread = v.max() - v.min()
        scale = max(abs(v.max()), abs(v.min()), 1.0)
        if spread > ATOL + RTOL * scale:
            worst[col] = spread / scale
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "collated"))
    ap.add_argument("--complete-only", action="store_true",
                    help="drop configs that have not landed every iteration")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write the output even if the consistency checks fail")
    args = ap.parse_args()

    if os.path.abspath(args.out_dir).startswith(os.path.abspath(args.runs_dir) + os.sep):
        sys.exit("--out-dir must not be inside --runs-dir (see the module docstring)")
    paths = find_configs(args.runs_dir)
    if not paths:
        sys.exit(f"no configs under {args.runs_dir}")

    manifest, frames, raws, pkls, problems = [], [], [], [], []
    for p in paths:
        cfg = read_config(p)
        run_dir, rows = best_run(cfg)
        complete = bool(cfg["expected"]) and rows >= cfg["expected"]
        manifest.append({"config": cfg["config"], "dataset": cfg["dataset"], "rows": rows,
                         "expected": cfg["expected"], "complete": complete,
                         "n_runs": len(glob.glob(os.path.join(os.path.dirname(p), "results",
                                                              cfg["config"], "*"))),
                         "run_dir": run_dir or ""})
        if run_dir is None or rows == 0 or (args.complete_only and not complete):
            continue
        frames.append(pd.read_csv(os.path.join(run_dir, "ModelResults.csv")))
        raw = os.path.join(run_dir, "RawDataEvaluation.csv")
        if os.path.exists(raw):
            r = pd.read_csv(raw)
            r["__config"] = cfg["config"]
            raws.append(r)
        pk = os.path.join(run_dir, "results.pkl")
        if os.path.exists(pk):
            with open(pk, "rb") as fh:
                pkls.extend(pickle.load(fh))

    man = pd.DataFrame(manifest)
    missing = man[man["rows"] == 0]
    short = man[(man["rows"] > 0) & ~man["complete"]]
    print(f"{len(man)} configs: {int(man['complete'].sum())} complete, {len(short)} partial, "
          f"{len(missing)} with no results")
    for _, r in short.iterrows():
        print(f"  partial  {r['config']}  {r['rows']}/{r['expected']}"
              + ("  (dropped: --complete-only)" if args.complete_only else ""))
    if len(missing) and len(missing) <= 40:
        print("  no results: " + ", ".join(missing["config"]))
    if not frames:
        sys.exit("nothing to collate yet")

    # Column order: the first file's, then anything later files add (a tuned model reports
    # BestParams_Tuned, an untuned one Model_Parameters).
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    res = pd.concat(frames, ignore_index=True)[cols]
    res = res.sort_values(KEY, kind="stable").reset_index(drop=True)

    dup = res[res.duplicated(KEY, keep=False)]
    if len(dup):
        problems.append(f"{len(dup)} rows share a (Dataset, embeddings, iteration, model) key:\n"
                        + dup[KEY].drop_duplicates().head(20).to_string(index=False))

    raw_cols = []
    if raws:
        rawdf = pd.concat(raws, ignore_index=True)
        raw_cols = [c for c in rawdf.columns if c not in ("Dataset", "__config")]
        for ds, g in rawdf.groupby("Dataset"):
            bad = _disagreement(g, raw_cols)
            if bad:
                problems.append(f"RawDataEvaluation of {ds} differs between its jobs: "
                                + ", ".join(f"{k} ({v:.2g})" for k, v in list(bad.items())[:6]))
        rawdf = rawdf.drop_duplicates("Dataset").drop(columns="__config")

    # The dataset-characteristic columns of every model row of one (Dataset, embedding,
    # iteration) were computed on that pass's embedded training split, by a different job
    # per model. They must agree.
    char_cols = [c for c in raw_cols if c in res.columns]
    n_groups, bad_groups = 0, []
    for key, g in res.groupby(["Dataset", "embeddings", "iteration"]):
        if len(g) < 2:
            continue
        n_groups += 1
        bad = _disagreement(g, char_cols)
        if bad:
            bad_groups.append((key, bad))
    if bad_groups:
        lines = [f"  {k[0]} {k[1]} iter {k[2]}: "
                 + ", ".join(f"{c} ({v:.2g})" for c, v in list(b.items())[:4])
                 for k, b in bad_groups[:15]]
        problems.append(f"{len(bad_groups)} of {n_groups} (dataset, embedding, iteration) "
                        f"passes saw DIFFERENT training features in different jobs:\n"
                        + "\n".join(lines))
    else:
        print(f"feature consistency: {n_groups} (dataset, embedding, iteration) passes "
              f"checked across jobs, {len(char_cols)} columns each, all agree")

    if problems:
        print("\n!! " + "\n!! ".join(problems), file=sys.stderr)
        if not args.allow_mismatch:
            sys.exit("not writing collated output (--allow-mismatch to write anyway)")

    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "ModelResults.csv"), index=False)
    if raws:
        rawdf.to_csv(os.path.join(args.out_dir, "RawDataEvaluation.csv"), index=False)
    tmp = os.path.join(args.out_dir, "results.pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(pkls, fh)
    os.replace(tmp, os.path.join(args.out_dir, "results.pkl"))
    man.to_csv(os.path.join(args.out_dir, "collate_manifest.csv"), index=False)
    print(f"wrote {len(res)} rows ({res['Dataset'].nunique()} datasets, "
          f"{res['model'].nunique()} models) to {args.out_dir}/ModelResults.csv")
    def short(p):
        rel = os.path.relpath(p)
        return rel if not rel.startswith("..") else os.path.abspath(p)
    print(f"next: ./analyze_pilot.py --results-root {short(args.out_dir)} "
          f"--kernels-root {short(args.runs_dir)}")


if __name__ == "__main__":
    main()
