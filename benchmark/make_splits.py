#!/usr/bin/env python
"""Freeze the outer CV folds as row-index manifests.

The benchmark's requirement is that every method, embedding and Optuna trial sees
byte-identical train, validation and test rows. A seed does not deliver that on its own:
the rows a seed selects depend on the row order of the file it is given, so the real
dependency is (file content, repeat, fold id). These manifests make that explicit -- each
records the sha256 of the CSV it was computed from, and QProfiler re-checks that hash
under ``split_mode: manifest`` before using it.

Protocol (make_splits/2.0, manifest schema_version 2): stratified k-fold repeated
``--repeats`` times, repeat r shuffled with seed + r, so repeat 0 is exactly the
make_splits/1.0 assignment for the same seed. Each fold record carries its validation
rows explicitly: under ``next_fold`` (TabZilla) the validation rows of fold f are the test
rows of fold (f + 1) mod k, and the tuning trials fit on ``train \\ val``. The schema is
documented in qbiocode/apps/qprofiler/split_manifest.py, and every manifest written here
is read back with that module before the run counts as a success.

Indices rather than materialised per-fold CSVs, for two reasons. Copying would multiply
disk by k on the wide datasets (colon_cancer p=2000, GAMETES_* p=1000), and a stray
``<name>_train.csv`` in a directory QProfiler globs would be re-split and reported as a
dataset in its own right -- silently, with plausible numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:
    from qbiocode.apps.qprofiler import split_manifest
except ImportError:  # run outside an env that has QBioCode installed: use this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qbiocode.apps.qprofiler import split_manifest

GENERATOR_VERSION = "make_splits/2.0"
SCHEMA_VERSION = split_manifest.SCHEMA_VERSION
VALIDATION = "next_fold"

# Per-fold class ratio must stay within this absolute tolerance of the whole dataset's.
# Stratification guarantees it up to integer rounding; the check exists to catch a
# mis-wired splitter rather than to police sklearn. It is a floor, not the whole rule:
# see balance_tolerance for the per-fold widening that small folds need.
BALANCE_TOLERANCE = 0.05

# Report-only floor: a validation side with fewer minority rows than this moves balanced
# accuracy in steps of 1/(2 * minority) or coarser, so validation-based selection there is
# noisy. Such datasets are listed in the run summary and spec.yaml, not dropped.
LOW_VAL_MINORITY = 3


def balance_tolerance(fold_size: int, rows: int = 1) -> float:
    """Allowed |fold positive rate - dataset positive rate| for a fold of ``fold_size`` rows.

    A fold's rate moves in steps of 1/fold_size, so a correctly stratified fold can sit
    one row off the dataset's rate. Below 20 rows that single-row step alone exceeds
    BALANCE_TOLERANCE (parity5, n=32, has test folds of 6-7 rows: one row is ~0.15), and a
    fixed 0.05 would reject correct folds. Widening to one row keeps the check meaningful
    -- a mis-wired splitter is off by far more than one row -- without failing on rounding.

    A side assembled from several folds inherits the rounding of each fold it excludes:
    the fit side is the training fold minus the validation fold, i.e. the complement of
    two folds, each of which may be one row off, so it is checked with ``rows=2``.

    Args:
        fold_size: Number of rows on the side of the split being checked.
        rows: Rounding steps the side may legitimately be off by.

    Returns:
        max(BALANCE_TOLERANCE, rows / fold_size).
    """
    return max(BALANCE_TOLERANCE, float(rows) / fold_size)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_manifest(path: Path, payload: dict) -> None:
    """Write JSON with each index list on one compact line.

    Fully pretty-printing would put one integer per line -- hundreds of thousands of lines
    for the larger datasets. This keeps the manifest readable and byte-deterministic, so
    re-running reproduces an identical file and the sha256 claim is testable.
    """
    lines = ["{"]
    scalars = [(k, v) for k, v in payload.items() if k != "folds"]
    for key, value in scalars:
        lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
    lines.append('  "folds": [')
    for position, fold in enumerate(payload["folds"]):
        comma = "," if position < len(payload["folds"]) - 1 else ""
        lines.append("    {")
        lines.append(f'      "repeat": {fold["repeat"]},')
        lines.append(f'      "fold": {fold["fold"]},')
        lines.append(f'      "train": {json.dumps(fold["train"], separators=(",", ":"))},')
        lines.append(f'      "val": {json.dumps(fold["val"], separators=(",", ":"))},')
        lines.append(f'      "test": {json.dumps(fold["test"], separators=(",", ":"))}')
        lines.append(f"    }}{comma}")
    lines.append("  ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def _minority(y_side: np.ndarray) -> int:
    """Rows of the rarer class on one side of a split (0 when a class is absent)."""
    return int(min(np.bincount(y_side, minlength=2)))


def _repeat_folds(frame: pd.DataFrame, y: np.ndarray, k: int, seed: int,
                  group_col: str | None) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """The (train, test) index pairs of one repeat, shuffled with ``seed``."""
    n = len(frame)
    if group_col:
        # StratifiedGroupKFold rather than GroupKFold: group disjointness *and* class
        # balance are both needed, and GroupKFold ignores y, which on a small imbalanced
        # set can hand back a fold with no positives at all. Unused for these 84 (all
        # group_col: null) but wired in now, since omics data will need it. Under
        # next_fold the validation rows are a whole test fold, so they stay group-
        # disjoint from the fit rows too.
        groups = frame[group_col].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n), y, groups=groups)), "StratifiedGroupKFold"
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n), y)), "StratifiedKFold"


def split_one(dataset_dir: Path, k: int, seed: int, repeats: int = 3) -> tuple[dict, list[dict]]:
    """Build the schema-2 manifest and the inventory rows of one curated dataset.

    Repeat r is a StratifiedKFold (StratifiedGroupKFold with a group_col) shuffled with
    ``seed + r``. Per-repeat seeds rather than one RepeatedStratifiedKFold stream make each
    repeat regenerable on its own, and keep repeat 0 identical to make_splits/1.0.

    Args:
        dataset_dir: ``<datasets>/<dataset_id>`` holding ``<dataset_id>.csv`` and meta.yaml.
        k: Folds per repeat (>= 3: next_fold validation needs rows left to fit on).
        seed: Seed of repeat 0.
        repeats: Number of repeats.

    Returns:
        (manifest payload for dump_manifest, one inventory row per (repeat, fold)).

    Raises:
        ValueError: On a changed CSV, k above the minority count, or bad arguments.
        AssertionError: When a split breaks disjointness, coverage, class presence or
            balance; the message names the dataset, repeat and fold.
    """
    if k < 3:
        raise ValueError(f"k={k}: next_fold validation needs k >= 3")
    if repeats < 1:
        raise ValueError(f"repeats={repeats}: need at least one repeat")

    meta = yaml.safe_load((dataset_dir / "meta.yaml").read_text())
    dataset_id = meta["dataset_id"]
    csv_path = dataset_dir / f"{dataset_id}.csv"

    # The CSV must be exactly the one curate.py hashed. If it has been touched since,
    # every manifest built from it would encode row positions that no longer mean
    # anything -- refuse rather than emit indices into an unknown file.
    actual = sha256_of(csv_path)
    if actual != meta["sha256"]:
        raise ValueError(
            f"{dataset_id}: {csv_path.name} has changed since curation\n"
            f"  meta.yaml sha256 : {meta['sha256']}\n"
            f"  file sha256      : {actual}\n"
            "  re-run curate.py, or restore the file"
        )

    frame = pd.read_csv(csv_path)
    y = frame["label"].to_numpy()
    n = len(frame)

    minority_n = int(pd.Series(y).value_counts().min())
    if k > minority_n:
        raise ValueError(
            f"{dataset_id}: k={k} exceeds minority class count {minority_n}; "
            "at least one fold would contain no minority sample"
        )

    group_col = meta.get("group_col")
    whole_rate = float(np.mean(y == 1))
    repeat_seeds = [int(seed) + r for r in range(repeats)]
    fold_records, inventory, protocol = [], [], None
    for repeat, repeat_seed in enumerate(repeat_seeds):
        folds, protocol = _repeat_folds(frame, y, k, repeat_seed, group_col)
        tests = [np.sort(test_idx) for _, test_idx in folds]
        for fold_id, (train_idx, _) in enumerate(folds):
            where = f"{dataset_id} repeat {repeat} fold {fold_id}"
            train_idx = np.sort(train_idx)
            test_idx = tests[fold_id]
            # next_fold: validate on the rows the next fold tests. Each row is then a
            # validation row exactly once per repeat, with no extra seed to record.
            val_idx = tests[(fold_id + 1) % k]
            fit_idx = np.setdiff1d(train_idx, val_idx)

            overlap = np.intersect1d(train_idx, test_idx)
            if overlap.size:
                raise AssertionError(f"{where}: {overlap.size} rows in both sides")
            if len(train_idx) + len(test_idx) != n:
                raise AssertionError(f"{where}: sides do not cover all {n} rows")
            if np.setdiff1d(val_idx, train_idx).size or len(fit_idx) + len(val_idx) != len(train_idx):
                raise AssertionError(f"{where}: validation rows are not inside the training fold")

            # Every side is checked, each against its own granularity (balance_tolerance).
            # The train side is the complement of the test side, so it is off by less;
            # checking it too catches a splitter that returns the sides swapped. The fit
            # side excludes two folds (val and test), so it may be off by two rows.
            # A side with a single class cannot be tuned or scored: that is an error
            # before it is a balance question.
            sides = (("test", test_idx, 1), ("train", train_idx, 1),
                     ("fit", fit_idx, 2), ("val", val_idx, 1))
            for side, idx, rows in sides:
                if np.unique(y[idx]).size < 2:
                    raise AssertionError(
                        f"{where}: the {side} side ({len(idx)} rows) holds a single class"
                    )
                rate = float(np.mean(y[idx] == 1))
                tolerance = balance_tolerance(len(idx), rows)
                if abs(rate - whole_rate) > tolerance:
                    raise AssertionError(
                        f"{where}: {side} positive rate {rate:.4f} differs "
                        f"from the dataset's {whole_rate:.4f} by more than {tolerance:.4f} "
                        f"({len(idx)} {side} rows)"
                    )

            fold_records.append({
                "repeat": repeat,
                "fold": fold_id,
                "train": [int(i) for i in train_idx],
                "val": [int(i) for i in val_idx],
                "test": [int(i) for i in test_idx],
            })
            inventory.append({
                "dataset_id": dataset_id, "repeat": repeat, "fold": fold_id,
                "iteration": repeat * k + fold_id + 1,
                "n_train": int(len(train_idx)), "n_fit": int(len(fit_idx)),
                "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
                "train_minority": _minority(y[train_idx]),
                "fit_minority": _minority(y[fit_idx]),
                "val_minority": _minority(y[val_idx]),
                "test_minority": _minority(y[test_idx]),
                "test_pos_rate": round(float(np.mean(y[test_idx] == 1)), 6),
                # gen_configs needs this: UMAP's default n_neighbors=30 exceeds the rows
                # of the smallest datasets. The tuning stage embeds the fit rows only
                # (parity5, n=32 -> 18-20 fit rows), so that is the binding limit.
                "n_neighbors_max": int(len(fit_idx) - 1),
                "n_components_max": int(meta["p"]),
            })

        # Every row is tested exactly once per repeat -- this is what distinguishes
        # K-fold from the repeated stratified holdout `iter` produces, whose test sets
        # overlap.
        covered = np.sort(np.concatenate(tests))
        if not np.array_equal(covered, np.arange(n)):
            raise AssertionError(
                f"{dataset_id} repeat {repeat}: test folds do not partition the rows "
                f"({len(covered)} test slots, {len(np.unique(covered))} distinct, n={n})"
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "sha256": meta["sha256"],
        "n": int(n),
        "k": int(k),
        "n_repeats": int(repeats),
        "protocol": protocol,
        "validation": VALIDATION,
        "seed": int(seed),
        "repeat_seeds": repeat_seeds,
        "group_col": group_col,
        "generator_version": GENERATOR_VERSION,
        "folds": sorted(fold_records, key=lambda f: (f["repeat"], f["fold"])),
    }
    return manifest, inventory


def verify_written(path: Path, dataset_dir: Path, manifest: dict) -> None:
    """Read a written manifest back the way QProfiler will, and refuse a mismatch.

    Loads it with split_manifest.load_manifest (schema, partition, next_fold and fit-side
    checks), then verify_dataset against the CSV (stem, sha256, row count, both classes
    on every fit / val / test side), and finally compares every fold with what was built.

    Raises:
        split_manifest.ManifestError: When the reader rejects the file.
        AssertionError: When the file reads back as different splits.
    """
    loaded = split_manifest.load_manifest(path)
    csv_path = dataset_dir / f"{manifest['dataset_id']}.csv"
    y = pd.read_csv(csv_path)["label"].to_numpy()
    split_manifest.verify_dataset(loaded, dataset_file=csv_path, n_rows=len(y), y=y)
    if len(loaded.splits) != len(manifest["folds"]):
        raise AssertionError(f"{path.name}: {len(loaded.splits)} splits read back, "
                             f"{len(manifest['folds'])} written")
    for split, record in zip(loaded.splits, manifest["folds"]):
        same = (split.repeat == record["repeat"] and split.fold == record["fold"]
                and split.train_idx.tolist() == record["train"]
                and split.val_idx.tolist() == record["val"]
                and split.test_idx.tolist() == record["test"])
        if not same:
            raise AssertionError(f"{path.name} repeat {record['repeat']} fold "
                                 f"{record['fold']}: reads back as different rows")


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", type=Path, default=here / "datasets")
    parser.add_argument("--out", type=Path, default=here / "splits" / "v2",
                        help="output directory (default splits/v2; splits/v1 is left alone)")
    parser.add_argument("-k", "--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeats of the k-fold; repeat r is shuffled with seed + r")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-path", type=Path, default=Path(sys.prefix),
                        help="runtime environment the benchmark will execute in (provenance; "
                             "default: the environment running this script)")
    args = parser.parse_args(argv)
    if args.folds < 3:
        parser.error(f"--folds {args.folds}: next_fold validation needs k >= 3")
    if args.repeats < 1:
        parser.error(f"--repeats {args.repeats}: need at least one repeat")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(d for d in args.datasets.resolve().iterdir()
                          if d.is_dir() and (d / "meta.yaml").exists())
    if not dataset_dirs:
        raise SystemExit(f"no curated datasets under {args.datasets}; run curate.py first")

    inventory, failures, written, low_val = [], [], [], {}
    for dataset_dir in dataset_dirs:
        path = None
        try:
            manifest, records = split_one(dataset_dir, args.folds, args.seed, args.repeats)
            path = out / f"{manifest['dataset_id']}.json"
            dump_manifest(path, manifest)
            verify_written(path, dataset_dir, manifest)
        except Exception as error:  # noqa: BLE001 - report every bad dataset, not just the first
            # A manifest that does not read back is worse than none: QProfiler would
            # refuse it mid-run, or worse, accept rows that are not the ones checked here.
            if path is not None and path.exists():
                path.unlink()
            failures.append({"dataset": dataset_dir.name, "error": str(error)})
            print(f"  FAIL  {dataset_dir.name}: {error}")
            continue
        written.append(manifest["dataset_id"])
        inventory.extend(records)
        smallest = min(r["val_minority"] for r in records)
        if smallest < LOW_VAL_MINORITY:
            low_val[manifest["dataset_id"]] = int(smallest)
        sizes = "/".join(str(r["n_fit"]) for r in records if r["repeat"] == 0)
        print(f"  ok    {manifest['dataset_id']:<34} n={manifest['n']:<6} "
              f"fit sizes (repeat 0) {sizes}  min val minority {smallest}")

    pd.DataFrame(inventory).to_csv(out / "index.csv", index=False)

    spec = {
        "version": "v2",
        "schema_version": SCHEMA_VERSION,
        "protocol": "StratifiedKFold",
        "k": int(args.folds),
        "n_repeats": int(args.repeats),
        "seed": int(args.seed),
        "repeat_seeds": [int(args.seed) + r for r in range(args.repeats)],
        "shuffle": True,
        "validation": VALIDATION,
        "split_mode": "manifest",
        "generator_version": GENERATOR_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_datasets": len(written),
        "n_splits": len(inventory),
        "low_validation_minority": {
            "threshold": LOW_VAL_MINORITY,
            "datasets": low_val,
        },
        "datasets_root": str(args.datasets.resolve()),
        "runtime_env": str(args.env_path.resolve()),
        "runtime_python": str((args.env_path / "bin" / "python").resolve()),
        "generated_by_python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "notes": [
            "Indices are row positions into datasets/<dataset_id>/<dataset_id>.csv as hashed "
            "in that dataset's meta.yaml; the sha256 in each manifest pins that exact file.",
            "Consumed with split_mode=manifest (split_dir = this directory, splits = 'all' "
            "or a list of global iterations repeat * k + fold + 1). QProfiler then checks "
            "each CSV's sha256, row count and per-side class coverage against its manifest. "
            "Under split_mode=internal QProfiler ignores these files and splits with "
            "train_test_split itself.",
            "Repeat r uses StratifiedKFold(shuffle=True, random_state=seed + r); repeat 0 "
            "equals the make_splits/1.0 (splits/v1) assignment for the same seed.",
            "validation=next_fold: the val rows of fold f are the test rows of fold "
            "(f + 1) mod k; tuning trials fit on train minus val and score on val, the "
            "chosen configuration is refit on train and tested once.",
            "low_validation_minority lists datasets whose smallest validation side has "
            "fewer minority rows than the threshold; they are kept, and their validation "
            "scores are coarse.",
        ],
    }
    (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    print(f"\nwrote {len(written)} manifests ({len(inventory)} splits) to {out}")
    if low_val:
        print(f"  note: validation minority below {LOW_VAL_MINORITY} (kept): "
              + ", ".join(f"{d}={m}" for d, m in sorted(low_val.items())))
    if failures:
        print(f"  FAILURES: {len(failures)}")
        for failure in failures:
            print(f"    {failure['dataset']}: {failure['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
