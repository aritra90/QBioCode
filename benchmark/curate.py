#!/usr/bin/env python
"""Freeze the benchmark dataset layer: one canonical, numeric CSV per dataset.

QProfiler globs ``*.csv`` from ``folder_path`` and treats *every* match as a complete
dataset, splitting it itself. That makes a materialised ``<name>_train.csv`` actively
dangerous: left anywhere QProfiler reads, it is re-split and silently reported as a
dataset in its own right. So this script writes exactly one CSV per dataset directory
and refuses to finish if anything train/test-shaped appears under the output root.

Everything here is deterministic: same inputs and same --seed reproduce byte-identical
CSVs, which is what the recorded sha256 in each meta.yaml is for. The split manifests
that reference those hashes are built separately by make_splits.py.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GENERATOR_VERSION = "curate/1.0"

SOURCES = {
    "libsvm": "libsvm_data",
    "openml": "openMLCC18",
    "pmlb": "pmlb_data",
}

# Dropped with the user: both are dirty enough that the preprocessing choices, not the
# learners, would drive their outcome. cylinder-bands has a 71-level `customer` column,
# two constant columns and 29% NaN in paper_mill_location; dresses-sales has duplicated
# case-variant levels (Low/low, Summer/winter/Winter, M/S/small) and 53% NaN in V11.
#
# The six below are CROSS-SOURCE COPIES: the same dataset reaching us from two of the three
# sources. They are excluded rather than clustered because two files of the same rows are
# one dataset, and keeping both counts one observation twice in every corpus-level test --
# the cluster is the unit of independence for meta_regression's wild cluster bootstrap and
# for holdout.py's draw, so a duplicate inflates G.
#
# Which copy is kept follows a source precedence, libsvm > openml > pmlb. That rule
# reproduces, on all six pairs, the dedup already recorded in
# experiments/cluster_map_draft.csv (which simply omits these six: 91 draft rows - 13
# synthetic = 78 real, and 84 curated - 6 = 78).
#
# NOT "keep the copy with more features", which is the intuitive rule and the wrong one
# here. It decides only two of the six -- diabetes, ionosphere and sonar are byte-identical
# at equal width, so it ties -- and it disagrees on german_numer. More columns almost never
# means more information, it means one-hot instead of ordinal: credit-g's 61 columns are
# the one-hot expansion of the same 20 attributes libsvm__german_numer carries in 24, and
# tic-tac-toe's 27 are 9 squares x 3 states against 9 ordinal. Width is also not neutral in
# this benchmark, because the quantum arms encode one qubit per feature and
# generate_pilot_configs.backend_for() bands a dataset by its width: pmlb__tic_tac_toe at
# 9 features would run unembedded on 9 qubits, while openml__tic-tac-toe at 27 is reduced
# to 8 components and runs the pca/umap arms. Choosing by width would quietly change which
# arms each dataset exercises.
EXCLUDE = {
    ("openml", "cylinder-bands"), ("openml", "dresses-sales"),
    ("openml", "breast-w"),    # = libsvm__breast_cancer, 683 rows (Wisconsin), 9 vs 10 cols
    ("openml", "credit-g"),    # = libsvm__german_numer, 1000 rows, one-hot 61 vs 24 cols
    ("openml", "diabetes"),    # = libsvm__diabetes, 768 x 8, identical
    ("pmlb", "ionosphere"),    # = libsvm__ionosphere, 351 x 33, identical
    ("pmlb", "sonar"),         # = libsvm__sonar, 208 x 60, identical
    ("pmlb", "tic_tac_toe"),   # = openml__tic-tac-toe, 958 rows, ordinal 9 vs one-hot 27
}

# Row-identifier columns dropped before anything else, keyed like EXCLUDE. Each one is a
# pure row identifier, and most also carry label signal (single-column label AUC on the
# raw file, measured 2026-09-30): some pmlb files are sorted or numbered in a way that
# tracks the class, so a learner can score above chance off the row number alone. clean1's
# molecule_name is the worst: it is the multiple-instance bag id, every conformation of a
# molecule shares its label, and it separates the classes perfectly (AUC 1.0).
#   analcatdata_bankruptcy Company 0.624, clean1 conformation_name 0.608 and
#   molecule_name 1.000, analcatdata_japansolvent Firm 0.538, backache id 0.511.
# A listed column that is absent from the raw file fails that dataset: the map is stale.
#
# The last two were added 2026-10-09, promoted from suspected_id_columns, which had
# reported and kept them. Their AUCs are measured on the CURATED csv rather than the raw
# file, so they are not directly comparable with the five above:
#
#   biomed Hospital_identification_number_for_blood_sample  0.784
#       against a permutation null whose 95th percentile is 0.582 at this n (209) and class
#       balance (134/75), so it is nowhere near chance. It is a blood-sample identification
#       number: cases and controls were evidently collected or numbered in blocks. Four real
#       measurements score higher (ml 0.868, m3 0.809, Age_of_patient 0.795, m4 0.789), which
#       is not a defence -- a measurement is meant to predict and an identifier is not.
#   prnn_crabs index  not measured exactly; below 0.625, the 5th-ranked column in the same
#       run, so plausibly near chance. Dropped on principle rather than on evidence: an
#       arbitrary row number is not a feature, and keeping it cannot help.
#
# What decided biomed was not the scores but the META-FEATURES. evaluate() computes the 141
# complexity measures on the training split, including this column, and those are the
# regressors the meta-analysis uses to explain when a quantum arm wins. Measured on a
# fixture of the same shape, adding one block-allocated id moves: max Fisher discriminant
# ratio +1194%, condition number +9.6%, effective rank +9.2%. The dataset's row in that
# design matrix would describe a dataset that does not exist.
#
# What did NOT decide it, because it was tested and did not hold: the idea that an
# axis-aligned leak favours the tree arms over the quantum kernels. Over 40 stratified
# splits every arm gained about the same (DecisionTree +0.214, RandomForest +0.202,
# LogisticRegression +0.193, SVC-rbf +0.220 balanced accuracy), so the leak inflates the
# dataset rather than tilting the comparison. It does compress the spread between arms
# (0.082 -> 0.068 there), which is its own problem: it costs the dataset the headroom the
# per-dataset margin test needs. A non-leaking id column cost nothing measurable (+-0.007).
# SVC-rbf stood in for a global kernel; the fidelity kernels were not themselves tested.
ID_COLUMNS = {
    ("pmlb", "analcatdata_bankruptcy"): ("Company",),
    ("pmlb", "analcatdata_japansolvent"): ("Firm",),
    ("pmlb", "backache"): ("id",),
    ("pmlb", "biomed"): ("Hospital_identification_number_for_blood_sample",),
    ("pmlb", "clean1"): ("molecule_name", "conformation_name"),
    ("pmlb", "prnn_crabs"): ("index",),
}

# Name tokens that mark a column as a probable identifier in suspected_id_columns. Matched
# against whole tokens of the lower-cased name (split on non-alphanumerics and camelCase),
# so "id" catches "patient_id" and "ID" but not "idle" or "width". A name alone is weak
# evidence -- pmlb has "Age_of_patient", "LIVER FIRM", "Refractive Index" -- so a hinted
# column is flagged only if it is also integer-coded (or non-numeric) with at least
# ID_HINT_UNIQUE_FRACTION distinct values.
ID_NAME_TOKENS = {
    "id", "ids", "identifier", "name", "company", "firm", "patient", "subject",
    "sample", "index", "key", "uuid", "accession",
}
ID_HINT_UNIQUE_FRACTION = 0.25

# A non-numeric column with at least this fraction of distinct values is near-unique --
# a free-text label per row, which one-hot encoding would turn into ~n dummy columns.
NEAR_UNIQUE_FRACTION = 0.9

# Datasets whose labels are exact Boolean concepts or engineered epistatic interactions.
# PCA at n_components=3 provably destroys high-order interactions, so pooling these into
# one headline mean is a confound -- tag them so results can be split by family later.
BOOLEAN_CONCEPT = {
    "parity5", "parity5+5", "mux6", "xd6", "threeOf9", "monk1", "monk2", "monk3",
}


def family_of(stem: str) -> str:
    if stem in BOOLEAN_CONCEPT:
        return "boolean_concept"
    if stem.startswith("GAMETES_"):
        return "epistasis"
    return "real_tabular"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(frame: pd.DataFrame) -> str:
    """Shape plus a row-order-independent digest of the numeric block.

    Sorting each column independently makes this invariant to row permutation, so two
    copies of one dataset that differ only in row order still collide. It is a
    near-duplicate *signal* for human review, not a proof of identity.
    """
    values = np.sort(frame.to_numpy(dtype=float), axis=0)
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode())
    digest.update(np.ascontiguousarray(values.round(10)).tobytes())
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    """Read comma- or tab-separated data, matching QProfiler's tolerance for both."""
    frame = pd.read_csv(path)
    if frame.shape[1] == 1:
        frame = pd.read_csv(path, sep="\t")
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: parsed {frame.shape[1]} column(s); need features + label")
    return frame


def _name_tokens(name: str) -> set[str]:
    """Lower-cased word tokens of a column name: ``patientID_2`` -> {patient, id, 2}."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return {token for token in re.split(r"[^0-9a-zA-Z]+", spaced.lower()) if token}


def suspected_id_columns(features: pd.DataFrame) -> list[str]:
    """Columns that look like row identifiers.

    A column is flagged when any of these holds:

    * it is numeric and a permutation of 0..n-1 or 1..n;
    * it is non-numeric and near-unique (at least NEAR_UNIQUE_FRACTION of the rows
      distinct) -- one-hot encoding would turn it into ~n dummy columns;
    * a token of its name is in ID_NAME_TOKENS (``id``, ``name``, ``company``, ...) and
      it is integer-coded or non-numeric with at least ID_HINT_UNIQUE_FRACTION of the
      rows distinct.

    Reported, never dropped: only the columns listed in ID_COLUMNS are removed. An ID
    column can predict the label outright when the file happens to be sorted by class,
    but a heuristic removing columns silently would also make these datasets
    incomparable with previously published runs. That trade-off is the user's -- adding
    a confirmed column to ID_COLUMNS is how it is made.

    Args:
        features: Feature block (label excluded), before one-hot encoding.

    Returns:
        Flagged column names, in column order.
    """
    found = []
    n = len(features)
    for name in features.columns:
        column = features[name]
        distinct = column.nunique(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(column)
        if numeric:
            values = column.to_numpy(dtype=float)
            integer_coded = bool(np.all(np.isfinite(values)) and np.all(values == np.floor(values)))
        else:
            integer_coded = False
        name_hinted = (bool(_name_tokens(name) & ID_NAME_TOKENS)
                       and (integer_coded or not numeric)
                       and distinct >= ID_HINT_UNIQUE_FRACTION * n)
        if name_hinted:
            found.append(str(name))
        elif not numeric:
            if n and distinct >= NEAR_UNIQUE_FRACTION * n:
                found.append(str(name))
        elif integer_coded:
            as_int = np.sort(values.astype(np.int64))
            if np.array_equal(as_int, np.arange(n)) or np.array_equal(as_int, np.arange(1, n + 1)):
                found.append(str(name))
    return found


def curate_one(path: Path, source: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """Apply the fixed pipeline to one raw file and return (curated frame, metadata)."""
    stem = path.stem
    dataset_id = f"{source}__{stem}"
    raw = read_table(path)
    steps: list[str] = []

    n_raw, p_raw = raw.shape
    label_col = str(raw.columns[-1])

    # Mapped identifier columns go first of all: before NaN removal, so a NaN in an ID
    # cannot cost a row, and before one-hot, where a string ID would become ~n dummies.
    id_dropped = list(ID_COLUMNS.get((source, stem), ()))
    if id_dropped:
        missing = [c for c in id_dropped if c not in raw.columns[:-1]]
        if missing:
            raise ValueError(
                f"{dataset_id}: ID_COLUMNS lists {missing}, absent from the raw feature "
                f"columns {list(raw.columns[:-1])[:10]}...; the map is stale -- fix ID_COLUMNS"
            )
        raw = raw.drop(columns=id_dropped)
        steps.append(f"dropped {len(id_dropped)} identifier column(s) listed in ID_COLUMNS: {id_dropped}")

    # NaN rows go first, before one-hot. get_dummies turns a NaN category into an
    # all-zero row rather than a NaN, so encoding first would hide exactly the rows we
    # mean to drop (credit-approval carries NaN in its categorical columns).
    before = len(raw)
    raw = raw.dropna(axis=0, how="any")
    nan_rows_dropped = before - len(raw)
    if nan_rows_dropped:
        steps.append(f"dropped {nan_rows_dropped} rows containing NaN")
    if raw.empty:
        raise ValueError(f"{dataset_id}: no rows survive NaN removal")

    features = raw.iloc[:, :-1]
    labels = raw.iloc[:, -1]

    id_columns = suspected_id_columns(features)

    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        features = pd.get_dummies(features, columns=non_numeric, drop_first=False, dtype=float)
        steps.append(f"one-hot encoded {len(non_numeric)} non-numeric column(s): {non_numeric}")

    constant = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
    if constant:
        features = features.drop(columns=constant)
        steps.append(f"dropped {len(constant)} zero-variance column(s): {constant}")
    if features.shape[1] == 0:
        raise ValueError(f"{dataset_id}: no feature columns survive curation")

    classes = sorted(pd.unique(labels))
    if len(classes) != 2:
        raise ValueError(
            f"{dataset_id}: {len(classes)} classes {classes[:5]}; QProfiler requires binary "
            "labels (model_evaluation.roc_auc_score is binary-only)"
        )
    mapping = {classes[0]: 0, classes[1]: 1}
    labels = labels.map(mapping).astype(int)
    steps.append(f"mapped labels to 0/1 by sorted unique order: {{{classes[0]}: 0, {classes[1]}: 1}}")

    features = features.astype(float)
    curated = pd.concat([features.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    curated.columns = [*features.columns, "label"]

    # Shuffle once, here, and never again. This freezes row order forever: it is what
    # makes the inner CV's StratifiedKFold(shuffle=False) safe once the outer split comes
    # from a manifest instead of from train_test_split, which used to shuffle as a side
    # effect. The manifest's sha256 then pins this exact ordering.
    permutation = np.random.default_rng(seed).permutation(len(curated))
    curated = curated.iloc[permutation].reset_index(drop=True)
    steps.append(f"shuffled rows once with numpy default_rng(seed={seed})")

    counts = curated["label"].value_counts()
    minority_n = int(counts.min())
    n, p = len(curated), curated.shape[1] - 1

    meta = {
        "dataset_id": dataset_id,
        "source": source,
        "original_file": str(path),
        "original_shape": [int(n_raw), int(p_raw)],
        "n": int(n),
        "p": int(p),
        "label_col_original": label_col,
        "class_counts": {int(k): int(v) for k, v in counts.sort_index().items()},
        "minority_n": minority_n,
        "minority_fraction": round(minority_n / n, 6),
        "n_components_max": int(p),
        "group_col": None,
        "family": family_of(stem),
        "dropped_id_columns": id_dropped,
        "suspected_id_columns": id_columns,
        "nan_rows_dropped": int(nan_rows_dropped),
        "one_hot_columns": non_numeric,
        "one_hot_drop_first": False,
        "dropped_constant_columns": constant,
        "curate_seed": int(seed),
        "generator_version": GENERATOR_VERSION,
        "steps": steps,
    }
    return curated, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=Path("/dccstor/cgq4hls/Q/qbc_data"))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent / "datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prune", action="store_true",
                        help="remove dataset directories an earlier run wrote that this "
                             "one did not -- what adding an EXCLUDE entry leaves behind")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()

    # The source tree belongs to another user and is read-only to us by intent. Writing
    # under it is the create_splits.ipynb bug -- os.path.join('train/', <abs path>)
    # returns the absolute path, so the notebook wrote into the source tree and the
    # intended output directories stayed empty.
    if out_root == source_root or source_root in out_root.parents:
        raise SystemExit(f"refusing to write inside the source tree: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows, fingerprints, failures, seen = [], {}, [], set()
    for source, directory in sorted(SOURCES.items()):
        for path in sorted((source_root / directory).glob("*.csv")):
            seen.add((source, path.stem))
            if (source, path.stem) in EXCLUDE:
                print(f"  skip     {source}__{path.stem} (excluded)")
                continue
            try:
                curated, meta = curate_one(path, source, args.seed)
            except Exception as error:  # noqa: BLE001 - report and continue over 86 files
                failures.append({"dataset": f"{source}__{path.stem}", "error": str(error)})
                print(f"  FAIL     {source}__{path.stem}: {error}")
                continue

            dataset_id = meta["dataset_id"]
            dataset_dir = out_root / dataset_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dataset_dir / f"{dataset_id}.csv"
            curated.to_csv(csv_path, index=False)
            meta["sha256"] = sha256_of(csv_path)
            (dataset_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

            fingerprints.setdefault(content_fingerprint(curated), []).append(dataset_id)
            rows.append(meta)
            flag = " [SUSPECTED ID COLUMN]" if meta["suspected_id_columns"] else ""
            print(f"  ok       {dataset_id:<34} n={meta['n']:<5} p={meta['p']:<5} "
                  f"minority={meta['minority_n']:<4} {meta['family']}{flag}")

    # Separation invariant: QProfiler globs folder_path for *.csv, so one stray per-fold
    # file here would be re-split and reported as its own dataset. Fail rather than let
    # that reach the generator.
    strays = [str(p) for p in out_root.rglob("*.csv")
              if "_train" in p.name or "_test" in p.name]
    if strays:
        raise SystemExit(f"train/test-shaped files under {out_root}: {strays}")
    for dataset_dir in sorted(out_root.iterdir()):
        if dataset_dir.is_dir():
            found = sorted(p.name for p in dataset_dir.glob("*.csv"))
            if len(found) != 1:
                raise SystemExit(f"{dataset_dir} holds {len(found)} csv files ({found}); need exactly 1")

    # Directories this run did not write, i.e. datasets curated by an EARLIER run and now
    # excluded or renamed. Nothing here overwrites or removes them, and every consumer
    # downstream discovers datasets by globbing for a meta.yaml -- the generator's
    # --datasets all is literally "directories holding a meta.yaml". So adding an entry to
    # EXCLUDE and re-running leaves the dropped dataset in the corpus, silently, and the
    # exclusion has no effect at all where it matters.
    curated_ids = {meta["dataset_id"] for meta in rows}
    stale = sorted(d for d in out_root.iterdir()
                   if d.is_dir() and (d / "meta.yaml").is_file() and d.name not in curated_ids)
    if stale:
        if args.prune:
            for dataset_dir in stale:
                shutil.rmtree(dataset_dir)
            print(f"\npruned {len(stale)} stale dataset directory(ies) this run did not write:")
            for dataset_dir in stale:
                print(f"    {dataset_dir.name}")
        else:
            raise SystemExit(
                f"\n{len(stale)} dataset directory(ies) under {out_root} were written by an "
                f"earlier run and not by this one:\n"
                + "".join(f"    {d.name}\n" for d in stale)
                + "They are still a part of the corpus for anything that globs meta.yaml "
                  "(the generator's --datasets all), so leaving them would undo the "
                  "exclusion.\nRe-run with --prune to remove them, or delete them by hand."
            )

    inventory = pd.DataFrame(rows)
    inventory_columns = ["dataset_id", "source", "n", "p", "minority_n", "minority_fraction",
                         "family", "nan_rows_dropped", "sha256"]
    inventory[inventory_columns].to_csv(out_root.parent / "inventory.csv", index=False)

    duplicates = []
    for fingerprint, ids in sorted(fingerprints.items()):
        if len(ids) > 1:
            for other in ids[1:]:
                duplicates.append({"kind": "identical_content", "dataset_a": ids[0],
                                   "dataset_b": other, "detail": f"fingerprint {fingerprint[:16]}"})
    stems: dict[str, list[str]] = {}
    for meta in rows:
        stems.setdefault(meta["dataset_id"].split("__", 1)[1], []).append(meta["dataset_id"])
    for stem, ids in sorted(stems.items()):
        if len(ids) > 1:
            for other in ids[1:]:
                duplicates.append({"kind": "stem_collision", "dataset_a": ids[0],
                                   "dataset_b": other, "detail": f"shared stem {stem!r}"})
    pd.DataFrame(duplicates, columns=["kind", "dataset_a", "dataset_b", "detail"]).to_csv(
        out_root.parent / "duplicates.csv", index=False)

    # A key naming a file that is not there is as stale as a column that is not there.
    for key in sorted(set(ID_COLUMNS) - seen):
        failures.append({"dataset": "__".join(key),
                         "error": "listed in ID_COLUMNS but no such raw file; the map is stale"})

    id_dropped = [m["dataset_id"] for m in rows if m["dropped_id_columns"]]
    id_flagged = [m for m in rows if m["suspected_id_columns"]]
    print(f"\ncurated {len(rows)} datasets into {out_root}")
    print(f"  duplicates.csv : {len(duplicates)} pair(s) to review")
    print(f"  id dropped     : {len(id_dropped)} dataset(s) via ID_COLUMNS: {id_dropped}")
    if id_flagged:
        # Kept, so loud: each of these may be leaking the label into every learner.
        print(f"  !! SUSPECTED ID COLUMNS KEPT in {len(id_flagged)} dataset(s) -- review, and add "
              "confirmed ones to ID_COLUMNS:")
        for meta in id_flagged:
            print(f"       {meta['dataset_id']}: {meta['suspected_id_columns']}")
    if failures:
        print(f"  FAILURES       : {len(failures)}")
        for failure in failures:
            print(f"    {failure['dataset']}: {failure['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
