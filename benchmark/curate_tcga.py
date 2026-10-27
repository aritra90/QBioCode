#!/usr/bin/env python3
"""Curate the TCGA RNA-seq cohorts into the corpus layout, one dataset per cancer.

    python benchmark/curate_tcga.py --source-root /dccstor/cgq4hls/Q/TCGA_data/processed \\
        --out-root $BENCH/data/datasets_tcga

Writes ``<out-root>/tcga__<CANCER>/tcga__<CANCER>.csv`` plus a ``meta.yaml`` per cancer,
in exactly the shape ``benchmark/make_splits.py`` and
``experiments/pilot10/generate_pilot_configs.py`` already consume -- features first,
label last, one CSV per directory.

Separate from ``curate.py`` rather than a fourth entry in its ``SOURCES``: that script's
work is one-hot encoding, identifier-column detection and a cross-source dedup list, none
of which applies here. A TCGA matrix is numeric throughout, has no identifier columns to
find, and reaches us from one source, so sharing the code would mean sharing the
conditionals rather than the behaviour.

What this does, and the choices it makes
----------------------------------------
* **Transposes.** ``matrices/*.parquet`` is genes x samples (its index holds
  ``ENSG...`` ids); every consumer here wants samples x genes.
* **Joins the labels** from ``splits/split_assignments.csv`` on ``sample_barcode``.
  Samples the matrix does not carry, and samples the label table does not list, are
  dropped and REPORTED -- a silent inner join would change *n* per cancer without
  saying so.
* **Keeps one label.** ``--label PFI_label`` by default. OS_label has missing values in
  every cohort and leaves PRAD with a minority class of 2, which ``make_splits.py``
  rejects outright (it needs ``k <= minority``); run it as a second, separate corpus if
  you want it, and expect fewer cohorts.
* **Selects no genes.** All ~20k protein-coding genes are written, and the pipeline's
  own ``feature_selection`` stage narrows them inside each training fold
  (:mod:`qbiocode.apps.qprofiler.feature_selection`). Doing it here instead would rank
  genes over every row of the file, including the rows each fold is scored on. The
  ``splits/rna/`` selections that ship with the data are not used for the same reason
  the splits themselves are not: see below.
* **Sets ``group_col: null``**, having checked that each sample barcode maps to a
  distinct patient. The upstream processing averaged multi-sample patients, so one row
  is one patient; the check is here because that is an assumption about someone else's
  pipeline and this is the last place it can be caught cheaply.

Why the shipped splits are not used
-----------------------------------
``splits/`` holds 10 independent 80/20 train/test draws per cancer. The corpus protocol
is repeated stratified K-fold: ``make_splits.py`` asserts that each repeat's test folds
*partition* the rows, so every row is a test row exactly once per repeat. Ten overlapping
draws cannot satisfy that, and the paired corpus-level tests downstream depend on it. The
shipped splits also carry no validation rows, which manifest mode needs for tuning.

Running ``make_splits.py`` over these CSVs instead gives the same 3x5 protocol as the
existing 171-dataset corpus, which is the precondition for collating the two together.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

#: ``rna_log2tpm_protein_coding_rescaled_<CANCER>.parquet`` -> ``<CANCER>``.
MATRIX_PATTERN = re.compile(r"_([A-Z0-9]+)\.parquet$")

#: The label columns the split table carries. PFI is the default: it is complete, while
#: OS_label is missing for 5-45 samples per cohort and leaves PRAD with 2 positives.
LABEL_COLUMNS = ("PFI_label", "OS_label")

#: ``meta_regression.dataset_family`` resolves a cluster per dataset, and 19 cohorts
#: assayed on one platform are not 19 independent datasets. Recorded here so the cluster
#: map generated later (``review_duplicates.py --emit-cluster-map``) can group them, and
#: so the choice is visible in the corpus rather than implicit in an analysis script.
FAMILY = "tcga_rna"

#: What to do when one patient contributes several samples. ``fail`` is the default
#: because the alternative is a decision about someone else's data made silently.
DUPLICATE_POLICIES = ("fail", "average")


def sample_type(barcode: str) -> str | None:
    """The TCGA sample-type code of a barcode: ``'01'`` from ``'TCGA-2F-A9KO-01A'``.

    A barcode's fourth field is the sample type (two digits) followed by the vial (a
    letter): ``01`` primary solid tumour, ``02`` recurrent, ``06`` metastatic, ``11``
    solid tissue normal. The distinction is what decides whether two samples of one
    patient may be averaged -- two vials of one tumour are replicates of the same
    tissue, a tumour and a normal are not the same measurement at all.

    Returns:
        str | None: the two-digit code, or None if the barcode does not carry one.
    """
    parts = str(barcode).split("-")
    if len(parts) < 4 or len(parts[3]) < 2 or not parts[3][:2].isdigit():
        return None
    return parts[3][:2]


def collapse_patients(X, y, patients, policy):
    """One row per patient, averaging the samples of each.

    Only ever averages samples that share a :func:`sample_type`. A patient with two
    sample types is a tumour/normal or tumour/metastasis pair, and the mean of those is
    not a measurement of anything -- so it raises rather than quietly producing one.

    The labels need no reconciliation: PFI and OS are patient-level outcomes, so every
    sample of a patient already carries the same value. That is asserted rather than
    assumed, because it is the premise that makes averaging meaningful at all.

    Args:
        X (pandas.DataFrame): samples x genes, indexed by sample barcode.
        y (pandas.Series): the label per sample barcode.
        patients (pandas.Series): the patient barcode per sample barcode.
        policy (str): one of :data:`DUPLICATE_POLICIES`.

    Returns:
        (DataFrame, Series, dict): the collapsed matrix and labels, indexed by patient
        barcode, and a report of what was done.

    Raises:
        ValueError: under ``policy='fail'`` when any patient has several samples; under
            either policy when one patient's samples disagree on the label or span more
            than one sample type.
    """
    counts = patients.value_counts()
    multi = counts[counts > 1]
    report = {
        "samples_in": int(len(X)),
        "patients": int(counts.size),
        "patients_collapsed": int(multi.size),
        "samples_merged_away": int(len(X) - counts.size),
        "duplicate_policy": policy,
        "sample_types": {},
    }
    types = patients.index.map(sample_type)
    report["sample_types"] = {
        str(code): int(n) for code, n in
        pd.Series(types).value_counts().sort_index().items()
    }

    if multi.empty:
        report["duplicate_policy"] = "none needed"
        return X, y, report

    if policy == "fail":
        raise ValueError(
            f"{multi.size} patient(s) contribute several samples "
            f"({int(len(X) - counts.size)} extra rows), so two rows of one patient can "
            f"land on opposite sides of a split. PFI and OS are patient-level outcomes, "
            f"so those rows carry the SAME label and a model that recognises the patient "
            f"recovers it -- which inflates every score. Pass "
            f"--on-duplicate-patient average to average each patient's samples, or "
            f"resolve it upstream."
        )

    # Label agreement, and tissue-type agreement, before anything is averaged.
    by_patient = pd.DataFrame({"patient": patients.to_numpy(),
                               "label": y.to_numpy(),
                               "type": list(types)}, index=X.index)
    disagree = by_patient.groupby("patient")["label"].nunique()
    if (disagree > 1).any():
        bad = list(disagree[disagree > 1].index[:5])
        raise ValueError(
            f"patient(s) {bad} have samples with DIFFERENT labels, which contradicts "
            f"PFI/OS being patient-level. Averaging would invent a label; resolve this "
            f"in the label table first."
        )
    mixed = by_patient.groupby("patient")["type"].nunique()
    if (mixed > 1).any():
        offenders = list(mixed[mixed > 1].index[:5])
        detail = {p: sorted(by_patient.loc[by_patient.patient == p, "type"].unique())
                  for p in offenders}
        raise ValueError(
            f"patient(s) with MORE THAN ONE sample type: {detail}. TCGA codes 01, 02, "
            f"06 and 11 are primary tumour, recurrence, metastasis and normal tissue -- "
            f"different measurements, whose mean is not a measurement. Averaging is "
            f"refused here; drop the unwanted sample types upstream and re-run."
        )

    grouped = X.groupby(patients.to_numpy(), sort=True)
    collapsed = grouped.mean()
    labels = y.groupby(patients.to_numpy(), sort=True).first()
    collapsed.index.name = "patient_barcode"
    return collapsed, labels.loc[collapsed.index], report


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(path: Path) -> pd.DataFrame:
    """One cancer's expression matrix as samples x genes.

    The file is genes x samples -- its index holds ``ENSG...`` ids and its columns hold
    ``TCGA-..`` barcodes -- so this transposes. The orientation is CHECKED rather than
    assumed: a future file written the other way round would otherwise be transposed
    into nonsense, and the error naming both axes is cheaper than discovering it as a
    20k-row dataset with 200 features.
    """
    frame = pd.read_parquet(path)
    index_is_genes = frame.index.astype(str).str.startswith("ENSG").mean() > 0.9
    cols_are_samples = frame.columns.astype(str).str.startswith("TCGA").mean() > 0.9
    if not (index_is_genes and cols_are_samples):
        raise ValueError(
            f"{path.name}: expected genes (ENSG...) on the index and samples (TCGA-...) "
            f"on the columns; got index {list(frame.index[:2])} and columns "
            f"{list(frame.columns[:2])}. Orientation is checked rather than assumed "
            f"because transposing the wrong way round yields a plausible-looking "
            f"matrix with the roles of rows and columns swapped."
        )
    return frame.T


def curate_one(matrix_path: Path, labels: pd.DataFrame, label_col: str,
               policy: str = "fail") -> tuple[pd.DataFrame, dict]:
    """``(curated, meta)`` for one cancer: features first, label last.

    Args:
        matrix_path: the cohort's parquet.
        labels: the whole ``split_assignments.csv``, deduplicated to one row per sample.
        label_col: which label column to use.

    Raises:
        ValueError: on a non-binary label, a cohort left with one class, or an
            orientation the matrix check rejects.
    """
    cancer = MATRIX_PATTERN.search(matrix_path.name).group(1)
    dataset_id = f"tcga__{cancer}"
    expression = read_matrix(matrix_path)
    n_genes = expression.shape[1]

    mine = labels[labels["cancer_type"] == cancer]
    labelled = mine.set_index("sample_barcode")[label_col]

    in_both = expression.index.intersection(labelled.index)
    expr_only = sorted(set(expression.index) - set(labelled.index))
    label_only = sorted(set(labelled.index) - set(expression.index))

    y = labelled.loc[in_both]
    # NaN labels are dropped here rather than imputed: a missing outcome is not a class,
    # and the alternative (treating it as negative) would invent censored events.
    missing_label = int(y.isna().sum())
    keep = y.dropna().index
    X = expression.loc[keep]
    y = y.loc[keep].astype(int)

    classes = sorted(pd.unique(y))
    if len(classes) != 2:
        raise ValueError(
            f"{dataset_id}: {label_col} has {len(classes)} class(es) {classes} after "
            f"dropping missing values; the evaluation is binary "
            f"(model_evaluation.roc_auc_score)"
        )

    # One row per PATIENT before anything else looks at the matrix. PFI and OS are
    # patient-level outcomes, so two samples of one patient carry the same label; left
    # as separate rows they can straddle a split, and a model that recognises the
    # patient from its expression recovers the label. Averaging replicates is what the
    # upstream processing was documented as doing and demonstrably did not.
    patients = mine.set_index("sample_barcode").loc[keep, "patient_barcode"]
    X, y, dup_report = collapse_patients(X, y, patients, policy)

    # Genes that are constant across the cohort carry no information and would occupy a
    # qubit or a PCA component for nothing. Dropped AFTER collapsing, because averaging
    # can make a gene constant that was not.
    constant = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
    if constant:
        X = X.drop(columns=constant)

    curated = X.reset_index(drop=True)
    curated["label"] = y.to_numpy()

    counts = y.value_counts()
    minority_n = int(counts.min())
    meta = {
        "dataset_id": dataset_id,
        "source": "tcga",
        "cancer_type": cancer,
        "original_file": str(matrix_path),
        "original_shape": [int(expression.shape[0]), int(n_genes)],
        "n": int(len(curated)),
        "p": int(X.shape[1]),
        "label_col_original": label_col,
        "class_counts": {int(k): int(v) for k, v in counts.sort_index().items()},
        "minority_n": minority_n,
        "minority_fraction": round(minority_n / len(curated), 6),
        "n_components_max": int(X.shape[1]),
        # None, and now TRUE BY CONSTRUCTION rather than by assumption: collapse_patients
        # has made one row one patient. group_col is not the alternative -- make_splits.py
        # reads it as frame[group_col], i.e. a COLUMN OF THE CSV, where every column but
        # the last is a feature. A patient barcode there would be handed to the models as
        # a near-unique identifier, which is a worse leak than the one it fixes.
        "group_col": None,
        "distinct_patients": int(dup_report["patients"]),
        "samples_before_collapse": int(dup_report["samples_in"]),
        "patients_collapsed": int(dup_report["patients_collapsed"]),
        "samples_merged_away": int(dup_report["samples_merged_away"]),
        "duplicate_policy": dup_report["duplicate_policy"],
        "sample_types": dup_report["sample_types"],
        "family": FAMILY,
        "dropped_constant_columns": len(constant),
        "samples_without_expression": len(label_only),
        "samples_without_label": len(expr_only),
        "samples_with_missing_label": missing_label,
    }
    # collapse_patients guarantees this; asserted because it is the property every
    # split downstream depends on and it costs nothing to check.
    assert meta["n"] == meta["distinct_patients"], (dataset_id, meta["n"],
                                                    meta["distinct_patients"])
    return curated, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path,
                        default=Path("/dccstor/cgq4hls/Q/TCGA_data/processed"))
    parser.add_argument("--out-root", type=Path, required=True,
                        help="a NEW directory, e.g. $BENCH/data/datasets_tcga -- keep it "
                             "separate from the existing corpus so the two can be "
                             "curated, split and run independently")
    parser.add_argument("--label", choices=LABEL_COLUMNS, default="PFI_label")
    parser.add_argument("--on-duplicate-patient", choices=DUPLICATE_POLICIES,
                        default="fail",
                        help="what to do when one patient contributes several samples. "
                             "'fail' (default) refuses, naming the count; 'average' "
                             "takes the mean of each patient's samples, which is valid "
                             "only for replicates of ONE tissue and is refused when a "
                             "patient's samples span several TCGA sample types")
    parser.add_argument("--min-minority", type=int, default=5,
                        help="skip a cohort whose minority class is smaller than this; "
                             "make_splits.py needs k <= minority and the default k is 5")
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    if out_root == source_root or source_root in out_root.parents:
        raise SystemExit(f"refusing to write inside the source tree: {out_root}")

    split_table = source_root / "splits" / "split_assignments.csv"
    if not split_table.is_file():
        raise SystemExit(f"no label table at {split_table}")
    assignments = pd.read_csv(split_table)
    for column in ("cancer_type", "sample_barcode", "patient_barcode", args.label):
        if column not in assignments.columns:
            raise SystemExit(f"{split_table}: no {column!r} column "
                             f"(has {list(assignments.columns)})")
    # The table has one row per (cancer, split, sample) -- 10 splits means 10 identical
    # label rows per sample. One row per sample is what the join needs; taking the first
    # is safe only if they agree, which is checked rather than assumed.
    per_sample = assignments.drop_duplicates(subset=["cancer_type", "sample_barcode"])
    disagreeing = (assignments.groupby(["cancer_type", "sample_barcode"])[args.label]
                   .nunique(dropna=False))
    if (disagreeing > 1).any():
        bad = list(disagreeing[disagreeing > 1].index[:5])
        raise SystemExit(f"{split_table}: {args.label} differs between split rows of the "
                         f"same sample, e.g. {bad}")

    matrices = sorted((source_root / "matrices").glob("*.parquet"))
    if not matrices:
        raise SystemExit(f"no *.parquet under {source_root / 'matrices'}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows, skipped, failures = [], [], []
    for path in matrices:
        match = MATRIX_PATTERN.search(path.name)
        if not match:
            failures.append((path.name, "file name does not end _<CANCER>.parquet"))
            continue
        try:
            curated, meta = curate_one(path, per_sample, args.label,
                                       args.on_duplicate_patient)
        except Exception as error:  # noqa: BLE001 - report every cohort, fail at the end
            failures.append((path.name, str(error)))
            print(f"  FAIL     {path.name}: {error}")
            continue

        if meta["minority_n"] < args.min_minority:
            skipped.append((meta["dataset_id"], meta["minority_n"]))
            print(f"  skip     {meta['dataset_id']:<14} minority={meta['minority_n']} "
                  f"< --min-minority={args.min_minority}: make_splits.py needs "
                  f"k <= minority")
            continue

        dataset_dir = out_root / meta["dataset_id"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dataset_dir / f"{meta['dataset_id']}.csv"
        curated.to_csv(csv_path, index=False)
        meta["sha256"] = sha256_of(csv_path)
        (dataset_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        rows.append(meta)
        note = ""
        if meta["samples_without_expression"] or meta["samples_with_missing_label"]:
            note = (f"  [-{meta['samples_without_expression']} no expression, "
                    f"-{meta['samples_with_missing_label']} no label]")
        print(f"  ok       {meta['dataset_id']:<14} n={meta['n']:<5} p={meta['p']:<6} "
              f"minority={meta['minority_n']:<4}{note}")

    # One CSV per directory: QProfiler globs folder_path for *.csv, so a second file
    # there would be re-split and reported as its own dataset.
    for dataset_dir in sorted(out_root.iterdir()):
        if dataset_dir.is_dir():
            found = sorted(p.name for p in dataset_dir.glob("*.csv"))
            if len(found) != 1:
                raise SystemExit(f"{dataset_dir} holds {len(found)} csv files ({found})")

    if rows:
        inventory = pd.DataFrame(rows)
        inventory.to_csv(out_root.parent / "inventory_tcga.csv", index=False)
        print(f"\ncurated {len(rows)} cohort(s) to {out_root}")
        print(f"  label           : {args.label}")
        print(f"  features        : {inventory.p.min()}-{inventory.p.max()} genes "
              f"(no selection here; the pipeline's feature_selection stage narrows "
              f"these inside each training fold)")
        print(f"  n               : {inventory.n.min()}-{inventory.n.max()} samples")
        print(f"  minority        : {inventory.minority_n.min()}-"
              f"{inventory.minority_n.max()}")
        merged = int(inventory.samples_merged_away.sum())
        if merged:
            print(f"  patients        : {merged} sample(s) merged into "
                  f"{int(inventory.patients_collapsed.sum())} patient(s) by "
                  f"--on-duplicate-patient average; one row is now one patient")
        print(f"  dropped samples : {int(inventory.samples_without_expression.sum())} "
              f"labelled but not in a matrix, "
              f"{int(inventory.samples_with_missing_label.sum())} with a missing label, "
              f"{int(inventory.samples_without_label.sum())} measured but unlabelled")
        print(f"  inventory       : {out_root.parent / 'inventory_tcga.csv'}")
        print(f"\n!! These {len(rows)} cohorts share one assay platform and are NOT "
              f"independent datasets.\n"
              f"!! Every meta.yaml carries family: {FAMILY!r}; group them into one "
              f"cluster before any\n"
              f"!! corpus-level test, or G is inflated by {len(rows) - 1} and every "
              f"cluster-robust interval\n"
              f"!! narrows spuriously.")
        print(f"\nnext: python benchmark/make_splits.py --datasets {out_root} "
              f"--out <BENCH>/data/splits/tcga1")
    if skipped:
        print(f"\nskipped {len(skipped)} cohort(s) for too small a minority class: "
              f"{skipped}")
    if failures:
        print(f"\n!! {len(failures)} cohort(s) FAILED:")
        for name, error in failures:
            print(f"     {name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
