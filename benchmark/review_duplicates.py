#!/usr/bin/env python3
"""Decide what to do about the duplicate pairs ``curate.py`` reports.

``curate.py`` ends with a line like::

    duplicates.csv : 4 pair(s) to review

and then leaves the review to a person, because the two possible answers are not
interchangeable and neither is mechanical:

* **Drop one copy.** Two files holding the same rows are one dataset, and keeping both
  counts one observation twice in every corpus-level test.
* **Keep both, in one cluster.** Two *encodings* of one source (``tic-tac-toe`` as 27
  one-hot columns and ``tic_tac_toe`` as 9 ordinal ones) are arguably two datasets, but
  they are certainly not independent, so they must share a cluster.

The cluster is the unit of independence throughout the analysis:
:func:`qbiocode.utils.meta_regression.dataset_family` resolves it, the wild cluster
bootstrap in that module clusters on it, and ``benchmark/holdout.py --cluster-map`` keeps a
cluster on one side of the hold-out. Counting copies separately inflates the number of
clusters ``G`` and so every cluster-robust test.

This script does not decide for you. It reports, against a cluster map:

1. which corpus datasets the map does **not** list -- the copies an earlier decision
   removed from it, plus anything genuinely new;
2. which clusters hold more than one corpus dataset -- the groups already recognised as
   non-independent;
3. each ``duplicates.csv`` pair, and whether the map already puts it in one cluster;
4. a ready-to-paste ``EXCLUDE`` snippet for the datasets you choose to drop.

With ``--emit-cluster-map`` it writes a frozen ``dataset_id,cluster`` CSV covering exactly
the corpus on disk, which is what ``holdout.py --cluster-map`` wants.

    python benchmark/review_duplicates.py --datasets $BENCH/data/datasets
    python benchmark/review_duplicates.py --datasets $BENCH/data/datasets \\
        --emit-cluster-map $BENCH/data/clusters.csv

Run it before ``make_splits.py``. The split manifests pin each CSV's sha256, so dropping a
dataset afterwards leaves a manifest for a file the corpus no longer has, and re-curating
afterwards invalidates every manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

#: The draft shipped with the repository. 91 datasets, and its ``stem`` column is not
#: unique (``breast_cancer`` is both the libsvm Wisconsin set and the PMLB Ljubljana one),
#: so it is always keyed by ``dataset_id``.
DEFAULT_MAP = Path(__file__).resolve().parent.parent / "experiments" / "cluster_map_draft.csv"
DEFAULT_KEY = "dataset_id"
#: ``meta_regression.CLUSTER_COLUMN``. The draft also carries a looser ``cluster_lib``.
DEFAULT_COLUMN = "cluster_cons"


def corpus(datasets: Path) -> pd.DataFrame:
    """One row per curated dataset: ``dataset_id``, ``source``, ``stem``, ``n``, ``p``.

    Read from each ``meta.yaml`` rather than ``inventory.csv``, so the answer describes
    what is on disk now even if the inventory is from an earlier run.
    """
    rows = []
    for meta_path in sorted(datasets.glob("*/meta.yaml")):
        meta = yaml.safe_load(meta_path.read_text())
        dataset_id = meta.get("dataset_id", meta_path.parent.name)
        source, _, stem = dataset_id.partition("__")
        rows.append({"dataset_id": dataset_id, "source": source, "stem": stem,
                     "n": meta.get("n"), "p": meta.get("p"),
                     "family": meta.get("family", "")})
    if not rows:
        raise SystemExit(f"no */meta.yaml under {datasets} -- run curate.py first")
    return pd.DataFrame(rows)


def cluster_of(datasets_frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    """Each dataset's cluster: the mapping where it lists one, else the fallback rules.

    The rules mirror :func:`qbiocode.utils.meta_regression.dataset_family` so a map that
    does not cover the whole corpus still yields the clusters the analysis will use --
    synthetic generators collapse to ``syn_<prefix>``, GAMETES variants to ``GAMETES``,
    and anything else is its own cluster.
    """
    from qbiocode.utils.meta_regression import SYNTHETIC_PREFIXES

    def one(row) -> str:
        if row.dataset_id in mapping:
            return mapping[row.dataset_id]
        stem = row.stem
        for prefix in SYNTHETIC_PREFIXES:
            if stem.startswith(prefix):
                return f"syn_{prefix.rstrip('_')}"
        if stem.startswith("GAMETES"):
            return "GAMETES"
        return stem

    return datasets_frame.apply(one, axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, required=True,
                        help="the curated tree: <id>/<id>.csv + meta.yaml")
    parser.add_argument("--duplicates", type=Path, default=None,
                        help="curate.py's duplicates.csv (default: beside --datasets)")
    parser.add_argument("--cluster-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--column", default=DEFAULT_COLUMN)
    parser.add_argument("--emit-cluster-map", type=Path, default=None, metavar="OUT",
                        help="write dataset_id,cluster for exactly the corpus on disk")
    args = parser.parse_args(argv)

    here = corpus(args.datasets)
    print(f"corpus: {len(here)} curated datasets under {args.datasets}")

    mapping: dict[str, str] = {}
    if args.cluster_map and args.cluster_map.is_file():
        table = pd.read_csv(args.cluster_map)
        for column in (args.key, args.column):
            if column not in table.columns:
                raise SystemExit(f"{args.cluster_map}: no {column!r} column "
                                 f"(has {list(table.columns)})")
        blank = table[table[args.column].isna()]
        if len(blank):
            raise SystemExit(f"{args.cluster_map}: blank {args.column} for "
                             f"{list(blank[args.key])}")
        mapping = dict(zip(table[args.key], table[args.column]))
        print(f"cluster map: {args.cluster_map} "
              f"({len(mapping)} datasets, {table[args.column].nunique()} clusters "
              f"in {args.column!r})")
    else:
        print(f"cluster map: none ({args.cluster_map} not found) -- using the fallback rules")

    here["cluster"] = cluster_of(here, mapping)

    # 1. Corpus datasets the map omits. Where an earlier dedup decision was recorded by
    #    leaving a copy out of the map, this is exactly that list.
    unlisted = here[~here.dataset_id.isin(mapping)]
    print(f"\n== {len(unlisted)} corpus dataset(s) NOT in the cluster map ==")
    if len(unlisted):
        print("   Each is either a copy an earlier decision dropped, or genuinely new.")
        for row in unlisted.itertuples():
            twins = here[(here.stem.str.replace("-", "_") == row.stem.replace("-", "_"))
                         & (here.dataset_id != row.dataset_id)]
            same_n = here[(here.n == row.n) & (here.dataset_id != row.dataset_id)]
            hint = ""
            if len(twins):
                hint = f"   same stem as {list(twins.dataset_id)}"
            elif len(same_n):
                hint = f"   same n as {list(same_n.dataset_id)}"
            print(f"     {row.dataset_id:<44} n={row.n:<6} p={row.p:<6}{hint}")

    # 2. Clusters holding several corpus datasets: the recognised non-independence.
    grouped = here.groupby("cluster").dataset_id.apply(list)
    multi = {c: ids for c, ids in grouped.items() if len(ids) > 1}
    print(f"\n== {len(multi)} cluster(s) holding more than one corpus dataset ==")
    for cluster, ids in sorted(multi.items()):
        print(f"     {cluster:<24} {len(ids):>2}  {ids}")
    print(f"\n   {len(here)} datasets -> {here.cluster.nunique()} clusters "
          f"(G for every cluster-robust test)")

    # 3. The pairs curate.py flagged, each resolved against the clusters above.
    dup_path = args.duplicates or args.datasets.parent / "duplicates.csv"
    if dup_path.is_file():
        dups = pd.read_csv(dup_path)
        by_id = dict(zip(here.dataset_id, here.cluster))
        print(f"\n== {len(dups)} pair(s) in {dup_path.name} ==")
        for row in dups.itertuples():
            a, b = row.dataset_a, row.dataset_b
            ca, cb = by_id.get(a), by_id.get(b)
            if a not in by_id or b not in by_id:
                verdict = "one side is no longer in the corpus -> already resolved"
            elif ca == cb:
                verdict = f"same cluster {ca!r} -> handled"
            else:
                verdict = f"DIFFERENT clusters {ca!r} vs {cb!r} -> decide"
            print(f"     {row.kind:<18} {a} | {b}\n        {verdict}")
    else:
        print(f"\n== no {dup_path} (run curate.py to produce it) ==")

    # 4. The snippet, for the copies you choose to drop.
    if len(unlisted):
        print("\n== to DROP a copy, add it to EXCLUDE in benchmark/curate.py and re-curate ==")
        print("   EXCLUDE = {")
        for row in unlisted.itertuples():
            print(f'       ("{row.source}", "{row.stem}"),')
        print("   }   # keep the two already there")
        print("   Then: re-run curate.py, and delete the stale directories it no longer writes.")
        print("   To KEEP them instead, add a row per dataset to the cluster map so each")
        print("   shares a cluster with its twin.")

    if args.emit_cluster_map:
        out = here[["dataset_id", "cluster"]].sort_values("dataset_id")
        args.emit_cluster_map.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.emit_cluster_map, index=False)
        print(f"\nwrote {len(out)} rows to {args.emit_cluster_map} "
              f"({out.cluster.nunique()} clusters); pass it as "
              f"holdout.py --cluster-map")
    return 0


if __name__ == "__main__":
    sys.exit(main())
