#!/usr/bin/env python3
"""Report the corpus's clusters, and freeze them into the map the analysis needs.

``curate.py`` ends with a line like::

    duplicates.csv : 1 pair(s) to review

and leaves the review to a person, because the two possible answers are not
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

The 6 cross-source copies that decision covered are now in ``curate.py``'s ``EXCLUDE``, so
this script's first job is done; what remains is the second. It reports, against a cluster
map:

1. which corpus datasets the map does **not** list, split into those a naming rule still
   groups (no action) and those left alone in a cluster (a decision, since each adds one
   to ``G``);
2. which clusters hold more than one corpus dataset -- the groups recognised as
   non-independent -- and the resulting ``G``;
3. each ``duplicates.csv`` pair, and whether the map already puts it in one cluster;
4. a ready-to-paste ``EXCLUDE`` snippet for the singletons you choose to drop.

Clusters come from :func:`qbiocode.utils.meta_regression.dataset_family`, the function the
cluster-robust tests call, so what this prints is what they will use.

With ``--emit-cluster-map`` it writes a frozen ``dataset_id,cluster`` CSV covering exactly
the corpus on disk. That is the reason to keep running it after the dedup: it is the only
thing that produces that file, and both consumers want it. ``holdout.py --cluster-map``
requires those two column names, which ``experiments/cluster_map_draft.csv`` does not have
(its cluster column is ``cluster_cons``) and which it does not cover the synthetic corpus
with at all.

    python benchmark/review_duplicates.py --datasets $BENCH/data/datasets
    python benchmark/review_duplicates.py --datasets $BENCH/data/datasets \\
        --emit-cluster-map $BENCH/data/clusters.csv

Run it before ``make_splits.py`` if it might change the corpus. The split manifests pin
each CSV's sha256, so dropping a dataset afterwards leaves a manifest for a file the corpus
no longer has, and re-curating afterwards invalidates every manifest. Emitting the cluster
map changes no dataset, so that is safe at any time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    import qbiocode  # noqa: F401
except ImportError:  # run outside an env that has QBioCode installed: use this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    """Each dataset's cluster: the mapping where it lists one, else the naming rules.

    Delegates to :func:`qbiocode.utils.meta_regression.dataset_family`, the function the
    analysis itself calls, rather than reimplementing its rules -- so what this script
    reports is what the cluster-robust tests will actually use. An earlier copy of those
    rules here applied them to the *source-stripped* stem and so fell back to
    ``breast_cancer`` for both ``libsvm__breast_cancer`` (Wisconsin) and
    ``pmlb__breast_cancer`` (Ljubljana), merging two different datasets into one cluster
    whenever the map did not happen to list them.
    """
    from qbiocode.utils.meta_regression import dataset_family

    return datasets_frame.dataset_id.map(
        lambda dataset_id: dataset_family(dataset_id, mapping=mapping or None,
                                          key="dataset_id"))


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

    # 1. Corpus datasets the map omits. Those a naming rule still groups are fine and are
    #    only counted; the ones that end up alone in a cluster are what needs a decision,
    #    because a singleton cluster adds one to G.
    unlisted = here[~here.dataset_id.isin(mapping)]
    grouped_by_rule = unlisted[unlisted.cluster != unlisted.dataset_id]
    singletons = unlisted[unlisted.cluster == unlisted.dataset_id]
    print(f"\n== {len(unlisted)} corpus dataset(s) NOT in the cluster map ==")
    if len(grouped_by_rule):
        by_cluster = grouped_by_rule.groupby("cluster").dataset_id.count().sort_values(ascending=False)
        print(f"   {len(grouped_by_rule)} of them a naming rule still groups, into "
              f"{len(by_cluster)} cluster(s) -- no action needed:")
        for cluster, count in by_cluster.items():
            print(f"     {cluster:<32} {count:>3} dataset(s)")
    if len(singletons):
        print(f"   {len(singletons)} end up ALONE in a cluster. Each is a copy an earlier "
              f"decision dropped, or genuinely new:")
        from qbiocode.utils.meta_regression import GENERATED_SOURCES

        # Only the real sources are searched for twins. A generated dataset's n and p are
        # arguments to its generator, not properties of it -- every shapes set is n=400,
        # d=8 by construction -- so a collision there is evidence of nothing, and matching
        # on it listed 92 irrelevant siblings per row. The duplicate question for a
        # generated corpus is whether two ids got the same SEED, which shows up as equal
        # file contents, not equal shapes; compare the dataset_sha256 of the manifests.
        real_only = here[~here.source.isin(GENERATED_SOURCES)]
        for row in singletons.itertuples():
            if row.source in GENERATED_SOURCES:
                print(f"     {row.dataset_id:<44} n={row.n:<6} p={row.p:<6}"
                      f"   generated: no twin search (n, p are generator arguments)")
                continue
            twins = real_only[
                (real_only.stem.str.replace("-", "_") == row.stem.replace("-", "_"))
                & (real_only.dataset_id != row.dataset_id)]
            same_n = real_only[(real_only.n == row.n)
                               & (real_only.dataset_id != row.dataset_id)]
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

    # 4. The snippet, for the copies you choose to drop. Only the singletons: a dataset a
    #    naming rule already grouped is not a duplicate, and nominating all 93 shapes sets
    #    for deletion was worse than printing nothing.
    if len(singletons):
        print("\n== to DROP a copy, add it to EXCLUDE in benchmark/curate.py and re-curate ==")
        print("   Only the datasets alone in a cluster are listed; review each before")
        print("   pasting, since 'new' and 'duplicate' look the same from here.")
        print("   EXCLUDE = {")
        for row in singletons.itertuples():
            print(f'       ("{row.source}", "{row.stem}"),')
        print("   }   # merge with the entries already there")
        print("   Then: re-run curate.py --prune to drop the stale directories too.")
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
