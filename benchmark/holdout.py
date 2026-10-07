#!/usr/bin/env python3
"""Draw the held-back datasets of the meta-analysis, before any result exists.

The meta-analysis (experiments/pilot10/meta_analysis.ipynb) asks which dataset properties
predict a quantum advantage. A model chosen and tested on the same datasets is optimistic,
so a random share of the corpus is held back: the meta-model is frozen on the discovery
set and then scored once on the held-back set. Every dataset is still run; only the
analysis splits them.

The unit drawn is the cluster, not the dataset: variants from one generator (one synthetic
family, or the same real data in two encodings) are not independent, and a held-back
variant whose sibling is in the discovery set would leak. The cluster of a dataset is its
meta.yaml ``family`` when that names a synthetic generator family, else the dataset itself,
unless a cluster map (``--cluster-map``, columns ``dataset_id,cluster``) says otherwise.

The draw is seeded and written once; rerunning with the same inputs reproduces it
byte for byte, and the output records its own sha256 so the analysis can pin it.

    python benchmark/holdout.py --datasets benchmark/datasets --fraction 0.2 --seed 20261006 \\
        --out benchmark/holdout.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GENERATOR_VERSION = "holdout/1.0"
#: meta.yaml families that are one generator each, so all their variants form one cluster.
SYNTHETIC_FAMILY_PREFIXES = ("shape_", "quantum_")


def dataset_clusters(datasets: Path, cluster_map: Path | None = None) -> pd.DataFrame:
    """One row per curated dataset under ``datasets``: ``dataset_id``, ``family``, ``cluster``."""
    rows = []
    for meta_path in sorted(datasets.glob("*/meta.yaml")):
        meta = yaml.safe_load(meta_path.read_text())
        ds = meta.get("dataset_id", meta_path.parent.name)
        family = str(meta.get("family", ""))
        cluster = family if family.startswith(SYNTHETIC_FAMILY_PREFIXES) else ds
        rows.append({"dataset_id": ds, "family": family, "cluster": cluster})
    frame = pd.DataFrame(rows, columns=["dataset_id", "family", "cluster"])
    if cluster_map is not None:
        mapping = pd.read_csv(cluster_map)
        if not {"dataset_id", "cluster"} <= set(mapping.columns):
            raise ValueError(f"{cluster_map}: needs dataset_id and cluster columns")
        frame["cluster"] = frame["dataset_id"].map(
            dict(zip(mapping["dataset_id"], mapping["cluster"]))).fillna(frame["cluster"])
    return frame


def draw_holdout(clusters: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """``clusters`` with a boolean ``holdout`` column: round(fraction x clusters) of them.

    Clusters are sorted before the draw, so the result depends only on their names, the
    fraction and the seed -- not on the order the datasets were listed in.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1); got {fraction}")
    names = sorted(clusters["cluster"].unique())
    n_hold = int(round(fraction * len(names)))
    if n_hold == 0 or n_hold == len(names):
        raise ValueError(f"{len(names)} clusters at fraction {fraction} leave an empty side")
    rng = np.random.default_rng(seed)
    held = set(rng.choice(names, size=n_hold, replace=False).tolist())
    out = clusters.copy()
    out["holdout"] = out["cluster"].isin(held)
    return out.sort_values("dataset_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", type=Path, required=True,
                        help="curated tree: <id>/<id>.csv + meta.yaml (curate.py, "
                             "create_synthetic_datasets.py)")
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="share of clusters held back (default 0.2)")
    parser.add_argument("--seed", type=int, required=True,
                        help="the draw's seed; record it in the analysis plan")
    parser.add_argument("--cluster-map", type=Path, default=None,
                        help="optional CSV dataset_id,cluster overriding the default clusters")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing draw (a second draw after seeing results "
                             "would defeat its purpose)")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} exists: the hold-out is drawn once. Pass --force only if no "
              f"result has been looked at.", file=sys.stderr)
        return 1
    table = draw_holdout(dataset_clusters(args.datasets, args.cluster_map),
                         args.fraction, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    n_c = table["cluster"].nunique()
    n_h = table.loc[table["holdout"], "cluster"].nunique()
    print(f"{GENERATOR_VERSION}: {n_h} of {n_c} clusters held back "
          f"({int(table['holdout'].sum())} of {len(table)} datasets), seed {args.seed}; "
          f"wrote {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
