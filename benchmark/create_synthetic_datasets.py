#!/usr/bin/env python3
"""Generate synthetic benchmark datasets in the curated format, ready for make_splits.py.

Two kinds, chosen by flag:

    --shapes   labelled manifolds and partitions (synthetic_shapes.py): torus, sphere,
               concentric_circles, concentric_spheres, checkerboard, random_manifold,
               swiss_roll, half_moons, parity, perm_parity, simple_linear
    --quantum  quantum families (synthetic_quantum.py): angle_encoding and the five
               qbiocode.data_generation families (gs_sparse, gs_e2e, te, hl, ql_zz, ql_evo,
               eng_qiskit, eng_unit)

Each takes a comma list or 'all'. Every requested family is generated at every
combination of --k, --d, --n and --seeds that the family accepts; a family whose k is
unused is generated once per (d, n, seed). A combination a family cannot take (k > d for
parity, d too small) is skipped with a message.

    python benchmark/create_synthetic_datasets.py --out-root benchmark/datasets \\
        --shapes torus,sphere,half_moons --quantum angle_encoding,ql_zz,eng_unit \\
        --k 2,5,8 --d 8 --n 400 --seeds 0

Each dataset becomes ``<out-root>/<id>/<id>.csv`` (features, then an integer ``label``
in {0, 1} last; rows shuffled once) with a ``meta.yaml`` in curate.py's format, plus
the generator, its parameters, the label rule, the label-cell count, the training points
per cell under 5-fold CV, the role (shape, positive/negative control, ...) and the
pre-registered prediction. ``latent.npz`` holds the latent coordinates and the
continuous target ``F`` in row order, so the label can be re-derived independently.
The rows are listed in ``<out-root>/../inventory_synthetic.csv`` (curate.py owns
inventory.csv). Then:

    python benchmark/make_splits.py --datasets benchmark/datasets --out benchmark/splits/v2
    python experiments/pilot10/generate_pilot_configs.py --split-mode manifest ... \\
        --datasets-file benchmark/inventory_synthetic.csv

Ids are ``shapes__<family>_k<k>_d<d>_n<n>_s<seed>`` and ``quantum__<family>_...`` (no
``k`` part when the family ignores it, and ``_bw<b>`` after ``d`` off bandwidth 1).
Output is byte-deterministic for given flags.

**Positive controls are gated.** A ``positive_control`` family with a matched kernel
(ql_zz, eng_qiskit, eng_unit) is written only if a pilot draw of its generator clears the
gates of control_gates.py: G1 entangling map, G2 not concentrated, G3 learnable by the
matched kernel, G4 geometric difference from RBF. None of them fits a classical model or
reads a benchmark row, so the choice is not circular. A failed configuration is skipped
with the reasons; ``--keep-failed-controls`` writes it as role ``gate_rejected`` (leave it
out of both the control and the discovery families), and ``--no-control-gates`` writes
every control ungated. The gate record goes into meta.yaml under ``control_gates``.

``--bandwidth`` (the ql families) encodes ``b * x``: the matched qsvc needs the same
``bandwidth`` to reach the labels. The ql families also draw their concept (the
Heisenberg Hamiltonian) from a seed fixed per (family, d, k), so their seeds are samples
of one concept.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control_gates as cg  # noqa: E402
import synthetic_quantum as sq  # noqa: E402
import synthetic_shapes as ss  # noqa: E402

GENERATOR_VERSION = "create_synthetic_datasets/1.1"
STRUCTURE_SEED = 20261006          # seeds the embedding maps; fixed, so seeds share a manifold
CV_K = 5                           # training points per cell are reported for this k-fold
INVENTORY_COLUMNS = ["dataset_id", "source", "n", "p", "minority_n", "minority_fraction",
                     "family", "nan_rows_dropped", "sha256", "generator", "k", "d", "seed",
                     "role", "n_cells", "train_per_cell", "bandwidth", "control_accepted"]


def structure_rng(family: str, d: int, k: int) -> np.random.Generator:
    """The fixed RNG of one configuration's embedding maps (independent of the seed)."""
    return np.random.default_rng([STRUCTURE_SEED, zlib.crc32(family.encode()), d, k])


def concept_seed_for(family: str, d: int, k: int) -> int:
    """The concept seed of one fixed-concept configuration (independent of the data seed)."""
    return int(structure_rng(family, d, k).integers(2 ** 31 - 1))


def balanced(X, F, latent, n, rng):
    """Exactly n // 2 rows per class (label 1[F > 0]), shuffled; ValueError if one is short."""
    y = (F > 0).astype(int)
    half = n // 2
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < half or len(neg) < half:
        raise ValueError(f"oversample too small: {len(pos)} positive and {len(neg)} negative "
                         f"rows for {half} per class")
    idx = np.concatenate([rng.choice(pos, half, replace=False), rng.choice(neg, half, replace=False)])
    rng.shuffle(idx)
    return X[idx], y[idx], F[idx], latent[idx]


def reflect_into_unit(X: np.ndarray) -> np.ndarray:
    """Fold values that noise pushed outside [0, 1] back in: |x| below 0, 2 - x above 1.

    Clipping, as the reference does, piles them up at exactly 0 and 1. At noise 0.01 that
    was about 5% of a torus's entries, in every column. Those atoms tie rows in the kNN
    and density measures, and make a scaler fitted on any subset of rows identical to one
    fitted on all of them. Reflection keeps every value in [0, 1] and leaves the rest as
    drawn. The final clip only matters for noise near 1.
    """
    X = np.abs(X)
    return np.clip(np.where(X > 1.0, 2.0 - X, X), 0.0, 1.0)


def make_shape(name: str, n: int, d: int, k: int, seed: int, noise: float):
    """(features frame, labels, F, latent, params) of one shape dataset."""
    shape = ss.SHAPES[name]
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), d, k])
    for factor in (shape.oversample, 4 * shape.oversample, 16 * shape.oversample):
        X, F, latent = shape.generate(factor * n, d, k, rng, structure_rng(name, d, k))
        try:
            X, y, F, latent = balanced(X, F, latent, n, rng)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"{name}: could not draw {n // 2} rows per class at k={k}, d={d}")
    if noise > 0:
        # After labelling, as in the reference: the label is a function of the latent, and
        # the features are its noisy observation.
        X = reflect_into_unit(X + rng.normal(0.0, noise, X.shape))
    frame = pd.DataFrame(X, columns=[f"x{j}" for j in range(d)])
    return frame, y, F, latent, {"noise": noise}


def make_quantum(name: str, n: int, d: int, k: int, seed: int, bandwidth: float = 1.0):
    """(features frame, labels, F, latent, params) of one quantum dataset."""
    fam = sq.QUANTUM[name]
    if fam.native is not None:
        rng = np.random.default_rng([seed, zlib.crc32(name.encode()), d, k])
        X, F, latent = fam.native(n, d, k, rng, structure_rng(name, d, k))
        X, y, F, latent = balanced(X, F, latent, n, rng)
        return pd.DataFrame(X, columns=[f"x{j}" for j in range(d)]), y, F, latent, {}
    frame, F, meta = sq._qbiocode_family(
        name, n, d, k, seed,
        concept_seed=concept_seed_for(name, d, k) if fam.fixed_concept else None,
        bandwidth=bandwidth if fam.uses_bandwidth else 1.0)
    y = frame.pop("label").to_numpy().astype(int)
    # The qbiocode generators write rows in generation order; shuffle once, like curate.py.
    order = np.random.default_rng([seed, zlib.crc32(name.encode()), d, k]).permutation(len(y))
    params = {"qbiocode_meta": _plain(meta)}
    return frame.iloc[order].reset_index(drop=True), y[order], F[order], F[order][:, None], params


def _plain(value):
    """JSON/YAML-safe copy: numpy scalars and arrays to Python values, recursively."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_id(source: str, family: str, k: int | None, d: int, n: int, seed: int,
               bandwidth: float = 1.0) -> str:
    kpart = f"_k{k}" if k is not None else ""
    bwpart = f"_bw{bandwidth:g}" if bandwidth != 1.0 else ""
    return f"{source}__{family}{kpart}_d{d}{bwpart}_n{n}_s{seed}"


def write_dataset(out_root: Path, ds_id: str, frame: pd.DataFrame, y: np.ndarray, F: np.ndarray,
                  latent: np.ndarray, meta: dict) -> dict:
    """Write <out_root>/<ds_id>/{<ds_id>.csv, meta.yaml, latent.npz}; return the inventory row."""
    folder = out_root / ds_id
    folder.mkdir(parents=True, exist_ok=True)
    data = frame.copy()
    data["label"] = y.astype(int)
    csv_path = folder / f"{ds_id}.csv"
    data.to_csv(csv_path, index=False)
    np.savez(folder / "latent.npz", latent=np.asarray(latent, float), F=np.asarray(F, float))
    counts = {int(c): int((y == c).sum()) for c in (0, 1)}
    minority = min(counts.values())
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    n_train = len(y) * (CV_K - 1) / CV_K
    cells = meta.get("n_cells")
    full = {
        "dataset_id": ds_id, "source": meta["source"],
        "original_file": None, "original_shape": [int(len(y)), int(frame.shape[1] + 1)],
        "n": int(len(y)), "p": int(frame.shape[1]), "label_col_original": "label",
        "class_counts": counts, "minority_n": minority,
        "minority_fraction": round(minority / len(y), 6),
        "n_components_max": int(frame.shape[1]), "group_col": None, "family": meta["family"],
        "dropped_id_columns": [], "suspected_id_columns": [], "nan_rows_dropped": 0,
        "one_hot_columns": [], "one_hot_drop_first": False, "dropped_constant_columns": [],
        "generator_version": GENERATOR_VERSION, "generator": meta["generator"],
        "k": meta["k"], "d": meta["d"], "seed": meta["seed"], "label_rule": meta["label_rule"],
        "role": meta["role"], "matched_arm": meta.get("matched_arm"),
        "prediction": meta.get("prediction", ""), "k_meaning": meta.get("k_meaning", ""),
        "n_cells": cells,
        "train_per_cell": None if cells in (None, 0) else round(n_train / cells, 3),
        "structure_seed": STRUCTURE_SEED, "params": _plain(meta.get("params", {})),
        "bandwidth": meta.get("bandwidth", 1.0), "concept_seed": meta.get("concept_seed"),
        "control_gates": _plain(meta.get("control_gates")),
        "control_accepted": (None if meta.get("control_gates") is None
                             else bool(meta["control_gates"]["accepted"])),
        "steps": ["generated by benchmark/create_synthetic_datasets.py",
                  "label = 1[F > 0] (median-thresholded F for the quantum families)",
                  "rows balanced to n // 2 per class and shuffled once"],
        "sha256": sha,
    }
    (folder / "meta.yaml").write_text(yaml.safe_dump(full, sort_keys=False))
    return {c: full.get(c) for c in INVENTORY_COLUMNS}


def write_inventory(out_root: Path) -> Path:
    """Rebuild ``<out_root>/../inventory_synthetic.csv`` from every synthetic meta.yaml on disk.

    Rebuilt rather than appended, so it always lists exactly the synthetic datasets that
    exist -- also after an interrupted run, and never a dataset that was deleted.
    """
    rows = []
    for meta_path in sorted(out_root.glob("*/meta.yaml")):
        meta = yaml.safe_load(meta_path.read_text())
        if str(meta.get("generator_version", "")).startswith("create_synthetic_datasets/"):
            rows.append({c: meta.get(c) for c in INVENTORY_COLUMNS})
    path = out_root.parent / "inventory_synthetic.csv"
    pd.DataFrame(rows, columns=INVENTORY_COLUMNS).sort_values("dataset_id").to_csv(path, index=False)
    return path


def parse_list(text: str | None, catalogue: dict, flag: str) -> list[str]:
    if text is None:
        return []
    names = sorted(catalogue) if text.strip() == "all" else [t.strip() for t in text.split(",") if t.strip()]
    unknown = [t for t in names if t not in catalogue]
    if unknown:
        raise SystemExit(f"{flag}: unknown {', '.join(unknown)}; choose from {', '.join(sorted(catalogue))} or all")
    return names


def int_list(text: str, flag: str) -> list[int]:
    try:
        vals = [int(t) for t in str(text).split(",") if t.strip()]
    except ValueError:
        raise SystemExit(f"{flag}: {text!r} is not a comma list of integers")
    if not vals or min(vals) < 0:
        raise SystemExit(f"{flag}: {text!r} needs non-negative integers")
    return vals


def float_list(text: str, flag: str) -> list[float]:
    try:
        vals = [float(t) for t in str(text).split(",") if t.strip()]
    except ValueError:
        raise SystemExit(f"{flag}: {text!r} is not a comma list of numbers")
    if not vals or min(vals) <= 0 or not all(np.isfinite(vals)):
        raise SystemExit(f"{flag}: {text!r} needs positive finite numbers")
    return vals


def plan(shapes, quantum, ks, ds, ns, seeds, bws=(1.0,)):
    """Every (kind, family, k, d, n, seed, bandwidth) to build, and the skips with reasons.

    ``bws`` reaches only the families that take a bandwidth; the rest are built at 1.
    """
    jobs, skips, seen = [], [], set()
    for kind, names, cat in (("shapes", shapes, ss.SHAPES), ("quantum", quantum, sq.QUANTUM)):
        for name in names:
            fam = cat[name]
            for d in ds:
                for k in ks:
                    kk = k if fam.uses_k else None
                    if d < fam.d_min or (kind == "quantum" and d > fam.d_max):
                        skips.append(f"{name} d={d}: needs {fam.d_min} <= d"
                                     + (f" <= {fam.d_max}" if kind == "quantum" else ""))
                        continue
                    if kk is not None and kind == "shapes" and (
                            kk < fam.k_min or (fam.k_max_d and kk > d)):
                        skips.append(f"{name} k={kk} d={d}: needs k >= {fam.k_min}"
                                     + (" and k <= d" if fam.k_max_d else ""))
                        continue
                    if name == "angle_encoding" and 2 * k > d:
                        skips.append(f"angle_encoding k={k} d={d}: needs d >= 2k")
                        continue
                    for n in ns:
                        n_max = getattr(fam, "n_max", None)
                        if n_max is not None and n > n_max(d):
                            skips.append(f"{name} d={d} n={n}: at most {n_max(d)} distinct rows")
                            continue
                        for bw in (bws if getattr(fam, "uses_bandwidth", False) else (1.0,)):
                            for seed in seeds:
                                key = (kind, name, kk, d, n, seed, float(bw))
                                if key not in seen:
                                    seen.add(key)
                                    jobs.append(key)
    return jobs, skips


def control_gate(name, k, d, n, bandwidth=1.0):
    """The gates of one positive-control configuration, from a pilot draw of its generator.

    The pilot uses control_gates.PILOT_SEED, never a data seed, and the matched kernel
    sees the inputs min-max scaled, as QProfiler would hand them over. Returns None for a
    family the gates do not apply to.
    """
    fam = sq.QUANTUM[name]
    if fam.role != "positive_control" or fam.matched_kernel is None:
        return None
    kk = k if k is not None else 2
    frame, y, _, _, _ = make_quantum(name, n, d, kk, cg.PILOT_SEED, bandwidth)
    X = cg.minmax(frame.to_numpy())
    K = fam.matched_kernel(X, kk, d, bandwidth)
    return cg.evaluate(X, y, K, n_train=n * (CV_K - 1) / CV_K, entangling=fam.entangling,
                       n_qubits=d if fam.fidelity else None)


def build(kind, name, k, d, n, seed, noise, bandwidth=1.0):
    """(frame, y, F, latent, meta) of one planned dataset."""
    if kind == "shapes":
        fam = ss.SHAPES[name]
        frame, y, F, latent, params = make_shape(name, n, d, k if k is not None else 0, seed, noise)
        meta = {"source": "shapes", "family": f"shape_{name}", "generator": f"synthetic_shapes.{name}",
                "role": fam.role, "label_rule": fam.label_rule, "n_cells": fam.cells(k or 0, d),
                "k_meaning": "unused" if k is None else ss.SHAPES[name].generate.__doc__.splitlines()[0]}
    else:
        fam = sq.QUANTUM[name]
        frame, y, F, latent, params = make_quantum(name, n, d, k if k is not None else 2, seed,
                                                   bandwidth)
        meta = {"source": "quantum", "family": f"quantum_{name}", "generator": f"synthetic_quantum.{name}",
                "role": fam.role, "label_rule": fam.label_rule, "matched_arm": fam.matched_arm,
                "prediction": fam.prediction, "k_meaning": fam.k_meaning,
                "n_cells": 2 ** k if name == "angle_encoding" else None,
                "bandwidth": float(bandwidth),
                "concept_seed": concept_seed_for(name, d, k if k is not None else 2)
                if fam.fixed_concept else None}
    meta.update(k=k, d=d, seed=seed, params={**params, "n": n})
    return frame, y, F, latent, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, default=HERE / "datasets",
                        help="curated tree to write into (default benchmark/datasets, gitignored)")
    parser.add_argument("--shapes", default=None, help="comma list of shape families, or all")
    parser.add_argument("--quantum", default=None, help="comma list of quantum families, or all")
    parser.add_argument("--k", default="2", help="complexity knob(s), comma list (default 2)")
    parser.add_argument("--d", default="8", help="feature / qubit count(s), comma list (default 8)")
    parser.add_argument("--n", default="400", help="rows per dataset, comma list (default 400)")
    parser.add_argument("--seeds", default="0", help="sample seeds, comma list (default 0)")
    parser.add_argument("--bandwidth", default="1",
                        help="input bandwidth(s) b of the ql families, which encode b*x; comma "
                             "list (default 1). The matched qsvc needs the same bandwidth.")
    parser.add_argument("--no-control-gates", action="store_true",
                        help="write positive controls without the control_gates.py check")
    parser.add_argument("--keep-failed-controls", action="store_true",
                        help="write a control that fails its gates, as role gate_rejected")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Gaussian feature noise after labelling, reflected back into [0, 1] at "
                             "the cube faces; shapes only (default 0.01)")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)

    if args.list:
        for title, cat in (("shapes", ss.SHAPES), ("quantum", sq.QUANTUM)):
            print(f"--{title}")
            for name, fam in sorted(cat.items()):
                print(f"  {name:20s} role {fam.role:20s} d >= {fam.d_min:<3d} "
                      f"{'k used' if fam.uses_k else 'k unused':9s} {fam.label_rule}")
        return 0
    shapes = parse_list(args.shapes, ss.SHAPES, "--shapes")
    quantum = parse_list(args.quantum, sq.QUANTUM, "--quantum")
    if not shapes and not quantum:
        parser.error("give --shapes and/or --quantum (a comma list or 'all'); --list shows them")
    ks, ds = int_list(args.k, "--k"), int_list(args.d, "--d")
    ns, seeds = int_list(args.n, "--n"), int_list(args.seeds, "--seeds")
    bws = float_list(args.bandwidth, "--bandwidth")
    if min(ns) < 20:
        parser.error("--n must be at least 20 rows")
    jobs, skips = plan(shapes, quantum, ks, ds, ns, seeds, bws)
    for s in skips:
        print(f"skipping {s}")
    # Gate every positive-control configuration once (the gates do not depend on the seed).
    gates = {}
    if not args.no_control_gates:
        for kind, name, k, d, n, seed, bw in jobs:
            key = (name, k, d, n, bw)
            if kind == "quantum" and key not in gates:
                gates[key] = control_gate(name, k, d, n, bw)
                if gates[key] is not None:
                    verdict = "accepted" if gates[key]["accepted"] else "REJECTED: " + cg.reasons(gates[key])
                    print(f"control gates {name} k={k} d={d} bw={bw:g} n={n}: {verdict}")
    kept = []
    for job in jobs:
        kind, name, k, d, n, seed, bw = job
        gate = gates.get((name, k, d, n, bw))
        if gate is not None and not gate["accepted"] and not args.keep_failed_controls:
            continue
        kept.append(job)
    dropped = len(jobs) - len(kept)
    print(f"{len(kept)} datasets to write under {args.out_root}"
          + (f" ({dropped} failed their control gates; --keep-failed-controls keeps them)"
             if dropped else ""))
    if args.dry_run:
        for kind, name, k, d, n, seed, bw in kept:
            print("  " + dataset_id(kind, name, k, d, n, seed, bw))
        return 0
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, name, k, d, n, seed, bw in kept:
        ds_id = dataset_id(kind, name, k, d, n, seed, bw)
        frame, y, F, latent, meta = build(kind, name, k, d, n, seed, args.noise, bw)
        gate = gates.get((name, k, d, n, bw))
        if gate is not None:
            meta["control_gates"] = gate
            if not gate["accepted"]:
                meta["role"] = "gate_rejected"
        row = write_dataset(args.out_root, ds_id, frame, y, F, latent, meta)
        rows.append(row)
        tpc = "" if row["train_per_cell"] is None else f"  {row['train_per_cell']:.1f} train/cell"
        print(f"  wrote {ds_id:52s} n={row['n']} p={row['p']}  role {row['role']}{tpc}")
    inv_path = write_inventory(args.out_root)
    print(f"inventory: {inv_path} ({len(pd.read_csv(inv_path))} synthetic datasets)")
    low = [r["dataset_id"] for r in rows if r["train_per_cell"] is not None and r["train_per_cell"] < 1]
    if low:
        print(f"note: {len(low)} datasets have fewer than one training point per label cell "
              f"(every method sits at chance there): {', '.join(low[:5])}{' ...' if len(low) > 5 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
