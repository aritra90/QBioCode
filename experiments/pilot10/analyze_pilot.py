#!/dccstor/boseukb/Q/envs/qbc/bin/python
"""Aggregate the pilot10 run into the corpus-level tables.

Run this AFTER the 12 LSF jobs finish. It walks the per-dataset result directories,
builds the delta-metric table, applies the fair-selection verdict rule to every
metric, and -- if the kernel dumps are present -- adds the kernel-geometry table.

    ./analyze_pilot.py                       # defaults below
    ./analyze_pilot.py --primary-metric f1_score
    ./analyze_pilot.py --no-kernels          # skip diagnostics (they are the slow part)

``test_size`` is READ FROM THE CONFIGS, not defaulted. The Nadeau-Bengio correction
needs the train/test ratio to turn per-iteration scatter into a standard error, and a
wrong value silently rescales every confidence interval instead of raising -- so a
hard-coded default here would quietly drift out of step the first time the configs were
regenerated with a different one. If the configs disagree with each other, that is a
corpus that cannot share one correction factor, and this script refuses to guess.

The configs read are the ones the jobs ran (``--config-dir``: the combined-layout
``configs`` by default, or a runs tree such as ``runs_cv/<run_id>``). Under ``split_mode: manifest`` (stratified k-fold repeated R times, each fold
with its own validation rows) the protocol is ``k`` and ``R`` from the split manifests
the configs name, the correction is ``r = 1/(k-1)`` with ``kR - 1`` degrees of freedom,
and winners are selected on validation (``select_winners(selection='validation')``)
instead of by leave-one-iteration-out.
"""
import argparse
import glob
import logging
import os
from typing import NamedTuple

import pandas as pd
import yaml

from qbiocode.utils.qc_winner_finder import aggregate_benchmark

HERE = os.path.dirname(os.path.abspath(__file__))


class RunProtocol(NamedTuple):
    """The evaluation protocol of a set of configs, as :func:`read_run_protocol` reads it.

    ``test_size`` is the value the Nadeau-Bengio / Bouckaert-Frank correction uses:
    the config's ``test_size`` under ``split_mode: internal``, ``1/k`` under
    ``split_mode: manifest`` (so ``r = test_size/(1-test_size) = 1/(k-1)``).
    ``n_iter`` is ``iter`` (internal) or, in manifest mode, the number of distinct
    global iterations the configs select through their ``splits`` key (``k * n_repeats``
    when every config runs ``'all'``); ``k`` and ``n_repeats`` are ``None`` in internal
    mode.
    """

    split_mode: str
    test_size: float
    n_iter: int
    k: int | None
    n_repeats: int | None
    configs: list

    @property
    def r(self):
        """``n_test / n_train``, the correction term."""
        return self.test_size / (1.0 - self.test_size)

    @property
    def selection(self):
        """The select_winners rule this protocol calls for."""
        return "validation" if self.split_mode == "manifest" else "loio"


def _load_config(path):
    """A config with its same-directory hydra defaults merged in (``_self_`` last unless
    listed), the way the generator composes split-layout job files. Top-level merge only:
    the protocol keys read here are all top-level."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg
    merged = {}
    for entry in defaults if "_self_" in defaults else [*defaults, "_self_"]:
        if entry == "_self_":
            merged.update(cfg)
        elif isinstance(entry, str):
            layer = os.path.join(os.path.dirname(path), f"{entry}.yaml")
            if os.path.exists(layer):
                merged.update(_load_config(layer))
    return merged


def _config_paths(config_dir):
    """The job configs under ``config_dir``: the combined layout's ``pilot*.yaml``, else a
    runs tree's ``<dataset>/*.yaml`` (the files the jobs actually ran), skipping the
    ``_``-prefixed layers (``_protocol.yaml``) that are not jobs themselves."""
    paths = sorted(glob.glob(os.path.join(config_dir, "pilot*.yaml")))
    if not paths:
        paths = sorted(p for p in glob.glob(os.path.join(config_dir, "*", "*.yaml"))
                       if not os.path.basename(p).startswith("_"))
    return paths


def _manifest_protocol(cfg, path):
    """``(k, n_repeats, iterations)`` of a manifest-mode config, from the manifests it
    names; ``iterations`` are the global iterations its ``splits`` key selects."""
    from qbiocode.apps.qprofiler.split_manifest import load_manifest, manifest_path

    split_dir = cfg.get("split_dir")
    files = cfg.get("file_dataset") or []
    files = [files] if isinstance(files, str) else list(files)
    if not split_dir or not files:
        raise SystemExit(f"{path}: split_mode manifest needs split_dir and file_dataset")
    shapes, iterations = set(), set()
    for csv in files:
        manifest = load_manifest(manifest_path(split_dir, csv))
        shapes.add((int(manifest.k), int(manifest.n_repeats)))
        iterations.update(int(sp.iteration) for sp in manifest.select(cfg.get("splits", "all")))
    if len(shapes) > 1:
        raise SystemExit(f"{path}: its datasets' manifests disagree on (k, R): {sorted(shapes)}")
    return (*shapes.pop(), iterations)


def read_run_protocol(config_dir, *, detail=False):
    """Return ``(test_size, n_iter, configs)`` as the configs actually declare them.

    ``config_dir`` is either a combined-layout config directory (``pilot*.yaml``) or a
    runs tree (``<dataset>/*.yaml``, the job files that were actually executed; their
    ``_protocol.yaml`` defaults layer is merged in). Under ``split_mode: manifest``,
    ``k`` and ``R`` are read from the split manifests the configs name
    (``split_dir``), ``test_size = 1/k``, and ``n_iter`` counts the distinct global
    iterations the configs' ``splits`` select (``k*R`` for a run over every split).

    Raises if the configs disagree: one shared protocol is an assumption baked into
    every interval the aggregator computes, so a split corpus has to be analysed as two.

    Args:
        config_dir (str): directory of configs, as above.
        detail (bool): return a :class:`RunProtocol` (with ``split_mode``, ``k``,
            ``n_repeats`` and ``r``) instead of the 3-tuple.
    """
    paths = _config_paths(config_dir)
    if not paths:
        raise SystemExit(f"no configs under {config_dir}; run generate_pilot_configs.py")
    seen, configs, iterations = {}, [], set()
    for path in paths:
        cfg = _load_config(path)
        configs.append((path, cfg))
        mode = cfg.get("split_mode") or "internal"
        if mode == "manifest":
            k_, reps_, its = _manifest_protocol(cfg, path)
            key = (mode, k_, reps_)
            iterations |= its
        else:
            key = (mode, cfg.get("test_size"), cfg.get("iter"))
        seen.setdefault(key, []).append(os.path.basename(path))
    if len(seen) > 1:
        detail_ = "; ".join(
            (f"split_mode=manifest, k={key[1]}, R={key[2]}" if key[0] == "manifest"
             else f"test_size={key[1]}, iter={key[2]}") + f": {len(v)} configs"
            for key, v in seen.items())
        # --test-size can re-scale an internal group, but is refused in manifest mode, so
        # the advice depends on what is mixed.
        advice = ("Point --config-dir at one runs tree per protocol."
                  if any(key[0] == "manifest" for key in seen)
                  else "Analyse each group separately with --test-size.")
        raise SystemExit(
            "configs disagree on the run protocol, so one correction factor cannot cover "
            f"them ({detail_}). {advice}"
        )
    key, _ = next(iter(seen.items()))
    if key[0] == "manifest":
        k, n_repeats = key[1], key[2]
        protocol = RunProtocol("manifest", 1.0 / k, len(iterations), k, n_repeats, configs)
    else:
        if key[1] is None or key[2] is None:
            raise SystemExit(
                f"configs under {config_dir} do not declare test_size and iter "
                f"({len(configs)} configs); cannot derive the correction factor."
            )
        protocol = RunProtocol("internal", float(key[1]), int(key[2]), None, None, configs)
    if detail:
        return protocol
    return protocol.test_size, protocol.n_iter, protocol.configs


def granularity_report(configs, test_size, epsilon):
    """Flag datasets where ``epsilon`` is finer than one step of balanced accuracy.

    ``test_size`` is the test fraction: ``1/k`` under ``split_mode: manifest``, so the
    test minority is about ``minority/k`` per outer fold.

    Balanced accuracy on a test set holding ``m`` minority rows can only take values on a
    grid of spacing ``0.5/m``. Where that spacing exceeds ``epsilon`` (the equivalence
    bound), an "equivalent" verdict is an artefact of the grid, not a measured effect,
    and no ``iter`` or
    ``test_size`` repairs it -- only more data does. Reported so those rows are read as
    granularity-limited rather than as null results.
    """
    rows, done = [], set()
    for path, cfg in configs:
        folder = cfg.get("folder_path", "")
        for csv in cfg.get("file_dataset", []) or []:
            full = os.path.join(folder, csv)
            # A split-layout runs tree has one config per job, many per dataset.
            if full in done or not os.path.exists(full):
                continue
            done.add(full)
            # Label is the last column -- the convention the rest of the repo uses when it
            # computes `df.shape[1] - 1` features.
            y = pd.read_csv(full).iloc[:, -1]
            minority = int(y.value_counts().min())
            m = int(round(minority * test_size))
            step = 0.5 / m if m else float("nan")
            rows.append({
                "Dataset": csv,
                "n": int(len(y)),
                "minority": minority,
                "test_minority": m,
                "ba_step": round(step, 4),
                "epsilon": round(epsilon, 4) if epsilon else None,
                "granularity_limited": bool(epsilon and step > epsilon),
            })
    return pd.DataFrame(rows)



def tuning_budget_report(configs):
    """Per-dataset search budgets for the two arms, which the pilot does NOT equalise.

    Both arms are tuned by the same tuner, but in the pilot (``split_mode: internal``) not
    for the same number of trials, nor on the same validation rows: classical tuners score
    an inner cross-validation, quantum ones an inner holdout. (``split_mode: manifest``
    gives every arm the same ``n_trials`` on the fold's shared validation rows, and the
    ratio below is then 1.) The classical budget is one number for the whole corpus; the
    quantum budget is chosen per dataset by the wall-clock model (see cost_model.py and
    ``generate_pilot_configs.py --budget-hours``), because quantum cost grows with row count
    -- quadratically for qsvc, whose kernel needs one circuit per training pair -- while
    classical cost effectively does not at these sizes.

    That asymmetry is a confound, and it has a direction: the arm allowed fewer trials
    searches less of its space, so the deficit is biased toward reading as a quantum LOSS,
    and it is largest exactly where quantum is most expensive -- the widest and longest
    datasets. It cannot be removed inside a fixed wall, only bought out with more hours. So
    it is reported here, beside the verdicts rather than buried in the configs, because a
    reader needs to know which rows carry it BEFORE reading a verdict on them.

    ``freeze_quantum_params`` is reported alongside for the same reason: with it on, the
    quantum search runs once on iteration 0 and every later resample refits the winning
    configuration, so the quantum arm also sees fewer independent searches than the
    classical arm, which re-searches every resample.
    """
    rows = []
    for path, cfg in configs:
        classical = cfg.get("n_trials")
        quantum = cfg.get("n_trials_quantum") if cfg.get("tune_quantum") else 0
        rows.append({
            "Config": os.path.basename(path).replace(".yaml", ""),
            "classical_trials": classical,
            "quantum_trials": quantum,
            # How many times more of its space the classical arm got to search.
            "ratio": (round(classical / quantum, 1)
                      if classical and quantum else float("inf")),
            "quantum_searches_once": bool(cfg.get("freeze_quantum_params")),
            "iter": cfg.get("iter"),
        })
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default=os.path.join(HERE, "results"))
    p.add_argument("--kernels-root", default=os.path.join(HERE, "kernels"))
    p.add_argument("--output-dir", default=os.path.join(HERE, "analysis"))
    p.add_argument("--tag", default="pilot10")
    p.add_argument("--primary-metric", default="balanced_accuracy")
    p.add_argument("--config-dir", default=os.path.join(HERE, "configs"),
                   help="the configs the jobs ran: a combined-layout dir (pilot*.yaml; "
                        "the default) or a runs tree (<dataset>/*.yaml), e.g. "
                        "runs_cv/ID for a split_mode: manifest run")
    p.add_argument("--test-size", type=float, default=None,
                   help="override; by default read from the configs (split_mode "
                        "internal only: under split_mode manifest it is 1/k)")
    p.add_argument("--epsilon", type=float, default=None,
                   help="TOST equivalence bound; omit to derive it per metric from the "
                        "resolution floor")
    p.add_argument("--margin", type=float, default=0.0,
                   help="pre-registered superiority margin a win must clear (default 0)")
    p.add_argument("--fdr", type=float, default=0.10,
                   help="Benjamini-Hochberg level of the adjusted verdicts (default 0.10)")
    p.add_argument("--no-kernels", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    protocol = read_run_protocol(args.config_dir, detail=True)
    configs, cfg_test_size = protocol.configs, protocol.test_size
    if protocol.split_mode == "manifest" and args.test_size is not None:
        raise SystemExit("--test-size does not apply to split_mode manifest: the "
                         f"correction is r = 1/(k-1) with k={protocol.k} from the manifests")
    test_size = args.test_size if args.test_size is not None else cfg_test_size
    if args.test_size is not None and abs(args.test_size - cfg_test_size) > 1e-12:
        # Loud, because every interval below is scaled by it and a silent override would
        # make the tables look precise while being wrong.
        print(f"!! overriding config test_size={cfg_test_size:g} with {args.test_size:g}")
    if protocol.split_mode == "manifest":
        print(f"protocol from configs: split_mode=manifest  k={protocol.k}  "
              f"R={protocol.n_repeats}  r=1/(k-1)={protocol.r:.4g}  "
              f"splits run={protocol.n_iter}/{protocol.k * protocol.n_repeats}  "
              f"({len(configs)} configs); validation-selected winners")
    else:
        print(f"protocol from configs: test_size={cfg_test_size:g}  iter={protocol.n_iter}  "
              f"({len(configs)} configs)")

    out = aggregate_benchmark(
        results_root=args.results_root,
        output_dir=args.output_dir,
        tag=args.tag,
        primary_metric=args.primary_metric,
        epsilon=args.epsilon,
        test_size=test_size,
        margin=args.margin,
        fdr=args.fdr,
        kernels_root=None if args.no_kernels else args.kernels_root,
        selection=protocol.selection,
        k=protocol.k,
    )

    print("\n=== inventory (did every arm land?) ===")
    print(out["inventory"].to_string(index=False))
    print("\n=== verdicts, every metric ===")
    print(out["summary"].to_string(index=False))

    kd = out["kernel_diagnostics"]
    if kd is not None and len(kd):
        cols = [c for c in ("Dataset", "model", "quantum_kernel_form", "kta_quantum",
                            "kta_classical", "kta_ratio", "g_cq_lam0.01",
                            "margin_quantum", "margin_classical") if c in kd.columns]
        print(f"\n=== kernel geometry ({len(kd)} splits) ===")
        print(kd[cols].to_string(index=False))
    else:
        print("\n(no kernel diagnostics -- no dumps found under %s)" % args.kernels_root)

    eps = out["epsilon"].get(args.primary_metric)
    gran = granularity_report(configs, test_size, eps)
    if len(gran):
        limited = gran[gran["granularity_limited"]]
        print(f"\n=== metric granularity (epsilon={eps:.4f} for {args.primary_metric}) ===")
        print(gran.to_string(index=False))
        gran.to_csv(os.path.join(args.output_dir, f"{args.tag}_granularity.csv"), index=False)
        if len(limited):
            print(f"\n!! {len(limited)} dataset(s) are GRANULARITY-LIMITED: "
                  f"{', '.join(limited['Dataset'])}")
            print("   One step of balanced accuracy there is coarser than epsilon, so a "
                  "verdict on\n   those rows reflects the grid, not an effect. More data is "
                  "the only fix.")

    budget = tuning_budget_report(configs)
    if len(budget):
        print("\n=== hyperparameter search budget per arm (NOT equal -- report this) ===")
        print(budget.to_string(index=False))
        budget.to_csv(os.path.join(args.output_dir, f"{args.tag}_tuning_budget.csv"),
                      index=False)
        worst = budget.iloc[0]
        if worst["ratio"] > 1:
            print(f"\n!! the classical arm searched up to {worst['ratio']:g}x more "
                  f"configurations than the quantum arm ({worst['Config']}: "
                  f"{worst['classical_trials']} vs {worst['quantum_trials']}).")
            print("   The budget was set by wall-clock cost, not by anything about the "
                  "methods, and the\n   deficit favours the classical arm -- so a classical "
                  "win on the high-ratio rows is\n   not separable from a search-budget "
                  "effect. Quote this table in the methods.")

    print("\nwrote:", sorted(os.listdir(args.output_dir)))


if __name__ == "__main__":
    main()
