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
"""
import argparse
import glob
import logging
import os

import pandas as pd
import yaml

from qbiocode.utils.qc_winner_finder import aggregate_benchmark

HERE = os.path.dirname(os.path.abspath(__file__))


def read_run_protocol(config_dir):
    """Return ``(test_size, n_iter, configs)`` as the configs actually declare them.

    Raises if the configs disagree: one shared ``test_size`` is an assumption baked into
    every interval the aggregator computes, so a split corpus has to be analysed as two.
    """
    paths = sorted(glob.glob(os.path.join(config_dir, "pilot*.yaml")))
    if not paths:
        raise SystemExit(f"no configs under {config_dir}; run generate_pilot_configs.py")
    seen, configs = {}, []
    for path in paths:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        configs.append((path, cfg))
        seen.setdefault((cfg.get("test_size"), cfg.get("iter")), []).append(
            os.path.basename(path)
        )
    if len(seen) > 1:
        detail = "; ".join(f"test_size={k[0]}, iter={k[1]}: {len(v)} configs" for k, v in seen.items())
        raise SystemExit(
            "configs disagree on the run protocol, so one correction factor cannot cover "
            f"them ({detail}). Analyse each group separately with --test-size."
        )
    (test_size, n_iter), _ = next(iter(seen.items()))
    return float(test_size), int(n_iter), configs


def granularity_report(configs, test_size, epsilon):
    """Flag datasets where ``epsilon`` is finer than one step of balanced accuracy.

    Balanced accuracy on a test set holding ``m`` minority rows can only take values on a
    grid of spacing ``0.5/m``. Where that spacing exceeds ``epsilon`` (the equivalence
    bound), an "equivalent" verdict is an artefact of the grid, not a measured effect,
    and no ``iter`` or
    ``test_size`` repairs it -- only more data does. Reported so those rows are read as
    granularity-limited rather than as null results.
    """
    rows = []
    for path, cfg in configs:
        folder = cfg.get("folder_path", "")
        for csv in cfg.get("file_dataset", []) or []:
            full = os.path.join(folder, csv)
            if not os.path.exists(full):
                continue
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

    Both arms are tuned by the same tuner over the same inner validation split, but not for
    the same number of trials. The classical budget is one number for the whole corpus; the
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
    p.add_argument("--config-dir", default=os.path.join(HERE, "configs"))
    p.add_argument("--test-size", type=float, default=None,
                   help="override; by default read from the configs")
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

    cfg_test_size, cfg_iter, configs = read_run_protocol(args.config_dir)
    test_size = args.test_size if args.test_size is not None else cfg_test_size
    if args.test_size is not None and abs(args.test_size - cfg_test_size) > 1e-12:
        # Loud, because every interval below is scaled by it and a silent override would
        # make the tables look precise while being wrong.
        print(f"!! overriding config test_size={cfg_test_size:g} with {args.test_size:g}")
    print(f"protocol from configs: test_size={cfg_test_size:g}  iter={cfg_iter}  "
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
