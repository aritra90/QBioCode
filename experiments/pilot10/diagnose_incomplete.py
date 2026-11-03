#!/usr/bin/env python3
"""Why each unfinished config of an array run is unfinished, and what to do about it.

    ./diagnose_incomplete.py --runs-dir $B/runs_cv/full1-qt-rdm2
    ./diagnose_incomplete.py --runs-dir ... --list     one line per config, not a summary

status.py says WHICH configs are partial or failed. This says WHY, from the array logs
submit_array.sh writes (array_logs/<job>.<jobid>.<index>.out/.err) -- which
status.py --why cannot read, because it looks for submit_runs.sh's per-config logs.

Each unfinished config is put in exactly one bucket, decided in this order:

  wall      the newest attempt's .out says TERM_RUNLIMIT: killed at its -W limit.
            Fix: resubmit with a longer wall (SKIP_EXISTING=1 keeps finished splits).
  memory    TERM_MEMLIMIT: killed for exceeding its memory request. Fix: MEM=16 or 32.
  error     the .err ends in a Python exception. The message is shown, grouped.
  head      a qpl config whose rows hold some heads but not others, with no kill and
            no error -- what a missing tabpfn looks like, since compute_qpl drops an
            unavailable head with a warning the tuning trials silence.
  no-log    no array log names this config: it was never started, or its log was
            removed. Fix: resubmit.
  unknown   started and stopped with none of the above.

The newest attempt is the one with the highest LSF job id, because job ids grow and an
earlier attempt's log says nothing about why the latest one stopped.
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status  # noqa: E402  (one definition of done/partial/failed)

TASK_LINE = re.compile(r"^task \d+ \(array index \d+ \+ offset \d+\): (\S+)\s+\[wall ([^\]]*)\]")
LOG_NAME = re.compile(r"\.(\d+)\.(\d+)\.out$")
EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))\b:?\s*(.*)$")


def latest_logs(log_dir):
    """{config: (jobid, out_path, wall)} -- the newest array attempt of each config."""
    newest = {}
    for out in glob.glob(os.path.join(log_dir, "*.out")):
        m = LOG_NAME.search(out)
        if not m:
            continue
        jobid = int(m.group(1))
        try:
            with open(out, errors="replace") as fh:
                for line in fh:
                    t = TASK_LINE.match(line)
                    if t:
                        config, wall = t.group(1), t.group(2)
                        if config not in newest or jobid > newest[config][0]:
                            newest[config] = (jobid, out, wall)
                        break
        except OSError:
            continue
    return newest


def last_exception(err_path):
    """The final exception line of an .err, or None."""
    if not os.path.exists(err_path):
        return None
    last = None
    with open(err_path, errors="replace") as fh:
        for line in fh:
            m = EXCEPTION.match(line.strip())
            if m:
                last = f"{m.group(1)}: {m.group(2)}"[:200]
    return last


def heads_present(cfg):
    """{model: n_iterations} in the config's best run, or {} when it has none."""
    run_dir, _ = status.best_run(cfg)
    path = os.path.join(run_dir, "ModelResults.csv") if run_dir else None
    if not path or not os.path.exists(path):
        return {}
    seen = defaultdict(set)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seen[row.get("model")].add(row.get("iteration"))
    return {model: len(its) for model, its in seen.items()}


def cfg_heads(cfg):
    """The heads a qpl config asked for, from its own YAML."""
    import ast
    with open(cfg["yaml"]) as fh:
        text = fh.read()
    m = re.search(r"['\"]classical_models['\"]\s*:\s*(\[[^\]]*\])", text)
    return list(ast.literal_eval(m.group(1))) if m else list(status.QPL_DEFAULT_HEADS)


def lsf_end(text):
    """How LSF says the job ended, from its .out: the TERM_ line or the exit line."""
    for pattern in (r"^(TERM_\w+:[^\n]*)", r"^(Exited with exit code \d+[^\n]*)",
                    r"^(Successfully completed[^\n]*)"):
        m = re.search(pattern, text, re.M)
        if m:
            return m.group(1).strip()
    return "no LSF exit line -- the job may still be running, or was killed hard"


def classify(cfg, log):
    """(bucket, detail) for one unfinished config."""
    if log is None:
        return "no-log", "no array log names this config"
    jobid, out, wall = log
    with open(out, errors="replace") as fh:
        text = fh.read()
    # LSF exit 140 is 128 + SIGUSR2's 12 -- the run-limit kill. A job can be killed hard
    # enough that its TERM_RUNLIMIT line never lands in the .out while the exit line does,
    # and this campaign had such jobs reported as "unknown".
    if "TERM_RUNLIMIT" in text or re.search(r"^Exited with exit code 140\b", text, re.M):
        return "wall", f"job {jobid} killed at -W {wall}"
    if "TERM_MEMLIMIT" in text:
        return "memory", f"job {jobid} exceeded its memory request"
    exc = last_exception(out[:-4] + ".err")
    if exc:
        return "error", exc
    heads = heads_present(cfg)
    if cfg["config"].endswith("_qpl") and heads:
        # A head is missing only if the config ASKED for it and no row names it. Every
        # head present at the same short count means the job stopped early, which is a
        # kill or a crash -- not a missing head. An earlier version reported "ABSENT"
        # whenever the heads it saw agreed with each other, so it called an
        # early-stopped job a missing tabpfn.
        wanted = cfg_heads(cfg)
        absent = sorted(h for h in wanted if not any(m.endswith("_" + h) for m in heads))
        if absent:
            return "head", f"heads {absent} wrote NO row; present {heads}"
    lsf = lsf_end(text)
    return "unknown", f"job {jobid}: {lsf}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--list", action="store_true", help="one line per unfinished config")
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    cfgs = [status.read_config(p) for p in status.find_configs(runs_dir)]
    if not cfgs:
        sys.exit(f"no configs under {runs_dir}")
    for c in cfgs:
        c["state"] = status.classify(c, {})
    todo = [c for c in cfgs if c["state"] in ("partial", "failed", "todo")]
    print(f"{runs_dir}: {len(cfgs)} configs, {len(todo)} not done "
          f"({dict(Counter(c['state'] for c in todo))})")
    if not todo:
        return

    logs = latest_logs(os.path.join(runs_dir, "array_logs"))
    buckets = defaultdict(list)
    for c in todo:
        bucket, detail = classify(c, logs.get(c["config"]))
        buckets[bucket].append((c, detail))

    order = ("wall", "memory", "error", "head", "no-log", "unknown")
    for bucket in order:
        items = buckets.get(bucket)
        if not items:
            continue
        arms = Counter(c["config"].rsplit("_", 1)[1] for c, _ in items)
        print(f"\n== {bucket}: {len(items)}  (by arm {dict(arms)})")
        if bucket == "error":
            for msg, n in Counter(d for _, d in items).most_common(8):
                print(f"   {n:4d}  {msg}")
        elif bucket == "wall":
            walls = Counter(d.rsplit("-W ", 1)[1] for _, d in items)
            print(f"   walls they were killed at: {dict(walls)}")
        elif bucket == "head":
            for _, d in items[:5]:
                print(f"   {d}")
        if bucket == "unknown":
            for msg, n in Counter(d.split(": ", 1)[1] for _, d in items).most_common(5):
                print(f"   {n:4d}  {msg}")
        if args.list or bucket in ("unknown", "no-log"):
            for c, d in items:
                print(f"   {c['config']:52s} rows {c['rows']}/{c['expected']:<4d} {d}")

    print("\nnext:")
    if buckets.get("wall"):
        print("  wall   -> SKIP_EXISTING=1 WALL=<longer> UMASK=002 ./submit_array.sh "
              "(re-runs only the unfinished; finished splits are adopted)")
    if buckets.get("memory"):
        print("  memory -> MEM=32 SKIP_EXISTING=1 UMASK=002 ./submit_array.sh")
    if buckets.get("head"):
        print("  head   -> check that tabpfn imports in the job's env: "
              "$PY -c 'import tabpfn'")
    if buckets.get("error"):
        print("  error  -> read the messages above; resubmitting will fail the same way")


if __name__ == "__main__":
    main()
