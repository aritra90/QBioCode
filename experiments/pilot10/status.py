#!/dccstor/boseukb/Q/envs/qbc/bin/python
"""Where every split-layout job stands: done, running, pending, partial, failed or todo.

    ./status.py                      per-dataset counts and totals
    ./status.py --list running       one line per running job: host, elapsed, expected hours
    ./status.py --list failed --why  ...and the tail of each failure's .err log
    ./status.py --todo               yaml paths not done and not live (what submit_runs.sh
                                     submits); failed and partial configs are included
    ./status.py --no-lsf             from the result files alone, without asking bjobs
    watch -n 120 ./status.py         a live board

The unit is one YAML under runs/<dataset>/, i.e. one (dataset, embedding, model) job. The
states, decided in this order:

  done     a run directory holds a ModelResults.csv row for every expected (embedding,
           iteration, model), i.e. `iter` rows for a split config
  running  LSF says RUN (or suspended) for the job named p10_<config>
  pending  LSF says PEND
  partial  not live, and the best run directory has some rows but not all: killed at the
           wall, or died mid-way. Resubmitting reruns it from iteration 1 in a new run
           directory; collate_results.py picks the complete one.
  failed   not live, no rows at all, and there is evidence of an attempt (an LSF log)
  todo     never attempted

Done is read from the result files, never from LSF: a DONE job whose ModelResults.csv is
short is not done, and qprofiler exits 0 after logging some failures.
"""
import argparse
import ast
import csv
import functools
import glob
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
#: submit_runs.sh names every job this, so bjobs can be joined back to its YAML.
JOB_PREFIX = "p10_"
STATES = ("done", "running", "pending", "partial", "failed", "todo")
LIVE = {"RUN": "running", "USUSP": "running", "SSUSP": "running",
        "PEND": "pending", "PSUSP": "pending"}


def is_job(path):
    """A job config -- not the _protocol.yaml that a directory's job files pull in."""
    return not os.path.basename(path).startswith("_")


def find_configs(runs_dir):
    """Every job config under runs_dir, sorted."""
    return sorted(p for p in glob.glob(os.path.join(runs_dir, "*", "*.yaml")) if is_job(p))


@functools.lru_cache(maxsize=None)
def _text(path):
    with open(path) as fh:
        return fh.read()


def _layers(path):
    """The texts hydra merges into this config, lowest precedence first: whatever its
    defaults list names (a composed job's _protocol.yaml), then the file itself.

    Read in the one form generate_pilot_configs.py writes -- a block list of names in the
    same directory, and _self_ -- which is enough for the few keys status reads.
    """
    text = _text(path)
    block = re.search(r"^defaults:[ \t]*\n((?:[ \t]+-.*(?:\n|$))+)", text, re.M)
    if not block:
        return [text]
    names = re.findall(r"^[ \t]+-[ \t]*([^\s#]+)", block.group(1), re.M)
    layers = [text if n == "_self_" else _text(os.path.join(os.path.dirname(path), n + ".yaml"))
              for n in names]
    return layers if "_self_" in names else layers + [text]


def _value(layers, key):
    """The text of ``key``'s value in the layer that wins the merge: the last to set it."""
    for text in reversed(layers):
        m = re.search(rf"^{key}:[ \t]*(.*)$", text, re.M)
        if m:
            return m.group(1)
    return ""


def _yaml_list(layers, key):
    m = re.match(r"\[.*?\]", _value(layers, key))
    return ast.literal_eval(m.group(0)) if m else []


def read_config(path):
    """The fields status needs, read by regex: 208 full YAML parses would dominate a watch.

    A composed job file sets only its job keys; iter comes from the _protocol.yaml its
    defaults list names, read once per directory.
    """
    layers = _layers(path)
    it = re.match(r"\d+", _value(layers, "iter"))
    n_iter = int(it.group(0)) if it else 0
    n_emb = max(1, len(_yaml_list(layers, "embeddings")))
    n_models = len(_yaml_list(layers, "classical_model")) + len(_yaml_list(layers, "quantum_model"))
    name = os.path.splitext(os.path.basename(path))[0]
    return {"yaml": os.path.abspath(path), "config": name,
            "dataset": os.path.basename(os.path.dirname(os.path.abspath(path))),
            "iter": n_iter, "expected": n_iter * n_emb * n_models}


def run_dirs(cfg):
    """Every hydra run directory this config has produced, oldest first."""
    root = os.path.join(os.path.dirname(cfg["yaml"]), "results", cfg["config"])
    return sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))


def count_rows(csv_path):
    """Distinct (embedding, iteration, model) rows in one ModelResults.csv.

    Distinct, not raw lines, so a row written twice cannot make a short run look complete.
    The csv module, not pandas: BestParams_Tuned is a quoted dict full of commas, which
    csv handles, and importing pandas would triple the time of a status call.
    """
    try:
        with open(csv_path, newline="") as fh:
            rows = csv.DictReader(fh)
            return len({(r.get("embeddings"), r.get("iteration"), r.get("model")) for r in rows})
    except (FileNotFoundError, csv.Error):
        return 0


def best_run(cfg):
    """(run_dir, rows) of the run that counts: the latest complete one, else the fullest."""
    runs = [(d, count_rows(os.path.join(d, "ModelResults.csv"))) for d in run_dirs(cfg)]
    complete = [r for r in runs if cfg["expected"] and r[1] >= cfg["expected"]]
    if complete:
        return complete[-1]
    # max() keeps the first maximum it meets, so scanning newest-first breaks ties to the
    # latest attempt.
    return max(reversed(runs), key=lambda r: r[1]) if runs else (None, 0)


def lsf_jobs():
    """{config: (jobid, stat, host, run_seconds)} for this user's p10_ jobs, latest wins."""
    try:
        out = subprocess.run(
            ["bjobs", "-a", "-noheader", "-o",
             "jobid stat job_name exec_host run_time delimiter='|'"],
            capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"!! bjobs unavailable ({exc}); live states unknown, use --no-lsf to silence",
              file=sys.stderr)
        return {}
    jobs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[2].startswith(JOB_PREFIX):
            continue
        jobid, stat, name, host, rt = parts[:5]
        secs = int(re.match(r"\s*(\d+)", rt).group(1)) if re.match(r"\s*\d+", rt) else 0
        host = host.split(":")[0].split("*")[-1] if host not in ("", "-") else ""
        cfg = name[len(JOB_PREFIX):]
        # Job ids grow, so the highest one is the latest attempt.
        if cfg not in jobs or int(jobid) > int(jobs[cfg][0]):
            jobs[cfg] = (jobid, stat, host, secs)
    return jobs


def latest_log(cfg, ext):
    logs = glob.glob(os.path.join(os.path.dirname(cfg["yaml"]), "lsf_logs",
                                  f"{cfg['config']}.*.{ext}"))
    # By job id, not mtime: the id is in the name and is monotonic.
    def jid(p):
        m = re.search(r"\.(\d+)\." + ext + "$", p)
        return int(m.group(1)) if m else -1
    return max(logs, key=jid) if logs else None


def classify(cfg, jobs):
    run_dir, rows = best_run(cfg)
    job = jobs.get(cfg["config"])
    cfg.update(run_dir=run_dir, rows=rows, job=job)
    if cfg["expected"] and rows >= cfg["expected"]:
        return "done"
    if job and job[1] in LIVE:
        return LIVE[job[1]]
    if rows > 0:
        return "partial"
    if latest_log(cfg, "err") or latest_log(cfg, "out") or (job and job[1] == "EXIT") \
            or run_dir is not None:
        return "failed"
    return "todo"


def load_manifest(runs_dir):
    path = os.path.join(runs_dir, "MANIFEST.tsv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {r["config"]: r for r in csv.DictReader(fh, delimiter="\t")}


def why(cfg, n=12):
    out = []
    o = latest_log(cfg, "out")
    if o:
        with open(o, errors="replace") as fh:
            term = [l.strip() for l in fh if l.startswith(("TERM_", "Exited", "Successfully"))]
        out += [f"    {os.path.basename(o)}: {t}" for t in term[-2:]]
    e = latest_log(cfg, "err")
    if e:
        with open(e, errors="replace") as fh:
            tail = fh.readlines()[-n:]
        out += [f"    | {l.rstrip()}" for l in tail]
    return out or ["    (no LSF log)"]


def main():
    # `./status.py --list all | head` should stop quietly, not with a BrokenPipe traceback.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("yamls", nargs="*", help="restrict to these configs (default: all)")
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--list", choices=STATES + ("all",), help="one line per job in this state")
    ap.add_argument("--todo", action="store_true",
                    help="print the yaml path of every config that is not done and not live")
    ap.add_argument("--why", action="store_true", help="with --list: show the LSF log tail")
    ap.add_argument("--no-lsf", action="store_true", help="do not ask bjobs")
    args = ap.parse_args()

    # A shell glob such as runs/heart/*.yaml also picks up the directory's _protocol.yaml.
    paths = [p for p in args.yamls if is_job(p)] if args.yamls else find_configs(args.runs_dir)
    if not paths:
        sys.exit(f"no configs under {args.runs_dir}; run generate_pilot_configs.py --layout split")
    jobs = {} if args.no_lsf else lsf_jobs()
    manifest = load_manifest(args.runs_dir)
    cfgs = [read_config(p) for p in paths]
    for c in cfgs:
        c["state"] = classify(c, jobs)

    if args.todo:
        for c in cfgs:
            if c["state"] in ("todo", "failed", "partial"):
                print(c["yaml"])
        return

    by_ds = defaultdict(Counter)
    for c in cfgs:
        by_ds[c["dataset"]][c["state"]] += 1
    width = max(len(d) for d in by_ds) + 2
    print(time.strftime("%Y-%m-%d %H:%M:%S") + ("   (no LSF)" if args.no_lsf else ""))
    print(f"{'dataset':{width}s}{'total':>6s}" + "".join(f"{s:>9s}" for s in STATES))
    tot = Counter()
    for ds in sorted(by_ds):
        cnt = by_ds[ds]
        tot.update(cnt)
        print(f"{ds:{width}s}{sum(cnt.values()):6d}" + "".join(f"{cnt[s]:9d}" for s in STATES))
    print(f"{'TOTAL':{width}s}{sum(tot.values()):6d}" + "".join(f"{tot[s]:9d}" for s in STATES))
    landed = sum(min(c["rows"], c["expected"]) for c in cfgs)
    need = sum(c["expected"] for c in cfgs)
    print(f"result rows landed: {landed}/{need} ({100.0 * landed / max(need, 1):.1f}%)   "
          f"processed configs: {tot['done']}/{len(cfgs)}")

    if args.list:
        print()
        for c in cfgs:
            if args.list != "all" and c["state"] != args.list:
                continue
            m = manifest.get(c["config"], {})
            job = c["job"]
            host, el = (job[2], job[3] / 3600.0) if job else ("", 0.0)
            exp_h, bound_h = m.get("exp_h") or "", m.get("bound_h") or ""
            flag = ""
            if c["state"] == "running" and bound_h and el > float(bound_h):
                flag = "  !! past its frozen-winner bound"
            print(f"{c['config']:45s} {c['state']:8s} rows {c['rows']}/{c['expected']:<3d} "
                  f"{('job ' + job[0]) if job else '':12s} {host:12s} "
                  f"{('%.2fh' % el) if job else '':>7s} exp {exp_h or '-':>6s} "
                  f"bound {bound_h or '-':>6s}{flag}")
            if args.why and c["state"] in ("failed", "partial", "running"):
                print("\n".join(why(c)))


if __name__ == "__main__":
    main()
