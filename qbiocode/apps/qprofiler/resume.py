# Copyright 2026, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adopt the results an earlier run of this config already computed.

Why
---
A job killed at its LSF wall keeps every pass that finished: ``ModelResults.csv`` is
appended to as each model returns. But a resubmit starts a *new* hydra run directory and
recomputes from the first split, so the work in the old directory is thrown away.
``status.py`` calls that state ``partial``, and ``collate_results.py`` takes the fullest
single run directory per config -- so a config that needs two walls to finish never
finishes, however often it is resubmitted. On the quantum arms one pass is hours, which
makes this the difference between a run that converges and one that does not.

With ``skip_existing`` set, a run reads the ``ModelResults.csv`` of the earlier run
directories of the same config, and for every (embedding, split, model) cell they already
hold it:

* does not fit that model again;
* copies the earlier row into this run's ``ModelResults.csv`` **verbatim**, so this run
  directory is the complete one and ``status.py`` can call the config ``done``;
* copies that model's ``oof/``, ``trials/`` and ``val_predictions/`` rows across, so the
  sidecars stay complete too;
* records where each adopted row came from in ``adopted.csv``.

A pass whose every model is already present is skipped whole -- no embedding is read, no
complexity measures are recomputed, no model is fitted. A pass that is missing some of its
models runs only those, and the adopted rows are appended beside the fresh ones.

The rows are copied as text, not re-serialised through pandas, so an adopted row is byte
for byte the row the earlier run wrote.

What it does NOT check
----------------------
That the earlier run used the same settings. The only thing tying a run directory to this
config is that it sits under ``results/<config_file_name>/``, so editing a config between
two runs and resuming across the edit mixes two protocols in one table, silently. Under
``split_mode: manifest`` the rows carry ``dataset_sha256`` and ``manifest_sha256`` and
those *are* checked (:func:`scan`'s ``require``), which covers the case that actually
bites -- a dataset or its frozen splits changing underneath a resume. Internal mode has
nothing comparable to check, so delete the old run directories rather than resume when the
config has changed.

``results.pkl`` of a fully adopted pass is carried over from the earlier run when it holds
one; a *partially* adopted pass contributes only the summary of the models that ran here,
because the earlier run's summary of that pass describes a different set of models.
"""

import csv
import glob
import os
import pickle
import socket

#: The per-model results table a run writes into its own run directory.
RESULTS_CSV = "ModelResults.csv"
#: Where the summaries of every pass are accumulated.
RESULTS_PKL = "results.pkl"
#: The provenance file this module writes: one row per adopted result row.
ADOPTED_CSV = "adopted.csv"
ADOPTED_COLUMNS = ("Dataset", "embeddings", "iteration", "model", "source_run_dir")

#: What identifies one result row, and the key of the index built below. It is
#: ``collate_results.KEY``, which is the same tuple the duplicate check there rejects --
#: so a row this module adopts is exactly a row that would otherwise collide.
KEY_COLUMNS = ("Dataset", "embeddings", "iteration", "model")

#: Where an earlier ``ModelResults.csv`` is looked for under the resolved root: the root
#: itself, and one or two directory levels below it. Two levels because the packaged
#: config's run directory is ``results/<config>/dataset=<csv>/<backend>_<stamp>``, so a
#: root given as ``results/<config>`` still reaches the run directories.
_DEPTHS = ("", "*", os.path.join("*", "*"))

#: Values of ``skip_existing`` that mean "off" and "the earlier runs of this config".
_OFF = ("", "false", "no", "0", "none", "off")
_ON = ("true", "yes", "1", "on")


class ResumeError(ValueError):
    """``skip_existing`` names something that cannot be resumed from."""


def resolve_root(value, run_dir=None):
    """The directory holding the earlier run directories, or None when resuming is off.

    Args:
        value: the ``skip_existing`` config value. False, None or an "off" spelling
            disables it. True (or an "on" spelling) means the parent of ``run_dir``,
            i.e. the other run directories of this config. A string path names that
            directory explicitly and must be absolute, for the reason
            ``embedding_cache`` must be: hydra runs every job from its own output
            directory, so a relative path names a different place in every job.
        run_dir: the current run directory; the working directory by default, which is
            where hydra puts a job.

    Returns:
        str or None: an absolute directory, not necessarily one that exists yet -- the
        first run of a config has no earlier runs, which is not an error.

    Raises:
        ResumeError: for a relative path, or a value that is neither a flag nor a path.
    """
    if value is None or value is False:
        return None
    run_dir = os.path.abspath(run_dir if run_dir is not None else os.getcwd())
    if value is True:
        return os.path.dirname(run_dir)
    if not isinstance(value, str):
        raise ResumeError(
            f"skip_existing is true (resume from the earlier runs of this config), false, "
            f"or the directory holding them; got {value!r}."
        )
    text = value.strip()
    if text.lower() in _OFF:
        return None
    if text.lower() in _ON:
        return os.path.dirname(run_dir)
    path = os.path.expanduser(text)
    if not os.path.isabs(path):
        raise ResumeError(
            f"skip_existing must be true, false, or an ABSOLUTE directory; got {value!r}. "
            f"Hydra runs every job from its own output directory, so a relative path would "
            f"name a different directory in each job."
        )
    return os.path.normpath(path)


def _as_iteration(value):
    """The ``iteration`` cell as an int, or None when it is not one.

    Read loosely on purpose: the column is written by ``str()`` of a Python int, but a
    file that has been through a spreadsheet carries '3.0'.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        return int(number) if number.is_integer() else None


def row_key(row):
    """``(Dataset, embeddings, iteration, model)`` of one result row, or None.

    None when the row is missing one of them, which is what a row from some other tool
    looks like.
    """
    values = [row.get(column) for column in KEY_COLUMNS[:2]] + [
        _as_iteration(row.get("iteration")), row.get("model")
    ]
    if any(value is None or value == "" for value in values):
        return None
    return (str(values[0]), str(values[1]), values[2], str(values[3]))


class CompletedResults:
    """The result rows earlier runs already hold, keyed by :data:`KEY_COLUMNS`.

    Attributes:
        rows: {key: (row dict, the ModelResults.csv it was read from)}.
        sources: every CSV that was read, in the order it was read.
        rejected: {key: why it was not adopted} -- a row whose provenance columns
            disagree with this run's. Reported rather than dropped quietly: a resume
            that silently declines to adopt looks exactly like one with nothing to adopt.
    """

    def __init__(self):
        self.rows = {}
        self.sources = []
        self.rejected = {}

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.rows

    def get(self, dataset, embedding, iteration, model):
        """``(row, source csv)`` for one cell, or ``(None, None)``."""
        return self.rows.get((str(dataset), str(embedding), int(iteration), str(model)),
                             (None, None))

    def run_dir(self, dataset, embedding, iteration, model):
        """The run directory one cell was computed in, or None."""
        _, source = self.get(dataset, embedding, iteration, model)
        return os.path.dirname(source) if source else None


def scan(root, exclude=(), require=None):
    """Index every result row under ``root`` that this run could adopt.

    Args:
        root: the directory :func:`resolve_root` returned. A root that does not exist
            yields an empty index rather than an error: the first run of a config has
            no earlier runs.
        exclude: directories whose rows must be ignored -- the current run directory,
            which is a sibling of the others and already holds whatever this run wrote.
        require: {column: value} a row must agree with to be adopted. A row that does
            not carry the column is accepted (an older run wrote no such column); a row
            that carries a different value is rejected and recorded. Under
            ``split_mode: manifest`` this is ``dataset_sha256`` and ``manifest_sha256``,
            which is what catches a dataset or a frozen split changing under a resume.

    Returns:
        CompletedResults: later sources override earlier ones. The sources are sorted by
        path, and a run directory is named ``<backend>_%Y-%m-%d_%H-%M-%S``, so that
        ordering is chronological and the newest run of a cell wins.
    """
    found = CompletedResults()
    if not root or not os.path.isdir(root):
        return found
    excluded = {os.path.abspath(path) for path in exclude}
    candidates = []
    for depth in _DEPTHS:
        candidates += glob.glob(os.path.join(root, depth, RESULTS_CSV))
    for path in sorted(set(candidates)):
        if os.path.abspath(os.path.dirname(path)) in excluded:
            continue
        rows = _read_rows(path)
        if rows is None:
            continue
        found.sources.append(path)
        for row in rows:
            key = row_key(row)
            if key is None:
                continue
            reason = refused(row, require)
            if reason:
                found.rejected[key] = f"{reason} (in {path})"
                found.rows.pop(key, None)
                continue
            found.rejected.pop(key, None)
            found.rows[key] = (row, path)
    return found


def _read_rows(path):
    """Every row of one ModelResults.csv as a dict of the raw cell text, or None.

    None for a file that cannot be read as a CSV -- a run killed inside its first write
    leaves one. That is not a reason to refuse the resume; the rows simply are not there.
    """
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle, restval=""))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def refused(row, require):
    """Why ``row`` may not be adopted under ``require``, or None.

    A run with several datasets reads one index for all of them (the key carries the
    dataset) but has a different ``require`` per dataset, so this is applied at lookup
    as well as in :func:`scan`.
    """
    for column, wanted in (require or {}).items():
        if wanted in (None, ""):
            continue
        present = row.get(column)
        if present in (None, ""):
            continue
        if str(present) != str(wanted):
            return f"its {column} is {present!r}, this run's is {str(wanted)!r}"
    return None


def provenance_require(manifest, dataset_sha256):
    """The ``require`` mapping of a manifest-mode run: the bytes its rows were computed from.

    Args:
        manifest: the dataset's SplitManifest, or None outside manifest mode.
        dataset_sha256: sha256 of the dataset CSV, or None when the run did not hash it.

    Returns:
        dict: empty outside manifest mode, where the rows carry no such columns.
    """
    if manifest is None:
        return {}
    return {"dataset_sha256": dataset_sha256 or manifest.sha256,
            "manifest_sha256": manifest.file_sha256}


# ---------------------------------------------------------------------------
# Carrying one cell across: the row, its sidecar rows, and its pass summary.
# ---------------------------------------------------------------------------
def _write_rows_atomic(path, fieldnames, rows):
    """Replace ``path`` with ``rows`` through a temporary file and a rename."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def record_adopted(adopted, path=ADOPTED_CSV):
    """Append the provenance of the cells one pass adopted to ``adopted.csv``.

    Args:
        adopted: [{Dataset, embeddings, iteration, model, source_run_dir}] of this pass.
        path: where to write, relative to the run directory.
    """
    if not adopted:
        return
    existing = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        existing = _read_rows(path) or []
    _write_rows_atomic(path, list(ADOPTED_COLUMNS), existing + list(adopted))


def carry_sidecars(sources, data_key, models, directories, dest="."):
    """Merge the sidecar rows of adopted models into this run's sidecars.

    One pass writes one file per sidecar directory, ``<dir>/<data_key>.csv``, holding
    every model of the pass (see :mod:`qbiocode.evaluation.protocol`). A resumed pass
    writes only the models it fitted, so the models it adopted have to be copied in from
    the run directory each of them came from -- otherwise ``collate_results.py`` merges
    sidecars that are missing exactly the models this run did not repeat.

    Args:
        sources: {model label: the run directory it was computed in}.
        data_key: the pass's data_key, which names the sidecar file.
        models: the model labels to carry (the keys of ``sources`` to use).
        directories: the sidecar directory names, in writing order.
        dest: this run's directory.

    Returns:
        dict: {directory: rows carried}, for the log line.
    """
    carried = {}
    wanted = {str(model) for model in models}
    for directory in directories:
        target = os.path.join(dest, directory, f"{data_key}.csv")
        mine = _read_rows(target) or []
        fieldnames = _fieldnames(target) or []
        extra = []
        for model in sorted(wanted):
            source_dir = sources.get(model)
            if not source_dir:
                continue
            rows = _read_rows(os.path.join(source_dir, directory, f"{data_key}.csv"))
            if not rows:
                continue
            if not fieldnames:
                fieldnames = _fieldnames(os.path.join(source_dir, directory,
                                                      f"{data_key}.csv")) or []
            extra += [row for row in rows if str(row.get("model")) == model]
        if not extra:
            continue
        # The column names of whichever file was read; a sidecar's columns are fixed by
        # protocol.py, so the two agree, and 'config' (which collate_results.py inserts
        # into its own copy, never into these) is not among them.
        _write_rows_atomic(target, fieldnames, mine + extra)
        carried[directory] = len(extra)
    return carried


def _fieldnames(path):
    """The header of one CSV, or None if it has none."""
    try:
        with open(path, newline="") as handle:
            return next(csv.reader(handle), None)
    except OSError:
        return None


def carry_summary(source_run_dir, dataset, embedding, iteration):
    """The earlier run's ``results.pkl`` summary of one fully adopted pass, or None.

    Only a pass this run skips entirely may take its summary from the earlier run: the
    summary is per pass and names every model of it, so for a pass that ran some models
    here the two would describe different things.
    """
    path = os.path.join(source_run_dir, RESULTS_PKL)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            summaries = pickle.load(handle)
    except Exception:  # a truncated or foreign pickle is simply not a summary to carry
        return None
    if not isinstance(summaries, list):
        return None
    for summary in reversed(summaries):
        if not isinstance(summary, dict):
            continue
        if (str(summary.get("Dataset")) == str(dataset)
                and str(summary.get("embeddings")) == str(embedding)
                and _as_iteration(summary.get("iteration")) == int(iteration)):
            return summary
    return None
