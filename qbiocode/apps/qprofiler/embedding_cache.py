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

"""Embed each (dataset, embedding, split) once, and hand every job the same features.

Why
---
A QProfiler run embeds each split itself. That is only sound where the embedding is
reproducible, and seeded UMAP is reproducible on one CPU type but not across CPU types.
numba compiles its fastmath kernels for the host's instruction set, and the SGD epochs
amplify the last-bit differences. The 2026 pilot ran one LSF job per (dataset,
embedding, model) and the jobs landed on three host types. So the models of one
(dataset, embedding) were scored on two or three different sets of UMAP features, with
condition numbers up to 57% apart, and nothing reported it. PCA depends on BLAS and
LAPACK in the same way, but only in the last bits: its pilot features agreed across
hosts to 4e-15. It is cached all the same, so that no job's features depend on its host.

How
---
With ``embedding_cache: <absolute directory>`` in the config, ``qprofiler.main`` reads
every embedding except ``'none'`` from ``<directory>/emb_<data_key>.npz``. It never
computes one. Before fitting anything it checks that every file it will read exists and
was written under its own settings, and it stops if any is not. There is no fallback to
computing the embedding, because that fallback is the bug this cache removes. The files
are written beforehand, on one machine, by::

    python -m qbiocode.apps.qprofiler.embedding_cache CONFIG.yaml [CONFIG.yaml ...]

This reads each config as the job will. It reproduces the job's splits and scaling with
qprofiler's own functions. It writes each (dataset, embedding, split) once, however many
configs share it, and leaves files that are already correct alone.

A file holds the embedded train and test rows, the dataset rows each side came from, and
two JSON records. ``spec`` holds every input the embedding depends on, and each read
checks it. ``provenance`` records where, when and with which library versions the file
was computed, and is never checked.

``--check`` is the dry run: it lists every file the configs read, as ``current``,
``MISSING`` or ``STALE``, writes nothing, and exits 1 if anything would be written.
``--dry-run`` lists the same and writes nothing, but exits 0 unless a config is invalid,
for a preview of what a real run would write.

Under ``split_mode: manifest`` the splits are the dataset manifest's, and only those the
config's ``splits`` selects are written. Each split has two files: the final stage,
``emb_<data_key>.npz``, fitted on the outer training fold and applied to the test fold,
and the tuning stage, ``emb_<data_key>__tune.npz``, fitted on the fit rows and applied
to the validation rows (stored under the same array names). Their spec names the
manifest, the outer split, the stage and the embedding seed instead of the seed, test
size and stratification of a drawn split, so a file written in one split mode is stale
to a run in the other.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings

import numpy as np

#: The version of the file layout. It is part of the spec, so a file in another layout
#: is reported as stale rather than misread.
CACHE_VERSION = 1

#: How a missing file gets written. Every error quotes it.
PRECOMPUTE_COMMAND = "python -m qbiocode.apps.qprofiler.embedding_cache <config.yaml> ..."

#: The array names of a file. A tuning-stage file (``split_mode: manifest``) stores the
#: fit rows under the ``train`` names and the validation rows under the ``test`` names;
#: its spec's ``stage`` says which it is.
_ARRAYS = ("X_train", "X_test", "train_idx", "test_idx")

#: Appended to a pass's data_key to name its tuning-stage file, ``emb_<data_key>__tune.npz``.
TUNE_SUFFIX = "__tune"
_ABSENT = "<absent>"

log = logging.getLogger(__name__)


class EmbeddingCacheError(ValueError):
    """The cache cannot serve a run: a file it needs is missing, stale or unreadable."""


def cache_file(cache_dir, data_key):
    """Where the embedding of one (dataset, embedding, split) pass is stored."""
    return os.path.join(cache_dir, f"emb_{data_key}.npz")


def tune_key(data_key):
    """The cache key of a pass's tuning stage: its fit and validation rows, embedded with
    the embedding fitted on the fit rows (``split_mode: manifest`` only)."""
    return f"{data_key}{TUNE_SUFFIX}"


def file_sha256(path):
    """The hex sha256 of a file's bytes, which is how a spec identifies a dataset."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def features_digest(*arrays):
    """16 hex digits of a sha256 over each array's dtype, shape and values.

    qprofiler logs it for every pass, cached or not. Two jobs that used the same
    features log the same digest, so their logs alone show whether they did.
    """
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        if array.dtype == object:
            digest.update(repr(array.tolist()).encode())
        else:
            digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _plain(value):
    """``value`` with OmegaConf containers turned into dicts and lists, for JSON."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf ships with hydra
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _canonical(record):
    """``record`` as it reads back from JSON, so a spec compares equal to its stored copy."""
    return json.loads(json.dumps(record, sort_keys=True))


def embedding_spec(args, dataset, dataset_sha256, embedding, iteration, scaler_name,
                   manifest=None, stage="final"):
    """Every input that determines the embedding of one (dataset, embedding, split).

    A file serves a run exactly when its stored spec equals the run's. The model, the
    backend and the output paths are left out on purpose: they are what differ between
    the jobs that share one file.

    Under ``split_mode: manifest`` (``manifest`` given) the split is the manifest's, so
    the spec names the manifest, the outer split and the stage instead of the seed,
    test size and stratification a drawn split depends on. The two layouts share no
    spec, so a file written in one mode is stale to a run in the other.

    Args:
        args: the run config (a dict or an OmegaConf DictConfig).
        dataset (str): the CSV's file name, as ``file_dataset`` lists it.
        dataset_sha256 (str): :func:`file_sha256` of that CSV.
        embedding (str): the embedding name, as the config writes it.
        iteration (int): the split, 1-based (the global iteration in manifest mode).
        scaler_name (str): what ``_validate_config`` resolved ``scaling`` to.
        manifest: the dataset's :class:`~qbiocode.apps.qprofiler.split_manifest.SplitManifest`
            under ``split_mode: manifest``; None otherwise.
        stage (str): manifest mode only. ``'final'``: fitted on the outer training
            fold, applied to the test fold. ``'tune'``: fitted on the fit rows, applied
            to the validation rows.
    """
    from qbiocode.apps.qprofiler import qprofiler as qp

    if manifest is not None:
        if stage not in ("final", "tune"):
            raise ValueError(f"stage must be 'final' or 'tune'; got {stage!r}")
        split = manifest.split(iteration)
        return _canonical({
            "cache_version": CACHE_VERSION,
            "dataset": dataset,
            "dataset_sha256": dataset_sha256,
            "index_col": bool(args.get("index_col", False)),
            # The split: which manifest, which outer split, which side of it.
            "split_mode": "manifest",
            "manifest_sha256": manifest.file_sha256,
            "iteration": int(iteration),
            "repeat": int(split.repeat),
            "fold": int(split.fold),
            "stage": stage,
            "embed_seed": int(qp._split_seed(args, iteration)),
            "scaling": scaler_name,
            "embedding": embedding,
            **{key: _plain(value) for key, value in qp._embedding_settings(args).items()},
        })
    return _canonical({
        "cache_version": CACHE_VERSION,
        # The rows: which file, byte for byte, and how it is parsed.
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "index_col": bool(args.get("index_col", False)),
        # The split, and the scaling fitted on its training side.
        "iteration": int(iteration),
        "split_seed": int(qp._split_seed(args, iteration)),
        "test_size": float(args["test_size"]),
        "stratify": qp._is_stratified(args),
        "scaling": scaler_name,
        # The embedding: its name and everything _embed passes it apart from the data
        # and the seed, which is the split seed above.
        "embedding": embedding,
        **{key: _plain(value) for key, value in qp._embedding_settings(args).items()},
    })


def plan(dataset, dataset_sha256, n_features, args, scaler_name, manifest=None):
    """``[(key, spec)]``, one pair for each embedding a run reads for ``dataset``.

    That is every split, crossed with every embedding except ``'none'`` that
    ``resolve_embeddings`` keeps at this feature count. main's embedding loop makes the
    same choice. The key is the pass's data_key.

    Under ``split_mode: manifest`` (``manifest`` given) the splits are the ones the
    config's ``splits`` selects, and each has two entries: the final stage under its
    data_key and the tuning stage under :func:`tune_key` of it.

    Raises:
        ValueError: under ``split_mode: manifest``, for a transductive embedding.
    """
    from qbiocode import resolve_embeddings
    from qbiocode.apps.qprofiler import qprofiler as qp
    from qbiocode.embeddings import DEFAULT_EMBEDDING_MIN_FEATURES, is_transductive

    embeddings, _ = resolve_embeddings(
        args["embeddings"],
        n_features,
        min_features=args.get("embedding_min_features", DEFAULT_EMBEDDING_MIN_FEATURES),
    )
    if manifest is not None:
        # qprofiler._validate_config refuses these first; this keeps plan safe on its own.
        transductive = [e for e in embeddings if is_transductive(e)]
        if transductive:
            raise ValueError(
                f"split_mode: manifest needs inductive embeddings; {transductive} would "
                f"be fit on the validation rows at the tuning stage."
            )
        entries = []
        for split in manifest.select(qp._split_selection(args)):
            for embed in embeddings:
                if embed == "none":
                    continue
                data_key = qp._data_key(dataset, embed, args["n_components"], split.iteration)
                for stage, key in (("final", data_key), ("tune", tune_key(data_key))):
                    entries.append((key, embedding_spec(
                        args, dataset, dataset_sha256, embed, split.iteration, scaler_name,
                        manifest=manifest, stage=stage,
                    )))
        return entries
    return [
        (
            qp._data_key(dataset, embed, args["n_components"], iteration),
            embedding_spec(args, dataset, dataset_sha256, embed, iteration, scaler_name),
        )
        for iteration in range(1, args["iter"] + 1)
        for embed in embeddings
        if embed != "none"
    ]


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def provenance(config_path=None):
    """Where, when and with which library versions an entry was computed.

    The record is stored so that a disagreement can be traced. It is never compared:
    every job reads the same file, so the machine that wrote it does not matter.
    """
    versions = {}
    for dist in ("qbiocode", "numpy", "scipy", "scikit-learn", "umap-learn",
                 "pynndescent", "numba", "llvmlite"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        # What numba compiles for. "host" is LLVM's name for this machine's own CPU, which
        # "cpu" identifies. Asking llvmlite for the exact name would import a package that
        # qbiocode does not declare (tests/test_quvine_packaging.py).
        "numba_target": os.environ.get("NUMBA_CPU_NAME") or "host",
        "python": platform.python_version(),
        "versions": versions,
    }
    if config_path:
        record["written_by"] = os.path.abspath(config_path)
    return record


def write(path, X_train_emb, X_test_emb, train_idx, test_idx, spec, provenance_record=None):
    """Write one entry atomically.

    A reader sees either the previous file or this one, never part of either. That
    matters because running jobs may be reading the directory. The file goes to a
    temporary name first and is renamed into place.
    """
    arrays = {"X_train": np.asarray(X_train_emb), "X_test": np.asarray(X_test_emb)}
    for name, array in arrays.items():
        if array.dtype.kind not in "biufc":
            raise EmbeddingCacheError(
                f"{name} for {os.path.basename(path)} has dtype {array.dtype}. The cache "
                f"stores numeric arrays only, so reading one never needs pickle."
            )
    arrays["train_idx"] = np.asarray(train_idx, dtype=np.int64)
    arrays["test_idx"] = np.asarray(test_idx, dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                **arrays,
                spec=np.array(json.dumps(spec, sort_keys=True)),
                provenance=np.array(json.dumps(provenance_record or {}, sort_keys=True)),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _open(path):
    if not os.path.exists(path):
        raise EmbeddingCacheError(f"{path} does not exist")
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise EmbeddingCacheError(
            f"{path} is not a readable cache file ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(npz, np.lib.npyio.NpzFile):
        # np.load opens a bare .npy array too, whatever the file is called.
        raise EmbeddingCacheError(
            f"{path} is not a readable cache file: it holds a single array, not an archive"
        )
    return npz


def _member(npz, path, name):
    try:
        return npz[name]
    except Exception as exc:
        raise EmbeddingCacheError(
            f"{path} is not a readable cache file: cannot read {name!r} "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _record(npz, path, name):
    try:
        return json.loads(str(_member(npz, path, name)))
    except json.JSONDecodeError as exc:
        raise EmbeddingCacheError(
            f"{path} is not a readable cache file: its {name!r} is not JSON ({exc})"
        ) from exc


def read_spec(path):
    """The spec stored in one file. Its arrays are not read."""
    with _open(path) as npz:
        return _record(npz, path, "spec")


def read(path):
    """Everything in one file: the four arrays, and the ``spec`` and ``provenance`` records."""
    with _open(path) as npz:
        entry = {name: _member(npz, path, name) for name in _ARRAYS}
        entry["spec"] = _record(npz, path, "spec")
        entry["provenance"] = _record(npz, path, "provenance")
    return entry


def spec_differences(stored, wanted):
    """``[(key, stored value, wanted value)]`` for each key on which two specs differ."""
    return [
        (key, stored.get(key, _ABSENT), wanted.get(key, _ABSENT))
        for key in sorted(set(stored) | set(wanted))
        if stored.get(key, _ABSENT) != wanted.get(key, _ABSENT)
    ]


def _stale(path, stored, wanted):
    """How a file's spec differs from the one a run needs, or None if it does not."""
    differences = spec_differences(stored, wanted)
    if not differences:
        return None
    detail = "; ".join(
        f"{key}: the file has {old!r}, this config needs {new!r}"
        for key, old, new in differences
    )
    return f"{path} was written under other settings ({detail})"


def _same_rows(entry, train_idx, test_idx):
    return (
        np.array_equal(entry["train_idx"], train_idx)
        and np.array_equal(entry["test_idx"], test_idx)
        and len(entry["X_train"]) == len(train_idx)
        and len(entry["X_test"]) == len(test_idx)
    )


def check(cache_dir, entries):
    """One message for each entry of :func:`plan` that ``cache_dir`` cannot serve.

    An entry fails if its file is missing, unreadable, or written under other settings.
    The list is empty when the run can start.
    """
    problems = []
    for data_key, spec in entries:
        path = cache_file(cache_dir, data_key)
        try:
            problem = _stale(path, read_spec(path), spec)
        except EmbeddingCacheError as exc:
            problem = str(exc)
        if problem:
            problems.append(problem)
    return problems


def require(cache_dir, entries):
    """Raise unless ``cache_dir`` holds every entry of :func:`plan`, written for this run.

    Raises:
        EmbeddingCacheError: listing each file that is missing, unreadable or stale.
    """
    problems = check(cache_dir, entries)
    if not problems:
        return
    shown = "\n  ".join(problems[:8])
    more = f"\n  ... and {len(problems) - 8} more" if len(problems) > 8 else ""
    raise EmbeddingCacheError(
        f"embedding_cache {cache_dir} cannot serve this run: {len(problems)} of the "
        f"{len(entries)} embedded splits it reads are missing or stale.\n  {shown}{more}\n"
        f"The run reads embeddings from the cache and never computes them, so that every "
        f"job of an experiment uses the same features. Write the missing files with\n"
        f"    {PRECOMPUTE_COMMAND}\n"
        f"Add --force to replace files written under other settings. Only do that if "
        f"no other config still reads them."
    )


def load(cache_dir, data_key, spec, train_idx, test_idx):
    """``(X_train_emb, X_test_emb)`` for one pass, after checking the file belongs to it.

    The file must carry the same spec, and must have been embedded from the same rows on
    each side of the split.

    Raises:
        EmbeddingCacheError: if the file is missing, unreadable, stale, or from other rows.
    """
    path = cache_file(cache_dir, data_key)
    try:
        entry = read(path)
    except EmbeddingCacheError as exc:
        raise EmbeddingCacheError(f"{exc}. Write it with\n    {PRECOMPUTE_COMMAND}") from exc
    problem = _stale(path, entry["spec"], spec)
    if problem:
        raise EmbeddingCacheError(
            f"{problem}. Rewrite it with --force added to\n    {PRECOMPUTE_COMMAND}"
        )
    if not _same_rows(entry, train_idx, test_idx):
        if "split_mode" in spec:
            # Manifest mode: the spec pins the manifest's bytes, so the rows cannot have
            # changed with it; the file does not hold what its spec says.
            cause = "The manifest is the same, so the file itself is damaged"
        else:
            cause = "So the split itself has changed, probably with the scikit-learn version"
        raise EmbeddingCacheError(
            f"{path} was embedded from other rows than this run's split "
            f"{spec['iteration']}, although its settings match. {cause}. Rewrite the file "
            f"with --force added to\n    {PRECOMPUTE_COMMAND}"
        )
    return entry["X_train"], entry["X_test"]


def _status(path, spec, train_idx, test_idx):
    """``('current' | 'missing' | 'stale', reason)`` of one file, for the precompute."""
    if not os.path.exists(path):
        return "missing", f"{path} does not exist"
    try:
        entry = read(path)
    except EmbeddingCacheError as exc:
        return "stale", str(exc)
    problem = _stale(path, entry["spec"], spec)
    if problem:
        return "stale", problem
    if not _same_rows(entry, train_idx, test_idx):
        return "stale", f"{path} was embedded from other rows than split {spec['iteration']}"
    return "current", None


def _compose(config_path):
    """The config that a job started with ``--config-dir`` and ``--config-name`` sees."""
    from hydra import compose, initialize_config_dir

    directory, name = os.path.split(os.path.abspath(config_path))
    with warnings.catch_warnings():
        # hydra's notices about version_base and _self_. The job sees them too.
        warnings.simplefilter("ignore")
        with initialize_config_dir(config_dir=directory, version_base="1.1"):
            return compose(config_name=os.path.splitext(name)[0])


def _dataset(loaded, path, index_col):
    """``(X, y_encoded, sha256)`` of one CSV, read once however many configs name it."""
    from qbiocode.apps.qprofiler import qprofiler as qp

    key = (os.path.realpath(path), bool(index_col))
    if key not in loaded:
        X, _, y_encoded = qp._read_dataset(path, {"index_col": bool(index_col)})
        loaded[key] = (X, y_encoded, file_sha256(path))
    return loaded[key]


def precompute(config_paths, check_only=False, force=False, out=print):
    """Write every embedding that ``config_paths`` read from their ``embedding_cache``.

    With ``check_only`` it writes nothing and reports what is missing or stale.

    A ``split_mode: manifest`` config has its datasets checked against their manifests
    first, as the job does, and gets both stages of each split its ``splits`` selects.

    A file that already holds an entry's spec, embedded from the same rows, is left
    alone, because that is what running jobs already read. A file written under other
    settings is reported, not replaced, unless ``force`` is set. Two configs that need
    different contents in one file are a conflict in every mode.

    Args:
        config_paths: job YAMLs. Names starting with ``_`` are shared protocols and are
            skipped. So is a config without ``embedding_cache``.
        check_only (bool): report only.
        force (bool): replace stale files.
        out: where each line of the report goes.

    Returns:
        int: the number of problems. Zero means every job can start.
    """
    from qiskit_algorithms.utils import algorithm_globals

    from qbiocode.apps.qprofiler import qprofiler as qp

    loaded = {}
    claimed = {}  # path -> (spec, config): what this call has already decided for it
    written = current = problems = 0
    for config_path in config_paths:
        if os.path.basename(config_path).startswith("_"):
            out(f"skip     {config_path}: a _-prefixed config is a shared protocol, not a job")
            continue
        args = _compose(config_path)
        qp._resolve_model_lists(args, log)
        scaler_name = qp._validate_config(args, log)
        cache_dir = qp._embedding_cache_dir(args)
        if cache_dir is None:
            out(f"skip     {config_path}: no embedding_cache, so its run embeds for itself")
            continue
        path_to_input = qp._input_folder(args)
        manifest_mode = qp._split_mode(args) == "manifest"
        for file in qp._input_files(args, path_to_input):
            dataset_path = os.path.join(path_to_input, file)
            X, y_encoded, sha256 = _dataset(loaded, dataset_path, args.get("index_col", False))
            # Under split_mode: manifest the rows come from the dataset's manifest, which
            # is checked against the CSV here exactly as the job will check it.
            manifest = (
                qp._dataset_manifest(args, dataset_path, len(X), y_encoded, sha256)
                if manifest_mode else None
            )
            todo = []
            for data_key, spec in plan(file, sha256, X.shape[1], args, scaler_name,
                                       manifest=manifest):
                path = cache_file(cache_dir, data_key)
                if path in claimed:
                    other_spec, other_config = claimed[path]
                    if other_spec != spec:
                        problems += 1
                        detail = "; ".join(
                            f"{key}: {old!r} vs {new!r}"
                            for key, old, new in spec_differences(other_spec, spec)
                        )
                        out(f"CONFLICT {os.path.basename(path)}: {other_config} and "
                            f"{config_path} need different contents ({detail})")
                    continue
                claimed[path] = (spec, config_path)
                todo.append((spec, path))
            if not todo:
                continue
            # Seeded as main seeds each dataset, for any embedding that reads the global
            # stream rather than its random_state.
            np.random.seed(args["seed"])
            algorithm_globals.random_seed = args["q_seed"]
            # One (split, stage) at a time. Internal-mode specs have no stage: their one
            # file per split is the final stage.
            for iteration, stage in sorted(
                {(spec["iteration"], spec.get("stage", "final")) for spec, _ in todo}
            ):
                X_train, X_test, _, _, train_idx, test_idx = qp._split_and_scale(
                    X, y_encoded, args, iteration, scaler_name,
                    split=manifest.split(iteration) if manifest is not None else None,
                    stage=stage,
                )
                for spec, path in todo:
                    if (spec["iteration"], spec.get("stage", "final")) != (iteration, stage):
                        continue
                    name = os.path.basename(path)
                    status, reason = _status(path, spec, train_idx, test_idx)
                    if status == "current":
                        current += 1
                        out(f"current  {name}")
                        continue
                    if check_only or (status == "stale" and not force):
                        problems += 1
                        out(f"{status.upper():8s} {name}: {reason}")
                        continue
                    started = time.time()
                    X_train_emb, X_test_emb = qp._embed(
                        spec["embedding"], X_train, X_test, args,
                        qp._split_seed(args, iteration),
                    )
                    write(path, X_train_emb, X_test_emb, train_idx, test_idx, spec,
                          provenance(config_path))
                    written += 1
                    out(f"wrote    {name}  {len(X_train_emb)}+{len(X_test_emb)} rows x "
                        f"{X_train_emb.shape[1]} {X_train_emb.dtype}, sha256 "
                        f"{features_digest(X_train_emb, X_test_emb)}, "
                        f"{time.time() - started:.1f} s"
                        + (f" (replaced: {reason})" if status == "stale" else ""))
    out(f"{written} written, {current} already current, {problems} "
        f"problem{'' if problems == 1 else 's'}")
    return problems


def main(argv=None):
    """The command line: ``python -m qbiocode.apps.qprofiler.embedding_cache``."""
    parser = argparse.ArgumentParser(
        prog="python -m qbiocode.apps.qprofiler.embedding_cache",
        description=(
            "Write the embedded features that QProfiler configs with embedding_cache "
            "read, one file per (dataset, embedding, split), two under split_mode: "
            "manifest (final and tuning stage). Files that are already current are kept."
        ),
    )
    parser.add_argument(
        "configs", nargs="+", metavar="CONFIG.yaml",
        help="job configs, as given to qprofiler; _-prefixed names are skipped",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true",
        help="the dry run: list each file's status and write nothing; exit 1 if any "
             "file is missing or stale",
    )
    mode.add_argument(
        "--dry-run", action="store_true",
        help="list what would be written, as --check does, and write nothing; exit 0 "
             "unless a config is invalid",
    )
    mode.add_argument(
        "--force", action="store_true",
        help="replace files written under other settings instead of reporting them",
    )
    options = parser.parse_args(argv)
    for path in options.configs:
        if not os.path.isfile(path):
            parser.error(f"no such config: {path}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Seeded UMAP says it runs single-threaded once per fit, which is intended.
    warnings.filterwarnings("ignore", message=r"n_jobs value .* overridden")
    try:
        problems = precompute(options.configs, check_only=options.check or options.dry_run,
                              force=options.force)
    except ValueError as exc:  # a config that fails validation, or EmbeddingCacheError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1 if problems and not options.dry_run else 0


if __name__ == "__main__":
    # Run the importable module rather than this __main__ copy of it, so there is one
    # EmbeddingCacheError class: the one qprofiler raises.
    from qbiocode.apps.qprofiler.embedding_cache import main as _main

    sys.exit(_main())
