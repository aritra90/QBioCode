"""Read the frozen outer-fold manifests written by ``benchmark/make_splits.py``.

Under ``split_mode: manifest`` QProfiler does not split the data itself. It reads, for
each dataset CSV, a manifest that pins every outer split of a repeated stratified K-fold
together with the validation rows carved from each training fold, and it checks that
the CSV is byte-for-byte the file the manifest was computed from. Every method,
embedding and tuning trial then sees the same rows, and every row is a test row exactly
once per repeat.

Manifest format (``schema_version`` 2, one JSON file per dataset, ``<dataset_id>.json``)::

    {
      "schema_version": 2,
      "dataset_id": "pmlb__glass2",
      "sha256": "<sha256 of pmlb__glass2.csv>",
      "n": 163, "k": 5, "n_repeats": 3,
      "protocol": "StratifiedKFold",
      "validation": "next_fold",
      "seed": 42, "repeat_seeds": [42, 43, 44],
      "group_col": null,
      "generator_version": "make_splits/2.0",
      "folds": [
        {"repeat": 0, "fold": 0, "train": [...], "val": [...], "test": [...]},
        ...
      ]
    }

Indices are 0-based row positions in the CSV as ``pandas.read_csv`` returns it (header
excluded). ``val`` is a subset of ``train``; the trials of a tuner are fit on
``train \\ val`` (the *fit* rows) and scored on ``val``.

Each split has a 1-based global ``iteration`` = ``repeat * k + fold + 1``. It is what
QProfiler writes into the ``iteration`` column and the trailing token of the data_key,
so code that parses those keeps working; ``repeat`` and ``fold`` are recorded as their
own columns as well.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from qbiocode.evaluation.protocol import VALIDATION_SCHEMES

SCHEMA_VERSION = 2
MANIFEST_SUFFIX = ".json"


class ManifestError(ValueError):
    """A manifest is missing, malformed, or does not match the dataset it is used with."""


def file_sha256(path: str | os.PathLike) -> str:
    """sha256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(split_dir: str | os.PathLike, dataset_file: str | os.PathLike) -> Path:
    """Where the manifest of ``dataset_file`` lives: ``<split_dir>/<stem>.json``."""
    stem = os.path.splitext(os.path.basename(os.fspath(dataset_file)))[0]
    return Path(split_dir) / f"{stem}{MANIFEST_SUFFIX}"


@dataclass(frozen=True, eq=False)
class OuterSplit:
    """One outer split: the test rows, the training rows, and the tuning split of the latter.

    Attributes:
        repeat: 0-based repeat index.
        fold: 0-based fold index within the repeat.
        iteration: 1-based global id, ``repeat * k + fold + 1``.
        seed: Seed of the StratifiedKFold that produced this repeat.
        train_idx: Sorted row ids of the outer training fold (fit + validation).
        val_idx: Sorted row ids of the validation rows (a subset of train_idx).
        fit_idx: Sorted row ids of train_idx minus val_idx.
        test_idx: Sorted row ids of the outer test fold.
    """

    repeat: int
    fold: int
    iteration: int
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    fit_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_fit(self) -> int:
        return int(self.fit_idx.size)

    @property
    def n_val(self) -> int:
        return int(self.val_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)

    def val_positions(self) -> np.ndarray:
        """Positions of the validation rows inside ``train_idx`` (for slicing X_train)."""
        return np.searchsorted(self.train_idx, self.val_idx)

    def fit_positions(self) -> np.ndarray:
        """Positions of the fit rows inside ``train_idx``."""
        return np.searchsorted(self.train_idx, self.fit_idx)


@dataclass(frozen=True, eq=False)
class SplitManifest:
    """A validated manifest. Build it with :func:`load_manifest`."""

    path: str
    file_sha256: str
    dataset_id: str
    sha256: str
    n: int
    k: int
    n_repeats: int
    protocol: str
    validation: str
    seed: int
    repeat_seeds: tuple
    group_col: str | None
    generator_version: str
    splits: tuple

    @property
    def iterations(self) -> list[int]:
        return [s.iteration for s in self.splits]

    def split(self, iteration: int) -> OuterSplit:
        """The split with global id ``iteration`` (1-based)."""
        for candidate in self.splits:
            if candidate.iteration == int(iteration):
                return candidate
        raise ManifestError(
            f"{self.dataset_id}: no split with iteration {iteration} "
            f"(valid: 1..{len(self.splits)})"
        )

    def select(self, which: str | int | Iterable[int] | None = "all") -> list[OuterSplit]:
        """The splits a run should execute.

        Args:
            which: 'all' or None for every split; an int or an iterable of ints for
                those global iterations (duplicates rejected). This is the ``splits``
                config key, so one LSF job can run one fold.
        """
        if which is None or (isinstance(which, str) and which.strip().lower() == "all"):
            return list(self.splits)
        if isinstance(which, str):
            which = [int(token) for token in which.replace(",", " ").split()]
        elif isinstance(which, (int, np.integer)):
            which = [int(which)]
        wanted = [int(i) for i in which]
        if not wanted:
            raise ManifestError(f"{self.dataset_id}: empty split selection")
        if len(set(wanted)) != len(wanted):
            raise ManifestError(f"{self.dataset_id}: duplicate iterations in {wanted}")
        return [self.split(i) for i in wanted]

    def row_fields(self, split: OuterSplit) -> dict:
        """The SPLIT_COLUMNS values of ``split`` (see qbiocode.evaluation.protocol)."""
        return {
            "split_mode": "manifest",
            "repeat": int(split.repeat),
            "fold": int(split.fold),
            "split_k": int(self.k),
            "split_repeats": int(self.n_repeats),
            "split_validation": self.validation,
            "split_seed": int(split.seed),
            "manifest_sha256": self.file_sha256,
            "dataset_sha256": self.sha256,
            "split_generator": self.generator_version,
            "n_fit": split.n_fit,
            "n_val": split.n_val,
            "n_test": split.n_test,
        }


def _index_array(value, where: str, n: int) -> np.ndarray:
    if not isinstance(value, list):
        raise ManifestError(f"{where}: expected a list of row ids, got {type(value).__name__}")
    array = np.asarray(value, dtype=np.int64) if value else np.zeros(0, dtype=np.int64)
    if array.ndim != 1:
        raise ManifestError(f"{where}: row ids must be a flat list")
    if array.size and (array.min() < 0 or array.max() >= n):
        raise ManifestError(f"{where}: row ids outside [0, {n})")
    if np.unique(array).size != array.size:
        raise ManifestError(f"{where}: repeated row ids")
    return np.sort(array)


def load_manifest(path: str | os.PathLike) -> SplitManifest:
    """Read and validate a schema-2 manifest.

    Checks, per repeat: exactly k folds numbered 0..k-1; the test folds partition
    ``range(n)``; ``train`` is the complement of ``test``; ``val`` is a subset of
    ``train`` and non-empty; under ``next_fold`` ``val`` equals the next fold's test
    rows; every fit side keeps both classes is *not* checked here (the manifest carries
    no labels) -- QProfiler checks it once y is read.

    Raises:
        ManifestError: On any violation, naming the file and the offending split.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"split manifest not found: {path}")
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path}: not valid JSON ({error})") from error

    version = payload.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: schema_version {version}, expected {SCHEMA_VERSION}. "
            "Version-1 manifests hold a single repeat and no validation rows; regenerate "
            "with benchmark/make_splits.py --repeats 3 (make_splits/2.0)."
        )
    for key in ("dataset_id", "sha256", "n", "k", "n_repeats", "protocol", "validation",
                "seed", "repeat_seeds", "generator_version", "folds"):
        if key not in payload:
            raise ManifestError(f"{path}: missing key {key!r}")

    dataset_id = str(payload["dataset_id"])
    n, k, n_repeats = int(payload["n"]), int(payload["k"]), int(payload["n_repeats"])
    validation = str(payload["validation"])
    if validation not in VALIDATION_SCHEMES:
        raise ManifestError(f"{path}: unknown validation scheme {validation!r}")
    if k < 3 and validation == "next_fold":
        # With k=2 the next fold's test rows are the whole training fold: nothing is
        # left to fit on.
        raise ManifestError(f"{path}: next_fold validation needs k >= 3, got k={k}")
    repeat_seeds = tuple(int(s) for s in payload["repeat_seeds"])
    if len(repeat_seeds) != n_repeats:
        raise ManifestError(f"{path}: {len(repeat_seeds)} repeat_seeds for n_repeats={n_repeats}")

    folds = payload["folds"]
    if not isinstance(folds, list) or len(folds) != k * n_repeats:
        raise ManifestError(
            f"{path}: expected {k * n_repeats} fold records (k={k} x {n_repeats} repeats), "
            f"got {len(folds) if isinstance(folds, list) else type(folds).__name__}"
        )

    by_key: dict[tuple[int, int], dict] = {}
    for record in folds:
        key = (int(record["repeat"]), int(record["fold"]))
        if key in by_key:
            raise ManifestError(f"{path}: repeat {key[0]} fold {key[1]} listed twice")
        by_key[key] = record

    everything = np.arange(n)
    splits = []
    for repeat in range(n_repeats):
        tests = {}
        for fold in range(k):
            where = f"{path.name} repeat {repeat} fold {fold}"
            if (repeat, fold) not in by_key:
                raise ManifestError(f"{where}: missing")
            tests[fold] = _index_array(by_key[(repeat, fold)].get("test"), f"{where} test", n)
        covered = np.sort(np.concatenate(list(tests.values())))
        if not np.array_equal(covered, everything):
            raise ManifestError(
                f"{path.name} repeat {repeat}: test folds do not partition the {n} rows"
            )
        for fold in range(k):
            where = f"{path.name} repeat {repeat} fold {fold}"
            record = by_key[(repeat, fold)]
            test_idx = tests[fold]
            train_idx = _index_array(record.get("train"), f"{where} train", n)
            val_idx = _index_array(record.get("val"), f"{where} val", n)
            if not np.array_equal(train_idx, np.setdiff1d(everything, test_idx)):
                raise ManifestError(f"{where}: train is not the complement of test")
            if val_idx.size == 0:
                raise ManifestError(f"{where}: empty validation set")
            if np.setdiff1d(val_idx, train_idx).size:
                raise ManifestError(f"{where}: validation rows outside the training fold")
            if validation == "next_fold" and not np.array_equal(val_idx, tests[(fold + 1) % k]):
                raise ManifestError(f"{where}: val is not the test rows of fold {(fold + 1) % k}")
            fit_idx = np.setdiff1d(train_idx, val_idx)
            if fit_idx.size == 0:
                raise ManifestError(f"{where}: no rows left to fit on")
            splits.append(OuterSplit(
                repeat=repeat, fold=fold, iteration=repeat * k + fold + 1,
                seed=repeat_seeds[repeat], train_idx=train_idx, val_idx=val_idx,
                fit_idx=fit_idx, test_idx=test_idx,
            ))

    return SplitManifest(
        path=str(path.resolve()),
        file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        dataset_id=dataset_id,
        sha256=str(payload["sha256"]),
        n=n, k=k, n_repeats=n_repeats,
        protocol=str(payload["protocol"]),
        validation=validation,
        seed=int(payload["seed"]),
        repeat_seeds=repeat_seeds,
        group_col=payload.get("group_col"),
        generator_version=str(payload["generator_version"]),
        splits=tuple(splits),
    )


def verify_dataset(
    manifest: SplitManifest,
    *,
    dataset_file: str | os.PathLike | None = None,
    sha256: str | None = None,
    n_rows: int | None = None,
    y: Sequence | np.ndarray | None = None,
) -> str:
    """Refuse a dataset the manifest was not computed from.

    Args:
        manifest: The loaded manifest.
        dataset_file: The CSV; its stem must equal ``manifest.dataset_id`` and its bytes
            are hashed unless ``sha256`` is given.
        sha256: Precomputed sha256 of the CSV bytes.
        n_rows: Rows QProfiler read from it; must equal ``manifest.n``.
        y: Encoded labels, when available. Every fit side and every validation side
            must then hold both classes -- a single-class side cannot be tuned or scored.

    Returns:
        The dataset sha256 that was checked.

    Raises:
        ManifestError: On a stem, hash, row-count or class-coverage mismatch.
    """
    if dataset_file is not None:
        stem = os.path.splitext(os.path.basename(os.fspath(dataset_file)))[0]
        if stem != manifest.dataset_id:
            raise ManifestError(
                f"manifest {manifest.path} is for {manifest.dataset_id!r}, not {stem!r}"
            )
        if sha256 is None:
            sha256 = file_sha256(dataset_file)
    if sha256 is None:
        raise ManifestError("verify_dataset needs dataset_file or sha256")
    if sha256 != manifest.sha256:
        raise ManifestError(
            f"{manifest.dataset_id}: the CSV has changed since its splits were frozen\n"
            f"  manifest sha256 : {manifest.sha256}\n"
            f"  file sha256     : {sha256}\n"
            "  regenerate the manifests (make_splits.py) or restore the file"
        )
    if n_rows is not None and int(n_rows) != manifest.n:
        raise ManifestError(
            f"{manifest.dataset_id}: read {n_rows} rows, the manifest indexes {manifest.n}"
        )
    if y is not None:
        y = np.asarray(y).ravel()
        if y.size != manifest.n:
            raise ManifestError(f"{manifest.dataset_id}: {y.size} labels for n={manifest.n}")
        for split in manifest.splits:
            for side, idx in (("fit", split.fit_idx), ("val", split.val_idx),
                              ("test", split.test_idx)):
                if np.unique(y[idx]).size < 2:
                    raise ManifestError(
                        f"{manifest.dataset_id} repeat {split.repeat} fold {split.fold}: "
                        f"the {side} rows hold a single class"
                    )
    return sha256
