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

"""The embedding cache: every job of an experiment scores its models on the same features.

Seeded UMAP reproduces on one CPU type only, so jobs that embedded for themselves scored the
models of one (dataset, embedding) on two or three sets of features, depending on where they
ran (embedding_cache.py has the pilot's numbers). With ``embedding_cache`` set, qprofiler
reads every embedding but ``'none'`` from a file written beforehand and never computes one.
These tests hold the four things that promise rests on:

  1. A file reads back bit for bit, and one that is missing, broken, stale or embedded from
     other rows is refused, with a message that says how to fix it.
  2. The spec a file is checked against changes with every input of the embedding, and
     with nothing that differs between the jobs that share the file.
  3. The precompute writes exactly what the job would have computed, writes each file
     once however many configs read it, and leaves current files alone.
  4. main checks every file of every dataset before it fits anything, and then scores the
     file's features, not features of its own.

The cross-process half, a job started from the command line reading what the command-line
precompute wrote, is tests/integration/test_qprofiler_embedding_cache.py.
"""

import json
import logging
import os
import re
import shutil
import socket
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

emb_cache = import_module("qbiocode.apps.qprofiler.embedding_cache")
qp = import_module("qbiocode.apps.qprofiler.qprofiler")

SHIPPED_CONFIG = (
    Path(__file__).resolve().parents[1] / "qbiocode" / "apps" / "qprofiler" / "configs"
    / "config.yaml"
)
LOG = logging.getLogger("test")

#: A run that embeds and fits in seconds: the shipped config on a 60 x 5 dataset, with
#: both embeddings the pilot caches plus 'none', two splits and one classical model.
RUN = {
    "config_file_name": "embedding_cache_test",
    "file_dataset": "ALL",
    "embeddings": ["pca", "umap", "none"],
    # 5 features is under the default threshold, which would suppress both embeddings.
    "embedding_min_features": 0,
    "n_components": 2,
    "n_neighbors": 10,
    "iter": 2,
    "model": ["lr"],
    "n_jobs": 1,
    "grid_search": False,
    # The shipped config tunes quantum models, which needs grid_search.
    "tune_quantum": False,
    "seed": 7,
}
#: The (embedding, split) passes of RUN that read the cache.
EMBEDDED = [(embed, it) for it in (1, 2) for embed in ("pca", "umap")]


def _write_dataset(directory, name="tiny.csv", seed=0, rows=60):
    """A learnable binary dataset: the label follows feature 0, plus noise."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=rows) > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame["label"] = y
    frame.to_csv(directory / name, index=False)
    return directory / name


def _write_config(path, data_dir, cache_dir, **overrides):
    """A job YAML: the shipped config with RUN, the data, the cache and ``overrides``."""
    config = OmegaConf.load(SHIPPED_CONFIG)
    settings = {
        **RUN,
        "folder_path": str(data_dir),
        "embedding_cache": None if cache_dir is None else str(cache_dir),
        **overrides,
    }
    for key, value in settings.items():
        config[key] = value
    OmegaConf.save(config, path)
    return path


def _key(embed, it, dataset="tiny.csv"):
    return qp._data_key(dataset, embed, RUN["n_components"], it)


def _job_args(config):
    """``(args, scaler_name)`` as main holds them once it has validated ``config``."""
    args = emb_cache._compose(config)
    qp._resolve_model_lists(args, LOG)
    qp._resolve_backend_alias(args, LOG)
    return args, qp._validate_config(args, LOG)


def _run_main(config, work_dir, monkeypatch):
    """qprofiler's main on ``config`` in this process, writing its results to ``work_dir``.

    Hydra is bypassed, so nothing changes directory for the run; this does.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)
    qp.main(emb_cache._compose(config))


def _stamp(directory):
    """Each file's inode and mtime: a file that was replaced, even by itself, changes."""
    return {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in directory.iterdir()}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """One precompute of RUN, for the tests that only read what it wrote."""
    root = tmp_path_factory.mktemp("embedding_cache")
    csv = _write_dataset(root / "data")
    config = _write_config(root / "job.yaml", root / "data", root / "cache")
    lines = []
    problems = emb_cache.precompute([str(config)], out=lines.append)
    return SimpleNamespace(
        csv=csv, data=root / "data", config=config, cache=root / "cache",
        lines=lines, problems=problems,
    )


# ---------------------------------------------------------------------------
# 1. One file
# ---------------------------------------------------------------------------
SPEC = {"cache_version": emb_cache.CACHE_VERSION, "dataset": "tiny.csv", "iteration": 1,
        "n_neighbors": 10}


def _entry(dtype=np.float32, seed=0):
    """``(X_train, X_test, train_idx, test_idx)``: 6 + 3 rows, indices in split order."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(6, 2)).astype(dtype), rng.normal(size=(3, 2)).astype(dtype),
            np.array([5, 0, 3, 8, 1, 7]), np.array([2, 6, 4]))


class TestAFileReadsBackAsWritten:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_bit_for_bit_and_in_its_own_dtype(self, tmp_path, dtype):
        # UMAP returns float32 and PCA float64. Casting either would change the features.
        X_train, X_test, train_idx, test_idx = _entry(dtype)
        path = emb_cache.cache_file(tmp_path, "k")
        emb_cache.write(path, X_train, X_test, train_idx, test_idx, SPEC, {"host": "h"})

        entry = emb_cache.read(path)
        for name, want in (("X_train", X_train), ("X_test", X_test)):
            assert entry[name].dtype == want.dtype
            assert entry[name].tobytes() == want.tobytes()
        assert entry["train_idx"].dtype == entry["test_idx"].dtype == np.int64
        np.testing.assert_array_equal(entry["train_idx"], train_idx)
        np.testing.assert_array_equal(entry["test_idx"], test_idx)
        assert entry["spec"] == emb_cache.read_spec(path) == SPEC
        assert entry["provenance"] == {"host": "h"}

        got_train, got_test = emb_cache.load(tmp_path, "k", SPEC, train_idx, test_idx)
        assert got_train.tobytes() == X_train.tobytes()
        assert got_test.tobytes() == X_test.tobytes()

    def test_it_reads_without_pickle(self, tmp_path):
        """Every job loads these files, so reading one must not be able to run code."""
        path = emb_cache.cache_file(tmp_path, "k")
        emb_cache.write(path, *_entry(), SPEC)
        with np.load(path, allow_pickle=False) as npz:
            assert sorted(npz.files) == sorted([*emb_cache._ARRAYS, "spec", "provenance"])
            for name in npz.files:
                npz[name]

    @pytest.mark.parametrize("dtype", [object, "U1"])
    def test_non_numeric_features_are_refused_before_anything_is_written(self, tmp_path,
                                                                         dtype):
        X = np.array([["a", "b"]], dtype=dtype)
        with pytest.raises(emb_cache.EmbeddingCacheError, match="numeric arrays only"):
            emb_cache.write(emb_cache.cache_file(tmp_path, "k"), X, X, [0], [1], SPEC)
        assert os.listdir(tmp_path) == []

    def test_the_directory_is_created(self, tmp_path):
        path = emb_cache.cache_file(tmp_path / "a" / "b", "k")
        emb_cache.write(path, *_entry(), SPEC)
        assert os.path.isfile(path)


class TestAWriteIsAtomic:
    """Running jobs may be reading the directory while the precompute replaces a file."""

    def test_it_leaves_the_file_and_nothing_else(self, tmp_path):
        path = emb_cache.cache_file(tmp_path, "k")
        emb_cache.write(path, *_entry(seed=0), SPEC)
        emb_cache.write(path, *_entry(seed=1), SPEC)
        assert os.listdir(tmp_path) == ["emb_k.npz"]
        assert emb_cache.read(path)["X_train"].tobytes() == _entry(seed=1)[0].tobytes()

    def test_a_write_that_dies_halfway_keeps_the_previous_file(self, tmp_path, monkeypatch):
        path = emb_cache.cache_file(tmp_path, "k")
        emb_cache.write(path, *_entry(seed=0), SPEC)

        def dies_halfway(fh, **arrays):
            fh.write(b"PK\x03\x04 the first bytes of an archive")
            raise OSError("No space left on device")

        monkeypatch.setattr(emb_cache.np, "savez", dies_halfway)
        with pytest.raises(OSError, match="No space left"):
            emb_cache.write(path, *_entry(seed=1), SPEC)
        monkeypatch.undo()

        assert os.listdir(tmp_path) == ["emb_k.npz"]
        assert emb_cache.read(path)["X_train"].tobytes() == _entry(seed=0)[0].tobytes()


def _savez(path, drop=(), **replace):
    """An archive laid out like a cache file, with members dropped or replaced."""
    X_train, X_test, train_idx, test_idx = _entry()
    members = {"X_train": X_train, "X_test": X_test, "train_idx": train_idx,
               "test_idx": test_idx, "spec": np.array(json.dumps(SPEC)),
               "provenance": np.array("{}"), **replace}
    np.savez(path, **{k: v for k, v in members.items() if k not in drop})


class TestAFileThatCannotServeTheRunIsRefused:
    def test_a_missing_file_says_how_to_write_it(self, tmp_path):
        with pytest.raises(emb_cache.EmbeddingCacheError) as caught:
            emb_cache.load(tmp_path, "k", SPEC, [0], [1])
        message = str(caught.value)
        assert f"{emb_cache.cache_file(tmp_path, 'k')} does not exist" in message
        assert emb_cache.PRECOMPUTE_COMMAND in message

    @pytest.mark.parametrize("content", ["empty", "garbage", "bare-npy"])
    def test_an_unreadable_file_is_reported_as_one(self, tmp_path, content):
        path = emb_cache.cache_file(tmp_path, "k")
        if content == "bare-npy":
            # np.load opens a bare array too, whatever the file is called.
            with open(path, "wb") as fh:
                np.save(fh, np.zeros(3))
        else:
            Path(path).write_bytes(b"" if content == "empty" else b"not an archive")
        with pytest.raises(emb_cache.EmbeddingCacheError, match="is not a readable cache file"):
            emb_cache.load(tmp_path, "k", SPEC, [0], [1])
        [problem] = emb_cache.check(tmp_path, [("k", SPEC)])
        assert "is not a readable cache file" in problem

    def test_a_file_without_one_of_its_arrays(self, tmp_path):
        _savez(emb_cache.cache_file(tmp_path, "k"), drop=("X_test",))
        with pytest.raises(emb_cache.EmbeddingCacheError, match="cannot read 'X_test'"):
            emb_cache.load(tmp_path, "k", SPEC, *_entry()[2:])

    def test_a_spec_that_is_not_json(self, tmp_path):
        _savez(emb_cache.cache_file(tmp_path, "k"), spec=np.array("{'not': json}"))
        with pytest.raises(emb_cache.EmbeddingCacheError, match="its 'spec' is not JSON"):
            emb_cache.load(tmp_path, "k", SPEC, *_entry()[2:])
        [problem] = emb_cache.check(tmp_path, [("k", SPEC)])
        assert "its 'spec' is not JSON" in problem

    def test_a_stale_file_names_every_setting_that_differs_and_both_values(self, tmp_path):
        emb_cache.write(emb_cache.cache_file(tmp_path, "k"), *_entry(), SPEC)
        wanted = {**SPEC, "n_neighbors": 15, "split_seed": 8}
        with pytest.raises(emb_cache.EmbeddingCacheError) as caught:
            emb_cache.load(tmp_path, "k", wanted, *_entry()[2:])
        message = str(caught.value)
        assert "n_neighbors: the file has 10, this config needs 15" in message
        assert "split_seed: the file has '<absent>', this config needs 8" in message
        assert "--force" in message and emb_cache.PRECOMPUTE_COMMAND in message
        [problem] = emb_cache.check(tmp_path, [("k", wanted)])
        assert "n_neighbors: the file has 10, this config needs 15" in problem

    def test_a_file_embedded_from_other_rows_is_refused(self, tmp_path):
        """Row i of the features is dataset row train_idx[i], so the order matters too."""
        X_train, X_test, train_idx, test_idx = _entry()
        emb_cache.write(emb_cache.cache_file(tmp_path, "k"), X_train, X_test, train_idx,
                        test_idx, SPEC)
        swapped = train_idx[[1, 0, 2, 3, 4, 5]]
        with pytest.raises(emb_cache.EmbeddingCacheError,
                           match="embedded from other rows than this run's split 1"):
            emb_cache.load(tmp_path, "k", SPEC, swapped, test_idx)
        emb_cache.load(tmp_path, "k", SPEC, train_idx, test_idx)


class TestRequireChecksEveryFileAtOnce:
    def test_it_passes_when_every_file_is_current(self, tmp_path):
        emb_cache.write(emb_cache.cache_file(tmp_path, "k"), *_entry(), SPEC)
        emb_cache.require(tmp_path, [("k", SPEC)])
        emb_cache.require(tmp_path, [])

    def test_it_counts_every_problem_and_lists_the_first_eight(self, tmp_path):
        entries = [(f"k{i:02d}", SPEC) for i in range(11)]
        emb_cache.write(emb_cache.cache_file(tmp_path, "k00"), *_entry(), SPEC)
        with pytest.raises(emb_cache.EmbeddingCacheError) as caught:
            emb_cache.require(tmp_path, entries)
        message = str(caught.value)
        assert "cannot serve this run: 10 of the 11 embedded splits" in message
        assert message.count("does not exist") == 8
        assert "... and 2 more" in message
        assert emb_cache.PRECOMPUTE_COMMAND in message


class TestTheFeaturesDigest:
    """What qprofiler logs for each pass, so that two jobs' logs show whether they agreed."""

    def test_it_sees_every_bit_the_dtype_the_shape_and_the_side(self):
        X = np.arange(6, dtype=np.float64).reshape(3, 2)
        digest = emb_cache.features_digest(X, X[:1])
        assert digest == emb_cache.features_digest(X.copy(), X[:1].copy())
        assert re.fullmatch(r"[0-9a-f]{16}", digest)
        last_bit = X.copy()
        last_bit[2, 1] = np.nextafter(last_bit[2, 1], np.inf)
        for other in [(last_bit, X[:1]), (X.astype(np.float32), X[:1]),
                      (X.reshape(2, 3), X[:1]), (X[:1], X)]:
            assert emb_cache.features_digest(*other) != digest

    def test_it_ignores_the_memory_layout(self):
        X = np.asfortranarray(np.arange(6.0).reshape(3, 2))
        assert emb_cache.features_digest(X) == emb_cache.features_digest(np.ascontiguousarray(X))


# ---------------------------------------------------------------------------
# 2. The spec
# ---------------------------------------------------------------------------
ARGS = {
    "seed": 7, "test_size": 0.25, "stratify": ["y"], "index_col": False,
    "n_components": 2, "n_neighbors": 10, "quvine_args": {"walk_length": 5},
    "iter": 2, "embeddings": ["pca", "umap", "none"], "embedding_min_features": 0,
}


def _spec(args=None, dataset="tiny.csv", sha="0" * 64, embedding="umap", iteration=1,
          scaler="MinMaxScaler"):
    return emb_cache.embedding_spec(ARGS if args is None else args, dataset, sha, embedding,
                                    iteration, scaler)


class TestTheSpec:
    def test_it_is_the_same_from_hydras_config_as_from_a_dict(self):
        assert _spec(OmegaConf.create(ARGS)) == _spec(ARGS)

    def test_it_survives_the_trip_through_the_file(self):
        """A file serves a run when its stored spec equals the run's, so JSON must not
        change it on the way."""
        assert json.loads(json.dumps(_spec())) == _spec()

    def test_it_records_every_setting_the_embedding_is_given(self):
        # _embed passes _embedding_settings, so a setting added there is recorded here.
        spec = _spec()
        for key, value in qp._embedding_settings(ARGS).items():
            assert spec[key] == value

    # The job keys of the split layout (tests/test_pilot_split_contract.py JOB_KEYS), and
    # the other settings that say how a job runs rather than what it is fed.
    @pytest.mark.parametrize("key, value", [
        ("config_file_name", "wdbc_umap_qsvc"), ("n_jobs", 13), ("embeddings", ["umap"]),
        ("classical_model", ["lr"]), ("quantum_model", ["qsvc"]), ("model", ["qsvc"]),
        ("quantum_param_dir", "/runs/wdbc/a"), ("kernel_dump_dir", "/runs/wdbc/a/k"),
        ("backend", "ibm_torino"), ("embedding_cache", "/elsewhere"), ("iter", 5),
        ("grid_search", True), ("n_trials_quantum", 3),
    ])
    def test_it_ignores_what_differs_between_the_jobs_that_share_a_file(self, key, value):
        assert _spec({**ARGS, key: value}) == _spec()

    @pytest.mark.parametrize("key, value, named", [
        ("seed", 8, "split_seed"),
        ("test_size", 0.3, "test_size"),
        ("stratify", [], "stratify"),
        ("index_col", True, "index_col"),
        ("n_components", 3, "n_components"),
        ("n_neighbors", 15, "n_neighbors"),
        ("quvine_args", {"walk_length": 6}, "quvine_args"),
    ])
    def test_it_changes_with_every_setting_the_features_depend_on(self, key, value, named):
        differences = emb_cache.spec_differences(_spec(), _spec({**ARGS, key: value}))
        assert [k for k, _, _ in differences] == [named]

    @pytest.mark.parametrize("change, named", [
        ({"embedding": "pca"}, ["embedding"]),
        ({"iteration": 2}, ["iteration", "split_seed"]),
        ({"scaler": "StandardScaler"}, ["scaling"]),
        ({"sha": "1" * 64}, ["dataset_sha256"]),
        ({"dataset": "other.csv"}, ["dataset"]),
    ])
    def test_it_changes_with_the_data_the_split_and_the_embedding(self, change, named):
        differences = emb_cache.spec_differences(_spec(), _spec(**change))
        assert [k for k, _, _ in differences] == named

    def test_it_changes_with_the_file_layout(self, monkeypatch):
        before = _spec()
        monkeypatch.setattr(emb_cache, "CACHE_VERSION", emb_cache.CACHE_VERSION + 1)
        assert [k for k, _, _ in emb_cache.spec_differences(before, _spec())] == [
            "cache_version"]

    def test_a_missing_n_neighbors_is_recorded_as_the_default_the_embedding_uses(self):
        args = {k: v for k, v in ARGS.items() if k != "n_neighbors"}
        assert _spec(args)["n_neighbors"] == qp._embedding_settings(args)["n_neighbors"]


class TestThePlan:
    """The files the preflight checks are the files the embedding loop reads."""

    def test_every_split_of_every_embedding_but_none(self):
        args = {**ARGS, "iter": 3}
        planned = emb_cache.plan("tiny.csv", "0" * 64, 5, args, "MinMaxScaler")
        passes = [(e, it) for it in (1, 2, 3) for e in ("pca", "umap")]
        assert [key for key, _ in planned] == [qp._data_key("tiny.csv", e, 2, it)
                                               for e, it in passes]
        assert [spec for _, spec in planned] == [_spec(args, embedding=e, iteration=it)
                                                 for e, it in passes]

    def test_a_dataset_too_narrow_to_embed_reads_nothing(self):
        # Below the threshold main runs 'none' alone, which it computes itself.
        from qbiocode import resolve_embeddings

        args = {**ARGS, "embedding_min_features": 18}
        assert resolve_embeddings(args["embeddings"], 5, min_features=18)[0] == ["none"]
        assert emb_cache.plan("tiny.csv", "0" * 64, 5, args, "MinMaxScaler") == []

    def test_without_a_threshold_it_uses_mains_default(self):
        # A dataset embeds when it has MORE features than the threshold.
        from qbiocode.embeddings import DEFAULT_EMBEDDING_MIN_FEATURES as default

        args = {k: v for k, v in ARGS.items() if k != "embedding_min_features"}
        assert emb_cache.plan("tiny.csv", "0" * 64, default, args, "MinMaxScaler") == []
        planned = emb_cache.plan("tiny.csv", "0" * 64, default + 1, args, "MinMaxScaler")
        assert len(planned) == 2 * args["iter"]


# ---------------------------------------------------------------------------
# 3. The config key
# ---------------------------------------------------------------------------
VALID_CONFIG = {
    "folder_path": "tutorial_test_data", "file_dataset": "ALL",
    "embeddings": ["pca", "none"], "n_components": 3, "model": ["svc"],
    "seed": 42, "q_seed": 42, "test_size": 0.3, "iter": 2,
    "scaling": ["True"], "backend": "simulator", "n_jobs": 4,
}


class TestTheConfigKey:
    @pytest.mark.parametrize("value", [None, "", "   "], ids=["null", "empty", "blank"])
    def test_unset_means_the_run_embeds_for_itself(self, value):
        assert qp._embedding_cache_dir({"embedding_cache": value}) is None

    def test_a_config_from_before_the_key_runs_as_it_did(self):
        assert qp._embedding_cache_dir({}) is None

    def test_an_absolute_path_is_the_cache(self, tmp_path):
        assert qp._embedding_cache_dir({"embedding_cache": str(tmp_path)}) == str(tmp_path)
        assert qp._embedding_cache_dir({"embedding_cache": f" {tmp_path} "}) == str(tmp_path)

    def test_the_home_directory_is_expanded(self):
        assert qp._embedding_cache_dir({"embedding_cache": "~/emb"}) == os.path.expanduser(
            "~/emb")

    @pytest.mark.parametrize("value", ["embeddings", "./embeddings", "../pilot10/embeddings"])
    def test_a_relative_path_is_refused(self, value):
        # Hydra runs each job from its own directory, so it would name a cache per job.
        with pytest.raises(ValueError, match="must be an absolute path"):
            qp._embedding_cache_dir({"embedding_cache": value})

    @pytest.mark.parametrize("value", [1, True, ["/a"], {"dir": "/a"}])
    def test_a_value_that_is_not_a_path_is_refused(self, value):
        with pytest.raises(ValueError, match="is the directory the embedded features are read"):
            qp._embedding_cache_dir({"embedding_cache": value})
        with pytest.raises(ValueError, match="is the directory the embedded features are read"):
            qp._embedding_cache_dir(OmegaConf.create({"embedding_cache": value}))

    def test_it_is_validated_with_the_rest_of_the_config(self):
        qp._validate_config({**VALID_CONFIG, "embedding_cache": None}, LOG)
        with pytest.raises(ValueError, match="must be an absolute path"):
            qp._validate_config({**VALID_CONFIG, "embedding_cache": "embeddings"}, LOG)


# ---------------------------------------------------------------------------
# 4. The precompute
# ---------------------------------------------------------------------------
class TestThePrecompute:
    def test_it_writes_one_file_per_split_of_each_cached_embedding(self, experiment):
        assert experiment.problems == 0, experiment.lines
        assert sorted(os.listdir(experiment.cache)) == sorted(
            f"emb_{_key(e, it)}.npz" for e, it in EMBEDDED)
        assert sum(line.startswith("wrote ") for line in experiment.lines) == 4
        assert experiment.lines[-1] == "4 written, 0 already current, 0 problems"

    def test_each_file_holds_what_the_job_would_have_computed(self, experiment):
        args, scaler = _job_args(experiment.config)
        X, _, y = qp._read_dataset(str(experiment.csv), args)
        sha = emb_cache.file_sha256(experiment.csv)
        for embed, it in EMBEDDED:
            X_train, X_test, _, _, train_idx, test_idx = qp._split_and_scale(
                X, y, args, it, scaler)
            want = qp._embed(embed, X_train, X_test, args, qp._split_seed(args, it))
            entry = emb_cache.read(emb_cache.cache_file(experiment.cache, _key(embed, it)))
            for got, expected in zip((entry["X_train"], entry["X_test"]), want):
                assert got.dtype == expected.dtype, (embed, it)
                assert got.tobytes() == expected.tobytes(), (embed, it)
            np.testing.assert_array_equal(entry["train_idx"], train_idx)
            np.testing.assert_array_equal(entry["test_idx"], test_idx)
            assert entry["spec"] == emb_cache.embedding_spec(
                args, "tiny.csv", sha, embed, it, scaler)

    def test_the_provenance_says_where_and_with_what(self, experiment):
        record = emb_cache.read(emb_cache.cache_file(experiment.cache, _key("umap", 1)))[
            "provenance"]
        assert record["host"] == socket.gethostname()
        assert record["written_by"] == os.path.abspath(experiment.config)
        assert {"numpy", "umap-learn", "numba"} <= set(record["versions"])
        # The CPU its UMAP was compiled for: what a disagreement would be traced to.
        assert record["numba_target"] == (os.environ.get("NUMBA_CPU_NAME") or "host")
        # Which CPU "host" was. None where the platform does not say (ARM /proc/cpuinfo).
        assert "cpu" in record

    def test_a_second_run_leaves_every_current_file_alone(self, experiment):
        before = _stamp(experiment.cache)
        lines = []
        assert emb_cache.precompute([str(experiment.config)], out=lines.append) == 0
        assert lines == [f"current  emb_{_key(e, it)}.npz" for e, it in EMBEDDED] + [
            "0 written, 4 already current, 0 problems"]
        assert _stamp(experiment.cache) == before

    def test_check_writes_nothing_and_reports_every_missing_file(self, experiment, tmp_path):
        config = _write_config(tmp_path / "job.yaml", experiment.data, tmp_path / "cache")
        lines = []
        assert emb_cache.precompute([str(config)], check_only=True, out=lines.append) == 4
        assert not (tmp_path / "cache").exists()
        assert sum(line.startswith("MISSING ") for line in lines) == 4
        assert lines[-1] == "0 written, 0 already current, 4 problems"

    def test_a_stale_file_is_reported_and_kept_until_forced(self, experiment, tmp_path):
        cache = tmp_path / "cache"
        shutil.copytree(experiment.cache, cache)
        before = {p.name: emb_cache.read(p) for p in cache.iterdir()}
        config = _write_config(tmp_path / "job.yaml", experiment.data, cache, n_neighbors=12)

        lines = []
        assert emb_cache.precompute([str(config)], out=lines.append) == 4
        stale = [line for line in lines if line.startswith("STALE ")]
        # Every setting is recorded for every embedding, so PCA's files go stale too. A
        # needless rewrite costs seconds; a missed one would serve features computed
        # under other settings.
        assert len(stale) == 4
        assert all("n_neighbors: the file has 10, this config needs 12" in s for s in stale)
        assert {p.name: emb_cache.read_spec(p) for p in cache.iterdir()} == {
            name: entry["spec"] for name, entry in before.items()}

        lines = []
        assert emb_cache.precompute([str(config)], force=True, out=lines.append) == 0
        assert sum("(replaced: " in line for line in lines) == 4
        for embed, it in EMBEDDED:
            name = f"emb_{_key(embed, it)}.npz"
            entry = emb_cache.read(cache / name)
            assert entry["spec"]["n_neighbors"] == 12
            same = entry["X_train"].tobytes() == before[name]["X_train"].tobytes()
            assert same == (embed == "pca"), f"{name}: PCA ignores n_neighbors, UMAP does not"

    def test_two_configs_that_need_different_contents_in_one_file_conflict(self, experiment,
                                                                           tmp_path):
        a = _write_config(tmp_path / "a.yaml", experiment.data, experiment.cache)
        b = _write_config(tmp_path / "b.yaml", experiment.data, experiment.cache, n_neighbors=12)
        lines = []
        assert emb_cache.precompute([str(a), str(b)], check_only=True, out=lines.append) == 4
        conflicts = [line for line in lines if line.startswith("CONFLICT ")]
        assert len(conflicts) == 4
        assert conflicts[0] == (f"CONFLICT emb_{_key('pca', 1)}.npz: {a} and {b} need "
                                f"different contents (n_neighbors: 10 vs 12)")

    def test_the_jobs_of_a_split_layout_are_served_by_one_write_per_file(self, experiment,
                                                                         tmp_path):
        """One config per (embedding, model), as the split layout writes them."""
        cache = tmp_path / "cache"
        jobs = [
            str(_write_config(tmp_path / f"tiny_{e}_{m}.yaml", experiment.data, cache,
                              config_file_name=f"tiny_{e}_{m}", embeddings=[e], model=[m]))
            for e in ("pca", "umap", "none") for m in ("lr", "svc")
        ]
        lines = []
        assert emb_cache.precompute(jobs, out=lines.append) == 0
        assert lines[-1] == "4 written, 0 already current, 0 problems"
        # And a second precompute on this machine wrote the same bytes as the first.
        for name in os.listdir(cache):
            ours, theirs = emb_cache.read(cache / name), emb_cache.read(experiment.cache / name)
            assert ours["X_train"].tobytes() == theirs["X_train"].tobytes(), name
            assert ours["X_test"].tobytes() == theirs["X_test"].tobytes(), name

    def test_protocols_and_configs_without_a_cache_are_skipped(self, experiment, tmp_path):
        protocol = tmp_path / "_protocol.yaml"
        protocol.write_text("embedding_cache: ???\n")
        uncached = _write_config(tmp_path / "uncached.yaml", experiment.data, None)
        lines = []
        assert emb_cache.precompute([str(protocol), str(uncached)], out=lines.append) == 0
        assert lines == [
            f"skip     {protocol}: a _-prefixed config is a shared protocol, not a job",
            f"skip     {uncached}: no embedding_cache, so its run embeds for itself",
            "0 written, 0 already current, 0 problems",
        ]


class TestTheCommandLine:
    """submit_pilot.sh and submit_runs.sh act on the exit status."""

    def test_zero_when_every_job_can_start(self, experiment, capsys):
        assert emb_cache.main([str(experiment.config)]) == 0
        assert "0 written, 4 already current, 0 problems" in capsys.readouterr().out

    def test_one_when_check_finds_a_missing_file(self, experiment, tmp_path, capsys):
        config = _write_config(tmp_path / "job.yaml", experiment.data, tmp_path / "cache")
        assert emb_cache.main(["--check", str(config)]) == 1
        assert "4 problems" in capsys.readouterr().out
        assert not (tmp_path / "cache").exists()

    def test_two_when_a_config_is_invalid(self, experiment, tmp_path, capsys):
        config = _write_config(tmp_path / "job.yaml", experiment.data, "embeddings")
        assert emb_cache.main([str(config)]) == 2
        assert "error: embedding_cache must be an absolute path" in capsys.readouterr().err

    @pytest.mark.parametrize("argv", [["missing.yaml"], ["--check", "--force", "job.yaml"], []],
                             ids=["no-such-config", "check-and-force", "no-config"])
    def test_usage_errors_exit_two_before_anything_runs(self, experiment, argv, monkeypatch,
                                                        capsys):
        monkeypatch.chdir(experiment.config.parent)
        with pytest.raises(SystemExit) as caught:
            emb_cache.main(argv)
        assert caught.value.code == 2


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def _nothing_ran(work_dir):
    return not any((work_dir / name).exists()
                   for name in ("ModelResults.csv", "RawDataEvaluation.csv"))


class TestTheRunReadsTheCache:
    def test_a_missing_file_stops_it_before_anything_is_fitted(self, experiment, tmp_path,
                                                                monkeypatch):
        config = _write_config(tmp_path / "job.yaml", experiment.data, tmp_path / "empty")
        with pytest.raises(emb_cache.EmbeddingCacheError,
                           match="cannot serve this run: 4 of the 4 embedded splits"):
            _run_main(config, tmp_path / "work", monkeypatch)
        assert _nothing_ran(tmp_path / "work")

    def test_a_file_written_under_other_settings_stops_it_too(self, experiment, tmp_path,
                                                              monkeypatch):
        config = _write_config(tmp_path / "job.yaml", experiment.data, experiment.cache,
                               n_neighbors=12)
        with pytest.raises(emb_cache.EmbeddingCacheError,
                           match="n_neighbors: the file has 10, this config needs 12"):
            _run_main(config, tmp_path / "work", monkeypatch)
        assert _nothing_ran(tmp_path / "work")

    def test_every_dataset_is_checked_before_the_first_one_runs(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        _write_dataset(data, "a.csv", seed=1)
        _write_dataset(data, "b.csv", seed=2)
        cache = tmp_path / "cache"
        only_a = _write_config(tmp_path / "a.yaml", data, cache, file_dataset=["a.csv"],
                               embeddings=["pca", "none"])
        assert emb_cache.precompute([str(only_a)], out=lambda line: None) == 0
        both = _write_config(tmp_path / "both.yaml", data, cache, embeddings=["pca", "none"])

        with pytest.raises(emb_cache.EmbeddingCacheError) as caught:
            _run_main(both, tmp_path / "work", monkeypatch)
        message = str(caught.value)
        assert "2 of the 4 embedded splits" in message
        assert f"{emb_cache.cache_file(cache, 'b_pca_2_1')} does not exist" in message
        assert "emb_a_" not in message
        # a.csv runs first, and none of it ran.
        assert _nothing_ran(tmp_path / "work")

    def test_a_dataset_edited_after_the_check_is_refused(self, experiment, tmp_path,
                                                         monkeypatch):
        data = tmp_path / "data"
        shutil.copytree(experiment.data, data)
        config = _write_config(tmp_path / "job.yaml", data, experiment.cache)
        real_require = emb_cache.require

        def require_then_edit(cache_dir, entries):
            real_require(cache_dir, entries)
            with open(data / "tiny.csv", "a") as fh:
                fh.write(",".join(["0.5"] * 5 + ["1"]) + "\n")

        monkeypatch.setattr(emb_cache, "require", require_then_edit)
        with pytest.raises(emb_cache.EmbeddingCacheError,
                           match="dataset_sha256: the file has"):
            _run_main(config, tmp_path / "work", monkeypatch)
        assert not (tmp_path / "work" / "ModelResults.csv").exists()

    def test_it_scores_the_files_features_not_its_own(self, experiment, tmp_path, monkeypatch,
                                                      caplog):
        """A run that embedded again would pass the tests above on one machine, where
        its features equal the file's. So one file is rewritten here with other features
        under the same spec and rows, and the run must score those."""
        cache = tmp_path / "cache"
        shutil.copytree(experiment.cache, cache)
        tampered = emb_cache.cache_file(cache, _key("umap", 1))
        entry = emb_cache.read(tampered)
        emb_cache.write(tampered, entry["X_train"][:, ::-1], entry["X_test"][:, ::-1],
                        entry["train_idx"], entry["test_idx"], entry["spec"],
                        entry["provenance"])
        config = _write_config(tmp_path / "job.yaml", experiment.data, cache)

        caplog.set_level(logging.INFO, logger=qp.__name__)
        _run_main(config, tmp_path / "work", monkeypatch)

        logged = {key: (digest, source) for key, digest, source in re.findall(
            r"Features of (\S+): \d+ columns, sha256 ([0-9a-f]{16}), (.*)$", caplog.text,
            re.M)}
        # The rewritten file's features, not the ones this machine computes for that split.
        assert logged[_key("umap", 1)][0] != emb_cache.features_digest(
            entry["X_train"], entry["X_test"])
        for embed, it in EMBEDDED:
            path = emb_cache.cache_file(cache, _key(embed, it))
            stored = emb_cache.read(path)
            assert logged.pop(_key(embed, it)) == (
                emb_cache.features_digest(stored["X_train"], stored["X_test"]),
                f"read from {path}")
        assert {key: source for key, (_, source) in logged.items()} == {
            _key("none", it): "computed in this run" for it in (1, 2)}

        results = pd.read_csv(tmp_path / "work" / "ModelResults.csv")
        assert sorted(results["embeddings"]) == sorted(["pca", "umap", "none"] * 2)
