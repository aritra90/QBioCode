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

"""The embedding cache across processes, the way the pilot's jobs use it.

tests/test_embedding_cache.py holds the cache's contract inside one interpreter. What
it cannot show is the seam the cache exists for: features written by one process, for
one CPU target, and read by jobs started later from the command line. Every run here is
a real subprocess on the shipped config, started with the commands
experiments/pilot10/submit_pilot.sh uses:

* the precompute, ``python -m qbiocode.apps.qprofiler.embedding_cache job.yaml``;
* a job that reads the cache, ``python -m qbiocode.apps.qprofiler.cli --config-dir ...``;
* the same job without a cache, which embeds for itself as every job did before;
* a second precompute with numba compiling for a generic x86-64 CPU instead of this
  one. It stands in for a host of another type, and a job on this host reads it.

Each process spends most of its time importing, so the independent ones run at once.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib import import_module
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from .conftest import CONFIG_DIR, metric_signature, subprocess_env, write_dataset

#: A run that embeds and fits in seconds, on the conftest's 60 x 5 dataset.
SETTINGS = {
    "file_dataset": "ALL",
    "embeddings": ["pca", "umap", "none"],
    # 5 features is under the default threshold, which would suppress both embeddings.
    "embedding_min_features": 0,
    "n_components": 2,
    "n_neighbors": 10,
    "iter": 2,
    "model": ["lr", "dt"],
    "n_jobs": 1,
    "grid_search": False,
    # The shipped config tunes quantum models, which needs grid_search.
    "tune_quantum": False,
    "seed": 7,
}
KEYS = {(e, it): f"tiny_{e}_2_{it}" for it in (1, 2) for e in ("pca", "umap", "none")}
#: The passes a job with a cache reads from it.
CACHED = [(e, it) for e, it in KEYS if e != "none"]
#: Seconds for one process, most of it imports from a shared filesystem.
TIMEOUT = 1500
#: The environment of a process that stands in for a host of another CPU type.
OTHER_CPU = {"NUMBA_CPU_NAME": "generic"}
FEATURES_LINE = re.compile(
    r"Features of (\S+): \d+ columns, sha256 ([0-9a-f]{16}), (.*)$", re.M
)


def _config(path, **settings):
    """A job YAML: the shipped config with SETTINGS and ``settings``."""
    config = OmegaConf.load(CONFIG_DIR / "config.yaml")
    for key, value in {**SETTINGS, "config_file_name": path.stem, **settings}.items():
        config[key] = value
    OmegaConf.save(config, path)


def _start(root, name, argv, cwd, **env):
    """``python <argv>`` in ``cwd``, its output to ``<root>/<name>.out``.

    A file rather than a pipe: a job logs more than a pipe buffers, and one that
    nothing reads would block.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    child_env = subprocess_env()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        child_env.setdefault(var, "1")
    child_env.update(env)
    with open(root / f"{name}.out", "w") as out:
        return subprocess.Popen([sys.executable, *argv], cwd=cwd, env=child_env,
                                stdout=out, stderr=subprocess.STDOUT)


def _precompute(root, config_name):
    return ["-m", "qbiocode.apps.qprofiler.embedding_cache", str(root / f"{config_name}.yaml")]


def _job(root, config_name):
    return ["-m", "qbiocode.apps.qprofiler.cli", f"--config-dir={root}",
            f"--config-name={config_name}"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Every process of the module, run once. Each result is ``(rc, out)``."""
    root = tmp_path_factory.mktemp("embedding_cache_cli")
    write_dataset(root / "data")
    caches = {"host": root / "embeddings", "other": root / "embeddings_other_cpu"}
    _config(root / "cached.yaml", folder_path=str(root / "data"),
            embedding_cache=str(caches["host"]))
    _config(root / "uncached.yaml", folder_path=str(root / "data"), embedding_cache=None)
    _config(root / "other_cpu.yaml", folder_path=str(root / "data"),
            embedding_cache=str(caches["other"]))
    # numba keys its on-disk cache by CPU as well, but a separate one leaves no doubt
    # that the other target compiled for itself.
    other_cpu = {**OTHER_CPU, "NUMBA_CACHE_DIR": str(root / "numba_other_cpu")}

    started, results = {}, {}

    def finish(name):
        started[name].wait(timeout=TIMEOUT)
        results[name] = SimpleNamespace(rc=started[name].returncode,
                                        out=(root / f"{name}.out").read_text())

    try:
        started["precompute"] = _start(root, "precompute", _precompute(root, "cached"), root)
        started["uncached"] = _start(root, "uncached", _job(root, "uncached"),
                                     root / "work_uncached")
        started["precompute_other_cpu"] = _start(
            root, "precompute_other_cpu", _precompute(root, "other_cpu"), root, **other_cpu)
        # Imported while the children import, rather than before any of them starts.
        emb_cache = import_module("qbiocode.apps.qprofiler.embedding_cache")

        finish("precompute")
        if results["precompute"].rc == 0:
            started["cached"] = _start(root, "cached", _job(root, "cached"),
                                       root / "work_cached")
        finish("precompute_other_cpu")
        if results["precompute_other_cpu"].rc == 0:
            # On this host's CPU target: a job reading what another host type wrote.
            started["reads_other_cpu"] = _start(root, "reads_other_cpu",
                                                _job(root, "other_cpu"),
                                                root / "work_reads_other_cpu")
        for name in ("uncached", "cached", "reads_other_cpu"):
            if name in started:
                finish(name)
    finally:
        for process in started.values():
            if process.poll() is None:
                process.kill()
                process.wait()
    return SimpleNamespace(root=root, caches=caches, results=results, emb_cache=emb_cache)


def _result(runs, name):
    result = runs.results.get(name)
    assert result is not None, f"{name} was not started, because the precompute it reads failed"
    assert result.rc == 0, f"{name} exited {result.rc}\n{result.out[-4000:]}"
    return result


def _model_results(work_dir):
    files = sorted(work_dir.glob("results/**/ModelResults.csv"))
    assert len(files) == 1, f"expected one ModelResults.csv under {work_dir}, found {files}"
    return pd.read_csv(files[0])


def _logged(result):
    """``{data_key: (features digest, where the features came from)}`` from a job's log."""
    return {key: (digest, source) for key, digest, source in FEATURES_LINE.findall(result.out)}


def _stored(runs, cache, embed, it):
    """``(path, features digest)`` of one cache file."""
    path = runs.emb_cache.cache_file(runs.caches[cache], KEYS[embed, it])
    entry = runs.emb_cache.read(path)
    return path, runs.emb_cache.features_digest(entry["X_train"], entry["X_test"])


class TestThePrecompute:
    @pytest.mark.parametrize("name, cache", [("precompute", "host"),
                                             ("precompute_other_cpu", "other")])
    def test_it_writes_every_split_of_every_cached_embedding(self, runs, name, cache):
        result = _result(runs, name)
        assert sorted(os.listdir(runs.caches[cache])) == sorted(
            f"emb_{KEYS[p]}.npz" for p in CACHED)
        assert "4 written, 0 already current, 0 problems" in result.out

    def test_each_file_records_the_cpu_target_it_was_compiled_for(self, runs):
        def target(cache):
            path = runs.emb_cache.cache_file(runs.caches[cache], KEYS["umap", 1])
            return runs.emb_cache.read(path)["provenance"]["numba_target"]

        assert target("other") == OTHER_CPU["NUMBA_CPU_NAME"]
        assert target("host") and target("host") != target("other")


class TestAJobThatReadsTheCache:
    def test_it_runs(self, runs):
        _result(runs, "cached")
        assert len(_model_results(runs.root / "work_cached")) == len(KEYS) * len(
            SETTINGS["model"])

    def test_it_scores_the_files_features_and_embeds_nothing(self, runs):
        logged = _logged(_result(runs, "cached"))
        for embed, it in CACHED:
            path, digest = _stored(runs, "host", embed, it)
            assert logged.pop(KEYS[embed, it]) == (digest, f"read from {path}")
        assert {key: source for key, (_, source) in logged.items()} == {
            KEYS["none", it]: "computed in this run" for it in (1, 2)}


class TestAJobWithoutACache:
    def test_on_this_cpu_it_computes_what_the_cache_holds(self, runs):
        """Why the pilot's jobs agreed within each host type, as its first check found."""
        logged = _logged(_result(runs, "uncached"))
        for embed, it in CACHED:
            _, digest = _stored(runs, "host", embed, it)
            assert logged[KEYS[embed, it]] == (digest, "computed in this run")

    def test_so_reading_the_cache_changes_no_result(self, runs):
        _result(runs, "cached")
        _result(runs, "uncached")
        pd.testing.assert_frame_equal(
            metric_signature(_model_results(runs.root / "work_cached")),
            metric_signature(_model_results(runs.root / "work_uncached")),
        )


class TestAnotherCpuType:
    """What the cache is for: a host of another type embeds the same split differently."""

    def test_its_pca_is_the_same(self, runs):
        _result(runs, "precompute_other_cpu")
        for it in (1, 2):
            assert _stored(runs, "other", "pca", it)[1] == _stored(runs, "host", "pca", it)[1]

    def test_its_umap_is_not(self, runs):
        _result(runs, "precompute_other_cpu")
        differ = [it for it in (1, 2)
                  if _stored(runs, "other", "umap", it)[1] != _stored(runs, "host", "umap", it)[1]]
        if not differ:
            pytest.skip("numba's code for this CPU and for a generic one gave the same UMAP "
                        "here, so this machine cannot show the difference")
        assert differ == [1, 2]

    def test_a_job_here_scores_the_features_the_other_type_wrote(self, runs):
        """The pilot's case: every job reads one file, whichever host type wrote it."""
        logged = _logged(_result(runs, "reads_other_cpu"))
        for embed, it in CACHED:
            path, digest = _stored(runs, "other", embed, it)
            assert logged[KEYS[embed, it]] == (digest, f"read from {path}")
