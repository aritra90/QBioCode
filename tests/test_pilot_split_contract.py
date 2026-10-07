"""The split layout is the combined experiment re-cut into jobs, and nothing else.

``generate_pilot_configs.py --layout split`` writes one config per (dataset, embedding,
model) under ``experiments/pilot10/runs/<dataset>/``, so that 208 single-slot LSF jobs run
at once instead of 12 sixteen-slot ones. That is only a scheduling change if three things
hold, and each has a way of failing silently:

  1. **Same science.** A split config must equal its combined parent in every key but the
     ones that say which job it is. A key that drifts -- a trial budget, a seed, a search
     space -- produces a table in which one arm was run under a different protocol, and
     nothing downstream can tell.
  2. **Same features.** The jobs of one (dataset, embedding) must score their models on
     the same features. Embedding in each job, that first failed because UMAP ran
     unseeded, from numpy's global stream, with numba-parallel SGD. In the combined layout
     that stream was the parent's and the models ran in loky workers, so it was at least
     one stream per dataset; split, each job's stream is advanced by its own model
     (``n_jobs: 1`` runs the model in-process), so from the second split on every model of
     a umap pass trained on different features. Randomised PCA (any matrix with a side
     over 500, i.e. colon_cancer) had the same hole. Seeding fixed that on one CPU type
     only: the 2026 pilot's UMAP jobs fell into three groups by host type, and the groups
     computed different features on every split. So every config now names one
     ``embedding_cache``, which the submit scripts fill before they submit, and the jobs
     read their features from it.
  3. **No shared writes.** 208 concurrent jobs must not write one file. The tuner dumps
     trial kernels with an empty data_key (``proj_pqk_.npz``), so a per-dataset kernel
     directory would be written by the pca and umap pqk jobs at once.

Also here: the shipped YAMLs are what the generator writes today, so a template fix that
was never regenerated -- the shots comment was one -- fails a test instead of shipping; and
the composed layout (a dataset's _protocol.yaml plus one small job file per job, joined by
hydra's defaults list) composes every job to exactly its self-contained config, the way
hydra itself composes it.
"""

import csv
import functools
import importlib.util
import os
import pathlib
import re

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments" / "pilot10"
CONFIG_DIR = PILOT / "configs"
RUNS_DIR = PILOT / "runs"

#: The keys that say which job a split config is. Everything else must equal the parent.
JOB_KEYS = {
    "config_file_name", "n_jobs", "embeddings", "classical_model", "quantum_model",
    "quantum_param_dir", "kernel_dump_dir", "hydra",
}


def _omegaconf():
    return pytest.importorskip("omegaconf").OmegaConf


@functools.lru_cache(maxsize=None)
def _module(name):
    spec = importlib.util.spec_from_file_location(name, PILOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path):
    """The config a job runs with: composed through its defaults list, if it has one."""
    OmegaConf = _omegaconf()
    return OmegaConf.to_container(_module("generate_pilot_configs").load_config(path),
                                  resolve=False)


def _jobs(paths):
    """The job configs among ``paths``; a _protocol.yaml is pulled in by jobs, not one."""
    return [p for p in paths if not p.name.startswith("_")]


@pytest.fixture(scope="module")
def parents():
    paths = sorted(CONFIG_DIR.glob("pilot*.yaml"))
    if not paths:
        pytest.skip(f"no combined configs at {CONFIG_DIR}")
    return {pathlib.Path(_load(p)["file_dataset"][0]).stem: (p, _load(p)) for p in paths}


@pytest.fixture(scope="module")
def splits():
    paths = _jobs(sorted(RUNS_DIR.glob("*/*.yaml")))
    if not paths:
        pytest.skip(f"no split configs under {RUNS_DIR}")
    return [(p, _load(p)) for p in paths]


@pytest.fixture(scope="module")
def generator():
    return _module("generate_pilot_configs")


def _run_dir_root(cfg):
    """hydra.run.dir minus the per-launch ``${backend}_${now:...}`` leaf.

    The directory is written ``results/${config_file_name}/...``; ``${now:}`` is a hydra
    resolver plain OmegaConf does not have, so the one interpolation that makes the path a
    job's own is substituted by hand.
    """
    d = cfg["hydra"]["run"]["dir"].replace("${config_file_name}", cfg["config_file_name"])
    assert "${config_file_name}" not in d and "$" not in str(pathlib.PurePosixPath(d).parent)
    return str(pathlib.PurePosixPath(d).parent)


class TestTheSplitIsTheSameExperiment:
    def test_every_parent_is_covered_by_exactly_its_own_jobs(self, parents, splits):
        got = {}
        for p, _ in splits:
            got.setdefault(p.parent.name, set()).add(p.stem)
        assert set(got) == set(parents), "dataset directories and combined configs disagree"
        for ds, (_, parent) in parents.items():
            want = {f"{ds}_{e}_{m}" for e in parent["embeddings"]
                    for m in parent["quantum_model"] + parent["classical_model"]}
            assert got[ds] == want, (
                f"{ds}: missing {sorted(want - got[ds])}, unexpected {sorted(got[ds] - want)}")

    def test_a_split_config_equals_its_parent_but_for_the_job_keys(self, parents, splits):
        bad = []
        for path, cfg in splits:
            _, parent = parents[path.parent.name]
            drift = {k for k in set(cfg) | set(parent)
                     if k not in JOB_KEYS and cfg.get(k) != parent.get(k)}
            if drift:
                bad.append(f"{path.name}: {sorted(drift)}")
        assert not bad, "split configs differ from their parent outside the job keys:\n" + \
            "\n".join(bad[:20])

    def test_each_job_key_says_one_model_on_one_embedding(self, parents, splits):
        for path, cfg in splits:
            ds = path.parent.name
            _, parent = parents[ds]
            emb, model = path.stem[len(ds) + 1:].split("_", 1)
            assert cfg["config_file_name"] == path.stem
            assert cfg["embeddings"] == [emb] and emb in parent["embeddings"]
            assert cfg["classical_model"] + cfg["quantum_model"] == [model]
            # The model stays on its own side of the quantum/classical split.
            side = "quantum_model" if model in parent["quantum_model"] else "classical_model"
            assert cfg[side] == [model], path.name
            assert cfg["n_jobs"] == 1
            # hydra differs in run.dir and nowhere else.
            h, ph = dict(cfg["hydra"]), dict(parent["hydra"])
            h["run"], ph["run"] = dict(h["run"]), dict(ph["run"])
            h["run"].pop("dir"), ph["run"].pop("dir")
            assert h == ph, path.name


class TestNoTwoJobsShareAWritablePath:
    @pytest.mark.parametrize("key", ["quantum_param_dir", "kernel_dump_dir", "run_dir"])
    def test_every_job_owns_its_directory(self, splits, key):
        seen = {}
        for path, cfg in splits:
            where = _run_dir_root(cfg) if key == "run_dir" else cfg[key]
            assert where not in seen, f"{path.name} and {seen[where]} share {key} {where}"
            seen[where] = path.name
            # ...and it lives under that job's dataset directory, where the scripts look.
            assert pathlib.PurePosixPath(where).is_relative_to(path.parent), (path.name, where)

    def test_no_split_path_reaches_into_the_combined_layouts_directories(self, splits):
        for path, cfg in splits:
            for where in (cfg["quantum_param_dir"], cfg["kernel_dump_dir"], _run_dir_root(cfg)):
                assert pathlib.PurePosixPath(where).is_relative_to(RUNS_DIR), (path.name, where)


class TestTheManifestMatchesTheYamls:
    def test_one_row_per_yaml_with_the_yamls_own_values(self, splits):
        mpath = RUNS_DIR / "MANIFEST.tsv"
        assert mpath.exists(), "MANIFEST.tsv missing; submit_runs.sh orders and filters by it"
        with open(mpath, newline="") as fh:
            rows = {r["config"]: r for r in csv.DictReader(fh, delimiter="\t")}
        assert set(rows) == {p.stem for p, _ in splits}
        for path, cfg in splits:
            r = rows[path.stem]
            assert pathlib.Path(r["yaml"]) == path.resolve(), "submit_runs.sh joins on this path"
            assert r["dataset"] == path.parent.name
            assert [r["embedding"]] == cfg["embeddings"]
            assert [r["model"]] == cfg["classical_model"] + cfg["quantum_model"]
            assert r["arm"] == ("quantum" if cfg["quantum_model"] else "classical")
            assert int(r["iter"]) == cfg["iter"]
            assert int(r["n_trials_quantum"]) == cfg["n_trials_quantum"]
            # Quantum rows carry hours (submit order, status overrun flag); classical none.
            assert bool(r["exp_h"]) == (r["arm"] == "quantum"), path.name


class TestTheShippedYamlsAreFresh:
    """Byte-for-byte what ``generate_pilot_configs.build`` produces from today's template."""

    def _rebuild(self, gen, cfg, composed=False, **job):
        ds = pathlib.Path(cfg["file_dataset"][0]).stem
        for idx, (folder, csv_name, rows, feats, why) in enumerate(gen.DATASETS, start=1):
            if csv_name[:-4] != ds:
                continue
            if composed:
                return gen.build_job(idx, folder, csv_name, feats, **job)[1]
            return gen.build(idx, folder, csv_name, rows, feats, why,
                             n_iter=cfg["iter"], test_size=cfg["test_size"],
                             n_trials_quantum=cfg["n_trials_quantum"],
                             embedding_cache=cfg.get("embedding_cache"), **job)[1]
        raise AssertionError(f"{ds} is not in the generator's DATASETS")

    def test_combined(self, generator, parents):
        stale = [p.name for p, cfg in parents.values()
                 if p.read_text() != self._rebuild(generator, cfg)]
        assert not stale, f"regenerate: python generate_pilot_configs.py ... ({stale})"

    def test_split(self, generator, splits):
        # Either layout: a directory with a protocol holds job files over it (the default),
        # one without holds self-contained configs (--self-contained; the pilot's runs/).
        stale = []
        for path, cfg in splits:
            model = (cfg["classical_model"] + cfg["quantum_model"])[0]
            composed = (path.parent / f"{generator.PROTOCOL}.yaml").exists()
            body = self._rebuild(generator, cfg, composed=composed, emb=cfg["embeddings"][0],
                                 model=model, runs_dir=str(path.parent.parent))
            if path.read_text() != body:
                stale.append(path.name)
        assert not stale, f"regenerate with --layout split ({len(stale)} stale, e.g. {stale[:5]})"

    def test_protocols(self, generator):
        stale = [p.parent.name
                 for p in sorted(RUNS_DIR.glob(f"*/{generator.PROTOCOL}.yaml"))
                 if p.read_text() != self._rebuild(generator, _load(p), protocol=True,
                                                   runs_dir=str(p.parent.parent))]
        assert not stale, f"regenerate with --layout split (stale protocols: {stale})"

    def test_the_shots_comment_no_longer_claims_the_statevector_is_exact(self, parents):
        # StatevectorSampler (V2) always samples; only the estimator arms are exact.
        for p, _ in parents.values():
            assert "Ignored by the exact statevector" not in p.read_text(), p.name


#: The keys a composed job file sets. n_jobs and hydra's run.dir are the same for every job
#: of a dataset, so they live in its protocol.
PER_JOB = JOB_KEYS - {"n_jobs", "hydra"}


def _unset(node, key=""):
    """The dotted keys whose value is MISSING (???), read without resolving anything:
    OmegaConf.missing_keys resolves, and the protocol's run dir interpolates a key that
    is itself MISSING there."""
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return {key[:-1]} if node == "???" else set()
    return set().union(*(_unset(v, f"{key}{k}.") for k, v in items))


@pytest.fixture(scope="module")
def layouts(generator, tmp_path_factory):
    """Every job at the default flags, written both ways: (composed, full, [(ds, name)]).

    Both are built with the composed tree as runs_dir, so the paths inside them agree; the
    self-contained twins are written elsewhere only so that both can exist at once.
    """
    root = tmp_path_factory.mktemp("layouts")
    composed, full = root / "composed", root / "full"
    jobs = []
    for idx, (folder, csv_name, rows, feats, why) in enumerate(generator.DATASETS, start=1):
        ds = csv_name[:-4]
        (composed / ds).mkdir(parents=True)
        (full / ds).mkdir(parents=True)
        (composed / ds / f"{generator.PROTOCOL}.yaml").write_text(generator.build(
            idx, folder, csv_name, rows, feats, why, protocol=True, runs_dir=str(composed))[1])
        for emb, model in generator.split_jobs(feats):
            name, body = generator.build_job(idx, folder, csv_name, feats, emb, model,
                                             runs_dir=str(composed))
            (composed / ds / f"{name}.yaml").write_text(body)
            (full / ds / f"{name}.yaml").write_text(generator.build(
                idx, folder, csv_name, rows, feats, why, emb=emb, model=model,
                runs_dir=str(composed))[1])
            jobs.append((ds, name))
    return composed, full, jobs


class TestTheComposedLayoutRunsTheSameConfig:
    """A job file over its dataset's _protocol.yaml is its self-contained config, key for key.

    That is the whole claim of the composed layout: it moves where the shared keys are
    written, not what any job runs with.
    """

    def test_every_job_composes_to_its_self_contained_config(self, generator, layouts):
        OmegaConf = _omegaconf()
        composed, full, jobs = layouts
        bad = [name for ds, name in jobs
               if _load(composed / ds / f"{name}.yaml")
               != OmegaConf.to_container(OmegaConf.load(full / ds / f"{name}.yaml"),
                                         resolve=False)]
        assert not bad, f"{len(bad)} of {len(jobs)} jobs compose differently, e.g. {bad[:5]}"

    def test_the_protocol_leaves_unset_exactly_the_keys_a_job_file_sets(self, generator,
                                                                         layouts):
        OmegaConf = _omegaconf()
        composed, _, _ = layouts
        for proto in sorted(composed.glob(f"*/{generator.PROTOCOL}.yaml")):
            cfg = OmegaConf.to_container(OmegaConf.load(proto), resolve=False)
            # MISSING, so a job file that forgets one -- or the protocol run by itself --
            # stops there instead of running with a placeholder.
            assert _unset(cfg) == PER_JOB, proto.parent.name
            assert cfg["n_jobs"] == 1
            assert "defaults" not in cfg

    def test_a_job_file_sets_its_job_keys_and_nothing_else(self, generator, layouts):
        OmegaConf = _omegaconf()
        composed, _, jobs = layouts
        for ds, name in jobs:
            raw = OmegaConf.to_container(OmegaConf.load(composed / ds / f"{name}.yaml"))
            assert set(raw) == {"defaults"} | PER_JOB, name
            # _self_ last: the job's keys override the protocol's.
            assert raw["defaults"] == [generator.PROTOCOL, "_self_"], name
            assert not _unset(raw), name

    # qprofiler's own version_base, so its deprecation notice is expected.
    @pytest.mark.filterwarnings("ignore:\\s*version_base=.1.1. selects")
    @pytest.mark.parametrize("ds,name", [
        ("heart", "heart_none_qsvc"),                # statevector band, quantum
        ("hepatitis", "hepatitis_none_pqk"),         # MPS band
        ("colon_cancer", "colon_cancer_umap_mlp"),   # embedded, classical
    ])
    def test_hydra_composes_it_the_same_way(self, layouts, ds, name):
        # load_config is only a stand-in for hydra; this is the check that it stands in
        # faithfully. qprofiler's @hydra.main is version_base='1.1'. Imported directly:
        # hydra-core is a base requirement, so a missing one is a broken install.
        from hydra import compose, initialize_config_dir

        OmegaConf = _omegaconf()
        composed, full, _ = layouts
        with initialize_config_dir(config_dir=str(composed / ds), version_base="1.1"):
            cfg = OmegaConf.to_container(compose(config_name=name, return_hydra_config=True),
                                         resolve=False)
        want = _load(full / ds / f"{name}.yaml")
        # Hydra adds its own defaults under hydra:, so compare the job's config and, from
        # the hydra node, the one key the config sets.
        assert cfg.pop("hydra")["run"]["dir"] == want.pop("hydra")["run"]["dir"]
        assert cfg == want

    def test_status_reads_the_same_job_either_way(self, layouts):
        status = _module("status")
        composed, full, jobs = layouts
        for ds, name in jobs:
            a = status.read_config(str(composed / ds / f"{name}.yaml"))
            b = status.read_config(str(full / ds / f"{name}.yaml"))
            assert a.pop("yaml") != b.pop("yaml")
            assert a == b, name
            assert a["expected"] == a["iter"] > 0, name

    def test_the_generator_writes_a_protocol_per_dataset_and_lists_only_jobs(
            self, generator, tmp_path, monkeypatch, capsys):
        import sys

        status = _module("status")
        monkeypatch.setattr(sys, "argv", ["generate_pilot_configs.py", "--layout", "split",
                                          "--runs-dir", str(tmp_path)])
        generator.main()
        capsys.readouterr()
        datasets = sorted(c[:-4] for _, c, _, _, _ in generator.DATASETS)
        assert sorted(p.parent.name for p in tmp_path.glob(f"*/{generator.PROTOCOL}.yaml")) \
            == datasets
        # submit_runs.sh submits what the manifest lists, and status and collate_results
        # count what find_configs finds: both must be the job files, and only those.
        with open(tmp_path / "MANIFEST.tsv", newline="") as fh:
            listed = sorted(r["yaml"] for r in csv.DictReader(fh, delimiter="\t"))
        jobs = status.find_configs(str(tmp_path))
        assert listed == jobs
        assert len(jobs) == sum(len(generator.split_jobs(f)) for *_, f, _ in generator.DATASETS)
        assert not any(generator.PROTOCOL in j for j in jobs)


class TestTheEmbeddingIsAFunctionOfTheSplit:
    """On one CPU type, a split embeds identically whatever ran before: in the precompute,
    and in a run without a cache."""

    @staticmethod
    def _data(n=60, p=30, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        return X[:48], X[48:]

    def _twice(self, method, X_tr, X_te, **kw):
        from qbiocode import get_embeddings

        out = []
        for noise in (1, 2):
            # Different global-stream state before each call: what a different model run
            # in the same process leaves behind.
            np.random.seed(noise)
            np.random.rand(noise * 7)
            out.append(get_embeddings(method, X_tr, X_te, **kw))
        return out

    def test_umap_with_random_state_is_reproducible(self):
        # No importorskip: umap-learn is a base requirement (test_suite_hygiene.py).
        X_tr, X_te = self._data()
        (a_tr, a_te), (b_tr, b_te) = self._twice("umap", X_tr, X_te, n_components=3,
                                                 n_neighbors=10, random_state=7)
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)

    def test_randomized_pca_with_random_state_is_reproducible(self):
        # A side over 500 is what sends sklearn's 'auto' solver to the randomized one.
        X_tr, X_te = self._data(n=60, p=600)
        (a_tr, a_te), (b_tr, b_te) = self._twice("pca", X_tr, X_te, n_components=8,
                                                 random_state=7)
        np.testing.assert_array_equal(a_tr, b_tr)
        np.testing.assert_array_equal(a_te, b_te)

    def test_qprofiler_seeds_the_embedding_with_the_split_seed(self):
        src = (ROOT / "qbiocode" / "apps" / "qprofiler" / "qprofiler.py").read_text()
        call = re.search(r"get_embeddings\((.*?)\n\s*\)", src, re.S)
        assert call and "random_state=split_seed" in call.group(1)


class TestEveryJobReadsOneEmbeddingCache:
    """Across CPU types: every job reads its features from one directory of files.

    A config that names another directory, or none, embeds for itself on whichever host
    it lands on, and nothing downstream can tell its features from the others'.
    """

    def test_every_config_names_the_same_absolute_directory(self, parents, splits):
        named = {}
        for path, cfg in [*parents.values(), *splits]:
            named.setdefault(cfg.get("embedding_cache"), []).append(path.name)
        assert len(named) == 1, "the configs name different caches:\n" + "\n".join(
            f"{cache}: {len(names)} configs, e.g. {names[:3]}" for cache, names in named.items())
        (cache,) = named
        assert cache and os.path.isabs(cache), cache

    @pytest.mark.parametrize("kind", ["combined", "protocol", "split"])
    def test_build_writes_the_directory_it_is_given(self, generator, kind):
        OmegaConf = _omegaconf()
        idx, (folder, csv_name, rows, feats, why) = next(
            (i, d) for i, d in enumerate(generator.DATASETS, start=1)
            if "pca" in generator.backend_for(d[3])[2])
        emb, model = generator.split_jobs(feats)[0]
        job = {"combined": {}, "protocol": {"protocol": True},
               "split": {"emb": emb, "model": model}}[kind]

        def written(cache):
            body = generator.build(idx, folder, csv_name, rows, feats, why,
                                   embedding_cache=cache, **job)[1]
            return OmegaConf.to_container(OmegaConf.create(body),
                                          resolve=False)["embedding_cache"]

        assert written("/x/emb") == "/x/emb"
        assert written(None) is None
        # qprofiler refuses a relative one, since every job runs from its own directory.
        with pytest.raises(ValueError, match="absolute"):
            written("emb")

    @pytest.mark.parametrize("layout", [["--layout", "combined"], ["--layout", "split"],
                                        ["--layout", "split", "--self-contained"]],
                             ids=["combined", "composed", "self-contained"])
    @pytest.mark.parametrize("flag", ["", "cache"], ids=["empty", "relative"])
    def test_the_generator_writes_its_flag_into_every_job(self, generator, tmp_path,
                                                          monkeypatch, capsys, layout, flag):
        import sys

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "out"
        combined = "combined" in layout
        monkeypatch.setattr(sys, "argv", [
            "generate_pilot_configs.py", *layout, "--config-dir" if combined else "--runs-dir",
            str(out), "--embedding-cache", flag])
        generator.main()
        capsys.readouterr()
        jobs = _jobs(sorted(out.glob("*.yaml" if combined else "*/*.yaml")))
        assert len(jobs) == (len(generator.DATASETS) if combined else
                             sum(len(generator.split_jobs(f)) for *_, f, _ in generator.DATASETS))
        # '' is null, so each job embeds for itself; a relative directory is made absolute
        # against the working directory.
        want = os.path.join(os.getcwd(), flag) if flag else None
        assert {_load(p).get("embedding_cache") for p in jobs} == {want}

    def test_the_shipped_cache_is_current_for_every_config(self, parents, splits):
        """What both submit scripts check before they submit, in one call.

        The combined and the split configs share the files, so a split job that needed
        other contents than its parent is a CONFLICT here, not a file that one submit
        script rewrites under the other's running jobs.
        """
        configs = [*parents.values(), *splits]
        absent = sorted({cfg["embedding_cache"] for _, cfg in configs
                         if cfg.get("embedding_cache")
                         and not os.path.isdir(cfg["embedding_cache"])})
        if absent:
            pytest.skip(f"no embedding cache at {absent}; the submit scripts write it")
        csvs = {os.path.join(cfg["folder_path"], name)
                for _, cfg in configs for name in cfg["file_dataset"]}
        absent = sorted(path for path in csvs if not os.path.isfile(path))
        if absent:
            pytest.skip(f"{len(absent)} datasets are not on this machine, e.g. {absent[0]}")
        from qbiocode.apps.qprofiler import embedding_cache as emb_cache

        lines = []
        problems = emb_cache.precompute([str(p) for p, _ in configs], check_only=True,
                                        out=lines.append)
        assert problems == 0, "\n".join(
            line for line in lines if not line.startswith(("current ", "skip ")))
        # Every embedded split once, however many configs read it.
        want = sum(len(set(cfg["embeddings"]) - {"none"}) * cfg["iter"]
                   for _, cfg in parents.values())
        assert lines[-1] == f"0 written, {want} already current, 0 problems"


class TestAnInProcessModelLeavesTheCallersStreamAlone:
    """With ``n_jobs: 1`` the model runs in qprofiler's own process."""

    def test_the_stream_is_restored_and_the_model_still_sees_its_seed(self):
        from qbiocode.evaluation.model_run import _call_with_global_seeds

        np.random.seed(123)
        before = np.random.get_state()
        drawn = _call_with_global_seeds(lambda: np.random.rand(5), 1, None)
        np.testing.assert_array_equal(drawn, np.random.RandomState(1).rand(5))
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])

    def test_the_stream_is_restored_when_the_model_raises(self):
        from qbiocode.evaluation.model_run import _call_with_global_seeds

        def boom():
            np.random.rand(3)
            raise RuntimeError("fit failed")

        np.random.seed(9)
        expected = np.random.RandomState(9).rand(4)
        with pytest.raises(RuntimeError):
            _call_with_global_seeds(boom, 1, None)
        np.testing.assert_array_equal(np.random.rand(4), expected)


# ---------------------------------------------------------------------------
# --split-mode manifest: one job per (dataset, split, embedding, group) under
# <runs-dir>/<run-id>/, every split from a precomputed manifest.
# ---------------------------------------------------------------------------

#: Fake curated datasets: id -> p. labor is the pilot's p=16 mps case (no embedding); spect
#: and spectf share a prefix, so 'spect' must resolve by the exact '__spect' suffix only.
FAKE_CURATED = {"pmlb__labor": 16, "pmlb__spect": 22, "pmlb__spectf": 44,
                "pmlb__glass2": 9, "libsvm__glass2": 9, "pmlb__appendicitis": 7}


def _fake_tree(root, with_manifest=None, repeats=3, k=5):
    """A curated tree (<id>/<id>.csv + meta.yaml) and schema-2 manifests for it."""
    import json

    data, splits = root / "datasets", root / "splits"
    splits.mkdir(parents=True)
    for i, (ds, p) in enumerate(sorted(FAKE_CURATED.items())):
        sha = f"{i:064x}"
        (data / ds).mkdir(parents=True)
        (data / ds / f"{ds}.csv").write_text("x,y\n")
        (data / ds / "meta.yaml").write_text(f"n: 60\np: {p}\nsha256: '{sha}'\n")
        if with_manifest is not None and ds not in with_manifest:
            continue
        folds = [{"repeat": r, "fold": f, "train": [], "val": [], "test": []}
                 for r in range(repeats) for f in range(k)]
        (splits / f"{ds}.json").write_text(json.dumps(
            {"schema_version": 2, "dataset_id": ds, "sha256": sha, "k": k,
             "n_repeats": repeats, "folds": folds}))
    return data, splits


def _manifest_run(generator, monkeypatch, tmp_path, *extra, run_id="t1", with_manifest=None):
    import sys

    data, splits = _fake_tree(tmp_path / "in", with_manifest=with_manifest)
    runs = tmp_path / "runs_cv"
    monkeypatch.setattr(sys, "argv", [
        "generate_pilot_configs.py", "--split-mode", "manifest", "--run-id", run_id,
        "--runs-dir", str(runs), "--datasets-root", str(data), "--split-dir", str(splits),
        *extra])
    generator.main()
    root = runs / run_id
    with open(root / "MANIFEST.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return root, rows


SHORT = ("--datasets", "appendicitis,labor,spect", "--models", "qsvc,pqk,qnn",
         "--splits", "1-5", "--n-trials", "30",
         "--wall", "classical=0:45,qsvc=2:00,pqk=2:00,qnn=4:00")


class TestTheManifestModeGenerator:
    """The short-pilot generation, run on a fake curated tree and fake manifests."""

    @pytest.fixture
    def short(self, generator, monkeypatch, tmp_path):
        return _manifest_run(generator, monkeypatch, tmp_path, *SHORT)

    def test_one_job_per_dataset_split_embedding_and_group(self, short):
        root, rows = short
        # appendicitis and labor: none; spect (p=22): pca + umap. 4 groups, 5 splits.
        assert len(rows) == (1 + 1 + 2) * 5 * 4
        keys = {(r["dataset"], r["iteration"], r["embedding"], r["group"]) for r in rows}
        assert len(keys) == len(rows)
        assert len(_jobs(sorted(root.glob("*/*.yaml")))) == len(rows)
        assert {r["group"] for r in rows} == {"classical", "qsvc", "pqk", "qnn"}
        assert {r["iteration"] for r in rows} == {"1", "2", "3", "4", "5"}

    def test_embeddings_follow_the_default_rule(self, short):
        _, rows = short
        emb = {}
        for r in rows:
            emb.setdefault(r["dataset"], set()).add(r["embedding"])
        assert emb == {"pmlb__appendicitis": {"none"}, "pmlb__labor": {"none"},
                       "pmlb__spect": {"pca", "umap"}}
        labor = {r["backend"] for r in rows if r["dataset"] == "pmlb__labor"}
        assert labor == {"mps_simulator"}

    def test_no_two_jobs_share_a_writable_path(self, short):
        root, rows = short
        seen = {}
        for r in rows:
            cfg = _load(pathlib.Path(r["yaml"]))
            for key in ("quantum_param_dir", "kernel_dump_dir"):
                seen.setdefault(key, []).append(cfg[key])
            seen.setdefault("run_dir", []).append(_run_dir_root(cfg))
            assert all(str(v).startswith(str(root)) for v in (cfg["quantum_param_dir"],
                                                              cfg["kernel_dump_dir"]))
        for key, values in seen.items():
            assert len(set(values)) == len(values), key

    def test_walls_are_per_group(self, short):
        _, rows = short
        want = {"classical": "0:45", "qsvc": "2:00", "pqk": "2:00", "qnn": "4:00"}
        assert all(r["wall"] == want[r["group"]] for r in rows)

    def test_the_subsets_and_the_run_are_recorded(self, short):
        _, rows = short
        header = list(rows[0])
        gen = _module("generate_pilot_configs")
        assert header == list(gen.MANIFEST_COLUMNS) + list(gen.MANIFEST_MODE_COLUMNS)
        r = rows[0]
        assert r["split_mode"] == "manifest" and r["run_id"] == "t1"
        assert r["sel_datasets"] == "pmlb__appendicitis,pmlb__labor,pmlb__spect"
        assert r["sel_splits"] == "1-5" and r["n_trials"] == "30"
        assert set(r["sel_models"].split(",")) == {"qsvc", "pqk", "qnn", "classical"}
        assert all(x["config"].startswith(f"{x['dataset']}_{x['embedding']}_i") for x in rows)

    def test_every_config_runs_the_manifest_protocol(self, short):
        _, rows = short
        for r in rows:
            cfg = _load(pathlib.Path(r["yaml"]))
            assert cfg["split_mode"] == "manifest"
            assert cfg["splits"] == [int(r["iteration"])]
            assert cfg["freeze_quantum_params"] is False
            assert cfg["n_trials"] == cfg["n_trials_quantum"] == 30
            for gone in ("iter", "test_size", "validation_split", "cross_validation"):
                assert gone not in cfg, gone
            ds = r["dataset"]
            assert cfg["file_dataset"] == [f"{ds}.csv"]
            assert cfg["folder_path"].endswith(f"/datasets/{ds}")
            if r["group"] == "classical":
                assert len(cfg["classical_model"]) == 9 and cfg["quantum_model"] == []
            else:
                assert cfg["classical_model"] == [] and cfg["quantum_model"] == [r["group"]]
            if r["group"] == "qnn":
                assert cfg["gridsearch_qnn_args"]["readout"] == ["global", "local"]

    def test_the_embedding_cache_lives_under_the_run(self, short):
        root, rows = short
        caches = {_load(pathlib.Path(r["yaml"]))["embedding_cache"] for r in rows}
        assert caches == {str(root / "embeddings")}

    def test_a_fixed_qnn_readout(self, generator, monkeypatch, tmp_path):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                                "--models", "qnn", "--splits", "3", "--qnn-readout", "local")
        # The classical group always comes along: naming no classical arm keeps all nine.
        assert sorted(r["group"] for r in rows) == ["classical", "qnn"]
        (r,) = [r for r in rows if r["group"] == "qnn"]
        cfg = _load(pathlib.Path(r["yaml"]))
        assert cfg["qnn_args"]["readout"] == "local"
        assert cfg["gridsearch_qnn_args"]["readout"] == ["local"]

    def test_all_splits_by_default(self, generator, monkeypatch, tmp_path):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets",
                                "pmlb__appendicitis", "--models", "qsvc")
        rows = [r for r in rows if r["group"] == "qsvc"]
        assert sorted(int(r["iteration"]) for r in rows) == list(range(1, 16))
        assert {(r["iteration"], r["repeat"], r["fold"]) for r in rows} >= {
            ("1", "0", "0"), ("6", "1", "0"), ("15", "2", "4")}

    def test_naming_a_classical_arm_splits_the_classical_group(self, generator, monkeypatch,
                                                               tmp_path):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                                "--models", "lr,svc", "--splits", "1")
        (r,) = rows
        cfg = _load(pathlib.Path(r["yaml"]))
        assert r["group"] == "classical" and cfg["classical_model"] == ["lr", "svc"]

    def test_an_ambiguous_dataset_name_is_an_error(self, generator, monkeypatch, tmp_path):
        with pytest.raises((SystemExit, ValueError)):
            _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "glass2")

    def test_a_dataset_without_a_manifest_is_skipped(self, generator, monkeypatch, tmp_path,
                                                     capsys):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets",
                                "labor,appendicitis", "--models", "qsvc", "--splits", "1",
                                with_manifest={"pmlb__appendicitis"})
        assert {r["dataset"] for r in rows} == {"pmlb__appendicitis"}
        assert "skipping pmlb__labor: no manifest" in capsys.readouterr().out

    def test_manifest_flags_are_refused_in_internal_mode(self, generator, monkeypatch):
        import sys

        monkeypatch.setattr(sys, "argv", ["generate_pilot_configs.py", "--splits", "1-5"])
        with pytest.raises(SystemExit):
            generator.main()

    def test_run_id_is_required(self, generator, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            _manifest_run(generator, monkeypatch, tmp_path, run_id="")


class TestTheShortPilotFlagParsers:
    def test_splits(self, generator):
        assert generator.parse_splits("1-5") == [1, 2, 3, 4, 5]
        assert generator.parse_splits("1,6,11") == [1, 6, 11]
        assert generator.parse_splits("all") == "all"
        for bad in ("0", "5-1", "a", "1,,2"):
            with pytest.raises(ValueError):
                generator.parse_splits(bad)

    def test_walls(self, generator):
        groups = ["qsvc", "classical"]
        assert generator.parse_wall("3:00", groups) == {"qsvc": "3:00", "classical": "3:00"}
        assert generator.parse_wall("classical=0:45,qsvc=2:00", groups) == {
            "qsvc": "2:00", "classical": "0:45"}
        for bad in ("qsvc=2:00", "nope=1:00,*=2:00", "2h", "1:75"):
            with pytest.raises(ValueError):
                generator.parse_wall(bad, groups)


class TestStatusAndCollateReadAManifestRun:
    def test_expected_rows_and_the_job_prefix(self, generator, monkeypatch, tmp_path):
        root, rows = _manifest_run(generator, monkeypatch, tmp_path, *SHORT, run_id="s-2")
        status = _module("status")
        assert status.job_prefix(str(root)) == "p10_s-2_"
        assert status.job_prefix(str(tmp_path)) == "p10_"
        for r in rows:
            cfg = status.read_config(r["yaml"])
            assert cfg["iter"] == 1
            assert cfg["expected"] == (9 if r["group"] == "classical" else 1)

    def test_bjobs_of_another_run_do_not_count(self, monkeypatch):
        status = _module("status")
        out = ("1|RUN|p10_a_x_none_i01_qsvc|h1|10 second(s)\n"
               "2|PEND|p10_b_x_none_i01_qsvc|-|0 second(s)\n")

        class Done:
            stdout = out

        monkeypatch.setattr(status.subprocess, "run", lambda *a, **k: Done())
        assert set(status.lsf_jobs("p10_a_")) == {"x_none_i01_qsvc"}
        assert status.lsf_jobs("p10_b_")["x_none_i01_qsvc"][1] == "PEND"

    def test_collate_keeps_its_key_and_reads_the_sidecars(self):
        collate = _module("collate_results")
        assert collate.KEY == ["Dataset", "embeddings", "iteration", "model"]
        assert set(collate.SIDECARS) == {"oof", "trials", "val_predictions"}


class TestTheManifestModeReviewFixes:
    """Dataset selection, stale files and flag refusals of --split-mode manifest."""

    def test_the_default_set_resolves_by_the_pilots_source(self, generator, monkeypatch,
                                                           tmp_path):
        # glass2 is curated from pmlb and libsvm here; the pilot ran pmlb_data/glass2.csv.
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--models", "qsvc",
                                "--splits", "1")
        assert {r["dataset"] for r in rows} == {
            "pmlb__appendicitis", "pmlb__glass2", "pmlb__labor", "pmlb__spect"}

    def test_a_name_and_its_id_make_one_set_of_jobs(self, generator, monkeypatch, tmp_path,
                                                    capsys):
        root, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets",
                                   "spect,pmlb__spect", "--models", "qsvc", "--splits", "1")
        assert len(rows) == 2 * 2                    # pca + umap, classical + qsvc
        assert len({r["config"] for r in rows}) == len(rows)
        assert rows[0]["sel_datasets"] == "pmlb__spect"
        assert "pmlb__spect is already selected" in capsys.readouterr().out

    def test_a_regeneration_with_fewer_datasets_leaves_no_yaml_behind(self, generator,
                                                                      monkeypatch, tmp_path):
        import sys

        root, _ = _manifest_run(generator, monkeypatch, tmp_path, "--datasets",
                                "labor,appendicitis", "--models", "qsvc", "--splits", "1")
        argv = [a if a != "labor,appendicitis" else "labor" for a in sys.argv]
        monkeypatch.setattr(sys, "argv", argv)
        generator.main()
        assert sorted(p.parent.name for p in root.glob("*/*.yaml")) == ["pmlb__labor"] * 2
        assert _module("status").find_configs(str(root)) == sorted(
            str(p) for p in root.glob("pmlb__labor/*.yaml"))

    @pytest.mark.parametrize("flag", [("--n-trials-quantum", "5"), ("--budget-hours", "3"),
                                      ("--layout", "split"), ("--self-contained",),
                                      ("--iter", "3"), ("--test-size", "0.3")])
    def test_internal_mode_flags_are_refused(self, generator, monkeypatch, tmp_path, flag):
        with pytest.raises(SystemExit):
            _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor", *flag)

    def test_the_comments_say_what_this_mode_does(self, generator, monkeypatch, tmp_path):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                                "--models", "pqk", "--splits", "1")
        body = pathlib.Path(rows[0]["yaml"]).read_text()
        assert "which no quantum space is" not in body
        assert "Where the frozen parameters are cached" not in body
        assert "classical RNG: splits" not in body


def _fake_results(run_dir, dataset, emb, iteration, models, oof_rows=(0, 1)):
    """One job's ModelResults.csv, and an oof/ and trials/ sidecar, under run_dir."""
    run_dir.mkdir(parents=True)
    lines = ["Dataset,embeddings,iteration,model,accuracy"]
    lines += [f"{dataset},{emb},{iteration},{m},0.5" for m in models]
    (run_dir / "ModelResults.csv").write_text("\n".join(lines) + "\n")
    key = f"{dataset}_{emb}_{iteration}"
    (run_dir / "oof").mkdir()
    (run_dir / "oof" / f"{key}.csv").write_text(
        "data_key,model,row_id,y_true,y_score\n"
        + "".join(f"{key},{m},{r},0,0.5\n" for m in models for r in oof_rows))
    (run_dir / "trials").mkdir()
    (run_dir / "trials" / f"{key}.csv").write_text(
        "data_key,model,trial,value\n" + "".join(f"{key},{m},0,0.5\n" for m in models))


class TestCollateAndStatusOnAManifestRun:
    @pytest.fixture
    def landed(self, generator, monkeypatch, tmp_path):
        """A two-job run (labor, classical + qsvc, split 1) with both jobs' results."""
        root, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                                   "--models", "qsvc", "--splits", "1",
                                   "--wall", "classical=0:45,qsvc=2:00")
        for r in rows:
            cfg = _load(pathlib.Path(r["yaml"]))
            models = cfg["classical_model"] + cfg["quantum_model"]
            _fake_results(root / r["dataset"] / "results" / r["config"] / "run_1",
                          r["dataset"], r["embedding"], r["iteration"], models)
        return root, rows

    def _collate(self, monkeypatch, root, out):
        import sys

        monkeypatch.setattr(sys, "argv", ["collate_results.py", "--runs-dir", str(root),
                                          "--out-dir", str(out)])
        _module("collate_results").main()

    def test_the_sidecars_are_concatenated_with_a_config_column(self, landed, monkeypatch,
                                                                tmp_path):
        import pandas as pd

        root, rows = landed
        out = tmp_path / "collated"
        self._collate(monkeypatch, root, out)
        res = pd.read_csv(out / "ModelResults.csv")
        assert len(res) == 10 and not res.duplicated(["Dataset", "embeddings", "iteration",
                                                      "model"]).any()
        oof = pd.read_csv(out / "oof.csv")
        assert list(oof.columns[:4]) == ["config", "data_key", "model", "row_id"]
        assert len(oof) == 10 * 2
        assert set(oof["config"]) == {r["config"] for r in rows}
        assert oof.equals(oof.sort_values(["data_key", "model", "row_id"], kind="stable")
                          .reset_index(drop=True))
        assert len(pd.read_csv(out / "trials.csv")) == 10
        assert not (out / "val_predictions.csv").exists()

    def test_a_duplicate_sidecar_key_fails(self, landed, monkeypatch, tmp_path):
        root, rows = landed
        (r,) = [r for r in rows if r["group"] == "qsvc"]
        oof = root / r["dataset"] / "results" / r["config"] / "run_1" / "oof"
        (oof / "again.csv").write_text(
            "data_key,model,row_id,y_true,y_score\n"
            f"{r['dataset']}_none_1,qsvc,0,0,0.5\n")
        with pytest.raises(SystemExit, match="not writing"):
            self._collate(monkeypatch, root, tmp_path / "collated")

    def test_status_lists_each_jobs_wall(self, landed, monkeypatch, capsys):
        import sys

        root, rows = landed
        monkeypatch.setattr(sys, "argv", ["status.py", "--runs-dir", str(root), "--no-lsf",
                                          "--list", "all"])
        _module("status").main()
        out = capsys.readouterr().out
        for r in rows:
            (line,) = [x for x in out.splitlines() if r["config"] in x]
            assert f"wall {r['wall']}" in line


class TestSubmitRunsDryOnAManifestRun:
    def test_each_job_gets_its_wall_and_a_run_aware_name(self, generator, monkeypatch,
                                                         tmp_path):
        import shutil
        import subprocess

        if not shutil.which("bash"):
            pytest.skip("no bash")
        root, rows = _manifest_run(generator, monkeypatch, tmp_path, *SHORT, run_id="dry1")
        # A stub for the cache step: this checks the bsub lines, not embedding_cache.
        stub = tmp_path / "stub_py"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
        env = dict(os.environ, DRY="1", FORCE="1", SPREAD="0", RUNS=str(root),
                   CACHE_PY=str(stub))
        done = subprocess.run(["bash", str(PILOT / "submit_runs.sh")], env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-2000:]
        subs = [x for x in done.stdout.splitlines() if "bsub" in x]
        assert len(subs) == len(rows)
        for r in rows:
            (line,) = [x for x in subs if f"-J p10_dry1_{r['config']} " in x]
            assert f"-W {r['wall']} " in line
        assert "would precompute the embedding cache for 80 configs" in done.stderr
        assert "WARNING: the embedding cache cannot serve" not in done.stderr

    def test_a_cache_the_check_finds_incomplete_warns(self, generator, monkeypatch,
                                                      tmp_path):
        """DRY=1 runs the cache step's --check, so a missing file still warns."""
        import shutil
        import subprocess

        if not shutil.which("bash"):
            pytest.skip("no bash")
        root, _ = _manifest_run(generator, monkeypatch, tmp_path, *SHORT, run_id="dry2")
        # Stands in for embedding_cache --check finding a missing file (exit 1).
        stub = tmp_path / "stub_py"
        stub.write_text('#!/bin/sh\necho "$@" > "$0.args"\nexit 1\n')
        stub.chmod(0o755)
        env = dict(os.environ, DRY="1", FORCE="1", SPREAD="0", RUNS=str(root),
                   CACHE_PY=str(stub))
        done = subprocess.run(["bash", str(PILOT / "submit_runs.sh")], env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr[-2000:]
        assert "WARNING: the embedding cache cannot serve these jobs yet" in done.stderr
        args = (tmp_path / "stub_py.args").read_text().split()
        assert args[:3] == ["-m", "qbiocode.apps.qprofiler.embedding_cache", "--check"]


# ---------------------------------------------------------------------------
# Scaling to the full corpus: --datasets all / --datasets-file, --splits-per-job and
# --wall auto (cost_model's manifest laws).
# ---------------------------------------------------------------------------

class TestDatasetSelectionAtScale:
    def test_all_takes_every_curated_dataset_with_a_manifest(self, generator, monkeypatch,
                                                              tmp_path, capsys):
        with_manifest = {"pmlb__labor", "pmlb__spect", "pmlb__glass2"}
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "all",
                                "--models", "qsvc", "--splits", "1",
                                with_manifest=with_manifest)
        assert {r["dataset"] for r in rows} == with_manifest
        assert {r["sel_datasets"] for r in rows} == {"all"}
        out = capsys.readouterr().out
        assert "skipping pmlb__appendicitis: no manifest" in out

    @pytest.mark.parametrize("kind", ["lines", "inventory"])
    def test_a_datasets_file_lists_them(self, generator, monkeypatch, tmp_path, kind):
        listing = tmp_path / ("ids.txt" if kind == "lines" else "inventory.csv")
        if kind == "lines":
            listing.write_text("# frozen membership\npmlb__labor\n\nspect   # by name\n")
        else:
            listing.write_text("dataset_id,n,p\npmlb__labor,60,16\npmlb__spect,60,22\n")
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets-file",
                                str(listing), "--models", "qsvc", "--splits", "1")
        assert {r["dataset"] for r in rows} == {"pmlb__labor", "pmlb__spect"}
        assert rows[0]["sel_datasets"] == f"file:{listing}"

    def test_datasets_and_a_datasets_file_are_alternatives(self, generator, monkeypatch,
                                                             tmp_path):
        listing = tmp_path / "ids.txt"
        listing.write_text("pmlb__labor\n")
        with pytest.raises(SystemExit):
            _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                          "--datasets-file", str(listing))

    def test_a_datasets_file_is_refused_in_internal_mode(self, generator, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv", ["generate_pilot_configs.py", "--datasets-file", "x"])
        with pytest.raises(SystemExit):
            generator.main()


class TestSplitsPerJob:
    @pytest.fixture
    def batched(self, generator, monkeypatch, tmp_path):
        return _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor,spect",
                             "--models", "qsvc", "--splits", "all",
                             "--splits-per-job", "classical=all,*=1")

    def test_the_classical_group_runs_every_split_in_one_job(self, batched):
        root, rows = batched
        classical = [r for r in rows if r["group"] == "classical"]
        quantum = [r for r in rows if r["group"] == "qsvc"]
        # labor: none; spect: pca + umap -> 3 passes; 15 splits each.
        assert len(classical) == 3 and len(quantum) == 3 * 15
        for r in classical:
            assert r["iteration"] == ";".join(str(i) for i in range(1, 16))
            assert r["n_splits"] == "15"
            assert r["config"].endswith("_i01-15_classical")
            cfg = _load(pathlib.Path(r["yaml"]))
            assert cfg["splits"] == list(range(1, 16))
        assert all(r["n_splits"] == "1" and ";" not in r["iteration"] for r in quantum)
        assert len({r["config"] for r in rows}) == len(rows)

    def test_status_expects_every_split_of_a_batched_job(self, batched):
        root, rows = batched
        status = _module("status")
        r = next(r for r in rows if r["group"] == "classical")
        cfg = status.read_config(r["yaml"])
        assert cfg["expected"] == 15 * len(_load(pathlib.Path(r["yaml"]))["classical_model"])

    def test_chunks_are_consecutive_and_cover_every_split(self, generator):
        assert generator.chunk_splits([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
        assert generator.chunk_splits([1, 6, 11], None) == [[1, 6, 11]]
        assert generator.manifest_job_name("d", "pca", [6, 11], "classical") == "d_pca_i06-11_classical"
        assert generator.manifest_job_name("d", "pca", 6, "qsvc") == "d_pca_i06_qsvc"

    def test_the_parser(self, generator):
        groups = ["qsvc", "classical"]
        assert generator.parse_splits_per_job("1", groups) == {"qsvc": 1, "classical": 1}
        assert generator.parse_splits_per_job("classical=all", groups) == {"qsvc": 1, "classical": None}
        assert generator.parse_splits_per_job("classical=5,*=2", groups) == {"qsvc": 2, "classical": 5}
        for bad in ("0", "x", "nope=2", "classical"):
            with pytest.raises(ValueError):
                generator.parse_splits_per_job(bad, groups)


class TestAutoWalls:
    def test_auto_fills_walls_and_expected_hours(self, generator, monkeypatch, tmp_path, capsys):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor,spect",
                                "--models", "qsvc,pqk", "--splits", "1-2", "--wall",
                                "classical=1:00,*=auto")
        cm = _module("cost_model")
        for r in rows:
            if r["group"] == "classical":
                assert r["wall"] == "1:00"
            else:
                bk = "mps" if r["backend"] == "mps_simulator" else "sv"
                exp = cm.manifest_job_hours(r["group"], bk, int(r["qubits"]), int(r["rows"]),
                                            k=5, n_trials=30, n_splits=1)
                assert float(r["exp_h"]) == pytest.approx(exp, abs=1e-3)
                assert r["wall"] == cm.auto_wall(exp)[0]
            assert float(r["bound_h"]) > 0
        assert "expected (cost model)" in capsys.readouterr().out

    def test_the_wall_parser_takes_auto(self, generator):
        groups = ["qsvc", "classical"]
        assert generator.parse_wall("auto", groups) == {"qsvc": "auto", "classical": "auto"}
        assert generator.parse_wall("classical=0:45,*=auto", groups) == {
            "qsvc": "auto", "classical": "0:45"}
        with pytest.raises(ValueError):
            generator.parse_wall("automatic", groups)


class TestTheManifestCostModel:
    """The laws reproduce the controlled run they were fitted on (ctrl1, 2026-10-05)."""

    #: (group, bk, qubits, rows, median measured seconds of the ctrl1 jobs)
    CTRL1 = [("classical", "sv", 8, 62, 167.0), ("classical", "sv", 8, 267, 241.5),
             ("classical", "mps", 16, 57, 173.0), ("pqk", "sv", 8, 62, 260.5),
             ("pqk", "sv", 8, 267, 750.0), ("pqk", "mps", 16, 57, 155.0),
             ("qsvc", "sv", 8, 62, 127.0), ("qsvc", "sv", 8, 267, 150.5),
             ("qsvc", "mps", 16, 57, 3681.0), ("qnn", "sv", 8, 62, 750.0),
             ("qnn", "sv", 8, 267, 2620.5), ("qnn", "mps", 16, 57, 1334.0)]

    @pytest.mark.parametrize("group, bk, q, rows, measured", CTRL1)
    def test_within_ten_percent_of_the_measured_jobs(self, group, bk, q, rows, measured):
        cm = _module("cost_model")
        expected = cm.manifest_job_hours(group, bk, q, rows, k=5, n_trials=30) * 3600
        assert expected == pytest.approx(measured, rel=0.10)

    def test_batching_adds_splits_not_startups(self):
        cm = _module("cost_model")
        one = cm.manifest_job_hours("classical", "sv", 8, 267)
        fifteen = cm.manifest_job_hours("classical", "sv", 8, 267, n_splits=15)
        assert fifteen == pytest.approx(cm.JOB_OVERHEAD_S / 3600 + 15 * (one - cm.JOB_OVERHEAD_S / 3600))

    def test_auto_wall_rounds_up_and_caps(self):
        cm = _module("cost_model")
        assert cm.auto_wall(0.01) == ("0:30", False)              # floor
        assert cm.auto_wall(1.0) == ("3:15", False)               # 3 x 1 h + 15 min
        assert cm.auto_wall(1.01) == ("3:30", False)              # rounded up to 15 min
        assert cm.auto_wall(100.0) == ("72:00", True)             # capped, and says so

    def test_extrapolation_is_flagged(self):
        cm = _module("cost_model")
        assert not cm.manifest_extrapolated("sv", 8, 267)
        assert cm.manifest_extrapolated("sv", 8, 2600)
        assert cm.manifest_extrapolated("mps", 20, 100)


class TestEmbedAboveAndTheDefaultArms:
    """The full run's choices: embed every width the statevector cannot take, no qnn."""

    def test_without_models_the_arms_are_qsvc_pqk_and_the_classical_group(
            self, generator, monkeypatch, tmp_path):
        _, rows = _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                                "--splits", "1")
        assert {r["group"] for r in rows} == {"classical", "qsvc", "pqk"}

    def test_qnn_is_still_there_when_named(self, generator):
        assert [g for g, _, _ in generator.model_groups(["qsvc", "pqk", "qnn"])] == [
            "qsvc", "pqk", "qnn", "classical"]

    def test_embed_above_13_moves_labor_from_mps_to_eight_embedded_qubits(
            self, generator, monkeypatch, tmp_path):
        args = ("--datasets", "labor,appendicitis", "--splits", "1")
        _, default = _manifest_run(generator, monkeypatch, tmp_path / "a", *args)
        root, rows = _manifest_run(generator, monkeypatch, tmp_path / "b", *args,
                                   "--embed-above", "13")
        labor = lambda rs: {(r["embedding"], r["backend"], r["qubits"]) for r in rs
                            if r["dataset"] == "pmlb__labor"}
        assert labor(default) == {("none", "mps_simulator", "16")}
        assert labor(rows) == {("pca", "statevector_simulator", "8"),
                               ("umap", "statevector_simulator", "8")}
        # appendicitis (7 features) is untouched, and the jobs tell QProfiler the same rule.
        assert {r["embedding"] for r in rows if r["dataset"] == "pmlb__appendicitis"} == {"none"}
        assert {r["embed_above"] for r in rows} == {"13"}
        cfg = _load(next(root.glob("pmlb__labor/*pca*qsvc*.yaml")))
        assert cfg["embedding_min_features"] == 13

    def test_embed_above_below_the_embedding_width_is_refused(self, generator, monkeypatch,
                                                              tmp_path):
        with pytest.raises(SystemExit):
            _manifest_run(generator, monkeypatch, tmp_path, "--datasets", "labor",
                          "--embed-above", "7")
