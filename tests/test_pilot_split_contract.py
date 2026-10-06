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
