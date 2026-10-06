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

"""QProfiler under ``split_mode: manifest``: the outer folds come from a frozen manifest.

What these tests hold:

  1. The config is checked for the protocol before any data is read (tuning on for both
     sides, no freeze, no grid tuner, a split directory), and ``iter``/``test_size`` stop
     being required.
  2. A dataset is run only against the manifest computed from its exact bytes.
  3. Each pass uses the manifest's rows: the final stage the training and test fold, the
     tuning stage the fit and validation rows, each scaled (and embedded) on its first
     side only; ``splits`` selects which outer splits run.
  4. Each pass records the PROTOCOL_COLUMNS, its row ids and its sidecars (oof, trials,
     val_predictions), and the out-of-fold predictions partition every repeat.
  5. The embedding cache keeps one file per stage, and never serves a file of one split
     mode to a run in the other.
  6. Internal mode is untouched: no new columns, keys, files or model_run arguments.

model_run is stubbed where a real fit would only add time; one test runs the real lr/nb
arms once model_run accepts ``validation=``.
"""

import inspect
import json
import logging
import pickle
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold

from qbiocode.apps.qprofiler import split_manifest as sm
from qbiocode.evaluation import protocol

emb_cache = import_module("qbiocode.apps.qprofiler.embedding_cache")
qp = import_module("qbiocode.apps.qprofiler.qprofiler")

SHIPPED_CONFIG = qp.__file__.rsplit("/", 1)[0] + "/configs/config.yaml"
LOG = logging.getLogger("test")
K, R = 3, 2


def _write_dataset(directory, name="tiny.csv", rows=36, seed=0):
    """A learnable binary dataset in curate's layout: features, then ``label`` last."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=rows) > 0).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame["label"] = y
    path = directory / name
    frame.to_csv(path, index=False)
    return path, y


def _write_manifest(split_dir, csv, y, k=K, repeats=R, seed=42, **overrides):
    """A schema-2 manifest of ``csv`` in split_manifest's format (next_fold validation)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)
    folds = []
    for r in range(repeats):
        tests = [np.sort(te) for _, te in StratifiedKFold(
            k, shuffle=True, random_state=seed + r).split(np.zeros(n), y)]
        for f in range(k):
            folds.append({
                "repeat": r, "fold": f,
                "train": np.setdiff1d(np.arange(n), tests[f]).tolist(),
                "val": tests[(f + 1) % k].tolist(), "test": tests[f].tolist(),
            })
    payload = {
        "schema_version": 2, "dataset_id": csv.stem, "sha256": sm.file_sha256(csv),
        "n": n, "k": k, "n_repeats": repeats, "protocol": "StratifiedKFold",
        "validation": "next_fold", "seed": seed,
        "repeat_seeds": [seed + r for r in range(repeats)], "group_col": None,
        "generator_version": "make_splits/2.0", "folds": folds, **overrides,
    }
    path = sm.manifest_path(split_dir, csv)
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def tree(tmp_path):
    """``(data_dir, split_dir, csv, y)``: one tiny dataset and its manifest."""
    csv, y = _write_dataset(tmp_path / "data")
    _write_manifest(tmp_path / "splits", csv, y)
    return tmp_path / "data", tmp_path / "splits", csv, y


def _args(data_dir, splits_root, **overrides):
    """A manifest-mode config: the shipped one, tiny, with one classical model."""
    config = OmegaConf.load(SHIPPED_CONFIG)
    settings = {
        "config_file_name": "manifest_mode_test",
        "folder_path": str(data_dir), "file_dataset": "ALL",
        "embeddings": ["none"], "embedding_min_features": 0, "n_components": 2,
        "n_neighbors": 5, "model": ["lr"], "n_jobs": 1,
        "grid_search": True, "tune_quantum": False, "freeze_quantum_params": False,
        "tuner": "optuna", "n_trials": 2, "seed": 7, "embedding_cache": None,
        "split_mode": "manifest", "split_dir": str(splits_root), "splits": "all",
    }
    settings.update(overrides)
    for key, value in settings.items():
        config[key] = value
    return config


def _validated(args):
    qp._resolve_model_lists(args, LOG)
    qp._resolve_backend_alias(args, LOG)
    return qp._validate_config(args, LOG)


class StubModelRun:
    """model_run's return shape for one model, 'lr', recording what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, X_train, X_test, y_train, y_test, data_key, args, **kwargs):
        self.calls.append({"data_key": data_key, "n_train": len(X_train),
                           "n_test": len(X_test), "kwargs": kwargs})
        y_pred = (X_test[:, 0] > np.median(X_train[:, 0])).astype(int)
        out = {
            "results_lr": {0: {"model": "lr", "accuracy": float(np.mean(y_pred == y_test))}},
            "y_test_lr": {0: np.asarray(y_test)},
            "y_predicted_lr": {0: y_pred},
            "y_score_lr": {0: X_test[:, 0].astype(float)},
        }
        vs = kwargs.get("validation")
        if vs is not None:
            records = [
                protocol.TrialRecord(t, {"C": float(t + 1)}, 0.5 + 0.1 * t,
                                     is_default=(t == 0),
                                     y_pred=np.zeros(vs.n_val, dtype=int),
                                     y_score=(np.asarray(vs.y_val, float) if t == 1 else None))
                for t in range(2)
            ]
            out["trials_lr"] = {0: protocol.trial_log(
                records, metric="balanced_accuracy", best=1,
                val_idx=vs.val_idx, y_val=vs.y_val)}
        return out


def _stub_evaluate(df, y, file, *a, **k):
    return pd.DataFrame([{"Dataset": file, "n_rows_evaluated": len(df)}])


@pytest.fixture
def stubbed(monkeypatch):
    stub = StubModelRun()
    monkeypatch.setattr(qp, "model_run", stub)
    monkeypatch.setattr(qp, "evaluate", _stub_evaluate)
    return stub


def _run(args, work_dir, monkeypatch):
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)
    qp.main(args)
    return work_dir


# ------------------------------------------------------------------ 1. the config
class TestTheConfig:
    def test_manifest_mode_needs_neither_iter_nor_test_size(self, tree):
        data_dir, split_dir, _, _ = tree
        args = _args(data_dir, split_dir)
        OmegaConf.set_struct(args, False)
        for key in ("iter", "test_size", "stratify"):
            del args[key]
        assert _validated(args) == "MinMaxScaler"

    def test_internal_mode_still_needs_them(self, tree):
        data_dir, split_dir, _, _ = tree
        args = _args(data_dir, split_dir, split_mode="internal")
        OmegaConf.set_struct(args, False)
        del args["iter"]
        with pytest.raises(ValueError, match="missing required key"):
            _validated(args)

    @pytest.mark.parametrize("overrides, message", [
        ({"grid_search": False}, "grid_search: True"),
        ({"freeze_quantum_params": True}, "freeze_quantum_params must be False"),
        ({"tuner": "grid"}, "tuner: optuna"),
        ({"model": ["lr", "qsvc"], "tune_quantum": False}, "tune_quantum: True"),
        ({"split_dir": None}, "needs split_dir"),
        ({"split_dir": "/nonexistent/qbc_splits"}, "is not a directory"),
        ({"splits": [0]}, "splits must be"),
        ({"splits": "three"}, "splits must be"),
        ({"split_mode": "kfold"}, "split_mode must be one of"),
        ({"embeddings": ["pca", "spectral"]}, r"inductive embeddings; \['spectral'\]"),
    ])
    def test_protocol_breaking_settings_are_refused(self, tree, overrides, message):
        data_dir, split_dir, _, _ = tree
        with pytest.raises(ValueError, match=message):
            _validated(_args(data_dir, split_dir, **overrides))

    def test_quantum_models_with_tune_quantum_pass(self, tree):
        data_dir, split_dir, _, _ = tree
        _validated(_args(data_dir, split_dir, model=["lr", "qsvc"], tune_quantum=True))

    @pytest.mark.parametrize("value, expected", [
        ("all", "all"), (None, "all"), (3, [3]), ([2, 5], [2, 5]), ("1, 4", [1, 4]),
    ])
    def test_splits_selection_forms(self, value, expected):
        assert qp._split_selection({"splits": value}) == expected

    def test_the_shipped_default_is_internal(self):
        config = OmegaConf.load(SHIPPED_CONFIG)
        assert config["split_mode"] == "internal"
        assert qp._split_mode(config) == "internal"
        assert qp._split_mode({}) == "internal"

    def test_a_single_file_dataset_string_is_matched_exactly(self, tmp_path):
        for name in ("heart.csv", "art.csv"):
            (tmp_path / name).write_text("a,label\n1,0\n")
        assert qp._input_files({"file_dataset": "heart.csv"}, str(tmp_path)) == ["heart.csv"]
        assert qp._input_files({"file_dataset": ["art.csv"]}, str(tmp_path)) == ["art.csv"]


# ------------------------------------------------------------------ 2. the dataset check
class TestTheDatasetCheck:
    def test_a_matching_dataset_is_accepted(self, tree):
        data_dir, split_dir, csv, y = tree
        args = _args(data_dir, split_dir)
        manifest = qp._dataset_manifest(args, str(csv), len(y), y)
        assert manifest.iterations == list(range(1, K * R + 1))

    def test_a_changed_csv_is_refused(self, tree):
        data_dir, split_dir, csv, y = tree
        frame = pd.read_csv(csv)
        frame.iloc[0, 0] += 1.0
        frame.to_csv(csv, index=False)
        with pytest.raises(sm.ManifestError, match="has changed since its splits were frozen"):
            qp._dataset_manifest(_args(data_dir, split_dir), str(csv), len(y), y)

    def test_a_dataset_without_manifest_is_refused(self, tree, tmp_path):
        data_dir, _, csv, y = tree
        empty = tmp_path / "no_splits"
        empty.mkdir()
        with pytest.raises(sm.ManifestError, match="has no split manifest"):
            qp._dataset_manifest(_args(data_dir, empty), str(csv), len(y), y)

    def test_a_selection_outside_the_manifest_is_refused(self, tree):
        data_dir, split_dir, csv, y = tree
        with pytest.raises(sm.ManifestError, match="no split with iteration 99"):
            qp._dataset_manifest(_args(data_dir, split_dir, splits=[99]), str(csv), len(y), y)

    def test_main_refuses_a_changed_csv_before_fitting(self, tree, stubbed, tmp_path,
                                                       monkeypatch):
        data_dir, split_dir, csv, _ = tree
        frame = pd.read_csv(csv)
        frame.iloc[3, 2] = 0.5
        frame.to_csv(csv, index=False)
        with pytest.raises(sm.ManifestError, match="has changed"):
            _run(_args(data_dir, split_dir), tmp_path / "run", monkeypatch)
        assert not stubbed.calls


# ------------------------------------------------------------------ 3. the rows
class TestTheSplit:
    def test_final_and_tune_stages_use_the_manifest_rows_scaled_on_their_first_side(
            self, tree):
        data_dir, split_dir, csv, y = tree
        X = pd.read_csv(csv).iloc[:, :-1].to_numpy()
        manifest = sm.load_manifest(sm.manifest_path(split_dir, csv))
        split = manifest.split(4)
        args = _args(data_dir, split_dir)
        for stage, (a, b) in (("final", (split.train_idx, split.test_idx)),
                              ("tune", (split.fit_idx, split.val_idx))):
            Xa, Xb, ya, yb, ia, ib = qp._split_and_scale(
                X, y, args, 4, "MinMaxScaler", split=split, stage=stage)
            np.testing.assert_array_equal(ia, a)
            np.testing.assert_array_equal(ib, b)
            assert np.all(np.diff(ia) > 0) and np.all(np.diff(ib) > 0)
            np.testing.assert_array_equal(ya, y[a])
            np.testing.assert_array_equal(yb, y[b])
            # The scaler saw the first side only: it spans exactly [0, 1] there.
            np.testing.assert_allclose(Xa.min(axis=0), 0.0, atol=1e-12)
            np.testing.assert_allclose(Xa.max(axis=0), 1.0, atol=1e-12)
            lo, hi = X[a].min(axis=0), X[a].max(axis=0)
            np.testing.assert_allclose(Xb, (X[b] - lo) / (hi - lo))

    def test_an_unknown_stage_is_refused(self, tree):
        data_dir, split_dir, csv, y = tree
        X = pd.read_csv(csv).iloc[:, :-1].to_numpy()
        split = sm.load_manifest(sm.manifest_path(split_dir, csv)).split(1)
        with pytest.raises(ValueError, match="stage"):
            qp._split_and_scale(X, y, {}, 1, "None", split=split, stage="val")


# ------------------------------------------------------------------ 4. the outputs
class TestAManifestRun:
    @pytest.fixture
    def run(self, tree, stubbed, tmp_path, monkeypatch):
        data_dir, split_dir, csv, y = tree
        monkeypatch.setenv("LSB_JOBID", "4242")
        work = _run(_args(data_dir, split_dir), tmp_path / "run", monkeypatch)
        manifest = sm.load_manifest(sm.manifest_path(split_dir, csv))
        return work, manifest, stubbed, y

    def test_model_run_gets_the_validation_split_of_each_fold(self, run):
        _, manifest, stub, y = run
        assert [c["data_key"] for c in stub.calls] == [
            f"tiny_none_2_{it}" for it in manifest.iterations]
        for call, split in zip(stub.calls, manifest.splits):
            vs = call["kwargs"]["validation"]
            assert isinstance(vs, protocol.ValidationSplit)
            np.testing.assert_array_equal(vs.fit_idx, split.fit_idx)
            np.testing.assert_array_equal(vs.val_idx, split.val_idx)
            np.testing.assert_array_equal(vs.y_val, y[split.val_idx])
            assert not np.intersect1d(vs.fit_idx, split.test_idx).size
            assert (call["n_train"], call["n_test"]) == (split.train_idx.size,
                                                        split.test_idx.size)

    def test_rows_carry_the_protocol_columns(self, run):
        work, manifest, _, _ = run
        rows = pd.read_csv(work / "ModelResults.csv", keep_default_na=False)
        assert set(protocol.PROTOCOL_COLUMNS) <= set(rows.columns)
        assert rows["iteration"].tolist() == manifest.iterations
        assert rows["repeat"].tolist() == [s.repeat for s in manifest.splits]
        assert rows["fold"].tolist() == [s.fold for s in manifest.splits]
        assert set(rows["split_mode"]) == {"manifest"}
        assert set(rows["manifest_sha256"]) == {manifest.file_sha256}
        assert set(rows["dataset_sha256"]) == {manifest.sha256}
        assert rows["embed_seed"].tolist() == [7 + it for it in manifest.iterations]
        assert set(rows["lsf_jobid"].astype(str)) == {"4242"}
        assert rows["host"].astype(str).str.len().min() > 0

    def test_rows_carry_the_chosen_trials_validation_tiebreak(self, run):
        work, _, _, _ = run
        rows = pd.read_csv(work / "ModelResults.csv")
        assert set(protocol.TIEBREAK_COLUMNS) <= set(rows.columns)
        # The stub's chosen trial scores its validation rows with their own labels.
        assert np.allclose(rows["val_auc"], 1.0)
        assert np.allclose(rows["val_log_loss"], 0.0, atol=1e-9)

    def test_results_pkl_keeps_the_row_ids_and_fields(self, run):
        work, manifest, _, _ = run
        with open(work / "results.pkl", "rb") as handle:
            results = pickle.load(handle)
        assert len(results) == len(manifest.splits)
        for summary, split in zip(results, manifest.splits):
            for key in ("train_idx", "fit_idx", "val_idx", "test_idx"):
                np.testing.assert_array_equal(summary[key], getattr(split, key))
            assert summary["repeat"] == split.repeat and summary["fold"] == split.fold
            assert summary["iteration"] == split.iteration
            assert "trials_lr" in summary

    def test_oof_predictions_partition_every_repeat(self, run):
        work, manifest, _, y = run
        oof = pd.concat(pd.read_csv(p) for p in sorted((work / protocol.OOF_DIR).glob("*.csv")))
        assert list(oof.columns) == list(protocol.OOF_COLUMNS)
        oof["iteration"] = oof["data_key"].str.rsplit("_", n=1).str[1].astype(int)
        oof["repeat"] = (oof["iteration"] - 1) // manifest.k
        for (_, model), rows in oof.groupby(["repeat", "model"]):
            assert sorted(rows["row_id"]) == list(range(manifest.n))
            np.testing.assert_array_equal(rows["y_true"].to_numpy(), y[rows["row_id"]])

    def test_trials_and_validation_predictions_are_written(self, run):
        work, manifest, _, _ = run
        for split in manifest.splits:
            name = f"tiny_none_2_{split.iteration}.csv"
            trials = pd.read_csv(work / protocol.TRIALS_DIR / name)
            assert list(trials.columns) == list(protocol.TRIAL_COLUMNS)
            assert trials["trial"].tolist() == [0, 1]
            assert trials["is_default"].tolist() == [True, False]
            val = pd.read_csv(work / protocol.VAL_PREDICTIONS_DIR / name)
            assert sorted(set(val["row_id"])) == split.val_idx.tolist()
        assert not list(work.rglob("*.tmp"))

    def test_splits_selects_the_outer_splits_that_run(self, tree, stubbed, tmp_path,
                                                      monkeypatch):
        data_dir, split_dir, _, _ = tree
        work = _run(_args(data_dir, split_dir, splits=[5, 2]), tmp_path / "sel", monkeypatch)
        assert [c["data_key"] for c in stubbed.calls] == ["tiny_none_2_5", "tiny_none_2_2"]
        rows = pd.read_csv(work / "ModelResults.csv")
        assert rows["iteration"].tolist() == [5, 2]
        assert sorted(p.name for p in (work / protocol.OOF_DIR).iterdir()) == [
            "tiny_none_2_2.csv", "tiny_none_2_5.csv"]


# ------------------------------------------------------------------ 6. internal mode
def test_internal_mode_writes_no_protocol_output(tree, stubbed, tmp_path, monkeypatch):
    data_dir, split_dir, _, _ = tree
    args = _args(data_dir, split_dir, split_mode="internal", iter=2, grid_search=False)
    work = _run(args, tmp_path / "internal", monkeypatch)
    assert all(call["kwargs"] == {} for call in stubbed.calls)
    assert len(stubbed.calls) == 2
    rows = pd.read_csv(work / "ModelResults.csv")
    assert not set(protocol.PROTOCOL_COLUMNS) & set(rows.columns)
    assert not set(protocol.TIEBREAK_COLUMNS) & set(rows.columns)
    with open(work / "results.pkl", "rb") as handle:
        results = pickle.load(handle)
    for summary in results:
        assert not set(qp._SPLIT_INDEX_KEYS) & set(summary)
        assert not set(protocol.PROTOCOL_COLUMNS) & set(summary)
    for directory in (protocol.OOF_DIR, protocol.TRIALS_DIR, protocol.VAL_PREDICTIONS_DIR):
        assert not (work / directory).exists()


# ------------------------------------------------------------------ 5. the cache
class TestTheEmbeddingCache:
    @pytest.fixture
    def cached(self, tree, tmp_path):
        data_dir, split_dir, csv, y = tree
        cache = tmp_path / "cache"
        args = _args(data_dir, split_dir, embeddings=["pca", "none"], splits=[1, 4],
                     embedding_cache=str(cache))
        config = tmp_path / "job.yaml"
        OmegaConf.save(args, config)
        lines = []
        assert emb_cache.precompute([str(config)], out=lines.append) == 0
        return args, config, cache, lines

    def test_precompute_writes_both_stages_of_the_selected_splits_only(self, cached):
        _, _, cache, lines = cached
        assert sorted(p.name for p in cache.iterdir()) == sorted(
            f"emb_tiny_pca_2_{it}{suffix}.npz" for it in (1, 4) for suffix in ("", "__tune"))
        assert lines[-1] == "4 written, 0 already current, 0 problems"

    def test_the_stage_files_hold_their_rows_and_spec(self, cached, tree):
        _, split_dir, csv, _ = tree
        _, _, cache, _ = cached
        split = sm.load_manifest(sm.manifest_path(split_dir, csv)).split(4)
        final = emb_cache.read(str(cache / "emb_tiny_pca_2_4.npz"))
        tune = emb_cache.read(str(cache / "emb_tiny_pca_2_4__tune.npz"))
        np.testing.assert_array_equal(final["train_idx"], split.train_idx)
        np.testing.assert_array_equal(final["test_idx"], split.test_idx)
        np.testing.assert_array_equal(tune["train_idx"], split.fit_idx)
        np.testing.assert_array_equal(tune["test_idx"], split.val_idx)
        for entry, stage in ((final, "final"), (tune, "tune")):
            spec = entry["spec"]
            assert spec["stage"] == stage and spec["split_mode"] == "manifest"
            assert (spec["repeat"], spec["fold"], spec["embed_seed"]) == (1, 0, 7 + 4)
            assert not {"split_seed", "test_size", "stratify"} & set(spec)

    def test_precompute_check_lists_without_writing(self, cached):
        _, config, cache, _ = cached
        (cache / "emb_tiny_pca_2_1__tune.npz").unlink()
        lines = []
        assert emb_cache.precompute([str(config)], check_only=True, out=lines.append) == 1
        assert any(line.startswith("MISSING  emb_tiny_pca_2_1__tune.npz") for line in lines)
        assert not (cache / "emb_tiny_pca_2_1__tune.npz").exists()

    def test_main_reads_both_stages_from_the_cache(self, cached, stubbed, tmp_path,
                                                   monkeypatch):
        args, _, cache, _ = cached
        seen = []
        real_load = emb_cache.load

        def spy(cache_dir, key, spec, a, b):
            seen.append(key)
            return real_load(cache_dir, key, spec, a, b)

        monkeypatch.setattr(emb_cache, "load", spy)
        _run(args, tmp_path / "cached_run", monkeypatch)
        assert seen == ["tiny_pca_2_1", "tiny_pca_2_1__tune",
                        "tiny_pca_2_4", "tiny_pca_2_4__tune"]
        # The tuning features handed to model_run are the cached ones.
        tune = emb_cache.read(str(cache / "emb_tiny_pca_2_4__tune.npz"))
        pca_calls = [c for c in stubbed.calls if c["data_key"] == "tiny_pca_2_4"]
        np.testing.assert_array_equal(pca_calls[0]["kwargs"]["validation"].X_fit,
                                      tune["X_train"])

    def test_a_file_of_the_other_split_mode_is_stale(self, cached, tree):
        data_dir, split_dir, csv, y = tree
        args, _, cache, _ = cached
        # The internal-mode spec of the same data_key differs, so the manifest file at
        # that name is refused rather than served.
        internal = _args(data_dir, split_dir, split_mode="internal", iter=4,
                         embeddings=["pca"], embedding_cache=str(cache))
        scaler = _validated(internal)
        entries = emb_cache.plan(csv.name, sm.file_sha256(csv), 5, internal, scaler)
        assert ("tiny_pca_2_4", entries[3][1]) == entries[3]
        problems = emb_cache.check(str(cache), entries)
        assert any("emb_tiny_pca_2_4.npz was written under other settings" in p
                   for p in problems)
        assert "split_mode" not in entries[3][1]

    def test_manifest_plan_differs_by_stage_and_split(self, tree):
        data_dir, split_dir, csv, y = tree
        args = _args(data_dir, split_dir, embeddings=["pca", "umap", "none"], splits=[2])
        scaler = _validated(args)
        manifest = sm.load_manifest(sm.manifest_path(split_dir, csv))
        entries = emb_cache.plan(csv.name, "sha", 5, args, scaler, manifest=manifest)
        assert [key for key, _ in entries] == [
            "tiny_pca_2_2", "tiny_pca_2_2__tune", "tiny_umap_2_2", "tiny_umap_2_2__tune"]
        specs = [spec for _, spec in entries]
        assert specs[0] != specs[1]
        assert emb_cache.spec_differences(specs[0], specs[1]) == [("stage", "final", "tune")]

    def test_manifest_plan_refuses_a_transductive_embedding(self, tree):
        data_dir, split_dir, csv, y = tree
        args = _args(data_dir, split_dir, embeddings=["pca"], splits=[2])
        scaler = _validated(args)
        args["embeddings"] = ["spectral"]  # past validation, which refuses it first
        manifest = sm.load_manifest(sm.manifest_path(split_dir, csv))
        with pytest.raises(ValueError, match="inductive embeddings"):
            emb_cache.plan(csv.name, "sha", 5, args, scaler, manifest=manifest)


# ------------------------------------------------------------------ a real run
def _accepts_validation():
    """True once model_run and the lr/nb ``_opt`` learners it calls take ``validation=``."""
    from qbiocode.evaluation.model_run import model_run
    from qbiocode.learning.compute_lr import compute_lr_opt
    from qbiocode.learning.compute_nb import compute_nb_opt
    return all("validation" in inspect.signature(f).parameters
               for f in (model_run, compute_lr_opt, compute_nb_opt))


@pytest.mark.skipif(not _accepts_validation(),
                    reason="model_run or the lr/nb learners do not accept validation= yet (G3)")
def test_a_real_lr_nb_run_writes_trials_and_oof(tree, tmp_path, monkeypatch):
    data_dir, split_dir, _, y = tree
    monkeypatch.setattr(qp, "evaluate", _stub_evaluate)
    args = _args(data_dir, split_dir, model=["lr", "nb"], splits=[1, 2, 3], n_trials=3)
    work = _run(args, tmp_path / "real", monkeypatch)
    rows = pd.read_csv(work / "ModelResults.csv")
    assert set(protocol.PROTOCOL_COLUMNS) <= set(rows.columns)
    assert len(rows) == 3 * 2
    assert rows["val_auc"].between(0, 1).all()
    assert rows["val_log_loss"].notna().all() and (rows["val_log_loss"] >= 0).all()
    oof = pd.concat(pd.read_csv(p) for p in (work / protocol.OOF_DIR).glob("*.csv"))
    for _, model_rows in oof.groupby("model"):
        assert sorted(model_rows["row_id"]) == list(range(len(y)))
    trials = pd.concat(pd.read_csv(p) for p in (work / protocol.TRIALS_DIR).glob("*.csv"))
    assert trials["model"].nunique() == 2
    per_model = trials.groupby(["data_key", "model"])
    assert (per_model["trial"].count() >= 2).all()
    # Trial 0 of every arm is its default configuration.
    assert per_model.apply(lambda t: bool(t.loc[t["trial"] == 0, "is_default"].all())).all()


class TestTiebreakScores:
    def test_auc_matches_sklearn_with_tied_scores(self):
        from sklearn.metrics import log_loss, roc_auc_score
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 40)
        s = np.round(rng.random(40), 1)          # coarse, so many scores tie
        out = protocol.binary_validation_scores(y, s)
        assert out["val_auc"] == pytest.approx(roc_auc_score(y, s))
        assert out["val_log_loss"] == pytest.approx(log_loss(y, np.clip(s, 1e-15, 1 - 1e-15)))

    def test_decision_scores_get_an_auc_but_no_log_loss(self):
        from sklearn.metrics import roc_auc_score
        y = np.array([0, 0, 1, 1, 0, 1])
        s = np.array([-2.0, -0.5, 0.3, 4.0, 0.1, -0.1])
        out = protocol.binary_validation_scores(y, s)
        assert out["val_auc"] == pytest.approx(roc_auc_score(y, s))
        assert np.isnan(out["val_log_loss"])

    def test_string_labels_take_the_larger_class_as_positive(self):
        out = protocol.binary_validation_scores(np.array(["no", "yes", "yes", "no"]),
                                                np.array([0.1, 0.9, 0.8, 0.2]))
        assert out["val_auc"] == pytest.approx(1.0)

    @pytest.mark.parametrize("y, s", [(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])),
                                      (np.array([0, 1]), np.array([np.nan, 0.5])),
                                      (np.array([0, 1, 1]), np.array([0.1, 0.9]))])
    def test_one_class_nan_or_misaligned_scores_give_nan(self, y, s):
        out = protocol.binary_validation_scores(y, s)
        assert np.isnan(out["val_auc"]) and np.isnan(out["val_log_loss"])

    def _log(self):
        y_val = np.array([0, 1, 0, 1])
        records = [protocol.TrialRecord(0, {"C": 1.0}, 0.5, y_pred=np.array([1, 0, 1, 0]),
                                        y_score=np.array([0.9, 0.1, 0.8, 0.2])),
                   protocol.TrialRecord(1, {"C": 2.0}, 0.9, y_pred=np.array([0, 1, 0, 1]),
                                        y_score=np.array([0.1, 0.9, 0.2, 0.8]))]
        return protocol.trial_log(records, metric="balanced_accuracy", best=1,
                                  val_idx=np.arange(4), y_val=y_val)

    def test_the_chosen_trial_is_scored(self):
        out = protocol.validation_tiebreak({0: self._log()})
        assert out["val_auc"] == pytest.approx(1.0)
        nan = protocol.validation_tiebreak(None)
        assert set(nan) == set(protocol.TIEBREAK_COLUMNS) and all(np.isnan(v) for v in nan.values())

    def test_sidecars_give_the_same_scores_as_the_log(self):
        trials, preds = protocol.trial_frames({"trials_lr": {0: self._log()}}, "k1")
        tb = protocol.tiebreak_from_sidecars(trials, preds)
        assert list(tb.columns) == ["data_key", "model", *protocol.TIEBREAK_COLUMNS]
        row = tb.iloc[0]
        direct = protocol.validation_tiebreak(self._log())
        assert (row["data_key"], row["model"]) == ("k1", "lr")
        assert row["val_auc"] == pytest.approx(direct["val_auc"])
        assert row["val_log_loss"] == pytest.approx(direct["val_log_loss"])
