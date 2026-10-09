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

"""``skip_existing``: a resumed run adopts the cells an earlier run already computed.

What these tests hold:

  1. ``skip_existing`` resolves to the earlier run directories of this config, to a given
     absolute directory, or to nothing; a relative path is an error, raised during
     validation rather than after the first pass has been fitted.
  2. The index is keyed by the label ``model_run`` files a model's results under, which is
     the name plus ``'_opt'`` exactly where that function appends one. Get this wrong and
     a resume adopts nothing while reporting success.
  3. A second run of a complete config fits NOTHING and still writes a complete
     ModelResults.csv -- the rows copied over verbatim, so ``status.py`` calls the config
     done and ``collate_results.py`` sees one complete run directory.
  4. A second run of a half-done config fits only the models that are missing, and its
     table holds both halves.
  5. Under ``split_mode: manifest`` the adopted models' oof/, trials/ and
     val_predictions/ rows come across too, so the sidecars match the table.
  6. A row computed from other dataset bytes, or under other frozen splits, is REFUSED and
     named -- the one case where resuming would merge two experiments.
  7. Off by default: without the key nothing is adopted and every pass is recomputed.

model_run is stubbed throughout: what is under test is which models a pass asks for and
what lands on disk, not any fit.
"""

import csv
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

qp = import_module("qbiocode.apps.qprofiler.qprofiler")
resume = import_module("qbiocode.apps.qprofiler.resume")

SHIPPED_CONFIG = qp.__file__.rsplit("/", 1)[0] + "/configs/config.yaml"
LOG = logging.getLogger("test")
K, R = 3, 1


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


def _write_manifest(split_dir, csv_path, y, k=K, repeats=R, seed=42):
    """A schema-2 manifest of ``csv_path``, next_fold validation."""
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
        "schema_version": 2, "dataset_id": csv_path.stem, "sha256": sm.file_sha256(csv_path),
        "n": n, "k": k, "n_repeats": repeats, "protocol": "StratifiedKFold",
        "validation": "next_fold", "seed": seed,
        "repeat_seeds": [seed + r for r in range(repeats)], "group_col": None,
        "generator_version": "make_splits/2.0", "folds": folds,
    }
    path = sm.manifest_path(split_dir, csv_path)
    path.write_text(json.dumps(payload))
    return path


def _args(data_dir, **overrides):
    """A tiny internal-mode config, two classical models, two splits."""
    config = OmegaConf.load(SHIPPED_CONFIG)
    settings = {
        "config_file_name": "skip_existing_test",
        "folder_path": str(data_dir), "file_dataset": "ALL",
        "embeddings": ["none"], "embedding_min_features": 0, "n_components": 2,
        "n_neighbors": 5, "model": ["lr", "nb"], "n_jobs": 1,
        "grid_search": True, "tune_quantum": False, "freeze_quantum_params": False,
        "tuner": "optuna", "n_trials": 2, "n_trials_quantum": 2, "seed": 7,
        "embedding_cache": None, "split_mode": "internal", "iter": 2, "test_size": 0.3,
        "skip_existing": False,
    }
    settings.update(overrides)
    OmegaConf.set_struct(config, False)
    for key, value in settings.items():
        config[key] = value
    return config


def _manifest_args(data_dir, split_dir, **overrides):
    """The same, under ``split_mode: manifest``."""
    settings = {
        "split_mode": "manifest", "split_dir": str(split_dir), "splits": "all",
        "grid_search": True, "tune_quantum": False, "freeze_quantum_params": False,
    }
    settings.update(overrides)
    args = _args(data_dir, **settings)
    for key in ("iter", "test_size"):
        if key in args:
            del args[key]
    return args


class StubModelRun:
    """model_run's return shape for whatever ``args['model']`` names, call by call.

    ``models`` is what each call was asked to fit, which is the assertion most of these
    tests turn on: a resumed pass must ask for the missing models and no others.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, X_train, X_test, y_train, y_test, data_key, args, **kwargs):
        requested = list(args["model"])
        self.calls.append({"data_key": data_key, "models": requested, "kwargs": kwargs})
        tuned = bool(args.get("grid_search", False))
        out = {}
        for name in requested:
            label = f"{name}_opt" if tuned else name
            y_pred = (X_test[:, 0] > np.median(X_train[:, 0])).astype(int)
            out[f"results_{label}"] = {0: {
                "model": label,
                "accuracy": float(np.mean(y_pred == y_test)),
                # A value whose text must survive the carry-over unchanged.
                "BestParams_Tuned": "{'C': 1.0}",
            }}
            out[f"y_test_{label}"] = {0: np.asarray(y_test)}
            out[f"y_predicted_{label}"] = {0: y_pred}
            out[f"y_score_{label}"] = {0: X_test[:, 0].astype(float)}
            vs = kwargs.get("validation")
            if vs is not None:
                records = [protocol.TrialRecord(
                    t, {"C": float(t + 1)}, 0.5 + 0.1 * t, is_default=(t == 0),
                    y_pred=np.zeros(vs.n_val, dtype=int),
                    y_score=(np.asarray(vs.y_val, float) if t == 1 else None))
                    for t in range(2)]
                out[f"{protocol.TRIALS_PREFIX}{label}"] = {0: protocol.trial_log(
                    records, metric="balanced_accuracy", best=1,
                    val_idx=vs.val_idx, y_val=vs.y_val)}
        return out

    @property
    def fitted(self):
        """Every (data_key, model) this stub was asked to fit."""
        return {(call["data_key"], m) for call in self.calls for m in call["models"]}


def _stub_evaluate(df, y, file, *a, **k):
    return pd.DataFrame([{"Dataset": file, "n_rows_evaluated": len(df)}])


@pytest.fixture
def stubbed(monkeypatch):
    stub = StubModelRun()
    monkeypatch.setattr(qp, "model_run", stub)
    monkeypatch.setattr(qp, "evaluate", _stub_evaluate)
    return stub


def _run(args, work_dir, monkeypatch):
    """One run of qprofiler in its own directory, as hydra would place it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)
    qp.main(OmegaConf.create(OmegaConf.to_container(args, resolve=False)))
    return work_dir


def _rows(run_dir):
    with open(run_dir / "ModelResults.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def _cells(run_dir):
    return {(r["embeddings"], r["iteration"], r["model"]) for r in _rows(run_dir)}


# ---------------------------------------------------------------- 1. the config value
class TestTheConfigValue:
    def test_off_forms(self):
        for value in (False, None, "", "false", "no", "off", "0"):
            assert resume.resolve_root(value) is None

    def test_true_means_the_sibling_run_directories(self, tmp_path):
        run = tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00"
        for value in (True, "true", "yes", "on"):
            assert resume.resolve_root(value, run_dir=str(run)) == str(run.parent)

    def test_a_path_is_taken_as_given(self, tmp_path):
        assert resume.resolve_root(str(tmp_path)) == str(tmp_path)

    @pytest.mark.parametrize("bad", ["earlier/runs", "./runs", 3.5])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(resume.ResumeError):
            resume.resolve_root(bad)

    def test_a_relative_path_fails_validation_not_the_first_pass(self, tmp_path):
        """Checked with embedding_cache, before a dataset is read."""
        csv_path, _ = _write_dataset(tmp_path / "data")
        args = _args(tmp_path / "data", skip_existing="earlier/runs")
        qp._resolve_model_lists(args, LOG)
        with pytest.raises(ValueError, match="ABSOLUTE directory"):
            qp._validate_config(args, LOG)


# ---------------------------------------------------------------- 2. the label
class TestTheLabelTheIndexIsKeyedBy:
    """``_model_label`` has to agree with model_run's dispatch or nothing is ever adopted."""

    def test_tuning_appends_opt_to_a_classical_model(self, tmp_path):
        args = _args(tmp_path, grid_search=True)
        assert qp._model_label("lr", args) == "lr_opt"

    def test_untuned_keeps_the_bare_name(self, tmp_path):
        args = _args(tmp_path, grid_search=False, tune_quantum=False)
        assert qp._model_label("lr", args) == "lr"

    def test_a_quantum_model_needs_tune_quantum_as_well(self, tmp_path):
        tuned = _args(tmp_path, grid_search=True, tune_quantum=True)
        half = _args(tmp_path, grid_search=True, tune_quantum=False)
        assert qp._model_label("qsvc", tuned) == "qsvc_opt"
        assert qp._model_label("qsvc", half) == "qsvc"

    def test_it_matches_what_a_run_actually_writes(self, tmp_path, stubbed, monkeypatch):
        _write_dataset(tmp_path / "data")
        args = _args(tmp_path / "data")
        run = _run(args, tmp_path / "r" / "one", monkeypatch)
        written = {r["model"] for r in _rows(run)}
        assert written == {qp._model_label(m, args) for m in ("lr", "nb")} == {"lr_opt", "nb_opt"}


# ---------------------------------------------------------------- 3. a complete resume
class TestResumingACompleteRun:
    @pytest.fixture
    def done(self, tmp_path, stubbed, monkeypatch):
        """One finished run under ``results/cfg/``, and the config that produced it."""
        _write_dataset(tmp_path / "data")
        args = _args(tmp_path / "data")
        first = _run(args, tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        return args, first, stubbed

    def test_the_first_run_fitted_every_cell(self, done):
        _, first, stub = done
        assert _cells(first) == {("none", str(i), m) for i in (1, 2)
                                 for m in ("lr_opt", "nb_opt")}
        assert len(stub.calls) == 2       # one per split

    def test_a_second_run_fits_nothing_and_is_still_complete(self, done, tmp_path,
                                                             monkeypatch):
        args, first, stub = done
        stub.calls.clear()
        second = _run(_args(tmp_path / "data", skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        assert stub.calls == [], "a complete config must fit nothing on resume"
        assert _cells(second) == _cells(first)

    def test_the_adopted_rows_are_copied_verbatim(self, done, tmp_path, monkeypatch):
        args, first, stub = done
        before = {(r["embeddings"], r["iteration"], r["model"]): r for r in _rows(first)}
        second = _run(_args(tmp_path / "data", skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        after = {(r["embeddings"], r["iteration"], r["model"]): r for r in _rows(second)}
        assert after.keys() == before.keys()
        for key, row in before.items():
            # Every cell, text for text: an adopted row is not re-serialised, so a score
            # cannot change in its last digits by passing through a resume.
            assert after[key]["accuracy"] == row["accuracy"]
            assert after[key]["BestParams_Tuned"] == row["BestParams_Tuned"]

    def test_every_adopted_row_names_where_it_came_from(self, done, tmp_path, monkeypatch):
        args, first, _ = done
        second = _run(_args(tmp_path / "data", skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        with open(second / resume.ADOPTED_CSV, newline="") as handle:
            adopted = list(csv.DictReader(handle))
        assert len(adopted) == 4
        assert {r["source_run_dir"] for r in adopted} == {str(first)}
        assert {r["model"] for r in adopted} == {"lr_opt", "nb_opt"}

    def test_the_pass_summaries_come_across_too(self, done, tmp_path, monkeypatch):
        args, first, _ = done
        second = _run(_args(tmp_path / "data", skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        with open(second / "results.pkl", "rb") as handle:
            carried = pickle.load(handle)
        assert [(s["embeddings"], s["iteration"]) for s in carried] == [("none", 1), ("none", 2)]

    def test_a_directory_given_explicitly_works_the_same(self, done, tmp_path, monkeypatch):
        args, first, stub = done
        stub.calls.clear()
        elsewhere = tmp_path / "elsewhere" / "run"
        second = _run(_args(tmp_path / "data", skip_existing=str(first)), elsewhere, monkeypatch)
        assert stub.calls == []
        assert _cells(second) == _cells(first)

    def test_the_current_run_directory_is_not_read_as_an_earlier_one(self, done, tmp_path,
                                                                    monkeypatch):
        """Resuming into the SAME directory must not adopt the rows it is writing."""
        args, first, stub = done
        stub.calls.clear()
        again = _run(_args(tmp_path / "data", skip_existing=True), first, monkeypatch)
        # Nothing else is under results/cfg/, so there is nothing to adopt and the run
        # recomputes -- rather than reading its own table and appending it to itself.
        assert len(stub.calls) == 2
        assert _cells(again) == {("none", str(i), m) for i in (1, 2)
                                 for m in ("lr_opt", "nb_opt")}


# ---------------------------------------------------------------- 4. a partial resume
class TestResumingAHalfDoneRun:
    @pytest.fixture
    def half(self, tmp_path, stubbed, monkeypatch):
        """An earlier run that fitted 'lr' only -- what a job killed mid-pass leaves."""
        _write_dataset(tmp_path / "data")
        first = _run(_args(tmp_path / "data", model=["lr"]),
                     tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        return first, stubbed

    def test_only_the_missing_model_is_fitted(self, half, tmp_path, monkeypatch):
        first, stub = half
        stub.calls.clear()
        second = _run(_args(tmp_path / "data", model=["lr", "nb"], skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        assert [call["models"] for call in stub.calls] == [["nb"], ["nb"]]
        assert _cells(second) == {("none", str(i), m) for i in (1, 2)
                                 for m in ("lr_opt", "nb_opt")}

    def test_the_model_list_is_put_back_after_the_pass(self, half, tmp_path, monkeypatch):
        """_only_models narrows args['model'] for one call, not for the run."""
        first, stub = half
        args = _args(tmp_path / "data", model=["lr", "nb"], skip_existing=True)
        _run(args, tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        assert list(args["model"]) == ["lr", "nb"]

    def test_a_split_the_earlier_run_never_reached_is_run_whole(self, tmp_path, stubbed,
                                                               monkeypatch):
        """The earlier run did one split of two; the resume fits both models of the other."""
        _write_dataset(tmp_path / "data")
        _run(_args(tmp_path / "data", iter=1),
             tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        stubbed.calls.clear()
        second = _run(_args(tmp_path / "data", iter=2, skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        assert [call["models"] for call in stubbed.calls] == [["lr", "nb"]]
        assert stubbed.calls[0]["data_key"].endswith("_2")
        assert _cells(second) == {("none", str(i), m) for i in (1, 2)
                                 for m in ("lr_opt", "nb_opt")}


# ---------------------------------------------------------------- 5. the sidecars
class TestTheSidecarsComeAcross:
    @pytest.fixture
    def half_manifest(self, tmp_path, stubbed, monkeypatch):
        csv_path, y = _write_dataset(tmp_path / "data")
        _write_manifest(tmp_path / "splits", csv_path, y)
        first = _run(_manifest_args(tmp_path / "data", tmp_path / "splits", model=["lr"]),
                     tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        return first, stubbed

    def test_an_adopted_models_oof_and_trials_land_in_the_new_run(self, half_manifest,
                                                                 tmp_path, monkeypatch):
        first, stub = half_manifest
        second = _run(_manifest_args(tmp_path / "data", tmp_path / "splits",
                                     model=["lr", "nb"], skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        for directory in (protocol.OOF_DIR, protocol.TRIALS_DIR,
                          protocol.VAL_PREDICTIONS_DIR):
            files = sorted((second / directory).glob("*.csv"))
            assert files, f"{directory} is empty after a resume"
            for path in files:
                models = {row["model"] for row in csv.DictReader(open(path, newline=""))}
                assert models == {"lr_opt", "nb_opt"}, (directory, path.name, models)

    def test_a_fully_adopted_pass_writes_its_sidecars_from_the_earlier_run(
            self, tmp_path, stubbed, monkeypatch):
        csv_path, y = _write_dataset(tmp_path / "data")
        _write_manifest(tmp_path / "splits", csv_path, y)
        args = _manifest_args(tmp_path / "data", tmp_path / "splits")
        first = _run(args, tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00",
                     monkeypatch)
        stubbed.calls.clear()
        second = _run(_manifest_args(tmp_path / "data", tmp_path / "splits",
                                     skip_existing=True),
                      tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00", monkeypatch)
        assert stubbed.calls == []
        for directory in (protocol.OOF_DIR, protocol.TRIALS_DIR,
                          protocol.VAL_PREDICTIONS_DIR):
            mine = sorted(p.name for p in (second / directory).glob("*.csv"))
            theirs = sorted(p.name for p in (first / directory).glob("*.csv"))
            assert mine == theirs and mine


# ---------------------------------------------------------------- 6. the provenance guard
class TestARowFromOtherBytesIsRefused:
    def test_a_changed_dataset_is_not_adopted(self, tmp_path, stubbed, monkeypatch, caplog):
        """The rows name the sha256 they were computed from; a new CSV must not reuse them."""
        csv_path, y = _write_dataset(tmp_path / "data")
        _write_manifest(tmp_path / "splits", csv_path, y)
        first = _run(_manifest_args(tmp_path / "data", tmp_path / "splits"),
                     tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        # Rewrite the rows' dataset_sha256 as if they came from another file, which is what
        # a resume across a regenerated dataset looks like from this run's side.
        rows = _rows(first)
        assert rows[0]["dataset_sha256"], "manifest mode writes this column"
        for row in rows:
            row["dataset_sha256"] = "0" * 64
        with open(first / "ModelResults.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), restval="")
            writer.writeheader()
            writer.writerows(rows)

        stubbed.calls.clear()
        with caplog.at_level(logging.WARNING):
            second = _run(_manifest_args(tmp_path / "data", tmp_path / "splits",
                                         skip_existing=True),
                          tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00",
                          monkeypatch)
        assert stubbed.calls, "every cell must be recomputed, not adopted"
        assert any("will NOT adopt" in r.message for r in caplog.records)
        assert not (second / resume.ADOPTED_CSV).exists()

    def test_the_index_itself_records_the_refusal(self, tmp_path):
        root = tmp_path / "results" / "cfg"
        run = root / "sv_2026-10-01_10-00-00"
        run.mkdir(parents=True)
        with open(run / "ModelResults.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "Dataset", "embeddings", "iteration", "model", "dataset_sha256"])
            writer.writeheader()
            writer.writerow({"Dataset": "d.csv", "embeddings": "pca", "iteration": "1",
                             "model": "lr_opt", "dataset_sha256": "AA"})
        found = resume.scan(str(root), require={"dataset_sha256": "BB"})
        assert len(found) == 0
        assert list(found.rejected) == [("d.csv", "pca", 1, "lr_opt")]
        assert "AA" in next(iter(found.rejected.values()))


# ---------------------------------------------------------------- 7. off by default
class TestItIsOffUnlessAskedFor:
    def test_the_shipped_config_ships_it_off(self):
        assert OmegaConf.load(SHIPPED_CONFIG)["skip_existing"] is False

    def test_a_config_that_never_names_it_recomputes_everything(self, tmp_path, stubbed,
                                                                monkeypatch):
        _write_dataset(tmp_path / "data")
        args = _args(tmp_path / "data")
        OmegaConf.set_struct(args, False)
        del args["skip_existing"]
        _run(args, tmp_path / "results" / "cfg" / "sv_2026-10-01_10-00-00", monkeypatch)
        stubbed.calls.clear()
        second = _run(args, tmp_path / "results" / "cfg" / "sv_2026-10-08_10-00-00",
                      monkeypatch)
        assert len(stubbed.calls) == 2
        assert not (second / resume.ADOPTED_CSV).exists()
