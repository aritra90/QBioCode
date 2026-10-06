"""Tests for the fair-selection wrappers in :mod:`qbiocode.utils.qc_winner_finder`.

The wrappers only forward to :func:`qbiocode.utils.fair_selection.select_winners` and
persist its frames, so the tests check the forwarding: ``test_size`` is required, the
``margin`` / ``fdr`` / ``controls`` keywords reach the selector, and the renamed columns
are the ones kept. :func:`resolution_floor_epsilon` is tested on its own, because its
pooling key decides the default equivalence bound.

Every fixture is synthetic and seeded; nothing reads the corpus.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from qbiocode.utils.fair_selection import iteration_floor_half_width
from qbiocode.utils.qc_winner_finder import (
    aggregate_benchmark,
    collect_model_results,
    delta_metric_table,
    fair_winner,
    missing_folds,
    resolution_floor_epsilon,
)


def results_frame(datasets=("a", "b", "c"), embeddings=("pca", "umap"), iters=6,
                  quantum_shift=0.0, seed=0):
    """A ModelResults-shaped frame: two classical and two quantum arms per pass."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in datasets:
        for emb in embeddings:
            for it in range(1, iters + 1):
                for m, shift in (("svc", 0.0), ("lr", 0.0), ("qsvc", quantum_shift),
                                 ("vqc", quantum_shift)):
                    score = rng.normal(0.75 + shift, 0.03)
                    rows.append(dict(Dataset=d, embeddings=emb, iteration=it, model=m,
                                     balanced_accuracy=score, mcc=2 * score - 1))
    return pd.DataFrame(rows)


class TestFairWinner:
    def test_test_size_is_required(self, tmp_path):
        with pytest.raises(ValueError, match="test_size"):
            fair_winner(results_frame(), str(tmp_path), "t", metric="balanced_accuracy")

    def test_forwards_margin_fdr_and_controls(self, tmp_path):
        rep = fair_winner(results_frame(quantum_shift=0.1), str(tmp_path), "t",
                          metric="balanced_accuracy", test_size=0.2, margin=0.01,
                          fdr=0.2, controls="c")
        assert (rep.margin, rep.fdr, tuple(rep.controls)) == (0.01, 0.2, ("c",))
        fam = rep.per_dataset.set_index("Dataset")["family"]
        assert fam["c"] == "control" and fam["a"] == "discovery"
        assert "c" not in rep.quantum_datasets
        saved = pd.read_csv(os.path.join(tmp_path, "t_fair_verdicts.csv"))
        assert {"p_value", "within_equivalence", "family"} <= set(saved.columns)
        assert "p_margin" not in saved.columns


class TestDeltaMetricTable:
    def test_test_size_is_required(self):
        with pytest.raises(ValueError, match="test_size"):
            delta_metric_table(results_frame(), metrics=("balanced_accuracy",))

    def test_keeps_the_new_columns_and_forwards_the_keywords(self):
        wide, reports = delta_metric_table(
            results_frame(quantum_shift=0.1), metrics=("balanced_accuracy", "mcc"),
            epsilon=0.05, test_size=0.2, margin=0.02, fdr=0.2, controls=["c"])
        for m in ("balanced_accuracy", "mcc"):
            for col in ("p_value", "within_equivalence", "family", "p_adjusted",
                        "verdict_raw", "verdict_adjusted", "epsilon", "margin"):
                assert f"{col}__{m}" in wide.columns
            assert f"p_margin__{m}" not in wide.columns
            assert reports[m].margin == 0.02 and reports[m].fdr == 0.2
        assert wide.set_index("Dataset").loc["c", "family__mcc"] == "control"


class TestAggregateBenchmark:
    def test_summary_counts_discovery_rows_and_records_the_rule(self, tmp_path):
        frame = results_frame(quantum_shift=0.1)
        for d, part in frame.groupby("Dataset"):
            os.makedirs(tmp_path / "runs" / d)
            part.to_csv(tmp_path / "runs" / d / "ModelResults.csv", index=False)
        out = aggregate_benchmark(str(tmp_path / "runs"), str(tmp_path / "out"),
                                  metrics=("balanced_accuracy",), epsilon=0.05,
                                  test_size=0.2, margin=0.01, fdr=0.2, controls=["c"])
        row = out["summary"].set_index("metric").loc["balanced_accuracy"]
        assert (row["margin"], row["fdr"]) == (0.01, 0.2)
        assert (row["n_datasets"], row["n_controls"]) == (2, 1)
        verdicts = ("quantum_wins", "classical_wins", "equivalent", "inconclusive")
        assert sum(row[v] for v in verdicts) <= 2
        assert "c" not in row["quantum_datasets"].split(";")


class TestResolutionFloorEpsilon:
    @staticmethod
    def two_pass_frame(offset):
        """One dataset, two passes with means ``offset`` apart and identical spread."""
        noise = np.array([-0.02, 0.0, 0.02, 0.01, -0.01])
        rows = []
        for emb, mean in (("pca", 0.7), ("umap", 0.7 + offset)):
            for m in ("svc", "qsvc"):
                for it, e in enumerate(noise, start=1):
                    rows.append(dict(Dataset="a.csv", embeddings=emb, iteration=it,
                                     model=m, balanced_accuracy=mean + e))
        return pd.DataFrame(rows)

    def test_embedding_passes_are_not_pooled(self):
        per_pass = float(np.std([-0.02, 0.0, 0.02, 0.01, -0.01], ddof=1))
        eps, sigma = resolution_floor_epsilon(self.two_pass_frame(0.1),
                                              "balanced_accuracy", 0.2)
        assert sigma == pytest.approx(per_pass)
        assert eps == pytest.approx(iteration_floor_half_width(per_pass, 0.2))
        # The mean gap between passes does not leak into sigma.
        _, sigma0 = resolution_floor_epsilon(self.two_pass_frame(0.0),
                                             "balanced_accuracy", 0.2)
        assert sigma == pytest.approx(sigma0)

    def test_without_an_embeddings_column_the_dataset_is_one_group(self):
        frame = self.two_pass_frame(0.1).drop(columns="embeddings")
        _, sigma = resolution_floor_epsilon(frame, "balanced_accuracy", 0.2)
        per_pass = float(np.std([-0.02, 0.0, 0.02, 0.01, -0.01], ddof=1))
        assert sigma > per_pass


def fold_results(datasets=("a", "b"), k=3, repeats=2, seed=0):
    """A split_mode=manifest frame: validation scores favour 'lr' whatever the test says."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in datasets:
        for g in range(k * repeats):
            for m, val in (("svc", 0.6), ("lr", 0.9), ("qsvc", 0.7), ("vqc", 0.5)):
                score = rng.normal(0.75, 0.03)
                rows.append(dict(Dataset=d, embeddings="pca", iteration=g + 1,
                                 repeat=g // k, fold=g % k, model=m,
                                 split_mode="manifest", split_k=k, split_repeats=repeats,
                                 balanced_accuracy=score, mcc=2 * score - 1,
                                 tuning_score=val, tuning_metric="balanced_accuracy"))
    return pd.DataFrame(rows)


class TestValidationSelectionForwarding:
    def test_fair_winner_forwards_selection_and_names_the_trace_file(self, tmp_path):
        rep = fair_winner(fold_results(), str(tmp_path), "t", metric="balanced_accuracy",
                          selection="validation")
        assert rep.selection_mode == "validation" and rep.k == 3
        assert (rep.selection["classical_arm"] == "pca|lr").all()
        saved = pd.read_csv(os.path.join(tmp_path, "t_fair_validation_selection.csv"))
        assert {"classical_val", "quantum_val", "repeat", "fold"} <= set(saved.columns)
        assert not os.path.exists(os.path.join(tmp_path, "t_fair_loio_selection.csv"))

    def test_fair_winner_forwards_validation_col_and_k(self, tmp_path):
        df = fold_results().rename(columns={"tuning_score": "val"}).drop(columns="split_k")
        rep = fair_winner(df, str(tmp_path), "t", metric="balanced_accuracy",
                          selection="validation", validation_col="val", k=3)
        assert rep.test_size == pytest.approx(1 / 3)

    def test_delta_metric_table_needs_no_test_size_under_validation(self):
        wide, reports = delta_metric_table(fold_results(), metrics=("balanced_accuracy",),
                                           selection="validation")
        rep = reports["balanced_accuracy"]
        assert rep.selection_mode == "validation"
        assert rep.test_size == pytest.approx(1 / 3)
        # The default epsilon is the resolution floor at r = 1/(k-1).
        _, sigma = resolution_floor_epsilon(fold_results(), "balanced_accuracy", 1 / 3)
        assert rep.epsilon == pytest.approx(iteration_floor_half_width(sigma, 1 / 3))
        assert set(wide["Dataset"]) == {"a", "b"}

    def test_aggregate_benchmark_forwards_selection_and_reports_missing_folds(self, tmp_path):
        frame = fold_results()
        # One fold of one quantum arm never landed.
        frame = frame[~((frame["Dataset"] == "a") & (frame["model"] == "vqc")
                        & (frame["iteration"] == 4))]
        for (d, g), part in frame.groupby(["Dataset", "iteration"]):
            os.makedirs(tmp_path / "runs" / d / str(g))
            part.to_csv(tmp_path / "runs" / d / str(g) / "ModelResults.csv", index=False)
        out = aggregate_benchmark(str(tmp_path / "runs"), str(tmp_path / "out"),
                                  metrics=("balanced_accuracy",), epsilon=0.05,
                                  selection="validation")
        assert out["reports"]["balanced_accuracy"].selection_mode == "validation"
        gaps = pd.read_csv(tmp_path / "out" / "benchmark_missing_folds.csv",
                           dtype={"missing": str})
        assert gaps[["Dataset", "model", "missing"]].values.tolist() == [["a", "vqc", "4"]]
        inv = out["inventory"]
        assert inv.loc[inv["Dataset"] == "a", "n_missing_folds"].eq(1).all()
        assert inv.loc[inv["Dataset"] == "b", "n_missing_folds"].eq(0).all()

    def test_internal_mode_inventory_is_unchanged(self, tmp_path):
        frame = results_frame()
        os.makedirs(tmp_path / "r")
        frame.to_csv(tmp_path / "r" / "ModelResults.csv", index=False)
        _, inv = collect_model_results(str(tmp_path))
        assert list(inv.columns) == ["path", "Dataset", "n_rows", "n_models", "n_iterations"]


class TestMissingFolds:
    def test_full_expects_split_k_times_repeats(self):
        df = fold_results(datasets=("a",))
        df = df[df["iteration"] != 6]           # no arm has fold 6: all four incomplete
        gaps = missing_folds(df, expected="full")
        assert len(gaps) == 4 and (gaps["missing"] == "6").all()
        assert (gaps["n_expected"] == 6).all() and (gaps["n_present"] == 5).all()

    def test_subset_run_is_not_flagged_by_default(self):
        # A run over splits 1-3 of kR = 6: every arm has the same three folds.
        df = fold_results(datasets=("a",))
        df = df[df["iteration"] <= 3]
        assert missing_folds(df).empty
        assert len(missing_folds(df, expected="full")) == 4
        gaps = missing_folds(df[~((df["model"] == "vqc") & (df["iteration"] == 2))])
        assert gaps[["model", "missing", "n_expected"]].values.tolist() == [["vqc", "2", 3]]

    def test_explicit_expected_iterations(self):
        df = fold_results(datasets=("a",))
        df = df[df["iteration"] <= 3]
        gaps = missing_folds(df, expected=[1, 2, 3, 4])
        assert len(gaps) == 4 and (gaps["missing"] == "4").all()
        with pytest.raises(ValueError, match="expected"):
            missing_folds(df, expected="bogus")

    def test_without_split_columns_the_union_is_expected(self):
        df = fold_results(datasets=("a",)).drop(columns=["split_k", "split_repeats"])
        assert missing_folds(df).empty
        df = df[~((df["model"] == "lr") & (df["iteration"].isin([2, 3])))]
        gaps = missing_folds(df)
        assert gaps[["model", "missing"]].values.tolist() == [["lr", "2;3"]]


def _analyze_pilot():
    """experiments/pilot10/analyze_pilot.py, loaded by path (experiments is not a package)."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), os.pardir, "experiments", "pilot10",
                        "analyze_pilot.py")
    spec = importlib.util.spec_from_file_location("analyze_pilot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(split_dir, csv, n=30, k=3, repeats=2):
    """A schema-2 manifest for ``csv`` in split_manifest's format."""
    import json

    from sklearn.model_selection import StratifiedKFold

    from qbiocode.apps.qprofiler import split_manifest as sm

    y = np.arange(n) % 2
    pd.DataFrame({"a": np.arange(n), "label": y}).to_csv(csv, index=False)
    folds = []
    for r in range(repeats):
        tests = [np.sort(te) for _, te in StratifiedKFold(k, shuffle=True, random_state=42 + r)
                 .split(np.zeros(n), y)]
        for f in range(k):
            folds.append({"repeat": r, "fold": f,
                          "train": np.setdiff1d(np.arange(n), tests[f]).tolist(),
                          "val": tests[(f + 1) % k].tolist(), "test": tests[f].tolist()})
    body = {"schema_version": 2, "dataset_id": "toy", "sha256": sm.file_sha256(csv), "n": n,
            "k": k, "n_repeats": repeats, "protocol": "StratifiedKFold",
            "validation": "next_fold", "seed": 42,
            "repeat_seeds": [42 + r for r in range(repeats)], "group_col": None,
            "generator_version": "make_splits/2.0", "folds": folds}
    sm.manifest_path(split_dir, csv).write_text(json.dumps(body))


class TestReadRunProtocol:
    def test_internal_runs_tree_merges_the_protocol_layer(self, tmp_path):
        ap = _analyze_pilot()
        ds = tmp_path / "runs" / "toy"
        ds.mkdir(parents=True)
        (ds / "_protocol.yaml").write_text("test_size: 0.2\niter: 5\n")
        for m in ("lr", "qsvc"):
            (ds / f"toy_none_{m}.yaml").write_text(f"defaults:\n  - _protocol\n  - _self_\n"
                                                   f"config_file_name: toy_none_{m}\n")
        test_size, n_iter, configs = ap.read_run_protocol(str(tmp_path / "runs"))
        assert (test_size, n_iter, len(configs)) == (0.2, 5, 2)
        proto = ap.read_run_protocol(str(tmp_path / "runs"), detail=True)
        assert proto.split_mode == "internal" and proto.selection == "loio"
        assert proto.k is None and proto.r == pytest.approx(0.25)

    def test_manifest_mode_reads_k_and_r_from_the_manifests(self, tmp_path):
        ap = _analyze_pilot()
        data, splits = tmp_path / "data", tmp_path / "splits"
        data.mkdir()
        splits.mkdir()
        _write_manifest(splits, str(data / "toy.csv"), k=3, repeats=2)
        ds = tmp_path / "runs" / "toy"
        ds.mkdir(parents=True)
        for it in (1, 2):
            (ds / f"toy_none_{it}_classical.yaml").write_text(
                f"split_mode: manifest\nsplit_dir: '{splits}'\nsplits: [{it}]\n"
                f"folder_path: '{data}'\nfile_dataset: ['toy.csv']\n")
        proto = ap.read_run_protocol(str(tmp_path / "runs"), detail=True)
        # n_iter counts the splits the jobs selected (1 and 2), not the manifest's kR.
        assert (proto.split_mode, proto.k, proto.n_repeats, proto.n_iter) == ("manifest", 3, 2, 2)
        assert proto.r == pytest.approx(0.5) and proto.test_size == pytest.approx(1 / 3)
        assert proto.selection == "validation"
        # Granularity reads each dataset once, however many jobs name it.
        gran = ap.granularity_report(proto.configs, proto.test_size, 0.05)
        assert len(gran) == 1 and gran["test_minority"].iloc[0] == 5

    def test_manifest_all_splits_counts_k_times_r(self, tmp_path):
        ap = _analyze_pilot()
        data, splits = tmp_path / "data", tmp_path / "splits"
        data.mkdir()
        splits.mkdir()
        _write_manifest(splits, str(data / "toy.csv"), k=3, repeats=2)
        ds = tmp_path / "runs" / "toy"
        ds.mkdir(parents=True)
        (ds / "toy_none_classical.yaml").write_text(
            f"split_mode: manifest\nsplit_dir: '{splits}'\nsplits: all\n"
            f"folder_path: '{data}'\nfile_dataset: ['toy.csv']\n")
        assert ap.read_run_protocol(str(tmp_path / "runs"), detail=True).n_iter == 6

    def test_mixed_manifest_protocols_do_not_advise_test_size(self, tmp_path):
        ap = _analyze_pilot()
        data, splits = tmp_path / "data", tmp_path / "splits"
        data.mkdir()
        splits.mkdir()
        _write_manifest(splits, str(data / "toy.csv"), k=3, repeats=2)
        (tmp_path / "m").mkdir()
        (tmp_path / "m" / "m.yaml").write_text(
            f"split_mode: manifest\nsplit_dir: '{splits}'\n"
            f"folder_path: '{data}'\nfile_dataset: ['toy.csv']\n")
        (tmp_path / "i").mkdir()
        (tmp_path / "i" / "i.yaml").write_text("test_size: 0.2\niter: 5\n")
        with pytest.raises(SystemExit, match="one runs tree per protocol"):
            ap.read_run_protocol(str(tmp_path))

    def test_mixed_protocols_are_refused(self, tmp_path):
        ap = _analyze_pilot()
        for name, body in (("a", "test_size: 0.2\niter: 5\n"), ("b", "test_size: 0.3\niter: 5\n")):
            (tmp_path / name).mkdir()
            (tmp_path / name / f"{name}.yaml").write_text(body)
        with pytest.raises(SystemExit, match="disagree"):
            ap.read_run_protocol(str(tmp_path))
