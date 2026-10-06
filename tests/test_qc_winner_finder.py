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
    delta_metric_table,
    fair_winner,
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
