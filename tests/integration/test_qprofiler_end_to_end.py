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

"""QProfiler, end to end, from the shipped config.

Three real runs back the whole module (session-scoped, so each happens once):
two at seed 7 and one at seed 99. Between them they pin the result schema, the
embedding widths, and both halves of the reproducibility contract -- the same
seed reproduces exactly, a different seed does not.

Classical models only (``lr``, ``dt``) and ``backend: simulator``, so nothing
here reaches for a quantum backend or the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from .conftest import (
    KEY_COLUMNS,
    METRIC_COLUMNS,
    N_FEATURES,
    metric_signature,
    run_qprofiler,
)

EMBEDDINGS = ["pca", "none"]
#: The feature count is reported by pyMFE now that QProfiler's complexity block is
#: pyMFE-backed; it was "# Features" before. See qbiocode/evaluation/mfe_features.py.
N_FEATURES_COLUMN = "mfe.nr_attr"
MODELS = ["lr", "dt"]
N_ITER = 2
N_COMPONENTS = 2

OVERRIDES = [
    f"embeddings=[{','.join(EMBEDDINGS)}]",
    # The synthetic dataset is 5 features wide, below the embedding_min_features
    # default of 18, so the embedding requested above would otherwise be suppressed
    # and collapsed to a single "none" pass. These tests exist to exercise the
    # embedding machinery, so they opt out of the threshold the way a user would.
    "embedding_min_features=0",
    f"model=[{','.join(MODELS)}]",
    f"iter={N_ITER}",
    f"n_components={N_COMPONENTS}",
    "n_jobs=2",
    # The shipped config tunes (grid_search and tune_quantum both True), which names the
    # rows lr_opt/dt_opt and writes BestParams_Tuned. This module pins the untuned
    # schema, so it asks for it explicitly, as the config's own comment says to; both
    # keys, because grid_search off with tune_quantum on is rejected.
    "grid_search=False",
    "tune_quantum=False",
]


@pytest.fixture(scope="session")
def run_seed7(tmp_path_factory):
    return run_qprofiler(tmp_path_factory.mktemp("seed7"), OVERRIDES + ["seed=7"])


@pytest.fixture(scope="session")
def rerun_seed7(tmp_path_factory):
    """A second, independent run at the same seed, in its own working directory."""
    return run_qprofiler(tmp_path_factory.mktemp("seed7-again"), OVERRIDES + ["seed=7"])


@pytest.fixture(scope="session")
def run_seed99(tmp_path_factory):
    return run_qprofiler(tmp_path_factory.mktemp("seed99"), OVERRIDES + ["seed=99"])


class TestTheResultsFile:
    def test_it_has_one_row_per_embedding_model_iteration(self, run_seed7):
        assert len(run_seed7) == N_ITER * len(EMBEDDINGS) * len(MODELS)

    def test_it_carries_the_documented_columns(self, run_seed7):
        expected = set(KEY_COLUMNS) | set(METRIC_COLUMNS) | {"time", "Model_Parameters"}
        missing = expected - set(run_seed7.columns)
        assert not missing, f"ModelResults.csv is missing {sorted(missing)}"

    def test_every_requested_embedding_and_model_ran(self, run_seed7):
        assert set(run_seed7["embeddings"]) == set(EMBEDDINGS)
        assert set(run_seed7["model"]) == set(MODELS)
        assert set(run_seed7["iteration"]) == set(range(1, N_ITER + 1))

    def test_the_embedding_actually_reduced_the_features(self, run_seed7):
        """``n_components`` must reach the embedding, not just the config file."""
        reduced = run_seed7[run_seed7["embeddings"] == "pca"][N_FEATURES_COLUMN].unique()
        unreduced = run_seed7[run_seed7["embeddings"] == "none"][N_FEATURES_COLUMN].unique()
        assert list(reduced) == [N_COMPONENTS]
        assert list(unreduced) == [N_FEATURES]

    @pytest.mark.parametrize("column", METRIC_COLUMNS)
    def test_metrics_are_finite_and_in_range(self, run_seed7, column):
        values = pd.to_numeric(run_seed7[column])
        assert values.notna().all(), f"{column} contains nan"
        assert ((values >= 0.0) & (values <= 1.0)).all(), f"{column} outside [0, 1]"

    def test_the_models_learned_something(self, run_seed7):
        """The dataset is separable; if the best model is at chance, the pipeline is broken."""
        assert pd.to_numeric(run_seed7["accuracy"]).max() > 0.6


@pytest.fixture(scope="session")
def defaulted_run(tmp_path_factory):
    """``run_seed7``'s configuration minus the ``embedding_min_features=0`` opt-out."""
    return run_qprofiler(
        tmp_path_factory.mktemp("threshold-default"),
        [
            f"embeddings=[{','.join(EMBEDDINGS)}]",
            f"model=[{','.join(MODELS)}]",
            f"iter={N_ITER}",
            f"n_components={N_COMPONENTS}",
            "n_jobs=2",
            "seed=7",
        ],
    )


class TestTheEmbeddingThresholdEndToEnd:
    """The ``embedding_min_features`` default, through the real CLI and config.

    Every other fixture in this module passes ``embedding_min_features=0`` so it can
    exercise the embedding machinery on 5-feature synthetic data. This class is the
    counterpart: it leaves the key at its shipped default and checks that the policy
    actually reaches a run -- otherwise the override in ``OVERRIDES`` would be the only
    behaviour under test, and the default could break without anything noticing.
    """

    def test_the_requested_pca_is_suppressed_on_narrow_data(self, defaulted_run):
        assert N_FEATURES < 18, "this test's premise is that the data is narrow"
        assert set(defaulted_run["embeddings"]) == {"none"}, (
            f"{N_FEATURES}-feature data must collapse ['pca', 'none'] to one unreduced "
            f"pass; got {sorted(set(defaulted_run['embeddings']))}"
        )

    def test_it_collapses_rather_than_running_none_twice(self, defaulted_run):
        """The row count is the whole point: one pass, not one no-op pass per name."""
        assert len(defaulted_run) == N_ITER * len(MODELS), (
            f"expected {N_ITER * len(MODELS)} rows (one 'none' pass per model per split), "
            f"got {len(defaulted_run)} -- a duplicated pass means every model was fitted "
            f"twice on identical data"
        )

    def test_the_features_are_the_unreduced_ones(self, defaulted_run):
        assert list(defaulted_run[N_FEATURES_COLUMN].unique()) == [N_FEATURES]

    def test_the_run_still_produces_usable_metrics(self, defaulted_run):
        """Suppressing the embedding must not degrade the run to nothing."""
        assert pd.to_numeric(defaulted_run["accuracy"]).max() > 0.6


class TestReproducibility:
    """Both halves matter: identical at one seed, and *not* identical across seeds.

    Only the first half was ever checked before, and it passes trivially if the
    seed is ignored entirely -- which is exactly what QProfiler used to do, since
    ``train_test_split`` was called with no ``random_state`` at all.
    """

    def test_the_same_seed_reproduces_every_metric(self, run_seed7, rerun_seed7):
        first, second = metric_signature(run_seed7), metric_signature(rerun_seed7)
        pd.testing.assert_frame_equal(first, second)

    def test_a_different_seed_produces_different_metrics(self, run_seed7, run_seed99):
        first, other = metric_signature(run_seed7), metric_signature(run_seed99)
        assert not first[METRIC_COLUMNS].equals(other[METRIC_COLUMNS]), (
            "seed 7 and seed 99 produced byte-identical metrics; the seed is not "
            "reaching the train/test split"
        )

    def test_the_two_iterations_are_different_splits(self, run_seed7):
        """``split_seed = seed + iter`` must vary the split, not just the seed."""
        per_iteration = {
            iteration: tuple(
                frame.sort_values(["embeddings", "model"])[METRIC_COLUMNS]
                .to_numpy()
                .ravel()
            )
            for iteration, frame in run_seed7.groupby("iteration")
        }
        assert len(set(per_iteration.values())) == N_ITER, (
            "every iteration produced identical metrics, so all iterations used "
            "the same split"
        )


class TestTheRawDataEvaluation:
    def test_it_summarizes_the_unembedded_dataset(self, run_seed7, tmp_path_factory):
        """Written alongside ModelResults, one row per input dataset."""
        # The frame under test is the sibling file of the results the fixture read.
        raw_files = sorted(
            path
            for base in tmp_path_factory.getbasetemp().glob("seed7*")
            for path in base.glob("results/**/RawDataEvaluation.csv")
        )
        assert raw_files, "no RawDataEvaluation.csv was written"
        raw = pd.read_csv(raw_files[0])
        assert len(raw) == 1
        assert raw[N_FEATURES_COLUMN].iloc[0] == N_FEATURES
