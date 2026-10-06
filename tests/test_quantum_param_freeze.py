"""Tune a quantum model once, reuse the result on later resamples.

Why the feature exists
----------------------
With ``grid_search: True`` and ``tune_quantum: False`` the benchmark searches every
classical learner's hyperparameters and no quantum learner's, then reports the comparison
as "classical vs quantum". Every quantum loss is then confounded with the fact that
nobody searched its space.

Turning ``tune_quantum: True`` on removes the confound and multiplies the quantum side's
cost by ``n_trials_quantum`` *per resample*. Freezing after the first resample keeps the
search and drops the repetition: ``n_trials + (iter - 1)`` fits per arm instead of
``iter * n_trials``.

What these tests pin
--------------------
The load-bearing ones are :meth:`TestTheSearchIsSkipped.test_the_second_resample_does_not_search_again`
(the saving is real, not just cached-looking) and
:class:`TestTheKeyIsCorrectlyScoped` (reuse must cross *only* the iteration axis -- a key
that also collapsed embeddings or datasets would hand one dataset's tuned configuration
to another, which is a worse bug than the unfairness being fixed).

A real quantum fit is far too slow for a test, so the compute function is a stub that
records its calls. That is the right level anyway: the behaviour under test is the
caching and dispatch, and substituting a real circuit would test Qiskit instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qbiocode.learning._param_cache import (
    CACHE_DIR_KEY,
    FREEZE_KEY,
    freeze_key,
    freezing_enabled,
    load_frozen_params,
    save_frozen_params,
)
from qbiocode.learning._tuning import build_search_space, run_function_study


def make_stub(calls, accuracy_of=None):
    """A ``compute_*``-shaped function that records its kwargs instead of fitting.

    Returns a frame with the one column ``_metric_of`` needs: a ``results_`` column
    holding a metrics dict. ``accuracy_of`` maps the trial's parameters to a score, so a
    test can make a particular configuration the winner deterministically. The score is
    written under ``balanced_accuracy`` too, the default ``tuning_metric``.
    """

    def stub(X_train, X_test, y_train, y_test, args, **kwargs):
        calls.append(dict(kwargs))
        score = accuracy_of(kwargs) if accuracy_of else 0.5
        return pd.DataFrame({"results_stub": [
            {"accuracy": score, "balanced_accuracy": score, "f1_score": score}
        ]})

    return stub


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)))
    y = pd.Series([0, 1] * 20)
    return X, y


@pytest.fixture
def args(tmp_path):
    """Freezing on, simulator backend, cache under tmp_path."""
    return {
        "backend": "simulator",
        "seed": 42,
        FREEZE_KEY: True,
        CACHE_DIR_KEY: str(tmp_path / "frozen"),
    }


SPACE = {"reps": [1, 2, 3], "entanglement": ["linear", "full"]}
#: Wide enough that a trial budget of 8 is not silently lowered to the size of the space
#: by ``_finite_size``, which would make a cost assertion measure the cap and not the
#: freeze.
WIDE_SPACE = {"reps": [1, 2, 3, 4], "entanglement": ["linear", "full", "circular"]}


class TestFreezeKey:
    """``data_key`` carries the iteration; the freeze key must not."""

    def test_the_trailing_iteration_token_is_dropped(self):
        """``qprofiler`` builds ``dataset_embed_ncomponents_iter``."""
        assert freeze_key("mydata_pca_3_0") == "mydata_pca_3"
        assert freeze_key("mydata_pca_3_4") == "mydata_pca_3"

    def test_every_iteration_of_one_arm_maps_to_one_key(self):
        keys = {freeze_key(f"mydata_umap_8_{i}") for i in range(5)}
        assert len(keys) == 1, f"iterations did not collapse to one key: {keys}"

    def test_a_key_with_no_separator_survives(self):
        assert freeze_key("solo") == "solo"

    def test_a_dataset_name_containing_underscores_is_not_truncated(self):
        """Only the last token goes. Real corpus names are full of underscores --
        ``GAMETES_Epistasis_2_Way_20atts_0.1H`` -- and losing more than the iteration
        would collapse distinct datasets onto one key."""
        assert freeze_key("GAMETES_Epistasis_2_Way_20atts_0.1H_pca_3_2") == \
            "GAMETES_Epistasis_2_Way_20atts_0.1H_pca_3"


class TestTheKeyIsCorrectlyScoped:
    """Reuse must cross the iteration axis and nothing else."""

    @pytest.mark.parametrize(
        "other_key,why",
        [
            ("mydata_umap_3_0", "a different embedding"),
            ("mydata_pca_8_0", "a different component count"),
            ("otherdata_pca_3_0", "a different dataset"),
        ],
    )
    def test_params_do_not_leak_across_anything_but_iterations(self, args, other_key, why):
        """Anything that changes what a good configuration *is* must miss the cache.

        Handing one dataset's tuned configuration to another would be a worse defect than
        the unfairness this feature exists to fix: it would silently make the reported
        configuration not the one that was searched.
        """
        save_frozen_params(args, "mydata_pca_3_0", "qsvc", {"reps": 2})
        assert load_frozen_params(args, other_key, "qsvc") is None, (
            f"frozen parameters leaked across {why}"
        )

    def test_params_do_not_leak_across_models(self, args):
        save_frozen_params(args, "mydata_pca_3_0", "qsvc", {"reps": 2})
        assert load_frozen_params(args, "mydata_pca_3_0", "qnn") is None

    def test_a_later_iteration_of_the_same_arm_hits(self, args):
        save_frozen_params(args, "mydata_pca_3_0", "qsvc", {"reps": 2})
        for i in range(1, 5):
            assert load_frozen_params(args, f"mydata_pca_3_{i}", "qsvc") == {"reps": 2}


class TestFreezingIsOptIn:
    def test_off_by_default(self):
        assert freezing_enabled({}) is False

    def test_nothing_is_written_when_it_is_off(self, tmp_path):
        off = {CACHE_DIR_KEY: str(tmp_path / "frozen")}
        assert save_frozen_params(off, "d_pca_3_0", "qsvc", {"reps": 2}) is None
        assert not (tmp_path / "frozen").exists()

    def test_nothing_is_read_when_it_is_off(self, args, tmp_path):
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2})
        off = dict(args)
        off[FREEZE_KEY] = False
        assert load_frozen_params(off, "d_pca_3_1", "qsvc") is None

    def test_a_caller_that_passes_no_data_key_never_freezes(self, args, data, tmp_path):
        """``data_key=None`` is the pre-feature behaviour, so an existing caller that
        does not forward it keeps searching every time even with the flag on."""
        X, y = data
        calls = []
        space = build_search_space("qsvc", SPACE)
        for _ in range(2):
            run_function_study(
                make_stub(calls), space, X, y, args,
                model="qsvc", n_trials=3, seed=1, data_key=None,
            )
        assert len(calls) == 6, "a data_key-less caller should search both times"
        assert not list(Path(args[CACHE_DIR_KEY]).glob("*.json"))


class TestTheSearchIsSkipped:
    """The cost saving must be real, not merely a cache that gets written."""

    def test_the_second_resample_does_not_search_again(self, args, data):
        """The whole point. Trial count must drop to zero on a reused resample.

        Asserted on the stub's call count rather than on wall clock, because the saving
        being claimed is "N fewer quantum fits" and the call count *is* the fit count.
        """
        X, y = data
        space = build_search_space("qsvc", SPACE)

        first = []
        best_first = run_function_study(
            make_stub(first), space, X, y, args,
            model="qsvc", n_trials=4, seed=1, data_key="mydata_pca_3_0",
        )
        assert len(first) == 4, "the first resample should run the full trial budget"

        second = []
        best_second = run_function_study(
            make_stub(second), space, X, y, args,
            model="qsvc", n_trials=4, seed=1, data_key="mydata_pca_3_1",
        )
        assert second == [], (
            f"the second resample ran {len(second)} trials; it should have reused the "
            f"frozen configuration and run none"
        )
        assert best_second == best_first, "reuse returned a different configuration"

    def test_the_frozen_configuration_is_one_the_search_actually_chose(self, args, data):
        """Reuse must return a searched configuration, not a default or a blank.

        The stub scores ``reps=3`` highest, so that is the configuration the search must
        settle on and the one later resamples must receive.
        """
        X, y = data
        space = build_search_space("qsvc", SPACE)
        best = run_function_study(
            make_stub([], accuracy_of=lambda kw: kw["reps"] / 10.0),
            space, X, y, args, model="qsvc", n_trials=6, seed=1,
            data_key="mydata_pca_3_0",
        )
        assert best["reps"] == 3
        assert load_frozen_params(args, "mydata_pca_3_1", "qsvc")["reps"] == 3

    def test_all_five_iterations_cost_one_search_plus_four_refits(self, args, data):
        """The cost claim, end to end: ``n_trials + (iter - 1)`` rather than
        ``iter * n_trials``."""
        X, y = data
        space = build_search_space("qsvc", WIDE_SPACE)
        n_trials, iters = 8, 5
        total = 0
        for i in range(iters):
            calls = []
            run_function_study(
                make_stub(calls), space, X, y, args,
                model="qsvc", n_trials=n_trials, seed=1, data_key=f"mydata_pca_3_{i}",
            )
            total += len(calls)
        assert total == n_trials, (
            f"{total} trial fits across {iters} resamples; freezing should cost "
            f"{n_trials} (the unfrozen cost would be {n_trials * iters})"
        )


class TestStalenessAndCorruption:
    """Every failure path must degrade to "search again", never to a wrong reuse."""

    def test_a_changed_search_space_discards_the_frozen_set(self, args):
        """Editing ``gridsearch_<model>_args`` between runs must invalidate the cache.

        Otherwise the run reports itself as tuned while feeding the old configuration
        into a space the user has since changed -- tuned to something that no longer
        exists.
        """
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2, "entanglement": "linear"})
        widened = {"reps": None, "entanglement": None, "C": None}
        assert load_frozen_params(args, "d_pca_3_1", "qsvc", space=widened) is None

    def test_an_unchanged_search_space_still_hits(self, args):
        params = {"reps": 2, "entanglement": "linear"}
        save_frozen_params(args, "d_pca_3_0", "qsvc", params)
        same = {"reps": None, "entanglement": None}
        assert load_frozen_params(args, "d_pca_3_1", "qsvc", space=same) == params

    def test_a_truncated_file_is_a_miss_not_a_crash(self, args, tmp_path):
        """The expected shape of a crash mid-write. A sweep must not die over it."""
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2})
        path = next(Path(args[CACHE_DIR_KEY]).glob("*.json"))
        path.write_text('{"params": {"reps"', encoding="utf-8")
        assert load_frozen_params(args, "d_pca_3_1", "qsvc") is None

    def test_a_file_without_a_params_mapping_is_a_miss(self, args, tmp_path):
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2})
        path = next(Path(args[CACHE_DIR_KEY]).glob("*.json"))
        path.write_text('{"model": "qsvc"}', encoding="utf-8")
        assert load_frozen_params(args, "d_pca_3_1", "qsvc") is None

    def test_an_empty_params_mapping_is_a_miss(self, args, tmp_path):
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2})
        path = next(Path(args[CACHE_DIR_KEY]).glob("*.json"))
        path.write_text('{"params": {}}', encoding="utf-8")
        assert load_frozen_params(args, "d_pca_3_1", "qsvc") is None

    def test_a_missing_directory_is_a_miss_not_an_error(self, tmp_path):
        never = {FREEZE_KEY: True, CACHE_DIR_KEY: str(tmp_path / "absent")}
        assert load_frozen_params(never, "d_pca_3_0", "qsvc") is None

    def test_an_empty_best_params_is_not_persisted(self, args):
        """A search that found nothing must not write a file that later reads as a hit."""
        assert save_frozen_params(args, "d_pca_3_0", "qsvc", {}) is None
        assert load_frozen_params(args, "d_pca_3_1", "qsvc") is None

    def test_no_scratch_files_are_left_behind(self, args):
        save_frozen_params(args, "d_pca_3_0", "qsvc", {"reps": 2})
        leftovers = list(Path(args[CACHE_DIR_KEY]).glob(".frozen_*"))
        assert not leftovers, f"atomic write left scratch files: {leftovers}"


class TestTheFileIsAuditable:
    """A frozen configuration is part of the experimental record."""

    def test_the_file_records_which_model_and_key_it_belongs_to(self, args):
        path = save_frozen_params(args, "mydata_pca_3_0", "qsvc", {"reps": 2})
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["model"] == "qsvc"
        assert payload["freeze_key"] == "mydata_pca_3"
        assert payload["data_key_written_from"] == "mydata_pca_3_0"
        assert payload["params"] == {"reps": 2}

    def test_the_filename_names_the_arm(self, args):
        path = Path(save_frozen_params(args, "mydata_pca_3_0", "qsvc", {"reps": 2}))
        assert "mydata_pca_3" in path.name and "qsvc" in path.name

    def test_non_json_parameter_values_do_not_break_the_write(self, args):
        """Search spaces can yield numpy scalars, which json refuses by default."""
        path = save_frozen_params(
            args, "d_pca_3_0", "qsvc", {"reps": np.int64(2), "C": np.float64(1.5)}
        )
        assert path is not None
        assert load_frozen_params(args, "d_pca_3_1", "qsvc") is not None
