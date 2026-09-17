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

"""An embedding is applied only to a dataset wide enough to justify one.

``embedding_min_features`` (default :data:`DEFAULT_EMBEDDING_MIN_FEATURES`, 18) is the
feature count a dataset must EXCEED before the ``embeddings`` list is applied. Below it
the list collapses to a single ``'none'`` pass.

Two properties are what make the policy safe to have on by default, and each is pinned
below:

* **It collapses rather than repeats.** Running each requested name as a no-op would fit
  every model once per name on identical data and write those rows under an
  ``embeddings`` column naming a reduction that never happened -- several times the
  runtime for one result, recorded misleadingly.
* **It is a default, not a rule.** ``embedding_min_features: 0`` restores the old
  behaviour exactly, which is what the two QProfiler tutorial configs set: their data is
  6 and 10 features wide and the comparison they draw is 'none' against 'pca'.
"""

import numpy as np
import pytest
import yaml

from qbiocode.embeddings import DEFAULT_EMBEDDING_MIN_FEATURES, resolve_embeddings

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


class TestTheThresholdItself:
    """``resolve_embeddings`` -- the whole policy, in one pure function."""

    def test_the_default_is_eighteen(self):
        """Pinned because it is a documented number, quoted in three config files."""
        assert DEFAULT_EMBEDDING_MIN_FEATURES == 18

    @pytest.mark.parametrize("n_features", [19, 20, 50, 20000])
    def test_a_wide_dataset_keeps_every_requested_embedding(self, n_features):
        requested = ["pca", "nmf", "none"]
        effective, skipped = resolve_embeddings(requested, n_features)
        assert effective == requested, "a wide dataset must be embedded as configured"
        assert skipped == []

    @pytest.mark.parametrize("n_features", [1, 2, 6, 10, 17, 18])
    def test_a_narrow_dataset_collapses_to_one_none_pass(self, n_features):
        """18 is included on purpose: the rule is *more than*, not *at least*."""
        effective, skipped = resolve_embeddings(["pca", "nmf", "none"], n_features)
        assert effective == ["none"], (
            f"{n_features} features is not more than {DEFAULT_EMBEDDING_MIN_FEATURES}, so "
            f"the list must collapse to exactly one unreduced pass, got {effective}"
        )
        assert skipped == ["pca", "nmf"], "the skipped names are what the caller reports"

    def test_the_collapse_does_not_duplicate_an_already_requested_none(self):
        """The failure this guards is quiet: a duplicate 'none' is a doubled run.

        ``['none', 'pca']`` -- what both tutorial configs write -- must not become
        ``['none', 'none']``. Every model would be fitted twice on identical data, and
        the two row groups would be indistinguishable in ``ModelResults.csv``.
        """
        effective, skipped = resolve_embeddings(["none", "pca"], 6)
        assert effective == ["none"]
        assert skipped == ["pca"]

    def test_a_list_that_was_only_none_is_untouched_and_reports_nothing_skipped(self):
        """No embedding was requested, so the threshold has no opinion and says so."""
        effective, skipped = resolve_embeddings(["none"], 6)
        assert effective == ["none"]
        assert skipped == [], (
            "reporting 'none' as skipped would make QProfiler warn about suppressing an "
            "embedding on every run of a config that asked for no embedding"
        )

    @pytest.mark.parametrize("disable", [0, None, False])
    def test_the_threshold_can_be_turned_off_entirely(self, disable):
        """The documented escape hatch, in each spelling YAML can produce."""
        requested = ["pca", "nmf"]
        effective, skipped = resolve_embeddings(requested, 6, min_features=disable)
        assert effective == requested
        assert skipped == []

    def test_case_and_space_do_not_defeat_the_none_check(self):
        """``check_embedding_name`` normalises later, so this has to normalise here."""
        effective, skipped = resolve_embeddings([" NONE ", "pca"], 6)
        assert effective == ["none"]
        assert skipped == ["pca"]

    @pytest.mark.parametrize("bad", [-1, 2.5, "18", True])
    def test_a_malformed_threshold_is_refused(self, bad):
        with pytest.raises(ValueError, match="embedding_min_features"):
            resolve_embeddings(["pca"], 6, min_features=bad)

    def test_the_input_list_is_not_mutated(self):
        """QProfiler passes ``args['embeddings']`` straight in, once per split."""
        requested = ["pca", "none"]
        resolve_embeddings(requested, 6)
        assert requested == ["pca", "none"]

    def test_a_numpy_width_is_accepted(self):
        """``X_train.shape[1]`` is a plain int, but an int64 must not be refused."""
        effective, _ = resolve_embeddings(["pca"], np.int64(6))
        assert effective == ["none"]


class TestQProfilerHonoursIt:
    """The wiring: the app must read the key, and validate it before loading data."""

    def _config(self, **overrides):
        args = {
            "folder_path": "data", "file_dataset": "ALL", "embeddings": ["pca"],
            "n_components": 3, "model": ["dt"], "seed": 42, "q_seed": 42,
            "test_size": 0.3, "iter": 1, "scaling": False, "backend": "simulator",
            "n_jobs": 1,
        }
        args.update(overrides)
        return args

    def test_a_malformed_threshold_is_refused_before_any_dataset_is_read(self):
        """The point of validating in ``_validate_config``: a typo costs a second.

        ``resolve_embeddings`` would also refuse it, but only from inside the split loop
        -- after the first dataset had been loaded, its complexity block computed and the
        split scaled, which on a real dataset is minutes.
        """
        import logging

        from qbiocode.apps.qprofiler.qprofiler import _validate_config

        with pytest.raises(ValueError, match="embedding_min_features"):
            _validate_config(
                self._config(embedding_min_features=-5), logging.getLogger("t")
            )

    def test_the_default_applies_when_the_key_is_absent(self):
        """An existing config that never heard of this key still gets the policy."""
        import logging

        from qbiocode.apps.qprofiler.qprofiler import _validate_config

        # No embedding_min_features at all -- must validate cleanly, not KeyError.
        _validate_config(self._config(), logging.getLogger("t"))

    def test_qprofiler_resolves_the_list_rather_than_iterating_args_directly(self):
        """A structural pin: the loop must consume the resolved list.

        Iterating ``args['embeddings']`` again anywhere below the resolve call would
        reinstate the old behaviour silently -- the threshold would be computed, logged,
        and then ignored.
        """
        import inspect

        from qbiocode.apps.qprofiler import qprofiler

        source = inspect.getsource(qprofiler.main)
        assert "for embed in effective_embeddings:" in source, (
            "the embedding loop must iterate the resolved list"
        )
        assert "for embed in args['embeddings']" not in source, (
            "args['embeddings'] is the REQUEST; iterating it bypasses the threshold"
        )


class TestTheShippedConfigsAgreeWithTheirData:
    """A config whose data is narrower than its threshold silently stops embedding.

    Both QProfiler tutorials ship data below the default and draw a 'none' vs 'pca'
    comparison, so both must opt out explicitly. Without this test the tutorials would
    keep running and quietly produce one embedding group where their notebooks -- and the
    committed correlation figures -- describe two.
    """

    @pytest.mark.parametrize(
        "config,data_dir",
        [
            ("tutorial/QProfiler/configs/config.yaml", "tutorial/QProfiler/data/ld_data"),
            (
                "tutorial/QProfiler_v2/configs/config.yaml",
                "tutorial/QProfiler_v2/data/ld_data_v2",
            ),
        ],
    )
    def test_a_tutorial_that_embeds_narrow_data_opts_out_explicitly(self, config, data_dir):
        import pandas as pd

        text = (REPO_ROOT / config).read_text()
        loaded = yaml.safe_load(text)
        requested = [str(e).lower().strip() for e in loaded["embeddings"]]
        if requested == ["none"]:
            pytest.skip("config asks for no embedding, so the threshold cannot bite")

        csvs = sorted((REPO_ROOT / data_dir).glob("*.csv"))
        assert csvs, f"no data found under {data_dir}"
        n_features = pd.read_csv(csvs[0], sep=r"\t|,", engine="python").shape[1] - 1

        effective, skipped = resolve_embeddings(
            requested,
            n_features,
            min_features=loaded.get(
                "embedding_min_features", DEFAULT_EMBEDDING_MIN_FEATURES
            ),
        )
        assert not skipped, (
            f"{config} asks for {requested} on {n_features}-feature data, but its "
            f"embedding_min_features leaves {skipped} suppressed. The notebook and its "
            f"committed figures describe {len(requested)} embedding groups; this run "
            f"would produce {len(effective)}. Set embedding_min_features: 0 there."
        )
