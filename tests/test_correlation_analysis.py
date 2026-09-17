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

"""The correlation analysis must not turn "unmeasurable" into "measured zero".

This is the last stage of the pipeline and the one whose output gets published, so a
value invented here reaches a figure with nothing upstream to contradict it. Two sources
of genuinely-missing values arrive at this module on every real run:

* ``auc`` is NaN wherever a model exposes no ranking. ``modeleval`` writes that NaN on
  purpose -- its docstring says a missing value is visible where a mislabelled one is not
  -- so anything here that fills it defeats the one guarantee it makes.
* A feature that is constant within a (model, embedding, dataset) group has no defined
  correlation. This is routine, not exotic: the embedding width is fixed inside a group,
  so ``mfe.nr_attr`` and its neighbours are constant by construction.

Both used to be destroyed, in different ways: ``np.median``/``spearmanr`` propagated the
first across a whole group, and ``fillna(0)`` rendered the second as the exact centre of
a diverging colormap.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from qbiocode.evaluation.model_run import QUANTUM_MODELS
from qbiocode.visualization.visualize_correlation import (
    _MISSING_COLOR,
    _QML_MODELS,
    compute_results_correlation,
    plot_results_correlation,
)

N_SPLITS = 15


def _results(models=("dt",), nan_auc_at=(), constant_feature=True):
    """A ModelResults-shaped frame: one row per (model, split)."""
    rng = np.random.default_rng(0)
    rows = []
    for model in models:
        for split in range(N_SPLITS):
            rows.append(
                {
                    "Dataset": "class_data-1.csv",
                    "model": model,
                    "embeddings": "pca",
                    "iteration": split + 1,
                    "accuracy": rng.uniform(0.5, 1.0),
                    "f1_score": rng.uniform(0.5, 1.0),
                    "time": rng.uniform(0.1, 2.0),
                    "auc": (
                        np.nan if (model, split) in nan_auc_at else rng.uniform(0.5, 1.0)
                    ),
                    # Constant inside the group, exactly as a fixed embedding width is.
                    "mfe.nr_attr": 3.0 if constant_feature else rng.uniform(1, 9),
                    "Intrinsic_Dimension": rng.uniform(1.0, 3.0),
                    "task.graph_hf_mass": rng.uniform(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def _correlate(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_results_correlation(df, **kwargs)[1]


class TestOneMissingValueDoesNotEraseAGroup:
    """``np.median`` and ``spearmanr`` both propagate NaN; neither may be used raw."""

    def test_a_single_nan_auc_leaves_the_other_fourteen_splits_usable(self):
        """The bug in its exact shape: 1 missing AUC in 15 blanked all 15.

        A decision tree of pure leaves has no ranking, so ``extract_binary_scores``
        returns None and ``modeleval`` records NaN. One such row among fifteen splits used
        to make ``median_metric`` NaN and EVERY feature correlation NaN for that group --
        the whole AUC analysis for that model, gone, with no warning and no empty cell to
        notice, because the figure then drew the NaNs as zeros.
        """
        corr = _correlate(_results(nan_auc_at={("dt", 3)}))
        auc = corr[corr["metric"] == "auc"]
        assert not auc.empty

        medians = auc["median_metric"].dropna()
        assert len(medians) == len(auc), (
            "median_metric must be computed from the observed values (nanmedian); a "
            "single NaN AUC left it undefined for the entire group"
        )
        assert (medians > 0).all()

        # The non-constant features must still produce a coefficient.
        measurable = auc[auc["feature"] != "mfe.nr_attr"]
        assert measurable["correlation"].notna().all(), (
            "a correlation must be computed from the complete pairs; propagating one NaN "
            "erased every feature's coefficient for this metric"
        )

    def test_the_fraction_above_threshold_excludes_missing_from_the_denominator(self):
        """A NaN is not a value below the threshold.

        With 14 observed AUCs the denominator is 14, not 15. Counting the missing row as
        a failure understates the fraction by exactly one row's worth, silently.
        """
        df = _results(nan_auc_at={("dt", 3)})
        corr = _correlate(df)
        observed = df["auc"].dropna()
        expected = float((observed > 0.7).sum() / len(observed))
        got = corr[corr["metric"] == "auc"]["frac_gt_thresh"].unique()
        assert len(got) == 1
        assert got[0] == pytest.approx(expected), (
            f"frac_gt_thresh must divide by the {len(observed)} observed AUCs, not by all "
            f"{len(df)} rows"
        )

    def test_a_metric_never_observed_reports_nan_rather_than_zero(self):
        """"Nothing cleared the bar" and "nothing was measured" are different facts."""
        df = _results(nan_auc_at={("dt", i) for i in range(N_SPLITS)})
        corr = _correlate(df)
        auc = corr[corr["metric"] == "auc"]
        assert auc["median_metric"].isna().all()
        assert auc["frac_gt_thresh"].isna().all(), (
            "a fraction of 0.0 would claim every model was scored and none passed"
        )

    def test_a_constant_feature_yields_nan_not_a_coefficient(self):
        """No variance means no correlation -- and it must say so, not say zero."""
        corr = _correlate(_results())
        constant = corr[corr["feature"] == "mfe.nr_attr"]
        assert not constant.empty
        assert constant["correlation"].isna().all()

    def test_a_varying_feature_is_measured_where_a_constant_one_is_not(self):
        """The control: NaN has to come from the data, not from the code path."""
        corr = _correlate(_results(constant_feature=False))
        assert corr[corr["feature"] == "mfe.nr_attr"]["correlation"].notna().all()


class TestTheCorrelationEngineIsHonouredOrRefused:
    """A name this function cannot compute must be an error, never an empty frame."""

    def test_pearson_is_computed_rather_than_silently_dropped(self):
        corr = _correlate(_results(), correlation="pearson")
        assert len(corr) > 0, (
            "correlation='pearson' used to fall through the spearman-only branch and "
            "append nothing, returning an empty frame and three blank figures"
        )
        assert corr[corr["feature"] == "Intrinsic_Dimension"]["correlation"].notna().any()

    def test_spearman_and_pearson_disagree_so_the_choice_is_real(self):
        """Guards against both names resolving to the same engine."""
        rho = _correlate(_results())["correlation"]
        r = _correlate(_results(), correlation="pearson")["correlation"]
        assert not np.allclose(rho.dropna(), r.dropna())

    def test_an_unknown_engine_is_refused(self):
        with pytest.raises(ValueError, match="Unknown correlation"):
            _correlate(_results(), correlation="kendal")


class TestMissingIsDrawnAsMissing:
    """The figure stage: NaN must survive into the render, distinctly coloured."""

    def test_the_missing_colour_is_outside_the_diverging_scale(self):
        """A grey that also appeared in the blue-white-red ramp would defeat the point."""
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(_MISSING_COLOR)
        assert r == g == b, "the missing colour must be neutral grey, not a scale colour"
        assert 0.5 < r < 0.9, "and mid-tone, so it reads as absent on a white ground"

    def test_all_three_figures_render_with_nan_present(self, tmp_path):
        """The end-to-end path, since `set_bad` and `mask` are easy to get wrong."""
        corr = _correlate(_results(models=("dt", "qsvc"), nan_auc_at={("dt", 3)}))
        out = tmp_path / "corr.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = plot_results_correlation(
                corr, metric="auc", save_file_path=str(out), show_plots=False
            )
        assert len(figures) == 4
        written = sorted(p.name for p in tmp_path.glob("*.png"))
        assert written == [
            "corr.png",
            "corr_heatmap.png",
            "corr_noncluster_heatmap.png",
        ], written

    def test_the_scatter_colormap_maps_nan_to_the_missing_colour(self):
        """Not an approximation of the fix -- the actual mapping the figure uses."""
        from matplotlib.colors import to_rgba

        import qbiocode.visualization.visualize_correlation as module

        corr = _correlate(_results())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scatter = module.plot_results_correlation(
                corr, metric="auc", show_plots=False
            ).scatter_ax.collections[0]
        cmap = scatter.get_cmap()
        assert cmap(np.nan) == to_rgba(_MISSING_COLOR), (
            "a NaN correlation must render in the missing colour; mapping it to the "
            "colormap's centre makes it indistinguishable from rho=0"
        )

    def test_a_zero_correlation_is_not_the_missing_colour(self):
        """The other half: a genuine zero must still read as a measured zero."""
        from matplotlib.colors import to_rgba

        import qbiocode.visualization.visualize_correlation as module

        corr = _correlate(_results())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scatter = module.plot_results_correlation(
                corr, metric="auc", show_plots=False
            ).scatter_ax.collections[0]
        cmap, norm = scatter.get_cmap(), scatter.norm
        assert cmap(norm(0.0)) != to_rgba(_MISSING_COLOR)


class TestAnUnmeasurableFeatureKeepsItsRow:
    """Grey means "could not be measured"; absent means "was never considered".

    Removing the ``fillna(0)`` had a second-order effect worth its own pin: a feature
    whose correlation is undefined in *every* group becomes an all-NaN row, and
    ``pivot_table`` drops those by default. On the real pyMFE block that silently removed
    28 of 141 rows from the heatmap -- every constant-within-group column -- which is a
    worse misreading than the zero it replaced, because a reader cannot tell a dropped
    feature from one that was never in the analysis.
    """

    def test_a_feature_that_is_never_measurable_still_appears_in_the_heatmap(self):
        import qbiocode.visualization.visualize_correlation as module

        corr = _correlate(_results(models=("dt", "qsvc")))
        # mfe.nr_attr is constant in both groups, so its correlation is NaN throughout.
        assert corr[corr["feature"] == "mfe.nr_attr"]["correlation"].isna().all()

        n_features = corr[corr["metric"] == "auc"]["feature"].nunique()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = module.plot_results_correlation(
                corr, metric="auc", show_plots=False
            )
        rows = len(figures.clustered_heatmap.data2d.index)
        assert rows == n_features, (
            f"the heatmap has {rows} rows for {n_features} features; a feature with no "
            f"measurable correlation anywhere must keep a fully-grey row rather than be "
            f"dropped from the figure"
        )


class TestTheSmallestUsefulRunPlots:
    """One model, one embedding, one dataset -- and therefore one column to cluster.

    That is the smallest run QProfiler can do and roughly what a first-time user starts
    from, and it used to crash the plotting stage: ``sns.clustermap`` asks scipy for a
    column dendrogram, a single column gives an empty distance matrix, and the error --
    "The number of observations cannot be determined on an empty distance matrix" --
    surfaced from four frames below seaborn naming nothing the user had configured.
    """

    def test_a_single_group_still_produces_all_three_figures(self, tmp_path):
        corr = _correlate(_results(models=("dt",)))
        assert corr["model_embed_datatype"].nunique() == 1
        out = tmp_path / "one.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = plot_results_correlation(
                corr, metric="f1_score", save_file_path=str(out), show_plots=False
            )
        assert len(figures) == 4
        assert len(sorted(tmp_path.glob("*.png"))) == 3

    def test_a_single_feature_row_also_plots(self, tmp_path):
        """The row-axis half of the same degeneracy."""
        corr = _correlate(_results(models=("dt", "qsvc")))
        one_feature = corr[corr["feature"] == "Intrinsic_Dimension"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figures = plot_results_correlation(
                one_feature, metric="f1_score", show_plots=False
            )
        assert len(figures) == 4


class TestQplIsTreatedAsQuantum:
    """One list of quantum models, and QPL is on it."""

    def test_the_figure_list_is_derived_from_the_dispatcher(self):
        assert set(_QML_MODELS) == {name.upper() for name in QUANTUM_MODELS}

    def test_qpl_is_included(self):
        """It was not. Three hardcoded copies all named four models and omitted QPL, so
        every QPL row was drawn in the classical colour and sorted with the classical
        models -- in the dot plot, in both heatmaps, and in the column colour bar."""
        assert "QPL" in _QML_MODELS
        assert "qpl" in QUANTUM_MODELS
