"""QSage holds out whole datasets, and never reads a missing metric as a score of zero.

Why these tests exist
---------------------
``QuantumSage.train_sub_sages`` fits one regressor per (model, metric) whose job is to
predict how well that model will do on a dataset it has never seen -- that is what
:meth:`~qbiocode.apps.sage.sage.QuantumSage.predict` is for. Two things in the original
split made the reported R-squared of that regressor an overstatement rather than a
measurement, and neither announced itself:

1. ``train_test_split`` shuffled ROWS. Complexity features are a property of the
   (embedded) dataset, so the ``iter`` rows of one (dataset, embedding) are byte-identical
   in every feature column. Shuffling rows therefore puts exact feature twins on both
   sides, and the score becomes recall of the training set.
2. The target was ``.fillna(0)``-ed. ``auc`` and ``pr_auc`` are NaN when a model exposed
   no usable ranking score; zero is not "unknown", it is "worse than random".

Both are silent -- they make the number look BETTER -- which is why they are pinned here
rather than left to a smoke test. The tests call the split helper directly: a real
``train_sub_sages`` run fits hundreds of regressors, and the behaviour under test is the
partitioning, not the regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbiocode.apps.sage.sage import QuantumSage

LEGACY_FEATURES = [
    "# Features", "# Samples", "Feature_Samples_ratio", "Intrinsic_Dimension",
    "Condition number", "Fisher Discriminant Ratio", "Total Correlations",
    "Mutual information", "# Non-zero entries", "# Low variance features",
    "Variation", "std_var", "Coefficient of Variation %", "std_co_of_v",
    "Skewness", "std_skew", "Kurtosis", "std_kurt", "Mean Log Kernel Density",
    "Isomap Reconstruction Error", "Fractal dimension", "Entropy", "std_entropy",
]
METRICS = ["accuracy", "f1_score", "auc"]
MODELS = ["rf", "svc"]


class Stub:
    """Enough of a QuantumSage for the split: the helper reads only ``_seed``."""

    _seed = 42


def arm(n_datasets=6, iterations=3, embeddings=("pca", "none")):
    """Rows for ONE model, shaped as a QProfiler results table.

    The feature vector repeats across every (embedding, iteration) row of a dataset,
    which is what makes a row-wise split leak: that is how a real table looks, because
    the complexity features describe the dataset and not the fit.
    """
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_datasets):
        features = {name: float(rng.uniform(1, 10)) for name in LEGACY_FEATURES}
        features["# Samples"] = 100.0
        for embedding in embeddings:
            for it in range(iterations):
                row = dict(features)
                row.update(Dataset=f"class_data-{d + 1}", embeddings=embedding,
                           iteration=it, f1_score=float(rng.uniform(0.5, 1.0)))
                rows.append(row)
    frame = pd.DataFrame(rows)
    X = frame[LEGACY_FEATURES]
    return X, frame["f1_score"], frame["Dataset"]


def split(X, y, groups, test_size=0.3, metric="f1_score", model="rf"):
    return QuantumSage._split_holding_out_datasets(
        Stub(), X, y, groups, test_size, metric, model
    )


class TestNoDatasetIsOnBothSides:
    """The property the whole fix exists for."""

    def test_train_and_test_datasets_are_disjoint(self):
        X, y, groups = arm()
        X_train, X_test, _, _ = split(X, y, groups)
        train_sets = set(groups.loc[X_train.index])
        test_sets = set(groups.loc[X_test.index])
        assert train_sets and test_sets
        assert not (train_sets & test_sets), (
            f"datasets on both sides of the split: {sorted(train_sets & test_sets)}"
        )

    def test_no_test_feature_row_has_a_twin_in_training(self):
        """The mechanism, stated directly: an identical feature vector on both sides.

        This is what the row-wise split was scoring. Asserted on the feature values
        rather than on the dataset label, because the label is not what the regressor
        sees -- a leak that preserved label disjointness but duplicated features would
        be just as wrong.
        """
        X, y, groups = arm()
        X_train, X_test, _, _ = split(X, y, groups)
        train_rows = {tuple(r) for r in X_train.to_numpy()}
        twins = [tuple(r) for r in X_test.to_numpy() if tuple(r) in train_rows]
        assert not twins, f"{len(twins)} test rows are exact feature twins of a training row"

    def test_the_row_wise_split_it_replaces_really_did_leak(self):
        """Guards the premise. If a plain shuffle happened not to leak on this fixture,
        the test above would pass for the wrong reason and prove nothing."""
        from sklearn.model_selection import train_test_split

        X, y, groups = arm()
        X_train, X_test, _, _ = train_test_split(
            X, y.to_numpy(), test_size=0.3, random_state=42
        )
        train_rows = {tuple(r) for r in X_train.to_numpy()}
        twins = [tuple(r) for r in X_test.to_numpy() if tuple(r) in train_rows]
        assert twins, (
            "the row-wise split did not leak on this fixture, so the grouped-split "
            "assertion above is not evidence of anything"
        )

    def test_every_row_lands_on_exactly_one_side(self):
        """No row silently dropped, none duplicated."""
        X, y, groups = arm()
        X_train, X_test, y_train, y_test = split(X, y, groups)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) == len(X_train) and len(y_test) == len(X_test)
        assert set(X_train.index).isdisjoint(X_test.index)

    def test_the_split_is_reproducible(self):
        X, y, groups = arm()
        first, second = split(X, y, groups), split(X, y, groups)
        assert list(first[1].index) == list(second[1].index)


class TestAnUndefinedMetricIsDroppedNotZeroed:
    """``auc`` NaN means "no usable ranking score", which is not a score of 0.0."""

    def test_nan_targets_are_absent_from_both_sides(self):
        X, y, groups = arm()
        y = y.copy()
        y.iloc[:8] = np.nan
        _, _, y_train, y_test = split(X, y, groups, metric="auc")
        assert not np.isnan(y_train).any() and not np.isnan(y_test).any()
        assert len(y_train) + len(y_test) == len(y) - 8

    def test_no_zero_is_invented_for_a_dropped_row(self):
        """The specific regression: ``.fillna(0)`` put 0.0 in the target vector."""
        X, y, groups = arm()
        y = y.copy()
        y.iloc[:8] = np.nan
        _, _, y_train, y_test = split(X, y, groups, metric="auc")
        assert not (y_train == 0.0).any() and not (y_test == 0.0).any(), (
            "a dropped row came back as auc = 0.0, which reads as worse than random"
        )

    def test_the_drop_is_logged(self, caplog):
        """A silently smaller training set is how the old defect stayed invisible."""
        X, y, groups = arm()
        y = y.copy()
        y.iloc[:8] = np.nan
        with caplog.at_level("WARNING"):
            split(X, y, groups, metric="auc")
        assert "undefined" in caplog.text and "auc" in caplog.text

    def test_a_fully_undefined_metric_raises_something_actionable(self):
        """Every row NaN is realistic -- one model, one metric it cannot produce.

        It must not fit a regressor on a column of zeros and report an R-squared for it,
        and the message has to name the two ways out, because neither is guessable.
        """
        X, y, groups = arm()
        y = pd.Series(np.nan, index=y.index, name="auc")
        with pytest.raises(ValueError) as excinfo:
            split(X, y, groups, metric="auc", model="svc")
        message = str(excinfo.value)
        assert "auc" in message and "svc" in message
        assert "drop" in message.lower()

    def test_a_target_that_is_genuinely_zero_is_kept(self):
        """0.0 is a legal F1. Only NaN is the sentinel."""
        X, y, groups = arm()
        y = y.copy()
        y.iloc[:4] = 0.0
        _, _, y_train, y_test = split(X, y, groups)
        assert len(y_train) + len(y_test) == len(y)
        assert (np.concatenate([y_train, y_test]) == 0.0).sum() == 4


class TestDegenerateInputs:
    def test_a_single_dataset_falls_back_and_says_so(self, caplog):
        """Nothing can be held out, so the split is row-wise -- but the R-squared it
        yields is not comparable with a multi-dataset run, and must not look like it is."""
        X, y, groups = arm(n_datasets=1)
        with caplog.at_level("WARNING"):
            X_train, X_test, _, _ = split(X, y, groups)
        assert len(X_train) and len(X_test)
        assert "1 dataset" in caplog.text
        assert "not comparable" in caplog.text

    def test_two_datasets_still_produce_a_non_empty_split(self):
        """GroupShuffleSplit allocates by group, so small group counts are the edge."""
        X, y, groups = arm(n_datasets=2)
        X_train, X_test, _, _ = split(X, y, groups)
        assert len(X_train) and len(X_test)
        assert set(groups.loc[X_train.index]).isdisjoint(groups.loc[X_test.index])

    def test_the_test_fraction_is_of_datasets_not_rows(self):
        """Documented behaviour, pinned so it is not read as a row fraction: with 10
        datasets and test_size 0.3, three datasets are held out."""
        X, y, groups = arm(n_datasets=10)
        X_train, X_test, _, _ = split(X, y, groups, test_size=0.3)
        assert groups.loc[X_test.index].nunique() == 3
        assert groups.loc[X_train.index].nunique() == 7
