"""Fold-internal feature selection: the properties, not the lines.

Three things have to hold, and each is constructed so that the obvious wrong
implementation fails it:

1. the selection reads the FIRST side only (no leakage from test/validation rows);
2. it runs on RAW values, before scaling -- MinMaxScaler destroys a variance ranking;
3. ``method: 'none'`` is byte-identical to no selection at all, so an existing embedding
   cache stays valid.
"""

import numpy as np
import pytest

from qbiocode.apps.qprofiler import feature_selection as fs


class TestNormaliseConfig:
    def test_none_block_is_the_no_selection_default(self):
        assert fs.normalise_config(None) == {
            "method": "none", "n_features": None, "fit_scope": "train_fold"}

    def test_method_none_drops_n_features_so_every_off_run_shares_one_spec(self):
        """Otherwise a stray n_features would fork the cache spec of runs that select
        nothing, and each variant would refuse the others' cache files."""
        a = fs.normalise_config({"method": "none", "n_features": 5000})
        b = fs.normalise_config({"method": "none"})
        assert a == b == {"method": "none", "n_features": None, "fit_scope": "train_fold"}

    def test_a_real_method_needs_n_features(self):
        with pytest.raises(ValueError, match="needs n_features"):
            fs.normalise_config({"method": "variance"})

    def test_unknown_method_and_scope_are_named(self):
        with pytest.raises(ValueError, match="not one of"):
            fs.normalise_config({"method": "mutual_info", "n_features": 10})
        with pytest.raises(ValueError, match="would leak"):
            fs.normalise_config({"method": "variance", "n_features": 10,
                                 "fit_scope": "cohort"})

    def test_non_positive_n_features_is_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            fs.normalise_config({"method": "variance", "n_features": 0})

    def test_values_are_cast_so_the_spec_is_json_safe(self):
        out = fs.normalise_config({"method": " Variance ", "n_features": "16"})
        assert out == {"method": "variance", "n_features": 16, "fit_scope": "train_fold"}
        assert isinstance(out["n_features"], int)


class TestNoLeakage:
    """The property ``fit_scope: train_fold`` names."""

    @staticmethod
    def _opposed(seed=0):
        """20 columns: 0-4 vary only in side A, 5-9 only in side B, 10-19 constant.

        An implementation that pools the two sides, or fits on the wrong one, picks
        different columns on this data -- so the assertion below is not satisfiable by
        accident.
        """
        rng = np.random.RandomState(seed)
        A = np.zeros((40, 20))
        B = np.zeros((40, 20))
        A[:, 0:5] = rng.normal(scale=10.0, size=(40, 5))
        B[:, 5:10] = rng.normal(scale=10.0, size=(40, 5))
        return A, B

    def test_selection_uses_the_first_side_only(self):
        A, B = self._opposed()
        cfg = fs.normalise_config({"method": "variance", "n_features": 5})
        _, _, idx = fs.select_features(A, B, cfg)
        assert sorted(idx) == [0, 1, 2, 3, 4]

    def test_swapping_the_sides_swaps_the_choice(self):
        A, B = self._opposed()
        cfg = fs.normalise_config({"method": "variance", "n_features": 5})
        _, _, idx = fs.select_features(B, A, cfg)
        assert sorted(idx) == [5, 6, 7, 8, 9]

    def test_the_second_side_cannot_change_the_choice(self):
        """Pure in the second argument: replace it entirely, same columns."""
        A, B = self._opposed()
        cfg = fs.normalise_config({"method": "variance", "n_features": 5})
        _, _, first = fs.select_features(A, B, cfg)
        _, _, second = fs.select_features(A, np.random.RandomState(1).normal(
            scale=100.0, size=B.shape), cfg)
        assert np.array_equal(first, second)

    def test_both_sides_are_restricted_to_the_same_columns(self):
        A, B = self._opposed()
        cfg = fs.normalise_config({"method": "variance", "n_features": 5})
        A_sel, B_sel, idx = fs.select_features(A, B, cfg)
        assert A_sel.shape == (40, 5) and B_sel.shape == (40, 5)
        assert np.array_equal(A_sel, A[:, idx])
        assert np.array_equal(B_sel, B[:, idx])


class TestOrderingAgainstScaling:
    """Selection must precede scaling, and this is why."""

    def test_minmax_scaling_destroys_the_variance_ranking(self):
        """The regression guard for the stage's position in the pipeline.

        Column 0 has a large spread; column 1 is mostly one value with a single far
        outlier, so its RANGE is larger while its variance is smaller. MinMaxScaler
        divides each column by its own range, which flattens exactly that distinction --
        so a ranking taken after scaling reports a different winner. The pipeline selects
        before scaling; if that order is ever reversed, this test says what breaks.
        """
        from qbiocode.utils.helper_fn import scale_train_test

        X = np.zeros((20, 2))
        X[:, 0] = np.linspace(-1.0, 1.0, 20)        # broad, high variance
        X[:, 1] = 0.0
        X[0, 1] = 50.0                              # one outlier: huge range, low variance

        raw = fs.variance_ranking(X)
        X_scaled, _ = scale_train_test(X, X.copy(), scaling="MinMaxScaler")
        scaled = fs.variance_ranking(X_scaled)

        assert raw[0] == 1, "column 1's single outlier dominates the raw variance"
        assert scaled[0] == 0, "after min-max, column 0's broad spread wins"
        assert raw[0] != scaled[0], (
            "the two orders must disagree, or this test cannot detect the mistake"
        )


class TestMethodNoneIsAPassthrough:
    """What keeps full1's embedding cache valid."""

    def test_arrays_are_returned_unchanged_and_indices_are_none(self):
        rng = np.random.RandomState(0)
        A, B = rng.normal(size=(10, 7)), rng.normal(size=(4, 7))
        A_out, B_out, idx = fs.select_features(A, B, fs.normalise_config(None))
        assert idx is None, "None distinguishes 'kept everything' from 'selected all'"
        assert A_out is A and B_out is B, "no copy: the arrays must pass straight through"


class TestVarianceRanking:
    def test_descending_by_variance(self):
        X = np.zeros((10, 3))
        X[:, 0] = np.linspace(0, 1, 10)      # smallest spread
        X[:, 1] = np.linspace(0, 5, 10)
        X[:, 2] = np.linspace(0, 10, 10)     # largest
        assert list(fs.variance_ranking(X)) == [2, 1, 0]

    def test_ties_break_by_column_order_and_are_stable(self):
        """Constant columns all have variance 0, and small folds produce many of them.
        An unstable sort would put a different one in the last kept slot per run."""
        X = np.zeros((8, 6))
        X[:, 3] = np.linspace(0, 1, 8)
        order = fs.variance_ranking(X)
        assert order[0] == 3
        assert list(order[1:]) == [0, 1, 2, 4, 5]
        for _ in range(5):
            assert np.array_equal(fs.variance_ranking(X), order)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            fs.variance_ranking(np.zeros(5))


class TestCapBehaviour:
    def test_n_features_above_the_width_keeps_everything(self):
        rng = np.random.RandomState(0)
        A, B = rng.normal(size=(10, 4)), rng.normal(size=(3, 4))
        cfg = fs.normalise_config({"method": "variance", "n_features": 99})
        A_sel, _, idx = fs.select_features(A, B, cfg)
        assert A_sel.shape[1] == 4
        assert list(idx) == [0, 1, 2, 3], "reported anyway, so the sidecar shape is fixed"

    def test_surviving_columns_keep_their_file_order(self):
        """Membership is decided by the ranking; layout stays the file's, so indices
        joined back to a gene list line up with the matrix a human would read."""
        X = np.zeros((10, 5))
        X[:, 4] = np.linspace(0, 9, 10)     # highest variance
        X[:, 1] = np.linspace(0, 3, 10)     # second
        cfg = fs.normalise_config({"method": "variance", "n_features": 2})
        _, _, idx = fs.select_features(X, X.copy(), cfg)
        assert list(idx) == [1, 4]

    def test_width_mismatch_is_rejected(self):
        cfg = fs.normalise_config({"method": "variance", "n_features": 2})
        with pytest.raises(ValueError, match="same width"):
            fs.select_features(np.zeros((5, 4)), np.zeros((5, 3)), cfg)


class TestSelectionDigest:
    def test_same_indices_same_digest_regardless_of_container(self):
        assert fs.selection_digest([3, 1, 4]) == fs.selection_digest(np.array([3, 1, 4]))

    def test_different_indices_differ(self):
        assert fs.selection_digest([1, 2, 3]) != fs.selection_digest([1, 2, 4])

    def test_order_is_part_of_the_digest(self):
        """It digests what was recorded, so a reordering shows up rather than being
        normalised away -- the indices are the record, the digest only checks them."""
        assert fs.selection_digest([1, 2]) != fs.selection_digest([2, 1])


class TestWiredIntoTheSplit:
    """The pipeline contract: order, scope, and what reaches the embedding."""

    @staticmethod
    def _data(seed=0):
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(60, 30))
        # Columns 0-3 carry a large spread everywhere, so they are selected whatever the
        # fold -- which makes the assertions below about WHICH columns stable.
        X[:, 0:4] *= 25.0
        y = np.array([0, 1] * 30)
        return X, y

    def test_off_by_default_returns_the_full_width(self):
        from qbiocode.apps.qprofiler import qprofiler as qp

        X, y = self._data()
        args = {"seed": 42, "test_size": 0.3, "stratify": ["y"]}
        X_tr, X_te, _, _, _, _, selected = qp._split_and_scale(X, y, args, 1, "None")
        assert X_tr.shape[1] == 30 and X_te.shape[1] == 30
        assert selected is None

    def test_selection_narrows_both_sides_and_reports_the_columns(self):
        from qbiocode.apps.qprofiler import qprofiler as qp

        X, y = self._data()
        args = {"seed": 42, "test_size": 0.3, "stratify": ["y"],
                "feature_selection": {"method": "variance", "n_features": 4}}
        X_tr, X_te, _, _, _, _, selected = qp._split_and_scale(X, y, args, 1, "None")
        assert X_tr.shape[1] == 4 and X_te.shape[1] == 4
        assert sorted(selected) == [0, 1, 2, 3]

    def test_selection_precedes_scaling_in_the_real_pipeline(self):
        """The columns chosen must be those of the RAW data, not the min-max scaled.

        Built so the two orders disagree: column 5 has one extreme outlier (large range,
        modest variance), columns 0-3 have a broad spread. Selecting first picks 0-3;
        scaling first would divide column 5 by its own huge range and change the
        ranking. Asserting the chosen set is the raw one pins the order.
        """
        from qbiocode.apps.qprofiler import qprofiler as qp

        X, y = self._data()
        X[:, 5] = 0.0
        X[0, 5] = 5000.0
        args = {"seed": 42, "test_size": 0.3, "stratify": ["y"],
                "feature_selection": {"method": "variance", "n_features": 4}}
        _, _, _, _, _, _, with_scaling = qp._split_and_scale(
            X, y, args, 1, "MinMaxScaler")
        _, _, _, _, _, _, without = qp._split_and_scale(X, y, args, 1, "None")
        assert np.array_equal(with_scaling, without), (
            "the selected columns must not depend on the scaler, which is only true if "
            "selection runs before it"
        )

    def test_the_two_stages_may_select_differently(self):
        """Expected, not a defect: 'tune' is fitted on the fit rows and 'final' on the
        whole training fold, so the variance ranking need not agree. It is also why the
        indices are recorded per (split, stage) rather than once per split."""
        from qbiocode.apps.qprofiler import qprofiler as qp
        rng = np.random.RandomState(3)
        X = rng.normal(size=(40, 12))
        y = np.array([0, 1] * 20)
        # A stub standing in for a manifest's OuterSplit: only the index arrays and
        # the repeat/fold labels are read by _split_and_scale.
        class _S:
            repeat, fold = 0, 0
            train_idx = np.arange(0, 32)
            val_idx = np.arange(0, 8)
            fit_idx = np.arange(8, 32)
            test_idx = np.arange(32, 40)
            n_fit, n_val, n_test = 24, 8, 8

        args = {"seed": 42, "feature_selection": {"method": "variance",
                                                  "n_features": 3}}
        _, _, _, _, _, _, final = qp._split_and_scale(
            X, y, args, 1, "None", split=_S(), stage="final")
        _, _, _, _, _, _, tune = qp._split_and_scale(
            X, y, args, 1, "None", split=_S(), stage="tune")
        assert len(final) == len(tune) == 3
        # They are selected from different row sets, so each must be a valid selection
        # of its own side; equality is permitted but not required.
        assert set(final) <= set(range(12)) and set(tune) <= set(range(12))


class TestCacheSpecInteraction:
    """What keeps an existing cache valid, and what must invalidate it."""

    @staticmethod
    def _args(**extra):
        base = {"seed": 42, "test_size": 0.3, "stratify": ["y"], "n_components": 8,
                "embeddings": ["pca"], "n_neighbors": 30}
        base.update(extra)
        return base

    def test_no_selection_leaves_the_spec_byte_identical(self):
        """The regression guard for full1's 171-dataset cache: a run that selects
        nothing must produce the spec it produced before this stage existed, so the
        files already on disk still serve it."""
        from qbiocode.apps.qprofiler import embedding_cache as emb_cache
        from qbiocode.apps.qprofiler import qprofiler as qp

        plain = emb_cache.embedding_spec(
            self._args(), "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        explicit_off = emb_cache.embedding_spec(
            self._args(feature_selection={"method": "none"}),
            "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        assert plain == explicit_off
        assert "feature_selection" not in plain
        assert qp._spec_settings(self._args()) == {}

    def test_a_selection_changes_the_spec(self):
        from qbiocode.apps.qprofiler import embedding_cache as emb_cache

        off = emb_cache.embedding_spec(
            self._args(), "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        on = emb_cache.embedding_spec(
            self._args(feature_selection={"method": "variance", "n_features": 16}),
            "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        assert off != on, "a selected run must not read an unselected run's file"

    def test_n_features_discriminates(self):
        """Two sweeps differing only in selection width must refuse each other's files;
        this is what makes n_features a safe axis to vary."""
        from qbiocode.apps.qprofiler import embedding_cache as emb_cache

        a = emb_cache.embedding_spec(
            self._args(feature_selection={"method": "variance", "n_features": 16}),
            "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        b = emb_cache.embedding_spec(
            self._args(feature_selection={"method": "variance", "n_features": 5000}),
            "d.csv", "a" * 64, "pca", 1, "MinMaxScaler")
        assert a != b


class TestSelectedFeaturesSidecar:
    """The artifact: which columns a fold actually used."""

    def test_rows_carry_rank_and_column_in_selection_order(self):
        from qbiocode.evaluation import protocol

        frame = protocol.selected_features_frame([7, 2, 5], "ds.csv|pca|i01", "final")
        assert list(frame.columns) == list(protocol.SELECTED_FEATURE_COLUMNS)
        assert list(frame["rank"]) == [0, 1, 2]
        assert list(frame["column"]) == [7, 2, 5]
        assert set(frame["stage"]) == {"final"}

    def test_no_selection_gives_an_empty_frame_with_the_header(self):
        """So a pass's sidecar set does not depend on the config: a reader finding no
        rows learns 'nothing was selected' instead of distinguishing a missing file."""
        from qbiocode.evaluation import protocol

        frame = protocol.selected_features_frame(None, "k", "final")
        assert frame.empty
        assert list(frame.columns) == list(protocol.SELECTED_FEATURE_COLUMNS)

    def test_columns_are_python_ints_so_the_csv_is_clean(self):
        from qbiocode.evaluation import protocol

        frame = protocol.selected_features_frame(np.array([3, 1], dtype=np.int64),
                                                 "k", "tune")
        assert [type(v) for v in frame["column"]] == [int, int] or \
            frame["column"].dtype.kind == "i"

    def test_resume_does_not_carry_this_sidecar(self):
        """resume.carry_sidecars filters rows by their 'model' column, and a
        selected-features row has none -- the selection belongs to the pass, not an arm.
        Listing the directory there would match nothing and silently drop the record."""
        from qbiocode.apps.qprofiler import qprofiler as qp
        from qbiocode.evaluation import protocol

        assert protocol.SELECTED_FEATURES_DIR not in qp._SIDECAR_DIRS
        assert "model" not in protocol.SELECTED_FEATURE_COLUMNS
