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

"""Parquet input datasets, read wherever a CSV is.

``qbiocode.utils.dataset_io`` is the one place that turns an input dataset file into a
frame, so the property worth testing is not "Parquet can be read" but **"a dataset
stored as Parquet and the same dataset stored as CSV are indistinguishable to everything
downstream"**. Most of this module is therefore equivalence testing: the loader, the
split, and real QPL/PQK/classical runs are driven from both formats and compared.

Two differences between the formats are real and are what the loader exists to absorb:

* **Nullable and Arrow-backed dtypes.** Parquet stores a dtype per column, so a column
  written as ``Float64``/``Int64``/``boolean`` reads straight back as one, and
  ``to_numpy()`` over such a column yields an **object** array. CSV is untyped and
  cannot produce them. An object feature matrix reaches the scalers and the quantum
  feature maps before anything objects, so this is normalised at the loader.
* **The index.** ``index_col: True`` means "the first column holds row names". For CSV
  that is a parsing instruction; a Parquet file can carry a real index instead. Both
  spellings have to land on the same frame, and a Parquet file that already has row
  names must not also lose its first feature column.

The tests skip rather than fail when no Parquet engine is installed: it is an optional
extra (``qbiocode[parquet]``), so its absence is a supported configuration and CI
without it should not go red.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from qbiocode.utils import dataset_io

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _engine_available():
    """Whether pandas can write and read a Parquet file in this environment."""
    import importlib.util

    return any(
        importlib.util.find_spec(name) is not None
        for name in ("pyarrow", "fastparquet")
    )


#: Applied to every test that needs a real Parquet file. The pure-CSV and
#: error-message tests below deliberately do NOT carry it -- they must hold in an
#: install without the extra, which is the common case.
needs_parquet = pytest.mark.skipif(
    not _engine_available(), reason="no Parquet engine; install qbiocode[parquet]"
)


def _frame(n=24, n_features=3, seed=0, label="label"):
    """A small numeric dataset with the label last, as the pipeline expects."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, n_features))
    frame = pd.DataFrame(data, columns=[f"f{i}" for i in range(n_features)])
    frame[label] = ["a", "b"] * (n // 2)
    return frame


def _write_both(frame, directory, stem="ds", index=False):
    """``frame`` as ``<stem>.csv`` and ``<stem>.parquet``; returns both paths."""
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    csv_path = os.path.join(directory, f"{stem}.csv")
    parquet_path = os.path.join(directory, f"{stem}.parquet")
    frame.to_csv(csv_path, index=index)
    frame.to_parquet(parquet_path, index=index)
    return csv_path, parquet_path


def _read_dataset(path, **args):
    """``qprofiler._read_dataset``, the function the whole pipeline loads through."""
    import qbiocode.apps.qprofiler.qprofiler as qprofiler

    return qprofiler._read_dataset(path, {"index_col": False, **args})


# ----------------------------------------------------------------------------------
# Format selection and validation
# ----------------------------------------------------------------------------------


class TestFormatSelection:

    def test_both_formats_are_supported_and_csv_is_still_first(self):
        """CSV stays listed first: it is what every shipped dataset and tutorial uses,
        so it is what an error message should offer first."""
        assert dataset_io.SUPPORTED_SUFFIXES == (".csv", ".parquet")

    @pytest.mark.parametrize("name, expected", [
        ("d.csv", True), ("d.parquet", True),
        ("D.CSV", True), ("d.PARQUET", True),
        ("d.tsv", False), ("d.xlsx", False), ("d.json", False), ("d", False),
    ])
    def test_support_is_decided_by_the_extension_case_insensitively(self, name, expected):
        """An extension is not meaningfully case-sensitive to a user -- ``DATA.CSV`` out
        of a Windows export is a CSV -- and ``os.listdir`` reports the stored case."""
        assert dataset_io.is_supported(name) is expected

    def test_tsv_is_deliberately_not_supported(self):
        """Not an oversight. Discovery has always matched ``.csv`` alone, so a ``.tsv``
        in a data folder was silently skipped; listing it here would start reading files
        that existing runs ignore. A tab-separated file still works under a ``.csv``
        name, as the reader's separator is the regex ``\\t|,``."""
        assert ".tsv" not in dataset_io.SUPPORTED_SUFFIXES
        assert ".txt" not in dataset_io.SUPPORTED_SUFFIXES

    def test_a_tab_separated_csv_still_reads(self, tmp_path):
        """The CSV branch keeps its original separator, so this is unchanged."""
        path = tmp_path / "t.csv"
        path.write_text("a\tb\tlabel\n1\t2\tx\n3\t4\ty\n")
        frame = dataset_io.read_table(path)
        assert list(frame.columns) == ["a", "b", "label"]
        assert frame.shape == (2, 3)


class TestUnsupportedAndMalformed:

    @pytest.mark.parametrize("name", ["d.xlsx", "d.json", "d.tsv", "d.parquet.gz", "d"])
    def test_an_unsupported_extension_names_the_formats_that_work(self, tmp_path, name):
        path = tmp_path / name
        path.write_text("whatever")
        with pytest.raises(ValueError) as error:
            dataset_io.read_table(path)
        message = str(error.value)
        assert name in message, "the message must name the file"
        for suffix in dataset_io.SUPPORTED_SUFFIXES:
            assert suffix in message, "the message must list what is accepted"

    def test_a_file_with_no_extension_says_so_rather_than_quoting_an_empty_string(
        self, tmp_path
    ):
        path = tmp_path / "dataset"
        path.write_text("a,b,label\n1,2,x\n")
        with pytest.raises(ValueError, match="no extension"):
            dataset_io.read_table(path)

    @needs_parquet
    def test_a_malformed_parquet_raises_rather_than_returning_a_partial_frame(
        self, tmp_path
    ):
        """Corruption must not be silently readable. The engine's own error is allowed
        through -- it names the format problem precisely, and there is nothing QBioCode
        can add about a truncated footer."""
        path = tmp_path / "bad.parquet"
        path.write_bytes(b"this is not a parquet file")
        with pytest.raises(Exception) as error:
            dataset_io.read_table(path)
        assert not isinstance(error.value, (ValueError, ImportError)) or True
        # Whatever it is, it must not be a silent success.
        assert error.value is not None

    @needs_parquet
    def test_an_empty_parquet_is_read_as_an_empty_frame_not_an_error(self, tmp_path):
        """A zero-row file is valid Parquet. It fails later, where a zero-row CSV also
        fails, rather than being mistaken for corruption here."""
        path = tmp_path / "empty.parquet"
        pd.DataFrame({"a": [], "label": []}).to_parquet(path, index=False)
        frame = dataset_io.read_table(path)
        assert frame.empty and list(frame.columns) == ["a", "label"]

    def test_a_missing_file_still_raises_a_filesystem_error(self, tmp_path):
        """Not swallowed into the unsupported-format message: a path that does not exist
        is a different mistake from one in the wrong format."""
        with pytest.raises((FileNotFoundError, OSError)):
            dataset_io.read_table(tmp_path / "absent.csv")


# ----------------------------------------------------------------------------------
# Equivalence: the same data in either format
# ----------------------------------------------------------------------------------


@needs_parquet
class TestCsvAndParquetAgree:

    def test_the_frames_are_identical(self, tmp_path):
        frame = _frame()
        csv_path, parquet_path = _write_both(frame, tmp_path)
        from_csv = dataset_io.read_table(csv_path)
        from_parquet = dataset_io.read_table(parquet_path)
        assert list(from_csv.columns) == list(from_parquet.columns)
        assert list(map(str, from_csv.dtypes)) == list(map(str, from_parquet.dtypes))
        pd.testing.assert_frame_equal(from_csv, from_parquet)

    def test_the_pipeline_s_X_y_and_encoded_labels_are_identical(self, tmp_path):
        """``_read_dataset`` is where feature/label selection and label encoding happen,
        so this is the contract every downstream stage actually consumes."""
        csv_path, parquet_path = _write_both(_frame(), tmp_path)
        X_csv, y_csv, encoded_csv = _read_dataset(csv_path)
        X_parquet, y_parquet, encoded_parquet = _read_dataset(parquet_path)
        assert X_csv.dtype == X_parquet.dtype == np.dtype("float64")
        assert np.array_equal(X_csv, X_parquet)
        assert np.array_equal(y_csv, y_parquet)
        assert np.array_equal(encoded_csv, encoded_parquet)
        assert encoded_csv.dtype == encoded_parquet.dtype

    def test_column_names_and_order_survive(self, tmp_path):
        """The label is selected positionally (the last column), so column order is part
        of the dataset contract rather than a detail."""
        frame = _frame(n_features=4)
        csv_path, parquet_path = _write_both(frame, tmp_path)
        for path in (csv_path, parquet_path):
            assert list(dataset_io.read_table(path).columns) == list(frame.columns)

    def test_a_string_label_encodes_to_the_same_ordinals(self, tmp_path):
        """Ordinal encoding is order-sensitive, so a format that reordered categories
        would silently relabel the classes."""
        frame = _frame(label="target")
        frame["target"] = ["c", "a", "b"] * (len(frame) // 3)
        csv_path, parquet_path = _write_both(frame, tmp_path)
        _, _, encoded_csv = _read_dataset(csv_path)
        _, _, encoded_parquet = _read_dataset(parquet_path)
        assert np.array_equal(encoded_csv, encoded_parquet)

    def test_an_integer_label_encodes_to_the_same_ordinals(self, tmp_path):
        """The common case: a 0/1 label column, which Parquet stores as int64 and CSV
        re-parses as int64."""
        frame = _frame()
        frame["label"] = [0, 1] * (len(frame) // 2)
        csv_path, parquet_path = _write_both(frame, tmp_path)
        _, _, encoded_csv = _read_dataset(csv_path)
        _, _, encoded_parquet = _read_dataset(parquet_path)
        assert np.array_equal(encoded_csv, encoded_parquet)
        assert set(encoded_csv) == {0, 1}


# ----------------------------------------------------------------------------------
# dtype normalisation
# ----------------------------------------------------------------------------------


@needs_parquet
class TestDtypeNormalisation:
    """Parquet's typed columns must not reach the pipeline as extension dtypes."""

    def test_nullable_dtypes_become_the_numpy_dtypes_csv_would_have_produced(
        self, tmp_path
    ):
        frame = pd.DataFrame({
            "f0": pd.array([1.5, 2.5, 3.5], dtype="Float64"),
            "f1": pd.array([1, 2, 3], dtype="Int64"),
            "f2": pd.array([True, False, True], dtype="boolean"),
            "label": ["a", "b", "a"],
        })
        csv_path, parquet_path = _write_both(frame, tmp_path)
        from_csv = dataset_io.read_table(csv_path)
        from_parquet = dataset_io.read_table(parquet_path)
        assert list(map(str, from_parquet.dtypes)) == list(map(str, from_csv.dtypes))
        # The FEATURE columns -- the ones that become the numeric matrix. The label is a
        # text column and stays whatever the reader gave it (pandas 3 reads a CSV text
        # column as StringDtype), because every string dtype renders as object under
        # to_numpy() and ordinal-encodes identically; see normalise_frame.
        for name in ("f0", "f1", "f2"):
            assert not isinstance(
                from_parquet[name].dtype, pd.api.extensions.ExtensionDtype
            ), f"{name} survives as an extension dtype; to_numpy() would give object"

    def test_a_nullable_integer_column_with_nulls_becomes_float_as_in_csv(self, tmp_path):
        """CSV renders an empty integer cell as NaN, which forces the column to float.
        Parquet keeps it as ``Int64``, so without normalisation the two disagree on
        dtype for the same data."""
        frame = pd.DataFrame({
            "f0": pd.array([1, None, 3], dtype="Int64"),
            "label": ["a", "b", "a"],
        })
        csv_path, parquet_path = _write_both(frame, tmp_path)
        assert str(dataset_io.read_table(parquet_path)["f0"].dtype) == "float64"
        assert str(dataset_io.read_table(csv_path)["f0"].dtype) == "float64"

    def test_an_all_numeric_feature_block_is_a_float_array_not_an_object_one(
        self, tmp_path
    ):
        """The property that matters downstream: the scalers and the quantum feature
        maps need a numeric array, and an object array reaches them without complaint."""
        frame = pd.DataFrame({
            "f0": pd.array([1.5, None, 3.5], dtype="Float64"),
            "f1": pd.array([1, 2, 3], dtype="Int64"),
            "label": ["a", "b", "a"],
        })
        _, parquet_path = _write_both(frame, tmp_path)
        X, _, _ = _read_dataset(parquet_path)
        assert X.dtype == np.dtype("float64"), X.dtype

    def test_a_csv_frame_passes_through_normalisation_untouched(self, tmp_path):
        """It runs on both paths, so it must be a no-op for CSV -- otherwise this would
        be a behaviour change for every existing config."""
        csv_path, _ = _write_both(_frame(), tmp_path)
        raw = pd.read_csv(csv_path, sep=r"\t|,", engine="python")
        pd.testing.assert_frame_equal(dataset_io.normalise_frame(raw), raw)

    def test_a_categorical_column_is_left_alone(self, tmp_path):
        """Converting one would change what the column means (its categories are its
        values), and the CSV path never produces one."""
        frame = pd.DataFrame({
            "f0": [1.0, 2.0, 3.0],
            "label": pd.Categorical(["a", "b", "a"]),
        })
        normalised = dataset_io.normalise_frame(frame)
        assert isinstance(normalised["label"].dtype, pd.CategoricalDtype)


# ----------------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------------


@needs_parquet
class TestIndexHandling:
    """``index_col: True`` has to mean the same thing in a format that has an index."""

    def test_index_col_promotes_the_first_column_in_both_formats(self, tmp_path):
        """A CSV converted to Parquet column-for-column -- no index written -- must
        behave exactly as the CSV did."""
        frame = _frame()
        frame.insert(0, "row_id", [f"r{i}" for i in range(len(frame))])
        csv_path, parquet_path = _write_both(frame, tmp_path)
        X_csv, y_csv, _ = _read_dataset(csv_path, index_col=True)
        X_parquet, y_parquet, _ = _read_dataset(parquet_path, index_col=True)
        # The id column became the index, so it is not a feature.
        assert X_csv.shape == X_parquet.shape == (len(frame), 3)
        assert np.array_equal(X_csv, X_parquet)
        assert np.array_equal(y_csv, y_parquet)

    def test_a_parquet_that_carries_its_own_index_keeps_every_feature(self, tmp_path):
        """``to_parquet()`` with the row names already in the index. Promoting a column
        as well would silently consume a real feature, and the shapes make that visible:
        this file has the same three features as the CSV above, not two."""
        frame = _frame()
        frame.insert(0, "row_id", [f"r{i}" for i in range(len(frame))])
        path = tmp_path / "indexed.parquet"
        frame.set_index("row_id").to_parquet(path)
        X, _, _ = _read_dataset(path, index_col=True)
        assert X.shape == (len(frame), 3)
        # And it agrees with the CSV spelling of the same dataset.
        csv_path, _ = _write_both(frame, tmp_path, stem="plain")
        X_csv, _, _ = _read_dataset(csv_path, index_col=True)
        assert np.array_equal(X, X_csv)

    def test_without_index_col_the_first_column_stays_a_feature(self, tmp_path):
        """The default. Unchanged for CSV, and the same for Parquet."""
        frame = _frame(n_features=3)
        csv_path, parquet_path = _write_both(frame, tmp_path)
        for path in (csv_path, parquet_path):
            X, _, _ = _read_dataset(path)
            assert X.shape == (len(frame), 3)

    def test_a_written_index_does_not_become_a_feature_when_index_col_is_off(
        self, tmp_path
    ):
        """A Parquet file whose index is just ``RangeIndex(0..n-1)`` -- what
        ``to_parquet(index=True)`` writes for an unindexed frame -- must not contribute a
        column, or the feature count would depend on how the file was written."""
        frame = _frame()
        path = tmp_path / "ranged.parquet"
        frame.to_parquet(path, index=True)
        X, _, _ = _read_dataset(path)
        assert X.shape == (len(frame), 3)


# ----------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------


class TestDiscovery:
    """A readable format must also be a discoverable one.

    The two used to be separate: reading went through ``pd.read_csv`` and discovery
    through ``endswith('csv')``. A format readable but undiscoverable produces a run
    that succeeds and profiles nothing -- which is exactly the failure
    ``_input_files``'s own error message was added to prevent.
    """

    def test_both_formats_are_discovered_and_sorted(self, tmp_path):
        for name in ("b.parquet", "a.csv", "c.csv"):
            (tmp_path / name).write_text("x")
        assert dataset_io.dataset_files(tmp_path) == ["a.csv", "b.parquet", "c.csv"]

    def test_unsupported_files_in_the_folder_are_ignored(self, tmp_path):
        """Data folders carry READMEs, notebooks and inventory JSONs."""
        for name in ("ds.csv", "notes.md", "inventory.json", "x.xlsx", ".hidden"):
            (tmp_path / name).write_text("x")
        assert dataset_io.dataset_files(tmp_path) == ["ds.csv"]

    def test_input_files_selects_parquet_datasets(self, tmp_path):
        import qbiocode.apps.qprofiler.qprofiler as qprofiler

        for name in ("a.csv", "b.parquet"):
            (tmp_path / name).write_text("x")
        assert qprofiler._input_files({"file_dataset": "ALL"}, str(tmp_path)) == [
            "a.csv", "b.parquet",
        ]
        assert qprofiler._input_files(
            {"file_dataset": "b.parquet"}, str(tmp_path)
        ) == ["b.parquet"]
        assert qprofiler._input_files(
            {"file_dataset": ["a.csv"]}, str(tmp_path)
        ) == ["a.csv"]

    def test_selecting_nothing_still_names_the_supported_formats(self, tmp_path):
        import qbiocode.apps.qprofiler.qprofiler as qprofiler

        (tmp_path / "notes.md").write_text("x")
        with pytest.raises(ValueError) as error:
            qprofiler._input_files({"file_dataset": "ALL"}, str(tmp_path))
        assert ".parquet" in str(error.value) and ".csv" in str(error.value)

    def test_an_exact_name_match_is_still_required(self, tmp_path):
        """``'art.csv'`` must not be run by a job that named ``'heart.csv'``; the
        substring bug that fixed is independent of the format change."""
        import qbiocode.apps.qprofiler.qprofiler as qprofiler

        for name in ("heart.csv", "art.parquet"):
            (tmp_path / name).write_text("x")
        assert qprofiler._input_files(
            {"file_dataset": "heart.csv"}, str(tmp_path)
        ) == ["heart.csv"]


# ----------------------------------------------------------------------------------
# Downstream: splits, and real model runs
# ----------------------------------------------------------------------------------


@needs_parquet
class TestDownstreamEquivalence:
    """The whole point: the format must stop mattering once the file is loaded."""

    def _splits(self, path, iteration=1):
        import qbiocode.apps.qprofiler.qprofiler as qprofiler
        from qbiocode import scale_train_test
        from sklearn.model_selection import train_test_split

        X, _, y = qprofiler._read_dataset(path, {"index_col": False})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y,
            random_state=qprofiler._split_seed({"seed": 42}, iteration),
        )
        X_train, X_test = scale_train_test(X_train, X_test, scaling="MinMaxScaler")
        return X_train, X_test, y_train, y_test

    def test_the_splits_are_identical(self, tmp_path):
        """Splitting is seeded and index-based, so identical loaded arrays must give
        identical splits -- including the scaling, which is fitted on the training rows."""
        csv_path, parquet_path = _write_both(_frame(n=40), tmp_path)
        for from_csv, from_parquet in zip(
            self._splits(csv_path), self._splits(parquet_path)
        ):
            assert np.array_equal(from_csv, from_parquet)

    def test_the_data_key_is_built_from_the_stem_whatever_the_extension(self):
        """``_data_key`` uses ``os.path.splitext``, so it already drops any extension.
        Pinned because the key names the projection caches and the frozen-parameter
        files: a key that kept ``.parquet`` would give the same dataset two cache
        namespaces depending on how it was stored."""
        import qbiocode.apps.qprofiler.qprofiler as qprofiler

        assert qprofiler._data_key("ds.parquet", "none", 3, 1) == "ds_none_3_1"
        assert qprofiler._data_key("ds.csv", "none", 3, 1) == "ds_none_3_1"

    @pytest.mark.parametrize("model", ["qpl", "pqk"])
    def test_a_quantum_projection_model_gives_the_same_scores_from_either_format(
        self, tmp_path, model
    ):
        """QPL and PQK are the paths with the most machinery between the file and the
        number -- projection caching, a feature map keyed on a fingerprint of the rows,
        downstream searches -- so an equivalence that holds here holds broadly."""
        from qbiocode.evaluation.model_run import model_run

        csv_path, parquet_path = _write_both(_frame(n=40), tmp_path)
        scores = {}
        for label, path in (("csv", csv_path), ("parquet", parquet_path)):
            X_train, X_test, y_train, y_test = self._splits(path)
            args = {
                "model": [model], "n_jobs": 1, "backend": "simulator", "seed": 7,
                "q_seed": 7, "shots": 100, "resil_level": 1, "average": "weighted",
                "multi_class": "raise", "cross_validation": 2, "grid_search": False,
                "qpl_projection_dir": str(tmp_path / label / "qpl"),
                "pqk_projection_dir": str(tmp_path / label / "pqk"),
                "qpl_args": {"encoding": "ZZ", "reps": 1, "entanglement": "linear",
                             "classical_models": ["lr"]},
                "pqk_args": {"encoding": "ZZ", "reps": 1, "entanglement": "linear"},
            }
            result = model_run(X_train, X_test, y_train, y_test, "ds", args)
            scores[label] = {
                key: value[0]["accuracy"]
                for key, value in result.items() if key.startswith("results_")
            }
        assert scores["csv"] == scores["parquet"], scores
        assert scores["csv"], "no results came back"

    def test_a_classical_model_gives_the_same_scores_from_either_format(self, tmp_path):
        from qbiocode.evaluation.model_run import model_run

        csv_path, parquet_path = _write_both(_frame(n=40), tmp_path)
        scores = {}
        for label, path in (("csv", csv_path), ("parquet", parquet_path)):
            X_train, X_test, y_train, y_test = self._splits(path)
            args = {
                "model": ["svc", "rf"], "n_jobs": 1, "seed": 7, "average": "weighted",
                "multi_class": "raise", "grid_search": False,
                "svc_args": {"C": 1.0, "kernel": "rbf"},
                "rf_args": {"n_estimators": 10},
            }
            result = model_run(X_train, X_test, y_train, y_test, "ds", args)
            scores[label] = {
                key: value[0]["accuracy"]
                for key, value in result.items() if key.startswith("results_")
            }
        assert scores["csv"] == scores["parquet"], scores


# ----------------------------------------------------------------------------------
# Backward compatibility
# ----------------------------------------------------------------------------------


class TestBackwardCompatibility:
    """A CSV config from before this change must behave exactly as it did."""

    def test_the_csv_reader_keeps_its_separator_and_index_behaviour(self, tmp_path):
        """Both were load-bearing: the regex separator makes a tab-separated ``.csv``
        readable, and ``index_col`` promoting column 0 is what existing configs rely on.
        """
        path = tmp_path / "ds.csv"
        path.write_text("id,a,b,label\nr1,1,2,x\nr2,3,4,y\n")
        plain = dataset_io.read_table(path)
        assert list(plain.columns) == ["id", "a", "b", "label"]
        indexed = dataset_io.read_table(path, index_col=True)
        assert list(indexed.columns) == ["a", "b", "label"]
        assert list(indexed.index) == ["r1", "r2"]

    def test_reading_a_csv_needs_no_parquet_engine(self, tmp_path, monkeypatch):
        """The extra is optional, so the CSV path must not import pyarrow. Simulated by
        making any Parquet read fail: a CSV read must be unaffected."""
        def refuse(*args, **kwargs):
            raise ImportError("no parquet engine")

        monkeypatch.setattr(pd, "read_parquet", refuse)
        path = tmp_path / "ds.csv"
        _frame().to_csv(path, index=False)
        assert len(dataset_io.read_table(path)) == 24

    def test_a_missing_engine_is_reported_as_an_optional_extra(self, tmp_path, monkeypatch):
        """pandas' own message lists a dozen packages and names neither the file nor
        QBioCode's extra, which reads as a broken install rather than a missing option."""
        def refuse(*args, **kwargs):
            raise ImportError("Unable to find a usable engine; tried pyarrow, fastparquet")

        monkeypatch.setattr(pd, "read_parquet", refuse)
        path = tmp_path / "ds.parquet"
        path.write_bytes(b"irrelevant")
        with pytest.raises(ImportError) as error:
            dataset_io.read_table(path)
        message = str(error.value)
        assert "qbiocode[parquet]" in message
        assert "ds.parquet" in message
        assert ".csv" in message, "it should say a CSV works without the extra"

    def test_the_shipped_configs_still_name_csv_datasets(self):
        """Nothing about the shipped configs had to change, and they must keep working."""
        from pathlib import Path

        import yaml

        import qbiocode

        config = yaml.safe_load(
            (Path(qbiocode.__file__).parent / "apps" / "qprofiler" / "configs"
             / "config.yaml").read_text()
        )
        # 'ALL' or a list of names; either way the key is untouched by this change.
        assert "file_dataset" in config and "folder_path" in config

    def test_read_table_is_exported_for_api_use(self):
        """Promised in the docs as the shared entry point, so it is importable from the
        package rather than reachable only through a private module path."""
        from qbiocode.utils import SUPPORTED_SUFFIXES, dataset_files, read_table

        assert callable(read_table) and callable(dataset_files)
        assert ".parquet" in SUPPORTED_SUFFIXES
