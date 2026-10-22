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

"""Reading an input dataset, whatever tabular format it is stored in.

One place decides which formats QProfiler accepts as *input datasets* and how a file in
each becomes the frame the rest of the pipeline sees. Everything downstream -- feature
and label selection, encoding, splitting, embedding, the classical and quantum learners
-- works on that frame and never on the file, so adding a format here adds it to the
whole pipeline rather than to one entry point.

This is deliberately only about **input datasets**. QBioCode also reads and writes its
own result tables (``ModelResults.csv``, ``RawDataEvaluation.csv``, the
``qc_winner_finder`` outputs) and those stay CSV: they are an output contract that
downstream tooling and published analyses already read, not a user-supplied input whose
format is the user's choice.

Why the format cannot simply be "whatever pandas guesses"
---------------------------------------------------------
CSV and Parquet do not round-trip the same frame, and two of the differences change the
numbers the pipeline computes rather than merely how it reads them:

* **Nullable and Arrow-backed dtypes.** Parquet stores a type per column, so a column
  written from ``pd.array([1.5, None], dtype='Float64')`` reads back as ``Float64``, and
  ``DataFrame.to_numpy()`` over such columns yields an **object** array. CSV has no types
  at all, so the same data parses as ``float64``. An object feature matrix does not fail
  loudly -- it reaches ``StandardScaler`` and the quantum feature maps, which is far
  downstream of anything that names the file. :func:`normalise_frame` converts those back
  to numpy dtypes so a Parquet input gives the pipeline exactly what a CSV input does.
* **The index.** ``index_col: True`` means "the first column holds row names". In CSV
  that is a parsing instruction, because a CSV has no index of its own. A Parquet file
  *can* carry a real index, and usually does not. Both spellings have to end up as the
  same frame; see :func:`read_table`.

Supported formats
-----------------
:data:`SUPPORTED_SUFFIXES` is the single list, used for reading, for discovering the
datasets in a folder, and for the error message naming what is accepted. ``.parquet``
needs an engine (``pyarrow`` or ``fastparquet``); it is an optional dependency, so the
error says how to install it rather than letting pandas raise about a missing module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pandas as pd

#: Input-dataset extensions QProfiler accepts, lowercase and dot-prefixed.
#:
#: ``.tsv``/``.txt`` are not listed even though the CSV reader below would parse them --
#: its separator is the regex ``\t|,`` -- because they never were: discovery has always
#: matched ``.csv`` alone, so a ``.tsv`` in a data folder was silently skipped rather
#: than read. Listing them here would start reading files that existing runs ignore,
#: which is a behaviour change dressed as a format addition. A tab-separated file still
#: works under a ``.csv`` name, as it always has.
SUPPORTED_SUFFIXES = (".csv", ".parquet")

#: Formats whose reader needs a third-party engine, mapped to the extra that provides it.
_ENGINE_HINTS = {".parquet": "pyarrow"}


def dataset_suffix(path: str | os.PathLike) -> str:
    """The lowercased extension of ``path``, for dispatch and validation.

    Lowercased because an extension is not meaningfully case-sensitive to a user --
    ``DATA.CSV`` out of a Windows export is a CSV -- and ``os.listdir`` reports whatever
    case the filesystem stored.
    """
    return os.path.splitext(os.fspath(path))[1].lower()


def is_supported(path: str | os.PathLike) -> bool:
    """Whether ``path``'s extension is one :func:`read_table` can read."""
    return dataset_suffix(path) in SUPPORTED_SUFFIXES


def _unsupported(path: str | os.PathLike) -> ValueError:
    """The error for a file whose extension names no reader."""
    suffix = dataset_suffix(path)
    described = repr(suffix) if suffix else "no extension"
    return ValueError(
        f"Cannot read input dataset {os.fspath(path)!r}: {described} is not a supported "
        f"format. QProfiler reads {', '.join(SUPPORTED_SUFFIXES)}. Convert the file, or "
        f"rename it if its contents are already in one of those formats."
    )


def _missing_engine(path: str | os.PathLike, suffix: str, error: Exception) -> ImportError:
    """The error for a supported format whose reader engine is not installed.

    pandas raises ``ImportError: Unable to find a usable engine...`` with a long list of
    packages and no mention of the file or of QBioCode's extra, which reads as a broken
    install rather than a missing optional dependency.
    """
    extra = _ENGINE_HINTS.get(suffix, "")
    return ImportError(
        f"Reading {os.fspath(path)!r} needs a {suffix} engine, which is not installed.\n"
        f"Error: {type(error).__name__}: {error}\n\n"
        f"Install it with:\n"
        f'  pip install "qbiocode[parquet]"\n'
        f"or:\n"
        f"  pip install {extra}\n\n"
        f"Alternatively, supply the dataset as .csv."
    )


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A frame whose columns hold numpy dtypes, as the CSV reader would have produced.

    The pipeline's contract is ``DataFrame.to_numpy()`` on the feature block giving a
    numeric array. A pandas nullable extension dtype (``Int64``, ``Float64``,
    ``boolean``) or an Arrow-backed one breaks that: ``to_numpy()`` over them returns an
    **object** array, and an object array reaches the scalers and the quantum feature
    maps before anything complains. CSV cannot produce such a column -- it is untyped, so
    integers with missing values parse as ``float64`` -- but Parquet stores the dtype and
    reads it straight back.

    Converted with ``convert_dtypes(dtype_backend='numpy_nullable')`` inverted via
    ``to_numpy``-compatible dtypes: each extension column is cast to its numpy
    equivalent, with integer columns holding nulls becoming ``float64`` exactly as the
    CSV parser would render them. Columns already holding numpy dtypes are untouched, so
    a CSV frame passes through unchanged and this is safe to call on both paths.

    **Only numeric columns are touched.** String columns are left exactly as the reader
    produced them, whatever dtype that is -- ``object``, pandas' ``StringDtype``, or an
    Arrow ``large_string``. All three render as ``object`` under ``to_numpy()`` and
    ordinal-encode to the same integers, so there is nothing to fix; and pandas 3 reads a
    text column out of a CSV as ``StringDtype``, so converting them would mean *changing*
    the CSV path -- a behaviour change for every existing config, to no end. The same
    goes for a categorical column, where the categories are the values.
    """
    converted = {}
    for name, column in frame.items():
        dtype = column.dtype
        # Extension dtypes are the ones to_numpy() renders as object. `pyarrow` dtypes
        # and pandas' own nullable ones both report True here; plain numpy ones do not.
        if not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            continue
        # Numeric only; see the docstring for why strings and categoricals are not.
        if not (
            pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_numeric_dtype(dtype)
        ) or isinstance(dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_bool_dtype(dtype):
            # A nullable boolean cannot be a numpy bool array, so nulls force float --
            # which is also what the pipeline needs, since a bool feature is used as 0/1.
            converted[name] = (
                column.astype("float64") if column.isna().any() else column.astype(bool)
            )
        elif pd.api.types.is_integer_dtype(dtype):
            # Nulls force float, exactly as the CSV parser renders an empty integer cell.
            converted[name] = (
                column.astype("float64") if column.isna().any() else column.astype("int64")
            )
        else:
            # Every remaining numeric extension dtype: float, which is what a CSV column
            # of the same values parses as.
            converted[name] = column.astype("float64")
    if not converted:
        return frame
    frame = frame.copy()
    for name, column in converted.items():
        frame[name] = column
    return frame


def read_table(
    path: str | os.PathLike,
    *,
    index_col: bool = False,
    args: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One input dataset as a frame, read by whichever reader its extension names.

    The one place an input dataset is turned into a frame. The result is normalised (see
    :func:`normalise_frame`) so that the same data stored as CSV and as Parquet gives
    frames the rest of the pipeline cannot tell apart.

    Args:
        path: The dataset file. Its extension selects the reader.
        index_col: Whether the first column holds row names rather than a feature.
            Handled per format, because the formats differ in what an index is:

            * **CSV** has no index, so this is a parsing instruction and the first
              column is promoted (``pd.read_csv(index_col=0)``), as it always was.
            * **Parquet** stores an index of its own. When the file already carries a
              non-trivial one, that index *is* the row names and nothing is promoted --
              promoting a column as well would consume a real feature. When the file
              carries only a default ``RangeIndex`` (what ``to_parquet(index=False)``
              writes), the first column is promoted, so a CSV converted to Parquet
              column-for-column behaves identically.
        args: Optional run config, read only for ``index_col`` when the argument is not
            given. Lets a caller pass the config through rather than unpacking it.

    Returns:
        pandas.DataFrame: The dataset. Features and label are selected downstream (the
        label is the last column), which is format-independent by construction.

    Raises:
        ValueError: If the extension names no supported reader.
        ImportError: If the format is supported but its engine is not installed.
    """
    if args is not None and not index_col:
        index_col = bool(args.get("index_col", False))
    suffix = dataset_suffix(path)
    if suffix not in SUPPORTED_SUFFIXES:
        raise _unsupported(path)

    if suffix == ".csv":
        # sep=r'\t|,' and the python engine: unchanged from the original single call
        # site, so a comma- or tab-separated .csv reads exactly as it did. Passed
        # explicitly to silence pandas' ParserWarning about falling back to it, which
        # this regex separator triggers on every read.
        frame = pd.read_csv(
            path, sep=r"\t|,", engine="python", index_col=0 if index_col else None
        )
        return normalise_frame(frame)

    try:
        frame = pd.read_parquet(path)
    except ImportError as error:
        raise _missing_engine(path, suffix, error) from error
    if index_col and _has_default_index(frame):
        # Only when the file brought no index of its own; see the `index_col` docs.
        frame = frame.set_index(frame.columns[0])
    return normalise_frame(frame)


def _has_default_index(frame: pd.DataFrame) -> bool:
    """Whether ``frame``'s index carries no information a column would.

    True for the positional ``RangeIndex`` a ``to_parquet(index=False)`` file reads back
    with, which means "this file has no row names". An index with a name, or one whose
    values are not ``0..n-1``, is real row names and must not be displaced.
    """
    index = frame.index
    if index.name is not None:
        return False
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def dataset_files(directory: str | os.PathLike) -> list[str]:
    """The names of every readable input dataset in ``directory``, sorted.

    The discovery counterpart of :func:`read_table`: both consult
    :data:`SUPPORTED_SUFFIXES`, so a format cannot be readable but undiscoverable (or
    the reverse, which is how a folder of ``.parquet`` files would produce a run that
    succeeds and profiles nothing).
    """
    return sorted(name for name in os.listdir(directory) if is_supported(name))


def describe_supported() -> str:
    """The supported formats as a phrase for an error or log message."""
    return " or ".join(SUPPORTED_SUFFIXES)


__all__ = [
    "SUPPORTED_SUFFIXES",
    "dataset_files",
    "dataset_suffix",
    "describe_supported",
    "is_supported",
    "normalise_frame",
    "read_table",
]
