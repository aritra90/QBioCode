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

# ====== Base class imports ======
import numpy as np
import pandas as pd
import logging
from collections.abc import Sequence
import pickle
import os
import re
import csv
import time
# ====== Hydra imports ======
import hydra

# ====== Scikit-learn imports ======
from sklearn.model_selection import train_test_split

# ====== Qiskit imports ======
from qiskit_algorithms.utils import algorithm_globals

import sys

#: Best-effort guess at the checkout root, kept because ``folder_path`` in every
#: shipped config is written relative to it. The substitution only lands when the
#: current directory really does sit under one literally named ``QBioCode``, which
#: is not true of a GitHub source zip (``QBioCode-main``), a lowercase clone, or a
#: pip install -- hence ``_resolve_input_folder`` below rather than this alone.
#:
#: :meta private:
#:
#: Private to autodoc deliberately: its value is whatever directory the *documenting*
#: process ran in, so publishing it wrote the doc builder's own absolute filesystem
#: path into the API page on GitHub Pages.
dir_home = re.sub( 'QBioCode.*', 'QBioCode', os.getcwd() )
if os.path.isdir(dir_home):
    sys.path.append( dir_home )


def _resolve_input_folder(folder_path):
    """Return an existing directory for ``folder_path``, or ``None``.

    Candidates, in order:

    1. ``folder_path`` itself -- absolute, or relative to the current directory.
    2. ``folder_path`` under each ancestor of the current directory. This is what
       makes the tutorials work from a checkout whose top directory is not called
       ``QBioCode``: run from ``QBioCode-main/tutorial/QProfiler`` with
       ``folder_path: tutorial/QProfiler/data/ld_data``, candidate 1 points two
       levels too deep and the ``QBioCode``-derived root below does not exist,
       while the ancestor walk finds the real one.
    3. ``folder_path`` under a checkout root derived from the *current* directory.
    4. ``folder_path`` under ``dir_home``, the same root derived from the directory
       this module was imported from. Last, deliberately: ``dir_home`` is frozen at
       import time, so in a long-lived process (a notebook kernel, a test session)
       it can name a different checkout than the one the caller is standing in, and
       ranking it above the ancestor walk let a stale root silently win.
    """
    here = os.path.abspath(os.getcwd())
    candidates = [folder_path]
    while True:
        candidates.append(os.path.join(here, folder_path))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    derived = re.sub("QBioCode.*", "QBioCode", os.path.abspath(os.getcwd()))
    candidates.append(os.path.join(derived, folder_path))
    candidates.append(os.path.join(dir_home, folder_path))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None

# ====== Scaling and encoding functions imports ======
from qbiocode import scale_train_test, feature_encoding
from qbiocode import get_embeddings, resolve_embeddings
from qbiocode.embeddings import DEFAULT_EMBEDDING_MIN_FEATURES
from qbiocode.embeddings import check_embedding_name
# ====== Evaluation functions imports ====
#from qmlbench.evaluation.dataset_evaluation_no_var_threshold import evaluate2 # use this for moons/circles data, otherwise you'll run into an error with finding no features with minimum variance threshold
from qbiocode import evaluate
from qbiocode import model_run
from qbiocode.apps.qprofiler import embedding_cache as emb_cache

#: Config keys ``main`` reads unconditionally. Reported together rather than one
#: KeyError at a time, so a hand-written config can be fixed in a single pass.
_REQUIRED_CONFIG_KEYS = (
    "folder_path", "file_dataset", "embeddings", "n_components", "model",
    "seed", "q_seed", "test_size", "iter", "scaling", "backend", "n_jobs",
)


def _resolve_scaling(scaling):
    """Resolve the ``scaling`` config value to a scaler name for ``scale_train_test``.

    The shipped config writes ``scaling: ['True']`` and the original code tested
    it with ``'True' in args['scaling']``. That substring test accepted
    ``'MinMaxScalerTrue'``, silently ignored ``['true']``, and raised
    ``TypeError: argument of type 'bool' is not iterable`` for the most natural
    YAML of all -- ``scaling: true``. All four spellings are accepted here, and
    anything else is named as an error instead of quietly disabling scaling.

    The single-element unwrapping tests ``Sequence`` rather than ``list``: Hydra
    hands the config over as ``omegaconf.ListConfig``, which is a ``Sequence``
    but *not* a ``list`` subclass, so an ``isinstance(value, list)`` test passes
    every dict-based unit test and then rejects the shipped config's own
    ``scaling: ['True']`` on the real CLI path.

    Returns:
        str: ``'MinMaxScaler'``, ``'StandardScaler'`` or ``'None'``.

    Raises:
        ValueError: if the value is not a recognized flag or scaler name.
    """
    value = scaling
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(
                f"scaling accepts a single value; got {list(value)!r}. Use "
                f"scaling: ['True'], scaling: false, or a scaler name."
            )
        value = value[0]
    if isinstance(value, bool):
        return "MinMaxScaler" if value else "None"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return "MinMaxScaler"
        if lowered in ("false", "no", "0", "none"):
            return "None"
        if lowered == "minmaxscaler":
            return "MinMaxScaler"
        if lowered == "standardscaler":
            return "StandardScaler"
    raise ValueError(
        f"Unrecognized scaling {scaling!r}. Accepted: true/false (or ['True'] / "
        f"['False'] as the shipped config writes it), 'MinMaxScaler', "
        f"'StandardScaler', or 'None'."
    )


def _resolve_backend_alias(args, log):
    """Rewrite ``args['backend']`` from a config-facing alias to the internal name.

    The aliases (``statevector_simulator``, ``mps_simulator``) say which *simulator*
    runs, which the internal pair (``backend``, ``sim_method``) only says together.
    :func:`qbiocode.utils.qutils.normalize_backend` does that translation, but it was
    applied inside ``get_backend_session`` -- and eight other sites read the raw string
    and branch on ``args["backend"] != "simulator"``
    (:mod:`qbiocode.embeddings.embed`, ``compute_qsvc``/``qnn``/``vqc``/``pqk``/``qpl``,
    and ``ensure_tuning_is_affordable``). An alias reaching those comparisons is not an
    unknown-value error: ``statevector_simulator != "simulator"`` is simply *true*, so
    every one of them would take its hardware branch -- building a pass manager, and
    refusing to tune at all -- while the session itself ran locally as asked. Nothing
    raises; the run is just wrong and slow.

    So the alias is resolved once, here, before any of that code sees the config.
    ``normalize_backend`` is idempotent, so ``get_backend_session`` normalising again
    later is harmless.

    Args:
        args: the run config, mutated in place.
        log: logger for the resolution line.
    """
    from qbiocode.utils.qutils import normalize_backend

    raw = args.get("backend")
    resolved = normalize_backend(args)
    if resolved.get("backend") == raw and "sim_method" not in resolved:
        return
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(args, DictConfig):
            OmegaConf.set_struct(args, False)
    except ImportError:  # pragma: no cover - omegaconf ships with hydra
        pass
    args["backend"] = resolved["backend"]
    if "sim_method" in resolved:
        args["sim_method"] = resolved["sim_method"]
    if resolved["backend"] != raw:
        log.info(
            f"backend {raw!r} resolved to backend={resolved['backend']!r}"
            + (
                f", sim_method={resolved['sim_method']!r}"
                if "sim_method" in resolved
                else ""
            )
        )


def _resolve_model_lists(args, log):
    """Build ``args['model']`` from ``classical_model`` plus ``quantum_model``.

    Two lists instead of one, because the single ``model`` list mixed two populations
    whose costs differ by three orders of magnitude and whose results are compared
    *against each other*. Reading a run's scope off one flat list meant counting which
    names happened to be quantum; and a model filed on the wrong side of that
    comparison -- ``qsvc`` written into what the reader believed was the classical
    arm -- was invisible, because ``model`` has no notion of sides. Splitting the key
    makes the side an assertion the config states and this function checks.

    ``model`` remains the internal name: everything downstream
    (:mod:`qbiocode.evaluation.model_run`, the results columns, the tuning dispatch)
    reads ``args['model']``, and rewriting those would be a much larger change for no
    gain. So the two config keys are joined here, once, before any validation runs.

    Precedence:
      * Either new key present -> ``model`` is *derived* (classical first, then
        quantum) and an explicitly written ``model`` is an error rather than a
        silently-ignored key or a third source of truth.
      * Neither present -> ``model`` is required, exactly as before. Every existing
        config keeps working.

    Args:
        args: the run config, mutated in place so the rest of the run sees ``model``.
        log: logger for the resolution line.

    Raises:
        ValueError: if a classical name appears under ``quantum_model`` or vice versa,
            if either list holds duplicates, or if ``model`` is written alongside them.
    """
    from qbiocode.evaluation.model_run import QUANTUM_MODELS

    has_split = "classical_model" in args or "quantum_model" in args
    if not has_split:
        if "model" not in args:
            raise ValueError(
                "Config names none of 'model', 'classical_model' or 'quantum_model'. "
                "Prefer the two-list form: classical_model: ['lr', ...] and "
                "quantum_model: ['qsvc', ...]."
            )
        return

    if "model" in args and args["model"]:
        raise ValueError(
            "Config sets 'model' as well as 'classical_model'/'quantum_model'. "
            "'model' is derived from the other two, so writing all three leaves no "
            "single answer to which models run. Delete the 'model' line."
        )

    classical = [str(m) for m in (args.get("classical_model") or [])]
    quantum = [str(m) for m in (args.get("quantum_model") or [])]

    # Checked here, where the two lists still exist as separate objects. Once they are
    # concatenated, model_run can only report that a name is unknown -- not that a
    # known name was filed under the wrong population, which is the mistake that
    # corrupts a quantum-vs-classical comparison without failing anything.
    misfiled_q = [m for m in classical if m in QUANTUM_MODELS]
    if misfiled_q:
        raise ValueError(
            f"classical_model names quantum model(s) {misfiled_q}. The quantum models "
            f"are {sorted(QUANTUM_MODELS)}; move these to quantum_model. Left here "
            f"they would run normally and be counted as classical baselines."
        )
    misfiled_c = [m for m in quantum if m not in QUANTUM_MODELS]
    if misfiled_c:
        raise ValueError(
            f"quantum_model names {misfiled_c}, which {'is' if len(misfiled_c) == 1 else 'are'} "
            f"not quantum. The quantum models are {sorted(QUANTUM_MODELS)}; move these "
            f"to classical_model."
        )

    merged = classical + quantum
    duplicated = sorted({m for m in merged if merged.count(m) > 1})
    if duplicated:
        raise ValueError(
            f"Duplicate model(s) {duplicated} across classical_model and "
            f"quantum_model; each model may appear once."
        )
    if not merged:
        raise ValueError(
            "classical_model and quantum_model are both empty; there is nothing to "
            "fit."
        )

    # Hydra composes the config in struct mode, which rejects assignment to a key the
    # YAML does not define -- and the point here is that the YAML no longer defines
    # 'model'. Unlocked for this one assignment, and only when the object is a
    # DictConfig; a plain dict (the unit tests, and any direct caller) needs nothing.
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(args, DictConfig):
            OmegaConf.set_struct(args, False)
    except ImportError:  # pragma: no cover - omegaconf ships with hydra
        pass
    args["model"] = merged
    log.info(
        f"model resolved from classical_model + quantum_model: "
        f"{len(classical)} classical {classical} + {len(quantum)} quantum {quantum}"
    )


def _validate_config(args, log):
    """Check the whole config before any dataset is read.

    A QProfiler run is long: loading data, splitting, embedding and fitting
    quantum models takes minutes to hours. Every check below used to fire deep
    into that run, or not at all -- an empty ``embeddings`` list, or an ``iter``
    of 0, simply produced no results and exited 0, which is indistinguishable
    from a run whose models all failed.

    Returns:
        str: the resolved scaler name, so ``main`` does not re-derive it.

    Raises:
        ValueError: naming the offending key, the value received, and what is
            accepted.
    """
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in args]
    if missing:
        raise ValueError(
            f"Config is missing required key(s): {missing}. Start from the "
            f"packaged configs/config.yaml, which defines all of "
            f"{list(_REQUIRED_CONFIG_KEYS)}."
        )

    if not isinstance(args["iter"], int) or isinstance(args["iter"], bool) or args["iter"] < 1:
        raise ValueError(
            f"iter is the number of train/test splits and must be a positive "
            f"integer; got {args['iter']!r}. A value of 0 produces no results at "
            f"all while still exiting successfully."
        )
    if not 0.0 < float(args["test_size"]) < 1.0:
        raise ValueError(
            f"test_size is a proportion and must be strictly between 0 and 1; "
            f"got {args['test_size']!r}."
        )
    n_components = args["n_components"]
    if (
        not isinstance(n_components, int)
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise ValueError(
            f"n_components is the embedding width and must be a positive integer; "
            f"got {n_components!r}."
        )
    if not isinstance(args["n_jobs"], int) or args["n_jobs"] == 0:
        raise ValueError(
            f"n_jobs must be a non-zero integer (-1 means all cores); got "
            f"{args['n_jobs']!r}."
        )

    # Validated here rather than left to resolve_embeddings, which runs after the first
    # dataset has been loaded, split and scaled -- minutes into a run, for a typo.
    min_features = args.get("embedding_min_features", DEFAULT_EMBEDDING_MIN_FEATURES)
    if (
        isinstance(min_features, bool)
        or not isinstance(min_features, int)
        or min_features < 0
    ):
        raise ValueError(
            f"embedding_min_features is the feature count a dataset must exceed before "
            f"an embedding is applied, and must be a non-negative integer; got "
            f"{min_features!r}. Use 0 to embed regardless of width; omit the key for the "
            f"default of {DEFAULT_EMBEDDING_MIN_FEATURES}."
        )

    embeddings = list(args["embeddings"])
    if not embeddings:
        raise ValueError(
            "embeddings is empty; there is nothing to profile. Use ['none'] to "
            "run the models on the unreduced features."
        )
    # Validated as a set, up front: a typo in the last of six embeddings used to
    # surface only after the first five had been embedded and modelled.
    for name in embeddings:
        check_embedding_name(name)

    models = list(args["model"])
    if not models:
        raise ValueError(
            "model is empty; there is nothing to fit. See "
            "qbiocode.evaluation.model_run for the available model names."
        )

    # Checked here, not when the first cached embedding is read: that is after the first
    # dataset has been loaded and evaluated.
    _embedding_cache_dir(args)

    scaler_name = _resolve_scaling(args["scaling"])
    log.info(f"Feature scaling resolved to: {scaler_name}")
    return scaler_name


# ---------------------------------------------------------------------------
# The steps from input folder to embedded split. Each is a function because two
# programs run them: main, and the embedding-cache precompute
# (qbiocode.apps.qprofiler.embedding_cache), which must reproduce main's splits and
# scaling exactly for the features it writes to be the ones a job would have computed.
# ---------------------------------------------------------------------------
def _embedding_cache_dir(args):
    """The directory ``embedding_cache`` names, or None when the run embeds in-process.

    It must be absolute: hydra runs each job from its own output directory, so a relative
    path would name a different directory in every job, and a cache is only useful when
    all the jobs share it.

    Raises:
        ValueError: if the value is not a string, or not an absolute path.
    """
    value = args.get("embedding_cache")
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"embedding_cache is the directory the embedded features are read from, or "
            f"null to compute them in this run; got {value!r}."
        )
    path = os.path.expanduser(value.strip())
    if not os.path.isabs(path):
        raise ValueError(
            f"embedding_cache must be an absolute path; got {value!r}. Hydra runs every "
            f"job from its own output directory, so a relative path would name a "
            f"different directory in each job."
        )
    return path


def _input_folder(args):
    """The directory ``folder_path`` resolves to.

    Raises:
        ValueError: if it resolves to no directory, naming where it was looked for.
    """
    # Normalize path separators for cross-platform compatibility
    folder_path = args['folder_path'].replace('/', os.sep).replace('\\', os.sep)
    path_to_input = _resolve_input_folder(folder_path)
    if path_to_input is None:
        raise ValueError(
            f"folder_path {args['folder_path']!r} is not a directory. It was looked "
            f"for relative to the current directory ({os.getcwd()!r}), relative to "
            f"the derived checkout root ({dir_home!r}), and under every parent of "
            f"the current directory. Give an absolute path, or run from a directory "
            f"from which the relative path resolves."
        )
    return path_to_input


def _input_files(args, path_to_input):
    """The CSVs in ``path_to_input`` that ``file_dataset`` selects, in the order they run.

    Raises:
        ValueError: if it selects none.
    """
    if args['file_dataset'] == 'ALL':
        input_files = [file for file in os.listdir(path_to_input) if file.endswith('csv')]
    else:
        input_files = [file for file in os.listdir(path_to_input) if file in args['file_dataset'] and file.endswith('csv')]
    if not input_files:
        # Previously this produced a successful run with no output whatsoever,
        # which reads exactly like a run whose models all silently failed.
        selector = (
            "every .csv file" if args['file_dataset'] == 'ALL'
            else f"file_dataset={args['file_dataset']!r}"
        )
        raise ValueError(
            f"No input datasets matched {selector} in {path_to_input!r}. "
            f"Directory contents: {sorted(os.listdir(path_to_input))[:10]}"
        )
    return sorted(input_files)


def _read_dataset(path, args, log=None):
    """``(X, y, y_encoded)`` from one input CSV: the features, the label column as read,
    and the labels ordinal-encoded to ``0 .. k-1``. The label is the last column."""
    # Load data with optional index column support
    if args.get('index_col', False):
        # First column contains row names/IDs
        rawdata = pd.read_csv(path, sep=r'\t|,', index_col=0)
        if log is not None:
            log.info(f"Loaded dataset with row names from first column")
    else:
        # Standard loading without index column
        rawdata = pd.read_csv(path, sep=r'\t|,')

    X = rawdata.iloc[:, :-1].to_numpy()
    y = rawdata.iloc[:,-1:].to_numpy()
    y_encoded = feature_encoding(y, feature_encoding='OrdinalEncoder')
    y_encoded = y_encoded.reshape(-1)
    y_encoded = y_encoded.astype(int)
    return X, y, y_encoded


def _is_stratified(args):
    """Whether the splits are stratified on the label.

    ``stratify`` can be ``['y']``, ``['Y']``, or an empty list / None for no
    stratification.
    """
    use_stratify = args.get('stratify', [])
    return bool(use_stratify and len(use_stratify) > 0)


def _split_seed(args, iter):
    """The ``random_state`` of split ``iter`` (1-based) -- of its train/test split and of
    its embedding alike.

    Distinct-but-reproducible split per iteration: random_state = seed + iter makes every
    split different from the others, yet deterministic across reruns and independent of
    any other RNG consumers (embeddings etc.) that run before it.
    """
    split_seed = args['seed'] + iter
    return split_seed


def _split_and_scale(X, y_encoded, args, iter, scaler_name):
    """Split ``iter`` of one dataset, scaled.

    Returns ``(X_train, X_test, y_train, y_test, train_idx, test_idx)``. The last two are
    the rows of ``X`` on each side, which is how a cached embedding is checked to belong
    to this split. They are passed through ``train_test_split`` as a third array, and that
    does not move any row: its permutation depends only on the row count, the labels
    (when stratified) and the seed, and each array is then indexed with it.
    """
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y_encoded, np.arange(len(y_encoded)),
        stratify=y_encoded if _is_stratified(args) else None,
        test_size=args['test_size'],
        random_state=_split_seed(args, iter),
    )
    # Scale the features: fit one scaler on TRAIN and apply it to TEST (never fit a
    # separate scaler on the test set -- that would use test-set statistics).
    if scaler_name != 'None':
        X_train, X_test = scale_train_test(X_train, X_test, scaling=scaler_name)
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def _data_key(file, embed, n_components, iter):
    """The name of one (dataset, embedding, split) pass, which every cache it feeds uses
    as its key: projections, kernel dumps, tuned parameters and the embedding cache."""
    # os.path.splitext, not re.sub(r'\..*'): that regex truncated at the
    # FIRST dot, so any dataset whose name carries a decimal parameter lost
    # everything after it and collapsed onto a shared key. Two pairs in the
    # curated 84-dataset corpus collide that way --
    # GAMETES_Epistasis_2_Way_20atts_0.1H vs _0.4H, and
    # GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50 vs _75 -- and since the
    # colliding members share (n=1600, p=20), the row-count and width checks in
    # compute_pqk/compute_qpl cannot tell them apart. The second dataset of each
    # pair was scored on the first one's cached projections, without a warning,
    # and the parameter that differs between them is the one that sets the
    # difficulty the comparison is meant to measure.
    return '_'.join( [os.path.splitext( file )[0], embed, str(n_components), str(iter)])


def _embedding_settings(args):
    """Everything an embedding is given apart from the data and the seed.

    One function for both consumers: :func:`_embed` passes these to the embedding, and
    the embedding cache records them in each file's spec. So a setting added here is
    passed to the embedding and checked by the cache, with no other change needed.
    """
    return {
        "n_neighbors": args.get("n_neighbors", 30),
        "n_components": args["n_components"],
        "method": None,
        "quvine_args": args.get("quvine_args", {}),
    }


def _embed(embed, X_train, X_test, args, split_seed):
    """Embed one split in this process: ``(X_train_emb, X_test_emb)``."""
    return get_embeddings(
        embed, X_train, X_test,
        **_embedding_settings(args),
        # Seeded by the split, like train_test_split. Unseeded, UMAP read numpy's
        # global stream -- which a model run in this process (n_jobs: 1, one model per
        # config) re-seeds and advances -- and ran numba-parallel SGD, which is not
        # reproducible even from a fixed stream. Two jobs of one (dataset, embedding)
        # but different models therefore scored those models on DIFFERENT features
        # from the second split on, and nothing failed to say so.
        #
        # The seed makes repeat runs on the same CPU type identical. It does not make
        # runs on different CPU types agree: numba compiles UMAP for the host's
        # instruction set, and SGD amplifies the last-bit differences. On the 2026
        # pilot the UMAP jobs fell into three groups by host type, and the groups
        # computed different features on every split. Jobs that must see the same
        # features read them from embedding_cache instead.
        random_state=split_seed,
    )


# Begin the main function and instatiate Hydra class
# config_path=None allows --config-dir to work properly
def _append_model_row(path, row):
    """Append one model's result row to ``path``, widening the file if it has to.

    ``ModelResults.csv`` is written incrementally -- one row per model, as each model
    finishes -- so that a run interrupted half way still leaves usable output. The
    obvious way to do that, ``csv.writer`` plus a header written when the file is
    empty, is wrong as soon as two models contribute different keys, and QProfiler has
    two that routinely do: a *tuned* classical model reports ``BestParams_Tuned`` while
    an *untuned* one reports ``Model_Parameters``, and ``grid_search: True`` with
    ``tune_quantum: False`` -- the natural way to run a sweep, since a quantum fit per
    trial is expensive -- puts both in the same run.

    The result was a file whose header was narrower than its later rows, which
    ``pandas.read_csv`` refuses outright::

        ParserError: Expected 150 fields in line 7, saw 151

    so nothing downstream could read the results at all. Writing through
    :class:`csv.DictWriter` against the header already on disk fixes that: a row that
    introduces new columns rewrites the file with the union header and pads the earlier
    rows, and a row missing a column writes an empty cell there rather than shifting
    every later value left by one.

    Widening rewrites the file, which is ``O(rows)``, but it happens at most once per
    distinct key set -- in practice once, when the first untuned model follows a tuned
    one -- so the cost does not grow with the length of the run.

    Args:
        path (str): Path to the CSV. Created with a header if absent or empty.
        row (dict): Column name -> value for exactly one model on one
            (dataset, iteration, embedding).
    """
    header = None
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline='') as csvfile:
            header = next(csv.reader(csvfile), None)

    if header is None:
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row), restval='')
            writer.writeheader()
            writer.writerow(row)
        return

    added = [name for name in row if name not in header]
    if not added:
        # restval covers a row that is MISSING a column the header has, which is the
        # other half of the same problem: without it, csv.writer would emit the row's
        # values positionally and silently misalign every column after the gap.
        with open(path, 'a', newline='') as csvfile:
            csv.DictWriter(csvfile, fieldnames=header, restval='').writerow(row)
        return

    with open(path, newline='') as csvfile:
        existing = list(csv.DictReader(csvfile))
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header + added, restval='')
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


@hydra.main(config_path=None, config_name='config', version_base='1.1')
def main(args):
    """
    Main function to run the qprofiler. It initializes logging, sets up the environment, and processes datasets.
    The function reads datasets from the specified folder, applies feature encoding, splits the data into training and test sets,
    applies scaling and embeddings, and evaluates the models using various quantum machine learning methods.
    It logs the results and saves them in a structured format for further analysis. 
    The function also handles parallel processing of multiple machine learning methods and datasets.

    Args:
        args (dict): Configuration parameters for the profiler, including dataset paths, model parameters, and evaluation settings.

    Returns:
        None
    """
    beg_time = time.time() 
    log = logging.getLogger(__name__)
    log.info(f"Main program initiated")
    # Validate before touching args, not after. These three log lines used to sit
    # above this call and indexed 'n_jobs', 'model' and 'backend' directly, so a
    # config missing any of them died with a bare KeyError raised from a logging
    # statement -- the exact "error attributed to the wrong thing" that
    # _validate_config exists to replace, and it reported one missing key where
    # the validator reports all of them at once.
    # Before _validate_config, which requires 'model': with the two-list form that key
    # does not exist in the YAML at all and is derived here.
    _resolve_model_lists(args, log)
    # Must precede every consumer of args['backend'], not just get_backend_session.
    _resolve_backend_alias(args, log)
    scaler_name = _validate_config(args, log)

    # Authorise TabPFN's weight download before any worker starts, if the model was asked
    # for. The token has to reach the estimator as $TABPFN_TOKEN, and setting it here puts
    # it in the environment that joblib's workers inherit -- doing it inside the worker
    # would mean reading the credentials file once per model per split. Only the *source*
    # is logged; the token itself is never written to the log, which is committed into
    # results directories and pasted into issues.
    if "tabpfn" in args["model"]:
        from qbiocode.learning.compute_tabpfn import (
            TABPFN_DEFAULT_VERSION,
            tabpfn_versions_requiring_token,
        )
        from qbiocode.utils.tabpfn_account import load_tabpfn_token

        source = load_tabpfn_token(args)
        # Only the restricted checkpoints need one. Warning whenever a token is merely
        # absent announced a failure that was not going to happen on every job of the
        # first pilot, all of which pin the token-free 'v2'.
        needs_token = tabpfn_versions_requiring_token(args)
        if source:
            log.info(f"TabPFN token {source}")
        elif needs_token:
            log.warning(
                f"'tabpfn' is in the model list and this config selects "
                f"{', '.join(needs_token)}, whose weights need a licence accepted "
                f"against a Prior Labs account, but no API token was found -- so they "
                f"cannot be downloaded and the model will fail. Put the key in "
                f"~/.config/qbiocode/tabpfn.json as {{\"token\": \"...\"}}, set "
                f"tabpfn_json_path in the config, or export TABPFN_TOKEN. Alternatively "
                f"use model_version {TABPFN_DEFAULT_VERSION!r}, which needs neither."
            )
        else:
            log.info(
                f"TabPFN needs no API token here: this config selects only token-free "
                f"weights (default {TABPFN_DEFAULT_VERSION!r}, Apache 2.0 plus "
                f"attribution), which download anonymously and cache locally."
            )

    log.info(f"The number of ML methods being parallelized is {min(args['n_jobs'], len(args['model']))}")
    log.info(f"Chosen backend for quantum algorithms is: {args['backend']}")
    path_to_input = _input_folder(args)
    log.info(f"Reading datasets from {path_to_input}")
    input_files = _input_files(args, path_to_input)
    # Set: every embedding but 'none' is read from here, and none is computed in this
    # run. See embedding_cache.py for why, and for how the files are written.
    cache_dir = _embedding_cache_dir(args)
    if cache_dir:
        log.info(f"Embedded features are read from embedding_cache {cache_dir}")
        # Every cached embedding the run reads, for every dataset, is checked before
        # anything is fitted. A run whose cache is missing a file, or holds one written
        # under other settings, stops here, not hours in at the dataset or split that
        # first needs it.
        entries = []
        for name in input_files:
            path = os.path.join(path_to_input, name)
            X_raw, _, _ = _read_dataset(path, args)
            entries += emb_cache.plan(
                name, emb_cache.file_sha256(path), X_raw.shape[1], args, scaler_name
            )
        emb_cache.require(cache_dir, entries)

    # need to populate raw data evaluation for each file, so start an empty list
    appended_raw_data_eval = []
    
    # start looping over datasets
    # start count
    file_count = 0 
    for file in input_files:
        print(f"Processing file: {file}")
        # this is where the seed needs to be set so the splits are consistent
        np.random.seed(args['seed'])
        algorithm_globals.random_seed = args['q_seed']

        dataset_start_time = time.time()
        summary = {}
        model_results = {}
        summary.update({'Dataset':file})
        model_results.update({'Dataset':file})

        dataset_path = os.path.join(path_to_input, file)
        X, y, y_encoded = _read_dataset(dataset_path, args, log)
        y_map = dict(zip(y_encoded.astype(str), y.tolist()))
        summary.update({'label_mapping': y_map})
        
        # Binary classification is a hard requirement, not a preference.
        #
        # This used to warn and continue, calling multi-class support "experimental". It is
        # not experimental, it is absent: `modeleval` scores every model with
        # `roc_auc_score(y_test, y_predicted)` and passes no `multi_class`, so any dataset
        # with more than two classes raised
        #
        #     ValueError: multi_class must be in ('ovo', 'ovr')
        #
        # several hundred lines later, from inside the metrics helper, after the embeddings
        # had been computed and the models fitted. The warning therefore bought nothing: it
        # invited the user to proceed into a failure that was certain, and then reported it
        # against a parameter name they had never heard of. Refusing here names the dataset,
        # the class count, and the fact that this is a boundary rather than a bug.
        n_classes = len(np.unique(y_encoded))
        if n_classes != 2:
            raise ValueError(
                f"Dataset {file!r} has {n_classes} classes, and QProfiler supports binary "
                f"classification only. Its label mapping is {y_map}.\n\n"
                f"This is a boundary of the tool rather than a defect: every model is scored "
                f"through qbiocode.evaluation.modeleval, whose AUC calculation is binary-only, "
                f"so a multi-class run cannot produce a result no matter which models are "
                f"selected. Continuing would fail later, after the embeddings and fits had "
                f"already been computed.\n\n"
                f"To proceed, either restrict the dataset to two classes (a one-vs-rest or "
                f"one-vs-one split of the labels above), or drop it from 'file_dataset'."
            )

        # Hashed again rather than kept from the check above: a file edited since then
        # no longer matches the specs it was checked against, and emb_cache.load refuses it.
        if cache_dir:
            dataset_sha256 = emb_cache.file_sha256(dataset_path)

        # call and run evaluation functions
        df_dataset = pd.DataFrame(X)
        raw_data_eval = evaluate(df_dataset, y_encoded, file)
        appended_raw_data_eval.append(raw_data_eval)

        # create csv file storing the evaluation of the raw, unembedded data
        all_raw_data_evaluation = pd.concat(appended_raw_data_eval)
        all_raw_data_evaluation.to_csv('RawDataEvaluation.csv', index=False)
        
        # log info
        log.info(f"Started processing data set {file}")
        log.info(f"Dataset has {n_classes} classes: {np.unique(y_encoded).tolist()}")
        
        iter = 0
        # makes number of iterations an argument from config
        for iter in range(args['iter']):
        ## run all this in a loop N_times, while leaving the seed fixed above. The train_test_split will change at each iteration, but will be based on the seed.
            iter=iter+1
            # track iteration time
            iter_start_time = time.time()

            split_seed = _split_seed(args, iter)
            X_train, X_test, y_train, y_test, train_idx, test_idx = _split_and_scale(
                X, y_encoded, args, iter, scaler_name
            )
            log.info(
                f"Begin processing iteration (split) {iter} of {args['iter']} "
                + ("with stratified sampling" if _is_stratified(args) else "without stratification")
            )

            # Skip feature reduction on a dataset too narrow to justify it.
            #
            # Resolved per split rather than once per dataset because it reads
            # X_train.shape[1] -- the width the embedding would actually be fitted on.
            # That equals the file's feature count today, but deriving it from the array
            # in hand is what keeps this correct if a future step drops a column.
            effective_embeddings, skipped_embeddings = resolve_embeddings(
                args['embeddings'],
                X_train.shape[1],
                min_features=args.get(
                    'embedding_min_features', DEFAULT_EMBEDDING_MIN_FEATURES
                ),
            )
            if skipped_embeddings and iter == 1:
                # Once per dataset, not once per split: the decision cannot change
                # between splits of one file, and repeating it `iter` times reads like
                # a recurring problem rather than a stated policy.
                log.warning(
                    f"{file}: {X_train.shape[1]} features is not more than "
                    f"embedding_min_features="
                    f"{args.get('embedding_min_features', DEFAULT_EMBEDDING_MIN_FEATURES)}, "
                    f"so {skipped_embeddings} will NOT be applied and the models run on "
                    f"the unreduced features instead. Below that width a reduction costs "
                    f"information without buying anything -- these models encode one "
                    f"qubit per feature and handle this many directly. To embed anyway, "
                    f"set embedding_min_features: 0 in the config."
                )

            # Embed the training data and test data separately
            for embed in effective_embeddings:
                if embed == 'none':
                    log.info(f"No feature reduction (embedding) applied in this iteration")
                else:
                    log.info(f"Feature reduction (embedding) applied with {embed}")
                data_key = _data_key(file, embed, args["n_components"], iter)
                if cache_dir and embed != 'none':
                    X_train_emb, X_test_emb = emb_cache.load(
                        cache_dir,
                        data_key,
                        emb_cache.embedding_spec(
                            args, file, dataset_sha256, embed, iter, scaler_name
                        ),
                        train_idx,
                        test_idx,
                    )
                    source = f"read from {emb_cache.cache_file(cache_dir, data_key)}"
                else:
                    X_train_emb, X_test_emb = _embed(embed, X_train, X_test, args, split_seed)
                    source = "computed in this run"
                # The digest is what makes two jobs' features comparable after the fact:
                # jobs that used the same features log the same one.
                log.info(
                    f"Features of {data_key}: {X_train_emb.shape[1]} columns, "
                    f"sha256 {emb_cache.features_digest(X_train_emb, X_test_emb)}, {source}"
                )
                summary.update({'embeddings': embed})
                model_results.update({'embeddings': embed})
                
                # TODO: move PQK here as an embedding?

                # call and run evalution functions again if data is embedded, save outputs in the log file
                df_dataset = pd.DataFrame(X_train_emb)
                evaluate_data = evaluate(df_dataset, y_train, file)
                evaluate_data_listofdict = evaluate_data.to_dict(orient='records')
                evaluate_data_dict = {k: v for d in evaluate_data_listofdict for k, v in d.items()}
                # print(evaluate_data_dict)
                model_results.update(evaluate_data_dict)
                #log.info(f"\nThe characteristics of the embedding train dataset are: \n{evaluate_data}")
                summary.update({'iteration': iter})
                model_results.update({'iteration': iter})
                # `summary` is created ONCE per dataset (line 396), above both the
                # iteration and the embedding loop, and is only ever updated in place --
                # so any key a pass does not itself write survives from the previous
                # pass. model_run ends in `pd.melt(pd.concat(results)).dropna()`
                # (model_run.py:501), and dropna DELETES a column whose single value is
                # None rather than preserving it. `y_score_<model>` is None for any model
                # that exposed no usable ranking score, so without the clear below such a
                # pass inherits the PREVIOUS pass's y_score array and pairs it with the
                # current pass's y_test -- a post-hoc ROC-AUC or PR-AUC computed from
                # results.pkl would then score one split's probabilities against another
                # split's labels. Silent, and wrong only for the threshold-free metrics,
                # which is precisely why it has to be closed here rather than noticed
                # later.
                #
                # This is the same bug class as the `row_base = dict(model_results)` fix
                # a few lines below, which cured it for ModelResults.csv; the
                # `summary` -> results.pkl side was never fixed. All four per-model
                # prefixes are cleared, not just y_score: that also stops a stale
                # `results_<model>` from being re-written to the CSV under the current
                # pass's iteration and embedding.
                for stale in [
                    key
                    for key in summary
                    if key.startswith(
                        ("results_", "y_test_", "y_predicted_", "y_score_")
                    )
                ]:
                    del summary[stale]
                summary.update(model_run(X_train_emb, X_test_emb, y_train, y_test, data_key, args))
                # print(summary)
                # Snapshot the per-(dataset, iteration, embedding) part ONCE, then build
                # each model's row from a fresh copy of it. Merging each model's results
                # into the shared `model_results` instead -- which is what this did --
                # let one model's columns persist into the next model's row: with
                # `grid_search: True` and `tune_quantum: False`, pqk's row carried the
                # preceding model's `BestParams_Tuned` value, so a naive-Bayes
                # `var_smoothing` was reported as pqk's tuned hyperparameters. The rows
                # are independent observations, so they must be built independently.
                row_base = dict(model_results)
                for outerkey, outervalue in summary.items():
                    if outerkey.startswith("results_"):
                        _append_model_row(
                            'ModelResults.csv', {**row_base, **outervalue[0]}
                        )
                # Read existing summary data from the file, if any
                try:
                    with open("results.pkl", "rb") as pklfile:
                        results = pickle.load(pklfile)
                except FileNotFoundError:
                    results = []
                # #Append the list with new summary data
                results.append(summary)
                # Dumped to a temporary file and renamed into place rather than opened
                # 'wb' over the live one. The three lines above are a read-modify-write of
                # the WHOLE history -- every pass loads the accumulated list, appends one
                # summary, and writes all of it back -- so 'wb' truncated the only copy of
                # every previous pass before writing the new one. A job killed inside that
                # window (an LSF wall kill, an OOM, a Ctrl-C) therefore lost not the pass
                # in flight but the entire dataset's history, and left a half-written
                # pickle whose `pickle.load` raises UnpicklingError rather than the
                # FileNotFoundError the reader above is written to tolerate. The window is
                # small but it is entered once per pass, and the wall kill is exactly the
                # failure this run is sized against. os.replace is atomic within a
                # filesystem, so a reader sees either the previous complete pickle or the
                # new one; ModelResults.csv needs no such care because it is appended to,
                # never rewritten.
                tmp_pkl = 'results.pkl.tmp'
                with open(tmp_pkl, 'wb') as pklfile:
                    pickle.dump(results, pklfile)
                    pklfile.flush()
                    os.fsync(pklfile.fileno())
                os.replace(tmp_pkl, 'results.pkl')
            iter_run_time = time.time() - iter_start_time
            
        # start logging times
            log.info(f"The run time for iteration (split) {iter} is: {iter_run_time}")
            
        file_count += 1
        dataset_run_time = time.time() - dataset_start_time
        log.info(f"The total run time for data set {file} is: \n{dataset_run_time}")
        log.info(f"Program has processed {file_count} out of {len(input_files)} data sets")
        log.info(f"Program has {len(input_files)-file_count} data sets left to process")
    
    # log total run time of entire job
    total_run_time = time.time() - beg_time
    log.info(f"\nThe total run time of program is: \n{total_run_time}")

if __name__ == "__main__":
    main()

