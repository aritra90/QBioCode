"""QPL's ``data_map`` selects how features become gate angles, as PQK's does.

``compute_qpl`` hard-coded a local copy of the unit-coefficient map, so no QPL arm
could use qiskit's default ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)`` -- the map the
``eng_zz`` synthetic datasets are generated with. ``data_map='qiskit'`` now passes no
``data_map_func``. ``'unit'`` stays the default and keeps the historical cache key and
results row; the two maps never share a projection file.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

import qbiocode.utils.qutils as qutils
from qbiocode.learning._tuning import build_search_space
from qbiocode.learning.compute_qpl import compute_qpl


def _tiny_problem(n_features=2, n_train=10, n_test=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_train + n_test, n_features))
    y = np.array([0, 1] * ((n_train + n_test) // 2))
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _args(tmp_path, **extra):
    args = dict(backend="simulator", seed=42, q_seed=42, shots=100, resil_level=1,
                average="weighted", multi_class="raise", grid_search=False,
                cross_validation=2, qpl_projection_dir=str(tmp_path))
    args.update(extra)
    return args


def _run(tmp_path, **kwargs):
    X_train, X_test, y_train, y_test = _tiny_problem()
    return compute_qpl(X_train, X_test, y_train, y_test, _args(tmp_path), data_key="t",
                       encoding="ZZ", entanglement="linear", reps=1,
                       classical_models=["lr"], **kwargs)


def _train_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith("_train.npy"))


def test_unit_keeps_the_historical_cache_name(tmp_path):
    X_train, X_test, _, _ = _tiny_problem()
    _run(tmp_path)
    parts = ("ZZ", "linear", 1, "estimator", 2,
             qutils.dataset_fingerprint(X_train, X_test), 100, "simulator")
    fp = hashlib.sha256(repr(parts).encode()).hexdigest()[:10]
    assert _train_files(tmp_path) == [f"qpl_projection_t_{fp}_train.npy"]


def test_the_two_maps_never_share_a_file_and_differ(tmp_path):
    _run(tmp_path, data_map="unit")
    unit = _train_files(tmp_path)
    frame = _run(tmp_path, data_map="qiskit")
    both = _train_files(tmp_path)
    assert len(both) == 2 and set(unit) < set(both)
    qiskit_file = (set(both) - set(unit)).pop()
    a = np.load(os.path.join(tmp_path, unit[0]))
    b = np.load(os.path.join(tmp_path, qiskit_file))
    assert a.shape == b.shape and not np.allclose(a, b)
    params = frame["results_qpl_lr"].iloc[0]["Model_Parameters"]
    assert "'data_map': 'qiskit'" in str(params)


def test_unit_rows_do_not_record_the_default(tmp_path):
    frame = _run(tmp_path)
    assert "data_map" not in str(frame["results_qpl_lr"].iloc[0]["Model_Parameters"])


def test_bool_spelling_is_accepted(tmp_path):
    _run(tmp_path / "q", data_map="qiskit")
    _run(tmp_path / "b", data_map=False)
    assert _train_files(tmp_path / "q") == _train_files(tmp_path / "b")


def test_an_unknown_map_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="data_map"):
        _run(tmp_path / "never", data_map="pi")
    assert not os.path.exists(tmp_path / "never")


def test_data_map_is_searchable():
    space = build_search_space("qpl", {"data_map": ["unit", "qiskit"]})
    assert "data_map" in space
