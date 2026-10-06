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

"""PQK's ``data_map`` selects how features become gate angles.

``compute_pqk`` used to hard-code
:func:`qbiocode.utils.qutils.unit_coefficient_data_map`, so a PQK arm could never
use qiskit's default map ``phi(x_i, x_j) = (pi - x_i)(pi - x_j)`` -- the map the
``eng_zz`` synthetic datasets are generated with -- and no aligned PQK arm could be
configured. ``data_map='qiskit'`` now passes no ``data_map_func``.

Two properties matter beyond the new option working:

1. ``'unit'`` (the default) is exactly the old behaviour: the same feature map and
   the same projection-cache filename, so caches already on disk stay valid.
2. ``'unit'`` and ``'qiskit'`` never share a cache file.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

import qbiocode.utils.qutils as qutils
from qbiocode.learning._tuning import build_search_space
from qbiocode.learning.compute_pqk import _resolve_data_map, compute_pqk, compute_pqk_opt

N_TRAIN, N_TEST, N_FEATURES = 10, 4, 2


def _dataset(n_train=N_TRAIN, n_test=N_TEST):
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(n_train, N_FEATURES)),
        rng.normal(size=(n_test, N_FEATURES)),
        np.array([0, 1] * (n_train // 2)),
        np.array([0, 1] * (n_test // 2)),
    )


def _args(projection_dir, **extra):
    args = {
        "backend": "simulator",
        "seed": 42,
        "shots": 100,
        "grid_search": False,
        "pqk_projection_dir": str(projection_dir),
    }
    args.update(extra)
    return args


def _projections(projection_dir):
    return sorted(
        name
        for name in os.listdir(projection_dir)
        if name.startswith("pqk_projection_") and name.endswith(".npy")
    )


def _run(projection_dir, args_extra=None, **kwargs):
    X_train, X_test, y_train, y_test = _dataset()
    params = {"encoding": "ZZ", "entanglement": "linear", "reps": 2, **kwargs}
    return compute_pqk(
        X_train, X_test, y_train, y_test,
        _args(projection_dir, **(args_extra or {})), data_key="ds", **params,
    )


def _legacy_fingerprint(encoding="ZZ", entanglement="linear", reps=2):
    """The cache digest as compute_pqk computed it before ``data_map`` existed.

    Reimplemented on purpose (unlike test_pqk_cache_key.py): the property under test
    is that the key did NOT change for the default, so it has to be pinned to the
    historical formula rather than to whatever the code does now.
    """
    X_train, X_test, _, _ = _dataset()
    parts = (encoding, entanglement, reps, "estimator", N_FEATURES)
    parts = parts + (qutils.dataset_fingerprint(X_train, X_test), 100, "simulator")
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:10]


def _model_parameters(frame, model):
    # record_tuned_params writes the tuned values under BestParams_Tuned when present.
    metrics = frame["results_" + model].iloc[0]
    return metrics.get("BestParams_Tuned", metrics.get("Model_Parameters"))


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("unit", "unit"), ("qiskit", "qiskit"), (True, "unit"), (False, "qiskit"),
     (np.bool_(True), "unit")],
)
def test_accepted_spellings(value, expected):
    assert _resolve_data_map(value) == expected


@pytest.mark.parametrize("value", ["Unit", "default", "", None, 1, 0, "qiskit "])
def test_invalid_values_raise_naming_the_options(value, tmp_path):
    with pytest.raises(ValueError, match=r"data_map must be one of \['unit', 'qiskit'\]"):
        _run(tmp_path, data_map=value)
    # Validation runs before the cache directory is created.
    assert not os.path.exists(os.path.join(tmp_path, "checkpoints"))


# --- the feature map --------------------------------------------------------


def _captured_feature_map(monkeypatch, tmp_path, **kwargs):
    """Run compute_pqk and return the ``data_map_func`` and circuit it built."""
    seen = {}
    real = qutils.get_feature_map

    def _spy(*a, **kw):
        fm = real(*a, **kw)
        seen["data_map_func"] = kw.get("data_map_func")
        seen["circuit"] = fm[0]
        return fm

    monkeypatch.setattr(qutils, "get_feature_map", _spy)
    _run(tmp_path, **kwargs)
    # Restore the real builder before the caller builds its reference map, or that
    # call would overwrite what compute_pqk built.
    monkeypatch.undo()
    return seen


def _gate_angles(circuit):
    return [
        (ins.operation.name, [str(p) for p in ins.operation.params])
        for ins in circuit.decompose().data
    ]


def _first_rotation_expr(circuit):
    """The angle expression of the first two-feature (ZZ) phase gate in the circuit."""
    for instruction in circuit.decompose().data:
        if instruction.operation.name in ("p", "rz") and instruction.operation.params:
            expr = instruction.operation.params[0]
            if len(getattr(expr, "parameters", ())) == 2:
                return expr
    raise AssertionError("no two-parameter rz found in the ZZ feature map")


def test_default_is_the_historical_unit_map(monkeypatch, tmp_path):
    seen = _captured_feature_map(monkeypatch, tmp_path)
    assert seen["data_map_func"] is qutils.unit_coefficient_data_map
    old, _ = qutils.get_feature_map(
        feature_map="ZZ", feat_dimension=N_FEATURES, reps=2, entanglement="linear",
        data_map_func=qutils.unit_coefficient_data_map,
    )
    # Compared as gate names and angle expressions: two builds hold distinct (if
    # identically named) ParameterVectors, so the circuits never compare equal.
    assert _gate_angles(seen["circuit"]) == _gate_angles(old)


def test_qiskit_map_is_qiskits_default_and_differs(monkeypatch, tmp_path):
    seen = _captured_feature_map(monkeypatch, tmp_path, data_map="qiskit")
    assert seen["data_map_func"] is None
    unit, _ = qutils.get_feature_map(
        feature_map="ZZ", feat_dimension=N_FEATURES, reps=2, entanglement="linear",
        data_map_func=qutils.unit_coefficient_data_map,
    )
    qiskit_expr = _first_rotation_expr(seen["circuit"])
    unit_expr = _first_rotation_expr(unit)
    assert str(qiskit_expr) != str(unit_expr)
    # The pair gate is P(2 * phi). Qiskit's phi is (pi - x_i)(pi - x_j), so x = (0, 0)
    # gives 2 * pi^2; the unit map's phi is x_i * x_j / 2, which gives 0.
    at_origin = lambda e: float(e.bind({p: 0.0 for p in e.parameters}))  # noqa: E731
    assert at_origin(qiskit_expr) == pytest.approx(2 * np.pi ** 2)
    assert at_origin(unit_expr) == pytest.approx(0.0)


# --- the cache key -----------------------------------------------------------


def test_unit_keeps_the_historical_cache_filename(tmp_path):
    _run(tmp_path)
    fp = _legacy_fingerprint()
    assert _projections(tmp_path) == [
        f"pqk_projection_ds_{fp}_test.npy",
        f"pqk_projection_ds_{fp}_train.npy",
    ]


def test_bool_true_hits_the_unit_cache(tmp_path):
    _run(tmp_path, data_map="unit")
    before = _projections(tmp_path)
    _run(tmp_path, data_map=True)
    assert _projections(tmp_path) == before


def test_unit_and_qiskit_do_not_share_a_cache_file(tmp_path):
    _run(tmp_path, data_map="unit")
    unit_files = set(_projections(tmp_path))
    frame = _run(tmp_path, data_map="qiskit")
    qiskit_files = set(_projections(tmp_path)) - unit_files
    assert len(qiskit_files) == 2, "data_map='qiskit' reused the 'unit' projections"
    unit_train = next(f for f in unit_files if f.endswith("_train.npy"))
    qiskit_train = next(f for f in qiskit_files if f.endswith("_train.npy"))
    a = np.load(os.path.join(tmp_path, unit_train))
    b = np.load(os.path.join(tmp_path, qiskit_train))
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the two data maps produced identical projections"
    assert np.isfinite(b).all()
    # The non-default map is recorded with the results; 'unit' rows are unchanged.
    assert _model_parameters(frame, "pqk")["data_map"] == "qiskit"
    assert "data_map" not in _model_parameters(_run(tmp_path, data_map="unit"), "pqk")


def test_projection_backend_threads_the_data_map(tmp_path):
    """The fast statevector projector must see the same map as the legacy path."""
    _run(tmp_path / "legacy", data_map="qiskit")
    _run(tmp_path / "sv", args_extra={"projection_backend": "statevector"},
         data_map="qiskit")
    legacy = [f for f in _projections(tmp_path / "legacy") if f.endswith("_train.npy")]
    fast = [f for f in _projections(tmp_path / "sv") if f.endswith("_train.npy")]
    np.testing.assert_allclose(
        np.load(tmp_path / "sv" / fast[0]), np.load(tmp_path / "legacy" / legacy[0]),
        atol=1e-9,
    )


# --- tuning -------------------------------------------------------------------


def test_build_search_space_takes_data_map_as_a_categorical():
    space = build_search_space("pqk", {"data_map": ["unit", "qiskit"], "reps": None})
    assert set(space) == {"data_map"}
    assert list(space["data_map"].values) == ["unit", "qiskit"]


def test_compute_pqk_opt_searches_data_map(tmp_path):
    X_train, X_test, y_train, y_test = _dataset(n_train=28, n_test=8)
    frame = compute_pqk_opt(
        X_train, X_test, y_train, y_test, _args(tmp_path),
        data_key="opt", encoding="ZZ", data_map=["unit", "qiskit"], n_trials=2,
    )
    assert _model_parameters(frame, "pqk_opt")["data_map"] in ("unit", "qiskit")
