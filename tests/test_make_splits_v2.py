"""benchmark/make_splits.py (make_splits/2.0) writes manifests QProfiler can trust.

Under ``split_mode: manifest`` QProfiler reads its outer splits, validation rows and all,
from these files instead of splitting itself, so the generator is where the protocol is
decided. The properties pinned here:

  1. **Repeat 0 is v1.** Repeat r is StratifiedKFold(seed + r); repeat 0 must be exactly
     the make_splits/1.0 assignment, so splits/v1 results stay comparable.
  2. **next_fold validation, explicit.** Each fold record carries ``val`` = the test rows
     of fold (f + 1) mod k, and the file reads back through split_manifest.load_manifest.
  3. **Byte-deterministic.** Same inputs, identical bytes (manifests and index.csv), so
     the manifest sha256 recorded per result row identifies the splits.
  4. **Loud failures.** A single-class fit or validation side is an error naming the
     dataset, repeat and fold, and a manifest that does not read back is not left behind.

benchmark/ is not a package: the script is loaded by path.
"""

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import StratifiedKFold

from qbiocode.apps.qprofiler import split_manifest

BENCHMARK = pathlib.Path(__file__).resolve().parents[1] / "benchmark"

pytestmark = pytest.mark.skipif(not (BENCHMARK / "make_splits.py").is_file(),
                                reason="benchmark/make_splits.py not in this checkout")


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location("make_splits_under_test",
                                                  BENCHMARK / "make_splits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _curated(root, dataset_id, n=40, n_pos=13, p=3, seed=0, groups=None, y=None):
    """One dataset in curate.py's layout: <root>/<id>/<id>.csv (label last) + meta.yaml."""
    rng = np.random.default_rng(seed)
    if y is None:
        y = np.zeros(n, dtype=int)
        y[rng.permutation(n)[:n_pos]] = 1
    frame = pd.DataFrame(rng.normal(size=(n, p)).round(6), columns=[f"x{i}" for i in range(p)])
    if groups is not None:
        frame.insert(0, "g", groups)
    frame["label"] = y
    directory = root / dataset_id
    directory.mkdir(parents=True)
    csv_path = directory / f"{dataset_id}.csv"
    frame.to_csv(csv_path, index=False)
    meta = {"dataset_id": dataset_id, "n": n, "p": frame.shape[1] - 1,
            "group_col": "g" if groups is not None else None,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    (directory / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return directory, y


@pytest.fixture
def tree(tmp_path):
    datasets = tmp_path / "datasets"
    _curated(datasets, "toy__a", n=40, n_pos=13, seed=1)
    _curated(datasets, "toy__b", n=23, n_pos=6, seed=2)
    return datasets


def _run(ms, datasets, out, *extra):
    return ms.main(["--datasets", str(datasets), "--out", str(out), "--env-path",
                    str(out.parent), *[str(e) for e in extra]])


def test_schema_and_reader_round_trip(ms, tree, tmp_path):
    out = tmp_path / "splits"
    assert _run(ms, tree, out) == 0
    payload = json.loads((out / "toy__a.json").read_text())
    assert list(payload) == ["schema_version", "dataset_id", "sha256", "n", "k", "n_repeats",
                             "protocol", "validation", "seed", "repeat_seeds", "group_col",
                             "generator_version", "folds"]
    assert payload["schema_version"] == 2 and payload["generator_version"] == "make_splits/2.0"
    assert payload["n_repeats"] == 3 and payload["repeat_seeds"] == [42, 43, 44]
    assert payload["validation"] == "next_fold"
    keys = [(f["repeat"], f["fold"]) for f in payload["folds"]]
    assert keys == [(r, f) for r in range(3) for f in range(5)]
    for record in payload["folds"]:
        assert list(record) == ["repeat", "fold", "train", "val", "test"]
        for side in ("train", "val", "test"):
            assert record[side] == sorted(record[side])
    by_key = {(f["repeat"], f["fold"]): f for f in payload["folds"]}
    for (r, f), record in by_key.items():
        assert record["val"] == by_key[(r, (f + 1) % 5)]["test"]

    csv_path = tree / "toy__a" / "toy__a.csv"
    manifest = split_manifest.load_manifest(split_manifest.manifest_path(out, csv_path))
    y = pd.read_csv(csv_path)["label"].to_numpy()
    split_manifest.verify_dataset(manifest, dataset_file=csv_path, y=y)
    assert manifest.iterations == list(range(1, 16))
    for s in manifest.splits:
        assert s.n_fit + s.n_val + s.n_test == 40


def test_repeat_zero_is_the_v1_assignment(ms, tree, tmp_path):
    out = tmp_path / "splits"
    assert _run(ms, tree, out, "--seed", 7) == 0
    for dataset_id in ("toy__a", "toy__b"):
        y = pd.read_csv(tree / dataset_id / f"{dataset_id}.csv")["label"].to_numpy()
        payload = json.loads((out / f"{dataset_id}.json").read_text())
        for r in range(3):
            # Per-repeat seeds, not one RepeatedStratifiedKFold stream: seed 7 + r.
            expected = StratifiedKFold(5, shuffle=True, random_state=7 + r).split(np.zeros(len(y)), y)
            got = [f for f in payload["folds"] if f["repeat"] == r]
            for record, (train, test) in zip(got, expected):
                assert record["train"] == sorted(train.tolist())
                assert record["test"] == sorted(test.tolist())
        assert payload["folds"][0]["test"] != payload["folds"][5]["test"]


def test_repeat_zero_matches_a_v1_manifest(ms, tree, tmp_path):
    """The make_splits/1.0 file format and repeat 0 agree fold for fold."""
    manifest, _ = ms.split_one(tree / "toy__a", 5, 42, repeats=1)
    v2 = [(f["fold"], f["train"], f["test"]) for f in manifest["folds"]]
    y = pd.read_csv(tree / "toy__a" / "toy__a.csv")["label"].to_numpy()
    v1 = [(i, sorted(tr.tolist()), sorted(te.tolist())) for i, (tr, te) in
          enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(40), y))]
    assert v2 == v1
    assert manifest["n_repeats"] == 1 and manifest["repeat_seeds"] == [42]


def test_output_is_byte_deterministic(ms, tree, tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    assert _run(ms, tree, first) == 0
    assert _run(ms, tree, second) == 0
    for name in ("toy__a.json", "toy__b.json", "index.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_inventory_and_spec(ms, tree, tmp_path):
    out = tmp_path / "splits"
    assert _run(ms, tree, out, "--repeats", 2) == 0
    index = pd.read_csv(out / "index.csv")
    for column in ("repeat", "fold", "iteration", "n_fit", "n_val", "fit_minority",
                   "val_minority", "n_neighbors_max"):
        assert column in index.columns
    assert len(index) == 2 * 5 * 2
    assert (index["n_neighbors_max"] == index["n_fit"] - 1).all()
    assert (index["n_fit"] + index["n_val"] == index["n_train"]).all()
    assert (index["iteration"] == index["repeat"] * 5 + index["fold"] + 1).all()
    assert (index["fit_minority"] >= 1).all() and (index["val_minority"] >= 1).all()

    spec = yaml.safe_load((out / "spec.yaml").read_text())
    assert spec["version"] == "v2" and spec["schema_version"] == 2
    assert spec["n_repeats"] == 2 and spec["validation"] == "next_fold"
    assert spec["n_datasets"] == 2 and spec["n_splits"] == 20
    assert "iter" not in spec
    # toy__b has 6 positives: 1-2 per validation fold, under the report-only floor.
    assert "toy__b" in spec["low_validation_minority"]["datasets"]


def test_default_out_is_v2():
    source = (BENCHMARK / "make_splits.py").read_text()
    assert 'here / "splits" / "v2"' in source


def test_single_class_validation_side_is_an_error(ms, tree, tmp_path, monkeypatch):
    """A splitter handing back a fold without positives must fail, naming the fold."""
    def unstratified(frame, y, k, seed, group_col):
        order = np.argsort(y, kind="stable")  # negatives first: early folds are one-class
        tests = np.array_split(order, k)
        n = len(y)
        return [(np.setdiff1d(np.arange(n), t), t) for t in tests], "StratifiedKFold"

    monkeypatch.setattr(ms, "_repeat_folds", unstratified)
    with pytest.raises(AssertionError, match=r"toy__a repeat 0 fold 0: the \w+ side .* single class"):
        ms.split_one(tree / "toy__a", 5, 42)


def test_balance_tolerance_widens_by_rows():
    """The fit side excludes two folds, so it may sit two rounding steps off."""
    spec = importlib.util.spec_from_file_location("make_splits_tol", BENCHMARK / "make_splits.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    assert ms.balance_tolerance(19, rows=2) == pytest.approx(2 / 19)
    assert ms.balance_tolerance(100) == ms.BALANCE_TOLERANCE
    assert ms.balance_tolerance(10) == pytest.approx(0.1)


def test_manifest_that_does_not_read_back_is_removed(ms, tree, tmp_path, monkeypatch):
    real_dump = ms.dump_manifest

    def lossy(path, payload):
        broken = dict(payload, folds=[dict(f, val=f["val"][:-1]) for f in payload["folds"]])
        real_dump(path, broken)

    monkeypatch.setattr(ms, "dump_manifest", lossy)
    out = tmp_path / "splits"
    assert _run(ms, tree, out) == 1
    assert not (out / "toy__a.json").exists() and not (out / "toy__b.json").exists()


def test_changed_csv_and_bad_arguments_are_refused(ms, tree, tmp_path):
    with pytest.raises(ValueError, match="k >= 3"):
        ms.split_one(tree / "toy__a", 2, 42)
    with pytest.raises(ValueError, match="exceeds minority"):
        ms.split_one(tree / "toy__b", 7, 42)
    csv_path = tree / "toy__a" / "toy__a.csv"
    csv_path.write_text(csv_path.read_text() + "0,0,0,1\n")
    with pytest.raises(ValueError, match="changed since curation"):
        ms.split_one(tree / "toy__a", 5, 42)


def test_group_col_keeps_groups_disjoint_per_repeat(ms, tmp_path):
    # 15 groups of 3, label constant within a group, 6 positive: each of k=3 folds can
    # take exactly 2 positive and 3 negative groups. (The balance check counts rows, so
    # a coarser group structure would need a group-sized tolerance.)
    n = 45
    groups = np.repeat(np.arange(15), 3)
    y = (groups % 5 < 2).astype(int)
    directory, _ = _curated(tmp_path / "datasets", "toy__g", n=n, seed=3, groups=groups, y=y)
    manifest, _ = ms.split_one(directory, 3, 42, repeats=2)
    assert manifest["protocol"] == "StratifiedGroupKFold" and manifest["group_col"] == "g"
    for record in manifest["folds"]:
        fit = np.setdiff1d(record["train"], record["val"])
        g_fit, g_val, g_test = (set(groups[i]) for i in (fit, record["val"], record["test"]))
        assert not (g_fit & g_val) and not (g_fit & g_test) and not (g_val & g_test)
    out = tmp_path / "splits"
    out.mkdir()
    ms.dump_manifest(out / "toy__g.json", manifest)
    ms.verify_written(out / "toy__g.json", directory, manifest)
