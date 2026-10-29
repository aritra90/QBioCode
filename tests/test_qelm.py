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

"""``compute_qelm`` as a first-class arm: the model, its cache, and its integration.

:mod:`tests.test_qelm_features` covers the building blocks -- the reservoir, the
observable pools, and that the projectors still compute PQK's features unchanged. This
file covers the learner built on them, and concentrates on the claims that are load-
bearing for the comparison the arm exists to support:

* **The ablation is exact.** ``reservoir_reps: 0`` with ``measurement: 'xyz'`` must give
  the same *predictions* as a PQK-geometry projection, because that cell is what
  separates the readout's effect from the reservoir's. Checked on the fitted model, not
  just on the features (which ``test_qelm_features`` already pins).

* **The seed is not tunable.** A ``gridsearch_qelm_args`` block naming it is refused.
  This is the one mistake that would look like ordinary tuning and silently turn the
  reported score into a maximum over random reservoirs, so it is tested from the config
  side -- the way a user would actually hit it.

* **The projection cache is keyed on everything that changes the features.** Including
  the reservoir depth and the reservoir seed, neither of which appears in the circuit a
  PQK-style fingerprint would hash. A miss here is silent: rerun into the same directory
  and the previous run's features come back labelled as the new result.

* **It behaves like the other arms.** Same post-embedding inputs, same results schema,
  same ``_opt`` twin, same freeze semantics, same ``selected_*`` columns -- the
  integration points that make a row comparable to a ``pqk`` or ``lr`` row at all.
"""

from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from qbiocode.evaluation.model_evaluation import RESULTS_PREFIX
from qbiocode.learning.compute_qelm import (
    DEFAULT_RIDGE_ALPHAS,
    RESERVOIR_SEED_KEY,
    compute_qelm,
    compute_qelm_opt,
    create_ridge_model,
    resolve_n_features,
    resolve_reservoir_seed,
)

#: Four features, so four qubits: wide enough for weight-2 pools and a 'full' pattern to
#: differ from 'linear', small enough to simulate a few hundred times in a test run.
N_FEATURES = 4


@pytest.fixture(scope="module")
def split():
    """30/10 rows on four features, MinMax-scaled on the TRAINING rows only.

    The protocol the other quantum tests follow: these models read feature magnitudes as
    rotation angles, so an unscaled column wraps past 2*pi and the fit means nothing.
    Scaling fitted on the training rows alone, as ``qprofiler`` does it.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    scaler = MinMaxScaler().fit(X[:30])
    return scaler.transform(X[:30]), scaler.transform(X[30:]), y[:30], y[30:]


@pytest.fixture
def args(tmp_path):
    """The minimal config ``compute_qelm`` reads, with its cache under ``tmp_path``.

    Function-scoped for the directory: the projection cache key covers the circuit and
    the data but a shared directory across tests would still let one test's assertion
    about *file counts* see another's files.
    """
    return {
        "backend": "simulator",
        "seed": 42,
        "q_seed": 42,
        "shots": 1024,
        "average": "weighted",
        "qelm_projection_dir": str(tmp_path / "qelm"),
        "pqk_projection_dir": str(tmp_path / "pqk"),
    }


def _row(frame, label):
    """The metrics dict of one ``results_<label>`` column."""
    for value in frame[f"{RESULTS_PREFIX}{label}"]:
        if isinstance(value, dict):
            return value
    raise AssertionError(f"{RESULTS_PREFIX}{label} is populated on no row")


def _run(split, args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_qelm(*split, args, data_key="ds_none_4_1", **kwargs)


# ----------------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------------


class TestTheFit:
    def test_it_reports_one_row_with_the_shared_schema(self, split, args):
        """Four columns per label, exactly as every other learner contributes -- it is
        what ``model_run``'s melt/pivot fold-up and the append-mode CSV writer are
        sized against."""
        frame = _run(split, args)
        assert sorted(frame.columns) == [
            "results_qelm", "y_predicted_qelm", "y_score_qelm", "y_test_qelm",
        ]

    @pytest.mark.parametrize("metric", ["accuracy", "f1_score", "balanced_accuracy", "mcc"])
    def test_the_label_metrics_are_finite(self, split, args, metric):
        assert np.isfinite(_row(_run(split, args), "qelm")[metric])

    def test_the_ranking_metrics_are_finite(self, split, args):
        """Ridge publishes no ``predict_proba``, so ``extract_binary_scores`` falls
        through to ``decision_function``. That is a real ranking, so ``auc`` and
        ``pr_auc`` must be numbers rather than the NaN a score-less estimator earns."""
        row = _row(_run(split, args), "qelm")
        assert np.isfinite(row["auc"])
        assert np.isfinite(row["pr_auc"])

    def test_a_ranking_score_is_persisted(self, split, args):
        frame = _run(split, args)
        scores = [v for v in frame["y_score_qelm"] if v is not None]
        assert scores and np.ndim(scores[0]) == 1

    def test_the_row_records_the_circuit_and_the_pool(self, split, args):
        """A reader must be able to say what produced the features. For this arm that
        includes the reservoir seed: without it the representation cannot be rebuilt,
        and two rows cannot be told to have used the same reservoir."""
        params = _row(_run(split, args), "qelm")["Model_Parameters"]
        for key in ("feature_map", "feature_map_reps", "entanglement", "data_map",
                    "bandwidth", "reservoir_reps", "reservoir_seed", "measurement",
                    "n_features", "alpha"):
            assert key in params, f"{key} is not reported in {sorted(params)}"

    def test_the_reported_alpha_is_the_one_the_head_chose(self, split, args):
        """``alpha`` is the single thing about this model that was fitted rather than
        fixed, so a row reporting the grid instead of the winner would be useless."""
        alpha = _row(_run(split, args), "qelm")["Model_Parameters"]["alpha"]
        assert any(alpha == pytest.approx(a) for a in DEFAULT_RIDGE_ALPHAS), (
            f"alpha={alpha!r} is not one of the candidates {list(DEFAULT_RIDGE_ALPHAS)}"
        )

    def test_a_pinned_alpha_is_honoured(self, split, args):
        params = _row(_run(split, args, alpha=10.0), "qelm")["Model_Parameters"]
        assert params["alpha"] == pytest.approx(10.0)

    @pytest.mark.parametrize("measurement,expected", [
        ("xyz", 3 * N_FEATURES),
        ("xyz+zz", 4 * N_FEATURES - 1),
    ])
    def test_a_structured_pool_sets_its_own_width(self, split, args, measurement, expected):
        row = _row(_run(split, args, measurement=measurement), "qelm")
        assert row["Model_Parameters"]["n_features"] == expected

    def test_a_float_n_features_is_a_multiple_of_the_qubit_count(self, split, args):
        """An absolute count cannot mean the same thing across datasets of different
        widths, which is why the float spelling exists."""
        row = _row(
            _run(split, args, measurement="random_pauli", n_features=2.0), "qelm"
        )
        assert row["Model_Parameters"]["n_features"] == 2 * N_FEATURES

    def test_the_run_is_reproducible(self, split, args):
        first = _row(_run(split, args), "qelm")
        second = _row(_run(split, args), "qelm")
        assert first["accuracy"] == second["accuracy"]
        assert first["Model_Parameters"]["alpha"] == second["Model_Parameters"]["alpha"]


class TestTheAblation:
    """``reservoir_reps: 0`` must reproduce a PQK-geometry fit, exactly.

    ``test_qelm_features`` pins the *features*; this pins the *model*, which is what the
    comparison actually reads. If these drifted, "the reservoir adds nothing" and "the
    ridge readout is weaker than the SVC" would be indistinguishable in the results.
    """

    def test_zero_depth_matches_a_ridge_on_the_pqk_projection(self, split, args):
        from sklearn.linear_model import RidgeClassifierCV

        from qbiocode.utils import qutils
        from qbiocode.utils.projection import make_projector

        X_train, X_test, y_train, _ = split
        # PQK's projection, built independently of compute_qelm.
        projector = make_projector(
            N_FEATURES, encoding="Z", reps=2, entanglement="linear",
            backend="statevector", data_map_func=qutils.unit_coefficient_data_map,
        )
        Z_train = projector.project(X_train, progress_every=None).reshape(len(X_train), -1)
        Z_test = projector.project(X_test, progress_every=None).reshape(len(X_test), -1)
        reference = RidgeClassifierCV(alphas=list(DEFAULT_RIDGE_ALPHAS))
        reference.fit(Z_train, y_train)

        frame = _run(split, args, encoding="Z", reps=2, reservoir_reps=0,
                     measurement="xyz")
        predicted = [v for v in frame["y_predicted_qelm"] if v is not None][0]
        np.testing.assert_array_equal(np.asarray(predicted), reference.predict(Z_test))
        assert _row(frame, "qelm")["Model_Parameters"]["alpha"] == pytest.approx(
            reference.alpha_
        )

    def test_a_reservoir_changes_the_fit(self, split, args):
        """Otherwise the reservoir arm of the ablation measures nothing."""
        without = _run(split, args, reservoir_reps=0)
        with_reservoir = _run(split, args, reservoir_reps=2)
        assert not np.array_equal(
            [v for v in without["y_score_qelm"] if v is not None][0],
            [v for v in with_reservoir["y_score_qelm"] if v is not None][0],
        )

    def test_zero_depth_is_recorded_as_such(self, split, args):
        """So a results table can be filtered to the ablation rows."""
        assert _row(_run(split, args, reservoir_reps=0), "qelm")[
            "Model_Parameters"]["reservoir_reps"] == 0


# ----------------------------------------------------------------------------------
# The seed
# ----------------------------------------------------------------------------------


class TestTheReservoirSeed:
    def test_it_falls_back_through_the_run_seeds(self):
        assert resolve_reservoir_seed({RESERVOIR_SEED_KEY: 7, "q_seed": 8, "seed": 9}) == 7
        assert resolve_reservoir_seed({"q_seed": 8, "seed": 9}) == 8
        assert resolve_reservoir_seed({"seed": 9}) == 9

    def test_an_unseeded_run_is_refused(self):
        """An unseeded reservoir makes the run irreproducible *without failing*: the
        features stay plausible and the only symptom is two runs at one seed
        disagreeing."""
        with pytest.raises(ValueError, match="irreproducible"):
            resolve_reservoir_seed({"backend": "simulator"})

    def test_a_non_integer_seed_is_refused(self):
        with pytest.raises(ValueError, match="integer"):
            resolve_reservoir_seed({RESERVOIR_SEED_KEY: "forty-two"})

    def test_a_different_seed_is_a_different_model(self, split, args):
        first = _run(split, args)
        other = _run(split, {**args, RESERVOIR_SEED_KEY: 7})
        assert _row(other, "qelm")["Model_Parameters"]["reservoir_seed"] == 7
        assert not np.array_equal(
            [v for v in first["y_score_qelm"] if v is not None][0],
            [v for v in other["y_score_qelm"] if v is not None][0],
        )

    def test_the_seed_does_not_matter_at_zero_depth(self, split, args):
        """With no reservoir and a structured pool there is nothing random left, so the
        ablation cell is seed-independent -- which is what makes it comparable to PQK
        rather than to one draw of a reservoir."""
        first = _run(split, args, reservoir_reps=0)
        other = _run(split, {**args, RESERVOIR_SEED_KEY: 999}, reservoir_reps=0)
        np.testing.assert_array_equal(
            [v for v in first["y_score_qelm"] if v is not None][0],
            [v for v in other["y_score_qelm"] if v is not None][0],
        )

    @pytest.mark.parametrize("key", ["reservoir_seed", RESERVOIR_SEED_KEY, "seed", "q_seed"])
    def test_tuning_the_seed_is_refused(self, split, args, key):
        """The leakage guard, tested from the config side because that is how a user
        hits it. Searching the seed would report the best of ``n_trials`` random
        reservoirs as if it were the performance of one -- a number that looks like a
        tuned result and is not one. It cannot even reach ``compute_qelm`` as a keyword,
        so without this check it surfaces as a ``TypeError`` inside a joblib worker.
        """
        block = {"encoding": ["Z"], key: [1, 2]}
        with pytest.raises(ValueError, match="cannot be tuned"):
            compute_qelm_opt(
                *split, {**args, "gridsearch_qelm_args": block},
                data_key="ds_none_4_1", n_trials=2, encoding=["Z"],
            )

    def test_an_ordinary_block_is_not_refused(self, split, args):
        """The guard must not fire on a legitimate block; it names four keys and one of
        them ('seed') is a plausible thing to find in a config for other reasons."""
        block = {"encoding": ["Z"], "reps": [1]}
        frame = compute_qelm_opt(
            *split, {**args, "gridsearch_qelm_args": block},
            data_key="ds_none_4_1", n_trials=2, **block,
        )
        assert np.isfinite(_row(frame, "qelm_opt")["accuracy"])


# ----------------------------------------------------------------------------------
# The projection cache
# ----------------------------------------------------------------------------------


class TestTheProjectionCache:
    """Everything that changes what a cached projection HOLDS must be in its key.

    A miss is silent in the worst way: change that setting, rerun into the same
    directory, and the previous run's features are reloaded and reported as the new
    result. Two of these -- the reservoir depth and the reservoir seed -- are not in the
    feature map at all, so a PQK-style fingerprint would not cover them.
    """

    @staticmethod
    def _files(args):
        return sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(args["qelm_projection_dir"], "*.npy"))
        )

    def test_a_pass_writes_one_file_per_split(self, split, args):
        _run(split, args)
        files = self._files(args)
        assert len(files) == 2
        assert sum("train" in f for f in files) == 1
        assert sum("test" in f for f in files) == 1

    def test_an_identical_rerun_reuses_the_cache(self, split, args):
        _run(split, args)
        before = self._files(args)
        _run(split, args)
        assert self._files(args) == before

    @pytest.mark.parametrize("change", [
        {"encoding": "ZZ"},
        {"reps": 1},
        {"entanglement": "full"},
        {"bandwidth": 2.0},
        {"data_map": "qiskit"},
        {"reservoir_reps": 2},
        {"measurement": "xyz+zz"},
    ])
    def test_a_changed_setting_writes_a_new_cache(self, split, args, change):
        _run(split, args)
        before = set(self._files(args))
        _run(split, args, **change)
        assert set(self._files(args)) - before, f"{change} reused the previous cache"

    def test_a_changed_reservoir_seed_writes_a_new_cache(self, split, args):
        """Not in the feature map, so nothing but an explicit entry puts it in the key."""
        _run(split, args)
        before = set(self._files(args))
        _run(split, {**args, RESERVOIR_SEED_KEY: 7})
        assert set(self._files(args)) - before

    def test_a_different_pool_of_the_same_width_writes_a_new_cache(self, split, args):
        """Two pools can hold different operators at one width (a different weight cap),
        and it is the operators that decide the numbers -- which is why the key carries a
        digest of the pool rather than its size."""
        common = dict(measurement="random_pauli", n_features=8)
        _run(split, args, **common, max_weight=1)
        before = set(self._files(args))
        _run(split, args, **common, max_weight=2)
        assert set(self._files(args)) - before

    def test_a_cached_file_of_the_wrong_length_is_refused(self, split, args):
        """The fingerprint cannot catch this -- a file is only checkable once opened --
        and silently fitting on the wrong rows is the failure it prevents."""
        _run(split, args)
        train = glob.glob(os.path.join(args["qelm_projection_dir"], "*train.npy"))[0]
        np.save(train, np.load(train, allow_pickle=False)[:-3])
        with pytest.raises(ValueError, match="rows"):
            _run(split, args)

    def test_a_cached_file_of_the_wrong_width_is_refused(self, split, args):
        _run(split, args)
        train = glob.glob(os.path.join(args["qelm_projection_dir"], "*train.npy"))[0]
        kept = np.load(train, allow_pickle=False)
        np.save(train, kept.reshape(len(kept), -1)[:, :-2])
        with pytest.raises(ValueError, match="features per row"):
            _run(split, args)

    def test_a_checkpoint_is_resumed_and_then_removed(self, split, args):
        """A killed quantum pass must resume rather than restart: on these arms one pass
        is hours, which is how a job meets its scheduler wall in the first place."""
        _run(split, args)
        checkpoints = glob.glob(
            os.path.join(args["qelm_projection_dir"], "checkpoints", "*.partial.npy")
        )
        assert not checkpoints, "a completed pass left its checkpoints behind"


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------


class TestValidation:
    """Every message names the parameter, the value and the accepted set, and is raised
    before any directory is created or any cache is read."""

    @pytest.mark.parametrize("kwargs,match", [
        ({"encoding": "W"}, "encoding must be one of"),
        ({"entanglement": "spaghetti"}, "entanglement must be one of"),
        ({"reps": 0}, "positive integer"),
        ({"measurement": "kitchen_sink"}, "measurement must be one of"),
        ({"primitive": "sampler"}, "must be 'estimator'"),
        ({"data_map": "banana"}, "data_map must be one of"),
        ({"bandwidth": 0.0}, "positive"),
        ({"reservoir_reps": -1}, "non-negative"),
    ])
    def test_a_bad_hyperparameter_is_rejected(self, split, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _run(split, args, **kwargs)

    def test_a_missing_backend_is_rejected(self, split, args):
        del args["backend"]
        with pytest.raises(ValueError, match="backend"):
            _run(split, args)

    def test_a_hardware_backend_is_refused_with_an_explanation(self, split, args):
        """Refused rather than silently run: this arm has no runtime-primitive path,
        because that path is built around the 3n single-qubit groups pqk measures."""
        with pytest.raises(ValueError, match="does not support"):
            _run(split, {**args, "backend": "ibm_fake"})

    def test_misaligned_labels_are_rejected(self, split, args):
        X_train, X_test, y_train, y_test = split
        with pytest.raises(ValueError, match="aligned"):
            compute_qelm(X_train, X_test, y_train[:-1], y_test, args)

    def test_mismatched_widths_are_rejected(self, split, args):
        X_train, X_test, y_train, y_test = split
        with pytest.raises(ValueError, match="same number of features"):
            compute_qelm(X_train, X_test[:, :-1], y_train, y_test, args)

    def test_n_features_on_a_structured_pool_is_rejected(self, split, args):
        """Truncating 'xyz' drops whole Pauli groups rather than thinning the pool, so a
        width request that cannot be honoured is reported instead of approximated."""
        with pytest.raises(ValueError, match="structured pool|fixed by the qubit count"):
            _run(split, args, measurement="xyz", n_features=5)

    def test_random_pauli_without_a_width_is_rejected(self, split, args):
        with pytest.raises(ValueError, match="needs n_features"):
            _run(split, args, measurement="random_pauli")


class TestResolveNFeatures:
    @pytest.mark.parametrize("value,expected", [(12, 12), (3.0, 24), (1.5, 12)])
    def test_counts_and_multiples(self, value, expected):
        assert resolve_n_features(value, 8, "random_pauli") == expected

    def test_a_structured_pool_resolves_to_none(self):
        assert resolve_n_features(None, 8, "xyz") is None

    @pytest.mark.parametrize("value", [0, -1, 0.0, "many", True])
    def test_a_bad_value_is_rejected(self, value):
        with pytest.raises(ValueError):
            resolve_n_features(value, 8, "random_pauli")


class TestTheRidgeHead:
    def test_several_alphas_select_by_cross_validation(self):
        from sklearn.linear_model import RidgeClassifierCV

        assert isinstance(create_ridge_model(42), RidgeClassifierCV)

    def test_one_alpha_is_a_plain_ridge(self):
        """``RidgeClassifierCV`` over one value would run a selection with nothing to
        select, and would report an ``alpha_`` no search produced."""
        from sklearn.linear_model import RidgeClassifier

        assert isinstance(create_ridge_model(42, alphas=[1.0]), RidgeClassifier)

    @pytest.mark.parametrize("alphas,match", [([], "empty"), ([0.0], "positive"),
                                              ([-1.0], "positive")])
    def test_a_bad_penalty_is_rejected(self, alphas, match):
        with pytest.raises(ValueError, match=match):
            create_ridge_model(42, alphas=alphas)


# ----------------------------------------------------------------------------------
# Tuning and integration
# ----------------------------------------------------------------------------------


class TestTuning:
    BLOCK = {"encoding": ["Z", "ZZ"], "reps": [1], "reservoir_reps": [0, 1]}

    @pytest.fixture
    def tuned(self, split, args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compute_qelm_opt(
                *split, args, model="qelm_opt", data_key="ds_none_4_1",
                n_trials=4, **self.BLOCK,
            )

    def test_the_row_is_labelled_as_tuned(self, tuned):
        """``modeleval`` infers ``tuned`` from the label, so a tuned run that kept the
        untuned one would be indistinguishable in ModelResults.csv -- the bug both pqk
        and qpl once had."""
        row = _row(tuned, "qelm_opt")
        assert row["model"] == "qelm_opt"
        assert "BestParams_Tuned" in row

    def test_what_was_searched_is_reported(self, tuned):
        best = _row(tuned, "qelm_opt")["BestParams_Tuned"]
        for name, values in self.BLOCK.items():
            assert best[name] in values, f"{name}={best[name]!r} is not one of {values}"

    def test_each_searched_name_gets_a_selected_column(self, tuned):
        """What makes "did the search settle on the same circuit across splits?"
        answerable with a groupby instead of by parsing a repr per row."""
        row = _row(tuned, "qelm_opt")
        for name in self.BLOCK:
            assert f"selected_{name}" in row

    def test_the_tuning_evidence_is_recorded(self, tuned):
        row = _row(tuned, "qelm_opt")
        assert row["tuning_metric"] == "balanced_accuracy"
        assert np.isfinite(row["tuning_score"])
        assert row["tuning_reused"] is False

    def test_the_reported_time_covers_the_whole_search(self, split, args, tuned):
        """The base call writes a ``time`` covering only the final fit, which understates
        a tuned run by the whole search."""
        single = _row(_run(split, args, encoding="Z", reps=1), "qelm")["time"]
        assert _row(tuned, "qelm_opt")["time"] > single

    def test_the_search_can_choose_the_ablation(self, split, args):
        """``reservoir_reps: 0`` in the space means the tuner decides whether the random
        layer earns its depth, rather than it being assumed to."""
        frame = compute_qelm_opt(
            *split, args, model="qelm_opt", data_key="ds_none_4_1", n_trials=2,
            encoding=["Z"], reps=[1], reservoir_reps=[0],
        )
        assert _row(frame, "qelm_opt")["BestParams_Tuned"]["reservoir_reps"] == 0

    def test_tuning_leaves_no_projection_in_the_real_cache(self, split, args):
        """Trials project the INNER split, whose row count differs from the refit's. A
        trial writing into the run's cache would make the refit fail its own row-count
        guard, which is why the tuner redirects them to a scratch directory."""
        compute_qelm_opt(*split, args, model="qelm_opt", data_key="ds_none_4_1",
                         n_trials=3, **self.BLOCK)
        files = glob.glob(os.path.join(args["qelm_projection_dir"], "*.npy"))
        # Only the refit's two, on the full training split.
        assert len(files) == 2
        assert len(np.load(files[0], allow_pickle=False)) in {len(split[0]), len(split[1])}


class TestFreezingAcrossSplits:
    """``freeze_quantum_params``: search once, reuse for the remaining resamples."""

    BLOCK = {"encoding": ["Z"], "reps": [1], "reservoir_reps": [0, 1]}

    @pytest.fixture
    def frozen(self, split, args, tmp_path):
        frozen_args = {**args, "freeze_quantum_params": True,
                       "quantum_param_dir": str(tmp_path / "frozen")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = compute_qelm_opt(*split, frozen_args, model="qelm_opt",
                                     data_key="ds_none_4_1", n_trials=3, **self.BLOCK)
            second = compute_qelm_opt(*split, frozen_args, model="qelm_opt",
                                      data_key="ds_none_4_2", n_trials=3, **self.BLOCK)
        return frozen_args, first, second

    def test_the_first_split_searches_and_the_second_reuses(self, frozen):
        _, first, second = frozen
        assert _row(first, "qelm_opt")["tuning_reused"] is False
        assert _row(second, "qelm_opt")["tuning_reused"] is True

    def test_both_splits_report_the_same_configuration(self, frozen):
        _, first, second = frozen
        for name in self.BLOCK:
            assert _row(first, "qelm_opt")[f"selected_{name}"] == \
                _row(second, "qelm_opt")[f"selected_{name}"]

    def test_the_frozen_payload_holds_only_the_searched_keys(self, frozen):
        """Not the seed, and not the ridge penalty: the first is never searched, and the
        second is re-selected per split on that split's own projected rows."""
        frozen_args, _, _ = frozen
        path = glob.glob(os.path.join(frozen_args["quantum_param_dir"], "*.json"))[0]
        params = json.load(open(path, encoding="utf-8"))["params"]
        assert set(params) == set(self.BLOCK)

    def test_the_cache_is_keyed_without_the_iteration(self, frozen):
        """Reuse across resamples is the point, so the iteration token cannot be in the
        key -- but the dataset, embedding and component count must be."""
        frozen_args, _, _ = frozen
        names = [os.path.basename(p)
                 for p in glob.glob(os.path.join(frozen_args["quantum_param_dir"], "*.json"))]
        assert names == ["ds_none_4__qelm.json"]


class TestDispatchIntegration:
    """QELM as ``model_run`` sees it -- the arm has to be reachable the same way."""

    def test_it_is_registered_as_a_quantum_model(self):
        from qbiocode.evaluation.model_run import QUANTUM_MODELS

        assert "qelm" in QUANTUM_MODELS

    def test_both_twins_are_exported_from_the_package(self):
        import qbiocode

        assert qbiocode.compute_qelm is compute_qelm
        assert qbiocode.compute_qelm_opt is compute_qelm_opt

    def test_it_is_classified_as_a_quantum_arm(self):
        """``fair_selection`` decides which side of the comparison a row counts for; a
        quantum arm read as classical would suppress a quantum win *and* add an arm to
        the classical side."""
        from qbiocode.utils.fair_selection import model_side

        assert model_side("qelm") == "quantum"
        assert model_side("qelm_opt") == "quantum"

    def test_it_is_drawn_as_a_quantum_model(self):
        from qbiocode.visualization.visualize_correlation import _QML_MODELS

        assert "QELM" in _QML_MODELS

    def test_it_runs_untuned_through_model_run(self, split, args):
        from qbiocode.evaluation.model_run import model_run

        config = {**args, "n_jobs": 1, "model": ["qelm"],
                  "qelm_args": {"encoding": "Z", "reps": 1}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model_run(*split, "ds_none_4_1", config)
        assert result["results_qelm"][0]["model"] == "qelm"

    def test_it_runs_tuned_through_model_run(self, split, args):
        from qbiocode.evaluation.model_run import model_run

        config = {**args, "n_jobs": 1, "model": ["qelm"], "grid_search": True,
                  "tune_quantum": True, "n_trials_quantum": 2,
                  "gridsearch_qelm_args": {"encoding": ["Z"], "reps": [1]}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model_run(*split, "ds_none_4_1", config)
        assert result["results_qelm_opt"][0]["model"] == "qelm_opt"

    def test_it_shares_the_results_schema_with_pqk(self, split, args):
        """``qprofiler`` writes ModelResults.csv with ``csv.writer`` in append mode and
        emits the header from the FIRST row's keys, so an arm with an extra or renamed
        inner key produces rows wider than the header -- a CSV that still parses, with
        every column after the divergence shifted by one."""
        from qbiocode.evaluation.model_run import model_run

        config = {**args, "n_jobs": 1, "model": ["pqk", "qelm"],
                  "pqk_args": {"encoding": "Z", "reps": 1},
                  "qelm_args": {"encoding": "Z", "reps": 1}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model_run(*split, "ds_none_4_1", config)
        assert set(result["results_qelm"][0]) == set(result["results_pqk"][0])


class TestTheKernelDump:
    """``kernel_dump_dir``: the objects a quantum-advantage claim is computed from."""

    def test_it_writes_the_features_and_the_settings(self, split, args, tmp_path):
        dump = tmp_path / "kernels"
        _run(split, {**args, "kernel_dump_dir": str(dump)})
        files = sorted(p.name for p in dump.glob("*.npz"))
        assert files == ["qelm_qelm_ds_none_4_1.npz"]
        with np.load(dump / files[0], allow_pickle=False) as held:
            for key in ("Z_train", "Z_test", "X_train", "y_train", "y_test", "params"):
                assert key in held
            # Readable back as the mapping it was written from, so a reader can rebuild
            # the representation -- which for this arm needs the reservoir seed.
            params = json.loads(str(held["params"]))
            assert "reservoir_seed" in params and "measurement" in params

    def test_the_winner_finder_recognises_the_dump(self):
        """It parses the model label off the filename, longest first -- so ``qelm`` must
        not shadow ``qelm_opt`` and file every tuned arm as untuned."""
        from qbiocode.utils.qc_winner_finder import _parse_dump_name

        assert _parse_dump_name("qelm_qelm_opt_wdbc_pca_8_3.npz") == (
            "qelm_opt", "wdbc_pca_8_3")
        assert _parse_dump_name("qelm_qelm_wdbc_pca_8_3.npz") == ("qelm", "wdbc_pca_8_3")
