"""The contract between a shipped config block and the function it is splatted into.

Four defects made every one of the 12 pilot configs fit nothing at all, and the whole
suite passed while they did. They share one shape: ``model_run`` splats a config block as
``**kwargs`` into a compute function, so a block and a signature that disagree is not a
config warning -- it is a ``TypeError`` raised inside a joblib worker. Because the
``delayed()`` list is built *before* ``Parallel`` starts, that exception takes down all 13
models having fit nothing, not just the one model whose key was wrong.

Nothing in the suite compared a block against the signature it reaches, so all four were
found by running qprofiler and reading the traceback. These tests are that comparison, done
statically over the shipped configs, so the next mismatch costs a second rather than a
24-hour LSF job that dies in its first minute.

The four:
  1. ``embeddings: ['none']`` with an ``n_components`` meant for the wide datasets raised
     ``ValueError: n_components=8 exceeds the 4 features in X_train`` before any fit --
     ``none`` is a pass-through and never produces components, so the guard must skip it.
  2. ``verbose`` in ``catboost_args`` collided with the ``verbose=False`` that ``model_run``
     already passes: ``TypeError: _call_with_global_seeds() got multiple values for keyword
     argument 'verbose'``.
  3. ``'thread_count': [1]`` is a *pass-through*, not a search dimension, so the list itself
     reached the estimator: ``RuntimeError: Cannot clone object CatBoostClassifier(...,
     thread_count=[1], ...)`` from sklearn's clone, inside cross-validation.
  4. ``gridsearch_dt_args`` named ``splitter``, which ``compute_dt`` accepts and
     ``compute_dt_opt`` did not: ``TypeError: compute_dt_opt() got an unexpected keyword
     argument 'splitter'``. With ``grid_search: True`` the tuned twin is the only one that
     runs, so this was unconditional.
  5. No xgb block pinned a thread cap, and ``compute_xgb`` had no argument to pin it with.
     Unset, XGBoost sizes its OpenMP pool from ``omp_get_max_threads()``: on a 128-core node
     one fit over a 42-row training set ran for more than 280 s and was killed, and took
     0.06 s once capped. This one does not raise, which is why it outlived the other four --
     it presents as a hang, and only on a machine with enough cores to hang on. Writing the
     obvious pin was itself the fifth defect: ``'nthread': 1`` gave ``TypeError:
     compute_xgb_opt() got an unexpected keyword argument 'nthread'``, so the setting that
     would have prevented the hang was unrepresentable in a config until the argument
     existed. ``TestEveryFitPinsItsThreadCap`` keeps the pins present; test 1 above keeps
     them spellable.

The sixth is a different shape, and worth separating rather than filing under the frame
above: the key was right and the signature accepted it. What was wrong was the value's
*type*. OmegaConf hands a YAML list over as a ``ListConfig``, which satisfies every
``Sequence`` test in ``build_search_space`` and whose scalar leaves read back as plain
``int`` and ``str`` -- so a list of scalars is indistinguishable from a plain one in
practice. A list *of lists* is not: ``gridsearch_mlp_args`` writes
``hidden_layer_sizes: [[20], [50], [100]]``, ``list(values)`` copied only the outer node,
and each choice stayed a ``ListConfig``. Optuna put the chosen value through
``json.dumps`` and raised ``TypeError: Object of type ListConfig is not JSON
serializable``.

That one is not hypothetical and it is the reason this section exists: it killed a real
submission of all 12 LSF jobs, every one exiting in its first minute having fit nothing,
while all 53 tests in this file passed. They passed because none of them built a search
space from a real OmegaConf object and then ran the tuner -- they compared config keys
against signatures, which is what the first five defects needed and what this one slips
straight past. Two further properties made it good at hiding, and both are now encoded in
``TestConfigValuesReachTheTunerAsPlainPython``:

  * It is version-dependent. The ``json.dumps`` is Optuna's constant-liar bookkeeping,
    and ``TPESampler`` only began defaulting ``constant_liar=True`` at Optuna 5. The same
    configs ran on Optuna 4, so the bug arrived with an environment change and no config
    or code change at all.
  * It is late. The sampler draws its first ``n_startup_trials`` -- 10 by default --
    independently at random, and only then consults the path that serialises. So the
    traceback names trial 10 of a 50-trial search, and any smoke test short enough to be
    comfortable in CI finishes before reaching it.
"""

import ast
import importlib
import inspect
import json
import pathlib
import re
import textwrap

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

import qbiocode
from qbiocode.apps.qprofiler.qprofiler import _resolve_model_lists
from qbiocode.learning._grid import one_value, to_plain
from qbiocode.learning.compute_tabpfn import (
    TABPFN_DEFAULT_VERSION,
    _RESTRICTED_VERSIONS,
    tabpfn_versions_requiring_token,
)
from qbiocode.learning._tuning import _Categorical, build_search_space, run_study

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "pilot10" / "configs"

#: Arguments ``model_run`` passes explicitly at every call site. A block naming one of
#: these duplicates it in the same namespace, which is defect 2.
RESERVED = ("model", "data_key", "n_trials", "validation_split", "cv", "tuner", "verbose")


def _configs():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    paths = sorted(CONFIG_DIR.glob("pilot*.yaml"))
    if not paths:
        pytest.skip(f"no pilot configs at {CONFIG_DIR}")
    return [(p.name, OmegaConf.load(p)) for p in paths]


def _compute_for(block):
    """``x_args`` -> ``compute_x``; ``gridsearch_x_args`` -> ``compute_x_opt``."""
    if block.startswith("gridsearch_"):
        stem = block[len("gridsearch_") : -len("_args")]
        return f"compute_{stem}_opt"
    return f"compute_{block[: -len('_args')]}"


def _searched(fn):
    """The keys the ``_opt`` twin actually searches, read off its ``candidates`` dict.

    Anything else in the block is a pass-through: it is handed to every trial's estimator
    verbatim, so a list there is defect 3 rather than a search space.
    """
    match = re.search(r"candidates\s*=\s*\{(.*?)\n    \}", inspect.getsource(fn), re.S)
    return set(re.findall(r'"([^"]+)":', match.group(1))) if match else None


class TestTheBlocksMatchTheSignaturesTheyReach:
    def test_every_args_key_is_accepted_by_the_compute_function_it_reaches(self):
        """Defect 4. The twins deliberately expose different *search* surfaces -- the
        ``_opt`` side is narrower on purpose, since searching fifteen MLP parameters is not
        a benchmark -- so this does not demand symmetry. It demands only that a key someone
        actually wrote is a key the function it lands in can take.
        """
        bad = []
        for name, cfg in _configs():
            for block in (k for k in cfg.keys() if k.endswith("_args")):
                fn = getattr(qbiocode, _compute_for(block), None)
                if fn is None:
                    bad.append(f"{name}: {block} -> no {_compute_for(block)} in qbiocode")
                    continue
                accepted = set(inspect.signature(fn).parameters)
                for key in cfg[block].keys():
                    if key not in accepted:
                        bad.append(f"{name}: {block}[{key!r}] -> {fn.__name__}() rejects it")
        assert not bad, (
            "a config key would reach a function that cannot take it, which is a TypeError "
            "inside a joblib worker that kills every model in the run:\n  "
            + "\n  ".join(sorted(set(bad))[:20])
        )

    def test_no_block_duplicates_an_argument_model_run_passes_explicitly(self):
        """Defect 2. ``verbose`` is the live example and a real CatBoost parameter, which is
        why it looked harmless: CatBoost's ``verbose`` is training chatter, while QBioCode's
        selects the result summary. Two different things, one namespace.
        """
        bad = [
            f"{name}: {block}[{key!r}]"
            for name, cfg in _configs()
            for block in (k for k in cfg.keys() if k.endswith("_args"))
            for key in cfg[block].keys()
            if key in RESERVED
        ]
        assert not bad, (
            "model_run passes these explicitly, so a block naming one raises TypeError "
            "while the delayed() list is being built -- before any model runs:\n  "
            + "\n  ".join(sorted(bad))
        )

    def test_a_pass_through_parameter_is_never_given_several_values(self):
        """Defect 3. A one-element list is fine and idiomatic -- every *searched* key in a
        ``gridsearch_*`` block is a list, so ``[1]`` is how a constant gets written there,
        and the compute function unwraps it. Several values mean the author believed the
        parameter was searched when it is not, and that is what must not ship.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        bad = []
        for name, cfg in _configs():
            for block in (k for k in cfg.keys() if k.startswith("gridsearch_")):
                fn = getattr(qbiocode, _compute_for(block), None)
                if fn is None:
                    continue
                searched = _searched(fn)
                if searched is None:
                    continue
                for key, raw in cfg[block].items():
                    value = OmegaConf.to_object(raw) if hasattr(raw, "_content") else raw
                    if key not in searched and isinstance(value, list) and len(value) != 1:
                        bad.append(f"{name}: {block}[{key!r}] = {value!r} is not searched")
        assert not bad, (
            "these are handed to the estimator verbatim, so a multi-value list becomes a "
            "sklearn clone RuntimeError during cross-validation:\n  " + "\n  ".join(bad)
        )


class TestNoneIsExemptFromTheComponentGuard:
    """Defect 1, which needs no config to reproduce -- only a narrow dataset."""

    def test_none_passes_data_through_when_n_components_exceeds_the_width(self):
        """``none`` returns X unchanged and never produces components, so ``n_components``
        does not apply to it. The pilot sets 8 for the wide band and pairs it with
        ``embeddings: ['none']`` everywhere, which killed the 4-, 6- and 7-feature datasets.
        """
        rng = np.random.RandomState(0)
        X_train, X_test = rng.rand(20, 4), rng.rand(8, 4)
        out_train, out_test = qbiocode.get_embeddings(
            "none", X_train, X_test, n_components=8
        )
        assert np.array_equal(out_train, X_train), "'none' must return X_train unchanged"
        assert np.array_equal(out_test, X_test), "'none' must return X_test unchanged"

    def test_a_real_embedding_still_refuses_more_components_than_features(self):
        """The guard is load-bearing for everything that does produce components; the
        exemption must not have turned it off in general.
        """
        rng = np.random.RandomState(0)
        with pytest.raises(ValueError, match=r"n_components"):
            qbiocode.get_embeddings("pca", rng.rand(20, 4), rng.rand(8, 4), n_components=8)


#: Models whose fits take every core unless a config says otherwise, and the block keys
#: that say so. XGBoost answers to either spelling of its cap; CatBoost's is its own name.
#: See defect 5 -- these are the pins whose absence is a hang rather than an exception.
THREAD_CAPS = {"xgb": ("n_jobs", "nthread"), "catboost": ("thread_count",)}


class _Log:
    """Minimal logger stand-in; ``_resolve_model_lists`` only ever calls .info()."""

    def info(self, msg, *a):
        pass


def _models(config):
    """The model list the run will actually assemble, not the one the file spells.

    A pilot config names ``classical_model`` and ``quantum_model`` and no ``model`` at all
    -- qprofiler joins them. Reading ``config['model']`` here therefore returns None for
    every shipped config, which is not an error anywhere: a test that iterates it simply
    checks nothing and passes. That is how the first draft of this class passed while the
    pins it was written to require were absent from one of the two blocks. So the join is
    done with the same function the run uses.
    """
    # OmegaConf is reached the way `_configs` reaches it. `_resolve_model_lists` is not:
    # tests/test_suite_hygiene.py forbids importorskip on a first-party module, because an
    # ImportError in one of ours must fail the suite rather than switch it off.
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    # resolve=False: the model lists are literal, and resolving the whole tree raises
    # `UnsupportedInterpolationType: now` on hydra.run.dir -- `${now:...}` is a resolver
    # hydra registers at runtime and nothing outside it can expand.
    resolved = OmegaConf.to_container(config, resolve=False)
    _resolve_model_lists(resolved, _Log())
    return resolved.get("model") or []


def _cap(block, keys, block_name):
    """The cap this block sets, read exactly as the compute function will read it.

    Unwrapping is delegated to :func:`qbiocode.learning._grid.one_value` rather than
    repeated here, for two reasons. It is what the production path calls, so this test
    cannot disagree with the run about what a block means. And a hand-rolled
    ``isinstance(value, (list, tuple))`` is wrong in this file: OmegaConf yields
    ``ListConfig``, which is a ``Sequence`` but neither a list nor a tuple, so the
    one-element list a search block spells its constant as came back unwrapped and
    ``[1] == 1`` failed for all 12 configs.

    Returns None when the block names none of ``keys`` -- the failure this class is about.
    """
    for key in keys:
        if key in block:
            return one_value(key, block[key], "it caps threads rather than tuning", block_name)
    return None


class TestEveryFitPinsItsThreadCap:
    """Defect 5: an unpinned fit is a hang, and it only hangs where the cores are.

    ``submit_pilot.sh`` exports ``OMP_NUM_THREADS=1``, which bounds XGBoost -- but not
    CatBoost, which runs its own thread pool, and not a ``qprofiler`` run launched by hand
    without that export. So the pin belongs in the config as well as the environment, and
    these tests are about the config.
    """

    @pytest.mark.parametrize("name,config", _configs())
    def test_the_capped_models_are_actually_in_the_run(self, name, config):
        """Guard the guard: the tests below are vacuous if the join returns nothing.

        Both tests below iterate a model list, and an empty list makes both pass without
        asserting anything. This states the precondition separately so that failure is
        reported as "the list is empty" rather than as silence.
        """
        models = _models(config)
        assert models, f"{name}: no model list could be resolved from this config."
        assert set(THREAD_CAPS) <= set(models), (
            f"{name}: resolves to {sorted(models)}, which is missing "
            f"{sorted(set(THREAD_CAPS) - set(models))}. Either the pilot dropped a model "
            f"whose threads need capping, or THREAD_CAPS names one that no longer runs."
        )

    @pytest.mark.parametrize("name,config", _configs())
    def test_the_block_that_grid_search_makes_live_pins_the_cap(self, name, config):
        live = "gridsearch_{}_args" if config.get("grid_search") else "{}_args"
        models = _models(config)
        for model, keys in THREAD_CAPS.items():
            if model not in models:
                continue
            block_name = live.format(model)
            block = config.get(block_name)
            assert block is not None, (
                f"{name}: model {model!r} runs but {block_name!r} is absent, so there is "
                f"nowhere for its thread cap to be pinned."
            )
            assert _cap(block, keys, block_name) == 1, (
                f"{name}: {block_name!r} does not pin any of {keys} to 1. Unset, {model} "
                f"takes every core on the node, and with n_jobs workers each running one "
                f"model that is n_jobs x ncores threads. This does not raise -- it hangs."
            )

    @pytest.mark.parametrize("name,config", _configs())
    def test_both_twins_pin_it_so_flipping_grid_search_stays_safe(self, name, config):
        """``grid_search`` selects which block is read; neither may be the unpinned one.

        It replaces the untuned path rather than adding to it, so a pin in only one block is
        correct only for the current value of one boolean. That is how defect 3 shipped --
        ``thread_count`` pinned where the run was not reading it.
        """
        models = _models(config)
        for model, keys in THREAD_CAPS.items():
            if model not in models:
                continue
            for block_name in (f"{model}_args", f"gridsearch_{model}_args"):
                block = config.get(block_name)
                if block is None:
                    continue
                assert _cap(block, keys, block_name) == 1, (
                    f"{name}: {block_name!r} exists but pins none of {keys} to 1. Whichever "
                    f"way 'grid_search' is set, the block that is read must cap its threads."
                )


class TestTunedQuantumReadsWhatTheConfigSays:
    """The tuned quantum path builds kwargs from ``gridsearch_<model>_args`` alone.

    ``_QUANTUM_PASSTHROUGH`` in ``model_run`` carries exactly one key -- qpl's
    ``classical_models`` -- so under ``grid_search`` and ``tune_quantum`` the
    ``<model>_args`` block is never read for the other quantum arms. A setting that lives
    only in ``<model>_args`` therefore falls back to the compute function's own default on
    the tuned path, silently, with no error and nothing in the log to say so.

    This does not demand that the gridsearch block restate everything -- for qsvc, pqk and
    vqc the function default already equals the configured value, so omitting it is
    correct. It demands that the EFFECTIVE value match. qnn is where that bites: qnn_args
    asks for the exact ``estimator`` while ``compute_qnn`` defaults to the shot-noisy
    ``sampler``, so without an explicit pin the tuned run uses a different primitive than
    the config documents. Verified against the real code path: with the pin ``compute_qnn``
    receives ``primitive='estimator'``, without it the sampler default at the configured
    shot count.
    """

    @pytest.mark.parametrize("name,config", _configs())
    def test_effective_tuned_value_matches_the_untuned_block(self, name, config):
        for model in ("qsvc", "pqk", "qnn", "vqc"):
            if model not in _models(config):
                continue
            untuned = config.get(f"{model}_args") or {}
            tuned = config.get(f"gridsearch_{model}_args") or {}
            fn = getattr(qbiocode, f"compute_{model}")
            defaults = {
                k: v.default for k, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            for key, want in untuned.items():
                if key not in defaults:
                    continue
                if key in tuned:
                    try:
                        got = one_value(key, tuned[key],
                                        "it is compared against the untuned block",
                                        f"gridsearch_{model}_args")
                    except ValueError:
                        continue    # several values: genuinely searched, nothing to match
                    source = f"gridsearch_{model}_args"
                else:
                    got = defaults[key]
                    source = f"compute_{model}'s own default"
                assert got == want, (
                    f"{name}: {model}_args sets {key}={want!r}, but the tuned path takes "
                    f"{key}={got!r} from {source}. The tuned path never reads "
                    f"{model}_args, so pin {key} in gridsearch_{model}_args."
                )


class _Stub(BaseEstimator, ClassifierMixin):
    """A clonable estimator that accepts the synthetic search space and does no work.

    ``run_study`` scores every trial with ``cross_val_score``, so the integration test
    below would otherwise pay for a real fit per trial at a trial count chosen to exceed
    Optuna's random startup phase. What is under test is the plumbing between the config
    and the sampler, not any estimator, so the fit is a no-op.

    The arguments are named explicitly rather than taken as ``**kwargs`` because sklearn's
    ``clone`` reads them off ``__init__``'s signature via ``get_params``; a ``**kwargs``
    estimator is not clonable and ``cross_val_score`` would fail for a reason that has
    nothing to do with this test.
    """

    def __init__(self, layers=None, mode="a", alpha=0.1):
        self.layers = layers
        self.mode = mode
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class TestConfigValuesReachTheTunerAsPlainPython:
    """Defect 6: the config key was right, the signature accepted it, the *type* leaked.

    See the module docstring. These three tests are layered deliberately, because the
    defect had two independent hiding places -- the wrapper type is invisible for scalar
    leaves, and the failing sampler path is not reached until trial 10 -- and no single
    test covers both cheaply.
    """

    def test_to_plain_unwraps_omegaconf_containers_at_every_depth(self):
        """The unit-level claim, stated against the type that actually caused it.

        Asserting ``type(...) is list`` rather than ``isinstance`` is the whole point:
        ``ListConfig`` passes ``isinstance(x, Sequence)`` and compares equal to the list it
        wraps, so every weaker assertion here would have passed before the fix.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        raw = OmegaConf.create(
            {"hidden_layer_sizes": [[20], [50, 50]], "alpha": {"low": 0.01, "high": 1.0}}
        )
        plain = to_plain(raw)

        assert type(plain) is dict
        assert type(plain["hidden_layer_sizes"]) is list
        assert [type(v) for v in plain["hidden_layer_sizes"]] == [list, list], (
            "the elements are what Optuna serialises, and they are what list(values) "
            "could not reach"
        )
        assert type(plain["alpha"]) is dict, "a range mapping is a DictConfig too"
        assert plain == {"hidden_layer_sizes": [[20], [50, 50]], "alpha": {"low": 0.01, "high": 1.0}}, (
            "the values must be preserved exactly -- best_params reports them back to the "
            "user, so this may not quietly coerce [20] to (20,) or to 20"
        )
        json.dumps(plain)      # the operation that raised in production

    @pytest.mark.parametrize("name,config", _configs())
    def test_every_shipped_gridsearch_block_builds_a_json_serializable_space(self, name, config):
        """The corpus-level claim: no shipped block can reach Optuna with a value it
        cannot serialise, for any model, not only the ``mlp`` block that happened to fail.

        Blocks are built one key at a time so that a single unsearched or malformed entry
        reports itself rather than aborting the whole block, and so the failure message can
        name the key. Keys the config leaves unset are skipped -- an empty value means "not
        tuned" and ``build_search_space`` rightly refuses to build a space from nothing.

        This test does not guarantee a *container*-valued choice is present to be checked:
        that depends on what the configs currently spell, and flattening
        ``hidden_layer_sizes`` would make it vacuous without failing. The integration test
        below closes that gap with a synthetic config, which cannot drift.
        """
        blocks = [k for k in config.keys() if k.startswith("gridsearch_")]
        assert blocks, f"{name}: no gridsearch_* block, so this config tunes nothing."
        bad = []
        for block in blocks:
            model = block[len("gridsearch_") : -len("_args")]
            for key, raw in config[block].items():
                if to_plain(raw) in (None, [], {}):
                    continue
                space = build_search_space(model, {key: raw})
                for spec_name, spec in space.items():
                    if not isinstance(spec, _Categorical):
                        continue
                    for value in spec.values:
                        try:
                            json.dumps(value)
                        except TypeError as exc:
                            bad.append(
                                f"{block}[{spec_name}] choice {value!r} "
                                f"({type(value).__name__}): {exc}"
                            )
        assert not bad, (
            f"{name}: Optuna serialises the chosen parameters, so these die inside a "
            f"joblib worker at the first trial the TPE sampler steers -- taking every "
            f"model of the pass with them:\n  " + "\n  ".join(bad)
        )

    def test_the_tuner_survives_the_trials_after_the_random_startup_phase(self):
        """The end-to-end claim, run past the boundary the production failure sat behind.

        The trial budget is read off ``TPESampler``'s own default rather than hardcoded,
        because that default is exactly what made this reachable at Optuna 5 and not at 4.
        Four trials past it is enough to enter the relative-sampling path repeatedly.

        The synthetic space is built with a float range alongside the categoricals on
        purpose. Without it the space is finite and ``run_study`` lowers ``n_trials`` to the
        number of distinct points -- which for a handful of categoricals is *below* the
        startup count, so the test would never reach the failing path and would pass
        against the unfixed code. It also exercises the ``DictConfig`` -> ``dict`` half of
        ``to_plain``, which the range branch depends on.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        # Imported, not importorskip'd like omegaconf above: optuna is in
        # requirements-base.txt, so a skip here could only ever hide a broken install --
        # which is what tests/test_suite_hygiene.py forbids.
        import optuna

        startup = inspect.signature(
            optuna.samplers.TPESampler.__init__
        ).parameters["n_startup_trials"].default
        n_trials = startup + 4

        block = OmegaConf.create({
            "layers": [[20], [50], [100], [20, 20]],
            "mode": ["a", "b"],
            "alpha": {"low": 0.01, "high": 1.0},
        })
        space = build_search_space("stub", block)
        assert any(
            isinstance(spec, _Categorical) and any(isinstance(v, list) for v in spec.values)
            for spec in space.values()
        ), "guard the guard: this test is only a regression test while a choice is a list"

        rng = np.random.RandomState(0)
        best = run_study(
            _Stub, space, rng.rand(24, 3), rng.randint(0, 2, 24),
            cv=3, n_trials=n_trials, model="stub", seed=0,
        )
        assert best["layers"] in [[20], [50], [100], [20, 20]], (
            f"best_params must report the value the config wrote, got {best['layers']!r}"
        )
        json.dumps(best)


class TestTheTabpfnTokenPreflightIsVersionAware:
    """Defect 7: a pre-flight warning that announced a failure that was not going to happen.

    Every one of the twelve pilot jobs logged, before doing any work::

        'tabpfn' is in the model list but no API token was found, so its pretrained
        weights cannot be downloaded and the model will fail.

    All twelve pin ``model_version: 'v2'``, whose weights are Apache 2.0 plus attribution
    and download anonymously -- no token, no licence acceptance, no account. So the line
    was false, and falsely specific: it named a failure, a cause and a remedy, none of
    which applied. ``compute_tabpfn`` had the correct statement of this all along (its
    ``_load_tabpfn_classifier`` error text says the pinned ``v2`` weights "need no token
    and no license acceptance"), so the two halves of the codebase disagreed.

    This is worth a test rather than a one-line edit because the failure mode is a
    *diagnostic* one, and those decay silently: nothing breaks when a warning is wrong, so
    nothing tells you. The cost is paid later, by a reader who is triaging a real failure
    and has learned that this log lies. It also actively misdirected the investigation of
    the pilot's genuine crash, which was a ``TypeError`` in the tuner and had nothing to do
    with TabPFN.

    The guard has to cut both ways, so these tests pin both directions: silent for the
    token-free versions, and still loud for the restricted ones, which really do need a
    token and really will fail without one.
    """

    def test_no_shipped_pilot_config_asks_for_a_token(self):
        """The twelve configs must produce an empty list -- they all pin the free ``v2``.

        Parametrising over the configs would report twelve results where there is one
        fact, so they are checked together and the failure names the offenders.
        """
        offenders = {
            name: tabpfn_versions_requiring_token(config)
            for name, config in _configs()
            if tabpfn_versions_requiring_token(config)
        }
        assert offenders == {}, (
            f"these configs would warn that TabPFN needs a token: {offenders}. Either they "
            f"stopped pinning {TABPFN_DEFAULT_VERSION!r}, or the version table changed."
        )

    def test_a_config_that_says_nothing_about_the_version_needs_no_token(self):
        """The default path must be silent, since the default version is the free one.

        Covers the three ways a config can decline to choose: no block at all, a block
        without ``model_version``, and the default named explicitly.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        for label, raw in (
            ("no tabpfn block", {}),
            ("block without model_version", {"tabpfn_args": {"n_estimators": 4}}),
            ("default named explicitly", {"tabpfn_args": {"model_version": TABPFN_DEFAULT_VERSION}}),
        ):
            assert tabpfn_versions_requiring_token(OmegaConf.create(raw)) == [], label

    @pytest.mark.parametrize("version", sorted(_RESTRICTED_VERSIONS))
    def test_each_restricted_version_is_still_reported(self, version):
        """The other direction: silence must not have been bought by never speaking.

        Driven off ``_RESTRICTED_VERSIONS`` rather than a literal list so that adding a
        version to the table without teaching this helper about it fails here.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        for block in ("tabpfn_args", "gridsearch_tabpfn_args"):
            config = OmegaConf.create({block: {"model_version": version}})
            assert tabpfn_versions_requiring_token(config) == [version], block

    def test_a_restricted_version_is_found_among_a_list_of_tuned_candidates(self):
        """A tuned block may search over versions, and one bad candidate is enough.

        This is the same OmegaConf-wrapper hazard as defect 6: the candidate list arrives
        as a ``ListConfig``, so a helper that tested ``isinstance(value, list)`` without
        going through ``to_plain`` would see a scalar, take the ``[requested]`` branch and
        silently clear a config that does need a token.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        restricted = sorted(_RESTRICTED_VERSIONS)[0]
        config = OmegaConf.create(
            {"gridsearch_tabpfn_args": {"model_version": [TABPFN_DEFAULT_VERSION, restricted]}}
        )
        assert tabpfn_versions_requiring_token(config) == [restricted]

    def test_an_unrecognised_version_is_treated_as_needing_a_token(self):
        """Unknown means unproven, and the two errors are not equally cheap.

        A typo cannot be shown to be token-free, and ``normalise_model_version`` will
        reject it at fit time regardless, so reporting it costs a spurious warning while
        staying silent would cost a job that dies at its first TabPFN fit.
        """
        OmegaConf = pytest.importorskip("omegaconf").OmegaConf
        config = OmegaConf.create({"tabpfn_args": {"model_version": "v9"}})
        assert tabpfn_versions_requiring_token(config) == ["v9"]

    def test_the_preflight_block_can_name_everything_it_prints(self):
        """Both message branches must be *executable*, not merely parseable.

        The first cut of this fix shipped a ``NameError``: the warning text interpolated
        ``TABPFN_DEFAULT_VERSION`` while the import next to it pulled in only the helper.
        ``ast.parse`` accepts that happily, and the pre-flight block runs before any
        model, so the run died at once with nothing to show -- the same "fails before it
        starts" shape as the defect it was fixing. Asserting the names are importable
        together from the module the block imports them from is what makes that a test
        failure instead of a dead job.
        """
        qprofiler = importlib.import_module("qbiocode.apps.qprofiler.qprofiler")
        source = inspect.getsource(qprofiler.main)
        preflight = source[source.index("if \"tabpfn\" in args[\"model\"]") :]
        preflight = preflight[: preflight.index("log.info(f\"The number of ML methods")]

        # Read the block with ast rather than a regex over its text: a regex also matches
        # the dotted path in `from qbiocode.utils.tabpfn_account import ...`, which is a
        # module name and not a value the block reads. ast.Name yields only the latter.
        tree = ast.parse(textwrap.dedent(preflight))
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        module = importlib.import_module("qbiocode.learning.compute_tabpfn")
        from_compute_tabpfn = {
            name for name in used if re.fullmatch(r"TABPFN_[A-Z_]+|tabpfn_[a-z_]+", name)
        }
        assert from_compute_tabpfn, (
            "found no TabPFN names in the pre-flight block -- the slice above probably "
            "stopped matching the source, so this test is no longer checking anything"
        )
        for name in sorted(from_compute_tabpfn):
            assert hasattr(module, name), (
                f"the TabPFN pre-flight block uses {name!r}, which "
                f"qbiocode.learning.compute_tabpfn does not define"
            )
