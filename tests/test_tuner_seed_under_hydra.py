"""The run's seed reaches every tuner when ``args`` is a Hydra ``DictConfig``.

The qsvc, pqk, qnn, vqc, qpl and nb tuners read the seed with
``args.get("seed") if isinstance(args, dict) else None``. Under the CLI ``args`` is a
``DictConfig``, which is not a ``dict``, so all six tuned with ``seed=None`` and TPE drew
its trials from OS entropy: two processes on the same features picked different
hyperparameters from the first trial on. Every other test passes a plain dict, which is
why none of them saw it -- so these build the config the way Hydra does.
"""
import importlib

import numpy as np
import pytest
from omegaconf import OmegaConf

from qbiocode.learning import _tuning

# By path: qbiocode.learning re-exports each compute_* function under its module's name,
# so `from qbiocode.learning import compute_nb` is the function, not the module.
compute_nb, compute_pqk, compute_qnn, compute_qpl, compute_qsvc, compute_vqc = (
    importlib.import_module(f"qbiocode.learning.{m}")
    for m in ("compute_nb", "compute_pqk", "compute_qnn", "compute_qpl", "compute_qsvc", "compute_vqc")
)


def _hydra_args():
    return OmegaConf.create(
        {"seed": 42, "q_seed": 42, "backend": "simulator", "shots": 1024, "grid_search": True}
    )


def test_a_dictconfig_is_not_a_dict():
    # The premise: if OmegaConf ever made DictConfig a dict, this file would test nothing.
    assert not isinstance(_hydra_args(), dict)


@pytest.mark.parametrize(
    "args, expected",
    [({"seed": 7}, 7), (OmegaConf.create({"seed": 7}), 7), ({}, None), (None, None)],
    ids=["dict", "dictconfig", "no-seed", "no-args"],
)
def test_seed_from(args, expected):
    assert _tuning.seed_from(args) == expected


class _Captured(Exception):
    pass


@pytest.mark.parametrize(
    "module, fn, entry, search",
    [
        (compute_qsvc, "compute_qsvc_opt", "run_function_study", {"C": [1.0]}),
        (compute_pqk, "compute_pqk_opt", "run_function_study", {"reps": [1]}),
        (compute_qnn, "compute_qnn_opt", "run_function_study", {"reps": [1]}),
        (compute_vqc, "compute_vqc_opt", "run_function_study", {"reps": [1]}),
        (compute_qpl, "compute_qpl_opt", "run_function_study", {"reps": [1]}),
        (compute_nb, "compute_nb_opt", "search_hyperparameters", {"var_smoothing": [1e-9]}),
    ],
    ids=["qsvc", "pqk", "qnn", "vqc", "qpl", "nb"],
)
def test_the_tuner_gets_the_run_seed_from_a_dictconfig(monkeypatch, module, fn, entry, search):
    seen = {}

    def capture(*args, **kwargs):
        seen["seed"] = kwargs.get("seed")
        raise _Captured

    monkeypatch.setattr(module, entry, capture)
    X = np.zeros((8, 2))
    y = np.array([0, 1] * 4)
    with pytest.raises(_Captured):
        getattr(module, fn)(X, X, y, y, _hydra_args(), **search)
    assert seen["seed"] == 42


def test_two_studies_from_a_dictconfig_sample_the_same_trials():
    """End to end through the real sampler: the property the bug broke."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    space = _tuning.build_search_space(
        "nb", {"var_smoothing": {"low": 1e-12, "high": 1e-1, "log": True}}
    )

    def trials():
        from sklearn.naive_bayes import GaussianNB

        chosen = []
        _tuning.optuna.logging.set_verbosity(_tuning.optuna.logging.WARNING)
        orig = _tuning.optuna.create_study

        def create_study(*a, **k):
            study = orig(*a, **k)
            opt = study.optimize
            study.optimize = lambda *oa, **ok: opt(
                *oa, **{**ok, "callbacks": [lambda s, t: chosen.append(t.params)]}
            )
            return study

        _tuning.optuna.create_study = create_study
        try:
            _tuning.run_study(
                GaussianNB, space, X, y, cv=3, n_trials=5, model="nb",
                seed=_tuning.seed_from(_hydra_args()),
            )
        finally:
            _tuning.optuna.create_study = orig
        return chosen

    assert trials() == trials()
