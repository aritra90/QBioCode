"""``summary`` must not carry per-model results from one pass into the next.

The bug this guards is not a crash and not visible in any output file's shape. It
produces a results.pkl in which one split's predicted scores sit beside another split's
true labels, for some models and not others, with no warning. Anything computed from it
afterwards -- a post-hoc ROC-AUC, a PR-AUC, a calibration curve -- is then silently
wrong, and only for the threshold-free metrics. Nothing downstream can detect it.

The mechanism takes three facts together, none of which is wrong on its own:

1. ``summary = {}`` is created once per *dataset*, above both the iteration and the
   embedding loop, and is only ever updated in place. A key that a pass does not
   itself write therefore survives from the previous pass.
2. ``model_run`` ends in ``pd.melt(pd.concat(results)).dropna()``. ``dropna`` *deletes*
   an entry whose single value is ``None`` rather than carrying it through as null.
3. ``y_score_<model>`` is ``None`` for any model that exposed no usable ranking score.

So a pass in which a model yields no score does not overwrite that model's
``y_score_``; it leaves the previous pass's array in place, now paired with the current
pass's ``y_test_``. This is the same bug class as the ``row_base = dict(model_results)``
fix that cured the ModelResults.csv side; the ``summary`` -> results.pkl side was never
fixed, and adding ``y_score`` to the persisted keys is what turned it from latent into
live.

The first test checks the melt/dropna behaviour directly, so the premise is verified
against the installed pandas rather than assumed. The rest assert structurally that the
clear is present, complete, and positioned before the update -- structural because the
loop is not factored out of ``main`` and driving it end to end would mean running real
models over real datasets.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

import qbiocode.apps.qprofiler.qprofiler as qprofiler

#: Every per-model key ``model_run`` contributes to ``summary``. All four are cleared,
#: not just ``y_score_``: a stale ``results_<model>`` would otherwise be rewritten to
#: ModelResults.csv under the *current* pass's iteration and embedding labels.
STALE_PREFIXES = ("results_", "y_test_", "y_predicted_", "y_score_")

SOURCE = Path(inspect.getfile(qprofiler)).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_melt_dropna_deletes_a_none_valued_key_but_keeps_an_array_valued_one():
    """The premise, verified rather than assumed.

    ``model_run`` returns ``pd.melt(pd.concat(results)).dropna()``. If ``dropna`` merely
    preserved a ``None`` as null, a pass with no ranking score would overwrite the
    previous pass's value and there would be nothing to clear. It does not: the key
    disappears entirely, which is what lets the stale array survive.
    """
    # shaped as model_run builds it: one single-row frame per persisted key
    frames = [
        pd.DataFrame({"y_score_lr": [[0.1, 0.9]]}),
        pd.DataFrame({"y_score_dt": [None]}),
    ]
    melted = pd.melt(pd.concat(frames)).dropna()
    kept = set(melted["variable"])
    assert "y_score_lr" in kept, "an array-valued score should survive the melt"
    assert "y_score_dt" not in kept, (
        "pandas no longer drops a None-valued key; if this ever changes, the stale-key "
        "clear in qprofiler is no longer load-bearing and this suite should be revisited"
    )


def _summary_update_with_model_run():
    """The ``summary.update(model_run(...))`` call node, plus its enclosing statements."""
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "update"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "summary"):
            continue
        if node.args and isinstance(node.args[0], ast.Call):
            inner = node.args[0].func
            if isinstance(inner, ast.Name) and inner.id == "model_run":
                return node
    return None


def test_the_model_run_update_is_still_where_this_suite_thinks_it_is():
    """Anchor. Every assertion below is about statements around this one call."""
    assert _summary_update_with_model_run() is not None, (
        "no `summary.update(model_run(...))` call found in qprofiler; the loop was "
        "refactored and the stale-key guard below needs to be re-pointed rather than "
        "assumed to still hold"
    )


def _deletion_nodes():
    """``del`` statements whose target is a subscript of ``summary``."""
    found = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Delete):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "summary"
            ):
                found.append(node)
    return found


def test_summary_is_cleared_of_per_model_keys_somewhere():
    assert _deletion_nodes(), (
        "nothing deletes per-model keys from `summary`. Because `summary` outlives both "
        "loops and melt/dropna omits None-valued keys, a pass with no ranking score "
        "will inherit the previous pass's y_score and pair it with this pass's y_test."
    )


def test_the_clear_names_every_persisted_per_model_prefix():
    """A partial clear is worse than none: it looks handled and is not.

    Searched over the whole module source rather than the one ``del`` node, since the
    prefixes live in the comprehension that feeds it.
    """
    missing = [p for p in STALE_PREFIXES if f'"{p}"' not in SOURCE and f"'{p}'" not in SOURCE]
    assert not missing, (
        f"these per-model prefixes are never named in qprofiler, so keys carrying them "
        f"are never cleared from `summary`: {missing}"
    )


def test_the_clear_happens_before_the_results_are_written():
    """Ordering is the whole point -- clearing after the update would erase the new
    results instead of the stale ones."""
    update = _summary_update_with_model_run()
    deletions = _deletion_nodes()
    assert deletions, "no deletion to order against"
    assert any(d.lineno < update.lineno for d in deletions), (
        f"every `del summary[...]` (lines {[d.lineno for d in deletions]}) sits after "
        f"the model_run update (line {update.lineno}); that deletes the current pass's "
        f"results and leaves the stale ones in place -- the inverse of the intent"
    )


def test_the_clear_sits_inside_the_embedding_loop_not_above_it():
    """Per *pass*, not per dataset or per iteration.

    A clear hoisted above the embedding loop would still leak between embeddings, which
    is the innermost axis and therefore the one that repeats most often -- the common
    case rather than an edge case.
    """
    update = _summary_update_with_model_run()
    deletions = _deletion_nodes()

    def loops_containing(line):
        depth = []
        for node in ast.walk(TREE):
            if isinstance(node, ast.For):
                end = getattr(node, "end_lineno", node.lineno)
                if node.lineno <= line <= end:
                    depth.append(node.lineno)
        return sorted(depth)

    update_loops = loops_containing(update.lineno)
    assert update_loops, "the model_run update is not inside any for-loop"
    innermost = update_loops[-1]
    assert any(innermost in loops_containing(d.lineno) for d in deletions), (
        "the per-model clear is not inside the same innermost loop as the model_run "
        "update, so `summary` still carries results across passes of that loop"
    )


@pytest.mark.parametrize("prefix", STALE_PREFIXES)
def test_each_prefix_is_cleared_by_a_single_shared_rule(prefix):
    """One ``startswith`` over a tuple, rather than four hand-written deletes.

    Enumerating them individually is how the next persisted key gets forgotten: adding
    ``y_score`` to ``model_evaluation`` is exactly what exposed this bug in the first
    place, and a shared rule means the next such addition is covered by construction.
    """
    assert "startswith" in SOURCE
    assert f'"{prefix}"' in SOURCE or f"'{prefix}'" in SOURCE
