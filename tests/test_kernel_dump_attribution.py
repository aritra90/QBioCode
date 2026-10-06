"""Kernel dumps that cannot be attributed to a pass must not become diagnostic rows.

The defect this guards is not a crash. ``collect_kernel_diagnostics`` globs every
``gram_*.npz``/``proj_*.npz`` under a kernels tree, and its only skip condition was an
unrecognised model label. A dump whose ``data_key`` is empty still has a recognised
label, so it was loaded, its kernel diagnostics were computed correctly, and it was
appended as a row reading ``Dataset=''``, ``embeddings=None``, ``iteration=None``.

That row is worse than a missing one, and the asymmetry is the point:

  * absent, and a per-dataset tally is short by one and the join is clean;
  * present-but-blank, and it joins against nothing in the delta-metric table while
    still counting toward every mean and every ``n``. Nothing looks wrong anywhere.

Empty keys are not hypothetical. ``compute_pqk``/``compute_qsvc`` declare
``data_key=""`` as their own default, so a tuning trial or a direct call writes to the
bare stem -- and every such call writes to the SAME stem, so the file on disk is also
whichever call finished last. The pilot run that died at optuna trial 10 left five of
them (``kernels/pilot0{2,4,6,7,8}/proj_pqk_.npz``).

What makes the guard safe rather than over-strict: ``qprofiler`` builds every real key as
``'_'.join([stem, embed, str(n_components), str(iter)])`` (qprofiler.py:732), whose last
two fields are always integers. So requiring that parse cannot reject a dump that a
profiling pass produced.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from qbiocode.utils.qc_winner_finder import (
    _parse_dump_name,
    _split_data_key,
    collect_kernel_diagnostics,
)

# Enough rows for the walker's own n >= 4 and two-class guards, so a skipped dump is
# skipped for its name and not for being too small to score.
N_ROWS, N_FEAT = 12, 3


def _write_dump(path, name, rng):
    """A PQK-shaped dump: projections plus the raw X and labels the walker needs."""
    path.mkdir(parents=True, exist_ok=True)
    y = np.array([0, 1] * (N_ROWS // 2))
    np.savez_compressed(
        path / name,
        Z_train=rng.normal(size=(N_ROWS, N_FEAT)),
        Z_test=rng.normal(size=(4, N_FEAT)),
        X_train=rng.normal(size=(N_ROWS, N_FEAT)),
        y_train=y,
        y_test=y[:4],
        best_params=np.asarray('{"kernel": "rbf", "gamma": "scale"}'),
    )


class TestUnattributableDumpsAreSkipped:
    def test_an_empty_data_key_produces_no_row(self, tmp_path, caplog):
        """The exact filename the dead pilot run left behind, five times over."""
        rng = np.random.default_rng(0)
        _write_dump(tmp_path / "pilot06_heart", "proj_pqk_.npz", rng)
        with caplog.at_level(logging.WARNING):
            df = collect_kernel_diagnostics(str(tmp_path))
        assert len(df) == 0, (
            f"an unattributable dump became {len(df)} diagnostic row(s): "
            f"{df.to_dict('records') if len(df) else ''}. A row with a blank Dataset "
            f"joins against nothing downstream but still moves every mean it lands in."
        )
        # getMessage(), not `record.message % record.args`: the latter re-applies the
        # format to an already-interpolated string and raises TypeError on the second %s.
        assert any("cannot be attributed" in r.getMessage() for r in caplog.records), (
            "the dump was dropped without saying so -- silence here is how five stale "
            "files went unnoticed in the first place"
        )

    def test_a_well_named_dump_still_produces_a_row_with_its_identity(self, tmp_path):
        """The negative control: the guard must not reject real passes."""
        rng = np.random.default_rng(1)
        _write_dump(tmp_path / "pilot06_heart", "proj_pqk_opt_heart_none_13_1.npz", rng)
        df = collect_kernel_diagnostics(str(tmp_path))
        assert len(df) == 1, f"a correctly-named dump was dropped: {len(df)} rows"
        row = df.iloc[0]
        assert (row["Dataset"], row["embeddings"], row["n_components"], row["iteration"]) \
            == ("heart.csv", "none", 13, 1)
        assert row["model"] == "pqk_opt"

    def test_the_two_are_distinguished_in_one_tree(self, tmp_path):
        """Mixed tree: the real pass survives, the trial dump beside it does not.

        This is the arrangement a finished pilot actually leaves on disk, so it is the
        one that decides whether the analysis table is trustworthy.
        """
        rng = np.random.default_rng(2)
        d = tmp_path / "pilot06_heart"
        _write_dump(d, "proj_pqk_opt_heart_none_13_1.npz", rng)
        _write_dump(d, "proj_pqk_.npz", rng)
        _write_dump(d, "gram_qsvc_.npz", rng)
        df = collect_kernel_diagnostics(str(tmp_path))
        assert list(df["Dataset"]) == ["heart.csv"], (
            f"expected exactly the one attributable dump, got {list(df['Dataset'])}"
        )

    @pytest.mark.parametrize("name", ["proj_pqk_.npz", "gram_qsvc_.npz",
                                      "proj_pqk_opt_.npz", "gram_qsvc_opt_.npz"])
    def test_every_bare_stem_the_compute_defaults_can_write(self, name):
        """All four arms default to ``data_key=""``, so all four bare stems are reachable.

        Asserted at the parse level so the case is pinned even if the walker is rewritten:
        a recognised model label plus an unparseable key is the combination that used to
        slip through, and `ncomp is None` is what the guard keys on.
        """
        model, data_key = _parse_dump_name(name)
        assert model is not None, f"{name}: label unrecognised, a different guard applies"
        _, _, ncomp, _ = _split_data_key(data_key)
        assert ncomp is None, (
            f"{name}: parsed a complete identity out of an empty key, so the walker's "
            f"guard would let it through"
        )

    def test_a_real_key_parses_to_a_complete_identity(self):
        """The other half of the guard's contract, at the same level.

        qprofiler.py:732 joins stem, embedding, n_components and iteration; the stem may
        itself contain underscores, which is why the split is from the right.
        """
        _, data_key = _parse_dump_name("gram_qsvc_opt_analcatdata_lawsuit_none_4_3.npz")
        assert _split_data_key(data_key) == ("analcatdata_lawsuit.csv", "none", 4, 3)
