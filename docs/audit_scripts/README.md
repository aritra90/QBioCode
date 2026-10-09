# Audit scripts

The four scripts behind `../AUDIT_quantum_datasets_qprofiler.md`. They are kept because
that document cites them as the evidence for specific findings (F1, F4, F12 and the
all-arms diagnostic table), not because they are part of the test suite.

They used to live in `tests/`, where they were mistaken for tests twice over: the names
look like tests, and pytest's `python_files = ["test_*.py"]` means it never collected
them, so nothing ever reported that they do not run.

## They do not run as committed

Every one of them starts with

```python
from qdata_gen import ...
```

and **`qdata_gen.py` is not in this repository** — it has never been committed on any
branch (`git log --all --diff-filter=A -- '*qdata_gen*'` is empty). The audit document
lists it first under "Supporting files", so it existed on the author's machine when the
audit was run. Without it all four scripts fail identically:

```
ModuleNotFoundError: No module named 'qdata_gen'
```

Two of them additionally read absolute paths that are not produced by anything in this
tree:

| Script | Reads |
|---|---|
| `audit_eng.py` | `/tmp/rep/x_view/eng_zz_n4_gq1_s0.csv` |
| `audit_ql.py` | `/tmp/rep/x_view/ql_zz_n4_tau1_s0.csv` |

`audit_diag.py` takes its CSVs as command-line arguments, and `audit_te.py` generates its
own data — so those two need only `qdata_gen.py`.

## To make the audit reproducible again

1. Recover `qdata_gen.py` and commit it here. It is a reference re-implementation of the
   generators (`pauli_sum`, `ising_terms`, `level_spacing_ratio`, `pool_local`,
   `sparse_observable`, `expvals`, `walsh_degree_profile`, `Pauli`, `zz_feature_state`)
   written independently of `qbiocode.data_generation` — that independence is the point,
   so it should not be replaced by an import from the package.
2. Regenerate the two CSVs, or change those two scripts to take a path argument as
   `audit_diag.py` does.

Until then, treat the numbers in the audit document as a record of a run that cannot
currently be repeated. The physics those scripts cross-check *is* pinned independently,
by `qbiocode.data_generation.quantum_selftest` and
`tests/test_quantum_data_generation.py::TestThePhysicsIsRight`, which do run in CI.
