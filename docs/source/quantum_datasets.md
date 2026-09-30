# Simulated quantum datasets

QBioCode generates five families of **binary classification** datasets from exact
statevector simulation. Every row is a classical description of a quantum instance, and
every label is a property of the corresponding quantum state. They exist so a QProfiler
sweep can include problems whose difficulty is known by construction rather than
measured after the fact: two classically-easy controls, one difficulty ladder with a
measured knob, and two positive controls built to favour a quantum kernel.

They are simulated, so they are not evidence of quantum advantage. The
[Non-claims](#non-claims) section below is part of the specification, not a disclaimer
appended to it.

## Verify before use

```bash
qdata-gen selftest      # must print SELFTEST PASS
```

Each check compares an implementation against an independent computation of the same
quantity:

| Check | Compared against |
|---|---|
| Bit-trick Pauli expectations | dense Kronecker products, all 4⁴ strings on 4 qubits |
| Sparse Hamiltonian assembly | the dense Hamiltonian |
| Even-sector ground state | its own eigenvalue residual, its ∏X parity, and the global minimum |
| Fast Walsh-Hadamard transform | its inverse, and Parseval's identity |
| Short-time quench | the analytic expansion ⟨Z_i(t)⟩ ≈ 1 − 2h_i²t² |
| Engineered labels | the bound they target: s_Q = 1 and s_C = g² |
| Native `ZZFeatureMap` | Qiskit's, for `linear`, `pairwise` and `full`, to ~1e-15 |

The same checks run in `pytest` as
`tests/test_quantum_data_generation.py::TestThePhysicsIsRight`, calling the same
functions, so the CLI's report and the test suite's verdict cannot disagree.

## Generate

Either from Python:

```python
from qbiocode import generate_ground_state_datasets

generate_ground_state_datasets(
    n_qubits=8, n_samples=400, label="sparse", kappa=0.5, save_path="qdata_out",
)
```

or through `generate_data`, which reaches the same generators by name and takes
everything past the qubit and row count in one `quantum_args` dict:

```python
from qbiocode import generate_data

generate_data(
    type_of_data="ground_state",     # or time_evolution, hamiltonian_learning,
    save_path="qdata_out",           #    quantum_labels, engineered_kernel
    dim=[8],                         # qubits
    n_samples=[400],                 # rows
    quantum_args={"label": "sparse", "kappa": 0.5},
)
```

or from the command line, one dataset per invocation. These eight commands are the
reference runsheet — the configuration every documented number below was measured at:

```bash
OUT=qdata_out
qdata-gen gs  --label sparse --n 8 --N 400 --kappa 0.5 --s 4 --seed 0 --out $OUT
qdata-gen gs  --label e2e    --n 8 --N 400 --kappa 0.5        --seed 0 --out $OUT
qdata-gen te  --n 10 --N 400 --s 4 --taus 0.25 0.5 1 2 4      --seed 0 --out $OUT
qdata-gen hl  --n 6 --N 400 --times 0.5 --shots 1000          --seed 0 --out $OUT
# aligned to the QSVC defaults -- reps, entanglement AND the qiskit data map qsvc uses
qdata-gen ql  --encoding zz  --n 8 --N 400 --tau 1 --reps 2 --entanglement linear   --seed 0 --out $OUT
# aligned to the PQK defaults: reps and entanglement, plus the unit data map pqk hardcodes.
# Dropping --data-map here would leave it aligned in depth but not in encoding, which is
# the larger of the two effects.
qdata-gen ql  --encoding zz  --n 8 --N 400 --tau 1 --reps 4 --entanglement pairwise --seed 0 --out $OUT --data-map unit --name ql_zz_pqkdefaults
# misaligned control
qdata-gen ql  --encoding evo --n 8 --N 400 --tau 1            --seed 0 --out $OUT
# one engineered set per arm: the default targets qsvc, --data-map unit targets pqk
qdata-gen eng --n 6 --N 300 --gamma_q 1.0                     --seed 0 --out $OUT
qdata-gen eng --n 6 --N 300 --gamma_q 1.0 --data-map unit      --seed 0 --out $OUT
```

The Python API's own defaults *are* this runsheet, so calling a generator with no
arguments reproduces its runsheet row -- except the two rows that deliberately depart from
the defaults, `ql_zz_pqkdefaults` and the `--data-map unit` engineered set, which name the
knobs they change. The CLI's defaults are the generic ones of the
original standalone script, and each invocation writes exactly one configuration.

### Layout written

| Path | Content | Run in QProfiler as |
|---|---|---|
| `$OUT/x_view/*.csv` | features = the classical description x (couplings, bitstrings, or raw inputs) | the main arm |
| `$OUT/phi_view/*.csv` | features = local Pauli expectations of the data state (`gs`, `te` only) | the "quantum-feature oracle" arm |
| `$OUT/meta/*.json` | parameters, label rule, threshold, diagnostics | ignored by QProfiler (only top-level `*.csv` is read) |
| `$OUT/meta/*_F.npy` | the continuous pre-threshold target F | ignored by QProfiler |

Labels are `1[F > median(F)]`, so the classes are balanced by construction. `--margin δ`
drops rows with `|F − median| < δ`, removing the ambiguous band around the boundary.
Keep `F` for any regression or margin analysis; do not treat `sign(F)` as `F`.

A dataset's name encodes the parameters that identify it, not every knob — no family's
name carries the row count, and `ql`'s carries neither `reps` nor `entanglement`. A sweep
over one of those would write several datasets to one path, so the generators refuse
before computing anything and name the knob to vary instead.

### The reference corpus

The runsheet writes 13 datasets: one per command, except `te`, which writes one per
entry of `--taus`. This is the corpus the benchmark runs read. Each dataset was
regenerated from this code and compared with the stored copy, and all 13 `x_view` and
all 7 `phi_view` CSVs are byte-identical. In `meta/`, the `eng` and `ql` sidecars'
`threshold` and pre-threshold target `F` (and `eng`'s `g`) agree to ~1e-14. That is
floating-point summation order, not a different dataset.

| Dataset | Role | Qubits | Rows | Views | Label | Diagnostic | Reproduces |
|---|---|---|---|---|---|---|---|
| `eng_zz_n6_gq1_s0` (pilot10) | engineered positive control | 6 | 300 | x | built for a projected kernel: RBF (gamma_q=1) on the Bloch vectors of ZZFeatureMap, reps 2, linear, data_map `qiskit` | g = 3.23 (continuous target) | byte-identical |
| `eng_zz_n6_gq1_s0_dmunit` | engineered positive control | 6 | 300 | x | built for a projected kernel: RBF (gamma_q=1) on the Bloch vectors of ZZFeatureMap, reps 2, linear, data_map `unit` | g = 2.15 (continuous target) | byte-identical |
| `gs_e2e_n8_k0.5_s0` | closed-form twin known | 8 | 400 | x, phi | ground-state `Z0Z7`, kappa 0.5 | min even-sector gap 2.12; product criterion 0.87 held out | byte-identical |
| `gs_sparse_n8_k0.5_s0` | classically-easy physics control | 8 | 400 | x, phi | ground-state `Z1Z2+X2X3+X6X7+Y5Y6`, kappa 0.5 | min even-sector gap 2.04 | byte-identical |
| `hl_n6_g0.5_shots1000_s0` | Hamiltonian learning (classically easy) | 6 | 400 | x | `1[mean(J) - mean(h) > median]`, quench t = [0.5], 1000 shots | -- | byte-identical |
| `ql_evo_n8_tau1_s0` | misaligned negative control | 8 | 400 | x | evo encoding, reps 2, then exp(-iH tau), tau 1; label <Z0> | -- | byte-identical |
| `ql_zz_n8_tau1_s0` | aligned positive control | 8 | 400 | x | ZZ encoder, reps 2, linear, data_map `qiskit`, then exp(-iH tau), tau 1; label <Z0> | -- | byte-identical |
| `ql_zz_pqkdefaults` | aligned positive control | 8 | 400 | x | ZZ encoder, reps 4, pairwise, data_map `unit`, then exp(-iH tau), tau 1; label <Z0> | -- | byte-identical |
| `te_n10_s4_seed0_tau0.25` | difficulty ladder (tau) | 10 | 400 | x, phi | sparse observable after exp(-iH tau), tau 0.25 | Walsh effective degree 2.01; level spacing 0.54 | byte-identical |
| `te_n10_s4_seed0_tau0.5` | difficulty ladder (tau) | 10 | 400 | x, phi | sparse observable after exp(-iH tau), tau 0.5 | Walsh effective degree 2.13; level spacing 0.54 | byte-identical |
| `te_n10_s4_seed0_tau1` (pilot10) | difficulty ladder (tau) | 10 | 400 | x, phi | sparse observable after exp(-iH tau), tau 1 | Walsh effective degree 1.56; level spacing 0.54 | byte-identical |
| `te_n10_s4_seed0_tau2` | difficulty ladder (tau) | 10 | 400 | x, phi | sparse observable after exp(-iH tau), tau 2 | Walsh effective degree 2.55; level spacing 0.54 | byte-identical |
| `te_n10_s4_seed0_tau4` | difficulty ladder (tau) | 10 | 400 | x, phi | sparse observable after exp(-iH tau), tau 4 | Walsh effective degree 3.47; level spacing 0.54 | byte-identical |

- **Label** is the quantity thresholded at its median; `meta/<name>.json` records every
  parameter and is the authoritative description of a dataset.
- **The two `eng` sets are built for a projected kernel**: an RBF kernel, bandwidth
  `gamma_q`, on the 1-local Bloch vectors of the `ZZFeatureMap` state at `reps` 2,
  `linear`, with the data map shown. A learner reproduces that kernel only with the same
  map, `reps`, entanglement and bandwidth. Neither shipped arm does so exactly. `qsvc`
  builds a fidelity kernel, not a projected one. `pqk` always uses the `unit` map, and
  `config_qdata_encoded` runs it at `reps` 4, `pairwise`. Compare an `eng` result with the
  construction above, not with the arm's name.
- **The `te` ladder is not monotone in its own difficulty measure.** The Walsh effective
  degree is 2.01, 2.13, 1.56, 2.55 and 3.47 for `tau` 0.25, 0.5, 1, 2 and 4: `tau` 1 is
  the lowest-degree rung, not a middle one. Use the measured degree, not `tau`, as the
  difficulty covariate.
- *(pilot10)* marks the two datasets the 12-dataset pilot ran.

## QProfiler settings for these datasets

Two configs ship ready to use:

```bash
qprofiler --config-dir=$(pwd)/qbiocode/apps/qprofiler/configs \
          --config-name=config_qdata_xview folder_path=$(pwd)/qdata_out/x_view
```

`config_qdata_xview.yaml` covers `gs`, `te` and `hl` in either view;
`config_qdata_encoded.yaml` covers `ql` and `eng`. What they set, and why:

- `folder_path` must name **one view directory**, never the parent. CSV discovery is a
  non-recursive `os.listdir`, so pointing it at `$OUT/` finds nothing at all.
- `embeddings: ['none']`. The features ARE the physics: compressing them to
  `n_components: 3` discards the structure the label depends on, and `nmf` cannot take the
  signed features of `phi_view` or `te` at all. QProfiler now also reaches this conclusion
  on its own for these datasets -- `embedding_min_features` defaults to 18 and every
  runsheet configuration here is narrower (15, 12, 10, 8, 6), so
  {func}`qbiocode.embeddings.resolve_embeddings` collapses a requested
  `['pca','nmf','none']` to a single `'none'` pass anyway. Setting it explicitly makes
  that independent of the threshold, so these configs behave the same if someone raises
  `embedding_min_features` or sets it to 0.
- Qubits = feature count under `none`: `gs` = 2n−1, `te` = n, `hl` = 2n × len(times),
  `ql`/`eng` = n. At the runsheet sizes that is 15, 10, 12, 8 and 6.
- `scaling: false` for `ql` and `eng`. Their inputs are generated in [0,1] and the hidden
  label rule was evaluated on exactly those values, so a per-fold `MinMaxScaler` would
  hand the encoder inputs the label generator never saw.
- **Encoder alignment matters for the positive controls.** QProfiler's QSVC default is
  ZZ/linear/reps 2 and its PQK default is ZZ/pairwise/reps 4. `pairwise` and `linear`
  generate the same state for `ZZFeatureMap` (verified in the selftest), but `reps` must
  match: in the smoke check a reps mismatch alone removed the `ql_zz` advantage
  (n=6, N=200, one seed).

```{note}
`phi_view` features are Pauli expectation values — signed, with means near zero. The
`Coefficient of Variation %` column of `RawDataEvaluation.csv` is `std/mean`, so for that
view it is huge or non-finite. That is arithmetic, not a failure, and the column is simply
not comparable between the two views. Every other column is fine.
```

```{note}
`RawDataEvaluation.csv` comes in two schemas, and
{func}`qbiocode.evaluation.dataset_evaluation.detect_complexity_schema` identifies which
one a table is in. The current `'pymfe'` block is 142 columns: `Dataset`, 10 native
measures kept from the older set, 115 `mfe.*` columns from
[pyMFE](https://github.com/ealcobaca/pymfe), and 16 `task.*` target-spectrum columns
(8 measures, each also as a `_z` score against a label-permutation null). The older
`'legacy'` block is 23 measures.

For the feature-count check above this matters in one place: `# Features` and `# Samples`
are `mfe.nr_attr` and `mfe.nr_inst` on the current block. `Coefficient of Variation %` is
one of the ten survivors, so the caveat in the previous note applies to both schemas.
```

## Worked example

[Simulated Quantum Datasets in QProfiler](tutorials/Quantum_Data/quantum_datasets_qprofiler.ipynb)
generates one dataset per family at notebook scale and runs both configs over them, then
checks the profiled feature count against the qubit count. It is deliberately too small to
support any accuracy claim -- it shows the mechanics and the audit checks.

## Families, roles, pre-registered predictions

The prediction column is what each family is *for*. A family that does not behave as
predicted is reporting a problem with the pipeline, and the audit trigger says where to
look.

| Family | Role | Prediction | Audit trigger |
|---|---|---|---|
| `gs` (x_view) | classically-easy physics control (gapped; the Huang et al. *Science* 2022 regime) | classical ≥ QSVC/PQK | any quantum win → construct the classical twin, or look for a bug |
| `gs --label e2e` | closed-form twin known | the product criterion sign(Σ log J − Σ log h) captures most of the label (0.87 held-out in the smoke check) | twin accuracy ≪ best classical → re-examine |
| `te` (x_view) | difficulty ladder in τ (Walsh effective degree recorded in the metadata) | every x-view learner degrades as the degree grows | no degradation with τ |
| `te` (phi_view) | measure-first learner, handed the right measurements (Molteni-type) | stays high; the φ − x gap grows with τ | the gap does not increase with τ |
| `hl` | Hamiltonian learning as classification | classically easy (the short-time expansion gives h_i directly) | classical fails → feature or label bug |
| `ql --encoding zz` | aligned positive control | PQK/QSVC > classical **when the encoder settings match, including the data map** | fails → pipeline bug, or a settings mismatch |
| `ql --encoding evo` | misaligned negative control | classical ≥ quantum | quantum wins → investigate |
| `eng` | engineered positive control (g(K_C‖K_Q) saturated for the continuous target) | PQK > classical, **only for the arm whose data map was targeted** | fails after binarisation → check `--data-map` first, then measure the boundary-complexity ratio |

### The data map decides which arm is aligned

QProfiler's two quantum models do not build the same feature map, so there is no
single setting that aligns with both:

| QProfiler model | `data_map_func` it passes | Generate with |
|---|---|---|
| `qsvc` | none → Qiskit's default, φ(xᵢ) = xᵢ and φ(xᵢ,xⱼ) = (π−xᵢ)(π−xⱼ) | `--data-map qiskit` (the default) |
| `pqk` | {func}`~qbiocode.utils.qutils.unit_coefficient_data_map`, φ(xᵢ) = xᵢ/2 and φ(xᵢ,xⱼ) = xᵢxⱼ/2 | `--data-map unit` |

These are different unitaries, not a reparameterisation, so `K_Q` is a different
kernel and the adversarial alignment `eng` is built on does not carry over. Getting
it wrong does not merely weaken the effect, it **inverts** it: on
`eng_zz_n4_gq1_s0`, projected-kernel accuracy over 40 stratified 70/30 splits is

| Labels built against | PQK accuracy |
|---|---|
| Qiskit's default map (`--data-map qiskit`) | **0.797 ± 0.073** |
| the `unit` map that `pqk` actually uses | 0.403 ± 0.108 |

Below chance in the mismatched case, because the labels are anti-aligned with the
other encoding's geometry. `reps` and `entanglement` have to match too, but they are
the *smaller* effect; the data map alone accounts for the whole collapse.

```{warning}
A `ql --encoding zz` or `eng` dataset carries its data map in
`meta/<name>.json` under `data_map`, and non-default names end in `_dmunit`.
Check that field against the arm you are running before reading anything into the
result. A mismatch looks exactly like a pipeline bug.
```

## Non-claims

- **Nothing here shows quantum advantage.** Every state at n ≤ 12 is exactly simulable,
  so the classical twin of any quantum arm or φ-view arm is "simulate, then fit".
- The aligned and engineered families are **circular by construction**: they validate the
  pipeline, not advantage.
- "Aligned" is relative to **one** consumer. `qsvc` and `pqk` use different data maps, so
  a family aligned with one is a negative control for the other, and a low score on the
  mismatched arm is the construction working as specified rather than a finding.
- The `evo` encoding is this generator's own definition, **not** Huang et al.'s E3.
- The engineered-label formula is derived here to satisfy eq. (5) of Huang et al. 2021;
  their Supplementary §7 procedure was not consulted.
- `g` is computed for the **continuous** target. The label on disk is that target's
  median binarisation, and thresholding is not guaranteed to preserve the separation, so
  `g` must not be quoted as an advantage for the classification task. The metadata carries
  the caveat next to the number.
- Smoke numbers are one seed, 5-fold CV, small grids, and their FQK/PQK arms are native
  re-implementations rather than QProfiler runs.
- QProfiler can express access levels (i) and (ii) only; coherent-processing (fully
  quantum) tasks are out of scope for it.

## Conventions

Little-endian: qubit q corresponds to bit q of the basis index, as in Qiskit. Pauli labels
in the metadata read `X3Z4` = X on qubit 3, Z on qubit 4. `gs` ground states are taken in
the ∏X = +1 sector.

## References

- H.-Y. Huang, R. Kueng, G. Torlai, V. V. Albert and J. Preskill,
  *Provably efficient machine learning for quantum many-body problems*,
  Science **376**, 1182 (2022).
- H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven and
  J. R. McClean, *Power of data in quantum machine learning*,
  Nature Communications **12**, 2631 (2021).
- S. Molteni, C. Gyurik and V. Dunjko, *Exponential quantum advantages for practical
  non-Hermitian eigenproblems*, npj Quantum Information **12**, 19 (2026).

## API reference

```{eval-rst}
.. autosummary::
    ~qbiocode.data_generation.make_ground_state.generate_ground_state_datasets
    ~qbiocode.data_generation.make_time_evolution.generate_time_evolution_datasets
    ~qbiocode.data_generation.make_hamiltonian_learning.generate_hamiltonian_learning_datasets
    ~qbiocode.data_generation.make_quantum_labels.generate_quantum_label_datasets
    ~qbiocode.data_generation.make_engineered_kernel.generate_engineered_kernel_datasets
    ~qbiocode.data_generation.quantum_selftest.run_selftest
```
