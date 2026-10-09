# Audit — "Simulated Quantum Datasets in QProfiler"

**Scope:** `quantum_datasets_qprofiler` notebook, PDF export dated 24 Sep 2026 (17 pages).
**Supporting files:** the reference generator `qdata_gen.py`; the audit scripts `audit_diag.py`, `audit_eng.py`, `audit_ql.py`, `audit_te.py` — now in [`audit_scripts/`](audit_scripts/).

> **The run below cannot currently be repeated.** `qdata_gen.py` was never committed to
> this repository, so all four audit scripts fail with `ModuleNotFoundError`. See
> [`audit_scripts/README.md`](audit_scripts/README.md) for what is missing and what it
> would take to restore. The physics these scripts cross-check is pinned independently by
> `qbiocode.data_generation.quantum_selftest`, which does run in CI.

---

## 0. Scope, method, limits

**What was checked**

- **Reference reproduction.** Every printed diagnostic was reproduced with the reference generator that the package ports.
- **QProfiler code paths.** The code the notebook exercises was read in QBioCode public `main`, commit `d6cd204` (15 Sep 2026).
- **Expected accuracies.** Each model's expected score at the notebook's sizes was calibrated from 40–100 repeated stratified 70/30 splits with exact statevectors. This mirrors QProfiler's single 70/30 split; the notebook's accuracies are k/18 and k/24.
- **Pre-registrations.** The notebook's predictions were re-tested on multi-seed data.
- **Citations.** Each was checked against its primary source.

**Limits**

- **Branch.** The quantum-data integration is not in public `main`: `qbc.generate_data(type_of_data="ground_state", …)`, `run_selftest`, and `config_qdata_*.yaml` live on your local QBC_v2 branch. Findings tagged **[public-main]** describe public code and must be re-checked on your branch.
- **Unseen artefacts.** I cannot see `ModelResults_*.csv`, the PQK projection cache, or your CSV files. A few notebook lines are clipped at the right margin of the PDF.
- **Expected accuracies are calibration, not benchmark results.** They come from my native re-implementation: exact Bloch vectors and fidelity kernels, and QBioCode-style `RandomizedSearchCV` for PQK. Each uses one dataset draw.

**Severity scale**

- **Critical** — invalidates a stated result, or can silently corrupt results.
- **High** — a stated purpose or pre-registration is wrong or untested.
- **Medium** — misleading text or weak statistics.
- **Low** — clarity or hygiene.

---

## 1. Verdict

**Strengths.** As a mechanics tutorial it is good, and most of its design decisions are right:
- It verifies the simulator before generating anything.
- It uses two output trees because scaling must differ.
- It asserts that QProfiler's feature count equals the generated feature count.
- It moves `RawDataEvaluation.csv` aside because QProfiler overwrites it on every call.
- It documents the `reps`/`entanglement` naming collision.
- It states that *g* applies to the continuous target, not the binarised label.
- It guards `n_samples ≤ 2ⁿ`.
- It ends with an honest non-claims list.

The package also reproduces the reference generator exactly (§2.1).

**Not yet safe as a benchmark front-end.** Six issues:

1. **F1 (Critical).** The positive control (`eng`) failed, and the notebook does not notice.
   - QProfiler's PQK scored 0.444.
   - A correctly aligned PQK at this size scores 0.80 ± 0.09, and scored ≤ 0.444 in 0 of 100 splits.
2. **F2 (Critical, [public-main]).** The PQK/QPL projection cache can silently serve the wrong dataset.
   - QProfiler builds the cache key from the filename cut at the **first "."**, so `tau0.25` and `tau0.5` collide, as do `shots1000` and `shots100`.
   - The key has no content hash.
3. **F3 (High).** No quantum model is ever run on `gs`/`te`/`hl`, and `phi_view` is never ingested. The question the corpus exists for (QML vs classical on quantum data) is therefore untested.
4. **F4 (High).** The `te` pre-registration ("falls monotonically in τ") is false per instance.
   - It fails in 3/8 seeds on accuracy and 5/8 seeds on effective degree.
   - The population-level trend does hold.
5. **F5 (High).** The `ql --encoding evo` expectation ("all learners mediocre") is contradicted by the data: classical RF scores 0.93 at the runsheet size.
6. **F6 (High, [public-main]).** QProfiler's low-variance-feature metric has two defects.
   - A `ddof` mismatch produces the "No feature is strong enough to keep" messages.
   - The metric is ≈ *n*/4 by construction, so it carries no information into QSage.

**The code, separately from the notebook's claims.** §1's six issues are about what the
notebook *asserts*; §9 and §10 are about whether the code is sound and whether the suite
qualifies. Those came out differently:

- **14 code defects found and fixed** (D1–D14), of which four were real correctness bugs
  rather than tidying: a reversed operator order in `evo_encoding_state`, a diagnostic
  scoring rows that were never shipped, a provenance index keyed to filenames that were
  never written, and a figure legend that dropped a swatch and mislabelled the rest. Two
  more were robustness defects that no test would ever have caught — a physics self-test
  that reports PASS having verified nothing under `python -O`, and nine JSON sidecars
  written at whatever the process locale happened to be. One (D14) was a *dormant*
  correctness defect: four of the five time-evolution operators spelled their adjoint
  `V.T` instead of `V.conj().T`, which is bitwise identical for the real Hamiltonians the
  package builds and wrong by 0.33 for a complex one — while still passing every unitarity
  check. It is reported as dormant rather than as a live bug, because that is what it is.
- **13 CI-breaking test failures closed** by gating on the `[tabpfn]` extra the way the
  repository already gates on `quimb`, proven in the install state CI is actually in
  (§10.3: 380 passed / 0 failed / 0 errors, against 6 failed / 7 errors before).
- **The suite qualifies on every tier** — **1457 unit** (16 xfailed, 0 skipped, 0 failed),
  **179 integration**, **6 notebooks**, and **1593 passed** under the exact command CI runs
  (coverage on, the `[tabpfn]` extra absent) — all exit 0, the Sphinx site builds clean from
  an empty doctree with **zero** warnings (§10.4c), and the one lint gate CI does not allow to
  fail reports **0** (§10.2, §10.7). Every skip in every tier is attributed by file and cause. The port adds **153 tests** and converts none of the
  existing suite into a skip. One of those tests failed on the first authoritative run and
  the failure was the suite's own hygiene gate catching an `importorskip` on `scipy` in the
  audit's own new tests — fixed, re-run, and written up in §10.2 rather than quietly
  corrected, because a gate that fires on the auditor is the only evidence it fires at all.
- **Numerics are unchanged**, proven by byte comparison against `origin/aritra/v2` rather
  than by inspection (§2.1, *Corpus fidelity after the fixes*).

---

## 2. What checks out

### 2.1 The package reproduces the reference generator exactly

Every diagnostic the notebook printed was regenerated with `qdata_gen.py` using the same arguments and `seed=0`:

| Dataset | Notebook | Reference | Match |
|---|---|---|---|
| `gs_e2e_n6_k0.5_s0` | gap_min 2.5816, gap_median 3.9384, product_criterion 0.825 | 2.5816, 3.9384, 0.825 | ✓ |
| `gs_sparse_n6_k0.5_s0` | gap_min 2.2759, gap_median 3.7865 | 2.2759, 3.7865 | ✓ |
| `te_n6_s4_seed0_tau{0.25,1,4}` | ⟨r⟩ 0.4266; eff. degree 1.1652 / 1.786 / 2.6535 | 0.4266; 1.1652 / 1.786 / 2.6535 | ✓ |
| `eng_zz_n4_gq1_s0` | g 2.6832, g² 7.1998 | 2.6832, 7.1998 | ✓ |

Value-level fingerprints for a direct CSV comparison are in Appendix B.

### 2.2 Other items verified

- **Self-test.** It reports seven checks with values consistent with the reference self-test. Differences in random draws explain the small numerical differences.
- **Feature counts.** 2n−1 (gs), n (te), 2n·|times| (hl), n (ql/eng), 5n−5 (gs φ), 6n−3 (te φ). All match the summary table (11, 6, 8, 4, 25, 33).
- **QProfiler defaults the notebook relies on [public-main]:**
  - The shipped `qsvc_args` is ZZ / linear / reps 2, so `ql_zz` at reps 2 is aligned with QSVC.
  - The shipped `pqk_args` is ZZ / pairwise / reps 4. For ZZFeatureMap, `pairwise` gives the same state as `linear` (verified to about 3e-15), so only `reps` is misaligned.
  - Overriding `pqk_args` with all four keys drops nothing.
- **Citations that are correct:**
  - Molteni–Gyurik–Dunjko, npj Quantum Information 12, 19 (2026).
  - Huang et al., "Power of data in quantum machine learning", Nat. Commun. 12, 2631 (2021).
- **Paired design across `ql_zz`, `ql_evo`, `eng`.** With the same seed, all three have **identical X**: the same first RNG draws, and the same `sum(X)` in Appendix B. They form a paired design (F20).

---

## 3. Findings

### F1 — Critical — The `eng` positive control failed, and the notebook doesn't flag it

**Observed (Arm B).** On `eng_zz_n4_gq1_s0`:

| Model | Accuracy |
|---|---|
| PQK | 0.444 (8/18) |
| QSVC | 0.500 |
| LR | 0.444 |
| RF | 0.389 |
| SVC | 0.444 |

**The notebook's own rules.** It states "If a quantum method cannot beat classical baselines here, the pipeline is misconfigured", and lists "it does not — check the encoder first" as the audit trigger. No cell checks either.

**Expected behaviour at the notebook's size.** Calibration on the regenerated `eng_zz_n4_gq1_s0` (n=4, N=60), 100 stratified 70/30 splits, PQK downstream model = QBioCode's `create_svc_model` search (C × gamma × kernel, 40 draws, 5-fold):

| PQK variant | Mean ± sd | P(acc ≤ 8/18) |
|---|---|---|
| Aligned (ZZ, linear, reps 2, raw x) | **0.798 ± 0.087** | **0.00** |
| Aligned, x MinMax-scaled per fold | 0.781 ± 0.093 | 0.00 |
| Shipped `pqk_args` (reps 4) | 0.546 ± 0.086 | 0.16 |
| Projections row-misaligned with labels (stale or mismatched cache) | 0.477 ± 0.144 | 0.44 |

The other models behave as expected (40 splits):

| Model | Expected | Notebook | Consistent? |
|---|---|---|---|
| LR | 0.38 ± 0.08 | 0.444 | ✓ (below chance by construction) |
| RF | 0.41 ± 0.10 | 0.389 | ✓ (below chance by construction) |
| SVC | 0.38 ± 0.10 | 0.444 | ✓ (below chance by construction) |
| QSVC | 0.49–0.51 | 0.500 | ✓ (eng is not engineered for the fidelity kernel) |
| PQK | ≈ 0.80 | 0.444 | ✗ **Only anomaly** |

**A same-run control.** In the same run, `ql_zz` PQK = 0.667.
- An aligned PQK gives 0.694 ± 0.101 with P(≥ 12/18) = 0.75.
- A reps-4 PQK gives 0.485 ± 0.088 with P(≥ 12/18) = 0.05.

So the `pqk_args` override most likely took effect for `ql_zz`. That points to something specific to the `eng` rows.

**Leading hypothesis (not confirmed).** A stale or misaligned PQK projection cache (see F2). Scaling is ruled out: MinMax still gives 0.78.

**Diagnose in this order**

1. In `ModelResults_encoded.csv`, read the `feature_map_reps` and `entanglement` fields of both `pqk` rows. Expect 2 and linear.
2. Delete the PQK projection directory (default `pqk_projections/` under the CWD in public `main`) and rerun Arm B only.
3. Load the cached `eng` train projection `.npy`. Compare it row-by-row with 1-local ⟨X⟩/⟨Y⟩/⟨Z⟩ recomputed from the CSV rows in the split order. They should agree to about 0.03 (1024-shot noise).
4. Compare the `eng` CSV with the fingerprint in Appendix B.

**Fix.** Add a gating cell after Arm B that fails loudly:

```python
eng = results_e[results_e.Dataset.str.startswith("eng_")].set_index("model").accuracy
assert eng["pqk"] >= 0.6 and eng["pqk"] > max(eng[["lr", "rf", "svc"]]), \
    "positive control failed: check pqk feature_map_reps/entanglement, clear pqk_projections/, rerun"
```

The 0.6 floor is about 2 sd below the aligned mean at n=4, N=60. Re-derive it for other sizes: at n=6, N=300, aligned PQK scored 0.91.

---

### F2 — Critical [public-main] — The PQK/QPL projection cache can silently reuse the wrong projections

**How the key is built.** `qprofiler.py` builds

```python
data_key = '_'.join([re.sub(r'\..*', '', file), embed, str(args["n_components"]), str(iter)])
```

**(a) Truncation at the first ".".** The regex removes everything after the **first** dot, not just `.csv`:

| File | Cache key stem |
|---|---|
| `te_n6_s4_seed0_tau0.25.csv` | `te_n6_s4_seed0_tau0` |
| `te_n6_s4_seed0_tau0.5.csv` | `te_n6_s4_seed0_tau0` ← **collides** |
| `hl_n4_g0.5_shots1000_s0.csv` | `hl_n4_g0` |
| `hl_n4_g0.5_shots100_s0.csv` | `hl_n4_g0` ← **collides** |
| `gs_sparse_n6_k0.5_s0.csv` vs `…k0.3…` | `gs_sparse_n6_k0` ← **collides** |

`compute_pqk` and `compute_qpl` add a feature-map fingerprint (encoding, primitive, entanglement, reps) but **no data content**. The only validation is row count and width, and datasets from the same generator call have the same N. Consequences:
- The notebook's statement "the shot count appears in the dataset name, so a sweep over it does not collide" is **false** for the PQK/QPL caches.
- It is true for the CSV files themselves.

**(b) No content, split, shots or backend in the key.** Any of the following reuses old projections for new data:
- regenerating a dataset under the same name and N (a generator fix, a changed parameter not in the name);
- changing `shots` or `backend`;
- a different train/test split.

The notebook's "re-runnable: start from nothing" cell deletes the data trees but **not the projection cache**.

**Scope.** QSVC does not cache (checked in `compute_qsvc.py`). PQK and QPL do.

**Fix** (§6 has patches):
- Use `os.path.splitext(file)[0]`.
- Add a SHA-256 of `X_train`/`X_test` bytes, plus `shots` and `backend`, to the cache fingerprint.
- In the notebook, delete the projection directories in the reset cell.
- In the generator, write `0p25` instead of `0.25` in names.

---

### F3 — High — The core comparison is never run

**What is missing.**
- **Arm A** runs `lr, dt, rf, nb, svc` only. `gs sparse` is labelled the negative control ("a quantum model winning clearly" is the trigger), but no quantum model ever runs on it.
- **`phi_view`** is generated for `gs`/`te` but never ingested. The φ–x gap is therefore never measured. That gap is the quantity that tracked the τ ladder in the reference smoke test: 0.00 → 0.20 as τ increases at n=10, N=400.

**Why it matters.** The originating question was whether QML methods beat classical ones on quantum data. The notebook does not test it for 3 of 5 families.

**Fix.** Add two arms. Both are feasible at the notebook's sizes (qubits = 11 / 6 / 8):
- **Arm A-q:** QSVC and PQK with a pinned encoder on `x_view` for `gs`/`te`/`hl`.
- **Arm C:** classical models on `phi_view` for `gs`/`te`.

Then report, for each `te` dataset, best(φ-view) − best(x-view).

---

### F4 — High — The `te` pre-registration is too strong

**The notebook's claims.**
- "The Walsh–Hadamard effective degree … grows with τ and every learner should degrade monotonically along it."
- Audit trigger: "flat or non-monotonic in τ".

**Evidence against per-instance monotonicity.** Setup: `audit_te.py`, n=8, N=200, 8 disorder/observable seeds, τ ∈ {0.25, 0.5, 1, 2, 4}, RF with 5-fold CV.

| Quantity | Result |
|---|---|
| Seeds with effective degree non-monotone in τ | **5 / 8** |
| Seeds where RF accuracy rises by > 0.02 somewhere along τ | **3 / 8** |
| Spearman(accuracy, τ), pooled over seeds × τ | **−0.82** |
| Spearman(accuracy, effective degree), pooled | **−0.75** |

The reference n=10 instance was also non-monotone in degree: 2.01, 2.13, 1.56, 2.55, 3.47. The notebook's own ladder is non-monotone too (RF 0.778 → 0.889 → 0.556), which by its stated rule is an audit trigger.

**Conclusion.** The trend is real at the population level, but it is not a per-instance law. Effective degree is a proxy: accuracy tracks τ at least as well as it tracks degree.

**Fix.** Pre-register a population statement instead:

> Over ≥ 8 (disorder, observable) draws at runsheet size: Spearman(accuracy, τ) < 0 with a bootstrap 95% CI excluding 0, for each model; the φ–x gap increases with τ in the same sense.

Per-instance non-monotonicity is expected and is not an audit trigger.

**Coverage.** The notebook's `te` uses n=6, N=60, which is **94%** of the 64 possible inputs. At that coverage, "generalisation" is close to interpolation over a truth table.
- Record `coverage = N / 2ⁿ` in the metadata.
- Warn above about 0.25.
- The runsheet size (n=10, N=400) gives 0.39; n=12 gives about 0.10.

**Ownership.** Part of this over-strong framing came from my earlier README, which said "all x-view learners degrade as degree grows". This finding corrects it.

---

### F5 — High — The `ql --encoding evo` expectation is wrong

**The notebook's claims.** Expected: "all learners mediocre". Audit trigger: "any learner excelling".

**Evidence.** The `evo` labels are misaligned with the quantum encoders, not hard for classical models.

| Setting | LR | RF | SVC | QSVC | PQK |
|---|---|---|---|---|---|
| Reference smoke (n=8, N=400, one seed) | — | 0.93 | 0.93 | 0.65 (FQK) | 0.69 |
| Calibration (n=4, N=60, 40 splits) | 0.73 | 0.73 | 0.76 | 0.73–0.75 | 0.50–0.56 |
| Notebook | 0.72 | 0.72 | 0.72 | 0.61 | 0.67 |

The notebook's numbers are consistent with calibration. At the runsheet size, its trigger would fire on correct behaviour.

**Fix.**
- Expected: classical ≥ PQK, and classical may be high.
- Trigger: PQK > classical with a CI excluding 0.
- Replace "evo matching zz" as the ql_zz trigger with an explicit statement:
  - On `ql_zz`, aligned QSVC/PQK > classical.
  - On `ql_evo`, PQK ≤ classical.

---

### F6 — High [public-main] — QProfiler's low-variance-feature metric: one bug and one design defect

This is the source of "No feature is strong enough to keep" (`qbiocode/evaluation/dataset_evaluation.py::get_low_var_features`):

```python
threshold = np.percentile(df.var(), 25)            # pandas: ddof=1
VarianceThreshold(threshold).fit(df)               # sklearn: np.nanvar, ddof=0
```

**(a) ddof mismatch.**
- The threshold is inflated by a factor N/(N−1) relative to the variances it is compared against.
- When feature variances lie within that factor of each other, every feature fails and `ValueError` is raised. That happens for balanced binary features (`te` bits) and for standardized data.
- The metric then becomes `None`.
- Reproduced on te-like data (N=60 distinct 6-bit inputs): pandas variances 0.2531–0.2542, threshold 0.25332, sklearn variances 0.2489–0.2500 → error. With a consistent ddof, 4 features are kept.
- A standardized 10-feature dataset also returns `None`.

**(b) Tautological by construction.**
- The threshold is the 25th percentile of the variances themselves.
- So the count of features at or below it is about n/4, **whatever the data**.
- Over 20 datasets per size with very different variance profiles, the metric took only the values 2 (n=8), 5–6 (n=20), 25–26 (n=100) and 3–4 (n=11).

**Why it matters.** QSage regresses ΔF1 on these metrics. This one is either missing or a relabelled feature count.

**Fix.**
- Use a consistent `ddof`.
- Use a data-relative absolute criterion, for example variance < 10⁻³ × median variance on the unscaled data, or drop the metric.
- Worth checking the other `dataset_evaluation` metrics for the same pattern while the QSage metrics are being redesigned.

---

### F7 — Medium — One mis-citation and one mis-description

**`hl` cites Huang et al., *Science* 376, 1182 (2022)** as "the shape of the learning-observables problem".
- That paper is "Quantum advantage in learning from experiments".
- It proves that quantum machines with quantum memory can learn from exponentially fewer experiments than **conventional experiments, which measure the system and post-process classically**.
- `hl` *is* the conventional side: measure-first, then classical post-processing.
- Molteni et al. (Table I) also mark identification-type Hamiltonian learning as classically easy.

Suggested replacement:
> "Hamiltonian learning as classification — the conventional-experiment baseline (measure, then process classically). Expected classically easy: at short times ⟨Z_i(t)⟩ ≈ 1 − 2h_i²t²."

For predicting properties from classical measurement data, the relevant reference is Huang, Kueng, Torlai, Albert, Preskill, *Science* 377, eabk3333 (2022). That fits the `gs` family rather than `hl`.

**`gs` is described as "a disordered transverse-field Ising Hamiltonian".** The generator adds −κ Σ XᵢXᵢ₊₁ with κ = 0.5, which makes it interacting and not free-fermion. Say so: that is exactly why κ ≠ 0 was chosen.

---

### F8 — Medium — "The data-complexity measures explain a lot of the accuracy ordering" is unsupported

**From the notebook's own tables (n = 9 datasets):**

| Measure | Evidence |
|---|---|
| Fisher Discriminant Ratio vs mean accuracy | Spearman ρ = 0.48, p = 0.19 |
| Fisher Discriminant Ratio vs max accuracy | Spearman ρ = 0.38, p = 0.31 |
| Fractal dimension | Constant (1.990–1.997) |
| Intrinsic dimension | Equals the feature count except for `hl` |

**Structural reason.** FDR is a between-class-means statistic. It cannot see labels defined by parities or interactions, which is exactly the `te` structure.

**Fix.** Remove the sentence, or replace it with a computed correlation and CI at the runsheet size.

---

### F9 — Medium — Statistical reporting

**Test-set sizes.** Each number is one 70/30 split with `iter=1`, so 18 or 24 test rows:

| Accuracy | Wilson 95% CI |
|---|---|
| 8/18 = 0.444 | [0.25, 0.66] |
| 13/18 = 0.722 | [0.49, 0.88] |
| 22/24 = 0.917 | [0.74, 0.98] |

**Multiple models.** With 5 models, the chance that at least one reaches ≥ 13/18 on a pure-noise label is about 0.22. This treats the models as independent, which is only approximate because they share a split.

**Fix.**
- Use `iter ≥ 10` (different split seeds) and report Wilson or bootstrap CIs.
- Use paired comparisons: same split across models, and same X across `ql_zz`/`ql_evo`/`eng` (F20).
- Add a label-permuted copy of every dataset as a pipeline null (§4). This matches the program's standing two-point-test protocol.

---

### F10 — Medium — `eng` expectations are under-specified

"Quantum kernel wins" should say which kernel wins.
- The labels were engineered against **RBF(γ_Q = 1) on 1-local Bloch vectors of the ZZ state**, which is PQK's hypothesis class.
- QSVC/FQK is not targeted and is expected near chance (0.49–0.51 at n=4, N=60).
- Classical models are expected **below** chance (0.38–0.41), because the labels are built along the direction where the classical kernel generalises worst. A below-chance classical score is a signature of a correctly generated `eng` set, not noise.

**Fix.** Add these numbers and the F1 gate to the family table and the text.

---

### F11 — Medium [public-main] — Asymmetric tuning and encoder search

**QSVC is untuned by default.** With `tune_quantum: False` (the default), QSVC runs at `qsvc_args` C = 0.01, while classical models are tuned. The measured effect here is small:

| Dataset | QSVC C = 0.01 | QSVC C tuned |
|---|---|---|
| ql_zz | 0.725 | 0.747 |
| eng | 0.494 | 0.508 |

It is still an asymmetry in a comparison benchmark.

**Turning tuning on breaks alignment.** With `tune_quantum: True`, `gridsearch_qsvc_args` and `gridsearch_pqk_args` also search encoding {Z, ZZ}, reps {1, 2} and entanglement {linear, full}. That can un-align a positive control. The search space also excludes the shipped PQK default (reps 4).

**Fix.** For aligned controls, pin the encoder and tune only C (plus PQK's downstream SVC).

---

### F12 — Medium — ⟨r⟩ at n = 6 is uninformative

**What the notebook does.** It prints ⟨r⟩ = 0.4266 and calls H "non-integrable".

**Spread across 10 disorder seeds (same construction):**

| n | ⟨r⟩ range |
|---|---|
| 6 | 0.40–0.62 |
| 8 | 0.50–0.59 |
| 10 | 0.50–0.56 |

**Fix.** Report ⟨r⟩ only for n ≥ 10, or averaged over seeds. Describe H as "a generically non-integrable mixed-field Ising chain" rather than implying the diagnostic establishes it.

---

### F13 — Low — `phi_view` is not always an "upper bound"

It is an oracle ceiling only when the label is linear in the pool (`gs sparse`, `te`). For `gs e2e`, the label observable Z₀Z₍ₙ₋₁₎ is deliberately excluded from the pool, so φ-view is not an upper bound there.

Suggested wording: "oracle-feature reference; a ceiling only when the label observable is in the pool."

---

### F14 — Low — The `hl` shots sweep is not run

The audit trigger "no dependence on shots" needs a sweep that reaches low shot counts, for example `shots ∈ {0, 1000, 100, 30, 10}`.

At t = 0.5, the across-row spread of ⟨Zᵢ⟩ in the regenerated `hl_n4` set is: std 0.15–0.22 per qubit, range 0.17–0.94. Shot noise at M = 1000 is 1/√M ≈ 0.03. I therefore *expect* little change until M ≲ 100, but I have not measured it. A flat curve over {0, 1000} alone would be uninformative, not a trigger.

---

### F15 — Low — Rendering defects

- **Admonitions.** The MyST blocks `{warning}`, `{note}`, `{important}` are indented, so they render as code blocks with raw `$…$`. Use fenced ```` ```{warning} ```` (Jupyter Book) or `> **Warning**` blockquotes.
- **Step-4 feature-count table.** It collapsed onto one line because its rows need newlines.
- **Lost math.** In the non-claims, "qubits is a regime…" is missing "n ≤ 12".

---

### F16 — Low — `gs` covers only the easy regime

- The minimum even-sector gap is 2.28 (sparse) and 2.58 (e2e) at n=6. Every row is deep in a gapped regime, which is the side where classical ML is provably efficient.
- If the harder regime is wanted, add a near-critical variant: narrow `h/J` around the finite-size crossover and report the gap histogram.
- Keep κ ≠ 0. At κ = 0 the model is free-fermion, so a classical twin exists at any n.

---

### F17 — Low — Silent failures

- **Empty folder.** QProfiler pointed at the parent directory finds nothing and reports nothing. The notebook documents this; the code should enforce it with `if not input_files: raise FileNotFoundError(...)`.
- **Empty `phi_view/`.** Don't create it for `hl`/`ql`/`eng`. An empty directory is a trap for anyone who points `folder_path` at it.

---

### F18 — Low — Dataset naming

- **`reps`/`entanglement` missing from names.** The notebook documents this, but documentation is weaker than a fix. Encode them automatically, for example `ql_zz_r2lin_n4_tau1_s0`, or append a short hash of `quantum_args`.
- **Dots in names.** Replace `.` with `p` (`tau0p25`, `k0p5`, `g0p5`). This also defuses F2(a) until QProfiler is patched.

---

### F19 — Info — "A reps mismatch alone is enough to remove the advantage" now has multi-split support

At n=4, N=60, going from the aligned encoder to reps 4 moves:
- `eng` PQK from 0.80 to 0.55 (100 splits);
- `ql_zz` PQK from 0.69 to 0.49 (60 splits).

Each rests on one dataset draw.

---

### F20 — Info — `ql_zz`, `ql_evo` and `eng` share identical X

With the same `random_state`, X is drawn first and is identical across the three families. That is useful:
- Label-only contrasts can be tested paired, on the same split.
- The three datasets are not independent samples. Don't count them as three independent confirmations.

---

## 4. What is missing

| Item | Why |
|---|---|
| Quantum models on `gs`/`te`/`hl` x-view (Arm A-q) | The core question (F3) |
| `phi_view` ingestion and the φ–x gap per τ (Arm C) | The quantity that carries the Molteni-type structure |
| Label-permuted null for every dataset | Pipeline chance calibration; matches the program's two-point protocol |
| `iter ≥ 10`, CIs, paired tests | F9 |
| A positive-control gate before any benchmark run | F1: nothing downstream is interpretable if `eng` fails |
| Multi-seed (disorder, observable) draws per family | F4: per-instance behaviour is not the pre-registered object |
| Runsheet-size runs (n=6–10, N=300–400) | All predictions refer to those sizes |
| `coverage = N/2ⁿ` in `te` metadata, with a warning | F4 |
| `hl` shots sweep | F14 |
| Near-critical `gs` variant | F16 |
| Label-aware complexity features for QSage | F6, F8. Add: Walsh degree profile (boolean inputs), kernel-target alignment and geometric difference g for the saved PQK/FQK/classical kernels. Hold out whole families in the QSage regression so family identity cannot proxy for complexity |

---

## 5. Corrected expectations table (replaces the notebook's)

Numbers are calibration from this audit (exact simulation, one dataset draw, repeated splits). Treat them as bands to be re-derived at runsheet size, not as results.

| Family | Role | Expected | Audit trigger |
|---|---|---|---|
| `gs --label sparse` | Negative control | Best classical ≥ best quantum on x-view | Quantum > classical with CI excluding 0 → construct the classical twin; check for leakage |
| `gs --label e2e` | Classical twin known | Product criterion ≈ 0.8+ held-out; LR high | Classical at chance |
| `te` | Difficulty ladder | Population Spearman(accuracy, τ) < 0; φ–x gap grows with τ | Population trend ≥ 0; φ-view < x-view |
| `hl` | Conventional-experiment baseline | Classical high; roughly flat until low shots | Classical at chance with `shots=0` |
| `ql --encoding zz` | Aligned positive control | n=4, N=60: QSVC ≈ 0.73–0.75, PQK ≈ 0.68–0.71, classical ≈ 0.45–0.48 | Aligned quantum ≤ classical |
| `ql --encoding evo` | Misaligned control | Classical ≥ PQK; classical may be high (0.93 at n=8, N=400) | PQK > classical with CI excluding 0 |
| `eng` | PQK positive control | n=4, N=60: PQK ≈ 0.80 ± 0.09, QSVC ≈ 0.5, classical ≈ 0.38–0.41 (below chance by design); n=6, N=300: PQK ≈ 0.91 | PQK < 0.6 → stop; run the F1 diagnostics |
| any, label-permuted | Pipeline null | Every model ≈ 0.5 | Any model > 0.6 |

---

## 6. Patches

### 6.1 QProfiler `data_key` [public-main, `qbiocode/apps/qprofiler/qprofiler.py`]

```python
# before:  re.sub(r'\..*', '', file)   -> truncates at the FIRST dot
data_key = '_'.join([os.path.splitext(file)[0], embed, str(args["n_components"]), str(iter)])
```

### 6.2 Content-addressed projection cache [public-main, `compute_pqk.py` and `compute_qpl.py`]

```python
data_digest = hashlib.sha256(
    np.ascontiguousarray(X_train, dtype=np.float64).tobytes()
    + np.ascontiguousarray(X_test, dtype=np.float64).tobytes()
).hexdigest()[:16]
feature_map_fingerprint = hashlib.sha256(repr((
    ("model", model), ("encoding", encoding), ("primitive", primitive),
    ("entanglement", entanglement), ("reps", int(reps)),
    ("shots", args.get("shots")), ("backend", args.get("backend")),
    ("data", data_digest),
)).encode("utf-8")).hexdigest()[:10]
```

### 6.3 Low-variance metric [public-main, `dataset_evaluation.py`]

```python
def get_low_var_features(df, num_features, rel_tol=1e-3):
    """Count features whose variance is negligible relative to the median feature variance."""
    v = df.var(ddof=0)
    med = float(np.median(v))
    return int((v <= rel_tol * med).sum()) if med > 0 else int(num_features)
```

### 6.4 Notebook reset cell

```python
for root in (XVIEW_ROOT, ENCODED_ROOT, Path("pqk_projections"), Path("qpl_projections")):
    shutil.rmtree(root, ignore_errors=True)
```

Use whatever `pqk_projection_dir` / `qpl_projection_dir` your config sets. The public defaults are relative to the CWD.

### 6.5 Generator naming

```python
def _fmt(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")   # 0.25 -> 0p25
# and include reps/entanglement for encoder families, e.g. f"ql_{enc}_r{reps}{ent[:3]}_n{n}_tau{_fmt(tau)}_s{seed}"
```

---

## 7. Notebook edits (cell by cell)

- **Header table.** Fix the `gs` description (interacting, κ XX term). Fix the `hl` citation (F7).
- **Warning admonition.** Fix the rendering (F15). Add "positive control must pass before any other number is read".
- **§1b (`te`).** Replace the monotonicity sentence with the population statement (F4). Print `coverage`. Drop or caveat ⟨r⟩ at n=6 (F12).
- **§1c (`hl`).** Fix the citation. Add a shots-sweep cell (F14).
- **§1d (`ql`).** Change the `evo` expectation (F5).
- **§1e (`eng`).** Name PQK as the targeted kernel. State the expected bands, including below-chance classical scores (F10).
- **Arm A.** Add `qsvc`/`pqk` with a pinned encoder (F3).
- **New Arm C.** `phi_view` for `gs`/`te`; compute the φ–x gap table (F3).
- **Arm B.** Add the F1 gate. Print `feature_map_reps`/`entanglement` from `ModelResults` for the quantum rows.
- **Step 4.** Remove the complexity-measures sentence, or compute it (F8). Fix the feature-count table (F15). Add Wilson CIs to every accuracy (F9).
- **Every arm.** Add a label-permuted null dataset (§4).
- **Expectations table.** Replace with §5.

---

## 8. Additions to the non-claims ledger

- **te.** The ladder is a population trend. Individual instances need not be monotone in τ or in effective degree.
- **Level-spacing ratio.** ⟨r⟩ at n < 10 does not establish non-integrability.
- **eng.** A PQK win shows that the pipeline reproduces an engineered kernel alignment. It says nothing about quantum hardware or real data. Classical below-chance scores on `eng` are by construction.
- **ql_evo.** High classical accuracy there is expected and is not a failure.
- **Low-variance metric.** In `RawDataEvaluation` it is ≈ n/4 by construction (until patched) and must not enter QSage.
- **Cached PQK/QPL results.** Results produced before the cache-key fix (F2) should be treated as unverified wherever two dataset names share a prefix up to their first ".". This matters beyond this notebook: it may also affect the ~800K-experiment corpus if any filenames contained dots.

---

## 9. Second-pass code audit (generator implementation)

Sections 1–8 audit the *claims* — the science, the notebook's expectations, the
statistics. This section audits the *code* of the eight new modules line by line
(`quantum_core.py`, the five `make_*` families, `quantum_cli.py`,
`quantum_selftest.py`) plus the diffs to the four modified library modules. It is a
separate pass with a separate method: read every line, then verify each suspicion
numerically before calling it a defect.

The governing constraint is the integration plan's **"preserve numerics exactly"**
mandate. The shipped corpus under `qdata/` was generated by this code, so where the
code and its documentation disagree, *the code is authoritative* and the
documentation is what gets corrected. Six defects were found; five were fixed, and
one fix was subsequently reverted on fidelity grounds.

### D1 — `evo_encoding_state`'s documented operator order was reversed

The docstring said `exp(-i t Σ x_i Z_i) exp(-i t Σ X_i X_{i+1})` — XX evolution first.
The code is `psi = Uxx @ (zdiag * psi)`, which applies the data-dependent Z phase
first. Verified against `scipy.linalg.expm`:

```
|| got - as_documented || = 0.223
|| got - as_coded       || = 4.0e-16
```

Fixed the **docstring**, not the code: the code defines the shipped `ql --encoding evo`
dataset. A reader implementing the encoding from the docstring would have produced a
different feature map and drawn the wrong conclusion about `ql_evo`.

### D2 — `make_time_evolution` validated its row count too late

`n_samples > 2**n` was checked inside the per-configuration write loop, so
`n_qubits=[10, 4]` wrote the five tau datasets of the first configuration and only
then raised — leaving a half-written sweep on disk. This was inconsistent with
`ensure_unique_names`, which is deliberately hoisted above the loop for exactly this
reason. Hoisted the guard to match. `te` is the only family that needs it: only it
draws basis states without replacement.

### D3 — the `gs` `e2e` classical-twin diagnostic scored rows that were never shipped

The guard tested `X[keep] > 0` but computed `z`, `y` and `half` from the *unfiltered*
arrays. Two consequences: a row dropped by `margin` could still hand `np.log` a
non-positive coupling — the precise silent NaN the guard existed to prevent — and the
reported `product_criterion_holdout_acc` was measured on rows absent from the
dataset, so it was not comparable with a model's accuracy on the shipped CSV. Now
scored on `X[keep]`, `y[keep]`.

Verified safe for the shipped corpus: at `margin=0`, `threshold` returns an all-True
`keep` (`|F - thr| >= 0` always; checked over 200 random draws plus a tie exactly at
the median), and `gs_e2e_n8_k0.5_s0.json` records `margin = 0.0`. The shipped
`product_criterion_holdout_acc = 0.87` is therefore unchanged.

### D4 — `write_dataset` always created `phi_view/`

`hl`, `ql` and `eng` document "only `x_view` is written", but the writer created
`phi_view/` unconditionally. Beyond the contradiction, an empty view directory is
actively harmful: QProfiler discovers CSVs with a non-recursive `os.listdir` +
`endswith('csv')`, so an empty `phi_view/` reads as *a corpus containing no datasets*
rather than as an error — a silently empty sweep. Now created only when `phi` is given.
Confirmed on disk: `hl`/`ql`/`eng` write `['meta', 'x_view']`, `gs` writes
`['meta', 'phi_view', 'x_view']`.

### D5 — `E741` in `make_quantum_labels.py`

`O = [Pauli(n, {0: "Z"})]` — flake8 `E741 ambiguous variable name`. Renamed to `obs`,
matching `make_hamiltonian_learning.py`. This was the only lint finding in new code
across the whole package; every other `F401`/`F841` under `data_generation/` is
pre-existing in the shipped classical generators.

### D6 — reverted: `zeros_like` → `empty_like` in `expvals`

`PS` is fully overwritten by a permutation (`PS[tgt] = ...`), so `empty_like` is
*logically* safe and skips a `calloc`. It was measurably the wrong trade. Regenerating
the corpus showed the `ql`/`eng` `meta/*_F.npy` diagnostics shifted by ~1e-14 — and
reverting to `zeros_like` cut that to ~1e-16, because the `calloc`'d buffer's
alignment is what the shipped diagnostics were computed against. Reverted; the
permutation assumption it documented is retained as a comment, since an operator with
a non-bijective target would otherwise contribute silent zeros to the sum.

### D7 — twelve tests hard-failed instead of skipping without the `[tabpfn]` extra

`tests/test_classical_models.py` carries `tabpfn` in `CLASSICAL_MODELS`,
`UNWRAPPED_MODELS` and `RANKING_MODELS` but consulted none of conftest's
`tabpfn_ready` / `tabpfn_skip_reason` probes. `tabpfn` is its own extra and `[dev]` does
not pull it in, so on the exact install `.github/workflows/ci.yml` performs —
`pip install -e ".[dev]"` — twelve nodes reached `compute_tabpfn` and raised
`ImportError` rather than skipping. None carries a `slow` or `requires_quantum` mark, so
the default `-m` filter did not deselect them, and the `Run tests` step has no
`continue-on-error`.

Measured by shadowing `tabpfn` on `sys.path` to reproduce a CI install:

```
12 failed, 115 deselected in 149.37s
FAILED ...TestEveryClassicalModelFits::...[tabpfn]              (8 nodes)
FAILED ...test_each_compute_function_labels_its_own_columns...[tabpfn-tabpfn]
FAILED ...TestTheRecordedParameterSchema::...[tabpfn]
FAILED ...TestTheAucColumnIsARankingAuc::...[tabpfn]            (2 nodes)
```

That count survives the instrument change described under *A wrong simulation of the
missing extra* below, and this is worth being precise about, because the 22-failure number
in that section did not. `compute_tabpfn` raises `ImportError` in **both** simulations —
under the shadow file from `importlib.import_module("tabpfn")` at `compute_tabpfn.py:241`,
under the faithful blocker from the guard at `:222` — so the same twelve nodes fail either
way and only the message differs. What the flawed shadow invented was the *extra* failures
in the four other files, where the code does consult the gate and would have taken the
skip branch.

This is pre-existing on `origin/aritra/v2` and independent of this port; it is fixed
here because it otherwise masks the port's own results. The fix follows the idiom the
suite already uses at
`tests/test_model_run_edges.py::test_a_direct_compute_call_is_not_covered_by_the_dispatcher`
— request the probe through `getfixturevalue` for the `tabpfn` node only, so the other
eight models do not pay for conftest's subprocess fit — as a single autouse fixture
keyed on the `model` parameter, which covers all three parametrizations at once.

Not a defect, checked and cleared: `xgb` and `qpl` have no `gridsearch_*_args` block in
the packaged `config.yaml`, but neither is in its default ten-model `models` list, and
both paths are deliberate and documented — `_model_args` (`model_run.py:345`) falls back
to estimator defaults and *logs* that it did, and the tuning path raises an actionable
`ValueError` naming the block to add and the recognised hyperparameters
(`_tuning.py:203`). Every one of the ten selected models has both an `_args` and a
`gridsearch_*_args` block.

### D8 — the same missing-extra failure in three more files

Re-running the no-`[tabpfn]` tier under the *corrected* blocker left **6 failures and 7
errors** that D7's fix does not reach, all of them the same defect in three more files:

| File | Nodes | Shape of the defect |
|---|---|---|
| `tests/test_tuner_engine_selection.py` | 3 failed | `tabpfn` is a row of the `RANGE_BLOCKS` table consumed by three parametrizations (`:217`, `:244`, `:326`) |
| `tests/test_opt_twins_dispatch.py` | 2 failed | `tabpfn` is a row of `CLASSICAL_BLOCKS`, consumed at `:341` and `:418` |
| `tests/test_model_contract_matrix.py` | 1 failed, 7 errors | `tabpfn` is in `CLASSICAL_KEYS`, which both the `sequential` fixture's model list *and* the expected-label sets are derived from |

The failure modes differ in an instructive way. In `test_tuner_engine_selection.py` the
grid half asserts `pytest.raises(ValueError)`; the absent extra makes
`compute_tabpfn_opt` raise `ImportError` from the gate before the engine branch is
reached, so the wrong exception type arrives and the test reports nothing about engine
selection — the thing the file exists to measure. In `test_model_contract_matrix.py` the
damage is worse than one node: the `sequential` fixture is module-scoped, so its
`ImportError` becomes a *setup error* for all seven tests that request it.

**The policy question, and why it was not decided by preference.** The file being fixed
carried a comment asserting the opposite:

> TabPFN carries no skip guard on purpose. QBioCode pins model version v2, whose weights
> are ungated — no API token and no license acceptance — so a tuned TabPFN either
> dispatches or the `[tabpfn]` extra is missing, and the latter is a broken dev install
> rather than a condition to tolerate quietly.

Half of that is correct and half of it conflates two independent gates. The **license**
gate genuinely does not apply: `TABPFN_DEFAULT_VERSION = "v2"` and
`_RESTRICTED_VERSIONS = frozenset({"v2.5", "v2.6", "v3"})`, so nothing here is gated on
acceptance. The **extra** gate is a separate question, and the repository answers it
itself in three places:

- `requirements/requirements-base.txt:19-22` — *"tabpfn is NOT here either, but for a
  different reason: weight. It brings torch's ecosystem plus mlx, lightgbm,
  huggingface-hub and safetensors, and `import qbiocode` must not pull torch in."*
  Deliberate exclusion from the core dependencies, stated as such.
- `tests/test_mps_backend.py:22` — `pytest.importorskip("quimb", reason="MPS backend
  requires the optional quimb dependency")`. An absent **optional extra** is skipped.
- `tests/test_openmp_import_order.py:42` — *"a missing torch or xgboost is a broken
  install, and these tests…"*. An absent **base dependency** is not skipped.

So the house rule is base-dependency versus optional-extra, and `[tabpfn]` is the latter,
exactly as `[mps]` is. The same criterion is already written down elsewhere in the very
file that denied it: the `DEFAULT_QPL_HEADS` comment (`:884`) justifies `xgb` and
`catboost` carrying no guard on the grounds that *"both are declared in
requirements-base.txt, so neither is absent in any supported install"* — which is the
right test, and which `tabpfn` fails. Both comments were corrected to say so rather than
left to contradict the code.

And the decisive practical point: `[dev]` does not pull in `[tabpfn]`, and
`.github/workflows/ci.yml:33` installs `pip install -e ".[dev]"`. TabPFN is absent on
**every** CI run, so "the extra is missing" is not a broken dev install — it is the only
state CI is ever in.

**The fix**, one gate per file, on the extra's presence at collection time:

```python
from qbiocode.learning.compute_tabpfn import tabpfn_is_available
...
RANGE_BLOCKS = [b for b in _ALL_RANGE_BLOCKS if b[0] != "tabpfn" or tabpfn_is_available()]
```

Filtering the table rather than marking the node with `skipif` is what
`test_model_contract_matrix.py` requires and what the other two then match: its
assertions are label-set **equalities** (`set(sequential) == {f"{prefix}_{key}" ...}`)
derived from `CLASSICAL_KEYS`, so a key the run cannot produce has to leave the
expectation as well, or the equality fails for a reason that is not the run's fault. The
file already had the pattern one line further down — `PARALLEL_KEYS = [k for k in
CLASSICAL_KEYS if k != "tabpfn"]`, excluded from the fan-out for an unrelated libomp
reason — so the fix mirrors an idiom that was already there.

Scope of the gate, stated so it is not mistaken for more: it is `tabpfn_is_available()`,
the extra's **presence**, which is the condition that differs between installs and the
one that breaks CI. It is not conftest's `tabpfn_ready`, which additionally probes a real
fit and so also covers unreachable weights. D7's fix does use `tabpfn_ready`, because
`test_classical_models.py` asserts on scores and needs a fit that actually succeeded;
these three assert on dispatch and engine routing, for which importability is the whole
requirement. Where a later test needs the stronger condition, the fixture exists and the
comments point at it.

Verified in both directions on the six TabPFN-touching files: 125 tests collected with
the extra installed, 120 with the blocker active — the difference being exactly the five
`[tabpfn]` parametrized nodes — and `tabpfn_is_available()` returning `True` and `False`
respectively.

**Pre-existence, established by inspection rather than by a second run.** Before today's
edits `git diff origin/aritra/v2` was empty for all three test files *and* for the entire
causal chain behind them — `qbiocode/learning/compute_tabpfn.py`, `tests/conftest.py` and
`requirements/requirements-tabpfn.txt`. The 13 failures were therefore produced by
unmodified upstream code, and the diff for the three files is now additive gating only
(71 insertions, 10 deletions, the deletions being the two renamed table bindings and the
corrected comment).

### D9 — two generators keyed their provenance index to filenames they never wrote

Unrelated to this port, found while checking that the new generators' `dataset_config.json`
round-trips. `dataset_config.json` is the provenance index that maps each generated CSV
back to the parameters that produced it. `make_circles.py` wrote its CSVs as
`circles_data-N.csv` but keyed the index `ld_data-N.csv`; `make_class.py` wrote
`class_data-N.csv` and keyed it `hd_data-N.csv`. Both were copy-paste residue from
`make_ld_data.py` / `make_hd_data.py`, and both make the index unjoinable to the data it
describes — every lookup misses, silently.

Reproduced empirically before changing anything (generate, then compare the index's keys
against `os.listdir`), confirmed that no test pinned the wrong keys, fixed both, and
re-verified that all **seven** generators now round-trip with index keys equal to the CSVs
on disk. Two dead stores went with them: `new_dataset = dataset.to_csv(path)` in each
file, where `to_csv` with a path argument returns `None`.

### D10 — the correlation figure's size legend dropped a swatch and mislabelled the rest

Found from a warning, not a failure. `tests/test_plotting_hygiene.py` passes, but every
run of it emitted

```
UserWarning: Mismatched number of handles and labels: len(handles) = 3 len(labels) = 4
  legend = ax.legend(
```

from `visualize_correlation.py:627`. Three lines above, the code asks matplotlib for four
size swatches and then *unconditionally* builds four captions for them:

```python
handles_size, labels_size = scatter.legend_elements(prop="sizes", num=4, ...)
smin, smax = np.min(data[size]), np.max(data[size])
labels_size = [f"{x:.2f}" for x in np.linspace(smin, smax, 4)]
```

`num=4` is a *target*, not a guarantee — `legend_elements` hands it to a tick locator,
which returns however many round levels the observed size range admits, and for the
metrics this figure plots that is routinely three. The consequence is worse than the
missing fourth entry that matplotlib warns about: with three handles and four captions,
matplotlib pairs them positionally and discards the tail, so the three swatches that *do*
render are captioned `smin`, `smin + Δ`, `smin + 2Δ` from a four-point grid — i.e. the
legend states the wrong marker sizes, silently, in a figure whose whole job is to let a
reader size-decode the scatter.

Fixed by sizing the caption list from the handles actually returned:

```python
labels_size = [f"{x:.2f}" for x in np.linspace(smin, smax, len(handles_size))]
```

The warning is gone and `tests/test_plotting_hygiene.py` still passes 17/17. This keeps
the author's intent — real metric values rather than matplotlib's internal `s` values —
and only corrects the count. It does not make the captions exactly right: `linspace`
endpoints are not the "nice" levels the locator picked, so the middle captions remain
approximations of their swatches. Making them exact means inverting the size norm the
scatter used, which is a design change to a plotting helper this port does not otherwise
touch, so it is recorded here rather than done. `visualize_correlation.py` is a
pre-existing file, and this bug is pre-existing on `origin/aritra/v2`.

### D11 — the physics self-test could report PASS having checked nothing

`qdata-gen selftest` is the entry point the plan designates as "the real regression guard
on the physics": seven checks covering the Pauli algebra, the sparse-vs-dense Hamiltonian,
the even-sector ground state, the Walsh transform, the short-time quench expansion, the
engineered-label bound, and the native-vs-Qiskit `ZZFeatureMap`. Each of the eleven
comparisons inside them is a bare `assert`, and `run_selftest` reports PASS for any check
that returns without raising `AssertionError`.

`python -O` removes `assert` statements from the bytecode entirely. So under `-O` — or with
`PYTHONOPTIMIZE` set in the environment, which is how it usually happens — every check
returns its measured quantities regardless of what they are, and the CLI prints

```
SELFTEST PASS
```

having verified nothing at all. That is the worst available failure mode for a self-test:
silent, and indistinguishable from success.

Two fixes were possible. Rewriting all eleven `assert X, msg` into
`if not X: raise AssertionError(msg)` works, but costs the `assert` form that
`pytest` introspects, in eleven places, to defend against a flag nobody normally passes.
Instead `run_selftest` now refuses outright:

```python
if not __debug__:
    raise RuntimeError(
        "run_selftest() cannot run under python -O: the checks assert their "
        "results, and -O strips assert statements, so every check would pass "
        "vacuously. Re-run without -O (or unset PYTHONOPTIMIZE)."
    )
```

Two lines, no change to any check, and a loud refusal replaces a false PASS. Verified both
ways: `python -O` raises that `RuntimeError`, and
`python -m qbiocode.data_generation.quantum_cli selftest` still reports all seven checks ok
(max Pauli error 1.4e-16, sparse-vs-dense exactly 0, ground-state residual 5.6e-15, FWHT
round trip 3.3e-16, quench expansion 2.3e-6 at t=0.02, `s_Q = 1.000000`,
`s_C/g² = 1.000000`, max ZZ infidelity 3.0e-15 over 36 comparisons) and exits 0.

### D12 — every JSON sidecar was written at the process locale's encoding

All nine `json.dump` call sites under `data_generation` — `quantum_core.write` plus the
eight classical generators' `dataset_config.json` writers — opened their file as
`open(path, "w")` with no `encoding`, so the bytes depended on
`locale.getpreferredencoding()`. Under `LC_ALL=C`, which is common in batch schedulers and
container images, that is ASCII, and a single non-ASCII character anywhere in a dataset
name or a label-rule string would abort generation with `UnicodeEncodeError` after the CSVs
had already been written — leaving a corpus with data but no provenance.

All nine now pass `encoding="utf-8"` explicitly. This **cannot** change a single byte of
existing output: `json.dump` defaults to `ensure_ascii=True`, so every sidecar the package
has ever written is pure ASCII, and ASCII is a subset of UTF-8. Demonstrated rather than
argued — the eight classical generators were re-run after the change and compared against
the pre-change tree: all 21 CSVs and all 8 `dataset_config.json` files byte-identical.

### D13 — the CLI's five-way dispatch was asserted nowhere, and its flags had two spellings

The coverage run (T4) put `quantum_cli.py` at **0% — 56 statements, every one missed**,
while the five family modules sat at 98% and `quantum_core.py` at 97%. (Re-run on the fixed
tree as job `970157`'s B2, the same module reads **100%, 0 missed**; §10.4b has the table.) Two separate causes,
and only one of them is a measurement artefact.

The artefact: `tests/integration/test_cli_smoke.py` exercises every console script through
`subprocess.run([sys.executable, "-c", code])`. A child process started that way is not
traced by the parent's `--cov`, so the CLI's statements read as unexecuted even though they
ran. That part of the number is not a gap.

The gap: **no in-process test imported `quantum_cli` at all.** The smoke test asserted only
that `qdata-gen --help` exits 0, and `.github/workflows/ci.yml` adds exactly one line for
this branch — `qdata-gen --help > /dev/null`. So the part of the CLI that can actually be
wrong was unchecked: `main()` translates argparse destinations into generator keyword
arguments by hand (`--n` → `n_qubits`, `--N` → `n_samples`, `--seed` → `random_state`,
`--out` → `save_path`, `--s` → `n_terms`, `--te_w` → `disorder`, `--g` →
`longitudinal_field`, and `--J_lo/--J_hi/--h_lo/--h_hi` into two tuples), then routes five
families through an `if/elif` chain. A transposed pair there produces a CLI that runs, exits
0, writes a plausible CSV, and silently generates the wrong dataset. `--help` catches none
of it.

`tests/test_quantum_cli.py` (43 tests) closes it in-process: it loads the module by path
under its real dotted name so the relative imports resolve, replaces all five generators with
recorders, and asserts for each family that **exactly one** generator is called and that
every flag arrives under its documented keyword. It also covers the two branches nothing
else reaches — the `--blas-threads 0 → None` sentinel, and `selftest` **failing**, which is
the only path that returns 1.

Writing those tests surfaced a second, smaller defect. The CLI had grown two naming
conventions for multi-word flags: six inherited from the standalone generator use
underscores (`--J_lo`, `--J_hi`, `--h_lo`, `--h_hi`, `--te_w`, `--gamma_q`) and the two
added during the port use dashes (`--data-map`, `--blas-threads`). This was **not** a
documentation bug — every invocation in `README.md`, `docs/source/quantum_datasets.md` and
the notebook uses the spelling its flag actually accepts, and that was checked flag by flag
before anything was changed. It is an internal inconsistency with a user-visible cost: a
flag that appears in `--help` rejects the other convention with `unrecognized arguments`.
All eight flags now declare both spellings as aliases of one destination. The fix is purely
additive — no documented command changes meaning, `--help` still lists the primary spelling,
and a parametrized test asserts each pair parses to an identical namespace.

One detail worth recording because it cuts the other way. Adding a second option string
lengthened four `add_argument` lines past the project's 100-character limit
(`pyproject.toml`, `[tool.black] line-length = 100`), so the fix as first written traded a
usability defect for three new `E501`s. Those four calls were wrapped to the file's own
continuation style, which also caught two lines that were **already** over before this
branch touched them: `quantum_cli.py` now reports **2** `E501`s where it reported 4. `E501`
is not in CI's hard gate (`E9,F63,F7,F82`) and the flake8 step is `continue-on-error`, so
nothing would have failed — which is exactly why it is easy to leave behind.

Because both edits touch a generation entry point, the byte contract was re-checked rather
than reasoned about: `gs --n 6 --N 40 --seed 0` through the CLI and the same configuration
through `generate_ground_state_datasets` produce **byte-identical** `x_view`, `phi_view`,
`meta/*.json` and `meta/*_F.npy` (4/4 artefacts, `_F.npy` by `np.array_equal`). The CLI adds
no numerical path of its own.

That check also surfaced a difference worth stating, since it is easy to trip over: **the
CLI's defaults are not the API's defaults.** `qdata-gen gs` defaults `--label` to the single
value `sparse`, mirroring the standalone generator's one-dataset-per-invocation behaviour,
whereas `generate_ground_state_datasets` defaults `label` to a two-element *sweep*. So
passing `name=` to the API with default arguments is a collision, and the D-series
`reject_name_for_sweep` guard refuses it by name — `name='bc' was given but the sweep has 2
configurations, whose outputs would collide under that one name` — instead of silently
overwriting one dataset with the other. The guard doing its job is why the byte comparison
above had to be set up deliberately, and it is the behaviour to keep.

For context on the 0% figure, the pre-existing classical generators are covered far less
than the new quantum ones: `make_spirals` 18%, `make_moons` 32%, `make_s_curve` 32%,
`make_swiss_roll` 34%, `make_blobs` 39%, `make_spheres` 45%. Those are out of scope here and
were left alone.

### D14 — four propagators spelled `V.T` where a fifth spelled `V.conj().T`

Every time-evolution operator in the package is built by diagonalising a Hamiltonian and
reassembling it: `E, V = np.linalg.eigh(H)` then `U = (V * exp(-i E t)) @ V<adjoint>`. Five
sites do this, and they did not agree on what the adjoint is:

| Site | spelling |
|---|---|
| `make_time_evolution.py:82` | `V.T` |
| `make_hamiltonian_learning.py:73` | `V.T` (as `c0 = V.T @ psi0`) |
| `quantum_selftest.py:250` | `V.T` |
| `quantum_core.py:722` (`evo_encoding_state`) | `V.T` |
| `make_quantum_labels.py:86` | `V.conj().T` |

**This is not a live bug, and the audit says so plainly.** `pauli_sum` deliberately
downcasts its result: `if H.nnz and abs(H.imag).max() < 1e-14: H = H.real.tocsr()`, with a
docstring that says "cast to a real matrix when its imaginary part is numerically zero
(which it is for every Hamiltonian built here)". Measured, for every Hamiltonian the package
actually constructs — disordered Ising (`te`, `hl`, the self-test), Heisenberg XX+YY+ZZ
(`ql`), and the XX chain inside `evo_encoding_state` — `H.dtype` comes back `float64`, so
`eigh` returns a **real** `V` and `V.T` is bitwise identical to `V.conj().T`. The Heisenberg
case is the one that looks wrong and is not: `Y ⊗ Y` is real, because the two factors of
`i` multiply out.

What makes it a defect rather than a style point is that the invariant is **conditional and
data-dependent**, and four sites depend on it silently while the fifth does not. Add one
genuinely complex term — a single-site `Y` field, a flux phase, a DMI coupling — and
`pauli_sum` correctly returns `complex128`, `eigh` returns a complex `V`, and the four `V.T`
sites stop computing the operator they document. Measured on `n=4, τ=0.7` against
`scipy.linalg.expm(-1j*H*τ)` computed independently:

| | real `H` (as built today) | complex `H` (one `Y` term) |
|---|---|---|
| `‖U(V.T) − expm(−iHτ)‖_max` | 8.95e-16 | **3.33e-01** |
| `‖U(V.conj().T) − expm(−iHτ)‖_max` | 8.95e-16 | 9.19e-16 |

The 0.33 is the whole finding. And the failure is close to undetectable by the obvious
guard: `V.T` with complex `V` still yields a **perfectly unitary** matrix — unitarity error
1.4e-15, because the transpose of a unitary is unitary — so a norm check, a trace check or a
"is it still a valid quantum evolution" assertion all pass while the dynamics are wrong. It
would surface as quietly wrong labels, not as an exception.

All five sites now spell the adjoint `V.conj().T`. The change is **bitwise neutral today** —
verified, not assumed, by re-running the byte-identity comparison — and correct if the
dtype invariant ever stops holding. A regression test builds a deliberately complex
Hermitian `H`, asserts `pauli_sum` keeps it `complex128`, and asserts the reassembled
propagator matches `expm` to 1e-12, which fails if anyone reverts the spelling.

### A methodology note on where the suite was run

Earlier runs of the parallel unit tier died with `[gwN] node down: Not properly
terminated` and `exit 137`, which reads as a test crash and was initially attributed to
one. It was not. These runs were on an **LSF cluster login node** shared with ten other
users: there is no cgroup memory cap (`memory.max = max`), no `ulimit`, and 354 GB free,
yet sustained CPU-saturating processes were reliably SIGKILLed — the login node polices
them. The same `exit 137` had earlier been blamed on this port's notebook, which in
isolation runs in 92.5 s at 0.48 GB peak.

Every result in section 10 was therefore taken from an `bsub` job on a dedicated compute
node. One trap is worth recording: the job script exports `OMP_NUM_THREADS=1` before
reporting its allocation, and `nproc` *honours* that variable, so it printed `CORES=1`
and looked like a one-core allocation. `nproc --all` and the affinity mask both report
128. Three healthy jobs were cancelled chasing that artefact before it was identified;
adding an `affinity[...]` clause to work around it is what would genuinely have pinned
the job to one core.

A second methodology point, recorded because it changes how section 10's table should be
read. Fixes kept landing while jobs were queued and running, so three of the jobs below
were started against a tree that later changed:

| Job | Started against | Superseded for |
|---|---|---|
| 969117 `qbc_align` | the `eng` control's inputs, which no fix touched | nothing — result stands |
| 969268 `qbc_gatefix` | the tree *before* D9's siblings, D10, D11, D12 | the three files it exists to test were not touched again, so its G1/G2/G3 verdicts stand |
| 969269 `qbc_final` | the same earlier tree | superseded by 969698 |
| **969698 `qbc_finaltree`** | **the final tree, all fixes in** | **authoritative** |

Rather than assert that the later edits were harmless, the claim is checked two ways: a
177-test targeted tier over exactly the files that changed (`test_data_generation`,
`test_generator_dispatch`, `test_openmp_import_order`, `test_quvine_packaging`,
`test_suite_hygiene`, `test_plotting_hygiene`, `test_credential_hygiene`,
`test_quantum_data_generation`) passed 177/177 after every edit, and job 969698 re-runs
every tier on the final tree. Where 969269 and 969698 disagree, 969698 is the number.

### What was installed into the `qbc` environment

None of this is a change to the package's declared dependencies; it is disclosed so the
test results can be reproduced, and so nothing here is mistaken for a dependency bump.
Every pinned version is where the plan recorded it: numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.1, torch 2.14.0, qiskit 2.2.0, pandas 2.3.3 — a `pip install` dry run
confirmed all fifteen `[tabpfn]` packages were *additions*, with zero version movement.

| installed | why | declared? |
|---|---|---|
| `pymfe` 0.4.4 + `gower` 0.1.2 | pymfe metafeatures | yes — `requirements-base.txt:44` |
| `quimb` 1.15.0, `cotengra` 0.8.2 (+ `autoray`, `cytoolz`, `toolz`) | the `[mps]` extra | yes — `[mps]` |
| `[tabpfn]`: `tabpfn` 9.0.0 + 14 transitive | the `[tabpfn]` extra | yes — `[tabpfn]` |
| `pytest-xdist` 3.8.0 | test runner only (`-n 8`) | **no, and correctly not** — it is not a package dependency |

Two generated fixtures were also created, both already gitignored (`.gitignore:99`), from
the tutorials' own recipes: `tutorial/QProfiler/data/ld_data` (66 CSVs) and
`tutorial/QProfiler_v2/data/ld_data_v2` (3 CSVs). Twenty-one tests take them as a
fixture and had been failing on `FileNotFoundError`.

One trap worth recording about TabPFN, because it produced a wrong conclusion before it
was caught: a bare `TabPFNClassifier()` defaults to the **v3** checkpoint, which is
licence-gated and raises `TabPFNLicenseError` with no interactive terminal. QBioCode pins
**v2**, which is ungated (Prior Labs License, commercial use permitted) — and conftest's
probe goes through `compute_tabpfn` for exactly this reason, documented at
`tests/conftest.py:63`. Probing with the bare constructor reported the whole environment
as gated when it was not; through the pinned path, TabPFN fits in 3.1 s at 0.92 GB.

### A wrong simulation of the missing extra, and what it cost

The first attempt at the no-`[tabpfn]` tier put a `tabpfn.py` on `PYTHONPATH` whose body
raised `ImportError`. It reported **22 failures and 6 errors** across five files. None of
them were real.

The gate is `compute_tabpfn.tabpfn_is_available`, and it asks
`importlib.util.find_spec("tabpfn")` — deliberately, so that merely *asking* does not drag
torch in (`compute_tabpfn.py:139-152`). A shadow file is perfectly locatable, so the gate
answered **True**, and execution walked past the actionable "TabPFN is not installed"
branch into `importlib.import_module("tabpfn")`, which then raised. That is a state no
install can actually be in: the module was findable but not importable. The 22 failures
were all artefacts of that impossible state.

The faithful simulation blocks at `sys.meta_path` instead, which is the idiom the suite
already uses for optional dependencies (the `Blocker` in
`tests/test_quvine_packaging.py:86-96`). `find_spec` then raises `ModuleNotFoundError`,
`tabpfn_is_available`'s `except (ImportError, ValueError)` catches it and returns False,
and the clean actionable error is raised at the guard — verified directly:

| | shadow file (wrong) | `meta_path` blocker (right) | extra installed |
|---|---|---|---|
| `find_spec("tabpfn")` | returns a spec | raises `ModuleNotFoundError` | returns a spec |
| `tabpfn_is_available()` | **True** | **False** | True |
| `_load_tabpfn_classifier()` | `ImportError` from the module body | actionable "TabPFN is not installed…" | returns the class |

The blocker lives in a `sitecustomize.py`, so `site` loads it at interpreter startup and
every pytest-xdist worker inherits it; a `conftest` hook would not have reached the
workers' own import of `qbiocode`.

Worth stating plainly: the earlier 22-failure number was mine, not the package's, and the
D7 finding below stands on the *corrected* simulation, not that one.

### Simplifications applied

- Six comma-tuple statements of the form `a.append(x), b.append(y)` — which evaluate
  to a discarded tuple — split into plain statements across `quantum_core.py`,
  `make_ground_state.py` and `make_quantum_labels.py`.
- `make_engineered_kernel.py` made 27 `engineered_labels` calls over a 26-point
  `gamma_c` grid: it scanned, then recomputed the winner. `engineered_labels` takes no
  RNG and is deterministic, so the scan's result is kept instead, dropping one dense
  `sqrtm`/`solve`/`eigh` per dataset.
- The name-versus-sweep guard was duplicated verbatim in all five family modules — six
  lines each, including a hand-assembled "x, y and z" list of that family's knobs —
  and is now one `reject_name_for_sweep(name, n_configurations, knobs)` in
  `quantum_core.py`, called once per module. 30 lines became 5 calls plus one helper.
  The `name=` prefix of the message is preserved deliberately:
  `tests/test_quantum_data_generation.py:414` matches on it, and it was confirmed that
  nothing pins the rest of the prose before the wording was unified.
- `make_quantum_labels.py:96-99` had a 145-character line, over CI's flake8 limit of
  127; wrapped.
- **Two dead-store families across the classical generators**, found by reading the
  `F841` findings rather than counting them. Every one of the seven generators built
  `config = "n_samples={}, noise={}".format(...)` and then used it in nothing but a
  commented-out `# print("Configuration {}/{}: {}".format(...))`; and five of them wrote
  `new_dataset = dataset.to_csv(...)`, binding the `None` that `to_csv` returns.
  `make_circles.py` and `make_class.py` had already been fixed here while D9 was being
  tracked down, which is exactly the problem — the fix had left the family
  inconsistent, with two members clean and five carrying the same two dead stores. All
  seven are now clean. `make_spheres.py:146` needed a second pass because it writes
  through `df.to_csv(`, not `dataset.to_csv(`, and `make_blobs.py:11` imported `numpy`
  without a single `np.` in the file. **Verified non-behavioural by byte comparison**,
  not by inspection: all eight generators were run from the `origin/aritra/v2` sources
  and from the current sources into separate trees at a fixed seed, and every output CSV
  is byte-identical (21 files). The only differing artefact is `dataset_config.json` for
  `circles` and `class`, whose keys change from `ld_data-N.csv`/`hd_data-N.csv` to
  `circles_data-N.csv`/`class_data-N.csv` — which is D9's fix, and the reason for
  running the comparison.
- `qbiocode/learning/__init__.py` imported `compute_pqk, compute_pqk_opt` twice, four
  lines apart. The second copy was under the `# Quantum ML algorithms` header, so the
  header was moved up to the first copy rather than the first copy deleted: that keeps
  the module's import *order* byte-for-byte, which matters because
  `tests/test_openmp_import_order.py` asserts on it.

### Defect classes that were scanned and came back clean

Not every audit pass finds something, and the passes that find nothing are worth
recording so they are not repeated blindly. Each of these was a whole-package `grep`
over `qbiocode/`, not a spot check:

| Pattern looked for | Why it matters | Result |
|---|---|---|
| `except:` / `except Exception: pass` | Swallowing an exception turns a crash into a silently wrong number | **0 sites.** Every handler in the package names its exception and does something with it |
| Mutable default arguments (`def f(x=[])`, `def f(x={})`) | State leaks between calls; the classic Python trap | **0 sites** |
| Division by a quantity that can be zero in the new quantum code | An `inf`/`nan` that propagates into a shipped feature column | **No reachable site.** `quantum_core.py:373,683,723` divide by `2**n`; `:785-786` by `np.trace(K)`, which is `N` exactly, both kernels having unit diagonal; `:746` clips before `sqrt`. `level_spacing_ratio` at `:405` is the one real candidate and is already guarded twice — `s = s[s > 1e-12]` drops zero gaps, then `len(s) < 2` returns `nan` rather than dividing, and `tests/test_quantum_data_generation.py:658` tests exactly that |
| `assert` used as a runtime check in shipped code paths | Stripped by `python -O` | One site, the self-test, which is what D11 is about. No others |

### Changes deliberately declined

- **One-einsum `expvals`** (`np.einsum("dm,d,dm->m", S[tgt].conj(), ph, S).real`)
  would change summation order and so break byte-identity with the shipped corpus.
- **`_popcount` → `np.bitwise_count`** would impose an undeclared `numpy>=2` floor
  (`requirements-base.txt` pins numpy unpinned), and it is cached via `Pauli._act`, so
  it is not a hot spot.
- **Duplicating `ql`'s upfront `entanglement` validation into `eng`** — `ql` needs it
  only because the `evo` encoding bypasses `zz_feature_state`, which `eng` always calls.
- **`black` on the eight new quantum modules.** The lint job runs
  `black --check --diff qbiocode/` and it is `continue-on-error: true`, so it reports and
  never blocks — which is just as well, because **101 of the 146 files under
  `qbiocode/` are not black-clean on this branch, and they are not clean on
  `origin/aritra/v2` either** (spot-checked: `evaluation/dataset_evaluation.py` at the
  merge base fails `black --check` on its own). Black-cleanliness is therefore not this
  repository's convention, and the new modules sit at 29 hunks / ~712 lines from it:

  | New module | hunks | lines black would change |
  |---|---|---|
  | `quantum_cli.py` | 2 | 223 |
  | `make_ground_state.py` | 4 | 99 |
  | `make_engineered_kernel.py` | 2 | 86 |
  | `make_quantum_labels.py` | 2 | 85 |
  | `make_time_evolution.py` | 3 | 82 |
  | `make_hamiltonian_learning.py` | 4 | 73 |
  | `quantum_selftest.py` | 7 | 43 |
  | `quantum_core.py` | 5 | 21 |

  Almost all of it is one thing: the argparse tables and the aligned trailing comments
  that carry the physics (`# (check diagnostics.level_spacing_ratio)`,
  `# even sector: prod X = +1`), which black explodes one-argument-per-line and pushes
  off the end. Two further reasons to decline it *here* rather than never: black must not
  land while a validation job is mid-flight against this exact working tree (four are, at
  the time of writing; each job's log records the `DIRTY_FILES` count of the tree it ran,
  so the correspondence is checkable); and although black cannot change semantics, the
  byte-identity proof in §2.1 was run against these sources, so reformatting means
  re-running it rather than assuming. `black qbiocode/data_generation/` plus a re-run of
  the round-trip harness is the whole job, and it is recommended as its own commit.

- **The 18 remaining `E501`s in the eight new modules.** At the project's own
  `line-length = 100`, the new modules carry 18 over-long lines: `make_ground_state` 5,
  `make_time_evolution` 5, `make_engineered_kernel` 2, `quantum_cli` 2, `quantum_selftest`
  2, `make_hamiltonian_learning` 1, `make_quantum_labels` 1. All are docstring prose or
  aligned help text, none is code. For scale, `qbiocode/` as a whole reports **510** of
  them, and 13 sit in `qprofiler.py` alone on `origin/aritra/v2`, so this is a
  package-wide convention that is not enforced rather than something this branch
  introduced. `E501` is not in CI's hard gate (`E9,F63,F7,F82`) and the flake8 step is
  `continue-on-error`. Declined here for the same reason as black: rewrapping prose in six
  generator files buys nothing and costs a full re-validation cycle of a tree that is
  currently green. The two `E501`s this branch *did* introduce, by lengthening
  `add_argument` lines with flag aliases, were fixed at the point of introduction (D13) —
  which is the rule worth keeping: don't add to it, don't sweep it.

- **`encoding=` on the 55 text-mode `open()` calls outside `data_generation`.**
  D12 fixed the nine JSON writers in `qbiocode/data_generation/` because those write the
  corpus this branch ships. The same pattern runs through the rest of the package —
  `qbiocode/apps/quvine/reproducibility/` (split, seed, registry, validator, method
  runner), `apps/quvine/cli.py`, `apps/qprofiler/qprofiler.py`'s five CSV handles,
  `utils/generate_qml_configs.py`, `utils/ibm_account.py` — 55 sites in total, **none of
  them in a file this branch touches**. Every one has the same failure mode as D12 (a
  non-UTF-8 `LC_CTYPE` writes a file that a UTF-8 reader cannot parse) and the same
  one-token fix, so this is a real worklist and not a style preference. Declined here
  only because it would add ~20 untouched files to a port's changeset;
  `grep -rn "open(" --include=*.py qbiocode/ | grep -v encoding=` is the worklist.

- **The rest of the package-wide lint sweep.**
  `flake8 qbiocode --select=F401,F811,F841,E711,E712,E714,E722,F632` reported **152**
  findings on this branch before today — 117 `F401`, 28 `F841`, 3 `F811`, 2 `E711`,
  2 `E712` — all pre-existing on `origin/aritra/v2`. Reading them rather than counting
  them is what turned up D9's siblings: 12 of the 28 `F841` were the dead `config` and
  `new_dataset` stores in `data_generation`, and 2 of the 3 `F811` were the duplicate
  `compute_pqk` import, so the sweep was not cosmetic and those 15 were fixed (see
  *Simplifications applied*). **137 remain** — 116 `F401`, 16 `F841`, 2 `E712`,
  2 `E711`, 1 `F811` — and they are declined for review size, not because they are
  all benign:

  | Kept | Where | Why it stays |
  |---|---|---|
  | 116 `F401` unused imports | `qutils.py` 10, `compute_qsvc.py` 10, `quvine/reproducibility/graph_generator.py` 9, then a long tail over ~30 files | Touching 30 files in the port's changeset would bury the port |
  | 3 `F841` `model_fit = model.fit(...)` | `compute_qnn.py:154`, `compute_qsvc.py:110`, `compute_vqc.py:105` | Harmless: `fit` returns `self` and the caller uses `model`, so nothing is lost |
  | 1 `F841` `correlation_missing` | `visualize_correlation.py:533` | A **genuine leftover** — computed, never read, made redundant when NaN handling moved to the colormap's `set_bad` at `:474`, which is in place and tested. Safe to delete, but it is a visualization file the port does not touch |
  | 12 other `F841` | `quvine/*`, `qprofiler_batchmode.py:204`, `graph_evaluation.py:886` | Each needs its own reading to tell a leftover from a deliberate unpack; not reviewable in bulk |
  | 2 `E711` / 2 `E712` | `qutils.py:502` `(a != None) & (b != None)`, `compute_qsvc.py:105` and `model_evaluation.py:258` `== True` | Correct as written on the values they see |

  Three of the 137 are in files this branch does touch, and each is left for a stated
  reason rather than overlooked:

  - `make_spheres.py:13` and `make_spirals.py:13`, `import matplotlib.pyplot as plt` —
    used only by the commented-out `# fig = plt.figure()` / `# plt.savefig(...)` blocks
    at `:151,155` and `:213,216`. Deleting the import would leave those blocks
    un-re-enableable without a reader noticing, and `matplotlib` is loaded by
    `qbiocode.visualization` regardless, so there is no import-cost argument either.
  - `compute_nb.py:5`, `OneVsRestClassifier` — one half of
    `from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier`, a line
    repeated verbatim in `compute_lr`, `compute_dt`, `compute_svc`, `compute_mlp`,
    `compute_rf` and `compute_xgb`. Trimming it in this one file would make it the only
    member of the family with a different import line — the same consistency argument
    that motivated cleaning all seven classical generators here argues for leaving it.

  The command above is the whole remaining worklist; recommended as its own commit.

- **A `CHANGELOG.md` entry for D13 and D14.** Checked and declined deliberately, not
  overlooked. Both are fixes to code that lives entirely inside the `## [Unreleased]` /
  `### Added` block's own `#### Simulated quantum datasets as binary-classification
  benchmarks` feature: the eight flag aliases and `tests/test_quantum_cli.py` are new
  surface on a CLI that has never shipped, and the `V.conj().T` correction changes no
  number any release ever produced (14 artefacts compared pre/post, 0 differ). A
  `### Fixed` entry describes a behaviour change a reader can have experienced; listing one
  here would tell every reader of the next release notes about a bug that was never in a
  release, and would imply their generated data might need regenerating when it provably
  does not. The convention the file already follows is to fold same-release corrections
  into the feature's own entry, which the existing text does. The place where D13 and D14
  *are* recorded is here, plus the regression tests that would fail if either regressed —
  `TestBothFlagSpellingsWork` and `TestTheAdjointIsAConjugateTranspose`.

### Corpus fidelity after the fixes

The whole 13-dataset corpus was regenerated with the patched generators and compared
against `qdata/` (`filecmp` for CSV/JSON, `np.array_equal` for `.npy`):

| Artefact | Result |
|---|---|
| all 13 `x_view/*.csv` and `phi_view/*.csv` | **byte-identical** |
| `gs`, `te`, `hl` `meta/*.json` and `_F.npy` | **byte-identical** |
| `ql`, `eng` `meta/*.json` `threshold`, `_F.npy` | differ by ≤ 8.9e-15 absolute |

**Nothing a model consumes changed**: every feature and every label is bit-identical,
so all benchmark results remain comparable. The residual is confined to the continuous
pre-threshold diagnostic and the threshold derived from it, at 1–4 ulp of float64.

That residual is *not* attributable to these fixes. Two runs of the current code are
bit-identical to each other (15/15 files), so the generators are reproducible; and
with the only numerics-touching change in `ql`'s path reverted (D6), `ql` still
differs from the shipped corpus by 2.2e-16 although nothing else in its path changed.
The shift therefore predates this pass and comes from the environment — the corpus was
generated before `quimb`, `cotengra` and `pymfe` were installed into `qbc`, which
moves BLAS/LAPACK dispatch. Regenerating the corpus would remove even this.

### Corpus completeness, re-checked against the contract

Independently of the numerics check, the corpus at `/dccstor/cgq4hls/Q/qbc_data/qdata` was
re-walked and every structural promise the loader depends on was verified rather than
assumed. 13 `x_view` CSVs, 7 `phi_view` CSVs, 13 `meta/*.json`, 13 `meta/*_F.npy`, no
orphan on either side, and no `phi_view` file without its `x_view` counterpart.

| dataset | family | rows | x features | φ features | balance |
|---|---|---|---|---|---|
| `eng_zz_n6_gq1_s0` | `eng` | 300 | 6 | — | 0.500 |
| `eng_zz_n6_gq1_s0_dmunit` | `eng` | 300 | 6 | — | 0.500 |
| `gs_e2e_n8_k0.5_s0` | `gs` | 400 | 15 | 35 | 0.500 |
| `gs_sparse_n8_k0.5_s0` | `gs` | 400 | 15 | 35 | 0.500 |
| `hl_n6_g0.5_shots1000_s0` | `hl` | 400 | 12 | — | 0.500 |
| `ql_evo_n8_tau1_s0` | `ql` | 400 | 8 | — | 0.500 |
| `ql_zz_n8_tau1_s0` | `ql` | 400 | 8 | — | 0.500 |
| `ql_zz_pqkdefaults` | `ql` | 400 | 8 | — | 0.500 |
| `te_n10_s4_seed0_tau{0.25,0.5,1,2,4}` | `te` | 400 | 10 | 57 | 0.500 |

The feature counts are the ones plan verification step 5 predicts from the qubit count:
`gs` = 2n−1 = 15 at n=8, `te` = n = 10, `hl` = 2n·|times| = 12 at n=6 with one time,
`ql` = `eng` = n. Every label column is last, is 0/1, and every class balance is exactly
0.500. For both `phi_view` families the φ row count equals the `x` row count and the φ label
column is *element-wise equal* to the `x` label column — checked with `Series.equals`, not
by comparing means, because equal means would not rule out a permutation.

### The `eng` positive control, measured end-to-end through QProfiler

Plan verification step 6 asks whether the families still behave as the runsheet predicts.
The `eng` family is the one that makes a falsifiable prediction, so it was run as a 2×1
control through the real `qprofiler` entry point (`config_qdata_encoded.yaml`,
`embeddings: ['none']`, `scaling: false`, `iter: 5`, LSF job 969117, 6344 s on a compute
node) over both arms of the `--data-map` switch. Mean accuracy over the five splits:

| dataset | `data_map` | `lr` | `svc` | `qsvc` | `pqk` |
|---|---|---|---|---|---|
| `eng_zz_n6_gq1_s0` | `qiskit` (default) | 0.569 | 0.549 | **0.587** | 0.502 |
| `eng_zz_n6_gq1_s0_dmunit` | `unit` | 0.836 | 0.813 | 0.600 | **0.913** |

Per-split standard deviations are 0.021–0.056, so the 0.913-vs-0.836 gap on the second row
is about 2× the spread and the flatness of the first row is not a sampling artefact.

Read against `docs/source/quantum_datasets.md:81,187`, which states that the default arm
targets `qsvc` and the `--data-map unit` arm targets `pqk`, and predicts "PQK > classical,
**only for the arm whose data map was targeted**":

- **The `unit` arm confirms the prediction outright.** `pqk` is the best of the four at
  0.913, ahead of the best classical model by 7.8 points, and `qsvc` — the fidelity kernel,
  which this arm was *not* built for — collapses to 0.600. That is the intended signature,
  and it is reproduced through the packaged config rather than a bespoke script.
- **The default arm reproduces the predicted *ranking* but at chance.** `qsvc` is indeed
  the best of the four, but the whole row sits in 0.50–0.59: the separation does not
  survive binarisation at n=6 / N=300. This is the failure mode the same doc row already
  names — "fails after binarisation → check `--data-map` first, then measure the
  boundary-complexity ratio" — and the metadata says why it is expected to be the harder
  arm: `gamma_c_adversary` is 15.85 for `qiskit` against 2.51 for `unit`, and
  `g_continuous` is 3.23 against 2.15. The continuous target is well separated in both;
  only the `unit` arm keeps that separation through the sign threshold.

So the control fires, and it fires on exactly one arm. Two consequences worth stating
plainly rather than leaving implicit:

1. The prediction in the docs is **correct as written**, including its caveat. Nothing
   needs changing there, and no advantage claim is being made: this is one engineered
   construction at one size, scored by accuracy on five splits.
2. The default-arm dataset in the shipped corpus is, empirically, a **near-chance dataset
   for all four models**. That is legitimate as a negative control and it is what the
   `margin` knob exists to sharpen, but a reader who takes `eng_*` to be the corpus's
   PQK-favourable rung will pick the wrong file. The `_dmunit` arm is the one that carries
   the signal, and it is in the corpus alongside it.

### F6(a) implemented, F6(b) still open

The `ddof` half of F6 is now fixed in `dataset_evaluation.py`: the threshold is taken
on `var(ddof=0)` to match `VarianceThreshold`'s `np.nanvar`. Measured effect:

| data | before (`ddof=1`) | after (`ddof=0`) |
|---|---|---|
| raw gaussian, varied scales | 2 | 2 |
| **StandardScaler'd** | **None** | 2 |
| **balanced binary** (`te` bits) | **None** | 2 |
| MinMaxScaler'd | 2 | 2 |

Standardized variances come out at exactly `N/(N-1) = 1.00502513` against a comparison
at `1.0`, so *every* feature was dropped and the metric was unmeasurable for any
standardized or balanced-binary dataset — including the `te` family added here.

**F6(b) is deliberately not implemented.** Patch 6.3 proposes replacing the 25th-percentile
criterion with a data-relative absolute one, because the current metric returns about
`n/4` whatever the data. That is a design change to a `public-main` metric that QSage
regresses on: it changes the meaning of a shipped feature, invalidates comparison with
previously generated `ModelResults`, and section 6's own text offers "or drop the
metric" as an equally valid option. It is a maintainer's decision, not a bug fix, and
is left flagged.

Fixing (a) did invalidate the premise of a pre-existing test,
`tests/integration/test_pymfe_seam_remainder.py::TestTheNativeBranchThatReturnsNone`.
That class exists to pin the `None` → NaN coercion seam, and reached it with six
shuffled copies of `np.linspace(-2, 2, 60)` — whose variances are equal only to ~4e-16
of floating-point noise. The old inflated threshold swamped that noise; the exact
threshold lets it decide which columns survive, so the helper returned a
noise-determined `2`. The fixture now uses a small-**integer** base
(`np.arange(-30, 30, dtype=float)`), for which the mean and every squared deviation are
exactly representable and the variances are therefore *bit-identical* across
permutations — `var(ddof=0).std() == 0.0` exactly, and `None` on all 200 seeds tested.
The seam the class guards is unchanged; only its premise was hardened.

---

## 10. Test verdict — does the suite qualify?

**Short answer: yes, on every tier, and the gate CI does not allow to fail is clean.**
The long answer is that "all tests pass" is only worth something if you can say *which*
tests ran, *where*, and *in what install state* — this box silently kills
CPU-saturating processes on the login node (see *A methodology note*), and the suite
behaves differently depending on whether the `[tabpfn]` extra is installed. So the
suite was run in tiers, on compute nodes, in both install states.

### 10.1 The tiers, and why there is more than one

`pyproject.toml` sets `testpaths = ["tests"]` and
`addopts = "-v -m 'not slow and not requires_quantum'"`. That single default hides two
whole populations of tests, so running `pytest` once is not a verdict:

| Tier | Command | What it exists to catch |
|---|---|---|
| T0 | `qdata-gen selftest` | The physics itself: seven independent checks, the tightest at 1e-15 |
| T0b | the same under `python -O` | That the self-test **refuses** rather than passing vacuously (D11) |
| T1 | `pytest tests --ignore=tests/integration -n 8` | Unit tier, the bulk of the suite |
| T2 | `pytest tests/integration` (serial) | Integration tier; serial because its own fixtures fan out over loky workers |
| T3 | `pytest tests -m "slow or requires_quantum"` | The two populations the default `-m` filter **deselects**, including all six notebooks |
| T4 | `pytest --cov=qbiocode` under a `sys.meta_path` blocker | Exactly what CI runs, in exactly the install state CI is in |
| T5 | `flake8 . --count --select=E9,F63,F7,F82` | The one lint gate in `ci.yml` **without** `continue-on-error` |
| T6 | `pytest tests/integration/test_docs_build.py` **with the env's `bin` on `PATH`**, and `make -C docs html` | The Sphinx site, which no CI job and no other tier here actually asserts (§10.4c) |

T4 needs explaining. CI runs `pip install -e ".[dev]"`, and `[dev]` does not pull
`tabpfn` — so CI's import graph is *not* this environment's. Simulating that by
shadowing the module is what produced 22 phantom failures earlier in this audit
(*A wrong simulation of the missing extra*); the honest simulation is a `sys.meta_path`
finder that raises `ModuleNotFoundError`, which is what `find_spec` sees for a genuinely
absent distribution.

T6 is a tier and not a line in T2 for a reason learned the hard way: its ten assertions live
under `tests/integration/`, so T2 and T4 collect them, but the module-scoped fixture skips
unless `sphinx` imports *and* a `pandoc` binary is on `PATH` — and a batch script that calls
the interpreter by absolute path has neither guarantee. Naming it separately is what makes
the skip visible instead of arithmetic. §10.4c has the measurement and the cause.

### 10.2 Results

| Tier | Result | Exit | Wall |
|---|---|---|---|
| T0 physics self-test | `SELFTEST PASS`, all 7 checks; max error 1.4e-16 (Pauli vs dense kron), 3.0e-15 (native vs Qiskit `ZZFeatureMap`, 36 comparisons) | **0** | ~40 s |
| T0b self-test under `-O` | `REFUSED` with the D11 `RuntimeError`; the "BUG: -O ran the selftest" branch not taken | **0** | ~5 s |
| T1 unit | **1410 passed, 1 skipped, 15 xfailed, 0 failed** | **0** | 4:46 / 34:14 |
| T2 integration | **179 passed, 9 skipped, 2 xfailed, 6 deselected, 0 failed** | **0** | 11:31 / 9:43 |
| T3 slow + `requires_quantum` | **6 passed, 1 skipped, 1615 deselected** — all six tutorial notebooks re-executed, including the new `quantum_datasets_qprofiler.ipynb` | **0** | 20:20 / 21:23 |
| T5 hard lint gate | **0 findings** (`E9,F63,F7,F82`) | **0** | 3 s |

Two wall times are given for T1–T3 because each tier was run twice: once during the fix
cycle, and once again on the tree of the time as job `969698`. The two runs agree on
every count — 1410/1/15, 179/9/2/6, 6/1/1615 — and differ only in wall time, because the
nodes differ in how many cores LSF actually gave the run. That agreement is the point:
the counts are a property of the suite, not of the machine.

**Then two more defects were found and fixed, so the table above is no longer the last
word.** D13 added `tests/test_quantum_cli.py` (43 tests) and the eight flag aliases; D14
changed four propagator sites and added 4 regression tests. Both landed *after* job
`969698` reported, which means its counts describe a tree that no longer exists — the
honest way to say that is to re-run, not to argue the changes were small. The unit tier
now collects **1473** where it collected 1426.

Re-running is also the part that is easy to get subtly wrong, and the first attempt was.
Job `970012` began collecting at 03:50:30 while two formatting-only edits were being
written at 03:51:52 and 03:52:28, so its unit tier read a tree that moved underneath it.
The counts it produced were almost certainly fine — both edits were line wrapping, and the
43 CLI tests passed before and after — but "almost certainly fine" is not a measurement.
Job `970062` is therefore the authoritative one: the last `.py` write is timestamped
**04:01:16** and the job started **04:03:22**, both recorded in its own log via
`find -newermt`, so the tree provably did not move under it. Where the two disagree,
`970062` is the number that stands.

**And that authoritative run failed, which makes it the most useful single result in this
section.** Job `970062`'s unit tier came back `1 failed, 1456 passed, 16 xfailed` — all
1473 collected items accounted for, one of them red:

    tests/test_suite_hygiene.py:164: AssertionError: pytest.importorskip on a module that
    is not optional. Import it directly, or -- if it really is optional -- declare it in
    an extra in pyproject.toml (or add it to CONDITIONAL_SHIMS with the reason):
      tests/test_quantum_data_generation.py:231 importorskip('scipy.linalg') -- scipy is
      declared in requirements-base.txt, so it is present in every install
      tests/test_quantum_data_generation.py:248 importorskip('scipy.linalg') -- scipy is
      declared in requirements-base.txt, so it is present in every install

The D14 regression tests reached `scipy.linalg.expm` through `pytest.importorskip`, and the
suite has a gate that forbids precisely that, for a reason it states in its own docstring:
*"Guarding a mandatory dependency is the same off switch, one step removed."* A package in
`requirements-base.txt` is present in every supported install, so a guard on it can only
ever fire when the environment is already broken — and then it hides the breakage instead
of reporting it. `scipy` is in `requirements-base.txt`. The fix is one line —
`from scipy.linalg import expm` at module scope, as the same file already imports numpy and
pandas, and the two local rebinds deleted — after which the three hygiene tests and both
new files pass together: **136 passed**, flake8 **0** at the project's own 100 columns.

Two things are worth stating rather than quietly fixing. First, this is the house rule of
§10.3 enforced against code written *for this audit*, which is the only real evidence that
the rule is enforced at all rather than merely documented: the same gate that decides
`tabpfn` may skip decides `scipy` may not, and it does not care who wrote the import.
Second, it is exactly why the frozen-tree discipline above is not pedantry — on a run
allowed to collect against a moving tree this failure would have been indistinguishable
from a flake, and the temptation would have been to re-run it rather than read it. Job
`970157` is `970062` with that one test-file fix and no package change of any kind; its
`NEWEST_PY` line records the last `.py` write at **04:30:20** against a **04:31** start.

That "no package change of any kind" is measured, not asserted:
`find qbiocode -name '*.py' -newermt '2026-09-26 04:03:22'` — 970062's start — returns
**nothing**, and across `qbiocode/` and `tests/` the only non-`__pycache__` file newer than
that timestamp is `tests/test_quantum_data_generation.py` at 04:30:20. So the two jobs ran
the same package and differ in one test file, which fixes the attribution cleanly: `970157`
is authoritative for the unit tier and the CI-exact tier, and `970062` remains
authoritative for the tiers that ran before the fix and cannot be affected by it —
integration, the six notebooks, and the lint gate. Nothing is quoted from a run that could
not have produced it.

Two earlier jobs were **cancelled rather than reported**: `970012`, whose unit tier had
already collected against a moving tree, and `969269`, whose F1 and F4 collected 2¼ hours
apart (see §10.3). Both would have produced numbers that then needed a paragraph of
explanation for why they should not be used, which is a worse outcome than not having them.

#### The authoritative results

Every row below names the job that produced it. `970062` ran tiers A0-A5 on the final
package tree; `970157` re-ran the two tiers the `importorskip` fix touches (B1, B2) and added
two more (B3 under the `[tabpfn]` blocker, B4 the lint gate); `970477` closed the docs build
(§10.4c). `970062`'s A4
shows exit 2 because it was **cancelled**, not because it failed: once `970157`'s B2 was
running the same command on the fixed tree there was nothing left for it to establish, and
A4 would have reproduced the A1 failure for the third time.

| Tier | Job | Result | Exit | Wall |
|---|---|---|---|---|
| unit, `-n 8`, `-rsx` | `970157` B1 | **1457 passed, 16 xfailed, 0 skipped, 0 failed** — 8 workers, all **1473** collected items accounted for | **0** | 48:38 |
| CI-exact: `pytest --cov=qbiocode`, `[tabpfn]` absent (T4) | `970157` B2 | **1593 passed, 47 skipped, 6 deselected, 18 xfailed, 0 failed**; coverage TOTAL 14394 statements, 7480 missed, **48%** | **0** | 27:49 |
| physics self-test, 7 checks | `970062` A0 | `SELFTEST PASS`; quench error 2.259e-06, native vs Qiskit `ZZFeatureMap` 3.0e-15 over 36 comparisons | **0** | ~40 s |
| self-test under `-O` (D11) | `970062` A0b | refuses with the D11 `RuntimeError` | **1** = pass | ~5 s |
| integration | `970062` A2 | **179 passed, 9 skipped, 6 deselected, 2 xfailed, 0 failed** | **0** | 9:26 |
| six notebooks (`slow or requires_quantum`) | `970062` A3 | **6 passed, 1 skipped, 1662 deselected** — every tutorial notebook re-executed, including the new `tutorial/Quantum_Data/quantum_datasets_qprofiler.ipynb` | **0** | 20:11 |
| unit under the `[tabpfn]` blocker | `970157` B3 | **1417 passed, 35 skipped, 16 xfailed, 0 failed** from **1468** collected — every skip named by `-rs` | **0** | 14:01 |
| hard lint gate `E9,F63,F7,F82` | `970062` A5, `970157` B4 | **0 findings**, both jobs | **0** | 3 s |

Job `970157` itself closed as **`Successfully completed`** with **empty stderr**, 5440 s wall
and 33 GB peak memory, and all four of its tiers exited 0. That line in the job's own `.out`
is the authoritative status, not `bhist`: LSF job ids are recycled on this cluster, so a
lookup by id can return a different job's record entirely.

The integration count is identical to what the earlier `969269` F2 reported — 179/9/6/2 — on
a different node and a different tree revision, which is the same point §10.2 made about
969698: these counts are a property of the suite, not of the machine.

The unit tier closes the arithmetic exactly: 1410 before D13 and D14, plus the 43 CLI tests,
plus the 4 adjoint regression tests, is **1457**, and 1457 + 16 xfailed is the 1473 collected.
Nothing is unaccounted for and nothing skipped — this node's Aer statevector limit falls
below 36, so `tests/test_qensemble.py:1161` ran and xfailed strictly rather than skipping,
which is the machine-dependent behaviour §10.2 traced above, now observed in its other state.

Collection is clean at every tier: `pytest --collect-only -q` reports **1658/1664
collected, 6 deselected** — the 6 being the notebook tests the default filter drops, which
is why T3 exists.

B2 closes on those same numbers from the other direction, which is the point of running it:
1593 passed + 47 skipped + 18 xfailed is **1658**, which is 1657 selected plus the one
module-level skip pytest counts in the total but not in the selection, and 1658 + 6
deselected is the 1664 discovered.

**Its skip count moved, and the reason is in the run script, not the branch.** Job `969268`'s
G3 ran this same command on the earlier tree and reported `1554 passed, 39 skipped, 6
deselected, 18 xfailed` from 1616 collected. B2 collected 1663 — exactly **+47**, the 43 CLI
tests of D13 and the 4 adjoint tests of D14 — and the two runs reconcile to the item:

    collected   1616 + 47 new                        = 1663   (+1 module skip = 1664)
    passed      1554 + 47 new - 8 moved to skipped   = 1593
    skipped       39 + 8 moved                       =   47
    xfailed       18 unchanged, deselected 6 unchanged

The 8 are all of `tests/integration/test_docs_build.py`'s Sphinx assertions, and they
skipped because `authoritative2/run.sh` invokes the interpreter by absolute path and never
exports `PATH`. The module's `build` fixture opens with `if shutil.which("pandoc") is None:
pytest.skip(...)`, `pandoc` lives in the env's `bin` (a 164 MB binary beside `python`), and
that directory was not on the job's `PATH`; `gatefix.sh:16` had exported it, which is why G3
ran all ten. So the guard fired on a missing binary, exactly as its docstring says it should
— *"missing toolchain, not broken documentation"* — and the difference between 39 and 47 is
a property of two shell scripts. It is recorded here rather than quietly re-run because a
skip that appears between two runs of the same command is indistinguishable from a
regression until someone names the cause. Job `970477` re-runs that module on the current
tree with `PATH` carrying the env's `bin`, and §10.4c reports it.

With B3 in hand, **every one of B2's 47 skips is accounted for by file and by cause** — which
is the standard this audit set for the suite and therefore owes its own tiers:

| Count | Where | Cause |
|---|---|---|
| 35 | five unit-tier files (§10.3 lists them) | the `[tabpfn]` extra absent — measured directly by B3 |
| 3 | `tests/integration/test_catboost_tabpfn_integration.py` | the same |
| 8 | `tests/integration/test_docs_build.py` | `pandoc` not on the job's `PATH` — closed by `970477`, §10.4c |
| 1 | `tests/integration/test_qprofiler_quvine.py:33` | module-level `importorskip("gensim")`; the `[quvine]` extra is not installed here, which is why pytest counts it at collection rather than as a test |
| **47** | | |

On the part that is a real comparison, B2 misses **67 fewer lines** than G3 (7480 against
7547) across an unchanged 14394 statements.

**One count did differ between runs, and it was chased down rather than waved through.**
The unit tier reported `1 skipped, 15 xfailed` on one node and `16 xfailed` on another —
same total, same passed count, zero failures either way, so nothing was failing, but a
count that moves is a count that is not understood. Re-running with `-rsx` named it:

    SKIPPED [1] tests/test_qensemble.py:1161: this machine's Aer statevector limit is
    36 qubits, so the guard's 36 is not above it and there is no gap to pin

`tests/test_qensemble.py:1160-1176` carries **two stacked markers** on one test — a
`@pytest.mark.skipif(AER_MAX_QUBITS >= 36, ...)` and a `@pytest.mark.xfail(strict=True,
...)` — where `AER_MAX_QUBITS = AerSimulator(method="statevector").num_qubits` is derived by
Aer from the node's available memory. So the outcome is a property of the node: where Aer's
limit lands below 36 the test runs and xfails strictly (the pin: `compute_qensemble.py:444`
hardcodes 36 while Aer derives its own), and where it lands at or above 36 there is no gap
to pin and the test skips. This login node reports 34; the two batch nodes reported below
and at 36 respectively. Either way the tier is green, and one test is accounted for in both
totals.

Worth stating for the record: `tests/test_qensemble.py` is **unchanged by this branch**, so
this is a pre-existing property of the suite that the port merely ran on more machines than
before. It is also why an earlier attempt to find it failed — an AST scan for an
xfail-marked test containing a `pytest.skip` *call* finds nothing, because the skip is a
stacked `skipif` *marker*, not a call in the body.

### 10.3 The `[tabpfn]`-absent state, measured

This is where the audit's largest single fix (D7 + D8, 13 CI-breaking failures) is
either proven or not. Two runs, both under the `sys.meta_path` blocker:

| Run | Scope | Before the fix | After the fix |
|---|---|---|---|
| G1 | the six files that touch TabPFN (`test_catboost_tabpfn`, `test_tabpfn_token`, `test_tuner_engine_selection`, `test_opt_twins_dispatch`, `test_classical_models`, `test_model_contract_matrix`) | 6 failed, 372 passed, 32 skipped, **7 errors** | **380 passed, 32 skipped, 0 failed, 0 errors** in 1443 s, exit **0** |
| G2 | the same six files with the extra **installed** (this environment) | — | **417 passed, 0 skipped, 0 failed** in 4793 s (1:19:53), exit **0** |

G1 is the decisive one and it is unambiguous: the 13 failures are closed, and closed by
skipping for a *stated reason* rather than by weakening an assertion. G2 is what makes that
claim checkable in both directions, and its count settles it arithmetically: **380 + 32 + 5 =
417**. Every one of G1's 32 skips *passes* when the extra is present — so the gate skips
exactly the tests that need `tabpfn` and no others — and the 5 extra items are the
parametrized cases enumerated below, which exist only when `tabpfn_is_available()` is true at
collection time. A gate that had been hiding a genuine failure would show up here as a G2
failure or as a G2 count below 412; neither happens. The gate follows
the house rule the repository already had — `requirements-base.txt` membership decides
whether an absent import skips or fails (`tests/test_mps_backend.py:22`
`importorskip("quimb")` vs `tests/test_openmp_import_order.py:42`) — and `tabpfn` is not
in `requirements-base.txt`, deliberately, because it would drag torch's ecosystem into
`import qbiocode`.

**What the blocker does to collection, exactly.** It removes five parametrized cases and
adds none, because `tabpfn_is_available()` is consulted at collection time to build the
parametrize lists:

    tests/test_opt_twins_dispatch.py::TestEveryClassicalTwinIsReachedThroughTheDispatcher
        ::test_the_reported_choice_is_one_the_config_block_offered
        ::test_the_tuned_run_is_reported_under_the_opt_label
    tests/test_tuner_engine_selection.py::TestEachEngineIsReallyReached
        ::test_a_config_that_names_the_grid_gets_the_grid_and_not_the_sampler
        ::test_the_grid_engine_refuses_a_range_it_could_never_enumerate
        ::test_the_optuna_engine_samples_the_very_range_the_grid_refused

So the unit tier collects **1473** with the extra installed and **1468** without — measured
by diffing `--collect-only -qq` node-id sets, not inferred, and then confirmed by actually
running it: job `970157`'s B3 reports `8 workers [1468 items]` and **1417 passed, 35 skipped,
16 xfailed**, which sums to 1468 exactly. Against B1's 1457/0/16 from 1473 the two runs
differ by precisely the two mechanisms and nothing else:

    collected   1473 - 5 parametrized cases not built   = 1468
    passed      1457 - 35 now skipped - 5 not collected = 1417
    xfailed     16 unchanged

`-rs` names all 35, and they fall in five files: `test_classical_models.py` 12,
`test_catboost_tabpfn.py` 12, `test_tabpfn_token.py` 8, `test_grid_search_partial.py` 2,
`test_model_run_edges.py` 1. Two distinct reason strings appear — `the [tabpfn] extra is not
installed`, which is the D7/D8 gate, and `could not import 'tabpfn'`, which is a
`pytest.importorskip` — and both are legal for the same reason: `tabpfn` is not in
`requirements-base.txt`. Nothing skips for an unstated reason.

One thing to know when reading the job log: `authoritative2/run.sh` labels this tier
`### B3: unit tier with the extra PRESENT`, and the label is wrong — the command on the next
line sets `PYTHONPATH` to the blocker, and the log it writes is `B3_unit_no_tabpfn.log`. The
tier measured what it was meant to; only the `echo` is mislabelled. This is worth pinning because
it is the *whole* of the difference: a run that reports more items under the blocker than
without it is measuring something other than the blocker. Job `969269` did exactly that —
its F1 reported 1426 items and its F4, the same selection under the blocker, reported 1464
— and the cause is the clock, not the extra. F1 collected at 01:26, before
`tests/test_quantum_cli.py` existed: 1473 − 43 − 4 = **1426**. F4 collected around 03:41,
after the 43 CLI tests landed and before the 4 D14 tests did: 1426 + 43 − 5 = **1464**. Both
numbers reconstruct exactly, neither is anomalous, and `969269` is authoritative for
neither tier. Note `-qq` and not `-q`: `addopts` already carries `-v`, verbosity is the sum
of the flags, and at net zero `--collect-only` prints a tree instead of node ids.

### 10.4 What the port did to the suite

Measured against `origin/aritra/v2` in the same environment, same tiers:

| Tier | `origin/aritra/v2` | this branch | delta |
|---|---|---|---|
| unit | 1304 passed | 1410 → **1457** passed | **+153 tests** |
| integration | 172 passed | 179 passed | **+7 tests** |
| slow / quantum | 5 passed | 6 passed | **+1 notebook** |

The unit-tier figure moved twice after the first measurement, which is why two numbers are
shown: 1410 was the count before D13 and D14, and the 43 CLI tests plus 4 adjoint
regression tests bring it to 1457, measured by job `970157`. The two new test files account for **133** of the added
tests on their own — `tests/test_quantum_data_generation.py` 90, `tests/test_quantum_cli.py`
43.

Skip and xfail counts are **identical** on both sides. That is the number that matters:
the port adds tests and converts none of the existing suite into a skip, so the green is
not green-by-subtraction.

### 10.4b Coverage, and what the number is worth

T4 measures line coverage across the whole package: **48% total**, 14394 statements with
7480 missed, as job `970157`'s B2 measured it on the final tree. That figure is dominated by
the pre-existing package and is not a useful verdict on this branch. The per-module numbers
for the code the branch adds are:

| New module | Statements | Missed | Coverage |
|---|---|---|---|
| `make_ground_state.py` | 62 | 1 | **98%** |
| `make_time_evolution.py` | 51 | 1 | **98%** |
| `make_hamiltonian_learning.py` | 46 | 1 | **98%** |
| `make_quantum_labels.py` | 48 | 1 | **98%** |
| `make_engineered_kernel.py` | 40 | 1 | **98%** |
| `quantum_core.py` | 264 | 8 | **97%** |
| `quantum_selftest.py` | 107 | 8 | **93%** |
| `quantum_cli.py` | 56 | **0** | **100%** (was 0%; see D13) |
| **the eight together** | **674** | **21** | **97%** |

The `quantum_cli.py` zero is the one that mattered, and its 0 → 100% is the single
clearest measured effect of any fix in this audit: 56 statements, every one missed before
D13, every one traced after, in the same command under the same absent extra. It is worth
being precise about why a coverage number caught something `--help` did not: the CLI was exercised only through
`subprocess.run`, which `--cov` cannot trace, *and* no in-process test imported it, so its
flag-to-keyword translation and five-way dispatch were unasserted. D13 records the
diagnosis and the 43 tests that close it.

Two things this does not claim. Line coverage is not behaviour coverage — 98% on a
generator means its lines ran, and the physics claim rests on the seven self-test checks and
the byte-identity proof in §2.1, not on the percentage. And the comparison with the
pre-existing classical generators (18–45%) is context, not an argument that this branch
improved them; it did not touch them.

### 10.4c The docs build, re-run because B2 skipped it

Job **`970477`** (`cccxc430`, HEAD `fd4b0b6`, 64 files dirty — the change set of §10.2) exists
only to close the eight skips §10.2 traced to a missing `PATH` export. It exports
`/dccstor/boseukb/Q/envs/qbc/bin` first, so `shutil.which("pandoc")` finds the env's
`pandoc 3.11`, and it runs on the current tree. Empty stderr, LSF `Done`, 1m49s total.

| Check | Command | Result | Exit |
|---|---|---|---|
| C1 | `pytest tests/integration/test_docs_build.py -q -rs` | **10 passed, 0 skipped** in 73.11 s — the eight B2 skipped, plus the two it ran | **0** |
| C2 | `make -C docs html` — what the CI docs job runs, with `-W` in the Makefile | **0** lines matching `WARNING` or `ERROR`; 19 changed documents re-read, every new `data_generation` page among them | **0** |
| C3 | the branch's own pages on the built site | `quantum_datasets.html`, `tutorials/Quantum_Data/quantum_datasets_qprofiler.html`, and **18** `api/qbiocode.data_generation.*` pages | — |

The two checks are not redundant, and the difference is the part worth stating. C1's `build`
fixture runs `python -m sphinx` into a `tmp_path_factory` directory, so it is a **clean build
from an empty doctree**; C2 reuses `docs/build`, so it is the **incremental** build a
developer and the CI job actually get. A docs change can pass one and fail the other — a
stale doctree hides a broken cross-reference, and a clean build re-resolves every one — which
is why `quantum_datasets.html` carries its earlier timestamp in C3's listing: sphinx found
its source unchanged and did not rewrite it. C1 is the evidence that the page builds from
nothing; C3 is the evidence that it is on the site.

What this does **not** extend to: `docs/AUDIT_quantum_datasets_qprofiler.md` — this file — is
under `docs/`, not `docs/source/`, and is referenced by no toctree, so no build reads it and
no warning can be raised about it. It is a repository document, not a page of the site.

### 10.5 Caveats, stated rather than buried

- **The suite is green in this environment, which is not CI's.** T4 closes the one
  difference that was known to matter (the missing extra). Others remain unexercised
  here: CI runs a matrix of OSes and Python versions, and nothing in this audit says
  anything about Windows or 3.12.
- **Six notebooks pass, but notebook tests are `@pytest.mark.slow`**, so CI's default
  `-m` filter deselects them. They are green here and unreached there. That is a
  pre-existing property of the suite, not something this branch changed, but it means
  "CI is green" and "the notebooks run" are two separate claims.
- **`tests/integration/test_notebook_execution.py` re-executes against a copy of each
  notebook's own directory**, so a notebook depending on corpus paths outside its
  directory would pass here and fail for a user who has no `/dccstor` mount.
- **F6(b) is still open** and no test covers it; see *F6(a) implemented, F6(b) still
  open*.
- **The docs build is split across two CI jobs, and neither runs what §10.4c ran.** CI's
  `test` job installs `.[dev]`, which carries no `sphinx` (`pyproject.toml`: `dev` is pytest,
  black, isort, flake8, mypy), so `tests/integration/test_docs_build.py` skips at its
  module-level `pytest.importorskip("sphinx")` and **all ten of its assertions are dead in
  CI**. CI's `docs` job does install `.[docs]` and `apt-get install -y pandoc`
  (`ci.yml:176-179`) but runs `make html` under pytest-free `run:` steps, so its zero-warning
  guarantee comes from `-W` in `docs/Makefile`, not from the test. Job `970477` is the only
  place both halves run in one environment; that is stronger than either CI job alone, and it
  is also the reason the test's own skip guards have to be read carefully rather than trusted
  to fire only in broken environments.

### 10.6 HPO on the quantum corpus, measured end-to-end

The tiers above say the code is correct. They do not say the benchmark is *runnable*
with hyperparameter optimisation turned on, which is a separate question with a separate
answer: **functionally yes, and the cost is concentrated in one model.**

Job `969116` ran the packaged QProfiler through both `*_opt` paths on two corpus
datasets, one split, `embeddings: ['none']`, all eight tunable models — 6 classical + 2
quantum. It completed cleanly (`Successfully completed`, empty stderr, 8634 s total,
2/2 datasets) and wrote a 150-column `ModelResults.csv`: the 141 features (10 QBioCode
raw-data descriptors + 115 `mfe.*` + 16 `task.*`) plus the 9 non-feature columns.

**Tuning actually happened.** Every row carries a populated `BestParams_Tuned`, and the
values are model-specific rather than defaults echoed back — `svc_opt` selected
`kernel='linear', C=0.0746, gamma=0.635`, `qsvc_opt` selected `C=4.2897` (a continuous
draw, so the Optuna engine and not the grid), `pqk_opt` reports both its feature-map
choice and the nested classical `best_params`.

| Dataset | Best | Second | Worst | Reading |
|---|---|---|---|---|
| `te_n10_s4_seed0_tau1` | `mlp_opt` 0.925 | `pqk_opt` 0.900 (auc 0.966) | `qsvc_opt` 0.817 | The `te` family is genuinely learnable and all eight tuned models clear 0.81 |
| `eng_zz_n6_gq1_s0` | `mlp_opt` 0.544 | `lr_opt`/`nb_opt`/`svc_opt` 0.533 | `dt_opt`/`pqk_opt` 0.456 | **All eight at chance**, independently reproducing the default-`data_map` result in *The `eng` positive control* — this time with tuning on, so it is not a tuning artefact |

**The cost, which is the part that constrains a full sweep:**

| Model | mean s / dataset / split | max |
|---|---|---|
| `qsvc_opt` | 4304 | **7820** |
| `pqk_opt` | 150 | 283 |
| `mlp_opt` | 20 | 20 |
| `rf_opt` | 3 | 3 |
| `dt_opt`, `lr_opt`, `nb_opt`, `svc_opt` | 0.1 | 0.1 |

`qsvc_opt` is **1500× the cost of the next-cheapest quantum model and 5×10⁴× the
classical ones**, and 7820 s of that was a single dataset at n = 10 on one split. The
`FidelityQuantumKernel` builds an N×N kernel by circuit evaluation and HPO multiplies
that by the trial count, so this grows with both row count and trial budget. For the
106-CSV corpus at the default `iter: 2` that is not a sweep anyone should launch
unscoped: **budget `qsvc_opt` separately, or exclude it from the broad sweep and run it
on a chosen subset.** Nothing is broken — this is a cost property of fidelity kernels,
not a defect — but "the benchmark is ready to run with HPO" is only true with that
qualification stated.

One process note for anyone re-checking these job records: **LSF job IDs on this cluster
are recycled**, and `bhist -l 969116` returns completion records from April and May 2026
for earlier jobs that held the same number. The authoritative exit status is the
`Successfully completed` line in the job's own `.out` file, not `bhist`.

### 10.7 Style, measured rather than asserted

The suite verdict above is about behaviour. Style is a separate question with a separate
answer, and the answer is only interesting if the numbers are stated:

| Check | CI treatment | Whole package | The 8 new modules |
|---|---|---|---|
| `flake8 . --count --select=E9,F63,F7,F82` | **no** `continue-on-error` — the real gate | **0** | **0** |
| `black --check qbiocode/` | `continue-on-error: true`, and scoped to `qbiocode/` only | **101 of 146 files** would be reformatted | **8 of 8** |
| `flake8 --max-line-length=100` (black's setting in `pyproject.toml:190`) | not run by CI in this form | **510** E501 | **1** — `quantum_selftest.py:180`, an assert message at 112 characters |

So: the gate that can fail the build is clean, and the advisory checks flag the new modules
the same way they flag most of the package. The repository is **not** black-formatted — 101
files predate this branch — and there is no `[flake8]` section in `setup.cfg`, `tox.ini` or
`.flake8`, so a bare `flake8 .` runs at the 79-column default and reports 13684 findings
across the tree. Nothing here is a regression the branch introduced; the one line it adds to
the 510 is named above.

**Why it is recorded instead of fixed.** Running black over the eight new modules changes no
behaviour, but it changes the tree — and every number in §10.2 and §10.4c depends on the tree
being frozen from the moment jobs `970157` and `970477` were submitted. Trading a validated
tree for a cosmetic diff is the wrong trade, and a black pass that touches 8 files while
leaving 93 identically-flagged ones alone is also the wrong pass: it makes the codebase
*less* uniform, not more, and it is the kind of change that should be one reviewable commit
over all 101 files. The one E501 is left in place for the same reason and named here so it is
not discovered as a surprise.

## Appendix A — Reproduction

```bash
python qdata_gen.py selftest
# regenerate the notebook's datasets with the reference generator
python qdata_gen.py eng --n 4 --N 60 --seed 0 --out /tmp/rep
python qdata_gen.py ql  --encoding zz  --n 4 --N 60 --tau 1 --seed 0 --out /tmp/rep
python qdata_gen.py ql  --encoding evo --n 4 --N 60 --tau 1 --seed 0 --out /tmp/rep
python qdata_gen.py te  --n 6 --N 60 --s 4 --taus 0.25 1 4 --seed 0 --out /tmp/rep
python qdata_gen.py gs  --label e2e    --n 6 --N 80 --seed 0 --out /tmp/rep
python qdata_gen.py gs  --label sparse --n 6 --N 80 --seed 0 --out /tmp/rep
python qdata_gen.py hl  --n 4 --N 80 --times 0.5 --shots 1000 --seed 0 --out /tmp/rep

# The four below need qdata_gen.py on sys.path; it is not in the repo (audit_scripts/README.md).
python docs/audit_scripts/audit_diag.py /tmp/rep/x_view/eng_zz_n4_gq1_s0.csv /tmp/rep/x_view/ql_zz_n4_tau1_s0.csv /tmp/rep/x_view/ql_evo_n4_tau1_s0.csv  # 40 splits, all arms
python docs/audit_scripts/audit_eng.py    # F1: 100 splits, aligned vs reps4 vs scaled vs row-misaligned
python docs/audit_scripts/audit_ql.py     # F1 control: ql_zz aligned vs reps4
python docs/audit_scripts/audit_te.py     # F4, F12: tau ladder over 8 seeds; <r> vs n
```

The `data_key` collision (F2) and the low-variance metric (F6) were reproduced directly against the public `main` source.

## Appendix B — CSV fingerprints (reference generator, seed 0)

Compare with your files: `X = df.iloc[:, :-1]`, `y = df.iloc[:, -1]`.

| File | rows | `X.sum()` | `(y * arange(rows)).sum()` | `X[0,0]` |
|---|---|---|---|---|
| `eng_zz_n4_gq1_s0` | 60 | 127.5849286190 | 853 | 0.636961687321 |
| `ql_zz_n4_tau1_s0` | 60 | 127.5849286190 | 917 | 0.636961687321 |
| `ql_evo_n4_tau1_s0` | 60 | 127.5849286190 | 783 | 0.636961687321 |
| `te_n6_s4_seed0_tau1` | 60 | 183.0000000000 | 929 | 0 |
| `gs_sparse_n6_k0.5_s0` | 80 | 945.8677282222 | 1537 | 1.043624991465 |
