"""Matrix-product-state simulation of QBioCode feature maps, via quimb.

Why this exists
---------------
The projected-kernel paths (:mod:`qbiocode.learning.compute_pqk`,
:mod:`qbiocode.learning.compute_qpl`) need one number per Pauli per qubit --
``<X_k>``, ``<Y_k>``, ``<Z_k>`` -- for every data row. Qiskit's
``StatevectorEstimator`` gets those by materialising the full ``2**n``
amplitude vector, so ``feat_dimension`` is capped near 30 qubits by RAM alone
(2**30 complex128 = 16 GiB) long before it is capped by time.

A feature map is not a generic circuit, though. ``ZFeatureMap`` has no
entanglement at all, and ``ZZFeatureMap``/``PauliFeatureMap`` with
``entanglement="linear"`` entangle only nearest neighbours for a handful of
reps. The Schmidt rank across any cut therefore stays tiny, and an MPS
represents the state *exactly* with a bond dimension that does not grow with
``n``. Cost becomes linear in the qubit count instead of exponential, which is
what makes 100+ feature columns reachable.

This module is deliberately separate from :mod:`qbiocode.utils.qutils`: quimb is
an optional dependency, and ``qutils`` is imported unconditionally by
``import qbiocode``.

Accuracy
--------
With ``max_bond=None`` (the default) and a small ``cutoff``, the simulation is
*exact* up to floating point -- verified against ``StatevectorEstimator`` to
~1e-15 for linear-entanglement ``ZZFeatureMap``. Set ``max_bond`` only when you
knowingly want an approximation (deep reps, or ``entanglement="full"``), and
read :meth:`MPSFeatureMapProjector.fidelity_estimate` afterwards to see what
the truncation cost you. A fidelity that is not ~1.0 means the features are
approximate, and that has to be reported alongside any accuracy number.

Examples
--------
>>> from qbiocode.utils.mps_backend import MPSFeatureMapProjector
>>> proj = MPSFeatureMapProjector(n_qubits=64, encoding="ZZ", reps=1)
>>> features = proj.project(X)          # (len(X), 3, 64)
>>> proj.fidelity_estimate()            # 1.0 -> exact
1.0
"""

from __future__ import annotations

import os

import numpy as np

#: Qiskit basis the feature map is lowered to before hand-off. Every name here
#: has a direct quimb counterpart in ``_QISKIT_TO_QUIMB``; transpiling to a
#: closed basis is what lets the translation below be a total function rather
#: than a best-effort lookup that fails on row 400 of a long run.
TRANSPILE_BASIS = ("u", "u3", "u2", "p", "rx", "ry", "rz", "h", "x", "y", "z", "cx", "cz", "rzz")

#: Qiskit instruction name -> (quimb gate label, number of angle parameters).
#: quimb's registry spells these in upper case; see
#: ``quimb.tensor.circuit.gates.ALL_GATES``.
_QISKIT_TO_QUIMB = {
    "h": ("H", 0),
    "x": ("X", 0),
    "y": ("Y", 0),
    "z": ("Z", 0),
    "s": ("S", 0),
    "sdg": ("SDG", 0),
    "t": ("T", 0),
    "tdg": ("TDG", 0),
    "sx": ("SX", 0),
    "rx": ("RX", 1),
    "ry": ("RY", 1),
    "rz": ("RZ", 1),
    "p": ("PHASE", 1),
    "u1": ("U1", 1),
    "u2": ("U2", 2),
    "u": ("U3", 3),
    "u3": ("U3", 3),
    "cx": ("CX", 0),
    "cy": ("CY", 0),
    "cz": ("CZ", 0),
    "swap": ("SWAP", 0),
    "iswap": ("ISWAP", 0),
    "cp": ("CPHASE", 1),
    "cu1": ("CU1", 1),
    "crx": ("CRX", 1),
    "cry": ("CRY", 1),
    "crz": ("CRZ", 1),
    "rxx": ("RXX", 1),
    "ryy": ("RYY", 1),
    "rzz": ("RZZ", 1),
    "ccx": ("CCX", 0),
}

#: Simulator backends quimb offers for this workload, and when each wins (measured on
#: the ZZ feature map, all 3n single-site Pauli expectations per row -- see
#: ``tutorial/MPS_vs_Statevector_Scaling.ipynb``):
#:
#: - ``"mps"`` (:class:`quimb.tensor.CircuitMPS`) -- the default. Exact and linear in
#:   ``n`` whenever entanglement is bounded (``linear``/``pairwise``/``circular``/``sca``).
#: - ``"permmps"`` (:class:`quimb.tensor.CircuitPermMPS`) -- tracks qubit ordering lazily
#:   instead of swapping non-local gates back. Worth 8-40% on the non-local patterns;
#:   it does not change any regime.
#: - ``"exact"`` (:class:`quimb.tensor.Circuit`) -- full tensor-network contraction, no
#:   truncation, cotengra-optimised path. Loses badly on bounded entanglement but is the
#:   *fastest of everything* for ``entanglement="full"`` at >= 24 qubits, where the dense
#:   statevector has blown up (measured n=24 full: exact 20 s, MPS 41 s, statevector 100 s).
SUPPORTED_SIMULATORS = ("mps", "permmps", "exact")

#: Instructions carrying no unitary action on the state. ``barrier`` and
#: ``measure`` are dropped rather than raised on: qiskit's transpiler inserts
#: barriers into feature maps, and dropping them is correct for expectation
#: values.
_IGNORED_INSTRUCTIONS = frozenset({"barrier", "measure", "delay", "id"})


def _require_quimb():
    """Import quimb, turning the ImportError into an actionable message.

    quimb is an optional dependency; the bare ``ModuleNotFoundError`` names the
    module but not the install command or the fact that ``cotengra`` is a hard
    requirement of quimb's contraction machinery.
    """
    try:
        import quimb
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The MPS backend requires quimb (and its cotengra dependency), "
            "which is not installed in this environment. Install it with "
            "`pip install 'quimb[recommended]'`, or, for a local checkout, "
            "`pip install -e /path/to/quimb cotengra`."
        ) from exc
    return quimb, qtn


def qiskit_circuit_to_quimb_gates(circuit):
    """Translate a *fully bound* qiskit circuit into a list of quimb gates.

    Args:
        circuit (QuantumCircuit): a circuit with no unbound ``Parameter``\\ s,
            already transpiled into :data:`TRANSPILE_BASIS` (or any subset of
            :data:`_QISKIT_TO_QUIMB`).

    Returns:
        list[quimb.tensor.circuit.Gate]: gates in application order.

    Raises:
        ValueError: if the circuit still holds unbound parameters, or contains
            an instruction with no quimb counterpart. Both are raised here,
            naming the offending instruction, rather than surfacing deep inside
            quimb as a ``KeyError`` on an upper-cased gate label.

    Notes:
        The QASM2 round trip (``qasm2.dumps`` ->
        ``quimb.tensor.circuit.qasm.parse_openqasm2_str``) is the other way to
        do this and is equivalent numerically, but it re-serialises and
        re-parses a string for every data row. Going through ``circuit.data``
        keeps the per-row cost proportional to the gate count.
    """
    _, qtn = _require_quimb()

    if circuit.num_parameters:
        raise ValueError(
            f"circuit still has {circuit.num_parameters} unbound parameter(s) "
            f"{[p.name for p in circuit.parameters][:5]}; bind the data row "
            f"with `circuit.assign_parameters(x)` before translating, because "
            f"quimb gates need numeric angles."
        )

    # Index lookup once: `circuit.find_bit` is a Python-level call per qubit per
    # gate otherwise, which dominates the translation for wide circuits.
    qubit_index = {bit: i for i, bit in enumerate(circuit.qubits)}

    gates = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in _IGNORED_INSTRUCTIONS:
            continue
        if name == "global_phase":  # pragma: no cover - not a real instruction
            continue

        try:
            label, n_params = _QISKIT_TO_QUIMB[name]
        except KeyError:
            raise ValueError(
                f"Instruction {name!r} has no quimb equivalent. Transpile the "
                f"circuit into a supported basis first, e.g. "
                f"`transpile(qc, basis_gates={list(TRANSPILE_BASIS)})`. "
                f"Supported names: {sorted(_QISKIT_TO_QUIMB)}."
            ) from None

        params = instruction.operation.params
        if len(params) != n_params:
            raise ValueError(
                f"Instruction {name!r} carries {len(params)} parameter(s) but "
                f"quimb's {label} takes {n_params}; this means the qiskit "
                f"instruction is not the gate this table assumes."
            )

        qubits = tuple(qubit_index[q] for q in instruction.qubits)
        gates.append(qtn.Gate(label, tuple(float(p) for p in params), qubits))

    # Qiskit tracks an overall global phase that QASM export folds into a gate;
    # it cancels in every expectation value, so it is intentionally dropped.
    return gates


def _project_chunk(projector, chunk):
    """joblib worker: project a contiguous block of rows.

    Returns the block's projections plus its truncation diagnostics, which the parent
    merges -- a worker's own attribute updates die with its process.
    """
    out = np.empty((len(chunk), 3, projector.n_qubits))
    for i, row in enumerate(chunk):
        out[i] = projector.project_row(row)
    return out, projector.fidelity_estimate(), projector.observed_max_bond()


class MPSFeatureMapProjector:
    """Compute per-qubit Pauli expectation values of a feature map via an MPS.

    This is a drop-in replacement for the ``StatevectorEstimator`` block inside
    :func:`qbiocode.learning.compute_qpl.compute_qpl` and
    :func:`qbiocode.learning.compute_pqk.compute_pqk`, and returns the same
    ``(n_rows, 3, n_qubits)`` layout those functions save to ``.npy``.

    Args:
        n_qubits (int): number of features / qubits.
        encoding (str): feature map name, as accepted by
            :func:`qbiocode.utils.qutils.get_feature_map` -- ``'Z'``, ``'ZZ'``
            or ``'P'``.
        reps (int): feature-map repetitions.
        entanglement (str): entanglement pattern. ``'linear'`` keeps the MPS
            bond dimension flat in ``n_qubits`` and is what makes large widths
            cheap; ``'full'`` makes every pair adjacent by swapping, which costs
            O(n^2) two-qubit gates and grows the bond dimension quickly.
        max_bond (int or None): bond-dimension cap. ``None`` means no cap, i.e.
            exact simulation -- the right default, because a silently truncated
            feature is worse than a slow one. Set an integer only to trade
            accuracy for speed deliberately, and check
            :meth:`fidelity_estimate`. Ignored when ``simulator="exact"``, which
            never truncates.
        simulator (str): one of :data:`SUPPORTED_SIMULATORS`. See that constant
            for which one wins where.
        cutoff (float): singular values below this are discarded. ``1e-12`` is
            effectively exact for these circuits.
        data_map_func (callable, optional): passed through to the feature map,
            e.g. :func:`qbiocode.utils.qutils.unit_coefficient_data_map`.

    Attributes:
        circuit (QuantumCircuit): the transpiled, still-parameterised feature
            map. Exposed so callers can inspect depth / gate counts.
    """

    def __init__(
        self,
        n_qubits,
        encoding="ZZ",
        reps=1,
        entanglement="linear",
        max_bond=None,
        cutoff=1e-12,
        data_map_func=None,
        simulator="mps",
    ):
        quimb, qtn = _require_quimb()
        from qiskit import transpile

        if max_bond is not None and simulator == "exact":
            raise ValueError(
                f"max_bond={max_bond!r} caps an MPS bond dimension, but "
                f"simulator='exact' contracts the full tensor network and performs no "
                f"truncation, so the cap would be silently ignored. Drop max_bond, or "
                f"use simulator='mps'/'permmps'."
            )
        if simulator not in SUPPORTED_SIMULATORS:
            raise ValueError(
                f"Unsupported simulator {simulator!r}. Expected one of "
                f"{SUPPORTED_SIMULATORS}; see SUPPORTED_SIMULATORS for which "
                f"one is fastest for which entanglement pattern."
            )

        from qbiocode.utils import qutils

        feature_map, _ = qutils.get_feature_map(
            feature_map=encoding,
            feat_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            data_map_func=data_map_func,
        )

        self.n_qubits = int(n_qubits)
        self.encoding = encoding
        self.reps = reps
        self.entanglement = entanglement
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.simulator = simulator
        # Held on the instance so the per-row path does no imports and no lookups.
        self._qtn = qtn
        self._circuit_cls = {
            "mps": qtn.CircuitMPS,
            "permmps": qtn.CircuitPermMPS,
            "exact": qtn.Circuit,
        }[simulator]

        # Transpiled once, parameterised. Per row we only re-bind angles, so the
        # transpiler -- which is the expensive part at 100+ qubits -- runs once
        # for the whole dataset rather than once per sample.
        self.circuit = transpile(
            feature_map,
            basis_gates=list(TRANSPILE_BASIS),
            optimization_level=1,
        )
        self._paulis = [quimb.pauli(p) for p in "XYZ"]
        # Dense 2x2 copies, for tracing against a reduced density matrix.
        self._pauli_arrays = [np.asarray(P) for P in self._paulis]
        self._last_fidelity = None
        self._last_max_bond = None

    def _build_state(self, x):
        """Bind one data row and evolve the simulator's state through the circuit."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.circuit.num_parameters:
            raise ValueError(
                f"x has {x.size} value(s) but the {self.encoding} feature map "
                f"on {self.n_qubits} qubits takes "
                f"{self.circuit.num_parameters} parameter(s)."
            )
        gates = qiskit_circuit_to_quimb_gates(self.circuit.assign_parameters(x))

        kwargs = {"N": self.n_qubits}
        if self.simulator != "exact":
            # CircuitMPS/CircuitPermMPS truncate; Circuit has no such knobs.
            kwargs.update(max_bond=self.max_bond, cutoff=self.cutoff)
        circ = self._circuit_cls(**kwargs)
        circ.apply_gates(gates)
        return circ

    def _read_expectations(self, circ):
        """All 3n single-site Pauli expectations, cheaply.

        For a plain MPS this forms each site's 2x2 reduced density matrix *once* and
        traces it against X, Y and Z, sweeping sites in increasing order. That matters
        because every expectation value moves the MPS orthogonality centre: asking for
        X on all sites, then Y, then Z drags the centre across the chain 3 times and
        recomputes each site's environment 3 times. One left-to-right sweep of RDMs
        measured 1.6-1.9x faster on shallow maps and 3.3x on ``reps=8``, to ~1e-14.

        ``permmps`` gets the same sweep, but over its *unpermuted* MPS: it stores the
        state in a lazily-permuted order, so site ``i`` holds qubit ``circ.qubits[i]``
        and results have to be scattered back to qubit indices. Skipping that and
        letting it fall back to the slow read is not a neutral choice -- it made
        ``CircuitPermMPS`` look 20% *slower* than ``CircuitMPS`` on ``linear`` in an
        earlier draft, which was a comparison of read strategies, not of simulators.

        ``exact`` has no MPS to canonicalize and goes through ``local_expectation``.
        """
        n = self.n_qubits
        out = np.empty((3, n))

        if self.simulator in ("mps", "permmps"):
            if self.simulator == "permmps":
                psi = circ.get_psi_unordered()
                sites_to_qubits = list(circ.qubits)
            else:
                psi = circ._psi
                sites_to_qubits = range(n)
            # Share quimb's OWN bookkeeping dict rather than a fresh one. It records
            # the orthogonality centre, and `fidelity_estimate()` computes a norm over
            # that window -- handing the sweep a private dict left quimb's copy stale
            # and made fidelity_estimate() return 2.0 on an untruncated state.
            info = circ.gate_opts["info"] if self.simulator == "mps" else {}
            for site, qubit in enumerate(sites_to_qubits):
                rho = np.asarray(
                    psi.partial_trace_to_dense_canonical(site, normalized=True, info=info)
                )
                for j, P in enumerate(self._pauli_arrays):
                    out[j, qubit] = np.trace(P @ rho).real
        else:
            for k in range(n):
                for j, P in enumerate(self._paulis):
                    out[j, k] = circ.local_expectation(P, k).real
        return out

    def _record_diagnostics(self, circ):
        """Track worst-case truncation seen so far. ``exact`` never truncates."""
        if self.simulator == "exact":
            self._last_fidelity = 1.0
            return
        fidelity = circ.fidelity_estimate()
        if self._last_fidelity is None or fidelity < self._last_fidelity:
            self._last_fidelity = fidelity
        psi = circ.get_psi_unordered() if self.simulator == "permmps" else circ.psi
        self._last_max_bond = max(self._last_max_bond or 0, psi.max_bond())

    def project_row(self, x):
        """Return the ``(3, n_qubits)`` array of ``<X>``, ``<Y>``, ``<Z>`` for one row.

        Args:
            x (array_like): one data row, of length ``circuit.num_parameters``.

        Returns:
            numpy.ndarray: shape ``(3, n_qubits)``, real valued.
        """
        circ = self._build_state(x)
        # BEFORE reading: reading canonicalises the state in place, and the fidelity
        # estimate is a norm over the current orthogonality window. Recorded here it
        # describes the state as the circuit left it, which is what callers mean.
        self._record_diagnostics(circ)
        return self._read_expectations(circ)

    def project(self, X, progress_every=100, n_jobs=1):
        """Project a whole design matrix.

        Args:
            X (array_like): shape ``(n_rows, n_features)``.
            progress_every (int or None): print a progress line every this many
                rows, matching the existing ``compute_qpl`` output. ``None``
                silences it. Ignored when ``n_jobs != 1``.
            n_jobs (int): rows are independent, so this splits them across
                processes with joblib. ``1`` (the default) stays in-process;
                ``-1`` uses every core. This is the parallelism that actually
                pays for this workload -- see the note below on why not threads,
                and not a GPU.

        Returns:
            numpy.ndarray: shape ``(n_rows, 3, n_qubits)`` -- the same layout
            ``compute_qpl``/``compute_pqk`` save and then reshape to
            ``(n_rows, 3 * n_qubits)``.

        Notes:
            **Processes, not threads.** The per-row work is numpy/numba inside one
            Python interpreter, so threads would contend on the GIL; and QBioCode
            already requires ``OMP_NUM_THREADS=1`` because several OpenMP runtimes
            are mapped into the process. joblib's loky backend pins inner thread
            counts in its workers, which is what makes this safe rather than
            oversubscribed.

            Truncation diagnostics are merged back from the workers, so
            :meth:`fidelity_estimate` and :meth:`observed_max_bond` remain
            meaningful after a parallel run.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_rows, n_features); got shape {X.shape}."
            )

        if n_jobs == 1:
            out = np.empty((len(X), 3, self.n_qubits))
            for i, row in enumerate(X):
                if progress_every and i % progress_every == 0:
                    print(f"at datapoint {i}")
                out[i] = self.project_row(row)
            return out

        from joblib import Parallel, delayed

        # One chunk per worker rather than one task per row: the projector (and its
        # transpiled circuit) is pickled once per chunk instead of once per row.
        n_workers = os.cpu_count() if n_jobs in (-1, None) else abs(n_jobs)
        n_chunks = max(1, min(n_workers, len(X)))
        chunks = np.array_split(X, n_chunks)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_project_chunk)(self, chunk) for chunk in chunks
        )
        arrays, fidelities, bonds = zip(*results)
        for f in fidelities:
            if f is not None and (self._last_fidelity is None or f < self._last_fidelity):
                self._last_fidelity = f
        for b in bonds:
            if b is not None:
                self._last_max_bond = max(self._last_max_bond or 0, b)
        return np.concatenate(arrays, axis=0)

    def fidelity_estimate(self):
        """Worst-case fidelity estimate seen so far, or ``None`` before any run.

        ``1.0`` means nothing was truncated and the projections are exact. Any
        value below that is the fraction of the state's norm retained, and is
        the number to quote when reporting results from a capped ``max_bond``.
        """
        return self._last_fidelity

    def observed_max_bond(self):
        """Largest bond dimension the state actually reached, or ``None``.

        Useful for choosing ``max_bond``: if this comes back well below your cap
        (or below ``2**(n//2)``), the MPS is compressing the state for free and
        there is nothing to gain from a cap.
        """
        return self._last_max_bond
