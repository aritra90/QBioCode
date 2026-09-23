"""Feature-map projection backends, and a factory that picks the right one.

The projected-kernel paths (:mod:`qbiocode.learning.compute_pqk`,
:mod:`qbiocode.learning.compute_qpl`) need one number per Pauli per qubit --
``<X_k>``, ``<Y_k>``, ``<Z_k>`` -- for every data row. Several simulators can
produce that, and which is fastest depends on the feature map, not on taste:

=========================  ==================================================
situation                  fastest measured backend
=========================  ==================================================
bounded entanglement,      Aer's MPS (``"aer_mps"``), ~10x quimb, ~210x dense
``reps <= 6``
bounded entanglement,      quimb MPS (``"quimb_mps"``) -- Aer degrades much
``reps >= 7``              faster with depth
``entanglement="full"``,   dense statevector (``"statevector"``)
``n <= ~20``
``entanglement="full"``,   quimb exact contraction (``"quimb_exact"``); the
``n >= ~24``               dense statevector has blown up by then
=========================  ==================================================

Those boundaries were measured on this machine with the ZZ feature map; see
``tutorial/MPS_vs_Statevector_Scaling.ipynb`` for the sweeps behind them and
:func:`choose_backend` for the rule itself.

Every projector here exposes the same two methods, ``project_row(x)`` ->
``(3, n_qubits)`` and ``project(X, n_jobs=...)`` -> ``(n_rows, 3, n_qubits)``,
which is the layout ``compute_qpl``/``compute_pqk`` save and then flatten.
"""

from __future__ import annotations

import os

import numpy as np

from qbiocode.utils import qutils
from qbiocode.utils.mps_backend import MPSFeatureMapProjector

#: Backend names accepted by :func:`make_projector`.
SUPPORTED_BACKENDS = ("statevector", "aer_mps", "quimb_mps", "quimb_permmps", "quimb_exact")

#: Entanglement patterns whose MPS bond dimension stays small -- single digits at
#: ``reps=1``. ``reverse_linear`` is the same nearest-neighbour chain as ``linear`` in
#: the opposite order, and measures identically (bond 2).
MPS_FRIENDLY_ENTANGLEMENT = frozenset(
    {"linear", "reverse_linear", "pairwise", "circular", "sca"}
)

#: Above this many repetitions quimb's MPS overtakes Aer's on bounded patterns
#: (measured crossover sits between reps 6 and 8 at n=20).
AER_MAX_REPS = 6

#: At or above this width, exact tensor-network contraction beats the dense
#: statevector on ``full`` entanglement (measured n=20 statevector wins, n=24 exact wins).
EXACT_MIN_QUBITS = 24


def choose_backend(n_qubits, entanglement="linear", reps=1):
    """Return the measured-fastest backend name for a feature-map configuration.

    Args:
        n_qubits (int): number of features / qubits.
        entanglement (str): the feature map's entanglement pattern.
        reps (int): feature-map repetitions.

    Returns:
        str: one of :data:`SUPPORTED_BACKENDS`.

    Notes:
        This is a lookup of measured results, not a cost model -- it does not
        extrapolate. Outside the ranges that were swept (``n <= 256``,
        ``reps <= 12``) treat it as a starting guess and re-measure.
    """
    if entanglement not in MPS_FRIENDLY_ENTANGLEMENT:
        # 'full' (or a custom pattern we have not characterised): an MPS cannot
        # compress this state, so the choice is dense vs exact contraction.
        return "quimb_exact" if n_qubits >= EXACT_MIN_QUBITS else "statevector"
    return "aer_mps" if reps <= AER_MAX_REPS else "quimb_mps"


class StatevectorFeatureMapProjector:
    """Dense ``2**n`` statevector projection.

    This exists because the primitive the pipelines currently use,
    ``qiskit.primitives.StatevectorEstimator``, **re-simulates the circuit once per
    observable**. Measured at ``n=20``: 0.516 s for one observable, 10.07 s for twenty,
    exactly linear -- so the 3n observables this workload needs cost 30.5 s, against
    0.53 s for the identical numbers computed here. That is a ~58x overhead unrelated
    to simulation method, and it is the single cheapest speedup available to
    ``compute_qpl``/``compute_pqk``.

    Args:
        n_qubits (int): number of features / qubits.
        encoding (str): ``'Z'``, ``'ZZ'`` or ``'P'``, as
            :func:`qbiocode.utils.qutils.get_feature_map` accepts.
        reps (int): feature-map repetitions.
        entanglement (str): entanglement pattern.
        data_map_func (callable, optional): passed through to the feature map.

    Notes:
        Memory is the binding constraint: ``2**30`` complex128 amplitudes is 16 GiB,
        so this is unusable much past 30 qubits regardless of time.
    """

    def __init__(self, n_qubits, encoding="ZZ", reps=1, entanglement="linear",
                 data_map_func=None):
        from qiskit.quantum_info import SparsePauliOp

        self.n_qubits = int(n_qubits)
        self.encoding = encoding
        self.reps = reps
        self.entanglement = entanglement

        self.circuit, _ = qutils.get_feature_map(
            feature_map=encoding, feat_dimension=n_qubits, reps=reps,
            entanglement=entanglement, data_map_func=data_map_func,
        )
        # Built once. Each is a single-term operator, so expectation_value is a cheap
        # sparse contraction against an already-computed amplitude vector.
        self._observables = [
            SparsePauliOp.from_sparse_list([(p, [k], 1.0)], num_qubits=self.n_qubits)
            for p in "XYZ" for k in range(self.n_qubits)
        ]

    def project_row(self, x):
        """Return the ``(3, n_qubits)`` array of ``<X>``, ``<Y>``, ``<Z>`` for one row."""
        from qiskit.quantum_info import Statevector

        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.circuit.num_parameters:
            raise ValueError(
                f"x has {x.size} value(s) but the {self.encoding} feature map on "
                f"{self.n_qubits} qubits takes {self.circuit.num_parameters}."
            )
        sv = Statevector(self.circuit.assign_parameters(x))
        # One simulation, then 3n reads off it -- the whole point of this class.
        evs = [sv.expectation_value(o).real for o in self._observables]
        return np.asarray(evs).reshape(3, self.n_qubits)

    def project(self, X, progress_every=100, n_jobs=1):
        """Project a design matrix; see :meth:`MPSFeatureMapProjector.project`."""
        return _project_matrix(self, X, progress_every, n_jobs)

    def fidelity_estimate(self):
        """Always exactly 1.0 -- a dense statevector performs no truncation."""
        return 1.0

    def observed_max_bond(self):
        """``None`` -- a dense statevector has no bond dimension."""
        return None


class AerMPSFeatureMapProjector:
    """Aer's matrix-product-state simulator, all 3n expectations in one run.

    Fastest of everything measured for bounded-entanglement maps at ``reps <= 6``.

    Notes:
        Two ordering traps, each of which costs 3-7x if missed:

        - Lower the feature map with ``basis_gates=...``, **not** by passing the
          simulator as a transpile target. A target applies Aer's default 63-qubit
          coupling map, which hard-fails above 63 qubits with
          ``CircuitTooWideForTarget`` and inserts routing SWAPs below it.
        - Transpile the *parameterised* circuit once and bind angles per row.
          Transpiling per row is 65-70% of the runtime at low reps and is not
          simulation work. ``save_expectation_value`` must be attached *after* the
          basis translation, which cannot represent it.
    """

    #: A basis Aer executes directly and every gate of which quimb also understands.
    BASIS = ("u", "p", "h", "rz", "rx", "ry", "cx", "cz")

    def __init__(self, n_qubits, encoding="ZZ", reps=1, entanglement="linear",
                 max_bond=None, data_map_func=None):
        from qiskit import transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator

        self.n_qubits = int(n_qubits)
        self.encoding = encoding
        self.reps = reps
        self.entanglement = entanglement
        self.max_bond = max_bond

        options = {"method": "matrix_product_state"}
        if max_bond:
            options["matrix_product_state_max_bond_dimension"] = max_bond
        self._sim = AerSimulator(**options)

        feature_map, _ = qutils.get_feature_map(
            feature_map=encoding, feat_dimension=n_qubits, reps=reps,
            entanglement=entanglement, data_map_func=data_map_func,
        )
        template = transpile(feature_map, basis_gates=list(self.BASIS),
                             optimization_level=0)
        for p in "XYZ":
            for k in range(self.n_qubits):
                template.save_expectation_value(SparsePauliOp(p), [k], label=f"{p}{k}")
        self.circuit = template

    def project_row(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_qubits:
            raise ValueError(
                f"x has {x.size} value(s) but this feature map takes {self.n_qubits}."
            )
        result = self._sim.run(self.circuit.assign_parameters(x), shots=1).result()
        if not result.success:
            raise RuntimeError(f"Aer simulation failed: {result.status}")
        data = result.data()
        return np.asarray(
            [[float(data[f"{p}{k}"]) for k in range(self.n_qubits)] for p in "XYZ"]
        )

    def project(self, X, progress_every=100, n_jobs=1):
        return _project_matrix(self, X, progress_every, n_jobs)

    def fidelity_estimate(self):
        """``None`` -- Aer exposes no directly comparable truncation readout.

        This is the reason to prefer a quimb backend when you need to *know* whether
        a configuration truncated, even though Aer is faster at low reps.
        """
        return None

    def observed_max_bond(self):
        """``None`` -- not reported by this path."""
        return None


def _project_chunk(projector, chunk):
    """joblib worker: project a contiguous block of rows."""
    out = np.empty((len(chunk), 3, projector.n_qubits))
    for i, row in enumerate(chunk):
        out[i] = projector.project_row(row)
    return out


def _project_matrix(projector, X, progress_every, n_jobs):
    """Shared ``project`` implementation for the non-quimb projectors."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_rows, n_features); got shape {X.shape}.")

    if n_jobs == 1:
        out = np.empty((len(X), 3, projector.n_qubits))
        for i, row in enumerate(X):
            if progress_every and i % progress_every == 0:
                print(f"at datapoint {i}")
            out[i] = projector.project_row(row)
        return out

    from joblib import Parallel, delayed

    n_workers = os.cpu_count() if n_jobs in (-1, None) else abs(n_jobs)
    chunks = np.array_split(X, max(1, min(n_workers, len(X))))
    blocks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_project_chunk)(projector, c) for c in chunks
    )
    return np.concatenate(blocks, axis=0)


def make_projector(n_qubits, encoding="ZZ", reps=1, entanglement="linear",
                   backend="auto", max_bond=None, data_map_func=None):
    """Build a projector, choosing the backend from measurements unless told otherwise.

    Args:
        n_qubits (int): number of features / qubits.
        encoding (str): ``'Z'``, ``'ZZ'`` or ``'P'``.
        reps (int): feature-map repetitions.
        entanglement (str): entanglement pattern.
        backend (str): ``"auto"`` (the default) defers to :func:`choose_backend`, or
            name one of :data:`SUPPORTED_BACKENDS` explicitly.
        max_bond (int or None): bond-dimension cap for the MPS backends. Leave at
            ``None`` (exact) unless you are deliberately trading accuracy for speed;
            then check ``fidelity_estimate()``, which is the only signal that the
            features became approximate.
        data_map_func (callable, optional): passed through to the feature map. The
            projected-kernel paths use
            :func:`qbiocode.utils.qutils.unit_coefficient_data_map`; omitting it
            silently produces different features from those pipelines.

    Returns:
        A projector exposing ``project_row``, ``project``, ``fidelity_estimate`` and
        ``observed_max_bond``.

    Raises:
        ValueError: if ``backend`` is not ``"auto"`` or a supported name.
    """
    if backend == "auto":
        backend = choose_backend(n_qubits, entanglement=entanglement, reps=reps)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. Expected 'auto' or one of "
            f"{SUPPORTED_BACKENDS}."
        )

    # Refuse rather than ignore. `max_bond` is a bond-dimension cap, which only the two
    # truncating MPS backends have; the dense statevector and the exact contraction have
    # no bond dimension at all. Silently dropping it would let a caller believe they had
    # requested an approximation and were measuring its cost, when they were measuring
    # the exact computation -- the same failure mode as a stale cache.
    if max_bond is not None and backend in ("statevector", "quimb_exact"):
        raise ValueError(
            f"max_bond={max_bond!r} caps an MPS bond dimension, which the {backend!r} "
            f"backend does not have ({'a dense statevector stores all amplitudes' if backend == 'statevector' else 'exact contraction performs no truncation'}). "
            f"Drop max_bond, or choose 'quimb_mps'/'quimb_permmps'/'aer_mps'."
        )

    common = dict(n_qubits=n_qubits, encoding=encoding, reps=reps,
                  entanglement=entanglement, data_map_func=data_map_func)

    if backend == "statevector":
        return StatevectorFeatureMapProjector(**common)
    if backend == "aer_mps":
        return AerMPSFeatureMapProjector(max_bond=max_bond, **common)

    simulator = {"quimb_mps": "mps", "quimb_permmps": "permmps",
                 "quimb_exact": "exact"}[backend]
    return MPSFeatureMapProjector(max_bond=max_bond, simulator=simulator, **common)
