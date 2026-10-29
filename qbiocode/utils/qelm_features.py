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

"""The two pieces a quantum extreme learning machine adds to a feature map.

A QELM reads its features off a circuit that is the data feature map followed by a
*fixed, untrained* random unitary -- the reservoir -- and it reads more of them than
the ``3n`` single-qubit Pauli expectations the projected-kernel paths take. This module
builds both halves and nothing else: no model, no fit, no projection. The projectors in
:mod:`qbiocode.utils.projection` and :mod:`qbiocode.utils.mps_backend` accept what it
returns, so one observable pool and one reservoir definition serve every simulator
backend.

Why these two are the whole of it
---------------------------------
Everything else a QELM needs already exists. The data encoding is
:func:`qbiocode.utils.qutils.get_feature_map`, the input scaling is
:func:`qbiocode.utils.qutils.apply_bandwidth`, and one state preparation per row with
the observables read off it is what
:class:`qbiocode.utils.projection.StatevectorFeatureMapProjector` was written to do.
What is missing is a circuit to append and a list of things to measure.

The seed is configuration, never a hyperparameter
-------------------------------------------------
:func:`qelm_reservoir` and :func:`qelm_observables` are deterministic in ``seed``, and
that seed must come from the run's configuration -- not from a search space. Tuning it
would select the reservoir that happens to suit the validation holdout, so the reported
score would be a maximum over however many draws the budget allowed rather than the
performance of one reservoir. Every other knob here (depth, pool, width) is a property
of the architecture and is tunable; the seed is not.

Observable specs
----------------
An observable is a ``(pauli, qubits)`` pair -- ``("ZZ", (0, 3))`` is ``Z_0 Z_3``. The
convention is **index-based**: character ``i`` of ``pauli`` acts on ``qubits[i]``. That
is exactly what :meth:`qiskit.quantum_info.SparsePauliOp.from_sparse_list` means by the
same pair, and it agrees with quimb's
``local_expectation(kron(A, B), (qa, qb))`` (both verified against a state with
``<X_0> = <Y_1> = 1``). It is *not* what a bare ``SparsePauliOp("XY")`` label means --
those are little-endian, so the rightmost character is qubit 0 -- which is why every
consumer here builds its operator through ``from_sparse_list`` rather than from the
label. Getting that backwards is silent: the features stay self-consistent and the
accuracy barely moves, because a permutation of columns is invisible to a rotationally
symmetric readout.

MPS friendliness
----------------
:data:`RESERVOIR_ENTANGLEMENT` is nearest-neighbour on purpose. An unstructured
(Haar) reservoir cannot be compressed by a matrix product state, which would cap a
QELM at the ~20-24 qubits a dense statevector reaches and give up the scaling the
optional ``[mps]`` extra buys for ``pqk``/``qpl``. Here the randomness lives in the
*angles* and the connectivity stays a brick wall, so the bond dimension stays bounded
and :func:`qbiocode.utils.projection.choose_backend` can still route a QELM to an MPS.
Pool choice matters for the same reason: ``'xyz'`` and ``'xyz+zz'`` are local, and
``'random_pauli'`` is capped at weight 2 by default.
"""

from __future__ import annotations

import math

import numpy as np

#: The single-qubit Paulis an observable may be built from. ``'I'`` is deliberately
#: absent: under the index-based convention above an identity character is a qubit the
#: operator does not act on, which is spelled by leaving that qubit out of ``qubits``.
#: Allowing both spellings would make two specs that name one observable compare
#: unequal, and the pools below de-duplicate by spec.
PAULI_CHARS = ("X", "Y", "Z")

#: How :func:`qelm_observables` may choose what to measure.
#:
#: * ``'xyz'`` -- ``<X_k>``, ``<Y_k>``, ``<Z_k>`` on every qubit: ``3n`` features, in
#:   the order and the layout the projected-kernel paths already use. This is the pool
#:   that makes a zero-depth QELM comparable to PQK (see :func:`qelm_reservoir`).
#: * ``'xyz+zz'`` -- the above plus ``<Z_k Z_{k+1}>`` along the chain: ``4n - 1``
#:   features. The cheapest way to add two-body information, and local, so it costs an
#:   MPS nothing.
#: * ``'random_pauli'`` -- ``n_features`` distinct Pauli strings drawn from the seeded
#:   pool of all non-identity Paulis of weight at most ``max_weight``. The only pool
#:   whose width is free, and so the one a search over the number of quantum features
#:   varies.
MEASUREMENTS = ("xyz", "xyz+zz", "random_pauli")

#: Default weight cap for ``'random_pauli'``: one- and two-body operators only. Higher
#: weights are permitted but grow the MPS work and, past weight ~3, mostly add features
#: whose expectation is near zero for these shallow circuits -- columns that cost a
#: measurement and carry little signal.
DEFAULT_MAX_WEIGHT = 2

#: The reservoir's entanglement pattern. Nearest-neighbour, for the reason in the module
#: docstring: it is what keeps a QELM MPS-simulable. Not a parameter, because a config
#: able to select ``'full'`` here would silently move the arm off the tensor-network
#: backends and onto a dense statevector.
RESERVOIR_ENTANGLEMENT = "linear"

#: Rotation axes of one reservoir layer. ``ry``/``rz`` spans SU(2) per qubit and lowers
#: into both the Aer and the quimb translation bases without an extra pass.
RESERVOIR_ROTATIONS = ("ry", "rz")


def _check_n_qubits(n_qubits):
    if isinstance(n_qubits, (bool, np.bool_)) or not isinstance(
        n_qubits, (int, np.integer)
    ) or n_qubits < 1:
        raise ValueError(
            f"n_qubits is the number of features (one qubit each) and must be a "
            f"positive integer; got {n_qubits!r}."
        )
    return int(n_qubits)


def _check_seed(seed, what):
    """Validate a seed. ``None`` is refused rather than defaulted.

    An unseeded reservoir or pool would make a run irreproducible, and because the
    features stay plausible the only symptom is that two runs at one ``seed`` disagree
    -- the failure :func:`qbiocode.learning._tuning.seed_from` exists to prevent on the
    tuner side.
    """
    if seed is None:
        raise ValueError(
            f"{what} must be seeded explicitly so a run reproduces; got None. Pass the "
            f"run's reservoir seed (the config's own, defaulting to q_seed). It is a "
            f"configuration value and must never be a tuned hyperparameter -- see the "
            f"module docstring."
        )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"{what} must be an integer seed; got {seed!r}.")
    return int(seed)


def qelm_reservoir(n_qubits, reps, seed):
    """The fixed random unitary a QELM appends to its data feature map.

    A brick-wall ``ry``/``rz`` + ``cx`` circuit whose angles are drawn once from
    ``seed`` and **bound**, so the returned circuit has no free parameters and the data
    feature map remains the only thing a row is bound into.

    Args:
        n_qubits (int): Width, matching the feature map's.
        reps (int): Reservoir depth. ``0`` means **no reservoir at all** and returns
            ``None``; see the note below, which is load-bearing for the ablation.
        seed (int): Seeds the angle draw. Required -- see :func:`_check_seed`.

    Returns:
        QuantumCircuit or None: An ``n_qubits``-wide circuit with
        ``num_parameters == 0``, or ``None`` when ``reps == 0``.

    Raises:
        ValueError: If ``n_qubits`` is not a positive integer, ``reps`` is negative, or
            ``seed`` is absent or not an integer.

    Note:
        ``reps=0`` returns ``None`` rather than ``efficient_su2(..., reps=0)``, and the
        difference is not cosmetic: qiskit's ``reps=0`` still emits the **final rotation
        layer** (8 parameters on 4 qubits, measured on qiskit 2.5), so it is a random
        single-qubit rotation on every qubit and not the identity. Building it that way
        would mean a "no reservoir" QELM silently measured a *rotated* feature map, and
        the ablation this depth exists to support -- that a zero-depth QELM reads
        exactly PQK's projection, up to the column order, and so isolates the readout --
        would quietly not hold. With ``None`` it holds exactly.
    """
    n_qubits = _check_n_qubits(n_qubits)
    if isinstance(reps, (bool, np.bool_)) or not isinstance(reps, (int, np.integer)) \
            or reps < 0:
        raise ValueError(
            f"reps is the reservoir depth and must be a non-negative integer "
            f"(0 means no reservoir); got {reps!r}."
        )
    reps = int(reps)
    if reps == 0:
        # See the note above: there is no zero-depth spelling of this circuit that is
        # the identity, so the absence of a circuit is represented by the absence of a
        # circuit.
        return None
    seed = _check_seed(seed, "the reservoir seed")

    # The function, not the deprecated EfficientSU2 class: the class warns on qiskit
    # 2.1+ and is removed in 3.0, and these circuits are built once per fit, so there is
    # no reason to carry the warning into every run's log.
    from qiskit.circuit.library import efficient_su2

    template = efficient_su2(
        n_qubits,
        su2_gates=list(RESERVOIR_ROTATIONS),
        entanglement=RESERVOIR_ENTANGLEMENT,
        reps=reps,
    )
    # Angles on [0, 2*pi): the Haar measure on a single qubit is not uniform in these
    # coordinates, so this is not a Haar-random unitary -- deliberately. A Haar
    # reservoir destroys MPS compressibility (module docstring), and what the reservoir
    # has to supply is a fixed, unbiased, reproducible mixing, which this does.
    angles = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, template.num_parameters)
    reservoir = template.assign_parameters(angles)
    if reservoir.num_parameters:  # pragma: no cover - assign_parameters binds all
        raise AssertionError(
            f"the reservoir must be fully bound before it is composed onto a feature "
            f"map, but {reservoir.num_parameters} parameter(s) remain; otherwise a data "
            f"row would be bound into the reservoir's angles."
        )
    return reservoir


def _single_qubit_pool(n_qubits):
    """``<X_k>``, ``<Y_k>``, ``<Z_k>`` on every qubit, in the projected-kernel order.

    The order is ``X`` on every qubit, then ``Y``, then ``Z`` -- which is what makes a
    flat read of this pool reshape to the ``(3, n_qubits)`` layout
    ``compute_pqk``/``compute_qpl`` save, entry for entry.
    """
    return [(pauli, (qubit,)) for pauli in PAULI_CHARS for qubit in range(n_qubits)]


def _chain_zz_pool(n_qubits):
    """``<Z_k Z_{k+1}>`` along the chain: ``n_qubits - 1`` operators, or none at width 1."""
    return [("ZZ", (qubit, qubit + 1)) for qubit in range(n_qubits - 1)]


def _weighted_pauli_pool(n_qubits, max_weight):
    """Every non-identity Pauli of weight 1..``max_weight``, as sorted specs.

    Enumerated rather than rejection-sampled so that ``n_features`` can be checked
    against what exists before anything is measured, and so the draw below is a
    permutation of a fixed list -- which makes it reproducible without depending on how
    many candidates a sampler happened to reject.
    """
    from itertools import combinations, product

    pool = []
    for weight in range(1, max_weight + 1):
        for qubits in combinations(range(n_qubits), weight):
            for paulis in product(PAULI_CHARS, repeat=weight):
                pool.append(("".join(paulis), qubits))
    return pool


def _pool_size(n_qubits, max_weight):
    """How many distinct Paulis of weight <= ``max_weight`` exist on ``n_qubits``."""
    return sum(
        3 ** weight * math.comb(n_qubits, weight)
        for weight in range(1, max_weight + 1)
    )


def qelm_observables(n_qubits, measurement="xyz", n_features=None, seed=None,
                     max_weight=DEFAULT_MAX_WEIGHT):
    """The observables a QELM reads its features off, as ``(pauli, qubits)`` specs.

    Deterministic in its arguments: the same call returns the same list in the same
    order, which is what lets a projection cache be keyed on these arguments rather than
    on the list itself.

    Args:
        n_qubits (int): Width of the circuit being measured.
        measurement (str): One of :data:`MEASUREMENTS`.
        n_features (int or None): How many observables to return. ``None`` means the
            whole pool. For the two structured pools it must equal the pool size if
            given -- see Raises. For ``'random_pauli'`` it is required and is the number
            drawn.
        seed (int or None): Seeds the draw. Required for ``'random_pauli'`` only; the
            structured pools are not random and ignore it.
        max_weight (int): Weight cap for ``'random_pauli'``. Defaults to
            :data:`DEFAULT_MAX_WEIGHT`.

    Returns:
        list[tuple[str, tuple[int, ...]]]: Character ``i`` of each ``pauli`` acts on
        ``qubits[i]`` -- the index-based convention of the module docstring.

    Raises:
        ValueError: If ``measurement`` is unknown; if ``n_features`` is given for a
            structured pool and does not match its size (a structured pool has no
            meaningful subset -- taking a prefix of ``'xyz'`` would drop every ``Z``
            before it drops any ``X``, which is a different measurement, not a smaller
            one, so the mismatch is reported with the size to write instead of being
            silently truncated); if ``'random_pauli'`` is asked for more distinct
            observables than exist at ``max_weight``; or if a required ``seed`` is
            absent.

    Example:
        The default pool is PQK's, in PQK's order, so a flat read reshapes to its layout::

            >>> qelm_observables(2)
            [('X', (0,)), ('X', (1,)), ('Y', (0,)), ('Y', (1,)), ('Z', (0,)), ('Z', (1,))]
    """
    n_qubits = _check_n_qubits(n_qubits)
    if measurement not in MEASUREMENTS:
        raise ValueError(
            f"measurement must be one of {list(MEASUREMENTS)}; got {measurement!r}."
        )
    if n_features is not None:
        if isinstance(n_features, (bool, np.bool_)) or not isinstance(
            n_features, (int, np.integer)
        ) or n_features < 1:
            raise ValueError(
                f"n_features is the number of observables to measure and must be a "
                f"positive integer or None (the whole pool); got {n_features!r}."
            )
        n_features = int(n_features)

    if measurement == "random_pauli":
        if isinstance(max_weight, (bool, np.bool_)) or not isinstance(
            max_weight, (int, np.integer)
        ) or max_weight < 1:
            raise ValueError(
                f"max_weight caps the Pauli weight and must be a positive integer; "
                f"got {max_weight!r}."
            )
        max_weight = min(int(max_weight), n_qubits)
        available = _pool_size(n_qubits, max_weight)
        if n_features is None:
            raise ValueError(
                f"measurement='random_pauli' needs n_features -- it is the number of "
                f"observables to draw, and the pool it draws from holds {available} "
                f"distinct Paulis of weight at most {max_weight} on {n_qubits} qubits. "
                f"Pass n_features, or use measurement='xyz' for the whole {3 * n_qubits}"
                f"-observable single-qubit pool."
            )
        if n_features > available:
            raise ValueError(
                f"measurement='random_pauli' was asked for {n_features} distinct "
                f"observables but only {available} exist at weight <= {max_weight} on "
                f"{n_qubits} qubits. Lower n_features, or raise max_weight (at most "
                f"{n_qubits})."
            )
        seed = _check_seed(seed, "the observable-pool seed")
        pool = _weighted_pauli_pool(n_qubits, max_weight)
        # A permutation of the enumerated pool, then its prefix: distinct by
        # construction, so no two features are the same operator. Drawing with
        # rejection instead would make the result depend on how many duplicates the
        # sampler happened to hit, which is reproducible but not *stable* -- adding one
        # to n_features would reshuffle every earlier pick.
        order = np.random.default_rng(seed).permutation(len(pool))
        return [pool[index] for index in order[:n_features]]

    if measurement == "xyz":
        pool = _single_qubit_pool(n_qubits)
    else:  # 'xyz+zz'
        pool = _single_qubit_pool(n_qubits) + _chain_zz_pool(n_qubits)
    if n_features is not None and n_features != len(pool):
        raise ValueError(
            f"measurement={measurement!r} on {n_qubits} qubits is a fixed pool of "
            f"{len(pool)} observables, but n_features={n_features} was requested. It "
            f"has no meaningful subset -- a prefix of it drops whole Pauli groups rather "
            f"than thinning the pool -- so either set n_features={len(pool)}, leave it "
            f"None, or use measurement='random_pauli', whose width is free."
        )
    return pool


def default_observables(n_qubits):
    """The ``3n`` single-qubit Pauli specs the projected-kernel paths measure.

    The projectors' own default, named here so the one place that defines "X, Y and Z on
    every qubit, in this order" is the same place a QELM pool comes from. Equal to
    ``qelm_observables(n_qubits, 'xyz')``.
    """
    return _single_qubit_pool(_check_n_qubits(n_qubits))


def check_observables(observables, n_qubits):
    """Validate a list of ``(pauli, qubits)`` specs and return it normalised.

    Called by every projector at construction, so a malformed pool is rejected before a
    single circuit is simulated rather than surfacing on some later row as a ``KeyError``
    on an upper-cased gate label or a qiskit shape complaint.

    Args:
        observables (iterable): ``(pauli, qubits)`` pairs.
        n_qubits (int): Width the qubit indices must fall inside.

    Returns:
        list[tuple[str, tuple[int, ...]]]: The specs, with ``qubits`` as tuples of
        ``int`` so they hash, compare and pickle predictably.

    Raises:
        ValueError: If the list is empty, a spec is malformed, a Pauli character is not
            in :data:`PAULI_CHARS`, the string length and the qubit count disagree, a
            qubit index is out of range or repeated within one observable.
    """
    n_qubits = _check_n_qubits(n_qubits)
    normalised = []
    for position, spec in enumerate(observables):
        # A bare string is rejected before the unpack, not by it. ``"ZZ"`` unpacks
        # happily into ``("Z", "Z")`` -- two characters, so the pair check passes -- and
        # then fails several lines down as ``int('Z')``, reporting "invalid literal for
        # int()" and naming neither the observable nor the shape it should have had.
        if isinstance(spec, (str, bytes)):
            raise ValueError(
                f"observable {position} is the bare string {spec!r}; it must be a "
                f"(pauli, qubits) pair such as ('ZZ', (0, 1)), which names the qubits "
                f"the string acts on."
            )
        try:
            pauli, qubits = spec
        except (TypeError, ValueError):
            raise ValueError(
                f"observable {position} is {spec!r}, not a (pauli, qubits) pair such as "
                f"('ZZ', (0, 1))."
            ) from None
        pauli = str(pauli)
        if not pauli:
            raise ValueError(f"observable {position} has an empty Pauli string.")
        bad = sorted(set(pauli) - set(PAULI_CHARS))
        if bad:
            raise ValueError(
                f"observable {position} ({pauli!r}) uses Pauli character(s) {bad}; only "
                f"{list(PAULI_CHARS)} are accepted. A qubit the operator does not act "
                f"on is left out of its qubit tuple rather than written as 'I'."
            )
        try:
            qubits = tuple(int(qubit) for qubit in qubits)
        except TypeError:
            raise ValueError(
                f"observable {position} ({pauli!r}) has qubits {qubits!r}, which is not "
                f"a sequence of qubit indices."
            ) from None
        if len(qubits) != len(pauli):
            raise ValueError(
                f"observable {position} is {pauli!r} on {list(qubits)}: the Pauli string "
                f"has {len(pauli)} character(s) but {len(qubits)} qubit(s) were named. "
                f"Character i acts on qubits[i], so the two must match."
            )
        if len(set(qubits)) != len(qubits):
            raise ValueError(
                f"observable {position} ({pauli!r} on {list(qubits)}) names a qubit "
                f"twice; one operator acts on each qubit at most once."
            )
        outside = sorted(q for q in qubits if not 0 <= q < n_qubits)
        if outside:
            raise ValueError(
                f"observable {position} ({pauli!r}) names qubit(s) {outside} outside "
                f"the circuit's {n_qubits} qubit(s) (0..{n_qubits - 1})."
            )
        normalised.append((pauli, qubits))
    if not normalised:
        raise ValueError(
            "observables is empty, so there is nothing to measure and every row would "
            "project to a zero-width feature vector. Pass None for the default "
            f"{3 * n_qubits}-observable single-qubit pool, or name at least one."
        )
    return normalised


def observables_fingerprint(observables):
    """A short, stable digest of an observable pool, for a projection cache key.

    The pool decides what a cached projection *holds*, so it has to be part of the key
    that names the file -- the failure otherwise is the one
    :func:`qbiocode.learning.compute_qpl.compute_qpl` documents at its own fingerprint:
    change what is measured, rerun into the same directory, and the previous run's
    features are silently reloaded and reported as the new result. A digest rather than
    the list itself, because a pool of a few hundred operators has no business in a
    filename.

    Computed from the normalised specs in order, since order fixes the column order of
    the projection and two pools that differ only by a permutation are different caches.
    """
    import hashlib

    payload = repr([(pauli, tuple(qubits)) for pauli, qubits in observables]).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


__all__ = [
    "DEFAULT_MAX_WEIGHT",
    "MEASUREMENTS",
    "PAULI_CHARS",
    "RESERVOIR_ENTANGLEMENT",
    "RESERVOIR_ROTATIONS",
    "check_observables",
    "default_observables",
    "observables_fingerprint",
    "qelm_observables",
    "qelm_reservoir",
]
