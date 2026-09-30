"""
Command-line interface to the quantum data generators.

Installed as ``qdata-gen``, and runnable as
``python -m qbiocode.data_generation.quantum_cli``. One subcommand per family plus
``selftest``::

    qdata-gen gs  --label sparse --n 8 --N 400 --kappa 0.5 --s 4 --seed 0 --out qdata
    qdata-gen te  --n 10 --N 400 --s 4 --taus 0.25 0.5 1 2 4 --seed 0 --out qdata
    qdata-gen hl  --n 6 --N 400 --times 0.5 --shots 1000     --seed 0 --out qdata
    qdata-gen ql  --encoding zz --n 8 --N 400 --tau 1 --reps 2 --entanglement linear \\
                  --seed 0 --out qdata
    qdata-gen eng --n 6 --N 300 --gamma_q 1.0                --seed 0 --out qdata
    qdata-gen selftest

The flags and their defaults are those of the original standalone ``qdata_gen.py``
script, so commands written against it keep working unchanged and produce
byte-identical output.

.. note::
   The **Python API's** sweep defaults are not the same as these CLI defaults. The
   module-level constants in each ``make_*`` module (e.g.
   :data:`~qbiocode.data_generation.make_time_evolution.N_QUBITS`) hold the
   documented runsheet configuration -- the datasets QBioCode actually ships -- so
   ``generate_time_evolution_datasets()`` with no arguments reproduces the runsheet,
   while ``qdata-gen te`` with no flags uses this script's generic ``--n 8``. Where
   they differ, the runsheet is what the benchmark was measured on.

Unlike the Python API, the CLI writes exactly one configuration per invocation: it
takes a scalar for each knob, not a list. Sweeps belong in Python, where the
``generate_*`` functions enumerate them and guard against two configurations
colliding on one filename.
"""

import argparse

from .make_engineered_kernel import generate_engineered_kernel_datasets
from .make_ground_state import generate_ground_state_datasets
from .make_hamiltonian_learning import generate_hamiltonian_learning_datasets
from .make_quantum_labels import generate_quantum_label_datasets
from .make_time_evolution import generate_time_evolution_datasets
from .quantum_selftest import run_selftest

#: Families the CLI accepts, plus the self-check.
FAMILIES = ("gs", "te", "hl", "ql", "eng", "selftest")


def build_parser():
    """Construct the argument parser.

    Returns
    -------
    argparse.ArgumentParser

    Notes
    -----
    A single flat parser with a ``family`` positional, rather than ``add_subparsers``,
    because that is what the original script used: subparsers would reject
    ``qdata-gen gs --tau 1`` where the flat parser silently ignores a flag the family
    does not read, and scripts written against the original rely on that.
    """
    parser = argparse.ArgumentParser(
        prog="qdata-gen",
        description="Generate simulated quantum binary-classification datasets.",
        epilog="Flags a family does not use are ignored. See the module docstring "
               "for the documented runsheet.",
    )
    parser.add_argument("family", choices=FAMILIES, help="dataset family, or 'selftest'")
    parser.add_argument("--out", default="qdata_out", help="output directory (default: qdata_out)")
    parser.add_argument("--name", default=None, help="override the generated dataset name")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--n", type=int, default=8, help="qubits, <=12 recommended (default: 8)")
    parser.add_argument("--N", type=int, default=400, help="rows before margin filtering (default: 400)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="drop rows with |F - threshold| < margin (default: 0, keep all)")
    parser.add_argument("--shots", type=int, default=0,
                        help="shots per Pauli for phi/hl features; 0 = exact (default: 0)")
    parser.add_argument("--s", type=int, default=4,
                        help="terms in the sparse observable, gs/te (default: 4)")
    parser.add_argument("--label", choices=["sparse", "e2e"], default="sparse",
                        help="gs: sparse local observable, or the end-to-end correlator (default: sparse)")
    parser.add_argument("--kappa", type=float, default=0.5,
                        help="gs: next-nearest-neighbour coupling (default: 0.5)")
    parser.add_argument("--J_lo", "--J-lo", type=float, default=0.5,
                        help="gs: lowest ZZ coupling (default: 0.5)")
    parser.add_argument("--J_hi", "--J-hi", type=float, default=1.5,
                        help="gs: highest ZZ coupling (default: 1.5)")
    parser.add_argument("--h_lo", "--h-lo", type=float, default=0.2,
                        help="gs: lowest field (default: 0.2)")
    parser.add_argument("--h_hi", "--h-hi", type=float, default=2.0,
                        help="gs: highest field (default: 2.0)")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0],
                        help="te: evolution times, one dataset each (default: 0.25 0.5 1 2 4)")
    parser.add_argument("--times", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="hl: measurement times, all in one dataset (default: 0.25 0.5 1 2)")
    parser.add_argument("--g", type=float, default=0.5,
                        help="hl: longitudinal field, breaks integrability (default: 0.5)")
    parser.add_argument("--te_w", "--te-w", type=float, default=0.1,
                        help="te: half-width of the coupling disorder (default: 0.1)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="ql: Heisenberg evolution time after encoding (default: 1.0)")
    parser.add_argument("--encoding", choices=["zz", "evo"], default="zz",
                        help="ql: feature map (default: zz)")
    parser.add_argument("--reps", type=int, default=2,
                        help="ql/eng: feature-map repetitions (default: 2)")
    parser.add_argument("--entanglement", choices=["linear", "pairwise", "full"], default="linear",
                        help="ql/eng: feature-map entanglement pattern (default: linear). "
                             "QProfiler defaults: qsvc ZZ/linear/reps 2, pqk ZZ/pairwise/reps 4")
    parser.add_argument("--data-map", "--data_map", choices=["qiskit", "unit"], default="qiskit",
                        dest="data_map",
                        help="ql (zz) / eng: feature-map data map, i.e. WHICH QProfiler model "
                             "the labels are aligned with (default: qiskit). 'qiskit' is the "
                             "stock ZZFeatureMap map and matches qsvc; 'unit' halves at every "
                             "step and matches pqk. There is no value that matches both, and "
                             "the wrong one inverts eng's advantage (0.797 -> 0.403). "
                             "Non-default adds a '_dmunit' suffix to the dataset name")
    parser.add_argument("--gamma_q", "--gamma-q", type=float, default=1.0,
                        help="eng: RBF bandwidth of the quantum kernel (default: 1.0)")
    parser.add_argument("--lam", type=float, default=1e-3,
                        help="eng: ridge on K_C before inversion (default: 1e-3)")
    parser.add_argument("--blas-threads", "--blas_threads", type=int, default=1,
                        dest="blas_threads",
                        help="BLAS threads for the small dense linear algebra; 0 leaves "
                             "threading untouched (default: 1, measured ~14x faster than "
                             "unpinned on a many-core host)")
    return parser


def main(argv=None):
    """Run the CLI.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` if ``selftest`` failed. Suitable for
        ``sys.exit(main())``.

    Examples
    --------
    >>> from qbiocode.data_generation.quantum_cli import main
    >>> main(["selftest"])                                          # doctest: +SKIP
    0
    """
    args = build_parser().parse_args(argv)
    blas_threads = None if args.blas_threads == 0 else args.blas_threads

    if args.family == "selftest":
        ok, _ = run_selftest()
        return 0 if ok else 1

    common = dict(n_qubits=args.n, n_samples=args.N, margin=args.margin,
                  save_path=args.out, name=args.name, random_state=args.seed,
                  blas_threads=blas_threads)
    if args.family == "gs":
        generate_ground_state_datasets(
            label=args.label, kappa=args.kappa, n_terms=args.s, shots=args.shots,
            J_range=(args.J_lo, args.J_hi), h_range=(args.h_lo, args.h_hi), **common)
    elif args.family == "te":
        generate_time_evolution_datasets(
            taus=args.taus, n_terms=args.s, disorder=args.te_w, shots=args.shots, **common)
    elif args.family == "hl":
        generate_hamiltonian_learning_datasets(
            times=args.times, shots=args.shots, longitudinal_field=args.g, **common)
    elif args.family == "ql":
        generate_quantum_label_datasets(
            encoding=args.encoding, tau=args.tau, reps=args.reps,
            entanglement=args.entanglement, data_map=args.data_map, **common)
    elif args.family == "eng":
        generate_engineered_kernel_datasets(
            gamma_q=args.gamma_q, reps=args.reps, entanglement=args.entanglement,
            data_map=args.data_map,
            lam=args.lam, **common)
    return 0


if __name__ == "__main__":       # pragma: no cover
    import sys

    sys.exit(main())
