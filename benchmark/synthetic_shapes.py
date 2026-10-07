"""Shape families for create_synthetic_datasets.py: labelled manifolds and partitions.

Every family is a top-level function ``(n, d, k, rng, srng) -> (X, F, latent)``:

- ``X``: ``n`` points in [0, 1]^d (before the driver's noise).
- ``F``: one value per point; the label is ``1[F > 0]``.
- ``latent``: the coordinates the label is a function of (angles, a radius, a latent
  vector), so a test can recompute the label without trusting ``X``.

``rng`` draws the points and differs per seed. ``srng`` draws the structure of the
embedding (rotations, harmonics) and is fixed per (family, d, k). Every seed of one
configuration is therefore a sample of the SAME manifold. The reference implementation
(kernel_exps/ultra_hard_datasets.py) re-drew those maps per seed.

Labels depend on the latent coordinates only, never on a column that carries a label
factor. The reference sphere and swiss roll put ``cos(k theta)`` in feature 3, which made
their labels an XOR of two observed columns; those columns are not reproduced here.

``k`` is each family's native complexity knob, and ``cells(k, d)`` turns it into the
number of label cells. That count, and training points per cell, is what compares across
families (the driver records both). Below about one training point per cell every method
sits at chance; the reference checkerboards with 256 or more cells did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

TWO_PI = 2.0 * np.pi


def _orthonormal(srng: np.random.Generator, d: int, m: int) -> np.ndarray:
    """A fixed d x m matrix with orthonormal columns: an m-plane inside [0,1]^d."""
    q, _ = np.linalg.qr(srng.standard_normal((d, m)))
    return q[:, :m]


def _into_cube(latent: np.ndarray, srng: np.random.Generator, d: int) -> np.ndarray:
    """Place points with |latent| <= 1 per row inside [0,1]^d by a fixed rotation.

    Each coordinate of ``latent @ Q.T`` lies in [-1, 1] because Q has orthonormal
    columns, so ``0.5 + 0.45 * ...`` stays inside the cube.
    """
    q = _orthonormal(srng, d, latent.shape[1])
    return 0.5 + 0.45 * latent @ q.T


def _harmonics(angles: np.ndarray, srng: np.random.Generator, n_extra: int) -> np.ndarray:
    """``n_extra`` extra columns ``(cos(a . angles + ph) + 1) / 2``, a in {1,2,3}^m, fixed."""
    m = angles.shape[1]
    out = np.empty((angles.shape[0], n_extra))
    for j in range(n_extra):
        a = srng.integers(1, 4, size=m)
        ph = srng.uniform(0.0, TWO_PI)
        out[:, j] = (np.cos(angles @ a + ph) + 1.0) / 2.0
    return out


# ---- families -----------------------------------------------------------------------

def torus(n, d, k, rng, srng):
    """Flat torus: angles (t, f), label sign(cos k t * cos k f), 4 k^2 cells.

    Columns 0-3 are (cos t+1)/2, (sin t+1)/2, (cos f+1)/2, (sin f+1)/2, as in the
    reference lids_torus; columns 4.. are fixed harmonics of (t, f). k = 2 at d = 4 is the
    configuration where an unentangled fidelity kernel beat tuned classical arms in the
    kernel_exps v3 study.
    """
    t, f = rng.uniform(0.0, TWO_PI, n), rng.uniform(0.0, TWO_PI, n)
    cols = [(np.cos(t) + 1) / 2, (np.sin(t) + 1) / 2, (np.cos(f) + 1) / 2, (np.sin(f) + 1) / 2]
    X = np.column_stack(cols)
    if d > 4:
        X = np.column_stack([X, _harmonics(np.column_stack([t, f]), srng, d - 4)])
    return X, np.cos(k * t) * np.cos(k * f), np.column_stack([t, f])


def sphere(n, d, k, rng, srng):
    """The 2-sphere, area-uniform: label sign(cos(k theta) * cos(phi)), 2 (k + 1) cells.

    theta = arccos(u), u ~ U(-1, 1), so points are uniform in area. The reference drew
    theta ~ U(0, pi), which crowds the poles. Columns 0-2 are (xyz + 1) / 2; columns 3.. are
    fixed harmonics of (theta, phi). No column holds cos(k theta).
    """
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, TWO_PI, n)
    xyz = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    X = (xyz + 1.0) / 2.0
    if d > 3:
        X = np.column_stack([X, _harmonics(np.column_stack([theta, phi]), srng, d - 3)])
    return X, np.cos(k * theta) * np.cos(phi), np.column_stack([theta, phi])


def concentric_circles(n, d, k, rng, srng):
    """Rings in a plane: radius r uniform in area in the unit disk, label sign(sin((k+1) pi r)).

    k + 1 rings (k interior boundaries), the 2-D plane rotated into d dimensions by a fixed
    orthonormal map. k = 1 is sklearn's make_circles.
    """
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    a = rng.uniform(0.0, TWO_PI, n)
    plane = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return _into_cube(plane, srng, d), np.sin((k + 1) * np.pi * r), np.column_stack([r, a])


def concentric_spheres(n, d, k, rng, srng):
    """Shells in the d-ball: direction uniform, radius r ~ U(0, 1), label sign(sin((k+1) pi r)).

    k + 1 shells. The radius is uniform rather than volume-uniform, because in d = 8 a
    volume-uniform radius puts almost every point in the outer shell.
    """
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rng.uniform(0.0, 1.0, n)
    ball = u * r[:, None]
    return 0.5 + 0.45 * ball, np.sin((k + 1) * np.pi * r), r[:, None]


def checkerboard(n, d, k, rng, srng):
    """k x k checkerboard on the first two axes, label sign(sin(k pi x0) sin(k pi x1)).

    k^2 cells; axes 2.. are irrelevant U(0, 1) features. Two signal axes, not d: the
    reference used every axis, which made k = 1 bit-identical to parity and put (2k)^d
    cells in the cube.
    """
    X = rng.uniform(0.0, 1.0, (n, d))
    return X, np.sin(k * np.pi * X[:, 0]) * np.sin(k * np.pi * X[:, 1]), X[:, :2]


def random_manifold(n, d, k, rng, srng):
    """A random 2-D manifold in [0,1]^d: latent z ~ U[0,1]^2, label sign(prod cos(k pi z_q)).

    (k + 1)^2 cells. Ambient column j is a fixed random sum of sinusoids of z (one map per
    (d, k), not per seed), scaled by its own bound into [0, 1].
    """
    z = rng.uniform(0.0, 1.0, (n, 2))
    cs, cc = srng.standard_normal((d, 2)), srng.standard_normal((d, 2))
    ph = srng.uniform(0.0, TWO_PI, (d, 2))
    freq = np.arange(1, 3)[None, :]                  # (q + 1) pi z_q, q = 0, 1
    X = np.empty((n, d))
    for j in range(d):
        col = (cs[j] * np.sin(freq * np.pi * z + ph[j]) + cc[j] * np.cos(freq * np.pi * z - ph[j])).sum(1)
        bound = float(np.abs(cs[j]).sum() + np.abs(cc[j]).sum())
        X[:, j] = 0.5 + 0.5 * col / bound
    return X, np.prod(np.cos(k * np.pi * z), axis=1), z


def swiss_roll(n, d, k, rng, srng):
    """Swiss roll (t cos t, h, t sin t), t ~ U(1.5 pi, 4.5 pi), h ~ U(-1, 1).

    Label sign(sin(k pi s) * h), s = (t - 1.5 pi) / (3 pi): k bands along the roll times
    two sides, 2k cells. The roll is rotated into d dimensions by a fixed orthonormal map;
    no column holds the band function (the reference's X3 did).
    """
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(-1.0, 1.0, n)
    rmax = 4.5 * np.pi
    roll = np.column_stack([t * np.cos(t) / rmax, h, t * np.sin(t) / rmax]) / np.sqrt(3.0)
    s = (t - 1.5 * np.pi) / (3.0 * np.pi)
    return _into_cube(roll, srng, d), np.sin(k * np.pi * s) * h, np.column_stack([t, h])


def half_moons(n, d, k, rng, srng):
    """k interleaved half-moons (k = 2 is sklearn's make_moons), label = moon index mod 2.

    Moon i: (i + (-1)^i cos a, 0.5 (i mod 2) + (-1)^i sin a), a ~ U(0, pi), each moon
    equally likely. k cells. The plane is rotated into d dimensions by a fixed map.
    """
    i = rng.integers(0, k, n)
    a = rng.uniform(0.0, np.pi, n)
    sgn = np.where(i % 2 == 0, 1.0, -1.0)
    x = i + sgn * np.cos(a)
    y = 0.5 * (i % 2) + sgn * np.sin(a)
    # Centre and scale the moon chain into the unit disk before rotating it in: x spans
    # [-1, k] and y [-0.5, 1], so after this |x| <= 1/sqrt(2) and |y| <= 0.75/sqrt(2)/1.5.
    half = (k + 1) / 2.0 * np.sqrt(2.0)
    x = (x - (k - 1) / 2.0) / half
    y = (y - 0.25) / half
    F = np.where(i % 2 == 0, 1.0, -1.0)
    return _into_cube(np.column_stack([x, y]), srng, d), F, np.column_stack([i, a])


def parity(n, d, k, rng, srng):
    """k-bit parity: X ~ U[0,1]^d, label sign(prod_{i<k} (x_i - 0.5)), 2^k cells; needs k <= d."""
    X = rng.uniform(0.0, 1.0, (n, d))
    return X, np.prod(X[:, :k] - 0.5, axis=1), X[:, :k]


def perm_parity(n, d, k, rng, srng):
    """Sign of the sorting permutation of x_0..x_{k-1}, k! cells; needs 2 <= k <= d.

    The one boundary of the kernel_exps study that went to the classical side in every
    quantum family.
    """
    X = rng.uniform(0.0, 1.0, (n, d))
    F = np.ones(n)
    for a in range(k):
        for b in range(a + 1, k):
            F *= np.sign(X[:, b] - X[:, a])
    return X, F, X[:, :k]


def simple_linear(n, d, k, rng, srng):
    """A random half-space through the centre of the cube, label sign(w . (x - 0.5)); k unused.

    A negative control: every reasonable classifier should solve it.
    """
    X = rng.uniform(0.0, 1.0, (n, d))
    w = srng.standard_normal(d)
    w /= np.linalg.norm(w)
    return X, (X - 0.5) @ w, (X - 0.5) @ w[:, None]


# ---- the catalogue --------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """One shape family: its generator and what its knobs mean."""

    generate: Callable
    d_min: int
    uses_k: bool
    cells: Callable[[int, int], int]
    label_rule: str
    role: str = "shape"
    k_min: int = 1
    k_max_d: bool = False       # k may not exceed d (parity-like families)
    oversample: int = 8


SHAPES = {
    "torus": Shape(torus, 4, True, lambda k, d: 4 * k * k,
                   "sign(cos(k t) cos(k f)), angles t, f of a flat torus"),
    "sphere": Shape(sphere, 3, True, lambda k, d: 2 * (k + 1),
                    "sign(cos(k theta) cos(phi)), area-uniform 2-sphere"),
    "concentric_circles": Shape(concentric_circles, 2, True, lambda k, d: k + 1,
                                "sign(sin((k+1) pi r)), r the radius in a rotated plane"),
    "concentric_spheres": Shape(concentric_spheres, 2, True, lambda k, d: k + 1,
                                "sign(sin((k+1) pi r)), r the radius in the d-ball"),
    "checkerboard": Shape(checkerboard, 2, True, lambda k, d: k * k,
                          "sign(sin(k pi x0) sin(k pi x1)); axes 2.. irrelevant", k_min=2),
    "random_manifold": Shape(random_manifold, 2, True, lambda k, d: (k + 1) ** 2,
                             "sign(prod_q cos(k pi z_q)), z the 2-D latent of a fixed random map"),
    "swiss_roll": Shape(swiss_roll, 3, True, lambda k, d: 2 * k,
                        "sign(sin(k pi s) h), s the arc position and h the height of the roll"),
    "half_moons": Shape(half_moons, 2, True, lambda k, d: k,
                        "moon index mod 2 of k interleaved half-moons", k_min=2, oversample=4),
    "parity": Shape(parity, 2, True, lambda k, d: 2 ** k,
                    "sign(prod_{i<k} (x_i - 0.5))", k_max_d=True),
    "perm_parity": Shape(perm_parity, 2, True, lambda k, d: int(np.prod(np.arange(1, k + 1))),
                         "sign of the sorting permutation of x_0..x_{k-1}",
                         role="classical_favoured", k_min=2, k_max_d=True),
    "simple_linear": Shape(simple_linear, 2, False, lambda k, d: 2,
                           "sign(w . (x - 0.5)), a fixed random half-space", role="negative_control",
                           oversample=4),
}
