"""``pairprod``: ``O(x) = Σ_{n1,n2} p(n1) p(n2) K(n1, n2)`` -- the full quadratic form ``p^T K p``.

The non-diagonal sibling of :mod:`.sq`.  The default kernel
``K(n1,n2) = (-1)^{Σ_i n1_i·n2_i}`` (parity of the two outcomes' occupation OVERLAP) is
deliberately NON-SEPARABLE: a separable ``K(n1,n2) = f(n1)·f(n2)`` would collapse to ``<f>^2``,
a product of two independently additive-estimable expectations -- straight back into the
classically easy single-copy regime.  The overlap does not factor over the sum, so ``p^T K p``
is a genuine two-point (2-copy) function.  The score is signed, so ``sign(O)`` yields two
classes with no centering or balancing.  Deterministic in ``(observable, m)`` -> reproducible /
hashable.

Below :func:`pairprod_kernel` sit alternative builders, each a drop-in for it (call
``f(keys, m)``); the seeded / parameterised ones take extra kwargs with defaults.  Governing
properties: every kernel here is non-separable EXCEPT ``base_overlay`` with a trivial core and
the pure ``rbf`` (a PSD Gram matrix -> ``p^T K p >= 0`` -> one class).  Only the SYMMETRIC part
of a kernel survives ``p^T K p`` (``p`` is real), so every builder returns symmetric K.
"""

from __future__ import annotations

import math

import numpy as np

from .base import (Observable, ObservableContext, ObservableFamily, QuadraticObservable,
                   register)

#: Cap on the Fock dimension for ``pairprod``'s dense ``(n_fock, n_fock)`` kernel.  8192 -> a
#: 256 MB fp32 matrix; above this, use a ``sq_<base>`` observable (diagonal, no dense kernel).
PAIRPROD_MAX_FOCK = 8192


def is_pairprod_observable(observable: str) -> bool:
    """True for the ``pairprod`` observable (score ``Σ_{n1,n2} p(n1) p(n2) (-1)^<n1,n2>``)."""
    return observable == "pairprod"


def check_fock_dim(n_fock: int) -> None:
    """Reject a Fock dimension whose dense pair kernel would not fit (see :data:`PAIRPROD_MAX_FOCK`)."""
    if n_fock > PAIRPROD_MAX_FOCK:
        raise ValueError(
            f"pairprod builds a dense {n_fock}x{n_fock} kernel (n_fock={n_fock} > "
            f"{PAIRPROD_MAX_FOCK}); lower (m, k) or use a sq_<base> observable")


def occ_matrix(keys, m: int) -> np.ndarray:
    """``(n_fock, m)`` int64 per-mode photon-count matrix for the Fock basis ``keys``."""
    return np.array([[int(key[i]) for i in range(m)] for key in keys], dtype=np.int64)


def finish_kernel(K: np.ndarray, zero_diagonal: bool) -> np.ndarray:
    """Coerce to float64 and, if requested, zero the diagonal (the distinct-sample form).

    Zeroing drops ``K[a,a]`` -- for the sign/parity kernels a fixed constant on the collision-free
    outcomes (a purity term that biases the whole score one way, all one class for even ``k``); the
    off-diagonal part is the genuine two-point (2-copy) functional and is mixed-sign, so
    ``sign(p^T K p)`` yields two classes with no balancing.
    """
    K = np.asarray(K, dtype=np.float64)
    if zero_diagonal:
        np.fill_diagonal(K, 0.0)
    return K


def pairprod_kernel(keys, m: int) -> np.ndarray:
    """The default kernel: ``K[a,b] = (-1)^{Σ_i (n^a_i + 1)(n^b_i + 1)}`` off-diagonal, ``0`` on it.

    The exponent is the overlap (dot product) of the two outcomes' per-mode occupation vectors,
    so ``K`` does not factor as ``f(a)·f(b)`` -- the quadratic form
    ``p^T K p = Σ_{a!=b} p(a) p(b) K[a,b]`` is a genuine two-point (distinct-sample, 2-copy)
    functional, not a product of two separately additive-estimable expectations.  The diagonal
    is zeroed on purpose (see :func:`finish_kernel`).  Returns a ``(n_fock, n_fock)`` float64
    array; deterministic in ``(keys, m)`` -> reproducible / hashable.

    NOTE: the counts are shifted by ``+1`` before the overlap (``n_i + 1``, not ``n_i``), unlike
    every other builder here, which use the raw counts via :func:`occ_matrix`.  The shift
    re-centres the sign split (an empty mode still contributes) and is what every existing
    ``pairprod`` dataset was generated with, so it is preserved verbatim.
    """
    occ = np.array([[int(key[i] + 1) for i in range(m)] for key in keys], dtype=np.int64)
    overlap = occ @ occ.T                               # (n_fock, n_fock) integer overlaps
    return finish_kernel(np.where(overlap % 2 == 0, 1.0, -1.0), zero_diagonal=True)


# --- alternative pair kernels: drop-in replacements for pairprod_kernel ------------------- #


def pairprod_random_phase_kernel(keys, m: int, *, seed: int = 0,
                                 zero_diagonal: bool = True) -> np.ndarray:
    """Random bilinear phase ``K[a,b] = cos(n^a^T W n^b)``, ``W`` a seeded symmetric matrix.

    Non-separable, indefinite, and naturally centred: the dense random coupling spreads
    ``n^a^T W n^b`` so the cosine sign is balanced (no sparse-overlap ``+`` bias).  ``W_ij ~ U[0, 2pi]``,
    symmetrised.  Deterministic in ``(keys, m, seed)``.
    """
    occ = occ_matrix(keys, m)
    A = np.random.default_rng(int(seed)).uniform(0.0, 2 * math.pi, size=(m, m))
    W = 0.5 * (A + A.T)
    return finish_kernel(np.cos(occ @ W @ occ.T), zero_diagonal)


def pairprod_random_parity_kernel(keys, m: int, *, seed: int = 0,
                                  zero_diagonal: bool = True) -> np.ndarray:
    """Random bilinear parity ``K[a,b] = (-1)^{n^a^T W n^b mod 2}``, ``W`` a seeded symmetric 0/1 matrix.

    The discrete (±1) sibling of :func:`pairprod_random_phase_kernel`.  Non-separable, indefinite,
    roughly centred.  Deterministic in ``(keys, m, seed)``.
    """
    occ = occ_matrix(keys, m)
    A = np.random.default_rng(int(seed)).integers(0, 2, size=(m, m))
    W = np.triu(A) + np.triu(A, 1).T                    # symmetric 0/1
    return finish_kernel(np.where(occ @ W @ occ.T % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_weighted_overlap_kernel(keys, m: int, *, seed: int = 0,
                                     zero_diagonal: bool = True) -> np.ndarray:
    """Weighted overlap parity ``K[a,b] = (-1)^{Σ_i w_i n^a_i n^b_i}``, ``w`` seeded integers in [1, 3].

    Non-separable, indefinite.  Non-uniform ``w`` breaks the ``|A∪B| = 2k - |A∩B|`` identity, so this
    is genuinely richer than the unit-weight overlap :func:`pairprod_kernel`.  Deterministic in
    ``(keys, m, seed)``.
    """
    occ = occ_matrix(keys, m)
    w = np.random.default_rng(int(seed)).integers(1, 4, size=m)
    return finish_kernel(np.where((occ * w) @ occ.T % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_weighted_or_parity_kernel(keys, m: int, *, seed: int = 0,
                                       zero_diagonal: bool = True) -> np.ndarray:
    """Weighted OR parity ``K[a,b] = (-1)^{Σ_i w_i (n^a_i ∨ n^b_i)}``, ``w`` seeded integers in [1, 3].

    Union (OR) rather than product (AND): ``a_i ∨ b_i = a_i + b_i - a_i b_i``, so this factors as
    ``s(a) s(b) (-1)^{Σ w_i a_i b_i}`` -- a separable ±1 sign overlay ``s(a) = (-1)^{Σ w_i a_i}`` on a
    non-separable weighted-overlap core.  The overlay re-centres (pushes ΣK toward 0) while the core
    keeps it two-copy; equals :func:`pairprod_kernel` at ``w ≡ 1`` (fixed-``k`` collision-free).
    Deterministic in ``(keys, m, seed)``.
    """
    occ = occ_matrix(keys, m)
    w = np.random.default_rng(int(seed)).integers(1, 4, size=m)
    supp = (occ > 0).astype(np.int64)
    ga = supp @ w                                       # Σ_i w_i supp_a_i
    H = (supp * w) @ supp.T                             # Σ_i w_i supp_a_i supp_b_i
    U = ga[:, None] + ga[None, :] - H                   # Σ_i w_i (a_i ∨ b_i)
    return finish_kernel(np.where(U % 2 == 0, 1.0, -1.0), zero_diagonal)


def pairprod_distance_sign_kernel(keys, m: int, *, d0: float | None = None,
                                  zero_diagonal: bool = True) -> np.ndarray:
    """Distance sign ``K[a,b] = +1 if ||n^a - n^b||_1 > d0 else -1``.

    Non-separable, indefinite.  ``d0`` defaults to the median off-diagonal L1 distance, which centres
    the split.  Deterministic in ``(keys, m, d0)``.
    """
    occ = occ_matrix(keys, m)
    n = occ.shape[0]
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(m):                                  # avoid the (n, n, m) broadcast intermediate
        col = occ[:, i]
        D += np.abs(col[:, None] - col[None, :])
    if d0 is None:
        off = D[~np.eye(n, dtype=bool)]
        d0 = float(np.median(off)) if off.size else 0.0
    return finish_kernel(np.where(D > float(d0), 1.0, -1.0), zero_diagonal)


def pairprod_shared_modes_kernel(keys, m: int, *, t: int = 1,
                                 zero_diagonal: bool = True) -> np.ndarray:
    """Relational ``K[a,b] = +1 if |supp(a) ∩ supp(b)| == t else -1`` (shared-mode count equals ``t``).

    Non-separable, indefinite; the split is tuned by ``t``.  Deterministic in ``(keys, m, t)``.
    """
    supp = (occ_matrix(keys, m) > 0).astype(np.int64)
    return finish_kernel(np.where(supp @ supp.T == int(t), 1.0, -1.0), zero_diagonal)


def pairprod_disjoint_kernel(keys, m: int, *, zero_diagonal: bool = True) -> np.ndarray:
    """HOM-like disjointness ``K[a,b] = +1 if a, b share no mode else -1`` (``|supp(a) ∩ supp(b)| == 0``).

    Non-separable, indefinite; biased (two random small supports are usually disjoint -> ``+``).
    The ``t = 0`` case of :func:`pairprod_shared_modes_kernel`.  Deterministic in ``(keys, m)``.
    """
    return pairprod_shared_modes_kernel(keys, m, t=0, zero_diagonal=zero_diagonal)


def pairprod_rbf_kernel(keys, m: int, *, gamma: float = 1.0,
                        zero_diagonal: bool = False) -> np.ndarray:
    """Gaussian/RBF Gram ``K[a,b] = exp(-gamma ||n^a - n^b||^2)``.

    NON-separable but **PSD** -> ``p^T K p >= 0`` -> ONE class (never centred); the definite-kernel
    reference.  ``zero_diagonal=False`` keeps it a true Gram matrix (set ``True`` to force it
    indefinite).  Deterministic in ``(keys, m, gamma)``.
    """
    occ = occ_matrix(keys, m).astype(np.float64)
    sq = (occ ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (occ @ occ.T)
    return finish_kernel(np.exp(-float(gamma) * d2), zero_diagonal)


def pairprod_base_overlay_kernel(base_vec, core) -> np.ndarray:
    """Base overlay ``K[a,b] = base(a) base(b) core(a,b)`` (``base_vec`` per-state, ``core`` a kernel).

    Separable **iff** ``core`` is trivial (``core ≡ 1`` -> ``K = v v^T`` -> ``<base>^2``, PSD, one class);
    with a non-separable ``core`` (any builder above) the product stays non-separable.  ``base_vec``
    can be built from the plain scorers (e.g. :func:`~.sq.sq_base_vec`).  Diagonal is inherited
    from ``core``.
    """
    v = np.asarray(base_vec, dtype=np.float64)
    return v[:, None] * v[None, :] * np.asarray(core, dtype=np.float64)


class PairprodFamily(ObservableFamily):
    describe = "pairprod"

    def matches(self, name: str) -> bool:
        return is_pairprod_observable(name)

    def build(self, name: str, ctx: ObservableContext) -> Observable:
        check_fock_dim(ctx.n_fock)
        return QuadraticObservable(pairprod_kernel(ctx.keys, ctx.m))


register(PairprodFamily())
