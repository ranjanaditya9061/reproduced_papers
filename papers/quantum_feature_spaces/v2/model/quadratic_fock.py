"""Poly-size classical control: random **quadratic** amplitudes over the Fock basis.

    a(x, n) = f(x)^T W phi(n)        for two independent random W  (W_re, W_im)
    p_x(n)  = (a_re^2 + a_im^2) / sum_n' (.)

``phi(n)`` is a fixed feature map of the *outcome* -- the occupation moments
``[n_i] + [n_i n_j]_{i<j}``, i.e. ``m(m+1)/2`` numbers.  ``f(x)`` is the Fourier expansion of the
input angles.  The model is then just two random matrices; no network, no training, no temperature.

**Why "quadratic".**  Expanding, ``p_x(n) ~ phi(n)^T M(x) phi(n)`` with
``M(x) = W_re^T f f^T W_re + W_im^T f f^T W_im`` -- a positive-semidefinite quadratic form of rank
<= 2, in ``phi(n)`` and equally in ``f(x)``.  Degree 2, which is the contrast with
:mod:`v2.model.mlp_fock`'s deep tanh stack.  (The legacy name was ``ebm_fock``, which is a
misnomer: it implies a Gibbs link ``p ~ exp(theta.phi)``, and that link was *rejected* -- see the
tail discussion below.  Renamed here.)

**Why this shape.**  The photonic map is ``2m^2 - 1`` numbers generating a distribution over
``C(m+k-1, k)`` outcomes: *poly parameters, exponential support*.  That asymmetry is what a quantum
circuit buys, so a control meant to answer "is a small classical model enough" must have it too.
This one does.  Its sibling ``mlp_fock`` emits the ``n_fock`` probabilities from a dense output
layer, so its *description* grows exponentially -- which hands the classical side the very resource
the quantum claim is about, and is why that one can only upper-bound.

Normalising here sums over every outcome, so exact inference costs exponential *work* -- but that is
our simulation, not the model, exactly as merlin computing the full Fock distribution is a
simulation artifact rather than what a boson sampler does.  ``phi`` is likewise scaffolding; the
model is ``W_re, W_im``.

**Parameter counting, and the two corrections v2 makes.**  The count is
``2 * d_x * d_phi`` with ``d_x = 2 * order * n_features`` and ``d_phi = m(m+1)/2``:

1. The legacy docstring reported ``O(m^3)`` (840 at ``m=6`` against the circuit's 72).  That was an
   artifact of ``n_features = m - 1``, which made ``d_x = O(m)`` multiply ``d_phi = O(m^2)``.  With
   ``n_features`` a fixed study invariant (:data:`v2.config.N_FEATURES`), ``d_x`` is a **constant**
   and the count is ``O(m^2)`` -- the circuit's scaling -- with no change to the model at all.
2. Same scaling is not the same count: full-rank at ``n_features=6, order=2`` is
   ``2 * 24 * 21 = 1008`` against ``71``.  So ``W = A B^T`` is optionally **low-rank**, giving
   ``2r(d_x + d_phi)`` parameters with ``r`` the single dial.  ``r`` is ``O(1)`` while
   ``d_phi = O(m^2)`` carries the scaling, so the match holds at every ``m``:

   ====  =======  ================  ==================  ==================
   m     d_phi    circuit 2m^2-1    rank-1, order=1     rank-2, order=1
   ====  =======  ================  ==================  ==================
   6     21       71                66                  132
   10    55       199               134                 268
   14    105      391               234                 468
   ====  =======  ================  ==================  ==================

   ``param_matched=True`` solves for the ``r`` whose count is closest to ``2m^2 - 1`` and records
   it in ``circuit_spec``.  ``rank=None`` keeps the legacy full-rank behaviour.

**Two amplitude sets, squared -- that is what fixes the tail.**  ``osc``/``ent`` live in the
small-``p`` tail, so a mismatch there silently rigs the comparison.  Reading the two random
projections as a complex amplitude and taking ``|a|^2`` mirrors the quantum structure and lands on
Porter-Thomas -- the law a Haar-random photonic circuit obeys -- because ``a_re, a_im`` are
near-Gaussian and ``a_re^2 + a_im^2`` is then exponential.  A single real amplitude set gives too
many near-zeros; a Gibbs link ``p ~ exp(theta.phi)`` gets the tail *width* right but the *shape*
wrong (Gaussian ``log p``, whose left tail is far thinner than Porter-Thomas's Gumbel-min:
``log10 p`` at the 0.1st percentile ``-3.87`` against the photonic ``-4.95``).

**Reading a result.**  ``parity`` is the calibration control, not just a baseline: a map with too
much high-frequency content is hard to learn *whatever* observable sits on top.  So ``parity`` must
come out easy here; if it does not, lower :data:`FOURIER_ORDER` until it does, and only then read
the nonlinear rows.

Carried from ``model/ebm_fock.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import DistributionModel
from .features import TEACHER_FOURIER_VERSION, fourier_dim, fourier_features
from circuit.fock import fock_keys, n_fock

if TYPE_CHECKING:
    from config import ExperimentConfig

#: Cap on ``n_fock``: bounds the ``(n_fock, d_phi)`` feature matrix and the normalising sum.
#: Unlike ``mlp_fock``'s cap this bounds *scaffolding*, not the model.
FOCK_MAX = 131_072

#: Fourier order of the input angles, so ``d_x = 2 * order * n_features``.  The map is linear in
#: these features, so this is the only frequency-content knob.
FOURIER_ORDER = 2


def occupation_features(keys, m: int) -> np.ndarray:
    """``(n_out, m(m+1)/2)`` outcome features ``[n_i] + [n_i n_j]_{i<j}``.

    ``m`` linear terms plus ``m(m-1)/2`` distinct pairwise products -- only ``O(m^2)`` of them, so
    the amplitude stays a poly-size description, but the pairwise block means the model carries
    inter-mode *correlations* rather than being a structureless product distribution.
    """
    occ = np.asarray([[int(key[i]) for i in range(m)] for key in keys], dtype=np.float64)
    iu, ju = np.triu_indices(m, k=1)
    return np.concatenate([occ, occ[:, iu] * occ[:, ju]], axis=1)


def matched_rank(m: int, d_x: int, d_phi: int) -> int:
    """The rank whose parameter count ``2r(d_x + d_phi)`` is closest to the circuit's ``2m^2 - 1``.

    Floored at 1, since rank 0 is no model at all.  Recorded in ``circuit_spec``, so a
    ``param_matched`` dataset is identified by the rank it resolved to rather than by the flag.
    """
    from circuit.photonic_circuit import n_circuit_parameters

    target = n_circuit_parameters(m)
    per_rank = 2 * (int(d_x) + int(d_phi))
    return max(1, int(round(target / per_rank)))


class QuadraticFockModel(DistributionModel):
    """``X -> probs``: a poly-size random quadratic distribution over the Fock basis.

    Drop-in comparable with :class:`~v2.model.photonic.PhotonicModel` and
    :class:`~v2.model.mlp_fock.MlpFockModel` at the same ``(m, k, n_features)`` -- same outcome
    basis, same scoring code, a poly-size classical map.  ``k`` is the photon number (it fixes the
    outcome count), not a depth.
    """

    name = "quadratic_fock"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 fourier_order: int = FOURIER_ORDER, rank: int | None = None,
                 param_matched: bool = False):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        n_out = n_fock(m, k)
        if n_out > FOCK_MAX:
            raise ValueError(f"n_fock = C(m+k-1, k) = {n_out} exceeds FOCK_MAX={FOCK_MAX} for "
                             f"(m={m}, k={k}); normalising sums over every outcome -- lower m or k")
        self.fourier_order = int(fourier_order)
        self._keys = fock_keys(self.m, self.k)

        phi = occupation_features(self._keys, self.m)
        # Standardised per feature so no single occupation moment dominates the amplitude, which
        # keeps a(x, .) close to Gaussian across outcomes -- hence p close to Porter-Thomas.
        phi = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-12)
        self.d_phi = phi.shape[1]
        self.register_buffer("phi", torch.tensor(phi, dtype=torch.float32))    # scaffolding

        d_x = fourier_dim(self.fourier_order, n_features)
        self.d_x = d_x
        if param_matched:
            if rank is not None:
                raise ValueError("give either model.rank or model.param_matched, not both")
            rank = matched_rank(self.m, d_x, self.d_phi)
        self.rank = None if rank is None else int(rank)
        self.param_matched = bool(param_matched)

        gen = torch.Generator().manual_seed(self.seed)
        if self.rank is None:
            # Legacy full-rank: W is drawn directly, scaled so a(x, .) has O(1) variance.
            scale = 1.0 / math.sqrt(d_x)
            for nm in ("W_re", "W_im"):
                self.register_buffer(nm, torch.randn(d_x, self.d_phi, generator=gen) * scale)
        else:
            # Low-rank W = A B^T.  The scale keeps Var(a) comparable to the full-rank case: a is a
            # sum of `rank` products of two O(1) projections, so each factor carries 1/sqrt of its
            # own fan-in and the sum carries 1/sqrt(rank).
            r = self.rank
            a_scale = 1.0 / math.sqrt(d_x)
            b_scale = 1.0 / math.sqrt(max(r, 1))
            for nm in ("A_re", "A_im"):
                self.register_buffer(nm, torch.randn(d_x, r, generator=gen) * a_scale)
            for nm in ("B_re", "B_im"):
                self.register_buffer(nm, torch.randn(self.d_phi, r, generator=gen) * b_scale)

        self._autosize_batch(n_out)

    def _weights(self):
        """``(W_re, W_im)``, materialised from the factors when low-rank."""
        if self.rank is None:
            return self.W_re, self.W_im
        return self.A_re @ self.B_re.T, self.A_im @ self.B_im.T

    def n_model_parameters(self) -> int:
        """The model proper (the projections), excluding the ``phi`` scaffolding."""
        if self.rank is None:
            return 2 * self.d_x * self.d_phi
        return 2 * self.rank * (self.d_x + self.d_phi)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        f = fourier_features(X, self.fourier_order)                  # (N, d_x)
        if self.rank is None:
            a_re = (f @ self.W_re) @ self.phi.T                      # (N, n_out)
            a_im = (f @ self.W_im) @ self.phi.T
        else:
            # Associativity keeps the low-rank path O(N r (d_x + d_phi)) instead of materialising W.
            a_re = ((f @ self.A_re) @ self.B_re.T) @ self.phi.T
            a_im = ((f @ self.A_im) @ self.B_im.T) @ self.phi.T
        w = a_re * a_re + a_im * a_im
        return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    def outcome_keys(self):
        return self._keys

    def circuit_spec(self) -> dict:
        return {"model": self.name,
                "fourier_order": self.fourier_order,
                "encoding": f"teacher_fourier_v{TEACHER_FOURIER_VERSION}",
                "amplitudes": "bilinear_random_complex",
                "outcome_features": "occupation_moments_1_2",
                # The resolved rank, not the flag: a param_matched dataset is identified by the
                # rank it landed on, so it collides with an explicit rank= that matches.
                "rank": self.rank}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "QuadraticFockModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed,
                   rank=cfg.model.rank, param_matched=cfg.model.param_matched)

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        if cfg.model.rank is not None and cfg.model.param_matched:
            raise ValueError("give either model.rank or model.param_matched, not both")
        if cfg.model.rank is not None and int(cfg.model.rank) < 1:
            raise ValueError(f"model.rank must be >= 1 (got {cfg.model.rank})")
        n_out = n_fock(cfg.problem.m, cfg.problem.k)
        if n_out > FOCK_MAX:
            raise ValueError(f"n_fock={n_out} exceeds FOCK_MAX={FOCK_MAX}")
