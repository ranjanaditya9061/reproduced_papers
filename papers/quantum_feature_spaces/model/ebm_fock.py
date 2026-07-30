"""Poly-size classical control: random bilinear amplitudes over the photonic Fock basis.

    a(x, n) = f(x)^T W phi(n)      for two independent random W
    p_x(n)  = (a_re^2 + a_im^2) / sum_n' (...)

``phi(n)`` is a fixed ``O(m^2)`` feature map of the *outcome* -- the occupation moments
``[n_i] + [n_i n_j]_{i<j}``, i.e. ``m(m+1)/2`` numbers, the same statistics
:mod:`embedding.features` uses for the photonic projected kernel.  ``f(x)`` is the Fourier
expansion of the input angles.  The model is then just two random matrices ``W_re, W_im`` -- no
network, no training, no temperature.

**Why this shape.**  The photonic map is ``2m^2`` numbers (two ``m x m`` unitaries) generating a
distribution over ``C(m+k-1, k)`` outcomes: *poly parameters, exponential support*.  That asymmetry
is exactly what a quantum circuit buys, so a control that is meant to answer "is a small classical
model enough" has to have it too.  This one does -- ``2 * dim f(x) * m(m+1)/2`` parameters, i.e.
``O(m^3)``, 840 of them at ``m=6`` against the photonic map's 72:

=========  ========  =================  ==================  ===============
m          n_fock    ebm_fock params    mlp_fock params     photonic 2m^2
=========  ========  =================  ==================  ===============
6            56           840                34.6k              72
10         2002          2,310              536k              200
14        77520          4,940             19.9M              392
=========  ========  =================  ==================  ===============

Its sibling :mod:`model.mlp_fock` emits the ``n_fock`` probabilities from a dense output layer, so
its *description* grows exponentially (19.9M at ``m=14``).  That hands the classical side the very
resource the quantum claim is about, which is why it can only upper-bound what an unconstrained
classical map could do.  Normalising here also sums over every outcome, so exact inference costs
exponential *work* -- but that is our simulation, not the model, exactly as ``merlin`` computing the
full Fock distribution is a simulation artifact rather than what a boson sampler does.  ``phi`` is
likewise scaffolding; the model is ``W_re, W_im``.

**Two amplitude sets, not one, and squared -- that is what fixes the tail.**  ``osc``/``ent`` live
in the small-``p`` tail, so a mismatch there silently rigs the comparison.  Reading the two random
projections as a complex amplitude and taking ``|a|^2`` mirrors the quantum structure
(``p = |amplitude|^2``) and lands on Porter-Thomas -- the same law a Haar-random photonic circuit
obeys -- because ``a_re, a_im`` are near-Gaussian sums and ``a_re^2 + a_im^2`` is then exponential.
A single real amplitude set gives too many near-zeros, and a Gibbs link ``p ~ exp(theta.phi)`` gets
the tail *width* right but the *shape* wrong (Gaussian ``log p``, whose left tail is far thinner
than Porter-Thomas's Gumbel-min: ``log10 p`` at the 0.1st percentile ``-3.87`` against the photonic
``-4.95``).  The measured comparison lives in :mod:`model.mlp_fock`, which uses the identical trick.

**Reading a result.**  ``parity`` is the calibration control, not just a baseline: a map with too
much high-frequency content is hard to learn *whatever* observable sits on top.  So ``parity`` must
come out easy here; if it does not, lower :data:`EBM_FOURIER_ORDER` until it does, and only then
read the nonlinear rows.
"""

from __future__ import annotations

import math
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .mlp import fourier_features
from .mlp_fock import fock_keys
from .photonic_observables import (ObservableContext, is_known_observable, observable_hash_spec,
                                   observable_help, resolve_observable)

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

#: Cap on ``n_fock``: bounds the ``(n_fock, m(m+1)/2)`` feature matrix and the normalising sum.
#: Unlike :data:`model.mlp_fock.MLP_FOCK_MAX` this bounds *scaffolding*, not the model.
EBM_FOCK_MAX = 131_072

#: Fourier order of the input angles, so ``dim f(x) = 2 * order * n_features``.  The map is linear
#: in these features, so this is the only frequency-content knob -- the dial to turn if the
#: ``parity`` calibration comes out hard.
EBM_FOURIER_ORDER = 2


def occupation_features(keys, m: int) -> np.ndarray:
    """``(n_fock, m(m+1)/2)`` outcome features ``[n_i] + [n_i n_j]_{i<j}``.

    The occupation moments: ``m`` linear terms plus ``m(m-1)/2`` distinct pairwise products.  Only
    ``O(m^2)`` of them, so the amplitude is a poly-size description -- but the pairwise block means
    the model still carries inter-mode *correlations*, unlike a product distribution which would be
    trivially structureless.
    """
    occ = np.asarray([[int(key[i]) for i in range(m)] for key in keys], dtype=np.float64)
    iu, ju = np.triu_indices(m, k=1)
    return np.concatenate([occ, occ[:, iu] * occ[:, ju]], axis=1)


class EbmFockTeacher(Teacher):
    """``X -> soft``: a poly-size random bilinear distribution over the Fock basis, scored by ``observable``.

    Drop-in comparable with :class:`~model.photonic.PhotonicTeacher` and
    :class:`~model.mlp_fock.MlpFockTeacher` at the same ``(m, k, observable, n_features)`` -- same
    outcome basis, same scoring code, a poly-size classical map.  ``k`` is the photon number (it
    fixes the outcome count).
    """

    name = "ebm_fock"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0,
                 fourier_order: int = EBM_FOURIER_ORDER,
                 n_vertices: int | None = None, graph_seed: int | None = None,
                 angle_seed: int | None = None, graph_density: float | None = None):
        super().__init__(n_features)
        if not is_known_observable(observable):
            raise ValueError(f"unknown observable {observable!r}; expected one of: "
                             f"{observable_help()}")
        n_fock = comb(m + k - 1, k)
        if n_fock > EBM_FOCK_MAX:
            raise ValueError(f"n_fock = C(m+k-1, k) = {n_fock} exceeds EBM_FOCK_MAX="
                             f"{EBM_FOCK_MAX} for (m={m}, k={k}); normalising sums over every "
                             f"outcome -- lower m or k")
        self.m, self.k, self.observable, self.nsample = int(m), int(k), observable, int(nsample)
        self.seed = int(seed)
        self.fourier_order = int(fourier_order)
        self._noise_seed = self.seed + 13
        self._capture = False
        self._dist_probs: list = []

        self._fock_keys = fock_keys(self.m, self.k)
        phi = occupation_features(self._fock_keys, self.m)
        # Standardised per feature so no single occupation moment dominates the amplitude, which
        # keeps a(x, .) close to Gaussian across outcomes -- hence p close to Porter-Thomas.
        phi = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-12)
        self.n_energy_features = phi.shape[1]
        self.register_buffer("phi", torch.tensor(phi, dtype=torch.float32))   # scaffolding

        # The model: two random projections from the x-features to the outcome-feature weights.
        d_x = 2 * self.fourier_order * n_features
        gen = torch.Generator().manual_seed(self.seed)
        scale = 1.0 / math.sqrt(d_x)
        for name in ("W_re", "W_im"):
            self.register_buffer(name, torch.randn(d_x, self.n_energy_features,
                                                   generator=gen) * scale)

        self.obs = resolve_observable(observable, ObservableContext(
            m=self.m, k=self.k, keys=self._fock_keys, seed=self.seed, graph_seed=graph_seed,
            angle_seed=angle_seed, n_vertices=n_vertices, graph_density=graph_density,
            input_state=None, reference_probs=self.exact_probs_at_zero))
        self.forward_batch = max(1, 33_554_432 // max(n_fock, 1))

    def n_model_parameters(self) -> int:
        """Parameters in the model proper (the two projections), excluding the ``phi`` scaffolding."""
        return self.W_re.numel() + self.W_im.numel()

    # --- the map ---------------------------------------------------------------------------- #

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, n_fock)`` distribution ``|a|^2`` normalised per row, ``a = f(x)^T W phi(n)``."""
        f = fourier_features(X, self.fourier_order)                   # (N, d_x)
        a_re = (f @ self.W_re) @ self.phi.T                           # (N, n_fock)
        a_im = (f @ self.W_im) @ self.phi.T
        w = a_re * a_re + a_im * a_im
        return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    @torch.no_grad()
    def exact_probs_at_zero(self) -> torch.Tensor:
        """``q``: the ``(n_fock,)`` distribution at ``x = 0`` -- the reference the ``xent`` family needs."""
        return self.probs(torch.zeros(1, self.n_features))[0]

    @property
    def score_vec(self) -> torch.Tensor:
        """The observable's per-outcome score vector (the linear/diagonal families)."""
        return self.obs.score_vec

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bs = self.forward_batch
        if bs is None or bs <= 0 or X.shape[0] <= bs:
            return self._forward_chunk(X)
        chunks = [self._forward_chunk(X[i:i + bs]) for i in range(0, X.shape[0], bs)]
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _forward_chunk(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.probs(X)
        if self.nsample > 0:
            probs = self._shot_sample(probs)
        if self._capture:
            self._dist_probs.append(probs.detach().cpu().numpy())
        return self.obs.score(probs).unsqueeze(-1)

    def _shot_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Replace each row by its ``nsample``-shot empirical distribution (finite-shot teacher)."""
        gen = torch.Generator().manual_seed(self._noise_seed)
        counts = torch.multinomial(probs.clamp(min=0), self.nsample, replacement=True,
                                   generator=gen)
        out = torch.zeros_like(probs)
        out.scatter_add_(1, counts, torch.ones_like(counts, dtype=probs.dtype))
        return out / self.nsample

    # --- distribution capture (same shape as PhotonicTeacher's) ----------------------------- #

    def enable_distribution_capture(self, enable: bool = True) -> None:
        """Record every forward's full distribution so it can be persisted and re-scored offline."""
        self._capture = bool(enable)
        self._dist_probs = []

    def captured_distributions(self) -> dict:
        """Recorded distributions, in the shape :func:`model.spoqc_magic.write_distributions` takes."""
        if not self._dist_probs:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        return {"keys": np.array(self._fock_keys, dtype=np.int16),
                "probs": np.vstack(self._dist_probs), "readout_modes": (),
                "m": self.m, "k": self.k, "observable": self.observable,
                "t_var": None, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        from .spoqc_magic import write_distributions
        return write_distributions(path, self.captured_distributions())

    # --- self-description -------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "EbmFockTeacher":
        p = cfg.problem
        return cls(m=p.m, k=p.k, n_features=cfg.resolved_n_features,
                   observable=p.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample, n_vertices=p.n_vertices,
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed,
                   graph_density=p.graph_density)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        """Observable identity (shared with the photonic teacher) plus the map's knobs."""
        p = cfg.problem
        spec = {"observable": p.observable, "nsample": cfg.generation.nsample,
                "fourier_order": EBM_FOURIER_ORDER,
                "amplitudes": "bilinear_random_complex",
                "energy_features": "occupation_moments_1_2"}
        spec.update(observable_hash_spec(p.observable, ObservableContext(
            m=p.m, k=p.k, seed=cfg.seeds.teacher_seed, graph_seed=p.graph_seed,
            angle_seed=p.angle_seed, n_vertices=p.n_vertices, graph_density=p.graph_density)))
        return spec