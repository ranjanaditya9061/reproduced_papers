"""Capacity-unbounded classical reference: a random MLP emitting a distribution over the Fock basis.

CAVEAT, read first: this model is NOT poly-size.  Its output layer is ``2 * n_fock`` wide, so its
parameter count grows exponentially in ``m`` -- 34.6k at ``m=6`` and 19.9M at ``m=14``, against the
photonic map's ``2 m^2`` = 72 and 392 (a 50,693x gap).  That hands the classical side the very
resource the quantum claim is about, so this teacher CANNOT answer "is a classical model enough";
it answers only the weaker "is the function learnable by an unconstrained classical map", i.e. it
is an upper bound on what any classical model could do.  For the actual poly-size control -- same
Fock basis, ``O(m^2)`` parameters -- use :mod:`model.ebm_fock`.

The point of this teacher is to answer one question about the nonlinear observables
(:mod:`model.photonic_observables.entropy`, :mod:`~model.photonic_observables.oscillatory`):
**is their hardness a property of the ``p -> score`` functional, or does it need the quantum
feature map?**  So it reproduces everything about :class:`~model.photonic.PhotonicTeacher` except
the map:

* same outcome basis -- the ``C(m+k-1, k)`` ``k``-photon occupation vectors over ``m`` modes
  (:func:`fock_keys`, pure combinatorics, no ``merlin``), so ``parity`` and friends are *literally
  the same function of the outcome* on both platforms;
* same scoring -- the distribution goes straight into
  :func:`~model.photonic_observables.base.resolve_observable`, i.e. the identical code path;
* different ``x -> p`` -- a fixed random tanh MLP over Fourier features instead of
  ``W2 P(x) W1`` boson sampling.  No permanents, nothing quantum, poly-time by construction.

Any difference in results is therefore attributable to the map alone.

**Why two heads and not a softmax.**  The MLP emits ``2 * n_fock`` numbers read as a complex
amplitude ``a = a_re + i a_im``, and ``p = |a|^2 / sum |a|^2``.  This mirrors the quantum structure
(``p = |amplitude|^2``) and, because the pre-activations are near-Gaussian, lands on the
Porter-Thomas law that a Haar-random photonic circuit also produces -- so the two platforms' ``p``
have matching *marginal* statistics and only their *structure* differs.  That matters more than it
sounds: ``osc``'s oscillation lives entirely in the small-``p`` tail, so a normalisation with the
wrong tail silently rigs the experiment.  Measured at ``m=6, k=3`` over 500 inputs:

=============================  ==================  ========  ============  =========
normalisation                  log10 p  min / med  entropy   frac p<1e-4   std(osc)
=============================  ==================  ========  ============  =========
photonic |Perm|^2  (target)      -6.95 / -1.96      3.509       0.011        0.1393
softmax(logits)                  -2.67 / -1.80      3.916       0.000        0.1014
softmax(4*logits)                -5.99 / -2.46      2.625       0.035        0.2949
|a|^2, one real head             -9.21 / -2.08      3.317       0.057        0.1588
|a_re + i a_im|^2  (used here)   -6.62 / -1.90      3.609       0.005        0.1298
=============================  ==================  ========  ============  =========

A plain softmax has no small-``p`` tail at all (nothing below ``1e-4``), which would understate
``osc`` and hand back a false "the functional is benign".  The complex-amplitude form matches the
target on every column; for reference, exact Porter-Thomas (``|Gaussian|^2``, ``n_fock=56``) gives
``-6.49 / -1.90``, ``H = 3.617``, ``0.006``, ``0.1346``.

**Reading a result.**  ``parity`` is the calibration control, not just a baseline: a random MLP
with too much high-frequency content is hard to learn *whatever* observable you put on it.  So
``parity`` must come out easy here; if it does not, lower :data:`MLP_FOURIER_ORDER` /
:data:`MLP_WEIGHT_GAIN` until it does, and only then read ``osc``.  With ``parity`` easy: ``osc``
easy too means the functional is benign and the photonic hardness belongs to the feature map;
``osc`` hard means the hardness rides on ``p -> score`` and reproduces with nothing quantum.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from .base import Teacher
from .mlp import fourier_features
from .photonic_observables import (ObservableContext, is_known_observable, observable_hash_spec,
                                   observable_help, resolve_observable)

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

#: Cap on ``n_fock = C(m+k-1, k)``: the output layer is ``2 * n_fock`` wide, so this bounds the
#: parameter count.  ``m=12, k=6`` -> 12376; ``m=16, k=8`` -> 490314, which is past it.
MLP_FOCK_MAX = 65_536

#: Fourier expansion order of the input angles, and the dominant frequency-content knob: raising
#: it makes ``x -> p`` wigglier and so harder to learn *for every observable*.  This is the dial to
#: turn if the ``parity`` calibration comes out hard (see the module docstring).
MLP_FOURIER_ORDER = 3

#: Hidden layers and width.  The width also sets how Gaussian the amplitudes are (each output is a
#: sum of ``MLP_HIDDEN`` tanh terms), hence how closely ``p`` tracks Porter-Thomas -- keep it
#: comfortably above ~64.
MLP_DEPTH = 2
MLP_HIDDEN = 128

#: Gain on the Xavier init.  Secondary frequency-content knob: >1 sharpens the map, <1 smooths it.
MLP_WEIGHT_GAIN = 1.0


def fock_keys(m: int, k: int) -> list[tuple[int, ...]]:
    """The ``C(m+k-1, k)`` ``k``-photon occupation vectors over ``m`` modes, canonically ordered.

    The same *set* of outcome labels a ``merlin`` ``QuantumLayer`` reports for ``(m, k)``, built
    combinatorially so this teacher needs no quantum dependency.  The enumeration order need not
    match merlin's: every observable scores ``Σ_n v(n) φ(p(n))`` with ``v`` and ``p`` indexed by the
    same enumeration, so the score is invariant to it.  (Align the order if you ever want to
    compare the two platforms' ``p`` vectors entry-by-entry rather than their scores.)
    """
    m, k = int(m), int(k)
    if m < 1 or k < 1:
        raise ValueError(f"need m >= 1 and k >= 1 (got m={m}, k={k})")
    keys = []
    for combo in combinations_with_replacement(range(m), k):
        occ = [0] * m
        for i in combo:
            occ[i] += 1
        keys.append(tuple(occ))
    return keys


class MlpFockTeacher(Teacher):
    """``X -> soft``: a random MLP's normalised amplitudes over the Fock basis, scored by ``observable``.

    Drop-in comparable with :class:`~model.photonic.PhotonicTeacher` at the same ``(m, k,
    observable, n_features)`` -- same outcome basis, same scorers, classical map.  ``k`` is the
    photon number (it fixes the outcome count), *not* the network depth; the architecture knobs are
    the module constants above and are surfaced in :meth:`hash_spec` so changing one re-identifies
    the dataset.
    """

    name = "mlp_fock"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0,
                 depth: int = MLP_DEPTH, hidden_size: int = MLP_HIDDEN,
                 fourier_order: int = MLP_FOURIER_ORDER, weight_gain: float = MLP_WEIGHT_GAIN,
                 n_vertices: int | None = None, graph_seed: int | None = None,
                 angle_seed: int | None = None, graph_density: float | None = None):
        super().__init__(n_features)
        if not is_known_observable(observable):
            raise ValueError(f"unknown observable {observable!r}; expected one of: "
                             f"{observable_help()}")
        n_fock = comb(m + k - 1, k)
        if n_fock > MLP_FOCK_MAX:
            raise ValueError(f"n_fock = C(m+k-1, k) = {n_fock} exceeds MLP_FOCK_MAX="
                             f"{MLP_FOCK_MAX} for (m={m}, k={k}); the output layer would be "
                             f"{2 * n_fock} wide -- lower m or k")
        self.m, self.k, self.observable, self.nsample = int(m), int(k), observable, int(nsample)
        self.seed = int(seed)
        self.fourier_order = int(fourier_order)
        self.depth, self.hidden_size = int(depth), int(hidden_size)
        self.weight_gain = float(weight_gain)
        self._noise_seed = self.seed + 13
        self._capture = False
        self._dist_probs: list = []

        self._fock_keys = fock_keys(self.m, self.k)
        assert len(self._fock_keys) == n_fock                 # stars and bars

        # Fixed random tanh MLP -> 2 * n_fock outputs, read as (a_re, a_im).
        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        in_size = 2 * self.fourier_order * n_features
        for _ in range(max(self.depth, 1)):
            lin = nn.Linear(in_size, self.hidden_size, bias=False)
            nn.init.xavier_uniform_(lin.weight,
                                    gain=self.weight_gain * nn.init.calculate_gain("tanh"))
            layers += [lin, nn.Tanh()]
            in_size = self.hidden_size
        out = nn.Linear(in_size, 2 * n_fock, bias=False)
        nn.init.xavier_uniform_(out.weight, gain=self.weight_gain)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.net.eval()

        self.obs = resolve_observable(observable, ObservableContext(
            m=self.m, k=self.k, keys=self._fock_keys, seed=self.seed, graph_seed=graph_seed,
            angle_seed=angle_seed, n_vertices=n_vertices, graph_density=graph_density,
            input_state=None, reference_probs=self.exact_probs_at_zero))
        # Match PhotonicTeacher: cap the per-call (N, n_fock) intermediate at ~128 MB fp32.
        self.forward_batch = max(1, 33_554_432 // max(n_fock, 1))

    # --- the map ---------------------------------------------------------------------------- #

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, n_fock)`` distribution: ``|a_re + i a_im|^2`` normalised per row.

        The two heads give the complex-amplitude / Porter-Thomas form the module docstring
        justifies.  ``eps`` in the denominator only guards the measure-zero all-zero row.
        """
        a = self.net(fourier_features(X, self.fourier_order))
        re, im = a.chunk(2, dim=-1)
        w = re * re + im * im
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
        return self.obs.score(probs).unsqueeze(-1)            # (N_chunk, 1)

    def _shot_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Replace each row by its ``nsample``-shot empirical distribution (the finite-shot teacher).

        The classical analogue of merlin's ``shots``: it makes the shot-estimability comparison
        run on both platforms.  Seeded off ``seed``, so a matched seed reproduces the draw.
        """
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
        """Recorded distributions, in the same dict shape :func:`model.spoqc_magic.write_distributions` takes."""
        if not self._dist_probs:
            raise RuntimeError("no distributions captured; call "
                               "enable_distribution_capture() before forward()")
        keys = np.array(self._fock_keys, dtype=np.int16)
        return {"keys": keys, "probs": np.vstack(self._dist_probs), "readout_modes": (),
                "m": self.m, "k": self.k, "observable": self.observable,
                "t_var": None, "seed": self.seed}

    def save_distributions(self, path):
        """Write the captured distributions to ``path`` (a ``.npz``); returns the path."""
        from .spoqc_magic import write_distributions
        return write_distributions(path, self.captured_distributions())

    # --- self-description -------------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "MlpFockTeacher":
        p = cfg.problem
        return cls(m=p.m, k=p.k, n_features=cfg.resolved_n_features,
                   observable=p.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample, n_vertices=p.n_vertices,
                   graph_seed=p.graph_seed, angle_seed=p.angle_seed,
                   graph_density=p.graph_density)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        """Observable identity (shared with the photonic teacher) plus the architecture knobs."""
        p = cfg.problem
        spec = {"observable": p.observable, "nsample": cfg.generation.nsample,
                "fourier_order": MLP_FOURIER_ORDER, "depth": MLP_DEPTH,
                "hidden_size": MLP_HIDDEN, "weight_gain": MLP_WEIGHT_GAIN,
                "normalisation": "complex_amplitude"}
        spec.update(observable_hash_spec(p.observable, ObservableContext(
            m=p.m, k=p.k, seed=cfg.seeds.teacher_seed, graph_seed=p.graph_seed,
            angle_seed=p.angle_seed, n_vertices=p.n_vertices, graph_density=p.graph_density)))
        return spec
