"""Capacity-unbounded classical reference: a random MLP emitting a distribution over Fock space.

CAVEAT, read first: this model is **NOT poly-size**.  Its output layer is ``2 * n_fock`` wide, so
its parameter count grows exponentially in ``m`` -- 35.3k at ``m=6`` and 19.9M at ``m=14``, against
the photonic map's ``2m^2 - 1`` = 71 and 391.  That hands the classical side the very resource the
quantum claim is about, so this model **cannot** answer "is a classical model enough"; it answers
only the weaker "is the function learnable by an unconstrained classical map", i.e. it is an upper
bound on what any classical model could do.  For the poly-size control -- same Fock basis,
``O(m^2)`` parameters, and optionally *exactly* parameter-matched -- use
:mod:`v2.model.quadratic_fock`.

Unlike ``quadratic_fock``, whose cubic scaling was an artifact of ``n_features = m - 1`` and
disappears once the input size is fixed, this model's exponential count is **structural**: the
``2 * n_fock`` output layer does not depend on ``n_features`` at all.  That is its job, and it
should keep saying so loudly.

The point is to answer one question about the nonlinear observables (``ent``, ``osc``): *is their
hardness a property of the ``p -> score`` functional, or does it need the quantum feature map?*  So
it reproduces everything about the photonic model except the map:

* same outcome basis -- the ``C(m+k-1, k)`` occupations (:func:`v2.model.fock.fock_keys`, pure
  combinatorics, no merlin), so ``parity`` and friends are *literally the same function of the
  outcome* on both platforms;
* same scoring -- the distribution goes straight into the shared observable registry;
* different ``x -> p`` -- a fixed random tanh MLP over Fourier features instead of ``W2 P(x) W1``
  boson sampling.  No permanents, nothing quantum, poly-*time* by construction.

**Why two heads and not a softmax.**  The MLP emits ``2 * n_fock`` numbers read as a complex
amplitude ``a = a_re + i a_im``, and ``p = |a|^2 / sum |a|^2``.  This mirrors the quantum structure
and, because the pre-activations are near-Gaussian, lands on the Porter-Thomas law a Haar-random
photonic circuit also produces -- so the two platforms' ``p`` have matching *marginal* statistics
and only their *structure* differs.  That matters more than it sounds: ``osc``'s oscillation lives
entirely in the small-``p`` tail, so a normalisation with the wrong tail silently rigs the
experiment.  Measured at ``m=6, k=3`` over 500 inputs:

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
``osc`` and hand back a false "the functional is benign".  For reference, exact Porter-Thomas
(``|Gaussian|^2``, ``n_fock=56``) gives ``-6.49 / -1.90``, ``H = 3.617``, ``0.006``, ``0.1346``.

**Reading a result.**  ``parity`` is the calibration control: a random MLP with too much
high-frequency content is hard to learn *whatever* observable you put on it.  So ``parity`` must
come out easy here; if it does not, lower :data:`FOURIER_ORDER` / :data:`WEIGHT_GAIN` until it
does, and only then read ``osc``.  With ``parity`` easy: ``osc`` easy too means the functional is
benign and the photonic hardness belongs to the feature map; ``osc`` hard means the hardness rides
on ``p -> score`` and reproduces with nothing quantum.

Carried from ``model/mlp_fock.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import DistributionModel
from .features import TEACHER_FOURIER_VERSION, fourier_dim, fourier_features
from circuit.fock import fock_keys, n_fock

if TYPE_CHECKING:
    from config import ExperimentConfig

#: Cap on ``n_fock``: the output layer is ``2 * n_fock`` wide, so this bounds the parameter count.
#: ``m=12, k=6`` -> 12376; ``m=16, k=8`` -> 490314, which is past it.
FOCK_MAX = 65_536

#: Fourier order of the input angles, and the dominant frequency-content knob: raising it makes
#: ``x -> p`` wigglier and so harder to learn *for every observable*.  Turn this down first if the
#: ``parity`` calibration comes out hard.
FOURIER_ORDER = 3

#: Hidden layers and width.  The width also sets how Gaussian the amplitudes are (each output is a
#: sum of ``HIDDEN`` tanh terms), hence how closely ``p`` tracks Porter-Thomas -- keep it well
#: above ~64.
DEPTH = 2
HIDDEN = 128

#: Gain on the Xavier init.  Secondary frequency-content knob: >1 sharpens the map, <1 smooths it.
WEIGHT_GAIN = 1.0


class MlpFockModel(DistributionModel):
    """``X -> probs``: a random MLP's normalised complex amplitudes over the Fock basis.

    Drop-in comparable with the photonic model at the same ``(m, k, n_features)``.  ``k`` is the
    photon number (it fixes the outcome count), *not* the network depth; the architecture knobs are
    the module constants above and are surfaced in ``circuit_spec`` so changing one re-identifies
    the dataset.
    """

    name = "mlp_fock"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 depth: int = DEPTH, hidden_size: int = HIDDEN,
                 fourier_order: int = FOURIER_ORDER, weight_gain: float = WEIGHT_GAIN):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        n_out = n_fock(m, k)
        if n_out > FOCK_MAX:
            raise ValueError(f"n_fock = C(m+k-1, k) = {n_out} exceeds FOCK_MAX={FOCK_MAX} for "
                             f"(m={m}, k={k}); the output layer would be {2 * n_out} wide -- "
                             "lower m or k")
        self.fourier_order = int(fourier_order)
        self.depth, self.hidden_size = int(depth), int(hidden_size)
        self.weight_gain = float(weight_gain)
        self._keys = fock_keys(self.m, self.k)

        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        in_size = fourier_dim(self.fourier_order, n_features)
        for _ in range(max(self.depth, 1)):
            lin = nn.Linear(in_size, self.hidden_size, bias=False)
            nn.init.xavier_uniform_(lin.weight,
                                    gain=self.weight_gain * nn.init.calculate_gain("tanh"))
            layers += [lin, nn.Tanh()]
            in_size = self.hidden_size
        out = nn.Linear(in_size, 2 * n_out, bias=False)
        nn.init.xavier_uniform_(out.weight, gain=self.weight_gain)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.net.eval()

        self._autosize_batch(n_out)

    def n_model_parameters(self) -> int:
        """Every weight in the net -- dominated by the ``2 * n_fock`` output layer, hence exponential."""
        return sum(p.numel() for p in self.net.parameters())

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        """``|a_re + i a_im|^2`` normalised per row; ``eps`` guards only the all-zero row."""
        a = self.net(fourier_features(X, self.fourier_order))
        re, im = a.chunk(2, dim=-1)
        w = re * re + im * im
        return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    def outcome_keys(self):
        return self._keys

    def circuit_spec(self) -> dict:
        return {"model": self.name,
                "fourier_order": self.fourier_order,
                "encoding": f"teacher_fourier_v{TEACHER_FOURIER_VERSION}",
                "depth": self.depth, "hidden_size": self.hidden_size,
                "weight_gain": self.weight_gain,
                "normalisation": "complex_amplitude"}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "MlpFockModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed)

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        n_out = n_fock(cfg.problem.m, cfg.problem.k)
        if n_out > FOCK_MAX:
            raise ValueError(f"n_fock={n_out} exceeds FOCK_MAX={FOCK_MAX} for "
                             f"(m={cfg.problem.m}, k={cfg.problem.k})")
