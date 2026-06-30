"""Spin-photon teacher built on ``perceval_spoqc`` (HybridProcessor).

Same idea as :class:`model.photonic.PhotonicTeacher`, but the photonic **input
state** is *prepared by spin qubits* instead of being a fixed Fock state:

    n_q = k qubits, each |0> -> H -> seeded Rx(a_q) Ry(b_q),
    emitted dual-rail into modes (2q, 2q+1),
    then the SAME embedding W1(Haar) -> PS(x_i) -> W2(Haar) and the SAME
    observable (parity / majority / bunching) as the photonic teacher.

The seeded ``Rx``/``Ry`` twists do two things:
- they push each qubit off the 50/50 point, so the photons' one-body matrix is no
  longer ``Gamma = 1/2 I`` -- which revives ``majority`` (otherwise pinned to 0,
  since a Haar interferometer preserves ``Gamma proportional to I``);
- being drawn from ``teacher_seed``, they make the teacher seed-dependent (so a
  matched-seed kernel vs a random-seed kernel is a meaningful distinction).

Dual-rail is forced by the simulator (a dim-2 qubit emits into exactly 2 modes),
so ``n_q`` qubits occupy ``2*n_q`` modes (need ``2*k <= m``).  ``perceval_spoqc``
evaluates one circuit per input (no batching) -> keep pools small.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .photonic import _bunching_score, _majority_score, _parity_score

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

OBSERVABLES = ("parity", "majority", "bunching")


def _sandwich_concrete(m: int, n_features: int, seed: int, x):
    """``W1(Haar) -> PS(x_i) -> W2(Haar)`` with concrete phase values (per sample).

    Same construction (and seeding) as :func:`model.photonic.build_sandwich_circuit`,
    but with numeric ``PS`` so it can be dropped into a HybridProcessor.
    """
    import perceval as pcvl

    pcvl.random_seed(seed)
    torch.manual_seed(seed)
    c = pcvl.Circuit(m, name="haar_phase_haar")
    c.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W1
    for i in range(n_features):
        c.add(i, pcvl.PS(float(x[i])))
    c.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W2
    return c


def _build_processor(x, *, m, n_q, n_features, seed, rx, ry):
    """Construct the spin-prepared photonic HybridProcessor for one input ``x``."""
    from perceval import Detector
    from perceval_spoqc import Gate, HybridProcessor

    p = HybridProcessor(num_sources=n_q, num_modes=m)
    rho0 = np.zeros((2 ** n_q, 2 ** n_q), dtype=complex)
    rho0[0, 0] = 1.0                                 # joint |0...0>
    p.with_initial_source_state(rho0)
    for q in range(n_q):
        p.gate.h(q)                                  # |0> -> |+>
        p.add_source_op(q, Gate.Rx(float(rx[q])))    # seeded random twists -> Gamma != 1/2 I
        p.add_source_op(q, Gate.Ry(float(ry[q])))
        p.emit(q, into=(2 * q, 2 * q + 1))           # dual-rail emission into its pair

    p.add(0, _sandwich_concrete(m, n_features, seed, x))   # same embedding as the photonic teacher
    for mode in range(m):
        p.add(mode, Detector())
    return p


def _spoqc_soft_row(x, *, m, n_q, n_features, observable, seed, rx, ry) -> float:
    """Continuous score for one input ``x`` from the spin-prepared photonic circuit."""
    p = _build_processor(x, m=m, n_q=n_q, n_features=n_features, seed=seed, rx=rx, ry=ry)

    parity_modes = tuple(range((m + 1) // 2))
    s = 0.0
    for key, pr in p.probabilities().items():
        if observable == "parity":
            s += pr * _parity_score(key, parity_modes)
        elif observable == "majority":
            s += pr * _majority_score(key, m, n_q)     # normalise by photon count n_q
        else:  # bunching
            s += pr * _bunching_score(key)
    return float(s)


def render_circuit(path="spoqc.png", *, m=6, k=3, n_features=5, seed=42, x=None):
    """Render the spoqc circuit once to ``path`` (structure is the same for all x)."""
    out = SpoqcPhotonicTeacher(m=m, k=k, n_features=n_features, seed=seed).render(path, x=x)
    print(f"[spoqc] saved circuit -> {out}")
    return out


class SpoqcPhotonicTeacher(Teacher):
    """Spin-prepared photonic teacher (``perceval_spoqc``); ``soft`` in ``[-1, 1]``."""

    name = "spoqc_photonic"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234):
        super().__init__(n_features)
        if observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
        if m % 2:
            raise ValueError("spoqc_photonic uses dual-rail photons -> requires even m")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        self.m, self.k, self.observable, self.seed = m, k, observable, int(seed)
        # seeded random single-qubit twists, one (Rx, Ry) per qubit
        rng = np.random.default_rng(self.seed)
        self.rx = rng.uniform(0.0, 2 * np.pi, size=k)
        self.ry = rng.uniform(0.0, 2 * np.pi, size=k)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xn = X.detach().cpu().numpy()
        vals = [
            _spoqc_soft_row(row, m=self.m, n_q=self.k, n_features=self.n_features,
                            observable=self.observable, seed=self.seed, rx=self.rx, ry=self.ry)
            for row in Xn
        ]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def render(self, path: str, x=None) -> str:
        """Render the spin-photon circuit (structure is identical for every x)."""
        import matplotlib.pyplot as plt
        from perceval_spoqc import Format, pdisplay

        xv = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        p = _build_processor(xv, m=self.m, n_q=self.k, n_features=self.n_features,
                             seed=self.seed, rx=self.rx, ry=self.ry)
        pdisplay(p, output_format=Format.MPLOT, expanded=True)
        plt.savefig(path)
        plt.close("all")                              # avoid the >20-figures leak
        return path

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "SpoqcPhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        return {"observable": cfg.problem.observable, "prep": "H_Rx_Ry_seeded"}