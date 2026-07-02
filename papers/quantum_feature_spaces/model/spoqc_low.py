"""Spin-photon teacher built on ``perceval_spoqc`` (HybridProcessor).

Same idea as :class:`model.photonic.PhotonicTeacher`, but the photonic **input
state** is prepared by spin qubits instead of a fixed Fock state:

    n_q = k qubits, each |0> -> H -> seeded Rx(a_q) Ry(b_q),
    optionally entangled by a CX chain (``problem.cx_pairs``),
    emitted dual-rail into modes (2q, 2q+1),
    then the SAME embedding W1(Haar) -> PS(x_i) -> W2(Haar) and the SAME
    observable (parity / majority / bunching) as the photonic teacher.

The whole spin prep (H, Rx, Ry, then CX) is built in numpy on the initial source
state -- spoqc has no two-qubit processor gate, and a CX on |0...0> is the
identity, so the single-qubit prep must precede the entangler.  ``cx_pairs`` is a
**spoqc-only** knob (like ``observable`` is photonic-only): it affects only this
teacher's artifact hash.  Helpers live in :mod:`model.spoqc_utils`.

``cx_pairs`` examples: ``None``/``[]`` (product, no entangler), ``[[0,1]]`` (one),
``[[0,1],[1,2]]`` (chain), ``[[0,1],[1,2],[2,0]]`` (ring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .spoqc_utils import (  # noqa: F401  (apply_cx re-exported for convenience)
    OBSERVABLES,
    apply_cx,
    _build_processor,
    _spoqc_soft_row,
)

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


def _normalize_cx_pairs(cx_pairs, k: int):
    """Validate + normalise cx_pairs to a list of ``(control, target)`` int tuples."""
    pairs = []
    for pair in (cx_pairs or []):
        c, t = int(pair[0]), int(pair[1])
        if c == t or not (0 <= c < k and 0 <= t < k):
            raise ValueError(f"cx pair {pair} invalid for k={k} qubits (need distinct 0<=i<k)")
        pairs.append((c, t))
    return pairs


class SpoqcLowPhotonicTeacher(Teacher):
    """Spin-prepared photonic teacher (``perceval_spoqc``); ``soft`` in ``[-1, 1]``.

    ``cx_pairs`` (from ``problem.cx_pairs``) sets the spin entangler; ``None`` = the
    product-state base.
    """

    name = "spoqc_low_photonic"

    #: rx/ry are drawn from the DISCRETE grid  {-0.1*L, ..., -0.1, 0.1, ..., 0.1*L}
    #: (multiples of 0.1 up to 0.1*ANGLE_LEVELS, excluding 0).  Raise for a coarser
    #: spread of small twists; folded into the hash so a change re-identifies the dataset.
    ANGLE_LEVELS = 3

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, cx_pairs=None,
                 angle_levels: int | None = None):
        super().__init__(n_features)
        if observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
        if m % 2:
            raise ValueError("spoqc_photonic uses dual-rail photons -> requires even m")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        self.m, self.k, self.observable, self.seed = m, k, observable, int(seed)
        self.cx_pairs = _normalize_cx_pairs(cx_pairs, k)
        self.angle_levels = self.ANGLE_LEVELS if angle_levels is None else int(angle_levels)
        if self.angle_levels < 1:
            raise ValueError(f"angle_levels must be >= 1 (got {self.angle_levels})")
        # discrete choices: {-0.1*L, ..., -0.1, +0.1, ..., +0.1*L}  (no 0)
        levels = 0.1 * np.arange(1, self.angle_levels + 1)
        choices = np.concatenate([-levels[::-1], levels])
        rng = np.random.default_rng(self.seed)
        self.rx = rng.choice(choices, size=k)
        self.ry = rng.choice(choices, size=k)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xn = X.detach().cpu().numpy()
        vals = [
            _spoqc_soft_row(row, m=self.m, n_q=self.k, n_features=self.n_features,
                            observable=self.observable, seed=self.seed, rx=self.rx, ry=self.ry,
                            cx_pairs=self.cx_pairs)
            for row in Xn
        ]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

    def render(self, path: str, x=None) -> str:
        """Render the spin-photon circuit (structure is identical for every x)."""
        import matplotlib.pyplot as plt
        from perceval_spoqc import Format, pdisplay

        xv = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        p = _build_processor(xv, m=self.m, n_q=self.k, n_features=self.n_features,
                             seed=self.seed, rx=self.rx, ry=self.ry, cx_pairs=self.cx_pairs)
        pdisplay(p, output_format=Format.MPLOT, expanded=True)
        plt.savefig(path)
        plt.close("all")
        return path

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "SpoqcLowPhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed,
                   cx_pairs=cfg.problem.cx_pairs, angle_levels=cfg.problem.angle_levels)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        # spoqc-only knobs -> only this teacher's hash sees observable / cx_pairs / angle_levels.
        # cx_pairs is added ONLY when non-empty, so a no-entangler config keeps the
        # same hash as before that knob existed (existing datasets stay valid).
        levels = cls.ANGLE_LEVELS if cfg.problem.angle_levels is None else int(cfg.problem.angle_levels)
        spec = {"observable": cfg.problem.observable, "prep": f"H_Rx_Ry_low_L{levels}"}
        pairs = _normalize_cx_pairs(cfg.problem.cx_pairs, cfg.problem.k)
        if pairs:
            spec["cx_pairs"] = [list(p) for p in pairs]
        return spec


def render_circuit(path="spoqc.png", *, m=6, k=3, n_features=5, seed=42, cx_pairs=None, x=None):
    """Render the spoqc circuit once to ``path`` (structure is the same for all x)."""
    out = SpoqcLowPhotonicTeacher(m=m, k=k, n_features=n_features, seed=seed,
                               cx_pairs=cx_pairs).render(path, x=x)
    print(f"[spoqc] saved circuit -> {out}")
    return out