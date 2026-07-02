"""Spin-photon teacher with prime-angle Rz gates (perceval_spoqc).

Same as :class:`model.spoqc_low.SpoqcLowPhotonicTeacher` (H -> discrete +-0.1 Rx/Ry
-> dual-rail emission -> W1.PS(x).W2 embedding), but each qubit ALSO gets an
``Rz`` at an increasing-prime angle: qubit q -> ``Rz(prime[q])`` with
``prime = 2, 3, 5, 7, 11, ...`` (radians).  The Rz is applied *between* Rx and Ry
(see :func:`model.spoqc_utils._spin_state`) so the following Ry rotates its phase
into the rail populations -- otherwise a trailing Rz on a product state is washed
out when the (unmeasured) spin is traced.

``cx_pairs`` and ``angle_levels`` are the same spoqc-only knobs as spoqc_low; the
prime Rz angles are fixed/deterministic (not seeded), so they don't add a knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import Teacher
from .spoqc_utils import OBSERVABLES, _build_processor, _spoqc_soft_row, apply_cx  # noqa: F401

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


def _first_primes(n: int) -> list[int]:
    """First ``n`` primes: 2, 3, 5, 7, 11, ..."""
    primes, c = [], 2
    while len(primes) < n:
        if all(c % p for p in primes):
            primes.append(c)
        c += 1
    return primes


def _normalize_cx_pairs(cx_pairs, k: int):
    pairs = []
    for pair in (cx_pairs or []):
        c, t = int(pair[0]), int(pair[1])
        if c == t or not (0 <= c < k and 0 <= t < k):
            raise ValueError(f"cx pair {pair} invalid for k={k} qubits (need distinct 0<=i<k)")
        pairs.append((c, t))
    return pairs


class SpoqcPrimePhotonicTeacher(Teacher):
    """spoqc_low + a per-qubit ``Rz(prime[q])`` gate; ``soft`` in ``[-1, 1]``."""

    name = "spoqc_prime_photonic"

    #: rx/ry drawn from {-0.1*L, ..., -0.1, 0.1, ..., 0.1*L} (as in spoqc_low)
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

        levels = 0.1 * np.arange(1, self.angle_levels + 1)
        choices = np.concatenate([-levels[::-1], levels])
        rng = np.random.default_rng(self.seed)
        self.rx = rng.choice(choices, size=k)
        self.ry = rng.choice(choices, size=k)
        self.rz = np.array(_first_primes(k), dtype=float)   # Rz(prime[q]) per qubit

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Xn = X.detach().cpu().numpy()
        vals = [
            _spoqc_soft_row(row, m=self.m, n_q=self.k, n_features=self.n_features,
                            observable=self.observable, seed=self.seed, rx=self.rx, ry=self.ry,
                            cx_pairs=self.cx_pairs, rz=self.rz)
            for row in Xn
        ]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

    def render(self, path: str, x=None) -> str:
        import matplotlib.pyplot as plt
        from perceval_spoqc import Format, pdisplay

        xv = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        p = _build_processor(xv, m=self.m, n_q=self.k, n_features=self.n_features,
                             seed=self.seed, rx=self.rx, ry=self.ry, cx_pairs=self.cx_pairs,
                             rz=self.rz)
        pdisplay(p, output_format=Format.MPLOT, expanded=True)
        plt.savefig(path)
        plt.close("all")
        return path

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "SpoqcPrimePhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed,
                   cx_pairs=cfg.problem.cx_pairs, angle_levels=cfg.problem.angle_levels)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        levels = cls.ANGLE_LEVELS if cfg.problem.angle_levels is None else int(cfg.problem.angle_levels)
        spec = {"observable": cfg.problem.observable, "prep": f"H_Rx_Rz_prime_Ry_low_L{levels}"}
        pairs = _normalize_cx_pairs(cfg.problem.cx_pairs, cfg.problem.k)
        if pairs:
            spec["cx_pairs"] = [list(p) for p in pairs]
        return spec
