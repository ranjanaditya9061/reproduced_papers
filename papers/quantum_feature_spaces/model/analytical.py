"""Analytical pairwise-phase interference teacher (deterministic, no params)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import Teacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


class AnalyticalTeacher(Teacher):
    """``soft(x) = (1/n_pairs) Σ_{i<j} sin(k (x_i − x_j))`` in ``[-1, +1]``.

    For ``n_features == 1`` this reduces to ``sin(k x_0)``.  ``k`` sets the
    spatial frequency of the (naturally balanced) decision boundary.
    """

    name = "analytical"

    def __init__(self, n_features: int, k: int):
        super().__init__(n_features)
        self.k = int(k)
        if n_features == 1:
            self.register_buffer("rows", torch.empty(0, dtype=torch.long))
            self.register_buffer("cols", torch.empty(0, dtype=torch.long))
            self.n_pairs = 1
        else:
            rows, cols = torch.triu_indices(n_features, n_features, offset=1)
            self.register_buffer("rows", rows)
            self.register_buffer("cols", cols)
            self.n_pairs = int(rows.shape[0])

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.n_features == 1:
            scores = torch.sin(self.k * X[:, 0])
        else:
            scores = torch.sin(self.k * (X[:, self.rows] - X[:, self.cols])).sum(dim=-1)
        return (scores / self.n_pairs).unsqueeze(-1)  # (N, 1)

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "AnalyticalTeacher":
        return cls(n_features=cfg.resolved_n_features, k=cfg.problem.k)