"""Random-teacher MLP over Fourier features (outputs softmax probabilities)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import Teacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


def fourier_features(X: torch.Tensor, order: int) -> torch.Tensor:
    """Expand angles ``(N, d)`` into ``[sin(jx), cos(jx)]_{j=1..order}`` -> ``(N, 2*order*d)``."""
    parts = []
    for j in range(1, order + 1):
        parts.append(torch.sin(j * X))
        parts.append(torch.cos(j * X))
    return torch.cat(parts, dim=1)


class MLPTeacher(Teacher):
    """A fixed random tanh MLP; ``soft`` is the ``(N, 2)`` softmax output.

    Zero-bias tanh on zero-mean Fourier features yields approximately balanced
    classes regardless of the random weights.  ``k`` = number of hidden layers.
    """

    name = "mlp"

    def __init__(self, n_features: int, k: int, seed: int,
                 fourier_order: int = 3, hidden_size: int | None = None):
        super().__init__(n_features)
        self.fourier_order = int(fourier_order)
        feat = 2 * fourier_order * n_features
        width = hidden_size if hidden_size is not None else max(2 * n_features, 8)
        n_layers = max(int(k), 1)

        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        in_size = feat
        for _ in range(n_layers):
            lin = nn.Linear(in_size, width, bias=False)
            nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain("tanh"))
            layers += [lin, nn.Tanh()]
            in_size = width
        out = nn.Linear(in_size, 2, bias=False)
        nn.init.xavier_uniform_(out.weight)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.net.eval()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.net(fourier_features(X, self.fourier_order))
        return torch.softmax(logits, dim=-1)  # (N, 2)

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "MLPTeacher":
        return cls(n_features=cfg.resolved_n_features, k=cfg.problem.k,
                   seed=cfg.seeds.teacher_seed)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        # default knobs, surfaced so changing them re-identifies the dataset
        return {"fourier_order": 3, "hidden_size": None}