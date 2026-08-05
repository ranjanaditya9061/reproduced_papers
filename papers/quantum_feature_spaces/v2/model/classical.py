"""The two non-Fock models, on a 2-outcome basis: a random MLP and a closed-form map.

Neither has a Fock structure, and in the legacy pipeline they were the reason the pipeline had two
shapes: ``MLPTeacher.forward`` returned ``(N, 2)`` softmax probabilities and
``AnalyticalTeacher.forward`` returned an ``(N, 1)`` signed score, so downstream code had to know
which it was holding and the observable registry did not apply to either.

Here they emit a distribution over a **2-element outcome basis**
(:func:`v2.model.fock.binary_keys`), which makes them satisfy the same
:class:`~v2.model.base.DistributionModel` contract as the boson sampler.  The pay-off is exact
rather than approximate: with ``keys = [(1,0), (0,1)]``, ``parity`` over the first
``ceil(2/2) = 1`` mode gives ``v = (-1, +1)``, so ``probs @ v = p_1 - p_0``.  Setting
``p_1 = (1 + s)/2`` therefore recovers ``s`` **exactly**, and no model needs a special-cased scalar
path.  (Asserted in the tests.)

Carried from ``model/mlp.py`` and ``model/analytical.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import DistributionModel
from .features import TEACHER_FOURIER_VERSION, fourier_dim, fourier_features
from ..circuit.fock import binary_keys

if TYPE_CHECKING:
    from config import ExperimentConfig

#: Fourier order of the input angles for :class:`MlpModel`.
MLP_FOURIER_ORDER = 3


class MlpModel(DistributionModel):
    """A fixed random tanh MLP; ``probs`` is the ``(N, 2)`` softmax output.

    Zero-bias tanh on zero-mean Fourier features yields approximately balanced classes regardless
    of the random weights.  ``k`` = number of hidden layers; ``m`` is unused.
    """

    name = "mlp"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 fourier_order: int = MLP_FOURIER_ORDER, hidden_size: int | None = None):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        self.fourier_order = int(fourier_order)
        feat = fourier_dim(self.fourier_order, n_features)
        width = hidden_size if hidden_size is not None else max(2 * n_features, 8)
        self.hidden_size = int(width)
        self.n_layers = max(int(k), 1)

        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        in_size = feat
        for _ in range(self.n_layers):
            lin = nn.Linear(in_size, self.hidden_size, bias=False)
            nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain("tanh"))
            layers += [lin, nn.Tanh()]
            in_size = self.hidden_size
        out = nn.Linear(in_size, 2, bias=False)
        nn.init.xavier_uniform_(out.weight)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.net.eval()
        self._keys = binary_keys(2)
        self._autosize_batch(2)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.net(fourier_features(X, self.fourier_order))
        return torch.softmax(logits, dim=-1)                      # (N, 2)

    def outcome_keys(self):
        return self._keys

    def n_model_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def circuit_spec(self) -> dict:
        return {"model": self.name, "fourier_order": self.fourier_order,
                "encoding": f"teacher_fourier_v{TEACHER_FOURIER_VERSION}",
                "hidden_size": self.hidden_size, "n_layers": self.n_layers}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "MlpModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed)


class AnalyticalModel(DistributionModel):
    """``s(x) = (1/n_pairs) sum_{i<j} sin(k (x_i - x_j))`` in ``[-1, 1]``, as a 2-outcome distribution.

    Deterministic and parameter-free -- ``k`` sets the spatial frequency of the (naturally balanced)
    decision boundary.  For ``n_features == 1`` this reduces to ``sin(k x_0)``.

    ``probs = [(1 - s)/2, (1 + s)/2]``, so ``parity`` on this basis returns ``s`` exactly (see the
    module docstring).  A genuine distribution: both entries are in ``[0, 1]`` and sum to 1 because
    ``|s| <= 1``.
    """

    name = "analytical"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        self.freq = int(k)
        if n_features == 1:
            self.register_buffer("rows", torch.empty(0, dtype=torch.long))
            self.register_buffer("cols", torch.empty(0, dtype=torch.long))
            self.n_pairs = 1
        else:
            rows, cols = torch.triu_indices(n_features, n_features, offset=1)
            self.register_buffer("rows", rows)
            self.register_buffer("cols", cols)
            self.n_pairs = int(rows.shape[0])
        self._keys = binary_keys(2)
        self._autosize_batch(2)

    def signed_score(self, X: torch.Tensor) -> torch.Tensor:
        """``(N,)`` signed score ``s(x)`` in ``[-1, 1]`` -- the quantity ``parity`` recovers."""
        if self.n_features == 1:
            scores = torch.sin(self.freq * X[:, 0])
        else:
            scores = torch.sin(self.freq * (X[:, self.rows] - X[:, self.cols])).sum(dim=-1)
        return scores / self.n_pairs

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        s = self.signed_score(X)
        return torch.stack([(1.0 - s) / 2.0, (1.0 + s) / 2.0], dim=1)

    def outcome_keys(self):
        return self._keys

    def n_model_parameters(self) -> int:
        """Zero -- the map is closed-form and carries no weights."""
        return 0

    def circuit_spec(self) -> dict:
        return {"model": self.name, "freq": self.freq, "form": "pairwise_sin"}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "AnalyticalModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed)
