"""Qubit IQP model (Havlicek et al. 2019), pure-torch statevector.

Feature map -- a random *sandwich*, mirroring the photonic ``W1 -> P(x) -> W2``::

    |phi(x)> = W(theta_trail) . U_phi(x) H^n U_phi(x) H^n . V(theta_lead) |0^n>

* ``V(theta_lead)``  -- leading seeded variational unitary (the ``W1`` analogue)
* ``U_phi(x)``       -- the fixed IQP data encoding
* ``W(theta_trail)`` -- trailing seeded variational unitary (the ``W2`` analogue)

Why the lead matters: in the fidelity kernel ``|<phi(x)|phi(x')>|^2`` the *trailing* unitary cancels
(a common unitary on both states) but the *leading* one does not -- exactly as ``W2`` cancels and
``W1`` survives in the photonic case.  So both platforms get a seed-dependent kernel from their
leading unitary.  ``lead=False`` drops ``V`` and recovers the plain fixed-IQP (Huang) kernel.

``n_qubits = n_features``, so the outcome basis is the ``2^n`` computational states -- a *different*
basis from the Fock models, which is why a cross-platform comparison goes through scores (and
through the input Fisher matrix, which is ``n_features x n_features`` on every model) rather than
through ``p`` entry-by-entry.

The gate primitives are pure torch and differentiable, so :mod:`v2.metrics` takes exact
input-Jacobians through this model.

Carried from ``model/qubit.py``; the ``forward`` now returns the distribution rather than a parity
score, with ``parity`` available as an ordinary registry observable over this basis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import DistributionModel

if TYPE_CHECKING:
    from config import ExperimentConfig


def _hadamard_n(n: int) -> torch.Tensor:
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    H = H1
    for _ in range(n - 1):
        H = torch.kron(H, H1)
    return H


def _basis_signs(n: int) -> torch.Tensor:
    """``(2^n, n)`` of ``+-1``: the ``Z_i`` eigenvalue on each computational basis state."""
    z = torch.arange(2 ** n)
    bits = (z.unsqueeze(-1) >> torch.arange(n)) & 1
    return (1 - 2 * bits).float()


def _ry(theta: torch.Tensor) -> torch.Tensor:
    h = theta / 2
    c, s = torch.cos(h).to(torch.complex64), torch.sin(h).to(torch.complex64)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _rz(theta: torch.Tensor) -> torch.Tensor:
    h = theta / 2
    em, ep = torch.exp(-1j * h.to(torch.complex64)), torch.exp(1j * h.to(torch.complex64))
    zero = torch.zeros_like(em)
    return torch.stack([torch.stack([em, zero]), torch.stack([zero, ep])])


def _apply_1q(state: torch.Tensor, gate: torch.Tensor, q: int, n: int) -> torch.Tensor:
    B = state.shape[0]
    high, low = 2 ** (n - q - 1), 2 ** q
    s = state.reshape(B, high, 2, low)
    return torch.einsum("oi,bhik->bhok", gate, s).reshape(B, -1)


def _apply_cz(state: torch.Tensor, i: int, j: int, n: int) -> torch.Tensor:
    z = torch.arange(2 ** n, device=state.device)
    both = ((z >> i) & 1) & ((z >> j) & 1)
    return state * (1 - 2 * both.to(torch.complex64))


def _variational(state: torch.Tensor, theta: torch.Tensor, n: int) -> torch.Tensor:
    """Alternating RY/RZ rotation layers with a fixed CZ-chain entangler between them."""
    theta = theta.reshape(-1, n, 2)
    n_layers = theta.shape[0]
    s = state
    for layer in range(n_layers):
        for q in range(n):
            s = _apply_1q(s, _ry(theta[layer, q, 0]), q, n)
            s = _apply_1q(s, _rz(theta[layer, q, 1]), q, n)
        if layer < n_layers - 1 and n >= 2:
            for q in range(n - 1):
                s = _apply_cz(s, q, q + 1, n)
    return s


class QubitFeatureMap(nn.Module):
    """``X (N, n) -> embedding states ``(N, 2^n)`` complex.

    Both unitaries' weights are seeded buffers drawn from one ``seed`` (lead then trail, fixed
    order), so the embedding is fully determined by ``(n_qubits, depth, seed, lead)`` and
    round-trips through ``state_dict``.  ``lead=False`` skips ``V_lead`` but still *draws* its
    weights, so ``theta_trail`` is identical with or without the lead.
    """

    def __init__(self, n_qubits: int, depth: int, seed: int, lead: bool = True):
        super().__init__()
        n = int(n_qubits)
        self.n_qubits = n
        self.depth = int(depth)
        self.seed = int(seed)
        self.lead = bool(lead)
        self.register_buffer("H", _hadamard_n(n).to(torch.complex64))
        self.register_buffer("signs", _basis_signs(n))
        self.register_buffer("pair_mask", torch.triu(torch.ones(n, n), diagonal=1).bool())

        torch.manual_seed(seed)
        size = (max(int(depth), 0) + 1) * n * 2
        self.register_buffer("theta_lead", 2 * math.pi * torch.rand(size))
        self.register_buffer("theta_trail", 2 * math.pi * torch.rand(size))

    def _iqp_diag(self, x: torch.Tensor) -> torch.Tensor:
        phi_single = x @ self.signs.T
        diff = math.pi - x
        diff_outer = (diff.unsqueeze(-1) * diff.unsqueeze(-2)) * self.pair_mask
        sign_outer = (self.signs.unsqueeze(-1) * self.signs.unsqueeze(-2)) * self.pair_mask
        phi_pair = (sign_outer.unsqueeze(0) * diff_outer.unsqueeze(1)).sum(dim=(-2, -1))
        return torch.exp(1j * (phi_single + phi_pair).to(torch.complex64))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n = self.n_qubits
        state = torch.zeros(X.shape[0], 2 ** n, dtype=torch.complex64, device=X.device)
        state[:, 0] = 1.0                                    # |0^n>
        if self.lead:
            state = _variational(state, self.theta_lead, n)  # V_lead (W1 analogue)
        state = state @ self.H.T                             # IQP: H^n
        state = state * self._iqp_diag(X)                    #      U_phi(x)
        state = state @ self.H.T                             #      H^n
        state = state * self._iqp_diag(X)                    #      U_phi(x)
        return _variational(state, self.theta_trail, n)      # W_trail (W2 analogue)


class QubitModel(DistributionModel):
    """``X -> probs``: the IQP sandwich's computational-basis distribution ``|amplitude|^2``.

    ``k`` is the variational depth of *each* block, and ``m`` is unused (the qubit count is
    ``n_features``) -- kept in the signature so the config shape is uniform across models.
    """

    name = "qubit"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 lead: bool = True):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        self.n_qubits = int(n_features)
        self.feature_map = QubitFeatureMap(self.n_qubits, depth=k, seed=seed, lead=lead)
        self._keys = [tuple((z >> i) & 1 for i in range(self.n_qubits))
                      for z in range(2 ** self.n_qubits)]
        self._autosize_batch(2 ** self.n_qubits)

    def amplitudes(self, X: torch.Tensor) -> torch.Tensor:
        """``(N, 2^n)`` complex embedding states (shared with the projected kernel)."""
        return self.feature_map(X)

    def _probs(self, X: torch.Tensor) -> torch.Tensor:
        a = self.feature_map(X)
        return (a.conj() * a).real

    def outcome_keys(self):
        """The ``2^n`` computational basis states as per-qubit bit tuples.

        Shaped like an occupation vector (one entry per "mode"), so the counting scorers apply
        unchanged: ``parity`` over the first ``ceil(n/2)`` qubits, and so on.
        """
        return self._keys

    def n_model_parameters(self) -> int:
        """``2 * (depth + 1) * n * 2`` -- the two variational blocks' angles."""
        return int(self.feature_map.theta_lead.numel() + self.feature_map.theta_trail.numel())

    def circuit_spec(self) -> dict:
        # `embedding` marks the V_lead -> IQP -> W_trail structure, so datasets made by an older
        # IQP-only map get a distinct identity rather than colliding.
        return {"model": self.name, "embedding": "Vlead-IQP-Wtrail",
                "depth": self.feature_map.depth, "lead": self.feature_map.lead,
                "basis": "computational_2^n"}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "QubitModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed)

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        if cfg.problem.n_features > 20:
            raise ValueError(f"qubit statevector is 2^n_features = 2^{cfg.problem.n_features} "
                             "wide; that will not fit")
