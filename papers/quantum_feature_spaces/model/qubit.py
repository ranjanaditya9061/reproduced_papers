"""Qubit IQP teacher (Havlicek et al. 2019), pure-torch statevector.

Feature map (a random *sandwich*, mirroring the photonic ``W1 -> P(x) -> W2``)::

    |phi(x)> = W(theta_trail) . U_phi(x) H^n U_phi(x) H^n . V(theta_lead) |0^n>

- ``V(theta_lead)``  -- leading seeded variational unitary (``W1`` analogue).
- ``U_phi(x)``       -- the fixed IQP data encoding.
- ``W(theta_trail)`` -- trailing seeded variational unitary (``W2`` analogue).

Why the lead matters: in the fidelity kernel ``|<phi(x)|phi(x')>|^2`` the *trailing*
unitary cancels (common unitary on both states) but the *leading* one does not --
exactly as ``W2`` cancels and ``W1`` survives in the photonic case.  So both
platforms get a seed-dependent fidelity kernel from their leading unitary.  Set
``lead=False`` to drop ``V`` and recover the plain fixed-IQP (Huang) kernel.

The teacher is ``feature map + parity readout``; the embedding is shared with the
qubit kernels (:mod:`kernel.qubit`).  Both unitaries' weights live in the
``state_dict`` (saved as ``teacher.pt``), so a matched-seed kernel reuses them.

``n_qubits = n_features``; ``k`` (depth) is the variational depth of *each* block.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import Teacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig


# --- structural tensors ---------------------------------------------------- #

def _hadamard_n(n: int) -> torch.Tensor:
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    H = H1
    for _ in range(n - 1):
        H = torch.kron(H, H1)
    return H


def _basis_signs(n: int) -> torch.Tensor:
    """``(2^n, n)`` of +-1: the Z_i eigenvalue on each computational basis state."""
    z = torch.arange(2 ** n)
    bits = (z.unsqueeze(-1) >> torch.arange(n)) & 1
    return (1 - 2 * bits).float()


def _popcount(t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(t)
    while t.any():
        out += t & 1
        t = t >> 1
    return out


# --- gates ----------------------------------------------------------------- #

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
    """Maps ``X (N, n)`` -> embedding states ``(N, 2^n)`` complex.

    ``|phi(x)> = W_trail . IQP(x) . V_lead . |0^n>``.  Both unitaries' weights are
    seeded buffers drawn from a single ``seed`` (lead then trail, fixed order), so
    the embedding is fully determined by ``(n_qubits, depth, seed, lead)`` and
    round-trips through ``state_dict``.  ``lead=False`` skips ``V_lead`` (but draws
    its weights anyway, so ``theta_trail`` is identical with or without the lead).
    """

    def __init__(self, n_qubits: int, depth: int, seed: int, lead: bool = True):
        super().__init__()
        n = n_qubits
        self.n_qubits = n
        self.depth = int(depth)
        self.seed = int(seed)
        self.lead = bool(lead)
        self.register_buffer("H", _hadamard_n(n).to(torch.complex64))
        self.register_buffer("signs", _basis_signs(n))
        self.register_buffer("pair_mask", torch.triu(torch.ones(n, n), diagonal=1).bool())

        # One seed -> both blocks, fixed draw order (lead then trail).  theta_lead is
        # always drawn (so theta_trail matches regardless of `lead`); it is applied
        # only when lead=True.
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

    @torch.no_grad()
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


class QubitTeacher(Teacher):
    """IQP feature map (random sandwich) + parity readout; outputs ``(N, 1)``."""

    name = "qubit_quantum"

    def __init__(self, n_features: int, k: int, seed: int, nsample: int = 0):
        super().__init__(n_features)
        n = n_features
        self.n_qubits = n
        self.nsample = int(nsample)
        self._noise_seed = seed + 13
        self.feature_map = QubitFeatureMap(n, depth=k, seed=seed, lead=True)
        self.register_buffer("parity_signs",
                             (1 - 2 * (_popcount(torch.arange(2 ** n)) % 2)).float())

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """The teacher's embedding states ``(N, 2^n)`` (shared with the kernels)."""
        return self.feature_map(X)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        state = self.feature_map(X)
        parity = (state.conj() * state).real @ self.parity_signs   # (N,) in [-1, 1]
        if self.nsample > 0:
            std = ((1.0 - parity ** 2) / self.nsample).clamp(min=0.0).sqrt()
            gen = torch.Generator().manual_seed(self._noise_seed)
            parity = (parity + torch.randn(parity.shape, generator=gen) * std).clamp(-1.0, 1.0)
        return parity.unsqueeze(-1)

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "QubitTeacher":
        return cls(n_features=cfg.resolved_n_features, k=cfg.problem.k,
                   seed=cfg.seeds.teacher_seed, nsample=cfg.generation.nsample)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        # `embedding` marks the V_lead -> IQP -> W_trail structure so datasets made
        # by the older (IQP -> W only) teacher get a distinct hash, not a collision.
        return {"nsample": cfg.generation.nsample, "embedding": "Vlead-IQP-Wtrail"}