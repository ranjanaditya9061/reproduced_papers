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

``n_qubits`` is set by ``m``, decoupled from ``n_features`` -- the qubit analogue of photonic's
``m`` (modes) vs ``n_features`` (encoded phases): ``U_phi(x)`` acts on the first ``n_features`` of
the ``m`` qubits, and the remaining ``m - n_features`` are held at the encoding's fixed point
(``x_i = pi``, see :meth:`QubitFeatureMap._iqp_diag`) so they carry no ``x``-dependent phase at
that layer -- mirroring ``circuit/encoding.py``'s ``PhaseEncoding``, which puts ``x_i`` on mode
``i`` for ``i < n_features`` and leaves modes ``n_features..m-1`` at identity phase.  Needs
``n_features <= m``.  The outcome basis is the ``2^m`` computational states -- a *different* basis
from the Fock models, which is why a cross-platform comparison goes through scores (and through
the input Fisher matrix, which is ``n_features x n_features`` on every model) rather than through
``p`` entry-by-entry.

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
    """``X (N, n_features) -> embedding states ``(N, 2^n_qubits)`` complex.

    Both unitaries' weights are seeded buffers drawn from one ``seed`` (lead then trail, fixed
    order), so the embedding is fully determined by ``(n_qubits, n_features, depth, seed, lead)``
    and round-trips through ``state_dict``.  ``lead=False`` skips ``V_lead`` but still *draws* its
    weights, so ``theta_trail`` is identical with or without the lead.

    ``n_features <= n_qubits``: :meth:`_iqp_diag` encodes ``x`` on the first ``n_features`` qubits
    and pads the rest to the encoding's fixed point (see there), so the unencoded qubits carry no
    ``x``-dependent phase at that layer -- though ``V_lead``/``W_trail`` still entangle them with
    the encoded qubits, which is the point of raising ``n_qubits`` at fixed ``n_features``.

    ``feature_map="iqp"`` (default) is the full Havlicek sandwich, two ``H^n -> U_phi(x)`` passes.
    ``feature_map="phase"`` is the qubit analogue of the photonic ``phase`` encoding: a single
    diagonal ``exp(i sum x_i Z_i)`` (linear term only, no pairwise ``Z_i Z_j``, no Hadamard of its
    own) sandwiched directly between ``V_lead`` and ``W_trail`` -- mirroring how the photonic
    ``phase`` encoding is one ``PS(x)`` between two Haar unitaries, with no beamsplitter mixer of
    its own beyond ``W_1``/``W_2``.
    """

    def __init__(self, n_qubits: int, n_features: int, depth: int, seed: int, lead: bool = True,
                 feature_map: str = "iqp"):
        super().__init__()
        n = int(n_qubits)
        if int(n_features) > n:
            raise ValueError(
                f"qubit feature map encodes one qubit per feature, so it needs "
                f"n_features <= n_qubits (got n_features={n_features}, n_qubits={n}). Raise "
                f"n_qubits (m), or lower n_features."
            )
        if feature_map not in ("iqp", "phase"):
            raise ValueError(f"feature_map must be 'iqp' or 'phase' (got {feature_map!r})")
        self.n_qubits = n
        self.n_features = int(n_features)
        self.depth = int(depth)
        self.seed = int(seed)
        self.lead = bool(lead)
        self.feature_map_kind = feature_map
        self.register_buffer("H", _hadamard_n(n).to(torch.complex64))
        self.register_buffer("signs", _basis_signs(n))
        self.register_buffer("pair_mask", torch.triu(torch.ones(n, n), diagonal=1).bool())

        torch.manual_seed(seed)
        size = (max(int(depth), 0) + 1) * n * 2
        self.register_buffer("theta_lead", 2 * math.pi * torch.rand(size))
        self.register_buffer("theta_trail", 2 * math.pi * torch.rand(size))

    def _iqp_diag(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` arrives at ``n_features`` width; padded here to ``n_qubits`` at the fixed point
        ``x_i = pi``, which is where ``diff = pi - x`` vanishes -- the unique padding value that
        makes the unencoded qubits' diagonal phase constant across every one of their bit patterns
        (an entrywise-``0`` pad does not: ``diff_i = pi`` there is nonzero, so a padded qubit's
        fixed ``Z_i`` sign would still shape a real, outcome-dependent phase).
        """
        if self.n_features < self.n_qubits:
            pad = x.new_full((x.shape[0], self.n_qubits - self.n_features), math.pi)
            x = torch.cat([x, pad], dim=1)
        phi_single = x @ self.signs.T
        diff = math.pi - x
        diff_outer = (diff.unsqueeze(-1) * diff.unsqueeze(-2)) * self.pair_mask
        sign_outer = (self.signs.unsqueeze(-1) * self.signs.unsqueeze(-2)) * self.pair_mask
        phi_pair = (sign_outer.unsqueeze(0) * diff_outer.unsqueeze(1)).sum(dim=(-2, -1))
        return torch.exp(1j * (phi_single + phi_pair).to(torch.complex64))

    def _phase_diag(self, x: torch.Tensor) -> torch.Tensor:
        """``exp(i sum_i x_i Z_i)`` -- the linear-only term of :meth:`_iqp_diag`, no pairwise
        ``Z_i Z_j``.  Unencoded qubits (``n_features < n_qubits``) are padded with ``x_i = 0``
        rather than ``pi``: with no pairwise term to protect, a plain zero phase already leaves
        those qubits' diagonal entry at ``1`` regardless of their bit pattern.
        """
        if self.n_features < self.n_qubits:
            pad = x.new_zeros((x.shape[0], self.n_qubits - self.n_features))
            x = torch.cat([x, pad], dim=1)
        phi_single = x @ self.signs.T
        return torch.exp(1j * phi_single.to(torch.complex64))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n = self.n_qubits
        state = torch.zeros(X.shape[0], 2 ** n, dtype=torch.complex64, device=X.device)
        state[:, 0] = 1.0                                    # |0^n>
        if self.lead:
            state = _variational(state, self.theta_lead, n)  # V_lead (W1 analogue)
        if self.feature_map_kind == "phase":
            state = state * self._phase_diag(X)               # single U(x), no Hadamard mixer
        else:
            state = state @ self.H.T                          # IQP: H^n
            state = state * self._iqp_diag(X)                 #      U_phi(x)
            state = state @ self.H.T                           #      H^n
            state = state * self._iqp_diag(X)                  #      U_phi(x)
        return _variational(state, self.theta_trail, n)      # W_trail (W2 analogue)


class QubitModel(DistributionModel):
    """``X -> probs``: the IQP sandwich's computational-basis distribution ``|amplitude|^2``.

    ``k`` is the variational depth of *each* block; ``m`` is the qubit count, decoupled from
    ``n_features`` (see the module docstring) -- ``m == n_features`` reproduces the legacy
    behaviour exactly, and ``m > n_features`` grows the circuit at fixed input dimension.
    """

    name = "qubit"

    def __init__(self, *, m: int, k: int, n_features: int, seed: int = 42,
                 lead: bool = True, feature_map: str = "iqp"):
        super().__init__(m=m, k=k, n_features=n_features, seed=seed)
        self.n_qubits = int(m)
        self.feature_map = QubitFeatureMap(self.n_qubits, n_features=self.n_features, depth=k,
                                           seed=seed, lead=lead, feature_map=feature_map)
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
        # `embedding` marks the V_lead -> {IQP,phase} -> W_trail structure, so datasets made by an
        # older IQP-only map (or the other feature_map) get a distinct identity rather than
        # colliding.  `n_qubits` is included so two datasets differing only in m (same n_features,
        # same seed) do not collide: m sets the statevector width and is otherwise absent from
        # hash_fields.
        embedding = ("Vlead-IQP-Wtrail" if self.feature_map.feature_map_kind == "iqp"
                    else "Vlead-Phase-Wtrail")
        return {"model": self.name, "embedding": embedding,
                "n_qubits": self.n_qubits, "depth": self.feature_map.depth,
                "lead": self.feature_map.lead, "basis": "computational_2^n"}

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "QubitModel":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.problem.n_features,
                   seed=cfg.seeds.model_seed,
                   feature_map=getattr(cfg.model, "feature_map", None) or "iqp")

    @classmethod
    def validate_config(cls, cfg: "ExperimentConfig") -> None:
        if cfg.problem.n_features > cfg.problem.m:
            raise ValueError(
                f"qubit feature map encodes one qubit per feature, so it needs "
                f"n_features <= m (got n_features={cfg.problem.n_features}, m={cfg.problem.m}). "
                f"Raise m, or lower n_features."
            )
        if cfg.problem.m > 20:
            raise ValueError(f"qubit statevector is 2^m = 2^{cfg.problem.m} wide; that will not "
                             "fit")
