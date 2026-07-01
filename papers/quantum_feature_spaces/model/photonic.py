"""Photonic sandwich teacher: W1(Haar) -> phase-encode(x) -> W2(Haar) -> measure.

The multiphoton interference is genuine boson sampling, so this wraps a Merlin
``QuantumLayer`` (perceval/merlin imported lazily, only on construction).

The circuit builder :func:`build_sandwich_circuit` is shared by the teacher and by
:class:`PhotonicFeatureMap` (used by the photonic kernels), so a matched-seed
kernel rebuilds the *identical* ``W1, W2`` (drawn sequentially from one seed, so
``W1 != W2``).  ``PhotonicTeacher.forward`` returns a continuous ``(N, 1)`` score
chosen by ``observable``; ``PhotonicFeatureMap`` exposes the full Fock-state
amplitudes (for the fidelity kernel) and probabilities (for the projected kernel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .base import Teacher

if TYPE_CHECKING:
    from Generator.config import ExperimentConfig

OBSERVABLES = ("parity", "majority", "bunching", "single_output", "n_first")


def _default_input_state(m: int, k: int) -> list[int]:
    """Inject k photons evenly spaced across m modes (no light-cone gaps)."""
    state = [0] * m
    for i in range(k):
        state[round(i * m / k)] = 1
    return state


def build_sandwich_circuit(m: int, n_features: int, seed: int):
    """``W1(Haar) -> PS(x_i) on first n_features modes -> W2(Haar)``.

    ``W1`` and ``W2`` are drawn sequentially from one seed -> reproducible and
    distinct.  Shared by the teacher and the kernel feature map so equal seeds
    give byte-identical circuits.
    """
    import perceval as pcvl

    torch.manual_seed(seed)
    pcvl.random_seed(seed)
    circuit = pcvl.Circuit(m, name="haar_phase_haar")
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W1
    for i in range(n_features):
        circuit.add(i, pcvl.PS(pcvl.P(f"x{i}")))
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W2
    return circuit


def _parity_score(key, parity_modes) -> int:
    n = sum(int(key[i]) for i in parity_modes)
    return 1 if n % 2 == 0 else -1


def _majority_score(key, m: int, k: int) -> float:
    split = m // 2
    n_left = sum(int(key[i]) for i in range(split))
    n_right = sum(int(key[i]) for i in range(split, m))
    return (n_left - n_right) / k


def _bunching_score(key) -> int:
    return 1 if max(int(n) for n in key) <= 1 else -1


def _first_mode_score(key) -> int:
    """Photon count in the first mode; dotted with probs gives ``E[n_0]`` (in [0, k])."""
    return int(key[0])


def _single_output_score(key, input_state) -> int:
    kl = [int(key[i]) for i in range(len(input_state))]
    if kl == list(input_state):
        return 1
    if kl == list(reversed(input_state)):
        return -1
    return 0


class PhotonicFeatureMap(nn.Module):
    """``|psi(x)> = W2 P(x) W1 |in>`` embedding (the W1->P(x)->W2 sandwich).

    ``amplitudes(X)`` -> ``(N, n_fock)`` complex Fock amplitudes (for the fidelity
    kernel); ``probs(X) = |amplitudes|^2`` and ``occ`` (per-Fock-state photon
    counts) feed the projected kernel's occupation moments.
    """

    def __init__(self, m: int, k: int, n_features: int, seed: int):
        super().__init__()
        self.m, self.k, self.seed = m, k, int(seed)
        import merlin as ML
        import perceval as pcvl

        circuit = build_sandwich_circuit(m, n_features, seed)
        self.input_state = _default_input_state(m, k)
        self.layer = ML.QuantumLayer(
            input_size=n_features,
            experiment=pcvl.Experiment(circuit),
            input_state=self.input_state,
            input_parameters=["x"],
            measurement_strategy=ML.MeasurementStrategy.amplitudes(ML.ComputationSpace.FOCK),
        )
        keys = list(self.layer.output_keys)
        occ = torch.tensor([[int(key[i]) for i in range(m)] for key in keys],
                           dtype=torch.float32)
        self.register_buffer("occ", occ)          # (n_fock, m) photon counts

    @torch.no_grad()
    def amplitudes(self, X: torch.Tensor) -> torch.Tensor:
        return self.layer.forward(X)               # (N, n_fock) complex

    @torch.no_grad()
    def probs(self, X: torch.Tensor) -> torch.Tensor:
        a = self.amplitudes(X)
        return (a.conj() * a).real                 # (N, n_fock)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.amplitudes(X)


class PhotonicTeacher(Teacher):
    name = "photonic_quantum"

    def __init__(self, m: int, k: int, n_features: int,
                 observable: str = "parity", seed: int = 1234, nsample: int = 0):
        super().__init__(n_features)
        if observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}")
        if observable == "majority" and m % 2:
            raise ValueError("observable 'majority' requires even m")
        self.m, self.k, self.observable, self.nsample = m, k, observable, int(nsample)

        import merlin as ML
        import perceval as pcvl

        circuit = build_sandwich_circuit(m, n_features, seed)
        input_state = _default_input_state(m, k)
        self.input_state = input_state
        self.layer = ML.QuantumLayer(
            input_size=n_features,
            experiment=pcvl.Experiment(circuit),
            input_state=input_state,
            input_parameters=["x"],
            measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        )

        keys = list(self.layer.output_keys)
        if observable == "parity":
            pm = tuple(range((m + 1) // 2))
            vec = [_parity_score(key, pm) for key in keys]
        elif observable == "majority":
            vec = [_majority_score(key, m, k) for key in keys]
        elif observable == "bunching":
            vec = [_bunching_score(key) for key in keys]
        elif observable == "n_first":
            vec = [_first_mode_score(key) for key in keys]   # soft = E[n_0]
        else:  # single_output
            vec = [_single_output_score(key, input_state) for key in keys]
        self.register_buffer("score_vec", torch.tensor(vec, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.layer.forward(X, shots=self.nsample if self.nsample > 0 else None)
        score = probs @ self.score_vec
        if self.observable == "single_output":
            score = score / probs.max(dim=1).values.clamp(min=1e-10)
        return score.unsqueeze(-1)  # (N, 1)

    @classmethod
    def from_config(cls, cfg: "ExperimentConfig") -> "PhotonicTeacher":
        return cls(m=cfg.problem.m, k=cfg.problem.k, n_features=cfg.resolved_n_features,
                   observable=cfg.problem.observable, seed=cfg.seeds.teacher_seed,
                   nsample=cfg.generation.nsample)

    @classmethod
    def hash_spec(cls, cfg: "ExperimentConfig") -> dict:
        return {"observable": cfg.problem.observable, "nsample": cfg.generation.nsample}
