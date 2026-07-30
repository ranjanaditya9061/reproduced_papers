"""The photonic sandwich circuit ``W1(Haar) -> phase-encode(x) -> W2(Haar)`` and its feature map.

The multiphoton interference is genuine boson sampling, so everything here wraps a Merlin
``QuantumLayer`` (perceval/merlin imported lazily, only on construction).

:func:`build_sandwich_circuit` is shared by :class:`model.photonic.PhotonicTeacher` and by
:class:`PhotonicFeatureMap` (used by the photonic kernels), so a matched-seed kernel rebuilds
the *identical* ``W1, W2`` (drawn sequentially from one seed, so ``W1 != W2``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def default_input_state(m: int, k: int) -> list[int]:
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


def build_quantum_layer(m: int, k: int, n_features: int, seed: int, *, measure: str):
    """``(layer, input_state)``: a Merlin ``QuantumLayer`` over the sandwich circuit.

    ``measure`` names the Fock-space measurement strategy -- ``"probs"`` (the teacher, which
    needs the full outcome distribution) or ``"amplitudes"`` (the feature map, whose fidelity
    kernel needs the complex amplitudes).
    """
    import merlin as ML
    import perceval as pcvl

    input_state = default_input_state(m, k)
    layer = ML.QuantumLayer(
        input_size=n_features,
        experiment=pcvl.Experiment(build_sandwich_circuit(m, n_features, seed)),
        input_state=input_state,
        input_parameters=["x"],
        measurement_strategy=getattr(ML.MeasurementStrategy, measure)(ML.ComputationSpace.FOCK),
    )
    return layer, input_state


class PhotonicFeatureMap(nn.Module):
    """``|psi(x)> = W2 P(x) W1 |in>`` embedding (the W1->P(x)->W2 sandwich).

    ``amplitudes(X)`` -> ``(N, n_fock)`` complex Fock amplitudes (for the fidelity
    kernel); ``probs(X) = |amplitudes|^2`` and ``occ`` (per-Fock-state photon
    counts) feed the projected kernel's occupation moments.
    """

    def __init__(self, m: int, k: int, n_features: int, seed: int):
        super().__init__()
        self.m, self.k, self.seed = m, k, int(seed)
        self.layer, self.input_state = build_quantum_layer(
            m, k, n_features, self.seed, measure="amplitudes")
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
