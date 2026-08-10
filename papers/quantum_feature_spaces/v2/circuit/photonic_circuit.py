"""The photonic sandwich ``W1(Haar) -> encode(x) -> W2(Haar)`` and its unitaries.

Carried from ``model/photonic_circuit.py`` and ``model/fermion.py``, with one change: the
encoding is a parameter (:mod:`v2.circuit.encoding`) instead of a hard-coded phase-shifter loop.

Two ways to get the same circuit, which must stay in lockstep:

* :func:`build_quantum_layer` -- a merlin ``QuantumLayer`` (genuine boson sampling; perceval and
  merlin imported lazily, only on construction);
* :func:`sandwich_unitaries` + :func:`sandwich_unitary_at` -- the same ``W1, W2`` as plain numpy
  and the resulting ``U(x)`` as a torch tensor, for the analytic ``det`` / ``Perm`` readouts.

The draw order (``torch.manual_seed`` then ``pcvl.random_seed``, then two sequential
``random_unitary(m)`` calls) is reproduced *exactly* by both, so a matched seed gives the same
circuit down to the entries.  That is what makes the boson-vs-determinant comparison controlled:
:mod:`v2.model.fermion` swaps ``Perm`` for ``det`` and holds literally everything else fixed.
"""

from __future__ import annotations

import numpy as np
import torch

from .encoding import Encoding, build_encoding


def default_input_state(m: int, k: int) -> list[int]:
    """Inject ``k`` photons evenly spaced across ``m`` modes (no light-cone gaps)."""
    state = [0] * m
    for i in range(k):
        state[round(i * m / k)] = 1
    return state


def build_sandwich_circuit(m: int, n_features: int, seed: int, encoding: Encoding | str = "phase",
                           *, x=None):
    """``W1(Haar) -> encoding -> W2(Haar)`` as a perceval circuit.

    ``x=None`` adds the encoding parameterised (free parameters ``x0..x{n-1}``, which a merlin
    layer binds); passing ``x`` adds concrete numeric values instead, which the per-row perceval
    paths need.  ``W1`` and ``W2`` are drawn sequentially from one seed, so they are reproducible
    and distinct.
    """
    import perceval as pcvl

    enc = build_encoding(encoding) if isinstance(encoding, str) else encoding
    torch.manual_seed(seed)
    pcvl.random_seed(seed)
    circuit = pcvl.Circuit(m, name="haar_encode_haar")
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)      # W1
    if x is None:
        enc.add_to(circuit, m=m, n_features=n_features)
    else:
        enc.add_concrete(circuit, x, m=m, n_features=n_features)
    circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)      # W2
    return circuit


def build_quantum_layer(m: int, k: int, n_features: int, seed: int, *, measure: str = "probs",
                        encoding: Encoding | str = "phase", input_state=None):
    """``(layer, input_state)``: a merlin ``QuantumLayer`` over the sandwich circuit.

    ``measure`` selects the Fock-space measurement strategy -- ``"probs"`` for the full outcome
    distribution (what every model wants) or ``"amplitudes"`` for the complex amplitudes (the
    projected-kernel feature map).  ``input_state`` defaults to :func:`default_input_state`;
    the spin preps override it.
    """
    import merlin as ML
    import perceval as pcvl

    state = default_input_state(m, k) if input_state is None else list(input_state)
    layer = ML.QuantumLayer(
        input_size=n_features,
        experiment=pcvl.Experiment(build_sandwich_circuit(m, n_features, seed, encoding)),
        input_state=state,
        input_parameters=["x"],
        measurement_strategy=getattr(ML.MeasurementStrategy, measure)(ML.ComputationSpace.FOCK),
    )
    return layer, state


def sandwich_unitaries(m: int, seed: int):
    """``(W1, W2)`` as numpy arrays, matching :func:`build_sandwich_circuit`'s draw exactly.

    Same two seeding calls in the same order, then two sequential ``random_unitary(m)`` draws --
    so a matched seed reproduces the circuit merlin builds.  Verified downstream by reproducing
    the merlin layer's probabilities from ``|Perm|^2`` to ~1e-7
    (:func:`v2.model.fermion.boson_probs_reference`).
    """
    import perceval as pcvl

    torch.manual_seed(int(seed))
    pcvl.random_seed(int(seed))
    W1 = np.array(pcvl.Matrix.random_unitary(m), dtype=np.complex128)
    W2 = np.array(pcvl.Matrix.random_unitary(m), dtype=np.complex128)
    return W1, W2


def sandwich_unitary_at(W1, W2, X: torch.Tensor, n_features: int,
                        encoding: Encoding | str = "phase") -> torch.Tensor:
    """``(N, m, m)`` complex ``U(x) = (W2 D(x) W1)^T`` for a batch of inputs.

    ``D(x)`` is the encoding's contribution -- a diagonal for encodings like ``phase``, or a full
    dense unitary for encodings that mix modes (``bs``, ``bs_phase``).  Transposing folds in as
    ``U = W1^T D W2^T``; the transpose is perceval/merlin's convention.

    Tries :meth:`Encoding.phases` first (the cheap ``O(m)``-per-row diagonal path) and falls back
    to :meth:`Encoding.unitary` (the general ``O(m^2)``-per-row path) when the encoding does not
    implement the former -- every shipped encoding implements exactly one, so this dispatch always
    resolves to the right cost for that encoding rather than needing a flag.

    **Differentiable in ``X``** for every encoding -- ``torch.exp(1j * x)`` / ``torch.cos``,
    ``torch.sin`` inside the encoding and plain ``einsum`` here -- which is what lets
    :mod:`v2.metrics` take exact input-Jacobians without any parametric re-implementation of the
    circuit.
    """
    enc = build_encoding(encoding) if isinstance(encoding, str) else encoding
    m = W1.shape[0]
    A = torch.as_tensor(np.ascontiguousarray(W1.T), dtype=torch.complex64)
    B = torch.as_tensor(np.ascontiguousarray(W2.T), dtype=torch.complex64)
    try:
        ph = enc.phases(X, m=m, n_features=n_features)
    except NotImplementedError:
        # General case: U(x) = (W2 D(x) W1)^T = W1^T D(x)^T W2^T.  D is diagonal in the phases()
        # branch above, where D^T = D collapses this to the einsum below; a mixing encoding's D is
        # not symmetric in general, so the transpose is not optional here.
        D = enc.unitary(X, m=m, n_features=n_features)                # (N, m, m)
        return torch.einsum("il,nlp,pj->nij", A, D.transpose(-1, -2), B)
    return torch.einsum("il,nl,lj->nij", A, ph, B)


def n_circuit_parameters(m: int) -> int:
    """``2m^2 - 1``: the sandwich's effective real parameter count.
    """
    return 2 * m * m - 1
