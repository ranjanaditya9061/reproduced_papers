"""Helpers for the spoqc spin-photon teacher (:mod:`model.spoqc`).

Builds the spin-prepared photonic ``HybridProcessor``: per qubit ``H -> Rx -> Ry``
then an optional ``CX`` entangler chain -- all applied in numpy on the initial
source state (spoqc has no two-qubit processor gate, and a ``CX`` on ``|0...0>``
is the identity, so the single-qubit prep must precede it) -- followed by dual-rail
emission, the ``W1.PS(x).W2`` embedding, and the observable scorers.
"""

from __future__ import annotations

import numpy as np
import torch

from .photonic import _bunching_score, _majority_score, _parity_score

OBSERVABLES = ("parity", "majority", "bunching")


def apply_cx(state, control: int, target: int, n_qubits: int | None = None):
    """Apply ``CNOT(control -> target)`` to a spin state and return the new state.

    ``state`` may be a state vector (1-D) or a density matrix (2-D); the return has
    the same shape.  Qubit 0 is the most-significant bit (leftmost tensor factor),
    matching the joint ordering of ``with_initial_source_state``.
    """
    state = np.asarray(state, dtype=complex)
    dim = state.shape[0]
    if n_qubits is None:
        n_qubits = int(round(np.log2(dim)))
    if control == target:
        raise ValueError("control and target must be different qubits")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError(f"control/target out of range for {n_qubits} qubits")

    cbit = 1 << (n_qubits - 1 - control)
    tbit = 1 << (n_qubits - 1 - target)
    perm = np.arange(dim)
    flip = (perm & cbit) != 0
    perm[flip] ^= tbit
    U = np.zeros((dim, dim), dtype=complex)
    U[perm, np.arange(dim)] = 1.0

    if state.ndim == 1:
        return U @ state
    return U @ state @ U.conj().T


def _sandwich_concrete(m: int, n_features: int, seed: int, x):
    """``W1(Haar) -> PS(x_i) -> W2(Haar)`` with concrete phase values (per sample)."""
    import perceval as pcvl

    pcvl.random_seed(seed)
    torch.manual_seed(seed)
    c = pcvl.Circuit(m, name="haar_phase_haar")
    c.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W1
    for i in range(n_features):
        c.add(i, pcvl.PS(float(x[i])))
    c.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)   # W2
    return c


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_1q(psi, q, g, n):
    """Apply 2x2 gate ``g`` to qubit ``q`` of an ``n``-qubit state vector (qubit 0 = MSB)."""
    U = np.array([[1.0 + 0j]])
    for i in range(n):
        U = np.kron(U, g if i == q else np.eye(2, dtype=complex))
    return U @ psi


def _spin_state(n_q, rx, ry, cx_pairs):
    """Joint source density matrix: H + seeded Rx/Ry per qubit, then the CX chain.

    The single-qubit prep happens *before* the CX (in numpy), since a CX on
    ``|0...0>`` is the identity and spoqc has no two-qubit processor gate.
    """
    psi = np.zeros(2 ** n_q, dtype=complex)
    psi[0] = 1.0                                     # |0...0>
    for q in range(n_q):
        psi = _apply_1q(psi, q, _H, n_q)             # |0> -> |+>
        psi = _apply_1q(psi, q, _rx(float(rx[q])), n_q)   # seeded twists -> Gamma != 1/2 I
        psi = _apply_1q(psi, q, _ry(float(ry[q])), n_q)
    for c, t in cx_pairs:
        psi = apply_cx(psi, c, t, n_q)               # entangler(s) on a real superposition
    return np.outer(psi, psi.conj())


def _build_processor(x, *, m, n_q, n_features, seed, rx, ry, cx_pairs=()):
    """Spin-prepared photonic HybridProcessor for one input ``x`` (spin prep in numpy)."""
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    p = HybridProcessor(num_sources=n_q, num_modes=m)
    p.with_initial_source_state(_spin_state(n_q, rx, ry, cx_pairs))
    for q in range(n_q):
        p.emit(q, into=(2 * q, 2 * q + 1))           # dual-rail emission into its pair

    p.add(0, _sandwich_concrete(m, n_features, seed, x))   # same embedding as the photonic teacher
    for mode in range(m):
        p.add(mode, Detector())
    return p


def _score_processor(p, *, m, n_q, observable) -> float:
    """Reduce a built processor's detection distribution to the observable score."""
    parity_modes = tuple(range((m + 1) // 2))
    s = 0.0
    for key, pr in p.probabilities().items():
        if observable == "parity":
            s += pr * _parity_score(key, parity_modes)
        elif observable == "majority":
            s += pr * _majority_score(key, m, n_q)     # normalise by photon count n_q
        else:  # bunching
            s += pr * _bunching_score(key)
    return float(s)


def _spoqc_soft_row(x, *, m, n_q, n_features, observable, seed, rx, ry, cx_pairs=()) -> float:
    """Continuous score for one input ``x`` (``cx_pairs`` selects the spin entangler)."""
    p = _build_processor(x, m=m, n_q=n_q, n_features=n_features, seed=seed, rx=rx, ry=ry,
                         cx_pairs=cx_pairs)
    return _score_processor(p, m=m, n_q=n_q, observable=observable)