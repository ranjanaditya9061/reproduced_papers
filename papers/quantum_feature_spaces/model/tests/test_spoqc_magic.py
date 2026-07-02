"""The magic-state teacher must inject genuine non-stabilizer coherence.

Every other spoqc teacher reduces to a classical mixture of Fock inputs (only
``|amplitude|^2`` survives the spin trace).  This one uses a photon-measurement +
feedforward gadget to teleport a non-Clifford ``T`` onto the data photons, so its
interferometer input is a *pure, non-stabilizer* state.  These tests pin that down
at the state level (fast, no perceval) and check the teacher runs end to end.
"""

from __future__ import annotations

import itertools

import numpy as np


def _emitter_train_state(k: int, t_var=1):
    """Numpy model of the emitter train + readout, post-selected on ``mu=0``.

    Returns the (2^k,) pure-state amplitudes of the k data photons (dual-rail ->
    one qubit each).  This mirrors ``_build_magic_processor`` gate-for-gate, with
    ``t_var`` T gates (one per cluster gap, first ``t_var`` gaps).
    """
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
    n = k + 1                                   # k data qubits + 1 readout

    def op1(g, q):
        U = np.array([[1]], complex)
        for i in range(n):
            U = np.kron(U, g if i == q else np.eye(2, dtype=complex))
        return U

    def cnot(c, t):
        d = 2 ** n
        U = np.zeros((d, d), complex)
        for x in range(d):
            b = [(x >> (n - 1 - i)) & 1 for i in range(n)]
            if b[c]:
                b[t] ^= 1
            U[sum(v << (n - 1 - i) for i, v in enumerate(b)), x] = 1
        return U

    psi = np.zeros(2 ** n, complex)
    psi[0] = 1.0
    psi = op1(H, 0) @ psi                       # spin |+>
    for j in range(k):
        psi = cnot(0, j + 1) @ psi               # emit data photon j+1 (qubit index j+1)
        psi = op1(H, 0) @ psi                    # cluster gap
        if j < t_var:
            psi = op1(T, 0) @ psi                # magic T in this gap
    psi = cnot(0, n - 1) @ psi                    # emit readout photon (last qubit)

    # post-select readout qubit = 0, factor out the (now separable) spin
    out = np.zeros((2, 2 ** k), complex)         # [spin, data]
    for x in range(2 ** n):
        b = [(x >> (n - 1 - i)) & 1 for i in range(n)]
        if b[n - 1] != 0:
            continue
        data = 0
        for i in range(1, k + 1):
            data = (data << 1) | b[i]
        out[b[0], data] += psi[x]
    phi = out[0] if np.linalg.norm(out[0]) >= np.linalg.norm(out[1]) else out[1]
    return phi / np.linalg.norm(phi)


def _pauli_expectations(phi):
    P = {"I": np.eye(2, dtype=complex),
         "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]], complex),
         "Z": np.diag([1, -1]).astype(complex)}
    k = int(round(np.log2(len(phi))))
    vals = []
    for combo in itertools.product("IXYZ", repeat=k):
        if all(c == "I" for c in combo):
            continue
        M = np.array([[1]], complex)
        for c in combo:
            M = np.kron(M, P[c])
        vals.append(abs(np.vdot(phi, M @ phi)))
    return np.array(vals)


def test_input_is_pure_and_non_stabilizer():
    phi = _emitter_train_state(3, t_var=1)
    assert abs(np.vdot(phi, phi) - 1.0) < 1e-9                     # pure, normalized
    vals = _pauli_expectations(phi)
    off = np.sum((vals > 1e-6) & (vals < 1 - 1e-6))
    assert off > 0, "a stabilizer state has all |<P>| in {0,1}; magic must break this"


def test_t_var_controls_magic():
    # t_var=0 -> pure cluster (stabilizer, no magic); t_var>=1 -> non-stabilizer.
    off0 = np.sum((_pauli_expectations(_emitter_train_state(3, t_var=0)) > 1e-6)
                  & (_pauli_expectations(_emitter_train_state(3, t_var=0)) < 1 - 1e-6))
    assert off0 == 0, "t_var=0 must yield a stabilizer state (no magic)"
    for tv in (1, 2, 3):
        vals = _pauli_expectations(_emitter_train_state(3, t_var=tv))
        assert np.sum((vals > 1e-6) & (vals < 1 - 1e-6)) > 0, f"t_var={tv} must be non-stabilizer"


def test_readout_gap_T_saturates():
    # t_var=k puts a T in the readout gap (phase before the Z readout) -> washed out,
    # so it is identical to t_var=k-1. The distinct levels are 0..k-1.
    k = 3
    phi_km1 = _emitter_train_state(k, t_var=k - 1)
    phi_k = _emitter_train_state(k, t_var=k)
    assert abs(abs(np.vdot(phi_km1, phi_k)) - 1.0) < 1e-9, "t_var=k must equal t_var=k-1"
    # and consecutive distinct levels are genuinely different states
    for tv in range(k - 1):
        a, b = _emitter_train_state(k, t_var=tv), _emitter_train_state(k, t_var=tv + 1)
        assert abs(np.vdot(a, b)) < 1 - 1e-6, f"t_var={tv} and {tv + 1} should differ"


def test_forward_runs_and_is_bounded():
    import torch

    from model import build_teacher
    from Generator.config import (ExperimentConfig, GenerationConfig,
                                  ProblemConfig, SeedConfig)

    cfg = ExperimentConfig()
    cfg.problem = ProblemConfig(m=6, k=3, observable="parity", n_features=5)
    cfg.generation = GenerationConfig(generator="spoqc_magic_photonic", size=4)
    cfg.seeds = SeedConfig(sample_seed=1, teacher_seed=7)
    t = build_teacher(cfg)
    X = torch.tensor(np.random.default_rng(0).uniform(0, 1, size=(3, 5)), dtype=torch.float32)
    soft = t(X)
    assert soft.shape == (3, 1)
    assert float(soft.abs().max()) <= 1.0 + 1e-6
