"""Confirm the numpy spin-prep gates match perceval_spoqc, and that preparing the
spin state in numpy is equivalent to applying the gates in the processor.

This matters because the spoqc teachers now bake H/Rx/Ry (and the CX entangler)
into the initial source state in numpy (the only way to entangle, since spoqc has
no two-qubit processor gate) instead of calling p.gate.h / add_source_op.
"""

from __future__ import annotations

import numpy as np


def test_gate_matrices_match_spoqc():
    from perceval_spoqc import Gate
    from model.spoqc_utils import _H, _rx, _ry

    assert np.allclose(_H, np.asarray(Gate.H)), "H mismatch"
    for t in (0.0, 0.3, 1.0, np.pi / 2, 2.5, 4.7):
        assert np.allclose(_rx(t), np.asarray(Gate.Rx(t))), f"Rx({t}) mismatch"
        assert np.allclose(_ry(t), np.asarray(Gate.Ry(t))), f"Ry({t}) mismatch"


def test_apply_cx_is_unitary_involution():
    from model.spoqc_utils import apply_cx

    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    once = apply_cx(psi, 0, 1, 3)
    twice = apply_cx(once, 0, 1, 3)
    assert np.allclose(twice, psi)                       # CX^2 = I
    assert abs(np.linalg.norm(once) - 1.0) < 1e-12       # norm preserved (unitary)


def test_numpy_prep_equiv_processor_gates():
    """Base prep (no CX): numpy initial-state prep vs processor gate ops -> same probs."""
    from perceval import Detector
    from perceval_spoqc import Gate, HybridProcessor

    from model.spoqc_utils import _sandwich_concrete, _spin_state

    m, n_q, nf, seed = 6, 3, 5, 42
    rng = np.random.default_rng(seed)
    rx, ry = rng.uniform(0, 2 * np.pi, n_q), rng.uniform(0, 2 * np.pi, n_q)
    x = np.linspace(0.1, 1.0, nf)

    # NEW path: full prep baked into the numpy initial state, processor only emits.
    pa = HybridProcessor(num_sources=n_q, num_modes=m)
    pa.with_initial_source_state(_spin_state(n_q, rx, ry, ()))
    for q in range(n_q):
        pa.emit(q, into=(2 * q, 2 * q + 1))
    pa.add(0, _sandwich_concrete(m, nf, seed, x))
    for md in range(m):
        pa.add(md, Detector())

    # OLD path: gates applied in the processor.
    pb = HybridProcessor(num_sources=n_q, num_modes=m)
    r0 = np.zeros((2 ** n_q, 2 ** n_q), dtype=complex)
    r0[0, 0] = 1.0
    pb.with_initial_source_state(r0)
    for q in range(n_q):
        pb.gate.h(q)
        pb.add_source_op(q, Gate.Rx(float(rx[q])))
        pb.add_source_op(q, Gate.Ry(float(ry[q])))
        pb.emit(q, into=(2 * q, 2 * q + 1))
    pb.add(0, _sandwich_concrete(m, nf, seed, x))
    for md in range(m):
        pb.add(md, Detector())

    A, B = pa.probabilities(), pb.probabilities()
    keys = set(A) | set(B)
    assert max(abs(A.get(k, 0.0) - B.get(k, 0.0)) for k in keys) < 1e-9