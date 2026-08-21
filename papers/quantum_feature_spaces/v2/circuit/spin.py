"""Spin-qubit state preparation in numpy, plus the per-row process pool.

Carried from ``model/spoqc_utils.py`` and the gate helpers of ``model/spoqc_magic.py``, with one
omission: the legacy ``_score_processor`` is **gone**.  It scored a detection distribution with a
hand-rolled if-chain over four observable names, which is exactly what the observable registry
(:mod:`v2.observable`) replaces -- the spin preps now return a full distribution like every other
model and are scored by the same code as the boson sampler.

The spin prep is built in numpy on the initial source state rather than with processor gates,
because spoqc has no two-qubit processor gate and a ``CX`` on ``|0...0>`` is the identity -- so
the single-qubit prep must precede the entangler.
"""

from __future__ import annotations

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def apply_1q(psi, q, g, n):
    """Apply the 2x2 gate ``g`` to qubit ``q`` of an ``n``-qubit state vector (qubit 0 = MSB)."""
    U = np.array([[1.0 + 0j]])
    for i in range(n):
        U = np.kron(U, g if i == q else np.eye(2, dtype=complex))
    return U @ psi


def apply_cx(state, control: int, target: int, n_qubits: int | None = None):
    """``CNOT(control -> target)`` on a spin state vector (1-D) or density matrix (2-D).

    Qubit 0 is the most-significant bit, matching the joint ordering of
    ``with_initial_source_state``.
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


def spin_state(n_q, rx, ry, cx_pairs, rz=None, layers=1, data_angles=None):
    """Joint source density matrix: an optional one-shot data encoding, then ``layers`` rounds of
    (per-qubit ``H -> Rx (-> Rz) -> Ry``, then the CX chain).

    ``rx``/``ry``/``rz`` are ``(layers, n_q)`` -- a fresh seeded draw per layer, not the same
    angles replayed, so ``layers`` genuinely deepens the circuit rather than repeating a gate that
    would partly cancel (``H`` applied twice in a row is its own inverse, so if the second layer's
    rotations were held at the first layer's values, ``H -> Rx -> Ry -> H -> Rx -> Ry`` would not
    reduce to a no-op, but the ``H``s would still be doing less work than two independently-seeded
    layers).  ``layers=1`` reproduces the legacy single-round circuit exactly, so ``rx``/``ry`` may
    also be passed as plain ``(n_q,)`` arrays for that case (see :func:`_broadcast_layer_angles`).

    ``data_angles``, when given, is ``(n_q, 2)`` -- per-qubit ``(rx, ry)`` data angles applied
    **once**, on ``|0...0>``, before the first layer.  Not per-layer like the seeded twists: this
    carries ``x`` itself, so repeating it ``layers`` times would inject ``layers`` copies of the
    same feature into the state rather than deepening the circuit around a single encoding -- the
    seeded ``rx``/``ry``/``rz`` are the part meant to vary with depth, ``x`` is not.

    ``rz`` (per-qubit angles, or ``None``) goes **between Rx and Ry** so the following Ry rotates
    its phase into the rail populations.  A *trailing* Rz is a complete no-op: dual-rail emission
    correlates each spin's Z-basis with its photon's rail, so tracing the unmeasured spins always
    leaves the photons in a classical mixture over Fock states (only ``|amplitude|^2`` survives),
    and CX is a Z-basis permutation, so it cannot rescue a pure phase -- true of the *last* layer's
    entangler only; an earlier layer's CX is not trailing and still acts on a complex amplitude,
    since a later layer's ``Rx``/``Ry`` follows it.
    """
    rx = _broadcast_layer_angles(rx, layers, n_q)
    ry = _broadcast_layer_angles(ry, layers, n_q)
    rz = None if rz is None else _broadcast_layer_angles(rz, layers, n_q)

    psi = np.zeros(2 ** n_q, dtype=complex)
    psi[0] = 1.0                                            # |0...0>
    if data_angles is not None:
        da = np.asarray(data_angles, dtype=float)
        if da.shape != (n_q, 2):
            raise ValueError(f"data_angles must be ({n_q}, 2), got {da.shape}")
        for q in range(n_q):
            psi = apply_1q(psi, q, _rx(float(da[q, 0])), n_q)
            psi = apply_1q(psi, q, _ry(float(da[q, 1])), n_q)
    for layer in range(layers):
        for q in range(n_q):
            psi = apply_1q(psi, q, _H, n_q)                  # |0> -> |+>
            psi = apply_1q(psi, q, _rx(float(rx[layer, q])), n_q)   # seeded twists -> Gamma != I/2
            if rz is not None:
                psi = apply_1q(psi, q, _rz(float(rz[layer, q])), n_q)
            psi = apply_1q(psi, q, _ry(float(ry[layer, q])), n_q)
        for c, t in cx_pairs:
            psi = apply_cx(psi, c, t, n_q)                    # entangler on a real superposition
    return np.outer(psi, psi.conj())


def _broadcast_layer_angles(angles, layers: int, n_q: int) -> np.ndarray:
    """``(layers, n_q)``, accepting a plain ``(n_q,)`` array at ``layers == 1`` unchanged."""
    a = np.asarray(angles, dtype=float)
    if a.ndim == 1:
        if layers != 1:
            raise ValueError(f"angles are (n_q,) but layers={layers}; draw (layers, n_q) angles")
        return a.reshape(1, n_q)
    if a.shape != (layers, n_q):
        raise ValueError(f"angles must be ({layers}, {n_q}), got {a.shape}")
    return a


def normalize_cx_pairs(cx_pairs, k: int):
    """Validate and normalise ``cx_pairs`` to a list of ``(control, target)`` int tuples.

    ``cx_pairs="chain"`` expands to the linear ladder ``[(0,1), (1,2), ..., (k-2,k-1)]`` at
    whatever ``k`` the config actually has, so a config can ask for "the ladder" without the
    caller pre-computing a ``k``-sized pair list by hand.
    """
    if cx_pairs == "chain":
        return [(i, i + 1) for i in range(k - 1)]
    pairs = []
    for pair in (cx_pairs or []):
        c, t = int(pair[0]), int(pair[1])
        if c == t or not (0 <= c < k and 0 <= t < k):
            raise ValueError(f"cx pair {pair} invalid for k={k} qubits (need distinct 0<=i<k)")
        pairs.append((c, t))
    return pairs


def first_primes(n: int) -> list[int]:
    """First ``n`` primes: 2, 3, 5, 7, 11, ...  (the ``rz_angles='prime'`` twist angles)."""
    primes, c = [], 2
    while len(primes) < n:
        if all(c % p for p in primes):
            primes.append(c)
        c += 1
    return primes


def discrete_angles(rng, k: int, levels: int) -> np.ndarray:
    """``k`` angles drawn from ``+-0.1 .. +-0.1*levels`` (the ``angle_levels`` knob)."""
    mags = 0.1 * rng.integers(1, int(levels) + 1, size=k)
    signs = rng.choice([-1.0, 1.0], size=k)
    return mags * signs


# --- the magic gap gate ------------------------------------------------------------------- #


def apply_gap_gate(p, gate_kind: str, gate_params, j: int) -> None:
    """Inject the non-Clifford "magic" gate into cluster gap ``j`` on the spin (mode 0).

    ``gate_kind`` is a *picklable string*, not a closure, so a whole task survives a process
    pool.  This one function replaces the four legacy ``spoqc_magic*`` subclasses, whose only
    difference was which branch here fires:

    - ``"t"``  -- the fixed ``T = P(pi/4)`` (legacy ``spoqc_magic``)
    - ``"rz"`` -- ``Rz(gate_params[j])``, e.g. increasing primes (``spoqc_magic_prime``)
    - ``"u3"`` -- Haar ``SU(2)`` as ``Rz(phi) Ry(theta) Rz(lam)`` (``spoqc_magic_rand``)
    - ``"u3_x"`` -- the same with ``Rx`` in place of ``Ry`` (``spoqc_magic_rand_x``), optionally
      angle-scaled by ``/5`` (``_m``) or ``/20`` (``_s``)
    """
    scale = 1.0
    if gate_kind.endswith("_m"):
        gate_kind, scale = gate_kind[:-2], 5.0
    elif gate_kind.endswith("_s"):
        gate_kind, scale = gate_kind[:-2], 20.0

    if gate_kind == "t":
        p.gate.t(0)
    elif gate_kind == "rz":
        p.gate.rz(0, float(gate_params[j]) / scale)
    elif gate_kind in ("u3", "u3_x"):
        theta, phi, lam = gate_params[j]
        rot = p.gate.ry if gate_kind == "u3" else p.gate.rx
        p.gate.rz(0, float(lam) / scale)                  # ZYZ Euler: Rz(phi) R(theta) Rz(lam)
        rot(0, float(theta) / scale)
        p.gate.rz(0, float(phi) / scale)
    else:
        raise ValueError(f"unknown gap gate_kind {gate_kind!r}")


def parse_gate_kind(gate_kind: str):
    """Split ``gate_kind`` into ``(magic_kind, encode_qubit, encode_iface)``.

    The prefix picks the gap gate (:func:`apply_gap_gate`); a suffix controls *where* ``x`` is
    encoded:

    - no suffix         -> interferometer only; the spin carries no data (legacy)
    - ``_rxry``         -> data via ``rx``/``ry`` on the spin, interferometer without the encoding
    - ``_rxry_iface``   -> data on the spin *and* in the interferometer

    In each ``_rxry*`` gap two features drive the spin (``rx(x[2j])``, ``ry(x[2j+1])``, indices
    mod ``n_features`` so they cycle when ``k`` gaps need more angles than there are features).
    """
    if gate_kind.endswith("_rxry_iface"):
        return gate_kind[: -len("_rxry_iface")], True, True
    if gate_kind.endswith("_rxry"):
        return gate_kind[: -len("_rxry")], True, False
    return gate_kind, False, True


def haar_su2_angles(rng, n: int) -> list[tuple[float, float, float]]:
    """``n`` Haar-random ``SU(2)`` ZYZ Euler triples ``(theta, phi, lam)``.

    ``phi, lam ~ U(0, 2pi)`` and ``cos(theta)`` uniform in ``[-1, 1]`` gives the uniform measure
    on the Bloch sphere.  A generic ``SU(2)`` element is non-Clifford with probability 1.
    """
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    lam = rng.uniform(0.0, 2 * np.pi, size=n)
    theta = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    return [(float(theta[i]), float(phi[i]), float(lam[i])) for i in range(n)]


def full_distribution(p, n_modes: int):
    """A built processor's full detection distribution as ``(keys, probs)`` plain arrays.

    ``keys`` is ``(n_states, n_modes)`` per-mode photon counts over **all** modes (readout modes
    included) and ``probs`` the matching probabilities.  Nothing is post-selected or thresholded
    here, so any selection *and* any observable can be recomputed offline from the saved arrays --
    which is the whole point of persisting the distribution rather than a score.
    """
    keys, probs = [], []
    for key, pr in p.probabilities().items():
        keys.append([int(key[i]) for i in range(n_modes)])
        probs.append(float(pr))
    return np.asarray(keys, dtype=np.int16), np.asarray(probs, dtype=np.float64)


# --- per-row parallelism ------------------------------------------------------------------- #
#
# The perceval prep paths evaluate one input row at a time and the rows are independent, so the
# loop parallelises trivially.  It must use *processes*, not threads: building the circuit calls
# ``pcvl.random_seed`` / ``torch.manual_seed`` (global state), which concurrent threads would race
# on.  Separate processes each hold their own global RNG, so results are deterministic and
# order-independent.  Carried verbatim from model/spoqc_utils.py -- this contract is load-bearing.


#: Every worker process's own intra-op thread pools (torch, and the BLAS backends numpy/sklearn
#: sit on) default to using ALL visible CPUs -- independent of, and invisible to, the process-count
#: gate below. N worker processes each spinning up their own full-width thread pool is N x
#: oversubscription (e.g. 3 processes x 12 threads each on a 12-core box), which is the actual
#: mechanism behind "parallelising gets stuck / no memory or CPU" even when the process count itself
#: was capped correctly -- the process gate alone never prevented it. `_init_worker_threads` runs
#: once per worker process, before any task, and pins every thread pool this repo's workloads
#: actually use (torch's own two settings, plus the OMP/MKL/OPENBLAS env vars numpy/scikit-learn's
#: linked BLAS reads at import time) down to 1 thread each, so the only parallelism left is across
#: processes -- exactly what the process-count gate is sized for.
def _init_worker_threads() -> None:
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


#: Fallback per-task memory estimate, used only when a caller passes neither ``per_worker_mb`` nor
#: ``bytes_per_task`` -- i.e. when the caller has no cheap way to know its own task's real memory
#: footprint. Deliberately conservative (an overestimate costs fewer workers and a slower run; an
#: underestimate is the OOM/thrash this whole gate exists to prevent), but a FLAT constant is wrong
#: on its own terms for any caller whose task memory actually scales with problem size -- a
#: photonic/spin row's distribution is `O(n_fock(m,k))`, combinatorial in `m`, so a constant sized
#: for `m=6` silently undercounts by orders of magnitude at `m=14+` (measured elsewhere in this
#: repo: `n_out=77520` at `m=14,k=7`, ~3 GB for a `(1e4, n_out)` fp32 matrix -- `model/base.py`'s
#: `FORWARD_ELEMENTS` docstring). Prefer passing ``bytes_per_task`` computed from the actual
#: problem size at the call site (see ``circuit/prep.py``'s callers) over relying on this default.
_DEFAULT_TASK_BYTES = 1536 * 1024 * 1024


def _resolve_workers(n_jobs, n_rows, *, per_worker_mb=None, bytes_per_task=None) -> int:
    """Worker count: ``n_jobs`` capped by CPUs and free RAM.

    ``1`` = serial (default), ``-1``/``0``/``None`` = auto (CPUs - 1), or an explicit count.
    Below two rows there is nothing to parallelise.

    ``bytes_per_task`` is the preferred way to size the memory gate: pass the actual estimated peak
    memory of ONE task, computed from the real problem size at the call site (e.g.
    ``n_fock(m, k) * n_bytes_per_outcome`` for a photonic/spin row-build task) -- not a fixed
    constant, since task memory here typically scales combinatorially with ``(m, k)`` and a
    constant sized for a small problem silently undercounts at a large one (see
    :data:`_DEFAULT_TASK_BYTES`'s docstring). ``per_worker_mb`` is kept as a coarser override for a
    caller that wants to hand-pick a flat MB figure directly; if both are given ``bytes_per_task``
    wins. If neither is given, falls back to :data:`_DEFAULT_TASK_BYTES`.

    The memory gate REQUIRES ``psutil``: a missing or broken ``psutil`` import is fatal here, not
    silently skipped -- a machine that hangs because the gate silently disabled itself is worse than
    one that errors loudly and tells you to ``pip install psutil``.
    """
    import os

    if n_rows < 2 or n_jobs == 1:
        return 1
    cpu = os.cpu_count() or 1
    want = max(1, cpu - 1) if n_jobs in (None, 0, -1) else int(n_jobs)
    want = min(want, n_rows, cpu)

    import psutil

    task_bytes = (bytes_per_task if bytes_per_task is not None
                 else per_worker_mb * 1024 * 1024 if per_worker_mb is not None
                 else _DEFAULT_TASK_BYTES)
    avail_bytes = psutil.virtual_memory().available
    want = min(want, max(1, int(0.7 * avail_bytes / max(1, task_bytes))))
    return max(1, want)


def parallel_row_map(worker, tasks, n_jobs, *, per_worker_mb=None, bytes_per_task=None):
    """Map ``worker`` over ``tasks``, returning results **in input order**.

    Serial when ``n_jobs == 1`` or there are <2 rows; otherwise a process pool whose ordered
    ``map`` keeps results row-aligned even though rows finish out of order.  ``worker`` must be a
    module-level (picklable) callable and each task a picklable tuple, so any accumulation the
    caller does afterwards stays deterministic.

    ``bytes_per_task``/``per_worker_mb`` size the memory gate -- see :func:`_resolve_workers`. Pass
    ``bytes_per_task`` computed from the real per-call problem size (``m``/``k``/dataset width)
    whenever the caller can, rather than relying on the flat fallback.

    Every pool worker process runs :func:`_init_worker_threads` once, before its first task, so
    torch/BLAS intra-op threading inside ``worker`` cannot oversubscribe on top of the process-level
    parallelism this function already provides (see that function's docstring) -- this is what
    actually fixes the "gets stuck, no memory or CPU" failure mode, not just the worker-count cap
    below, which was already RAM-aware before this fix.
    """
    workers = _resolve_workers(n_jobs, len(tasks), per_worker_mb=per_worker_mb,
                               bytes_per_task=bytes_per_task)
    if workers == 1:
        _init_worker_threads()               # serial path: still pin this process's own threads,
        return [worker(t) for t in tasks]    # so a serial mlp fit doesn't oversubscribe either

    import concurrent.futures as cf

    chunk = max(1, len(tasks) // (workers * 4))
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_threads) as ex:
        return list(ex.map(worker, tasks, chunksize=chunk))
