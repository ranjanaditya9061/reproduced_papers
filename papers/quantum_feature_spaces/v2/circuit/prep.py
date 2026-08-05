"""State preparation: how the photons entering the interferometer are produced.

This module is where the legacy ``spoqc`` family collapses.  Seven modules --
``spoqc.py``, ``spoqc_low.py``, ``spoqc_prime.py``, ``spoqc_magic.py``,
``spoqc_magic_prime.py``, ``spoqc_magic_rand.py``, ``spoqc_magic_rand_x.py`` -- were seven
``Teacher`` subclasses differing only in which gate went where.  Here they are **three preps and
their parameters**:

=========================  ==================================================================
prep                       replaces
=========================  ==================================================================
``fock``                   the plain fixed Fock input state (``model/photonic.py``)
``spin``                   ``spoqc`` (``cx_pairs``), ``spoqc_low`` (``angle_levels``),
                           ``spoqc_prime`` (``rz_angles="prime"``)
``spin_magic``             ``spoqc_magic`` (``gate_kind="t"``), ``_prime`` (``"rz"``),
                           ``_rand`` (``"u3"``), ``_rand_x`` (``"u3_x"``), each x ``t_var``
=========================  ==================================================================

Every prep returns a **full outcome distribution**, so all of them are scored by the shared
observable registry.  The legacy spin preps scored with a private if-chain over four observable
names, which is why ``osc``, ``ent``, ``prod_parity`` and the graph families were unavailable on
them; here they are available automatically.

A prep declares its own outcome basis, because ``spin_magic``'s includes two readout modes on top
of the ``m`` data modes.  Nothing is post-selected inside the prep: the ``mu = 0`` selection rides
in the saved distribution's readout modes so it can be applied -- or not -- offline.
"""

from __future__ import annotations

import numpy as np

from . import spin as _spin
from .photonic_circuit import build_quantum_layer

#: name -> StatePrep subclass, populated on subclassing.
PREPS: dict[str, type["StatePrep"]] = {}


class StatePrep:
    """Produces the photonic input and, with it, a full outcome distribution.

    Two kinds of implementation, distinguished by :attr:`is_batched`:

    * ``is_batched = True`` (``fock``) -- a merlin ``QuantumLayer`` evaluates a whole batch at
      once and is differentiable in ``X``, so :mod:`v2.metrics` can take input-Jacobians through
      it directly.
    * ``is_batched = False`` (the spin preps) -- perceval builds one processor per row, so the
      batch is a process-pool map and gradients are unavailable (finite differences only).
    """

    name: str | None = None

    #: True when ``probs`` takes a whole ``(N, n_features)`` batch and is autodiff-friendly.
    is_batched: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            PREPS[cls.name] = cls

    @classmethod
    def from_config(cls, cfg) -> "StatePrep":
        raise NotImplementedError

    def validate(self, *, m: int, k: int) -> None:
        return None

    def outcome_modes(self, m: int) -> int:
        """Number of modes the outcome keys span (``m`` unless the prep adds readout modes)."""
        return m

    def readout_modes(self, m: int) -> tuple:
        """Modes reserved for a post-selection readout; ``()`` when there is none."""
        return ()

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        """``(N, n_outcomes)`` distribution over this prep's outcome basis."""
        raise NotImplementedError

    def outcome_keys(self, *, m, k):
        """The outcome basis as a list of per-mode occupation tuples, or ``None`` to infer it.

        ``None`` means "whatever the backend reports", which the spin preps need because
        perceval prunes negligible-probability outcomes per row.
        """
        return None

    def spec(self) -> dict:
        """Identity fields folded into the artifact hash."""
        return {"prep": self.name}


class FockPrep(StatePrep):
    """A fixed Fock input state: ``k`` photons evenly spaced over ``m`` modes.

    The plain boson sampler.  Uses a merlin ``QuantumLayer``, so it is batched and
    differentiable in ``X``.
    """

    name = "fock"
    is_batched = True

    def __init__(self):
        self._layer = None
        self._key = None
        self.input_state = None

    @classmethod
    def from_config(cls, cfg) -> "FockPrep":
        return cls()

    def _get_layer(self, *, m, k, n_features, seed, encoding):
        key = (m, k, n_features, int(seed), getattr(encoding, "name", encoding))
        if self._layer is None or self._key != key:
            self._layer, self.input_state = build_quantum_layer(
                m, k, n_features, int(seed), measure="probs", encoding=encoding)
            self._key = key
        return self._layer

    def validate(self, *, m: int, k: int) -> None:
        # Bosons may bunch, so k > m is legal here (unlike the fermion model, which needs
        # k <= m by Pauli exclusion -- that check lives on the model, not the prep).
        return None

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        layer = self._get_layer(m=m, k=k, n_features=n_features, seed=seed, encoding=encoding)
        return layer.forward(X)

    def outcome_keys(self, *, m, k):
        from circuit.fock import fock_keys

        return fock_keys(m, k)


class SpinPrep(StatePrep):
    """Dual-rail photons emitted by ``k`` spin qubits (``perceval_spoqc``).

    Per qubit ``|0> -> H -> Rx(a) (-> Rz) -> Ry(b)``, optionally entangled by a CX chain, then
    emitted dual-rail into modes ``(2q, 2q+1)``.  The spin prep is built in numpy on the initial
    source state (:func:`v2.circuit.spin.spin_state`) because spoqc has no two-qubit processor
    gate and a CX on ``|0...0>`` is the identity.

    Absorbs three legacy classes as parameters: ``cx_pairs`` (``spoqc``), ``angle_levels``
    (``spoqc_low``, which drew the twists from ``+-0.1 .. +-0.1*levels`` instead of uniformly)
    and ``rz_angles="prime"`` (``spoqc_prime``, an extra ``Rz`` at increasing-prime radians
    between Rx and Ry).
    """

    name = "spin"
    is_batched = False

    def __init__(self, *, cx_pairs=None, angle_levels=None, rz_angles=None):
        self.cx_pairs_raw = cx_pairs
        self.angle_levels = None if angle_levels is None else int(angle_levels)
        self.rz_angles = rz_angles
        if rz_angles not in (None, "prime"):
            raise ValueError(f"rz_angles must be None or 'prime' (got {rz_angles!r})")

    @classmethod
    def from_config(cls, cfg) -> "SpinPrep":
        m = cfg.model
        return cls(cx_pairs=m.cx_pairs, angle_levels=m.angle_levels, rz_angles=m.rz_angles)

    def validate(self, *, m: int, k: int) -> None:
        if m % 2:
            raise ValueError("prep 'spin' emits dual-rail photons -> requires even m")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        _spin.normalize_cx_pairs(self.cx_pairs_raw, k)

    def _angles(self, k: int, seed: int):
        rng = np.random.default_rng(int(seed))
        if self.angle_levels is None:
            rx = rng.uniform(0.0, 2 * np.pi, size=k)
            ry = rng.uniform(0.0, 2 * np.pi, size=k)
        else:
            rx = _spin.discrete_angles(rng, k, self.angle_levels)
            ry = _spin.discrete_angles(rng, k, self.angle_levels)
        rz = None
        if self.rz_angles == "prime":
            rz = np.asarray(_spin.first_primes(k), dtype=float)
        return rx, ry, rz

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        import torch

        rx, ry, rz = self._angles(k, seed)
        pairs = _spin.normalize_cx_pairs(self.cx_pairs_raw, k)
        enc_name = getattr(encoding, "name", encoding)
        rows = X.detach().cpu().numpy()
        tasks = [(row, m, k, n_features, int(seed), rx, ry, pairs, rz, enc_name)
                 for row in rows]
        results = _spin.parallel_row_map(_spin_row_worker, tasks, n_jobs)
        keys, probs = _align_rows(results, n_modes=m)
        self._keys = keys
        return torch.as_tensor(probs, dtype=torch.float32)

    def outcome_keys(self, *, m, k):
        return getattr(self, "_keys", None)

    def spec(self) -> dict:
        # Knobs are added only when set, so a config that leaves them at their defaults hashes
        # the same as one written before the knob existed.  Pairs are already validated by
        # `validate`, so this only canonicalises them to lists of ints.
        spec = {"prep": self.name, "source": "H_Rx_Ry_seeded"}
        pairs = [[int(a), int(b)] for a, b in (self.cx_pairs_raw or [])]
        if pairs:
            spec["cx_pairs"] = pairs
        if self.angle_levels is not None:
            spec["angle_levels"] = self.angle_levels
        if self.rz_angles is not None:
            spec["rz_angles"] = self.rz_angles
        return spec


class SpinMagicPrep(StatePrep):
    """Emitter-train cluster state with a non-Clifford gate injected into ``t_var`` gaps.

    One spin emits ``k`` data photons dual-rail plus a readout photon; a "magic" gate in gap
    ``j < t_var`` makes the teleported interferometer input non-stabilizer.  Absorbs the four
    legacy ``spoqc_magic*`` classes as ``gate_kind`` (``"t"`` | ``"rz"`` | ``"u3"`` | ``"u3_x"``,
    optionally ``_m``/``_s`` angle-scaled, optionally ``_rxry[_iface]`` to move the data encoding
    onto the spin) -- see :func:`v2.circuit.spin.apply_gap_gate`.

    The outcome basis spans ``m + 2`` modes: the two extra are the readout pair, reported by
    :meth:`readout_modes`.  **The ``mu = 0`` post-selection is NOT applied here** -- it stays in
    the saved distribution so it can be applied, varied or dropped offline, which the legacy
    teacher could not do because it collapsed to a score inside ``forward``.
    """

    name = "spin_magic"
    is_batched = False

    #: gates injected when ``t_var`` is unset
    DEFAULT_T_VAR = 1
    DEFAULT_GATE_KIND = "t"

    def __init__(self, *, t_var=None, gate_kind=None):
        self.t_var = t_var
        self.gate_kind = self.DEFAULT_GATE_KIND if gate_kind is None else str(gate_kind)

    @classmethod
    def from_config(cls, cfg) -> "SpinMagicPrep":
        return cls(t_var=cfg.model.t_var, gate_kind=cfg.model.gate_kind)

    def resolved_t_var(self, k: int) -> int:
        return self.DEFAULT_T_VAR if self.t_var is None else int(self.t_var)

    def validate(self, *, m: int, k: int) -> None:
        if m % 2:
            raise ValueError("prep 'spin_magic' emits dual-rail photons -> requires even m")
        if k < 2:
            raise ValueError("magic teleportation needs k >= 2 data photons "
                             "(the gate sits before the second emission)")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        t = self.resolved_t_var(k)
        if not 0 <= t <= k:
            raise ValueError(f"t_var must be in [0, k] = [0, {k}] (got {t})")
        _spin.parse_gate_kind(self.gate_kind)

    def outcome_modes(self, m: int) -> int:
        return m + 2

    def readout_modes(self, m: int) -> tuple:
        return (m, m + 1)

    def _gate_params(self, k: int, seed: int):
        """Per-gap gate parameters, drawn once from ``seed`` (they are the fixed circuit weights)."""
        magic_kind, _, _ = _spin.parse_gate_kind(self.gate_kind)
        base = magic_kind[:-2] if magic_kind.endswith(("_m", "_s")) else magic_kind
        if base == "t":
            return None
        rng = np.random.default_rng(int(seed))
        if base == "rz":
            return [float(p) for p in _spin.first_primes(k)]
        return _spin.haar_su2_angles(rng, k)

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        import torch

        t_var = self.resolved_t_var(k)
        params = self._gate_params(k, seed)
        enc_name = getattr(encoding, "name", encoding)
        rows = X.detach().cpu().numpy()
        tasks = [(row, m, k, t_var, n_features, int(seed), self.gate_kind, params, enc_name)
                 for row in rows]
        results = _spin.parallel_row_map(_magic_row_worker, tasks, n_jobs)
        keys, probs = _align_rows(results, n_modes=m + 2)
        self._keys = keys
        return torch.as_tensor(probs, dtype=torch.float32)

    def outcome_keys(self, *, m, k):
        return getattr(self, "_keys", None)

    def spec(self) -> dict:
        return {"prep": self.name, "gate_kind": self.gate_kind,
                "t_var": self.t_var, "gadget": "emitter_train_readout_pair"}


# --- picklable process-pool workers -------------------------------------------------------- #
#
# Module-level, tuple-argument: required by ``parallel_row_map``'s process pool.


def _spin_row_worker(task):
    """One row of :class:`SpinPrep`: build the processor, return its full distribution."""
    row, m, k, n_features, seed, rx, ry, pairs, rz, enc_name = task
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    from .photonic_circuit import build_sandwich_circuit

    p = HybridProcessor(num_sources=k, num_modes=m)
    p.with_initial_source_state(_spin.spin_state(k, rx, ry, pairs, rz=rz))
    for q in range(k):
        p.emit(q, into=(2 * q, 2 * q + 1))
    p.add(0, build_sandwich_circuit(m, n_features, seed, enc_name, x=row))
    for mode in range(m):
        p.add(mode, Detector())
    return _spin.full_distribution(p, m)


def _magic_row_worker(task):
    """One row of :class:`SpinMagicPrep`: emitter train + gap gate + readout pair."""
    row, m, k, t_var, n_features, seed, gate_kind, params, enc_name = task
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    from .photonic_circuit import build_sandwich_circuit

    magic_kind, encode_qubit, encode_iface = _spin.parse_gate_kind(gate_kind)
    r0, r1 = m, m + 1
    p = HybridProcessor(num_sources=1, num_modes=m + 2, num_records=m + 2,
                        allow_carry_over=True)
    zero = np.array([1.0, 0.0], dtype=complex)
    p.with_initial_source_state(np.outer(zero, zero.conj()))

    for j in range(k):
        p.gate.h(0)
        if encode_qubit:                                  # data on the spin: 2 features per gap
            p.gate.rx(0, float(row[(2 * j) % n_features]))
            p.gate.ry(0, float(row[(2 * j + 1) % n_features]))
        if j < t_var:
            _spin.apply_gap_gate(p, magic_kind, params, j)
        p.emit(0, into=(2 * j, 2 * j + 1))
    p.gate.h(0)
    p.emit(0, into=(r0, r1))                              # readout photon
    p.add(r0, Detector(), record=r0)
    p.add(r1, Detector(), record=r1)

    # A zeroed x leaves the Haar scrambler but puts no data in the interferometer, which is the
    # qubit-only (``*_rxry``) case; PS(0) is the identity.
    iface_x = row if encode_iface else np.zeros_like(np.asarray(row, dtype=float))
    p.add(0, build_sandwich_circuit(m, n_features, seed, enc_name, x=iface_x))
    for md in range(m):
        p.add(md, Detector(), record=md)
    return _spin.full_distribution(p, m + 2)


def _align_rows(results, *, n_modes: int):
    """Align per-row ``(keys, probs)`` onto one shared basis, zero-filling missing outcomes.

    perceval prunes negligible-probability outcomes, and *which* ones it prunes can differ per
    row, so the rows must be unioned rather than stacked.  Returns ``(keys, probs)`` with
    ``keys`` a ``(n_states, n_modes)`` int array and ``probs`` ``(n_rows, n_states)``.
    """
    index: dict[tuple, int] = {}
    basis: list[tuple] = []
    for keys, _ in results:
        for row in keys:
            t = tuple(int(v) for v in row)
            if t not in index:
                index[t] = len(basis)
                basis.append(t)
    out = np.zeros((len(results), len(basis)), dtype=np.float64)
    for i, (keys, probs) in enumerate(results):
        for row, pr in zip(keys, probs):
            out[i, index[tuple(int(v) for v in row)]] = pr
    return np.asarray(basis, dtype=np.int16), out


def build_prep(cfg) -> StatePrep:
    """Instantiate the prep named by ``cfg.model.prep`` and validate its geometry."""
    name = cfg.model.prep
    if name not in PREPS:
        raise ValueError(f"unknown prep {name!r}; registered: {sorted(PREPS)}")
    prep = PREPS[name].from_config(cfg)
    prep.validate(m=cfg.problem.m, k=cfg.problem.k)
    return prep
