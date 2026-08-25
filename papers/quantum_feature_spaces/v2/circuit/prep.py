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
from .encoding import build_encoding
from .fock import binary_keys, fock_keys, n_fock
from .photonic_circuit import build_quantum_layer

#: name -> StatePrep subclass, populated on subclassing.
PREPS: dict[str, type["StatePrep"]] = {}

#: Rough per-outcome byte cost of one row's built distribution inside a `parallel_row_map` worker:
#: a perceval `probabilities()` dict entry is a Python tuple key (several ints, ~28 bytes each in
#: CPython on top of the tuple's own overhead) plus a Python float value (~24 bytes) plus dict
#: bucket overhead -- call it ~200 bytes/outcome as a working estimate for the live dict a worker
#: holds mid-build, before `_align_rows`/`_align_rows_fixed` compact it into the numpy arrays
#: actually returned. Not measured precisely; deliberately on the high side per
#: `circuit.spin._DEFAULT_TASK_BYTES`'s own conservative-by-design convention.
_BYTES_PER_OUTCOME = 200


#: Bytes/complex128 amplitude in the qubit joint-state vector :func:`circuit.spin.spin_state`
#: builds (``np.zeros(2**n_q, dtype=complex)``) -- 16 bytes/element (complex128) plus roughly
#: another 16 bytes/element of working-copy overhead across the ``layers`` rounds of
#: ``apply_1q``/entangler calls that each allocate a new array rather than mutate in place.
_BYTES_PER_QUBIT_AMPLITUDE = 32


def _row_task_bytes(m: int, k: int, *, basis_multiplier: int = 1, n_spin_qubits: int = 0) -> int:
    """Estimated peak memory of ONE row's distribution build, for
    :func:`~circuit.spin.parallel_row_map`'s ``bytes_per_task`` -- computed from the actual
    ``(m, k)`` at the call site rather than a flat constant, since the outcome count this scales
    with is combinatorial in ``m``/``k`` (:func:`~circuit.fock.n_fock`), not a fixed size regardless
    of problem scale.

    ``basis_multiplier`` accounts for a prep whose basis is a fixed multiple of the plain
    ``m``-mode Fock basis (``spin_magic``'s ``m+2``-mode basis is exactly ``n_fock(m,k) * 2`` -- the
    ``m``-mode Fock support times its 2-way readout photon placement, per
    :func:`SpinMagicPrep._magic_basis`'s own docstring -- so pass ``basis_multiplier=2`` there, not
    a per-extra-mode exponent that would not match that prep's actual basis size).

    ``n_spin_qubits`` covers a SEPARATE cost the photonic outcome basis alone misses entirely:
    :class:`SpinPrep` builds a joint qubit state over ``k`` independently-prepared spin qubits
    BEFORE any photon ever reaches the interferometer -- :func:`circuit.spin.spin_state` allocates
    ``np.zeros(2**n_q, dtype=complex)`` (line ``psi = np.zeros(2 ** n_q, ...)``), exponential in
    qubit count, and this is genuinely additive to (not a proxy for, and not dominated by) the
    photonic ``n_fock(m,k)`` term, since the two live in different parts of one row's build.  Pass
    ``n_spin_qubits=k`` for :class:`SpinPrep` (its ``num_sources=k``, `circuit/prep.py`'s own
    ``HybridProcessor(num_sources=k, ...)`` call). **Not** relevant to :class:`SpinMagicPrep`: that
    prep reuses a SINGLE emitter sequentially (``num_sources=1``, a fixed ``2``-dimensional
    single-qubit state throughout, per its own module docstring's "single reused emitter"), so it
    carries no exponential-in-``k`` term at all -- leave ``n_spin_qubits=0`` (the default) there.
    """
    n_outcomes = n_fock(int(m), int(k)) * max(1, int(basis_multiplier))
    photonic_bytes = max(1, n_outcomes) * _BYTES_PER_OUTCOME
    qubit_bytes = (2 ** int(n_spin_qubits)) * _BYTES_PER_QUBIT_AMPLITUDE if n_spin_qubits else 0
    return photonic_bytes + qubit_bytes


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
        enc = build_encoding(encoding) if isinstance(encoding, str) else encoding
        return layer.forward(X, *enc.extra_inputs(X))

    def outcome_keys(self, *, m, k):
        from circuit.fock import fock_keys

        return fock_keys(m, k)


class SpinPrep(StatePrep):
    """Dual-rail photons emitted by ``k`` spin qubits (``perceval_spoqc``).

    ``layers`` rounds of, per qubit, ``H -> Rx(a) (-> Rz) -> Ry(b)``, then one CX chain per round,
    then emitted dual-rail into modes ``(2q, 2q+1)``.  The spin prep is built in numpy on the
    initial source state (:func:`v2.circuit.spin.spin_state`) because spoqc has no two-qubit
    processor gate and a CX on ``|0...0>`` is the identity.

    Absorbs three legacy classes as parameters: ``cx_pairs`` (``spoqc``), ``angle_levels``
    (``spoqc_low``, which drew the twists from ``+-0.1 .. +-0.1*levels`` instead of uniformly)
    and ``rz_angles="prime"`` (``spoqc_prime``, an extra ``Rz`` at increasing-prime radians
    between Rx and Ry).

    Two more knobs, both new:

    * ``layers`` -- depth of the seeded rotate-then-entangle block (default ``1``, the legacy
      single-round circuit).  Each layer draws its own ``rx``/``ry``/``rz`` from the same seeded
      stream rather than replaying layer 1's angles, so depth is genuine, not a repeated no-op --
      see :func:`v2.circuit.spin.spin_state`.
    * ``encode_on_spin`` -- when set, ``rx(x[2q]), ry(x[2q+1])`` (indices mod ``n_features``) are
      applied **once**, per qubit, on ``|0...0>`` before the first layer -- not per layer, since
      this carries ``x`` itself rather than a fixed seeded weight (see ``spin_state``'s docstring
      for why repeating it would inject copies of the same feature instead of adding depth).
    """

    name = "spin"
    is_batched = False

    def __init__(self, *, cx_pairs=None, angle_levels=None, rz_angles=None, layers=None,
                 encode_on_spin=False, encode_circuit=True):
        self.cx_pairs_raw = cx_pairs
        self.angle_levels = None if angle_levels is None else int(angle_levels)
        self.rz_angles = rz_angles
        self.layers = 1 if layers is None else int(layers)
        self.encode_on_spin = bool(encode_on_spin)
        #: Whether ``x`` reaches the interferometer at all.  ``True`` is the legacy/default
        #: behaviour (every existing ``spin`` dataset was generated this way); ``False`` zeroes
        #: ``x`` before :func:`~circuit.photonic_circuit.build_sandwich_circuit`, mirroring
        #: :class:`SpinMagicPrep`'s knob of the same name, so ``x`` reaches the outcome only
        #: through ``encode_on_spin`` (if that is also set).
        self.encode_circuit = bool(encode_circuit)
        #: Set by :meth:`validate` once ``k`` is known -- see there for why ``spec()`` reads
        #: this instead of re-resolving ``cx_pairs_raw`` itself.
        self._resolved_pairs: list[tuple[int, int]] | None = None
        if rz_angles not in (None, "prime"):
            raise ValueError(f"rz_angles must be None or 'prime' (got {rz_angles!r})")

    @classmethod
    def from_config(cls, cfg) -> "SpinPrep":
        m = cfg.model
        return cls(cx_pairs=m.cx_pairs, angle_levels=m.angle_levels, rz_angles=m.rz_angles,
                   layers=getattr(m, "layers", None),
                   encode_on_spin=bool(getattr(m, "encode_on_spin", False)),
                   encode_circuit=bool(getattr(m, "encode_circuit", None) is not False))

    def validate(self, *, m: int, k: int) -> None:
        if m % 2:
            raise ValueError("prep 'spin' emits dual-rail photons -> requires even m")
        if 2 * k > m:
            raise ValueError(f"need 2*k <= m for dual-rail emission (k={k}, m={m})")
        if self.layers < 1:
            raise ValueError(f"layers must be >= 1 (got {self.layers})")
        # Resolved here (k is always known by this point -- build_prep validates immediately
        # after construction) and cached, so spec() can report the concrete pairs a "chain"
        # request resolved to rather than the literal string, and two "chain" configs at
        # different k hash apart.
        self._resolved_pairs = _spin.normalize_cx_pairs(self.cx_pairs_raw, k)

    def _angles(self, k: int, seed: int):
        rng = np.random.default_rng(int(seed))
        shape = (self.layers, k)
        if self.angle_levels is None:
            rx = rng.uniform(0.0, 2 * np.pi, size=shape)
            ry = rng.uniform(0.0, 2 * np.pi, size=shape)
        else:
            rx = np.stack([_spin.discrete_angles(rng, k, self.angle_levels)
                          for _ in range(self.layers)])
            ry = np.stack([_spin.discrete_angles(rng, k, self.angle_levels)
                          for _ in range(self.layers)])
        rz = None
        if self.rz_angles == "prime":
            # One fixed round of increasing primes, broadcast identically to every layer -- the
            # primes are a fixed reference sequence (spoqc_prime), not something to redraw per
            # layer the way the seeded rx/ry twists are.
            rz = np.tile(np.asarray(_spin.first_primes(k), dtype=float), (self.layers, 1))
        return rx, ry, rz

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        import torch

        rx, ry, rz = self._angles(k, seed)
        pairs = _spin.normalize_cx_pairs(self.cx_pairs_raw, k)
        enc_name = getattr(encoding, "name", encoding)
        rows = X.detach().cpu().numpy()
        tasks = [(row, m, k, n_features, int(seed), rx, ry, pairs, rz, self.layers,
                  self.encode_on_spin, self.encode_circuit, enc_name)
                 for row in rows]
        results = _spin.parallel_row_map(_spin_row_worker, tasks, n_jobs,
                                         bytes_per_task=_row_task_bytes(m, k, n_spin_qubits=k))
        keys, probs = _align_rows(results, n_modes=m)
        self._keys = keys
        return torch.as_tensor(probs, dtype=torch.float32)

    def outcome_keys(self, *, m, k):
        return getattr(self, "_keys", None)

    def spec(self) -> dict:
        # Knobs are added only when set, so a config that leaves them at their defaults hashes
        # the same as one written before the knob existed.  Reads the pairs `validate` already
        # resolved (never re-derives from cx_pairs_raw here), so "chain" hashes as the concrete
        # ladder it resolved to -- two "chain" configs at different k hash apart, and the hash
        # reflects the actual circuit rather than the shorthand that produced it.
        spec = {"prep": self.name, "source": "H_Rx_Ry_seeded"}
        pairs = [[int(a), int(b)] for a, b in (self._resolved_pairs or [])]
        if pairs:
            spec["cx_pairs"] = pairs
        if self.angle_levels is not None:
            spec["angle_levels"] = self.angle_levels
        if self.rz_angles is not None:
            spec["rz_angles"] = self.rz_angles
        if self.layers != 1:
            spec["layers"] = self.layers
        if self.encode_on_spin:
            spec["encode_on_spin"] = self.encode_on_spin
        if not self.encode_circuit:
            spec["encode_circuit"] = self.encode_circuit
        return spec


class SpinMagicPrep(StatePrep):
    """Emitter-train cluster state, optionally with a non-Clifford gate injected into ``t_var``
    gaps.

    One spin emits ``k`` data photons dual-rail plus a readout photon.  Two independent axes:

    * ``structure`` -- the per-gap gate pattern applied to the spin *before every* emission:

      =============  ============================================================================
      ``structure``  per-gap gate(s), ``j = 0 .. k-1`` -- see :func:`_apply_structure_gate`
      =============  ============================================================================
      ``linear``     branch gate at ``phi=0``, every gap          (default; legacy behaviour)
      ``ghz``        branch gate at ``phi=pi``, every gap
      ``linear_u3``  ``linear``'s branch gate then a Haar-random ``U3``, every gap
      =============  ============================================================================

      Both ``linear`` and ``ghz`` re-apply the branch gate every gap -- they differ only in
      ``phi``, not in cadence.  ``phi=0`` reproduces a linear cluster state (each gap extends the
      chain); ``phi=pi`` instead gives a caterpillar/GHZ connection (each gap's emission inherits
      correlation from the same shared branch rather than extending it).  ``linear_u3`` adds a
      fixed Haar-random single-qubit rotation to every gap on top of the ``linear`` branch gate,
      independent of whether a magic gate is also injected.

    * ``gate_kind``/``t_var`` -- an *additional*, sparser non-Clifford gate injected into the
      first ``t_var`` gaps on top of whichever ``structure`` is chosen; absorbs the four legacy
      ``spoqc_magic*`` classes as before (``"t"`` | ``"rz"`` | ``"u3"`` | ``"u3_x"``, optionally
      ``_m``/``_s`` angle-scaled) -- see :func:`v2.circuit.spin.apply_gap_gate`.  ``t_var=0``
      (or a ``structure`` that already injects its own ``U3``) makes this axis inert, so
      ``linear_u3`` is not a special case of the gate-injection machinery -- it is orthogonal to
      it, and the two compose (both may fire in the same gap).

    Data encoding is two more independent flags, replacing the legacy ``gate_kind`` suffix
    parsing (``_rxry``, ``_rxry_iface``) with explicit booleans -- the suffix form still works,
    for configs that set it, but ``encode_on_spin``/``encode_circuit`` take precedence when given:

    * ``encode_on_spin`` -- ``rx(x[2j]), ry(x[2j+1])`` (indices mod ``n_features``) on the spin
      after the gap's structural gate(s), before that gap's emission.  Composes with every
      ``structure`` identically: the branch gate fires every gap regardless of ``structure``, so
      every ``encode_on_spin`` gap rotates a state that was just re-branched.
    * ``encode_circuit`` -- whether ``x`` reaches the interferometer at all.  ``False`` zeroes
      ``x`` before :func:`~v2.circuit.photonic_circuit.build_sandwich_circuit`, so the encoding
      becomes the identity there (a fixed Haar scrambler with no data), leaving ``x`` reaching
      the outcome only through ``encode_on_spin``, if that is also set.

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
    DEFAULT_STRUCTURE = "linear"
    STRUCTURES = ("linear", "ghz", "linear_u3")

    def __init__(self, *, t_var=None, gate_kind=None, structure=None,
                 encode_on_spin=None, encode_circuit=None):
        self.t_var = t_var
        self.gate_kind = self.DEFAULT_GATE_KIND if gate_kind is None else str(gate_kind)
        self.structure = self.DEFAULT_STRUCTURE if structure is None else str(structure)
        #: ``None`` means "read it from gate_kind's legacy suffix" (see :meth:`_resolved_encoding`);
        #: an explicit ``True``/``False`` here always wins, so old configs keep working unchanged
        #: while new code sets these directly instead of encoding them into the ``gate_kind`` string.
        self.encode_on_spin = encode_on_spin
        self.encode_circuit = encode_circuit

    @classmethod
    def from_config(cls, cfg) -> "SpinMagicPrep":
        return cls(t_var=cfg.model.t_var, gate_kind=cfg.model.gate_kind,
                   structure=getattr(cfg.model, "structure", None),
                   encode_on_spin=getattr(cfg.model, "encode_on_spin", None),
                   encode_circuit=getattr(cfg.model, "encode_circuit", None))

    def resolved_t_var(self, k: int) -> int:
        return self.DEFAULT_T_VAR if self.t_var is None else int(self.t_var)

    def _resolved_encoding(self):
        """``(encode_on_spin, encode_circuit)``, explicit flags winning over the legacy
        ``gate_kind`` suffix (``_rxry`` -> spin only, ``_rxry_iface`` -> spin and circuit, no
        suffix -> circuit only)."""
        _, suffix_qubit, suffix_iface = _spin.parse_gate_kind(self.gate_kind)
        on_spin = suffix_qubit if self.encode_on_spin is None else bool(self.encode_on_spin)
        on_circuit = suffix_iface if self.encode_circuit is None else bool(self.encode_circuit)
        return on_spin, on_circuit

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
        if self.structure not in self.STRUCTURES:
            raise ValueError(f"structure must be one of {self.STRUCTURES} (got {self.structure!r})")
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

    def _structure_u3_params(self, k: int, seed: int):
        """Per-gap Haar ``U3`` angles for ``structure='linear_u3'``, seeded independently of the
        ``gate_kind`` magic-gate draw (a different, offset seed) so the two axes' random draws
        never collide even when both are Haar ``U3``."""
        if self.structure != "linear_u3":
            return None
        rng = np.random.default_rng(int(seed) + 1_000_003)      # offset: distinct from _gate_params
        return _spin.haar_su2_angles(rng, k)

    def probs(self, X, *, m, k, n_features, seed, encoding, n_jobs=1):
        import torch

        t_var = self.resolved_t_var(k)
        params = self._gate_params(k, seed)
        structure_params = self._structure_u3_params(k, seed)
        encode_on_spin, encode_circuit = self._resolved_encoding()
        enc_name = getattr(encoding, "name", encoding)
        rows = X.detach().cpu().numpy()
        tasks = [(row, m, k, t_var, n_features, int(seed), self.gate_kind, params,
                  self.structure, structure_params, encode_on_spin, encode_circuit, enc_name)
                 for row in rows]
        results = _spin.parallel_row_map(_magic_row_worker, tasks, n_jobs,
                                         bytes_per_task=_row_task_bytes(m, k, basis_multiplier=2))
        keys, probs = _align_rows_fixed(results, basis=_magic_basis(m, k))
        self._keys = keys
        return torch.as_tensor(probs, dtype=torch.float32)

    def outcome_keys(self, *, m, k):
        # Fixed in advance (see :func:`_magic_basis`), not discovered per call -- so two separate
        # `probs` calls at the same (m, k) always align to the same columns without needing to be
        # routed through one shared batch.
        return _magic_basis(m, k)

    def spec(self) -> dict:
        encode_on_spin, encode_circuit = self._resolved_encoding()
        return {"prep": self.name, "gate_kind": self.gate_kind, "t_var": self.t_var,
                "structure": self.structure, "encode_on_spin": encode_on_spin,
                "encode_circuit": encode_circuit, "gadget": "emitter_train_readout_pair"}


# --- picklable process-pool workers -------------------------------------------------------- #
#
# Module-level, tuple-argument: required by ``parallel_row_map``'s process pool.


def _spin_row_worker(task):
    """One row of :class:`SpinPrep`: build the processor, return its full distribution."""
    (row, m, k, n_features, seed, rx, ry, pairs, rz, layers, encode_on_spin, encode_circuit,
     enc_name) = task
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    from .photonic_circuit import build_sandwich_circuit

    data_angles = None
    if encode_on_spin:
        # (k, 2): rx(x[2q]), ry(x[2q+1]) per qubit, indices mod n_features -- same convention as
        # SpinMagicPrep's encode_on_spin (see its module docstring).
        data_angles = np.array(
            [[float(row[(2 * q) % n_features]), float(row[(2 * q + 1) % n_features])]
             for q in range(k)])

    p = HybridProcessor(num_sources=k, num_modes=m)
    p.with_initial_source_state(
        _spin.spin_state(k, rx, ry, pairs, rz=rz, layers=layers, data_angles=data_angles))
    for q in range(k):
        p.emit(q, into=(2 * q, 2 * q + 1))
    # A zeroed x leaves the Haar scrambler but puts no data in the interferometer -- the
    # circuit-encoding-off case, matching SpinMagicPrep's identical iface_x branch.
    iface_x = row if encode_circuit else np.zeros_like(np.asarray(row, dtype=float))
    p.add(0, build_sandwich_circuit(m, n_features, seed, enc_name, x=iface_x))
    for mode in range(m):
        p.add(mode, Detector())
    return _spin.full_distribution(p, m)


def _apply_structure_gate(p, *, phi: float) -> None:
    """``Ry(pi/2) . Rz(phi) . Ry(pi/2)`` on the spin -- the branch gate that distinguishes a
    linear cluster (``phi=0``) from a GHZ/caterpillar branch (``phi=pi``); not a Hadamard, despite
    an earlier version of this code using ``H`` here."""
    p.gate.ry(0, np.pi / 2)
    p.gate.rz(0, float(phi))
    p.gate.ry(0, np.pi / 2)


def _magic_row_worker(task):
    """One row of :class:`SpinMagicPrep`: emitter train + gap gate(s) + readout pair.

    Three independent per-gap ingredients, each optional and composable: the ``structure`` gate
    (:func:`_apply_structure_gate`, ``ghz``'s first-gap-only version, or that plus an unconditional
    Haar ``U3``), the sparser ``gate_kind`` magic gate (``j < t_var`` only), and the data-on-spin
    ``rx``/``ry`` pair.  Order within a gap is structure, then data, then magic gate, then emission
    -- so a magic gate always acts on whatever the structure/data steps just prepared, matching the
    legacy ``spoqc_magic*`` order (structure gate then data then gate) with ``structure``'s extra
    ``U3`` slotted in next to it.
    """
    (row, m, k, t_var, n_features, seed, gate_kind, params, structure, structure_params,
     encode_on_spin, encode_circuit, enc_name) = task
    from perceval import Detector
    from perceval_spoqc import HybridProcessor

    from .photonic_circuit import build_sandwich_circuit

    magic_kind, _, _ = _spin.parse_gate_kind(gate_kind)
    r0, r1 = m, m + 1
    p = HybridProcessor(num_sources=1, num_modes=m + 2, num_records=m + 2,
                        allow_carry_over=True)
    zero = np.array([1.0, 0.0], dtype=complex)
    p.with_initial_source_state(np.outer(zero, zero.conj()))

    for j in range(k):
        # caterpillar/GHZ (phi=pi) vs linear (phi=0) -- every gap, both branches re-apply it.
        _apply_structure_gate(p, phi=np.pi if structure == "ghz" else 0.0)
        if structure == "linear_u3":
            theta, phi, lam = structure_params[j]           # ZYZ Euler, as in apply_gap_gate's u3
            p.gate.rz(0, float(lam))
            p.gate.ry(0, float(theta))
            p.gate.rz(0, float(phi))
        if encode_on_spin:                                 # data on the spin: 2 features per gap
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
    # circuit-encoding-off case; PS(0) is the identity.
    iface_x = row if encode_circuit else np.zeros_like(np.asarray(row, dtype=float))
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


def _magic_basis(m: int, k: int) -> list[tuple[int, ...]]:
    """The fixed ``spin_magic`` outcome basis: ``m``-mode ``k``-photon Fock keys, each paired with
    both dual-rail readout outcomes -- ``n_fock(m, k) * 2`` keys total, ``m + 2`` long.

    Unlike :func:`_align_rows`, this basis does not depend on what any particular batch's perceval
    calls happened to report: it is the same combinatorial enumeration :func:`circuit.fock.fock_keys`
    gives every other Fock-basis model, crossed with the readout pair's two outcomes
    (:func:`circuit.fock.binary_keys`). Fixing it here is what lets two separate ``probs`` calls
    (e.g. :func:`eval.sweep_delta.probs_batch`'s ``x0`` and ``X_ball`` project) share one basis
    without needing to route every call through a single batched union.
    """
    return [data + readout for data in fock_keys(m, k) for readout in binary_keys(2)]


def _align_rows_fixed(results, *, basis: list[tuple[int, ...]]):
    """Scatter per-row ``(keys, probs)`` onto the fixed ``basis``, zero-filling unreported outcomes.

    Like :func:`_align_rows`, perceval prunes negligible-probability outcomes and *which* ones it
    prunes can differ per row -- but here the destination columns are fixed in advance rather than
    discovered from the batch, so two different batches (or two different single-row calls) always
    align to the same columns. Returns ``(keys, probs)`` with ``keys`` a ``(n_states, n_modes)`` int
    array (``basis``, unchanged) and ``probs`` ``(n_rows, n_states)``.
    """
    index = {key: i for i, key in enumerate(basis)}
    out = np.zeros((len(results), len(basis)), dtype=np.float64)
    for i, (keys, probs) in enumerate(results):
        for row, pr in zip(keys, probs):
            t = tuple(int(v) for v in row)
            if t not in index:
                raise ValueError(
                    f"perceval reported outcome {t} ({sum(t)} photons) outside the fixed "
                    f"spin_magic basis ({len(basis)} outcomes over {len(basis[0])} modes) -- "
                    "the circuit no longer conserves photon number as assumed."
                )
            out[i, index[t]] = pr
    return np.asarray(basis, dtype=np.int16), out


def build_prep(cfg) -> StatePrep:
    """Instantiate the prep named by ``cfg.model.prep`` and validate its geometry."""
    name = cfg.model.prep
    if name not in PREPS:
        raise ValueError(f"unknown prep {name!r}; registered: {sorted(PREPS)}")
    prep = PREPS[name].from_config(cfg)
    prep.validate(m=cfg.problem.m, k=cfg.problem.k)
    return prep
