"""How the input ``x`` enters the circuit: an open registry of encodings.

An *encoding* adds the ``x``-dependent components to a circuit and declares its own geometry
constraint.  Deliberately open-ended: the legacy pipeline hard-coded one phase shifter per mode
on the first ``n_features`` modes inside ``build_sandwich_circuit``, so there was no way to try
another scheme without editing every model.  Here the model takes an encoding by name.

Four ship: ``phase`` (legacy behaviour, the default), ``bs`` (a beamsplitter ladder), ``bs_phase``
(beamsplitter + phase shifter per feature).  The abstraction exists so that data re-uploading,
MZI-angle encodings, scaled/multi-frequency encodings or a dense mixing encoding drop in as
modules without touching any model.  Note the *state preparation* is a separate axis
(:mod:`v2.circuit.prep`) -- a spin-prepared photonic circuit is the same ``phase`` encoding with
``prep="spin"``.

**Diagonal vs. general encodings, and the two analytic hooks.**  ``phase`` is diagonal in the mode
basis (each feature only ever multiplies its own mode by a phase), so the pure-torch ``det``/
``Perm`` readouts (:mod:`v2.model.fermion`, :func:`~v2.model.fermion.boson_probs_reference`) can
treat it as ``diag(e^{i x_0}, .., e^{i x_{n-1}}, 1, .., 1)`` via :meth:`Encoding.phases`, which is
cheap (``O(m)`` per row) and is what :func:`~v2.circuit.photonic_circuit.sandwich_unitary_at` uses
when available.  A beamsplitter mixes amplitude *between* two modes, so it has no diagonal form;
those encodings instead implement :meth:`Encoding.unitary`, returning the full ``(N, m, m)``
per-row unitary the encoding contributes, and ``sandwich_unitary_at`` falls back to a batched
matrix product when ``phases`` is not implemented.  Every encoding must implement at least one of
the two; implementing both is never required.

Adding one: subclass :class:`Encoding` with a ``name`` and it auto-registers.
"""

from __future__ import annotations

import math

#: name -> Encoding subclass, populated on subclassing.
ENCODINGS: dict[str, type["Encoding"]] = {}


class Encoding:
    """Adds the ``x``-dependent components to a circuit, and validates its own geometry."""

    name: str | None = None

    #: Extra named ``input_parameters`` groups (beyond the default ``"x"``) this encoding needs
    #: bound into a merlin ``QuantumLayer`` -- e.g. ``["y"]`` for a derived-feature product that
    #: perceval cannot compute from two Parameters internally (see :meth:`extra_inputs` and
    #: :class:`HavlicekEncoding`). Empty for every diagonal/single-feature encoding.
    extra_input_names: tuple[str, ...] = ()

    def extra_input_widths(self, n_features: int) -> list[int]:
        """Column width of each group in :attr:`extra_input_names`, in order.

        Defaults to ``n_features`` per group (one value per feature, the common case). A group
        carrying one value per feature *pair* (:class:`HavlicekEncoding`'s ``b``/``d``) overrides
        this to report ``n_features * (n_features - 1) // 2`` instead, so
        :func:`~circuit.photonic_circuit.build_quantum_layer` can size ``input_size`` correctly.
        """
        return [n_features] * len(self.extra_input_names)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None):
            ENCODINGS[cls.name] = cls

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        """Raise if this encoding cannot represent ``n_features`` inputs on ``m`` modes.

        This is where the legacy global ``n_features <= m - 1`` check belongs: it is a property
        of the encoding, not of the problem.  A dense/mixed encoding would not need it at all.
        """
        return None

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        """Add the ``x``-dependent components to a perceval ``circuit`` in place.

        ``parameterised=True`` adds free parameters named ``x0..x{n-1}`` (what a merlin
        ``QuantumLayer`` binds to its input); ``False`` is used by the per-row perceval paths,
        which build a concrete circuit per sample via :meth:`add_concrete`.
        """
        raise NotImplementedError

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        """Add the encoding with *concrete* numeric values from ``x`` (per-row perceval paths)."""
        raise NotImplementedError

    def phases(self, X, *, m: int, n_features: int):
        """``(N, m)`` complex per-mode phase factors, for the pure-torch analytic paths.

        The fast, diagonal-only hook: valid only for encodings that are diagonal in the mode
        basis.  Encodings that mix modes (a beamsplitter) cannot implement this and should leave
        it raising -- implement :meth:`unitary` instead.  :func:`~v2.circuit.photonic_circuit.
        sandwich_unitary_at` tries this first and falls back to :meth:`unitary` when it raises.
        """
        raise NotImplementedError

    def unitary(self, X, *, m: int, n_features: int):
        """``(N, m, m)`` complex unitary this encoding contributes, for non-diagonal encodings.

        The general analytic hook: any encoding, diagonal or not, can be expressed this way, so
        this is what the ``det``/``Perm`` readouts fall back to when :meth:`phases` is not
        implemented.  Diagonal encodings should prefer :meth:`phases` (an ``O(m)`` diagonal beats
        an ``O(m^2)`` dense matrix per row) and need not override this.
        """
        raise NotImplementedError

    def extra_inputs(self, X):
        """``(N, n_features)`` tensors, one per name in :attr:`extra_input_names`, in order.

        Computed from ``X`` (the model's real, stored input -- this method derives from it, it
        does not replace it) right before a merlin ``QuantumLayer`` call, so a merlin-bound
        product like :class:`HavlicekEncoding`'s ``x_i * x_{(i+1) mod n}`` can be supplied as a
        second named parameter group without perceval needing to multiply two Parameters
        internally (see :meth:`add_to`'s per-encoding note on why that is not a circuit
        primitive).  Empty for every encoding with no ``extra_input_names``.
        """
        return []

    def spec(self) -> dict:
        """Identity fields folded into the artifact hash.

        ``self.name`` alone is enough to keep every encoding's datasets from colliding -- see
        :mod:`v2.model.photonic` and :mod:`v2.model.fermion`, whose ``circuit_spec`` both spread
        this in, and :mod:`v2.pipeline.artifact`, which hashes ``circuit_spec``.  A subclass with
        its own tunable knobs (an angle scale, a mode-pairing choice) should extend this rather
        than replace it, so two differently-configured instances of the *same* encoding also hash
        apart.
        """
        return {"encoding": self.name}


class PhaseEncoding(Encoding):
    """One phase shifter ``PS(x_i)`` on mode ``i``, for ``i < n_features``.

    The legacy behaviour, kept as the default so ``v2`` reproduces existing datasets.  Diagonal
    in the mode basis, so ``phases`` is available and the analytic ``det`` / ``Perm`` readouts
    work.  Needs one mode per feature, hence ``n_features <= m``.
    """

    name = "phase"

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        if n_features > m:
            raise ValueError(
                f"encoding 'phase' puts one phase shifter per feature on its own mode, so it "
                f"needs n_features <= m (got n_features={n_features}, m={m}). Raise m, or use "
                f"an encoding that mixes features across modes."
            )

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        import perceval as pcvl

        for i in range(n_features):
            circuit.add(i, pcvl.PS(pcvl.P(f"x{i}")))

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        import perceval as pcvl

        for i in range(n_features):
            circuit.add(i, pcvl.PS(float(x[i])))

    def phases(self, X, *, m: int, n_features: int):
        """``diag(e^{i x_0}, .., e^{i x_{n_features-1}}, 1, .., 1)`` as an ``(N, m)`` tensor."""
        import torch

        ph = torch.ones(X.shape[0], m, dtype=torch.complex64)
        ph[:, :n_features] = torch.exp(1j * X[:, :n_features].to(torch.complex64))
        return ph


class RepeatedPhaseEncoding(Encoding):
    """``L`` repeated ``phase`` blocks (data re-uploading): ``PS(x) -> W -> PS(x) -> W -> ... ->
    PS(x)``, ``L-1`` internal Haar unitaries ``W`` interleaved between ``L`` phase-shifter layers,
    all inside this one :meth:`add_to`/:meth:`add_concrete` call.

    Registered as ``phase2`` through ``phase10`` (:data:`REPEATED_PHASE_DEPTHS`), each a fixed
    ``L``.  ``phase1`` is not registered separately -- it would be identical to plain ``phase``,
    which already exists.

    **Why this needs no change to** :mod:`circuit.photonic_circuit`.  ``build_sandwich_circuit``
    seeds ``torch.manual_seed``/``pcvl.random_seed`` once, draws its own first Haar unitary
    (``W1``), then calls ``encoding.add_to(circuit, ...)`` -- still inside that same seeded
    stream -- before drawing its own final unitary.  This class's ``add_to`` draws its ``L-1``
    internal unitaries with plain ``pcvl.Matrix.random_unitary(m)`` calls from right there, so
    they continue the *same* stream ``W1`` came from, and the sandwich's own final draw becomes
    this encoding's ``L``-th (last) unitary -- giving ``L+1`` total Haar unitaries end to end
    (``W1`` from the sandwich, ``L-1`` internal to this class, ``W_{L+1}`` from the sandwich again)
    with no seed threaded through explicitly.  A depth-``L`` and depth-``L+1`` run therefore share
    their first ``L+1`` unitaries exactly, by construction of the shared stream -- not by any
    bookkeeping this class does itself.

    **Photonic (boson-sampling / ``model.kind=photonic``) only.**  The analytic ``det``/``Perm``
    readouts (:func:`~circuit.photonic_circuit.sandwich_unitaries`/``sandwich_unitary_at``, used by
    ``model.kind=fermion``) draw only two unitaries total from the seed and hand them to the
    encoding as fixed ``W1``/``W2`` arguments -- there is no hook for an encoding to draw further
    internal unitaries of its own on that path, so :meth:`unitary` raises rather than silently
    running a shorter, wrong circuit.  Every merlin/boson-sampling path (:func:`~circuit.
    photonic_circuit.build_quantum_layer`, hence :class:`~model.photonic.PhotonicModel`) works
    unmodified, since it only ever goes through :meth:`add_to`/:meth:`add_concrete`.
    """

    n_layers: int = 2

    def __init_subclass__(cls, **kwargs):
        # Each depth needs its own class (not one shared instance) so ENCODINGS[f"phase{L}"]
        # resolves to a distinct n_layers -- registration itself is unchanged (Encoding's own
        # __init_subclass__, invoked via super() below).
        super().__init_subclass__(**kwargs)

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        if n_features > m:
            raise ValueError(
                f"encoding {self.name!r} puts one phase shifter per feature on its own mode "
                f"(same geometry as 'phase', repeated), so it needs n_features <= m (got "
                f"n_features={n_features}, m={m}). Raise m, or lower n_features."
            )

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        import perceval as pcvl

        # One Parameter object per feature, reused (not re-created) at every layer -- data
        # re-uploading means the SAME x_i drives every occurrence, and perceval rejects two
        # distinct Parameter objects sharing one name ("two parameters with the same name in the
        # circuit"), so the object itself, not just the name string, must be shared.
        params = [pcvl.P(f"x{i}") for i in range(n_features)]
        for layer in range(self.n_layers):
            for i in range(n_features):
                circuit.add(i, pcvl.PS(params[i]))
            if layer < self.n_layers - 1:
                circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        import perceval as pcvl

        for layer in range(self.n_layers):
            for i in range(n_features):
                circuit.add(i, pcvl.PS(float(x[i])))
            if layer < self.n_layers - 1:
                circuit.add(0, pcvl.Unitary(pcvl.Matrix.random_unitary(m)), merge=True)

    def unitary(self, X, *, m: int, n_features: int):
        raise NotImplementedError(
            f"encoding {self.name!r} draws its own internal Haar unitaries inside add_to/"
            "add_concrete, which the analytic det/Perm path (sandwich_unitaries/"
            "sandwich_unitary_at, used by model.kind='fermion') has no hook for -- it only ever "
            "draws two fixed unitaries from the seed. Use model.kind='photonic' (the merlin/"
            "boson-sampling path), which calls add_to directly and needs no such hook."
        )

    def spec(self) -> dict:
        return {"encoding": self.name, "n_layers": self.n_layers}


#: phase2..phase10 -- one class per depth, registered by name. phase1 is not registered
#: separately since it would duplicate plain PhaseEncoding ("phase").
REPEATED_PHASE_DEPTHS = range(2, 11)
for _L in REPEATED_PHASE_DEPTHS:
    globals()[f"_RepeatedPhaseEncoding{_L}"] = type(
        f"RepeatedPhaseEncoding{_L}", (RepeatedPhaseEncoding,),
        {"name": f"phase{_L}", "n_layers": _L},
    )
del _L


def _bs_block(theta):
    """``(N, 2, 2)`` beamsplitter unitary at angle(s) ``theta``, perceval's ``BS`` (``Rx``)
    convention: ``[[cos(t/2), i sin(t/2)], [i sin(t/2), cos(t/2)]]``.  Verified against
    ``perceval.BS(theta=t).compute_unitary()`` to float32 round-off.
    """
    import torch

    half = theta.to(torch.complex64) / 2
    c, s = torch.cos(half), 1j * torch.sin(half)
    row0 = torch.stack([c, s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _bs_ladder_unitary(X, *, m: int, n_features: int):
    """``(N, m, m)`` -- one ``BS(x_i)`` on modes ``(i, i+1)`` for ``i < n_features``, composed in
    add-order (mode ``i`` added before mode ``i+1``, later components left-multiplying the running
    unitary -- perceval's own composition order; see :func:`BeamsplitterEncoding.add_to`).
    """
    import torch

    N = X.shape[0]
    U = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
    for i in range(n_features):
        block = _bs_block(X[:, i])                    # (N, 2, 2)
        step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
        step[:, i:i + 2, i:i + 2] = block
        U = torch.einsum("nij,njk->nik", step, U)
    return U


class BeamsplitterEncoding(Encoding):
    """One beamsplitter ``BS(theta=x_i)`` on modes ``(i, i+1)``, for ``i < n_features``.

    Mixes amplitude between adjacent modes instead of phase-shifting one mode alone, so it is
    *not* diagonal in the mode basis: :meth:`phases` is not implemented, and the analytic
    ``det``/``Perm`` readouts go through :meth:`unitary` instead (see the module docstring).  The
    ladder shares a mode between consecutive features (``BS(x_0)`` on ``(0,1)``, ``BS(x_1)`` on
    ``(1,2)``, ...), so it needs one more mode than feature, hence ``n_features <= m - 1`` --
    one more than ``phase``'s ``n_features <= m``, since here the last feature's second mode has
    nowhere further to go.
    """

    name = "bs"

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        if n_features > m - 1:
            raise ValueError(
                f"encoding 'bs' puts one beamsplitter per feature on modes (i, i+1), so it needs "
                f"n_features <= m - 1 (got n_features={n_features}, m={m}). Raise m, or lower "
                f"n_features."
            )

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        import perceval as pcvl

        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=pcvl.P(f"x{i}")))

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        import perceval as pcvl

        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=float(x[i])))

    def unitary(self, X, *, m: int, n_features: int):
        return _bs_ladder_unitary(X, m=m, n_features=n_features)


class BeamsplitterPhaseEncoding(Encoding):
    """``BS(theta=x_i)`` on modes ``(i, i+1)`` then ``PS(x_i)`` on mode ``i``, per feature.

    The same beamsplitter ladder as :class:`BeamsplitterEncoding`, with a phase shifter on the
    ladder's leading mode driven by the *same* ``x_i`` -- one encoded parameter reaches the circuit
    through two component types rather than one, without adding a second input per feature.  Also
    not diagonal (the ``BS`` half mixes modes), so :meth:`unitary` is the analytic hook, composing
    the ``PS`` after its ``BS`` -- matching :meth:`add_to`'s order, in which the phase shifter is
    added second and so left-multiplies the running unitary, exactly as perceval would build it.
    Same geometry constraint as ``bs``: ``n_features <= m - 1``.
    """

    name = "bs_phase"

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        if n_features > m - 1:
            raise ValueError(
                f"encoding 'bs_phase' puts one beamsplitter per feature on modes (i, i+1), so it "
                f"needs n_features <= m - 1 (got n_features={n_features}, m={m}). Raise m, or "
                f"lower n_features."
            )

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        import perceval as pcvl

        for i in range(n_features):
            # The same Parameter object on both components -- not two Parameters sharing a name,
            # which perceval rejects ("two parameters with the same name in the circuit") -- so
            # one merlin input x_i drives both the BS angle and the PS angle together.
            p = pcvl.P(f"x{i}")
            circuit.add(i, pcvl.BS(theta=p))
            circuit.add(i, pcvl.PS(p))

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        import perceval as pcvl

        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=float(x[i])))
            circuit.add(i, pcvl.PS(float(x[i])))

    def unitary(self, X, *, m: int, n_features: int):
        import torch

        N = X.shape[0]
        U = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
        for i in range(n_features):
            bs_step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
            bs_step[:, i:i + 2, i:i + 2] = _bs_block(X[:, i])
            U = torch.einsum("nij,njk->nik", bs_step, U)

            ps_step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
            ps_step[:, i, i] = torch.exp(1j * X[:, i].to(torch.complex64))
            U = torch.einsum("nij,njk->nik", ps_step, U)
        return U


def _bs_ladder_fixed_unitary(m: int, n_features: int, N: int) -> torch.Tensor:
    """``(N, m, m)`` -- one fixed ``BS(pi/2)`` on modes ``(i, i+1)`` for ``i < n_features``,
    composed in the same add-order as :func:`_bs_ladder_unitary` (data-independent, so every row
    is identical, but broadcast to ``N`` for uniform composition with the data-carrying layers).
    """
    import torch

    theta = torch.full((N,), math.pi / 2, dtype=torch.float32)
    U = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
    for i in range(n_features):
        block = _bs_block(theta)
        step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
        step[:, i:i + 2, i:i + 2] = block
        U = torch.einsum("nij,njk->nik", step, U)
    return U


def _bs_ladder_pairwise_unitary(X: torch.Tensor, *, m: int, n_features: int) -> torch.Tensor:
    """``(N, m, m)`` -- one ``BS(x_i * x_j)`` on modes ``(i, j)``, for every ``i < j <
    n_features``.  The pairwise term, matching IQP's full ``sum_{i<j}`` rather than only adjacent
    pairs: a beamsplitter is a genuine two-mode gate, so its ``2x2`` block is embedded directly at
    rows/columns ``(i, j)`` of the running unitary -- ``i`` and ``j`` need not be adjacent, since
    this is the analytic matrix path with no physical adjacency constraint (unlike a perceval
    circuit, where every component acts on physically adjacent modes; see :meth:`add_to`).
    """
    import torch

    N = X.shape[0]
    U = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
    for i in range(n_features):
        for j in range(i + 1, n_features):
            theta = X[:, i] * X[:, j]
            block = _bs_block(theta)
            step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
            step[:, i, i] = block[:, 0, 0]
            step[:, i, j] = block[:, 0, 1]
            step[:, j, i] = block[:, 1, 0]
            step[:, j, j] = block[:, 1, 1]
            U = torch.einsum("nij,njk->nik", step, U)
    return U


def _pair_indices(n_features: int) -> list[tuple[int, int]]:
    """``[(0,1), (0,2), .., (n-2,n-1)]`` -- every ``i < j`` pair, in a fixed order shared by
    :meth:`HavlicekEncoding.add_to`/``add_concrete``/``extra_inputs`` so a pair's index into
    ``pairs`` always lines up with its column in the ``b``/``d`` input groups.
    """
    return [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]


def _route_pair(m: int, i: int, j: int) -> list[int]:
    """A permutation sending mode ``i`` to port ``0`` and mode ``j`` to port ``1``, every other
    mode filling the remaining ports in order -- so a plain adjacent ``BS`` on ports ``(0, 1)``
    after this permutation couples the original, possibly non-adjacent, modes ``i`` and ``j``.

    ``perm[k]`` is perceval's ``PERM`` convention: the output port that input mode ``k``'s content
    is routed to (verified against ``PERM([1,2,3,4,0]).compute_unitary()``, which sends input ``k``
    to output ``(k+1) mod 5``, i.e. ``U[out, in]`` is ``1`` exactly at ``out = perm[in]``).
    """
    rest = [k for k in range(m) if k not in (i, j)]
    perm = [0] * m
    perm[i], perm[j] = 0, 1
    for offset, k in enumerate(rest):
        perm[k] = 2 + offset
    return perm


def _add_routed_bs(circuit, m: int, i: int, j: int, bs) -> None:
    """Add ``bs`` (a 2-mode component) coupling modes ``i`` and ``j``, routing them adjacent via
    :func:`_route_pair` first and back afterward when they are not already adjacent.  A plain
    adjacent add when ``j == i + 1`` avoids two no-op ``PERM`` layers in the common case (the
    ladder-style ``BS(pi/2)`` layers and most ``i < j`` pairs at small ``n_features``).
    """
    import perceval as pcvl

    if j == i + 1:
        circuit.add(i, bs)
        return
    perm = _route_pair(m, i, j)
    inv_perm = [0] * m
    for k in range(m):
        inv_perm[perm[k]] = k
    circuit.add(0, pcvl.PERM(perm))
    circuit.add((0, 1), bs)
    circuit.add(0, pcvl.PERM(inv_perm))


class HavlicekEncoding(Encoding):
    """The photonic analogue of the qubit IQP feature map's ``H -> U_phi(x) -> H -> U_phi(x)``
    sandwich, mode-for-mode:

    .. math::
        \\mathrm{BS}(\\pi/2) \\to \\mathrm{PS}(x) \\to \\mathrm{BS}(x^2)
        \\to \\mathrm{BS}(\\pi/2) \\to \\mathrm{PS}(x) \\to \\mathrm{BS}(x^2)
        \\to \\mathrm{BS}(\\pi/2)

    Three fixed ``BS(pi/2)`` layers (the balanced 50:50 mixer -- the photonic analogue of the
    qubit Hadamard, since ``BS(pi/2)`` is ``[[1, i], [i, 1]] / sqrt(2)`` -- applied ladder-style,
    one per adjacent mode pair) sandwich two data layers, each itself split into a single-mode
    phase ``PS(x_i)`` (IQP's linear ``x_i Z_i`` term) followed by a two-mode ``BS(x_i * x_j)`` for
    *every* pair ``i < j`` (IQP's full pairwise ``sum_{i<j} Z_i Z_j`` term -- linear optics has no
    direct two-mode *phase* gate, so the pairwise coupling is carried by a beamsplitter, a genuine
    two-mode primitive, rather than an artificial product-phase gate).  A beamsplitter only acts on
    physically adjacent modes, so each non-adjacent pair ``(i, j)`` is realised as ``PERM -> BS ->
    PERM``: route ``i`` and ``j`` to adjacent ports, apply the coupling there, then permute back --
    everything else passes through the permutation unchanged (see :func:`_route_pair`).

    Same geometry as ``bs``/``bs_phase``: needs one more mode than feature, hence
    ``n_features <= m - 1``.  Not diagonal (every layer but ``PS`` mixes modes), so :meth:`unitary`
    is the analytic hook.

    ``BS(x_i * x_j)`` needs a *product* of two features, which perceval's Parameter algebra cannot
    express directly (it composes Parameters additively/affinely, not by multiplying two distinct
    free Parameters together) -- so :meth:`add_to` cannot bind it to raw ``x`` the way
    ``phase``/``bs``/``bs_phase`` do.  Instead every data value the circuit needs is precomputed
    outside perceval and supplied as its own named input group (:attr:`extra_input_names`,
    :meth:`extra_inputs`): merlin's ``spec_mappings`` keys parameters by a plain *string-prefix*
    match against the group name, so the four data layers get single-letter, mutually-non-prefixing
    group names -- ``x`` (the real input, first linear layer, width ``n_features``), ``b`` (first
    pairwise layer, one value per ``i < j`` pair, width ``n_features * (n_features - 1) / 2``),
    ``c`` (second linear layer, same values as ``x``), ``d`` (second pairwise layer, same values as
    ``b``) -- fed the *same* numbers twice over: physically the same encoding applied twice,
    mechanically four independent input groups.
    """

    name = "havlicek"
    extra_input_names = ("b", "c", "d")

    def extra_input_widths(self, n_features: int) -> list[int]:
        n_pairs = n_features * (n_features - 1) // 2
        return [n_pairs, n_features, n_pairs]

    def validate(self, *, m: int, k: int, n_features: int) -> None:
        if n_features > m - 1:
            raise ValueError(
                f"encoding 'havlicek' needs a beamsplitter ladder that spans every feature on "
                f"modes (i, i+1), so it needs n_features <= m - 1 (got n_features={n_features}, "
                f"m={m}). Raise m, or lower n_features."
            )

    def extra_inputs(self, X):
        import torch

        pairs = _pair_indices(X.shape[1])
        xcorr = torch.stack([X[:, i] * X[:, j] for i, j in pairs], dim=1)
        return [xcorr, X, xcorr]

    def add_to(self, circuit, *, m: int, n_features: int, parameterised: bool = True) -> None:
        import perceval as pcvl

        pairs = _pair_indices(n_features)
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))
        for i in range(n_features):
            circuit.add(i, pcvl.PS(pcvl.P(f"x{i}")))
        for idx, (i, j) in enumerate(pairs):
            _add_routed_bs(circuit, m, i, j, pcvl.BS(theta=pcvl.P(f"b{idx}")))
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))
        for i in range(n_features):
            circuit.add(i, pcvl.PS(pcvl.P(f"c{i}")))
        for idx, (i, j) in enumerate(pairs):
            _add_routed_bs(circuit, m, i, j, pcvl.BS(theta=pcvl.P(f"d{idx}")))
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))

    def add_concrete(self, circuit, x, *, m: int, n_features: int) -> None:
        import perceval as pcvl

        pairs = _pair_indices(n_features)
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))
        for i in range(n_features):
            circuit.add(i, pcvl.PS(float(x[i])))
        for i, j in pairs:
            _add_routed_bs(circuit, m, i, j, pcvl.BS(theta=float(x[i] * x[j])))
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))
        for i in range(n_features):
            circuit.add(i, pcvl.PS(float(x[i])))
        for i, j in pairs:
            _add_routed_bs(circuit, m, i, j, pcvl.BS(theta=float(x[i] * x[j])))
        for i in range(n_features):
            circuit.add(i, pcvl.BS(theta=math.pi / 2))

    def unitary(self, X, *, m: int, n_features: int):
        import torch

        N = X.shape[0]

        def _ps_unitary(x_row: torch.Tensor) -> torch.Tensor:
            U = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
            step = torch.eye(m, dtype=torch.complex64).unsqueeze(0).expand(N, m, m).clone()
            for i in range(n_features):
                step[:, i, i] = torch.exp(1j * x_row[:, i].to(torch.complex64))
            return torch.einsum("nij,njk->nik", step, U)

        U = _bs_ladder_fixed_unitary(m, n_features, N)
        U = torch.einsum("nij,njk->nik", _ps_unitary(X), U)
        U = torch.einsum("nij,njk->nik", _bs_ladder_pairwise_unitary(X, m=m, n_features=n_features), U)
        U = torch.einsum("nij,njk->nik", _bs_ladder_fixed_unitary(m, n_features, N), U)
        U = torch.einsum("nij,njk->nik", _ps_unitary(X), U)
        U = torch.einsum("nij,njk->nik", _bs_ladder_pairwise_unitary(X, m=m, n_features=n_features), U)
        U = torch.einsum("nij,njk->nik", _bs_ladder_fixed_unitary(m, n_features, N), U)
        return U


def build_encoding(name: str) -> Encoding:
    """Instantiate the encoding registered under ``name``."""
    if name not in ENCODINGS:
        raise ValueError(f"unknown encoding {name!r}; registered: {sorted(ENCODINGS)}")
    return ENCODINGS[name]()
