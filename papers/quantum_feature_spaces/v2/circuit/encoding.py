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

#: name -> Encoding subclass, populated on subclassing.
ENCODINGS: dict[str, type["Encoding"]] = {}


class Encoding:
    """Adds the ``x``-dependent components to a circuit, and validates its own geometry."""

    name: str | None = None

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


def build_encoding(name: str) -> Encoding:
    """Instantiate the encoding registered under ``name``."""
    if name not in ENCODINGS:
        raise ValueError(f"unknown encoding {name!r}; registered: {sorted(ENCODINGS)}")
    return ENCODINGS[name]()
