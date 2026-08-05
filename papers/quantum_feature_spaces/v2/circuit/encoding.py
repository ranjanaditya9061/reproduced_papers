"""How the input ``x`` enters the circuit: an open registry of encodings.

An *encoding* adds the ``x``-dependent components to a circuit and declares its own geometry
constraint.  Deliberately open-ended: the legacy pipeline hard-coded one phase shifter per mode
on the first ``n_features`` modes inside ``build_sandwich_circuit``, so there was no way to try
another scheme without editing every model.  Here the model takes an encoding by name.

Only ``phase`` ships (today's behaviour, and the default).  The abstraction exists so that data
re-uploading, MZI-angle encodings, scaled/multi-frequency encodings or a dense mixing encoding
drop in as modules without touching any model.  Note the *state preparation* is a separate axis
(:mod:`v2.circuit.prep`) -- a spin-prepared photonic circuit is the same ``phase`` encoding with
``prep="spin"``.

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

        The fermion and reference-permanent readouts build ``U(x)`` themselves rather than going
        through perceval, so they need the encoding as a diagonal instead of as components.
        Encodings that are not diagonal in the mode basis should raise here, which keeps those
        readouts honest about what they support rather than silently ignoring the encoding.
        """
        raise NotImplementedError

    def spec(self) -> dict:
        """Identity fields folded into the artifact hash."""
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


def build_encoding(name: str) -> Encoding:
    """Instantiate the encoding registered under ``name``."""
    if name not in ENCODINGS:
        raise ValueError(f"unknown encoding {name!r}; registered: {sorted(ENCODINGS)}")
    return ENCODINGS[name]()
