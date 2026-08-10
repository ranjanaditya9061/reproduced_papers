"""Typed experiment config, with the input size as a study-level invariant.

Sections, each with one responsibility::

    problem:    n_features, m, k          # n_features is REQUIRED and pinned (see below)
    model:      kind, prep, encoding, +knobs   # the circuit -> determines the artifact
    generation: size, shots, n_jobs, max_dist_bytes
    split:      test_fraction, split_seed
    seeds:      sample_seed, model_seed, shot_seed

**There is deliberately no ``observable`` field.**  An observable is a readout applied to a
saved distribution, so it belongs to the scoring stage (:mod:`v2.pipeline.score`) and to a
learner config -- never to the data config, whose identity must not move when the readout
changes.  This is the single most important difference from ``Generator/config.py``.

**``n_features`` is required, and must be held fixed across any one comparison.**  Every
complexity measure in :mod:`v2.metrics` is denominated in it: the input Fisher matrix is
``n_features x n_features``, ``r_eff`` lies in ``[0, n_features]``, and the description cost
``(1/2) sum_i log(1 + lambda_i/eps^2)`` has ``n_features`` terms.  Vary it *within* a comparison
and you conflate "more input dimensions" with "harder map", which is exactly the objection that
rules out a weight-space FIM.

So it is carried per config rather than as a module-level constant -- a study is free to choose its
own input size -- and :func:`check_commensurable` is what enforces that a grid or sweep does not mix
sizes.  The value itself is not validated against anything global; the only geometry constraint is
the encoding's own (a phase encoding needs one mode per feature, a dense one need not), applied in
:meth:`ExperimentConfig.validate`.

What that buys: at fixed ``n_features`` the ``(m, k)`` sweep leaves the FIM at one size while the
circuit grows, so spectra stay stackable across sizes *and* across model families.  The legacy
default ``n_features = m - 1`` coupled the two, so every change of circuit size silently changed the
FIM's dimension and no such comparison was possible; there is no such fallback here.

There is also no ``prepare`` section: margin filtering and class balancing are a diagnostic
applied downstream, never a load-time transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

#: Peak bytes allowed for one artifact's ``probs`` matrix before generation refuses to run.
#: ``probs`` is ``(N, n_outcomes)`` float32, so this grows as ``C(m+k-1, k)``: at ``N = 10k``,
#: ``n_fock = 2002`` is 80 MB, ``12376`` is 495 MB and ``77520`` is 3.1 GB.  The guard errors
#: with the computed size rather than downcasting silently.
DEFAULT_MAX_DIST_BYTES = 2 * 1024 ** 3


@dataclass
class ProblemConfig:
    """Problem geometry.  ``n_features`` is required -- there is no ``m - 1`` fallback."""

    n_features: int
    m: int = 6
    k: int = 3


@dataclass
class ModelConfig:
    """The circuit.  Everything here feeds the artifact hash via ``model.circuit_spec()``.

    ``prep`` and ``encoding`` are registry names, so a spin-prepared photonic circuit is this
    model with ``prep="spin"`` rather than its own class -- which is what collapses the seven
    legacy ``spoqc*`` modules.  The remaining fields are per-family knobs, applied only by the
    families that declare them and hashed only when set.
    """

    kind: str = "photonic"
    prep: str = "fock"
    encoding: str = "phase"

    # --- prep="spin" / "spin_magic" ---------------------------------------------------- #
    cx_pairs: list | None = None      # spin entanglers, e.g. [[0,1],[1,2]]
    angle_levels: int | None = None   # discretise the rx/ry twists
    rz_angles: str | None = None      # None | "prime"
    #: spin: depth of the seeded rotate-then-entangle block (H->Rx->[Rz]->Ry, then cx_pairs),
    #: each layer independently seeded rather than replaying the same angles.  None/1 = the legacy
    #: single-round circuit.  See v2.circuit.prep.SpinPrep / v2.circuit.spin.spin_state.
    layers: int | None = None
    t_var: int | None = None          # spin_magic: number of injected gap gates
    gate_kind: str | None = None      # spin_magic: "t" | "rz" | "u3" | "u3_x"
    #: spin_magic: per-gap gate pattern -- "linear" (H every gap, default) | "ghz" (H at gap 0
    #: only) | "linear_u3" (H + an unconditional Haar U3 every gap).  Orthogonal to gate_kind/
    #: t_var's sparser magic-gate injection; the two compose.  See v2.circuit.prep.SpinMagicPrep.
    structure: str | None = None
    #: spin: puts x on the spin once, before any layer (rx(x[2q]), ry(x[2q+1]) per qubit on
    #: |0...0>).  spin_magic: same idea but per-gap and overriding gate_kind's legacy "_rxry"/
    #: "_rxry_iface" suffix when set.  encode_circuit (spin_magic only) puts x in the
    #: interferometer as well (False -> the encoding sees x=0, i.e. the identity).
    encode_on_spin: bool | None = None
    encode_circuit: bool | None = None

    # --- kind="fermion" ----------------------------------------------------------------- #
    #: Modulus exponent of the phase-power columns, the dial that sets how much mass the bunched
    #: sector carries.  ``None`` is the ``k/m`` rule, which matches the boson model's bunched mass
    #: to ~2% with no free parameter -- so the boson/determinant Fisher comparison is controlled on
    #: support *and* bunching rather than being partly a comparison of support sizes.  See
    #: :mod:`v2.model.fermion`.
    bunching_s: float | None = None

    #: Retired.  The flavoured-fermion readout degenerated to ``Perm(|U|^2)`` -- the classical
    #: distinguishable-particle distribution, with no determinant left -- at ``flavours = k``, since
    #: every flavour block is then ``1 x 1``.  Kept only so old configs fail with an explanation
    #: rather than a missing-key error; anything other than ``1`` is rejected.
    flavours: int = 1

    # --- kind="ebm_fock" ---------------------------------------------------------------- #
    #: Rank of the bilinear weight matrix ``W = A B^T``, the parameter dial.  ``None`` keeps
    #: the legacy full-rank behaviour; ``param_matched`` solves for the ``rank`` whose count is
    #: closest to the circuit's ``2m^2 - 1``.
    rank: int | None = None
    param_matched: bool = False


@dataclass
class GenerationConfig:
    size: int = 10000
    #: Target shots per input for the **shots branch**; ``0`` = exact distribution only.
    #:
    #: Rounded UP to a whole number of :data:`v2.pipeline.shots.BLOCK`-sized blocks, so ``45_000``
    #: realises ``50_000``.  That quantisation is what makes the format additive: shot draws are
    #: stored as cumulative integer counts, and counts cannot be truncated *within* a block
    #: (recovering 45k from a 50k count vector would need the individual shot indices).  Block
    #: boundaries are the only points a budget can be cut at.
    #:
    #: The realised count is **not** part of the artifact identity -- see
    #: :func:`v2.pipeline.artifact.hash_fields`.  Growing it extends the same store, exactly as
    #: growing ``size`` extends the same directory.
    shots: int = 0
    n_jobs: int = 1                   # per-row parallelism for the perceval prep paths
    max_dist_bytes: int = DEFAULT_MAX_DIST_BYTES


@dataclass
class SplitConfig:
    test_fraction: float = 0.20
    split_seed: int = 0


@dataclass
class SeedConfig:
    sample_seed: int = 42   # the input pool X
    model_seed: int = 42    # the circuit's fixed weights
    #: Which shot *realisation* to draw.  A first-class field rather than a value derived from
    #: ``model_seed``, because shot noise is a property of the MEASUREMENT: deriving it from the
    #: circuit seed made it impossible to draw a second realisation of the same circuit, which is
    #: exactly what error bars on the SNR -> R^2 ceiling need.  It selects a substream, not a
    #: circuit, so it enters the SHOT identity and never the circuit identity.
    shot_seed: int = 0


@dataclass
class ExperimentConfig:
    problem: ProblemConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    @property
    def n_features(self) -> int:
        """The input dimension, straight from the config; no ``m - 1`` fallback exists."""
        return self.problem.n_features

    def validate(self) -> None:
        """Check the invariant, the registries, and the encoding's own geometry constraint."""
        from circuit.encoding import ENCODINGS, build_encoding
        from circuit.prep import PREPS
        from model import MODELS

        p, m_, g = self.problem, self.model, self.generation

        if p.m < 2:
            raise ValueError(f"problem.m must be >= 2 (got {p.m})")
        if p.k < 1:
            raise ValueError(f"problem.k must be >= 1 (got {p.k})")

        if m_.kind not in MODELS:
            raise ValueError(f"unknown model kind {m_.kind!r}; choose from {sorted(MODELS)}")
        if m_.encoding not in ENCODINGS:
            raise ValueError(f"unknown encoding {m_.encoding!r}; choose from {sorted(ENCODINGS)}")
        if m_.prep not in PREPS:
            raise ValueError(f"unknown prep {m_.prep!r}; choose from {sorted(PREPS)}")

        # The n_features-vs-m constraint belongs to the encoding, not here: a phase encoding
        # needs one mode per feature, a dense/mixed one would not.
        build_encoding(m_.encoding).validate(m=p.m, k=p.k, n_features=p.n_features)
        MODELS[m_.kind].validate_config(self)

        if g.size <= 0:
            raise ValueError("generation.size must be positive")
        if g.shots < 0:
            raise ValueError("generation.shots must be >= 0 (0 = exact distribution only)")
        if not 0.0 < self.split.test_fraction < 1.0:
            raise ValueError("split.test_fraction must be in (0, 1)")


def _section(raw: dict, key: str, cls, **required):
    """Build a dataclass section, rejecting unknown keys so a typo is never silently ignored."""
    data = dict(raw.get(key, {}) or {})
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys in '{key}': {sorted(unknown)} "
                         f"(allowed: {sorted(allowed)})")
    for name, why in required.items():
        if name not in data:
            raise ValueError(f"'{key}.{name}' is required: {why}")
    return cls(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file into a validated :class:`ExperimentConfig`.

    Rejects an ``observable`` key anywhere with an explanatory error: putting the readout in
    the data config is the mistake this pipeline exists to prevent, so it fails loudly rather
    than being ignored.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {"problem", "model", "generation", "split", "seeds", "name"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")

    for section in ("problem", "model", "generation"):
        if "observable" in (raw.get(section) or {}):
            raise ValueError(
                f"'{section}.observable' is not a config field. The observable is a READOUT "
                "applied to a saved distribution, cached per-observable by v2.pipeline.score, "
                "so it must not enter the dataset identity -- otherwise the same circuit and "
                "the same inputs are simulated once per readout. Pass observables to the "
                "scoring / learner stage instead."
            )

    cfg = ExperimentConfig(
        problem=_section(raw, "problem", ProblemConfig,
                         n_features=f"the study invariant (must be ); there is no "
                                    "'m - 1' default in v2"),
        model=_section(raw, "model", ModelConfig),
        generation=_section(raw, "generation", GenerationConfig),
        split=_section(raw, "split", SplitConfig),
        seeds=_section(raw, "seeds", SeedConfig),
    )
    cfg.validate()
    return cfg


def check_commensurable(cfgs) -> None:
    """Raise unless every config in a comparison shares the input size (and so is comparable).

    Called by the grid / sweep loaders.  Since ``n_features`` is carried per config and checked
    against nothing global at load, this is the *only* thing standing between a stale artifact or a
    hand-built config and a grid of incomparable spectra -- so it is load-bearing, not a second belt.
    """
    sizes = {int(c.problem.n_features) for c in cfgs}
    if len(sizes) > 1:
        raise ValueError(
            f"configs in one comparison have different n_features: {sorted(sizes)}. "
            "Fisher spectra are n_features x n_features, so mixing sizes in a grid compares "
            "different-dimensional objects."
        )
