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

**``n_features`` is a study invariant, not a knob.**  Every complexity measure in
:mod:`v2.metrics` is denominated in it: the input Fisher matrix is ``n_features x n_features``,
``r_eff`` lies in ``[0, n_features]``, and the description cost
``(1/2) sum_i log(1 + lambda_i/eps^2)`` has ``n_features`` terms.  Vary it and you conflate
"more input dimensions" with "harder map", which is exactly the objection that rules out a
weight-space FIM.  So :data:`N_FEATURES` is fixed once for the whole study and every config
must match it; a mismatch is a load error naming the invariant.

What that buys: at fixed ``n_features`` the ``(m, k)`` sweep leaves the FIM at
``6 x 6`` while the circuit grows, so spectra stay stackable across sizes *and* across model
families.  The legacy default ``n_features = m - 1`` coupled the two, so every change of
circuit size silently changed the FIM's dimension and no such comparison was possible.

There is also no ``prepare`` section: margin filtering and class balancing are a diagnostic
applied downstream, never a load-time transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

#: The fixed input dimension for the entire study.  See the module docstring: this is an
#: invariant, not a parameter.  Changing it invalidates every cross-model and cross-``(m, k)``
#: comparison in :mod:`v2.metrics`, because it is the dimension those comparisons are
#: denominated in -- a different value is a different, non-comparable study, not an extension.

#: Peak bytes allowed for one artifact's ``probs`` matrix before generation refuses to run.
#: ``probs`` is ``(N, n_outcomes)`` float32, so this grows as ``C(m+k-1, k)``: at ``N = 10k``,
#: ``n_fock = 2002`` is 80 MB, ``12376`` is 495 MB and ``77520`` is 3.1 GB.  The guard errors
#: with the computed size rather than downcasting silently.
DEFAULT_MAX_DIST_BYTES = 2 * 1024 ** 3


@dataclass
class ProblemConfig:
    """Problem geometry.  ``n_features`` is required and validated against :data:`N_FEATURES`."""

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
    t_var: int | None = None          # spin_magic: number of injected gap gates
    gate_kind: str | None = None      # spin_magic: "t" | "rz" | "u3" | "u3_x"

    # --- kind="fermion" ----------------------------------------------------------------- #
    #: Internal states per mode.  ``1`` is strict free fermions (bunched outcomes exactly 0 by
    #: Pauli exclusion); ``r >= 2`` allows occupations up to ``r``, which is what makes the
    #: boson/determinant Fisher comparison apples-to-apples rather than partly a comparison of
    #: support sizes.
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
        """The input dimension.  Always :data:`N_FEATURES`; no ``m - 1`` fallback exists."""
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

    Called by the grid / sweep loaders.  Individually each config already had to match
    :data:`N_FEATURES` at load, so this is a second belt against a stale artifact or a
    hand-built config entering a grid -- the failure mode that would silently produce
    incomparable spectra.
    """
    sizes = {int(c.problem.n_features) for c in cfgs}
    if len(sizes) > 1:
        raise ValueError(
            f"configs in one comparison have different n_features: {sorted(sizes)}. "
            "Fisher spectra are n_features x n_features, so mixing sizes in a grid compares "
            "different-dimensional objects."
        )
