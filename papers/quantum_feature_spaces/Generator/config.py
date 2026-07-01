"""Typed experiment config loaded from YAML.

Each section has a single responsibility.  The *generation* parameters determine
the saved raw pool (and therefore the artifact hash); the *split* defines the one
shared train/test partition every consumer agrees on; the seeds each do one job.
There is deliberately **no** ``prepare`` section and **no** ``model_seed`` here:

    problem:    m, k, observable, n_features
    generation: generator, size, nsample                   # affect the saved pool
    split:      test_fraction, split_seed                  # the shared partition
    seeds:      sample_seed, teacher_seed                  # define X and the labels

Margin filtering / class balancing are a *diagnostic* applied by the analyzer
(see :func:`Generator.prepare.prepare`), not a load-time transform — for learning
nothing is discarded.  ``model_seed`` belongs to a learner, not to the data.

See ``configs/example_photonic.yaml`` for a complete example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

OBSERVABLES = ("parity", "majority", "bunching", "single_output", "n_first")


@dataclass
class ProblemConfig:
    m: int = 6
    k: int = 3
    observable: str = "parity"
    n_features: int | None = None
    cx_pairs: list | None = None   # spoqc-only: CX entanglers in the spin prep, e.g. [[0,1],[1,2]]


@dataclass
class GenerationConfig:
    generator: str = "photonic_quantum"
    size: int = 10000
    nsample: int = 0


@dataclass
class SplitConfig:
    test_fraction: float = 0.20   # fraction held out for test
    split_seed: int = 0           # seeds the (prepare-independent) train/test partition


@dataclass
class SeedConfig:
    sample_seed: int = 42      # the input points X (sample_X)
    teacher_seed: int = 42     # the teacher's weights (the labeling function)


@dataclass
class ExperimentConfig:
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    @property
    def resolved_n_features(self) -> int:
        """Encoded feature dimension (defaults to ``m - 1``)."""
        nf = self.problem.n_features
        return self.problem.m - 1 if nf is None else nf

    def validate(self) -> None:
        from model import TEACHERS  # lazy: avoid a Generator <-> model import cycle

        p, g = self.problem, self.generation
        if g.generator not in TEACHERS:
            raise ValueError(
                f"unknown generator {g.generator!r}; choose from {sorted(TEACHERS)}"
            )
        if p.observable not in OBSERVABLES:
            raise ValueError(
                f"unknown observable {p.observable!r}; choose from {list(OBSERVABLES)}"
            )
        if p.m < 2:
            raise ValueError(f"m must be >= 2 (got {p.m})")
        if p.n_features is not None and p.n_features > p.m - 1:
            raise ValueError(
                f"n_features ({p.n_features}) must be <= m-1 = {p.m - 1}"
            )
        if p.observable == "majority" and g.generator == "photonic_quantum" and p.m % 2:
            raise ValueError("observable 'majority' requires an even m")
        if g.size <= 0:
            raise ValueError("generation.size must be positive")
        if not 0.0 < self.split.test_fraction < 1.0:
            raise ValueError("split.test_fraction must be in (0, 1)")


def _section(raw: dict, key: str, cls):
    """Build a dataclass section, rejecting unknown keys early."""
    data = raw.get(key, {}) or {}
    allowed = cls.__dataclass_fields__
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"unknown keys in '{key}': {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file into a validated :class:`ExperimentConfig`."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    # 'name' is an optional display-only label (used by the grid to label configs);
    # it does not affect the config, its validation, or the artifact hash.
    unknown = set(raw) - {"problem", "generation", "split", "seeds", "name"}
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
    cfg = ExperimentConfig(
        problem=_section(raw, "problem", ProblemConfig),
        generation=_section(raw, "generation", GenerationConfig),
        split=_section(raw, "split", SplitConfig),
        seeds=_section(raw, "seeds", SeedConfig),
    )
    cfg.validate()
    return cfg