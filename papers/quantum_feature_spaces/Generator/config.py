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

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

OBSERVABLES = ("parity", "majority", "bunching", "single_output", "n_first", "max_prob")

#: Base scorers allowed under a photonic ``loop_path_<base>`` graph observable
#: (mirrors :data:`model.photonic.GRAPH_BASES`).
GRAPH_BASES = ("parity", "majority", "bunching", "n_first", "loop", "path")


@dataclass
class ProblemConfig:
    m: int = 6
    k: int = 3
    observable: str = "parity"
    n_features: int | None = None
    cx_pairs: list | None = None   # spoqc-only: CX entanglers in the spin prep, e.g. [[0,1],[1,2]]
    angle_levels: int | None = None  # spoqc_low-only: rx/ry drawn from +-0.1 .. +-0.1*angle_levels
    t_var: int | None = None       # spoqc_magic-only: # of gates (magic); 0=none .. k=one per emit
    gate_kind: str | None = None    #spoqc-magic only: Type of gate
    # photonic loop_path_<base>-only: graph interpretation of the Fock outcomes.  The loop /
    # path selection is encoded IN the observable string via ``__L``/``__P`` suffixes, e.g.
    # ``loop_path_parity__L0-1__P2-3`` (dash-joined counts to keep); an omitted segment -- or
    # no suffix at all -- keeps every count on that dimension.
    n_vertices: int | None = None  # even; vertices of the fixed graph G (M_0 has V/2 edges)
    graph_seed: int | None = None  # seeds G + M_0 (defaults to teacher_seed); folded into the hash

@dataclass
class GenerationConfig:
    generator: str = "photonic_quantum"
    size: int = 10000
    nsample: int = 0
    save_dist: bool = False   # also dump the full per-input distribution next to the pool
    #                           (spoqc_magic only; side output, does not affect the artifact hash)
    n_jobs: int = 1           # spoqc_magic per-row parallelism: 1=serial, -1=auto (CPUs-1), N=workers
    #                           (performance only; does not affect the artifact hash)


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
        # photonic loop_path_<base>: reinterpret Fock outcomes as graph edge sets
        # (matching -> overlay with M_0 -> loop/path pre-selection -> score <base>).
        if p.observable.startswith("loop_path_"):
            # Parse via the model helper so encoded ``__L``/``__P`` var suffixes are
            # stripped before the base is validated (see model.photonic).
            from model.photonic import parse_graph_observable

            _, base, _, _ = parse_graph_observable(p.observable)
            if base not in GRAPH_BASES:
                raise ValueError(
                    f"unknown loop_path base {p.observable!r}; choose base from {list(GRAPH_BASES)}"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("loop_path_<base> observables are photonic_quantum only")
            if p.n_vertices is None:
                raise ValueError("loop_path_<base> observables require problem.n_vertices")
            if p.n_vertices < 2 or p.n_vertices % 2:
                raise ValueError(f"n_vertices must be a positive even int (got {p.n_vertices})")
            half = p.n_vertices // 2
            if not p.k <= half <= p.m:
                raise ValueError(
                    f"loop_path_ needs k <= n_vertices//2 <= m "
                    f"(k={p.k}, n_vertices//2={half}, m={p.m})"
                )
        elif p.observable.startswith("prod_parity"):
            # (-1)^P(n) with P a sum of square-free monomials in the photon counts; the monomial
            # set is encoded in the observable string, or derived from (m, k) for the
            # consecutive variant (see model.photonic).
            from model.photonic import is_prod_family, prod_family_monomials

            if not is_prod_family(p.observable):
                raise ValueError(
                    f"malformed prod_parity observable {p.observable!r} (expected "
                    "prod_parity[__<preset|M<i>-<j>-...|N<i>-<j>-...>...] or prod_parity_consecutive)"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("prod_parity observables are photonic_quantum only")
            prod_family_monomials(p.observable, p.m, p.k)   # validate segments / (m, k)
        else:
            # spoqc_magic allows a ``match{N}_<base>`` prefix (half-agreement pre-selection);
            # validate the base observable and let the teacher check N against m.
            base_observable = re.sub(r"^match\d+_", "", p.observable)
            if base_observable not in OBSERVABLES:
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