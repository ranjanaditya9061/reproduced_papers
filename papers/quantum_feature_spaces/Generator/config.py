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

#: Base scorers allowed under a photonic ``connected_<base>`` observable
#: (mirrors :data:`model.photonic.CONNECTED_BASES`).
CONNECTED_BASES = ("parity", "majority", "bunching", "n_first", "maxcc")


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
    # photonic connected_<base>-only: modes map to the V=m vertices of a fixed graph G, and
    # graph_density is the fraction of the C(m, 2) possible edges present (edge count =
    # round(graph_density * C(m, 2))); denser G -> fewer clicked subsets stay independent.
    graph_density: float | None = None
    graph_seed: int | None = None  # seeds G + M_0 (defaults to teacher_seed); folded into the hash
    # photonic prod_parity_*_random-only: seeds the per-monomial angles theta ~ U[0, pi] of the
    # phase observable cos(P(n)) (defaults to teacher_seed); folded into the hash so a different
    # draw is a different dataset.  Unused by prod_parity_*_pi (theta = pi is deterministic).
    angle_seed: int | None = None

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
        elif p.observable.startswith("connected_"):
            # Sibling of loop_path_<base>: each mode is a *vertex* of a fixed graph G on V=m
            # vertices (build_vertex_graph), and each Fock outcome scores a global property of the
            # clicked vertices' induced subgraph -- ``maxcc``, its largest connected component
            # size -- as a plain E[<base>] over all outcomes (no selection; see model.photonic).
            # G's density rides on graph_density, not m/n_vertices.
            from model.photonic import parse_connected_observable

            _, base = parse_connected_observable(p.observable)
            if base not in CONNECTED_BASES:
                raise ValueError(
                    f"unknown connected base {p.observable!r}; choose base from {list(CONNECTED_BASES)}"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("connected_<base> observables are photonic_quantum only")
            if p.m < 2:
                raise ValueError(f"connected_ needs m >= 2 vertices (got m={p.m})")
            if p.graph_density is None:
                raise ValueError("connected_<base> observables require problem.graph_density")
            if not 0.0 < p.graph_density <= 1.0:
                raise ValueError(
                    f"connected_ needs graph_density in (0, 1] (got {p.graph_density})"
                )
        elif p.observable.startswith("prod_parity"):
            # (-1)^P(n) with P a sum of square-free monomials in the photon counts; the monomial
            # set is encoded in the observable string, or derived from (m, k) for the
            # consecutive variant.  The angle variants (prod_parity_{consecutive,second}_{pi,random})
            # instead score cos(P(n)) with per-monomial angles (see model.photonic).
            from model.photonic import (angle_monomials, is_prod_family,
                                        is_prod_parity_angle, prod_family_monomials)

            if not is_prod_family(p.observable) and not is_prod_parity_angle(p.observable):
                raise ValueError(
                    f"malformed prod_parity observable {p.observable!r} (expected "
                    "prod_parity[__<preset|M<i>-<j>-...|N<i>-<j>-...>...], prod_parity_consecutive"
                    " / prod_parity_second, or an angle variant "
                    "prod_parity_{consecutive,second}_{pi,random})"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("prod_parity observables are photonic_quantum only")
            if is_prod_parity_angle(p.observable):
                aseed = self.seeds.teacher_seed if p.angle_seed is None else int(p.angle_seed)
                angle_monomials(p.observable, p.m, p.k, aseed)   # validate (m, k)
            else:
                prod_family_monomials(p.observable, p.m, p.k)    # validate segments / (m, k)
        elif (p.observable.startswith("sq_") or p.observable == "pairprod"
                or p.observable == "ent" or p.observable.startswith("ent_")
                or p.observable == "osc" or p.observable.startswith("osc_")):
            # Nonlinear observables.  Degree-2 in p (2-copy/collision quantities, not
            # additive-error estimable from single-shot samples): sq_<base> = Σ_n <base>(n) p(n)^2
            # (p^T K p with K diagonal), or pairprod = Σ_{n1,n2} p(n1) p(n2) (-1)^<n1,n2> (a dense,
            # non-separable ±1 kernel).  Not polynomial in p at all -- so not the expectation of
            # any fixed observable -- are the pointwise families Σ_n <base>(n) φ(p(n)):
            # ent[_<base>] with φ = p log p (bare ent being -H(p)), and osc[_<base>] with
            # φ = p sin(1/(p + eps)).  sq_/ent_/osc_<base> are signed so no balancing is required;
            # bare ent is non-positive, so label it by a threshold rather than by sign.  NOTE osc
            # trades learnability for hardness -- see model.photonic_observables.oscillatory.
            from model.photonic import (SQ_BASES, is_ent_observable, is_osc_observable,
                                        is_pairprod_observable, is_sq_observable)

            if not (is_sq_observable(p.observable) or is_pairprod_observable(p.observable)
                    or is_ent_observable(p.observable) or is_osc_observable(p.observable)):
                raise ValueError(
                    f"unknown nonlinear observable {p.observable!r}; expected sq_<base>, "
                    f"ent[_<base>] or osc[_<base>] (base in {list(SQ_BASES)}), or pairprod"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("sq_<base> / ent[_<base>] / osc[_<base>] / pairprod observables "
                                 "are photonic_quantum only")
            if p.observable in ("sq_majority", "ent_majority", "osc_majority") and p.m % 2:
                raise ValueError(f"{p.observable} requires an even m")
        elif p.observable == "xent" or p.observable.startswith("xent_"):
            # xent[_<base>] = Σ_n <base>(n) p(n) log q(n), scoring against the FIXED reference q =
            # the circuit's output with every encoded feature at zero.  Because q does not depend
            # on x this is LINEAR in p -- the plain expectation <log q>, not a hardness witness
            # like ent -- but it is the other half of KL(p||q) = ent - xent.  q is reproducible
            # from (m, k, n_features, seed), which already fix the dataset identity, so no extra
            # hashed knob is needed; it is not persisted in distributions.npz, though, so an
            # offline re-score must be handed reference_probs explicitly.
            from model.photonic import SQ_BASES, is_xent_observable

            if not is_xent_observable(p.observable):
                raise ValueError(
                    f"unknown cross-entropy observable {p.observable!r}; expected xent or "
                    f"xent_<base> (base in {list(SQ_BASES)})"
                )
            if g.generator != "photonic_quantum":
                raise ValueError("xent[_<base>] observables are photonic_quantum only")
            if p.observable == "xent_majority" and p.m % 2:
                raise ValueError("xent_majority requires an even m")
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