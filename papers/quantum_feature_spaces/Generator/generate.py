"""Generation stage: draw a raw teacher-output pool and persist it.

Pipeline: a shared seeded sampler draws ``X``; the teacher named by the config
maps it to a continuous ``soft`` output; the pair is saved.  Labelling, margin
filtering and balancing are deferred to load time (:mod:`Generator.prepare`).

Prefix stability / extension
----------------------------
``sample_X`` is a single contiguous draw from a fixed ``sample_seed``, and the
teacher is deterministic in ``teacher_seed``, so ``generate(size=2N)`` reproduces
the first ``N`` rows of ``generate(size=N)`` exactly (the teacher is unchanged).
Extending a pool therefore means re-drawing at the larger size: existing rows
are byte-for-byte identical.

CLI::

    python -m Generator --config Generator/configs/example_photonic.yaml [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Support both `python -m Generator` and `python Generator/generate.py`: when run
# as a bare script there is no package context, so put the paper root on the path
# and set the package so the relative imports below resolve.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Generator"

import torch

from model import build_teacher, sample_X
from model.spoqc_magic import load_distributions, merge_distributions, write_distributions

from .artifact import artifact_path, load_raw, save_pool
from .config import ExperimentConfig, load_config
from .prepare import derive_confidence, derive_labels
from .seeding import seed_everything


DIST_FILENAME = "distributions.npz"


def _simulate(cfg: ExperimentConfig, X_rows: torch.Tensor, *, capture: bool):
    """Run the teacher over ``X_rows``; return ``(soft, teacher)`` (capture optional)."""
    teacher = build_teacher(cfg)
    if capture:
        if not hasattr(teacher, "enable_distribution_capture"):
            raise ValueError(
                f"generation.save_dist is set but teacher {cfg.generation.generator!r} "
                "does not support distribution capture (spoqc_magic_photonic only)"
            )
        teacher.enable_distribution_capture()
    return teacher(X_rows), teacher


def draw_pool(cfg: ExperimentConfig, size: int):
    """Draw a raw ``(X, soft, teacher)`` of ``size`` samples (full, non-incremental)."""
    X = sample_X(size, cfg.resolved_n_features, cfg.seeds.sample_seed)
    soft, teacher = _simulate(cfg, X, capture=cfg.generation.save_dist)
    return X, soft, teacher


def generate(cfg: ExperimentConfig, *, out_root: str | Path = "datasets",
             force: bool = False) -> Path:
    """Create (or reuse/extend) the artifact for ``cfg``; return its path.

    Extension is *incremental*: only rows the cached pool lacks are simulated, and
    the newly computed ``soft`` (and distributions, when ``save_dist``) are appended
    to what is on disk.  ``sample_X`` is prefix-stable, so the reused prefix is
    byte-identical to a full re-draw.
    """
    seed_everything(cfg.seeds.sample_seed)
    path = artifact_path(cfg, out_root)
    target = cfg.generation.size
    save_dist = cfg.generation.save_dist
    dist_path = path / DIST_FILENAME

    if not path.exists() or force:
        return _write_fresh(cfg, path, target, save_dist, dist_path)

    _, soft_old, meta = load_raw(path)
    existing = int(meta["size"])
    final = max(existing, target)                     # never shrink an existing pool

    old_dist = load_distributions(dist_path) if save_dist and dist_path.exists() else None
    n_dist_old = int(old_dist["probs"].shape[0]) if old_dist is not None else 0

    dist_done = (not save_dist) or n_dist_old >= final
    if final == existing and dist_done:
        print(f"[generate] cache hit: {path} (size={existing} >= {target})")
        return path

    # Incremental append is exact only for deterministic, row-independent teachers.
    # Shot sampling (nsample > 0) makes each row's value depend on the RNG stream
    # position, so a reused prefix would not match a full re-draw -> recompute all.
    if cfg.generation.nsample > 0:
        print(f"[generate] extending {path} ({existing} -> {final}) "
              "[shot sampling -> full recompute]")
        return _write_fresh(cfg, path, final, save_dist, dist_path)

    # Simulate only the rows we are missing.  soft is missing beyond `existing`;
    # distributions are missing beyond `n_dist_old` (0 if the file is absent) -- so
    # a save_dist backfill of an older pool re-simulates from `n_dist_old`, reusing
    # the cached soft prefix but recomputing the (unavailable) distribution prefix.
    sim_start = min(existing, n_dist_old) if save_dist else existing
    print(f"[generate] extending {path}: soft {existing}->{final}"
          + (f", dist {n_dist_old}->{final}" if save_dist else ""))

    X_full = sample_X(final, cfg.resolved_n_features, cfg.seeds.sample_seed)
    soft_sim, teacher = _simulate(cfg, X_full[sim_start:], capture=save_dist)

    soft_full = torch.cat([soft_old, soft_sim[existing - sim_start:]], dim=0)  # keep old soft prefix
    save_pool(path, cfg, X_full, soft_full, teacher=teacher)

    if save_dist:
        new_dist = teacher.captured_distributions()   # rows [sim_start, final)
        old_slice = None if old_dist is None else {**old_dist, "probs": old_dist["probs"][:sim_start]}
        merged = new_dist if old_slice is None else merge_distributions(old_slice, new_dist)
        write_distributions(dist_path, merged)
        print(f"[generate] appended distributions ({merged['probs'].shape[0]} rows) -> {dist_path}")

    _print_summary(cfg, path, X_full, soft_full)
    return path


def _write_fresh(cfg: ExperimentConfig, path: Path, target: int,
                 save_dist: bool, dist_path: Path) -> Path:
    """Full (non-incremental) generation of ``target`` rows from scratch."""
    X, soft, teacher = draw_pool(cfg, target)
    save_pool(path, cfg, X, soft, teacher=teacher)
    if save_dist:
        write_distributions(dist_path, teacher.captured_distributions())
        print(f"[generate] wrote full Fock distributions ({target} rows) -> {dist_path}")
    _print_summary(cfg, path, X, soft)
    return path


def _print_summary(cfg, path, X, soft) -> None:
    y = derive_labels(soft)
    n = X.shape[0]
    pos = int((y == 1).sum())
    print(
        f"[generate] wrote {path}\n"
        f"           pool={n}  soft in [{float(soft.min()):.3f}, {float(soft.max()):.3f}]  "
        f"class balance: 0->{n - pos}  1->{pos}  "
        f"(margin filtering / balancing is a load-time diagnostic; see the analyzer)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and save raw dataset pool(s).")
    parser.add_argument("--config", help="path to a single YAML experiment config")
    parser.add_argument("--configs-dir", help="generate every *.yaml in this folder")
    parser.add_argument("--out-root", default="datasets", help="root dir for artifacts")
    parser.add_argument("--force", action="store_true",
                        help="regenerate from scratch even if the artifact exists")
    args = parser.parse_args(argv)

    if bool(args.config) == bool(args.configs_dir):
        parser.error("give exactly one of --config or --configs-dir")

    if args.config:
        paths = [args.config]
    else:
        paths = [str(p) for p in sorted(Path(args.configs_dir).glob("*.yaml"))]
        if not paths:
            parser.error(f"no *.yaml configs found in {args.configs_dir}")
        print(f"[generate] {len(paths)} config(s) in {args.configs_dir}")

    for p in paths:
        generate(load_config(p), out_root=args.out_root, force=args.force)


if __name__ == "__main__":
    main()