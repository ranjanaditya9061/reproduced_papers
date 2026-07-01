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

from model import build_teacher, sample_X

from .artifact import artifact_path, load_raw, save_pool
from .config import ExperimentConfig, load_config
from .prepare import derive_confidence, derive_labels
from .seeding import seed_everything


def draw_pool(cfg: ExperimentConfig, size: int):
    """Draw a raw ``(X, soft, teacher)`` of ``size`` samples."""
    X = sample_X(size, cfg.resolved_n_features, cfg.seeds.sample_seed)
    teacher = build_teacher(cfg)
    soft = teacher(X)
    return X, soft, teacher


def generate(cfg: ExperimentConfig, *, out_root: str | Path = "datasets",
             force: bool = False) -> Path:
    """Create (or reuse/extend) the artifact for ``cfg``; return its path."""
    seed_everything(cfg.seeds.sample_seed)
    path = artifact_path(cfg, out_root)
    target = cfg.generation.size

    if path.exists() and not force:
        meta = load_raw(path)[2]
        existing = int(meta["size"])
        if existing >= target:
            print(f"[generate] cache hit: {path} (size={existing} >= {target})")
            return path
        print(f"[generate] extending {path} ({existing} -> {target})")

    X, soft, teacher = draw_pool(cfg, target)
    save_pool(path, cfg, X, soft, teacher=teacher)
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