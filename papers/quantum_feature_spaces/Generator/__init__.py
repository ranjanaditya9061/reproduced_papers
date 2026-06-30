"""Generation stage of the quantum-feature-spaces pipeline.

A config-driven, reproducible replacement for the inline data generation that
used to live inside each learner.  It builds a teacher from :mod:`model`, draws a
shared seeded input pool, and persists the **raw** per-sample teacher output to
disk.  :func:`load_split` then partitions the *full* pool into train/test
(nothing discarded); margin filtering / class balancing are a separate diagnostic
(:mod:`Generator.prepare`), not part of the load path.

Typical use::

    from Generator import load_config, generate, load_split

    cfg  = load_config("configs/example_photonic.yaml")
    path = generate(cfg)                       # draw + save (or reuse cache)
    split = load_split(path,                    # the SAME loader every model uses
                       test_fraction=cfg.split.test_fraction,
                       split_seed=cfg.split.split_seed)
    # split.train_idx / split.test_idx index the full pool (and any stored embedding)

CLI::

    python -m Generator --config <yaml> [--force]
"""

from __future__ import annotations

from .artifact import (
    artifact_path,
    compute_hash,
    load_raw,
    save_pool,
)
from .config import (
    ExperimentConfig,
    GenerationConfig,
    ProblemConfig,
    SeedConfig,
    SplitConfig,
    load_config,
)
from .generate import draw_pool, generate
from .prepare import derive_confidence, derive_labels, prepare, prepare_indices
from .seeding import seed_everything
from .split import Split, load_split

__all__ = [
    "ExperimentConfig",
    "ProblemConfig",
    "GenerationConfig",
    "SplitConfig",
    "SeedConfig",
    "load_config",
    "seed_everything",
    "compute_hash",
    "artifact_path",
    "save_pool",
    "load_raw",
    "draw_pool",
    "generate",
    "derive_labels",
    "derive_confidence",
    "prepare",
    "prepare_indices",
    "load_split",
    "Split",
]