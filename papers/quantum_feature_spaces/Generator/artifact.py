"""On-disk dataset artifact: a raw teacher-output pool + provenance sidecar.

Layout (one directory per logical dataset)::

    datasets/<generator>_m<m>_k<k>_<8-char-hash>/
    ├── data.pt      # torch.save({"X", "soft"}) — RAW output, no label/filter/balance
    ├── teacher.pt   # the teacher's state_dict (model parameters), if available
    └── meta.json    # provenance, including the model-specific hash spec

We save only ``X`` and the continuous teacher output ``soft``; labelling, margin
filtering and balancing are derived at load time (:mod:`Generator.prepare`).

The content **hash** is built from generic fields *plus* the teacher's own
``hash_spec(cfg)`` — so a new model's extra knobs ("variances") affect the
artifact identity automatically.  ``size`` is deliberately excluded, so growing
the pool keeps the same directory; ``meta.json["size"]`` records the current length.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from model import TEACHERS

from .config import ExperimentConfig

FORMAT_VERSION = 2


def _hash_fields(cfg: ExperimentConfig) -> dict:
    """Generic identity fields + the model-specific spec."""
    return {
        "format_version": FORMAT_VERSION,
        "generator": cfg.generation.generator,
        "m": cfg.problem.m,
        "k": cfg.problem.k,
        "n_features": cfg.resolved_n_features,
        "sample_seed": cfg.seeds.sample_seed,
        "teacher_seed": cfg.seeds.teacher_seed,
        "spec": TEACHERS[cfg.generation.generator].hash_spec(cfg),
    }


def compute_hash(cfg: ExperimentConfig) -> str:
    """8-char content hash over generation-only fields (excludes ``size``)."""
    blob = json.dumps(_hash_fields(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def artifact_dirname(cfg: ExperimentConfig) -> str:
    return f"{cfg.generation.generator}_m{cfg.problem.m}_k{cfg.problem.k}_{compute_hash(cfg)}"


def artifact_path(cfg: ExperimentConfig, out_root: str | Path = "datasets") -> Path:
    return Path(out_root) / artifact_dirname(cfg)


def save_pool(path: str | Path, cfg: ExperimentConfig, X: torch.Tensor,
              soft: torch.Tensor, teacher: torch.nn.Module | None = None) -> Path:
    """Write ``data.pt`` (+ optional ``teacher.pt``) + ``meta.json`` to ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save({"X": X.cpu(), "soft": soft.cpu()}, path / "data.pt")
    if teacher is not None:
        torch.save(teacher.state_dict(), path / "teacher.pt")

    meta = {
        **_hash_fields(cfg),
        "hash": compute_hash(cfg),
        "size": int(X.shape[0]),
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "soft_shape": list(soft.shape),
        "soft_dtype": str(soft.dtype),
        "soft_min": float(soft.min()),
        "soft_max": float(soft.max()),
        "has_teacher_state": teacher is not None,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_raw(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load ``(X, soft, meta)`` from an artifact directory."""
    path = Path(path)
    blob = torch.load(path / "data.pt")
    meta = json.loads((path / "meta.json").read_text())
    return blob["X"], blob["soft"], meta