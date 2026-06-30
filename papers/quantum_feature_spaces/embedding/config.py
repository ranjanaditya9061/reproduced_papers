"""Embedding-stage config: a dataset reference + the kernels to embed with.

Separate from the data config (which the dataset hash depends on) -- swapping
kernels must never change the dataset.  Example::

    dataset: configs/example_photonic.yaml      # path to a data config
    embeddings:
      - {type: rbf}
      - {type: fourier_rbf, fourier_order: 3}
      - {type: qubit_projected, depth: 3, seed: 42}   # 42 == teacher_seed -> matched
      - {type: qubit_projected, depth: 3, seed: 7}    # unmatched
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    dataset: str                       # path to the data (generation) config
    embeddings: list[dict] = field(default_factory=list)


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"dataset", "embeddings"}
    if unknown:
        raise ValueError(f"unknown embedding-config keys: {sorted(unknown)}")
    if "dataset" not in raw:
        raise ValueError("embedding config must reference a 'dataset' config path")
    specs = raw.get("embeddings", []) or []
    for s in specs:
        if "type" not in s:
            raise ValueError(f"each embedding entry needs a 'type'; got {s}")
    return EmbeddingConfig(dataset=raw["dataset"], embeddings=specs)
