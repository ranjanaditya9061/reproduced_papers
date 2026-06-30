"""Embedding stage: kernel representations of a dataset, with their own config + store.

A teacher labels a dataset; an *embedding* maps that dataset's X through a feature
map and saves the resulting feature matrix under ``embeddings/<dataset_hash>/`` --
separate from the dataset, with provenance that marks the dataset's sample/teacher
seeds vs the embedding's own seed.  The kernel (RBF) is applied at load time.

    from embedding import build_embeddings
    results, dcfg, meta = build_embeddings("configs/embed_qubit.yaml")

CLI::

    python -m embedding --config configs/embed_qubit.yaml
"""

from __future__ import annotations

from .config import EmbeddingConfig, load_embedding_config
from .features import EMBEDDINGS, Embedding, build_embedding, projected_features
from .store import (
    build_embeddings,
    build_embeddings_for,
    embedding_path,
    load_embedding,
    save_embedding,
)

__all__ = [
    "EmbeddingConfig",
    "load_embedding_config",
    "Embedding",
    "EMBEDDINGS",
    "build_embedding",
    "projected_features",
    "build_embeddings",
    "build_embeddings_for",
    "save_embedding",
    "load_embedding",
    "embedding_path",
]
