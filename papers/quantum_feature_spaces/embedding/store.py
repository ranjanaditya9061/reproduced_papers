"""Embedding stage: compute + persist kernel representations of a dataset.

Owns the on-disk embedding artifacts (separate from ``datasets/``), driven by an
embedding config that references a dataset.  Each saved blob records its full
provenance -- the dataset's ``sample_seed`` and ``teacher_seed``, the embedding's
own ``seed``, and whether they match (the rigged/sanity cell)::

    embeddings/<dataset_hash>/<embedding-name>_<spec-hash>.pt

Features are computed over the **full** ``X`` (the whole raw pool), not a margin-
filtered/balanced view -- so a blob is a pure function of its cache key
``(dataset_hash, spec)`` and can be sliced by any train/test split downstream.
The blob holds the feature matrix ``features(X)``; the kernel applied on top
(a Gaussian RBF, in :mod:`kernel`) is reconstructed at load time, so nothing
kernel-specific is stored here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from Generator import artifact_path, load_config, load_raw

from .config import load_embedding_config
from .features import build_embedding


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def embedding_path(embeddings_root, dataset_hash: str, emb) -> Path:
    return Path(embeddings_root) / dataset_hash / f"{emb.name}_{_spec_hash(emb.spec())}.pt"


def save_embedding(embeddings_root, dataset_meta: dict, emb, X: torch.Tensor) -> Path:
    """Compute and persist ``emb``'s feature matrix of the full ``X`` with provenance."""
    path = embedding_path(embeddings_root, dataset_meta["hash"], emb)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb_seed = emb.spec().get("seed")
    teacher_seed = dataset_meta.get("teacher_seed")
    matched = (emb_seed == teacher_seed) if emb_seed is not None else None
    torch.save(
        {
            "data": emb.features(X).cpu(),
            "spec": emb.spec(),
            "dataset_hash": dataset_meta["hash"],
            "sample_seed": dataset_meta.get("sample_seed"),
            "teacher_seed": teacher_seed,
            "embedding_seed": emb_seed,
            "matched": matched,
            "n": int(X.shape[0]),
        },
        path,
    )
    return path


def load_embedding(embeddings_root, dataset_hash: str, emb):
    path = embedding_path(embeddings_root, dataset_hash, emb)
    return torch.load(path) if path.exists() else None


def build_embeddings(embed_config_path, embeddings_root="embeddings",
                     dataset_root="datasets", use_cache=True):
    """Build (or reuse) every embedding in an embedding *config*; return result dicts.

    Each result is ``{spec, embedding, blob}`` where ``blob`` carries the stored
    feature matrix + provenance (incl. ``matched``).
    """
    ecfg = load_embedding_config(embed_config_path)
    return build_embeddings_for(
        ecfg.dataset, ecfg.embeddings, embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )


def build_embeddings_for(data_config, specs, *, embeddings_root="embeddings",
                         dataset_root="datasets", use_cache=True):
    """Build (or reuse) a shared ``specs`` list against one *data* config.

    The learner list is dataset-independent, so a grid can author it **once** and
    pair it with each data config here -- no per-dataset embedding config needed.
    ``data_config`` may be a path or an already-loaded ``ExperimentConfig``.
    """
    dcfg = data_config if hasattr(data_config, "problem") else load_config(data_config)
    path = artifact_path(dcfg, dataset_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not generated: {path}\n"
            f"Run first:  python -m Generator --config <the data config>"
        )
    X, _, meta = load_raw(path)            # full pool — no margin filter / balance

    results = []
    for spec in specs:
        emb = build_embedding(spec, dcfg)
        emb.check_input(X)             # fail loudly if the map's n_features != X's width
        blob = load_embedding(embeddings_root, meta["hash"], emb) if use_cache else None
        if blob is None:
            save_embedding(embeddings_root, meta, emb, X)
            blob = load_embedding(embeddings_root, meta["hash"], emb)
        results.append({"spec": spec, "embedding": emb, "blob": blob})
    return results, dcfg, meta
