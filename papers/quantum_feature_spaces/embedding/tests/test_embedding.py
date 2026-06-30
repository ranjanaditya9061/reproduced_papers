"""Tests for the embedding stage (config-driven kernel representations + provenance).

Embeddings are computed over the full pool and cached under
``embeddings/<dataset_hash>/<name>_<spec-hash>.pt`` with seed/matched provenance.
"""

from __future__ import annotations

from Generator import generate, load_config
from embedding import build_embedding, build_embeddings, load_embedding


def _setup(tmp_path):
    """Write a data config + embedding config, generate the dataset."""
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(
        "problem: {m: 4, k: 2, observable: parity}\n"
        "generation: {generator: qubit_quantum, size: 100}\n"
        "split: {test_fraction: 0.25, split_seed: 0}\n"
        "seeds: {sample_seed: 1, teacher_seed: 1}\n"
    )
    generate(load_config(data_cfg), out_root=tmp_path)

    embed_cfg = tmp_path / "embed.yaml"
    embed_cfg.write_text(
        f"dataset: {data_cfg.as_posix()}\n"
        "embeddings:\n"
        "  - {type: rbf}\n"
        "  - {type: qubit_projected, depth: 2, seed: 1}\n"    # matched (== teacher_seed)
        "  - {type: qubit_projected, depth: 2, seed: 99}\n"   # unmatched
    )
    return data_cfg, embed_cfg


def test_build_embeddings_and_provenance(tmp_path):
    _, embed_cfg = _setup(tmp_path)
    results, _, meta = build_embeddings(embed_cfg, embeddings_root=tmp_path / "emb",
                                        dataset_root=tmp_path)
    assert len(results) == 3

    by_name = {}
    for r in results:
        by_name.setdefault(r["embedding"].name, []).append(r["blob"])

    # classical kernel: no seed -> matched is None
    assert by_name["rbf"][0]["matched"] is None
    # qubit projected: seed 1 matches teacher_seed=1, seed 99 does not
    matched_flags = sorted(b["matched"] for b in by_name["qubit_projected"])
    assert matched_flags == [False, True]
    # provenance records the dataset seeds distinctly
    proj = by_name["qubit_projected"][0]
    assert proj["sample_seed"] == 1 and proj["teacher_seed"] == 1
    # features are stored over the FULL pool (size 100), not a filtered subset
    assert proj["n"] == 100 and proj["data"].shape[0] == 100


def test_embeddings_saved_under_embeddings_root(tmp_path):
    _, embed_cfg = _setup(tmp_path)
    build_embeddings(embed_cfg, embeddings_root=tmp_path / "emb", dataset_root=tmp_path)
    files = list((tmp_path / "emb").glob("*/*.pt"))
    assert len(files) == 3                       # one per embedding, under embeddings/<hash>/


def test_cache_reuse(tmp_path):
    data_cfg, embed_cfg = _setup(tmp_path)
    build_embeddings(embed_cfg, embeddings_root=tmp_path / "emb", dataset_root=tmp_path)
    # second lookup hits the cache (load_embedding returns the stored blob)
    from Generator import artifact_path, load_raw
    dcfg = load_config(data_cfg)
    h = load_raw(artifact_path(dcfg, tmp_path))[2]["hash"]
    emb = build_embedding({"type": "qubit_projected", "depth": 2, "seed": 1}, dcfg)
    blob = load_embedding(tmp_path / "emb", h, emb)
    assert blob is not None and blob["matched"] is True
