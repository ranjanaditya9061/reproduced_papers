"""Build embeddings for a dataset: ``python -m embedding --config <embed.yaml>``."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "embedding"

from .store import build_embeddings


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute + save kernel embeddings for a dataset.")
    ap.add_argument("--config", required=True, help="embedding config (references a data config)")
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args(argv)

    results, _, meta = build_embeddings(
        args.config, embeddings_root=args.embeddings_root,
        dataset_root=args.dataset_root, use_cache=not args.force,
    )
    print(f"[embedding] dataset {meta['hash']}  (sample_seed={meta.get('sample_seed')}, "
          f"teacher_seed={meta.get('teacher_seed')})")
    print(f"[embedding] built {len(results)} embeddings -> {args.embeddings_root}/{meta['hash']}/")
    for r in results:
        b = r["blob"]
        tag = "" if b["matched"] is None else ("  [MATCHED]" if b["matched"] else "  [unmatched]")
        seed = "" if b["embedding_seed"] is None else f" seed={b['embedding_seed']}"
        dim = "x".join(str(d) for d in b["data"].shape)
        print(f"    {r['embedding'].name:<18} feats={dim:<10}{seed}{tag}")


if __name__ == "__main__":
    main()
