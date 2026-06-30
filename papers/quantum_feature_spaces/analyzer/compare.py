"""Kernel-kernel matrix for a set of embeddings on a dataset.

Each embedding's Gram is a Gaussian RBF over its feature matrix (:mod:`kernel`).
Two entry points:
- :func:`compare_kernels` -- ad-hoc, from an explicit embedding list (computes
  Grams directly, no caching).
- :func:`compare_from_embeddings` -- config-driven, from an *embedding config*: it
  builds/loads the cached embeddings (:mod:`embedding`) and RBFs their stored
  features, labelling each by seed and whether it matched the teacher.
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m analyzer.compare` and `python analyzer/compare.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "analyzer"

from Generator import artifact_path, derive_labels, load_config, load_raw, prepare_indices
from kernel import RBFGram, gram_from_cache

from .metrics import kernel_kernel_matrix, target_alignment


def kernel_grams(config_path, embeddings, out_root="datasets"):
    """Ad-hoc: RBF Grams of an explicit embedding list over the dataset's full X."""
    cfg = load_config(config_path)
    path = artifact_path(cfg, out_root)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset artifact at {path}.  Generate it:  "
            f"python -m Generator --config {config_path}"
        )
    X = load_raw(path)[0]
    return [RBFGram().gram_from_features(e.features(X)) for e in embeddings], X


def compare_kernels(config_path, embeddings, names=None, out_root="datasets"):
    """Return ``(names, matrix)`` from an explicit embedding list."""
    grams, _ = kernel_grams(config_path, embeddings, out_root=out_root)
    if names is None:
        names = [getattr(e, "name", f"k{i}") for i, e in enumerate(embeddings)]
    return names, kernel_kernel_matrix(grams)


def compare_from_embeddings(embed_config_path, embeddings_root="embeddings",
                            dataset_root="datasets", use_cache=True):
    """Config-driven kernel-kernel matrix; names are auto-tagged with seed/matched."""
    from embedding import build_embeddings

    results, _, _ = build_embeddings(
        embed_config_path, embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )
    grams = [gram_from_cache(r["blob"]) for r in results]
    names = []
    for r in results:
        b, nm = r["blob"], r["embedding"].name
        if b["embedding_seed"] is not None:
            nm += f"@{b['embedding_seed']}" + ("*" if b["matched"] else "")
        names.append(nm)
    return names, kernel_kernel_matrix(grams)


def print_matrix(names, matrix) -> None:
    w = max(len(n) for n in names)
    print(" " * (w + 2) + "  ".join(f"{n[:8]:>8}" for n in names))
    for i, n in enumerate(names):
        row = "  ".join(f"{float(matrix[i, j]):8.3f}" for j in range(len(names)))
        print(f"{n:>{w}}  {row}")


def _grams_and_names(results, idx):
    """RBF Gram + a seed/matched-tagged name for each built embedding result.

    ``idx`` is the index tensor selecting which rows of the stored (full) feature
    matrix to use -- the kernel is O(N^2), so large datasets must be subsampled.
    """
    grams, names = [], []
    for r in results:
        b = r["blob"]
        grams.append(gram_from_cache(b, idx_a=idx))
        nm = r["embedding"].name
        if b["embedding_seed"] is not None:
            nm += f"@{b['embedding_seed']}" + ("*" if b["matched"] else "")
        names.append(nm)
    return grams, names


def main(argv=None) -> None:
    """``python -m analyzer compare --config <embed.yaml> [opts]``.

    By default the Gram is built over the full pool (capped by ``-n``).  The
    ``--min-margin`` / ``--balanced`` flags opt into the separability *diagnostic*:
    restrict the rows to the margin-filtered / class-balanced subset first.
    """
    import argparse

    import torch

    from embedding import build_embeddings

    ap = argparse.ArgumentParser(
        prog="analyzer compare",
        description="Compare a dataset's kernel embeddings via centered kernel alignment.",
    )
    ap.add_argument("--config", required=True,
                    help="embedding config (references a data config)")
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("-n", "--n", type=int, default=2000,
                    help="cap rows used for the (O(N^2)) Gram; 0 = use all")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="diagnostic: keep only samples with confidence >= this")
    ap.add_argument("--balanced", action="store_true",
                    help="diagnostic: class-balance the rows before comparing")
    ap.add_argument("--force", action="store_true",
                    help="recompute embeddings instead of loading the cache")
    ap.add_argument("--target", action="store_true",
                    help="also report each kernel's alignment with the labels")
    args = ap.parse_args(argv)

    results, dcfg, meta = build_embeddings(
        args.config, embeddings_root=args.embeddings_root,
        dataset_root=args.dataset_root, use_cache=not args.force,
    )
    avail = results[0]["blob"]["n"] if results else 0
    soft = load_raw(artifact_path(dcfg, args.dataset_root))[1]

    diagnostic = args.min_margin > 0.0 or args.balanced
    pool = (prepare_indices(soft, min_margin=args.min_margin, balanced=args.balanced,
                            seed=dcfg.split.split_seed)
            if diagnostic else torch.arange(avail))

    use = len(pool) if args.n in (0, None) else min(args.n, len(pool))
    idx = pool[:use]

    grams, names = _grams_and_names(results, idx)

    diag = ""
    if diagnostic:
        bits = [f"min_margin={args.min_margin}"] + (["balanced"] if args.balanced else [])
        diag = " [diagnostic: " + ", ".join(bits) + "]"
    note = "" if use == avail else f"  ({use} of {avail}; raise with -n / -n 0 for all)"
    print(f"[compare] dataset {meta['hash']}  ({len(names)} kernels, n={use}){diag}{note}")
    print("\nCentered kernel alignment  (1.000 = identical geometry; * = matched teacher seed):\n")
    print_matrix(names, kernel_kernel_matrix(grams))

    if args.target:
        y = derive_labels(soft)[idx]
        w = max(len(nm) for nm in names)
        print("\nKernel-target alignment  (higher -> more learnable for that kernel):\n")
        for nm, K in zip(names, grams):
            print(f"  {nm:>{w}}  {target_alignment(K, y):.4f}")


if __name__ == "__main__":
    main()
