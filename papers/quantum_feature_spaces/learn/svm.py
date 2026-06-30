"""Regress each embedding onto the teacher's continuous output (RBF-SVR, R^2).

No labelling, no balancing: the target is the teacher's raw ``soft[:, 0]`` and the
learner is an RBF-kernel *regressor* (:class:`sklearn.svm.SVR`), scored by ``R^2``.
This avoids the threshold/class-imbalance issues of hard labels entirely.

Pipeline (per the staged design):

1. :func:`embedding.build_embeddings` -> the feature matrix of every embedding,
   computed over the **full** pool and cached.
2. the target ``t = soft[:, 0]`` comes straight from the teacher output (no labels).
3. the shared train/test partition comes from ``split_seed`` / ``test_fraction``
   (the same convention as :func:`Generator.load_split`), as row indices.
4. for each embedding, slice ``F[train_idx]`` / ``F[test_idx]`` and fit an SVR.

``--n-train`` / ``--n-test`` cap the rows, because RBF-SVR is roughly O(N^2).
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn` and `python learn/svm.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import torch
from sklearn.svm import SVR

from Generator import artifact_path, load_raw


def _split_indices(soft, *, test_fraction, split_seed):
    """(train_idx, test_idx): a seeded shuffle of the full pool (no labelling).

    Same convention as ``Generator.load_split``; nothing is filtered or balanced.
    """
    n = soft.shape[0]
    pool = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
    n_test = int(n * test_fraction)
    return pool[n_test:], pool[:n_test]


def _tag(blob) -> str:
    """Embedding name annotated with its seed and matched/unmatched (if any)."""
    nm = blob["spec"].get("name", "?")
    seed = blob.get("embedding_seed")
    if seed is not None:
        nm += f"@{seed}" + ("*" if blob.get("matched") else "")
    return nm


def run_svm(
    embed_config_path,
    *,
    C: float = 1.0,
    gamma="scale",
    epsilon: float = 0.01,
    n_train: int = 8000,
    n_test: int = 2000,
    embeddings_root="embeddings",
    dataset_root="datasets",
    use_cache: bool = True,
):
    """Fit an RBF-SVR per embedding to **regress the teacher's continuous output**.

    No labelling: the target is ``soft[:, 0]`` (the signed score for an ``(N, 1)``
    teacher, or ``P(class 0)`` for an ``(N, c)`` one), and the score reported is
    ``R^2`` (chance = 0 = predicting the mean).  This sidesteps thresholding and
    class (im)balance entirely.  Each row: ``{name, dim, n_train, n_test,
    train_r2, test_r2}``.
    """
    from embedding import build_embeddings

    results, dcfg, meta = build_embeddings(
        embed_config_path, embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )
    soft = load_raw(artifact_path(dcfg, dataset_root))[1]
    t = soft[:, 0]                                   # continuous target (no labelling)

    tr, te = _split_indices(soft, test_fraction=dcfg.split.test_fraction,
                            split_seed=dcfg.split.split_seed)
    tr, te = tr[:n_train], te[:n_test]
    t_tr, t_te = t[tr].numpy(), t[te].numpy()

    rows = []
    for r in results:
        F = r["blob"]["data"]
        F_tr, F_te = F[tr].numpy(), F[te].numpy()
        reg = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=epsilon)
        reg.fit(F_tr, t_tr)
        rows.append({
            "name": _tag(r["blob"]),
            "dim": int(F.shape[1]),
            "n_train": len(tr),
            "n_test": len(te),
            "train_r2": float(reg.score(F_tr, t_tr)),
            "test_r2": float(reg.score(F_te, t_te)),
        })
    return rows, meta


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn",
        description="Regress each embedding onto the teacher's continuous output (RBF-SVR, R^2).",
    )
    ap.add_argument("--config", required=True, help="embedding config (references a data config)")
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation")
    ap.add_argument("--gamma", default="scale", help="RBF gamma ('scale', 'auto', or a float)")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--n-train", type=int, default=8000, help="cap training rows (O(N^2) SVR)")
    ap.add_argument("--n-test", type=int, default=2000, help="cap test rows")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    rows, meta = run_svm(
        args.config, C=args.C, gamma=gamma, epsilon=args.epsilon,
        n_train=args.n_train, n_test=args.n_test,
        embeddings_root=args.embeddings_root, dataset_root=args.dataset_root,
        use_cache=not args.force,
    )

    n_tr = rows[0]["n_train"] if rows else 0
    n_te = rows[0]["n_test"] if rows else 0
    print(f"[learn] dataset {meta['hash']}  RBF-SVR C={args.C} eps={args.epsilon}  "
          f"n_train={n_tr} n_test={n_te}  (target = soft[:,0]; score = R^2)\n")

    w = max((len(r["name"]) for r in rows), default=4)
    print(f"  {'embedding':>{w}}  {'dim':>4}  {'train R2':>9}  {'test R2':>8}")
    for r in sorted(rows, key=lambda r: r["test_r2"], reverse=True):
        print(f"  {r['name']:>{w}}  {r['dim']:>4}  {r['train_r2']:>9.3f}  {r['test_r2']:>8.3f}")


if __name__ == "__main__":
    main()
