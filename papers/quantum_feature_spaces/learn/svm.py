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
from Generator.generate import DIST_FILENAME


def load_target(dcfg, dataset_root, *, observable: str | None = None) -> torch.Tensor:
    """Continuous regression target for a dataset, as a 1-D float tensor of length N.

    Default (``observable=None``): the stored teacher output ``soft[:, 0]``.  When an
    ``observable`` is given, re-score the saved full distribution (``distributions.npz``,
    written by a distribution-capturing teacher with ``generation.save_dist``) under that
    observable instead -- so a single generated dataset can be swept across *different*
    measurements without regenerating.  Raises if no distribution file is present.

    The ``.npz`` format is shared, but the re-scorer is teacher-specific: ``photonic_quantum``
    uses :func:`model.photonic.score_from_distribution` (for a ``loop_path_<base>`` override the
    graph comes from ``dcfg.problem``'s ``n_vertices`` / ``graph_seed``, for a
    ``connected_<base>`` from ``graph_density`` / ``graph_seed``, while the base scorer (and the
    loop/path selection) ride in the observable string itself, e.g. ``loop_path_parity__L0-1__P2-3``
    or ``connected_maxcc``); every other
    teacher uses :func:`model.spoqc_magic.score_from_distribution`.
    """
    if observable is None:
        return load_raw(artifact_path(dcfg, dataset_root))[1][:, 0]

    from model.spoqc_magic import load_distributions

    dpath = artifact_path(dcfg, dataset_root) / DIST_FILENAME
    if not dpath.exists():
        raise FileNotFoundError(
            f"observable override {observable!r} needs a saved distribution, but "
            f"{dpath} is missing -- regenerate the dataset with generation.save_dist: true"
        )
    dist = load_distributions(dpath)

    if dcfg.generation.generator == "photonic_quantum":
        from model.photonic import (is_connected_observable, is_graph_observable,
                                     is_prod_parity_angle)
        from model.photonic import score_from_distribution as photonic_score

        if is_graph_observable(observable) or is_connected_observable(observable):
            # Both graph families need a fixed, seeded graph from the dataset config; the base
            # scorer *and* the selection ride in the observable string itself.  loop_path_<base>
            # maps modes to edges of a graph on n_vertices (loop/path __L/__P selection);
            # connected_<base> maps modes to the V=m vertices of a graph of density graph_density
            # and scores a global property (maxcc = largest connected component).
            p = dcfg.problem
            scores = photonic_score(dist, observable, n_vertices=p.n_vertices,
                                    graph_density=p.graph_density, graph_seed=p.graph_seed)
        elif is_prod_parity_angle(observable):
            # An angle prod_parity variant: its per-monomial angles ride on angle_seed (from the
            # dataset config; None -> the stored teacher seed, matching the generator's default).
            scores = photonic_score(dist, observable, angle_seed=dcfg.problem.angle_seed)
        else:
            scores = photonic_score(dist, observable)
    else:
        from model.spoqc_magic import score_from_distribution as magic_score

        scores = magic_score(dist, observable)
    return torch.tensor(scores, dtype=torch.float32)


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


def _fit_score(F_tr, t_tr, F_te, t_te, *, C, gamma, epsilon):
    """Standardize features **and** target on TRAIN stats, fit RBF-SVR, return
    ``(train_r2, test_r2)``.

    Per-feature standardization tames the single-``gamma`` RBF; standardizing the
    target makes ``C``/``epsilon`` mean the same thing across datasets whose ``soft``
    live on very different scales.  Stats are fit on train only (no leakage), and
    R^2 is scale/shift-invariant so it stays comparable to the raw-target R^2.
    """
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(F_tr)
    Ftr, Fte = xs.transform(F_tr), xs.transform(F_te)
    mu, sd = float(t_tr.mean()), (float(t_tr.std()) or 1.0)
    ytr, yte = (t_tr - mu) / sd, (t_te - mu) / sd
    reg = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=epsilon).fit(Ftr, ytr)
    return float(reg.score(Ftr, ytr)), float(reg.score(Fte, yte))


def _fit_score_rank(F_tr, t_tr, F_te, t_te, *, D, gamma="scale", alpha=1e-2, seed=0,
                    kernel="nystroem", F_lin_tr=None, F_lin_te=None):
    """RBF-dynamics-as-features + a LINEAR ridge readout; returns ``(train_r2, test_r2)``.

    The *model-size* analogue of :func:`_fit_score`: the capacity knob is ``D``, the
    number of explicit RBF features, at fixed dataset size.  Both back-ends realize the
    same object -- linear ridge on a finite-``D`` approximation of the RBF feature map,
    i.e. RBF kernel ridge -- so the test-R^2 curve climbs monotonically toward exact
    RBF kernel ridge and then *saturates*, with ``alpha`` keeping it from a bias/variance
    U-turn:

    - ``kernel='nystroem'`` (default): data-dependent landmarks; ``D`` is clamped to
      ``n_train`` (a Nystrom rank cannot exceed the sample count).
    - ``kernel='rff'``: random Fourier features (``RBFSampler``); ``D`` is unbounded, so
      it can exceed ``n_train`` and push further up the curve.

    **Junk separation.**  The RBF dynamics are built *only* on ``F`` -- keep that a
    clean feature map (``combo`` is).  ``F_lin`` (optional) bypasses the kernel and is
    concatenated straight into the linear ridge, so redundant/uninformative ("junk")
    columns are down-weighted by ridge instead of dilating the RBF distance (which the
    kernel cannot undo).  So: kernel sees ``F`` (clean); the linear layer trains on the
    RBF features *plus* ``F_lin`` (junk and all).  Same TRAIN-only feature/target
    standardization as :func:`_fit_score`; ``gamma='scale'`` maps to ``1/n_features``.
    """
    import numpy as np
    from sklearn.kernel_approximation import Nystroem, RBFSampler
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(F_tr)
    Ftr, Fte = xs.transform(F_tr), xs.transform(F_te)
    mu, sd = float(t_tr.mean()), (float(t_tr.std()) or 1.0)
    ytr, yte = (t_tr - mu) / sd, (t_te - mu) / sd

    g = 1.0 / Ftr.shape[1] if gamma in ("scale", "auto") else float(gamma)
    if kernel == "linear":
        Ptr, Pte = Ftr, Fte                             # no kernel step: plain linear ridge on F
    elif kernel == "rff":
        approx = RBFSampler(n_components=int(D), gamma=g, random_state=seed)
        Ptr, Pte = approx.fit(Ftr).transform(Ftr), approx.transform(Fte)
    elif kernel == "nystroem":
        d = min(int(D), Ftr.shape[0])                   # Nystrom rank <= n_train
        approx = Nystroem(kernel="rbf", gamma=g, n_components=d, random_state=seed)
        Ptr, Pte = approx.fit(Ftr).transform(Ftr), approx.transform(Fte)
    else:
        raise ValueError(f"kernel must be 'nystroem', 'rff', or 'linear', got {kernel!r}")

    if F_lin_tr is not None:                            # junk bypasses the kernel -> linear layer
        ls = StandardScaler().fit(F_lin_tr)
        Ptr = np.hstack([Ptr, ls.transform(F_lin_tr)])
        Pte = np.hstack([Pte, ls.transform(F_lin_te)])

    reg = Ridge(alpha=alpha).fit(Ptr, ytr)              # ridge keeps the curve monotone + kills junk
    return float(reg.score(Ptr, ytr)), float(reg.score(Pte, yte))


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
    observable: str | None = None,
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
    t = load_target(dcfg, dataset_root, observable=observable)   # soft[:,0] or re-scored

    tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                            split_seed=dcfg.split.split_seed)
    tr, te = tr[:n_train], te[:n_test]
    t_tr, t_te = t[tr].numpy(), t[te].numpy()

    rows = []
    for r in results:
        F = r["blob"]["data"]
        train_r2, test_r2 = _fit_score(F[tr].numpy(), t_tr, F[te].numpy(), t_te,
                                       C=C, gamma=gamma, epsilon=epsilon)
        rows.append({
            "name": _tag(r["blob"]),
            "dim": int(F.shape[1]),
            "n_train": len(tr),
            "n_test": len(te),
            "train_r2": train_r2,
            "test_r2": test_r2,
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
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation (smaller = more regularized)")
    ap.add_argument("--gamma", default="scale", help="RBF gamma ('scale', 'auto', or a float)")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--n-train", type=int, default=8000, help="cap training rows (O(N^2) SVR)")
    ap.add_argument("--n-test", type=int, default=2000, help="cap test rows")
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable instead of "
                         "the stored soft (spoqc_magic + generation.save_dist only)")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    rows, meta = run_svm(
        args.config, C=args.C, gamma=gamma, epsilon=args.epsilon,
        n_train=args.n_train, n_test=args.n_test,
        embeddings_root=args.embeddings_root, dataset_root=args.dataset_root,
        use_cache=not args.force, observable=args.observable,
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
