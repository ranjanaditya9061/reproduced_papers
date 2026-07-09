"""Fourier-kernel capacity curve: test R^2 vs Fourier order, one line per dataset.

Complements :mod:`learn.grid`.  The grid fixes the learner and sweeps datasets;
here we fix the learner *type* (Fourier-RBF) and sweep its *dimension* on the
x-axis -- the Fourier order (kernel dim = ``2 * order * n_features``) -- with each
dataset (discovered from a folder, same as the grid) drawn as a coloured line.

Reading it: how much kernel capacity does each dataset's target need?  A curve
that climbs with order needs high-frequency structure; one flat near 0 isn't
learnable by a (band-limited) Fourier kernel at all.

    python -m learn.capacity --configs-dir configs/datasets --orders 1 2 3 4 5 6

Passing ``--ranks`` instead switches the x-axis to *model size* at fixed dataset
size: a single fixed feature map (``--embedding-type``, default ``rbf``) with the
RBF kernel approximated at growing **Nystrom rank D**, fit by ridge.  Increasing D
climbs monotonically toward exact RBF kernel ridge and saturates (no overfitting
U-turn), so the plateau reads off the intrinsic kernel capacity the target needs.

    python -m learn.capacity --configs-dir configs/datasets --ranks 2 4 8 16 32 64 128
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.capacity` and `python learn/capacity.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import numpy as np

from Generator import generate, load_config

from .grid import discover_configs
from .svm import _fit_score, _fit_score_rank, _split_indices, load_target


def run_capacity(configs_dir, *, orders=(1, 2, 3, 4, 5, 6), n_train=2000, n_test=1000,
                 C=1.0, gamma="scale", epsilon=0.01, embeddings_root="embeddings",
                 dataset_root="datasets", use_cache=True, observable=None):
    """Return ``(results, orders)`` where results maps dataset label -> [R^2 per order].

    For each dataset, build the ``fourier_rbf`` embedding at each ``order`` through
    the embedding stage (cached), then regress the teacher's continuous target
    (``soft[:, 0]``, or the saved distribution re-scored under ``observable``) with an
    RBF-SVR on those features (same train/test split as the grid).
    """
    from embedding import build_embeddings_for

    dataset_map = discover_configs(configs_dir)
    orders = list(orders)
    results = {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                    # ensure the artifact exists
        t = load_target(dcfg, dataset_root, observable=observable)
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_train], te[:n_test]
        t_tr, t_te = t[tr].numpy(), t[te].numpy()

        r2s = []
        for order in orders:
            res, _, _ = build_embeddings_for(
                dcfg, [{"type": "fourier_rbf", "fourier_order": int(order)}],
                embeddings_root=embeddings_root, dataset_root=dataset_root, use_cache=use_cache,
            )
            F = res[0]["blob"]["data"]                           # cached fourier features (full X)
            _, test_r2 = _fit_score(F[tr].numpy(), t_tr, F[te].numpy(), t_te,
                                    C=C, gamma=gamma, epsilon=epsilon)
            r2s.append(test_r2)
        results[label] = r2s
    return results, orders


def run_capacity_rank(configs_dir, *, ranks=(2, 4, 8, 16, 32, 64, 128, 256),
                      embedding=None, n_train=8000, n_test=2000, gamma="scale",
                      alpha=1e-2, n_seeds=3, kernel="nystroem",
                      embeddings_root="embeddings", dataset_root="datasets",
                      use_cache=True, observable=None):
    """Return ``(results, ranks)``: test R^2 vs RBF-feature count D, one line per dataset.

    The *model-size* companion to :func:`run_capacity`.  Instead of growing the
    Fourier order, this fixes the feature map (``embedding`` spec, default the raw
    ``rbf`` angles) and sweeps ``D``, the number of explicit RBF features (``kernel``:
    ``nystroem`` landmarks or ``rff`` random Fourier features) fed to a linear ridge --
    a genuine capacity axis at *fixed* dataset size.  Each D's R^2 is averaged over
    ``n_seeds`` random draws for a smooth, climb-then-saturate curve.  See
    :func:`learn.svm._fit_score_rank`.
    """
    from embedding import build_embeddings_for

    spec = embedding or {"type": "rbf"}
    dataset_map = discover_configs(configs_dir)
    ranks = list(ranks)
    results = {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                    # ensure the artifact exists
        t = load_target(dcfg, dataset_root, observable=observable)
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_train], te[:n_test]
        t_tr, t_te = t[tr].numpy(), t[te].numpy()

        res, _, _ = build_embeddings_for(                        # one fixed feature map, cached
            dcfg, [spec], embeddings_root=embeddings_root,
            dataset_root=dataset_root, use_cache=use_cache,
        )
        F = res[0]["blob"]["data"]
        F_tr, F_te = F[tr].numpy(), F[te].numpy()

        r2s = []
        for D in ranks:
            seed_r2 = [_fit_score_rank(F_tr, t_tr, F_te, t_te, D=int(D), gamma=gamma,
                                       alpha=alpha, seed=s, kernel=kernel)[1]
                       for s in range(n_seeds)]
            r2s.append(float(np.mean(seed_r2)))                  # average over random draws
        results[label] = r2s
    return results, ranks


def _lineplot(results, xs, axis_label, save_path, *, show=False,
              xlabel="Fourier order   (kernel dim = 2 · order · n_features)",
              title="Fourier-kernel capacity"):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, r2s in results.items():
        ax.plot(xs, r2s, marker="o", label=label)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")               # R^2 = 0 -> predicting the mean
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test R²")
    ax.set_title(f"{title} across {axis_label}")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(title=axis_label)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[capacity] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.capacity",
        description="Fourier-kernel capacity curve (R^2 vs Fourier order), one line per dataset.")
    ap.add_argument("--configs-dir", default="configs/datasets",
                    help="folder of data configs (each *.yaml -> one line, labelled by its name:)")
    ap.add_argument("--orders", type=int, nargs="+", default=[1,2,3,4,5,6,7,8,9,10],
                    help="Fourier orders to sweep on the x-axis")
    ap.add_argument("--ranks", type=int, nargs="+", default=None,
                    help="if given, sweep Nystrom kernel rank D (model-size axis, RBF "
                         "Kernel Ridge) instead of Fourier order -- climbs then saturates")
    ap.add_argument("--embedding-type", default="rbf",
                    help="feature map the RBF kernel is built on for the rank sweep")
    ap.add_argument("--kernel", default="nystroem", choices=["nystroem", "rff"],
                    help="RBF-feature back-end for the rank sweep: 'nystroem' (D<=n_train) "
                         "or 'rff' random Fourier features (D unbounded)")
    ap.add_argument("--alpha", type=float, default=1e-2, help="ridge penalty (rank sweep)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Nystrom landmark draws to average per rank (rank sweep)")
    ap.add_argument("--axis-label", default=None, help="legend/title label (default: folder name)")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation (smaller = more regularized)")
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable instead of "
                         "the stored soft (needs generation.save_dist). Photonic graph "
                         "observables may encode the selection, e.g. "
                         "loop_path_parity__L0-1__P2-3")
    ap.add_argument("--save-dir", default="img")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    axis_label = args.axis_label or Path(args.configs_dir).name
    obs_tag = f"_{args.observable}" if args.observable else ""

    if args.ranks:                                               # model-size axis: Nystrom rank D
        results, xs = run_capacity_rank(
            args.configs_dir, ranks=args.ranks, embedding={"type": args.embedding_type},
            n_train=args.n_train, n_test=args.n_test, gamma=gamma, alpha=args.alpha,
            n_seeds=args.seeds, kernel=args.kernel, use_cache=not args.force,
            observable=args.observable)
        xlabel = f"RBF features D   ({args.kernel}; feature map: {args.embedding_type})"
        title, col = f"RBF-dynamics ({args.kernel}) + linear ridge capacity", "D"
        save_path = Path(args.save_dir) / f"capacity_rank_{axis_label}{obs_tag}.png"
    else:                                                        # capacity axis: Fourier order
        results, xs = run_capacity(
            args.configs_dir, orders=args.orders, n_train=args.n_train,
            n_test=args.n_test, C=args.C, gamma=gamma, epsilon=args.epsilon,
            use_cache=not args.force, observable=args.observable)
        xlabel = "Fourier order   (kernel dim = 2 · order · n_features)"
        title, col = "Fourier-kernel capacity", "o"
        save_path = Path(args.save_dir) / f"capacity_{axis_label}{obs_tag}.png"

    w = max(len(lbl) for lbl in results)
    print("  " + f"{'dataset':<{w}}  " + "  ".join(f"{col}={x:>3}" for x in xs))
    for label, r2s in results.items():
        print(f"  {label:<{w}}  " + "  ".join(f"{v:>5.2f}" for v in r2s))

    _lineplot(results, xs, axis_label, save_path, show=args.show, xlabel=xlabel, title=title)


if __name__ == "__main__":
    main()
