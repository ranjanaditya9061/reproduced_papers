"""Fourier-kernel capacity curve: test R^2 vs Fourier order, one line per dataset.

Complements :mod:`learn.grid`.  The grid fixes the learner and sweeps datasets;
here we fix the learner *type* (Fourier-RBF) and sweep its *dimension* on the
x-axis -- the Fourier order (kernel dim = ``2 * order * n_features``) -- with each
dataset (discovered from a folder, same as the grid) drawn as a coloured line.

Reading it: how much kernel capacity does each dataset's target need?  A curve
that climbs with order needs high-frequency structure; one flat near 0 isn't
learnable by a (band-limited) Fourier kernel at all.

    python -m learn.capacity --configs-dir configs/datasets --orders 1 2 3 4 5 6
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.capacity` and `python learn/capacity.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

from Generator import artifact_path, generate, load_config, load_raw

from .grid import discover_configs
from .svm import _fit_score, _split_indices


def run_capacity(configs_dir, *, orders=(1, 2, 3, 4, 5, 6), n_train=2000, n_test=1000,
                 C=1.0, gamma="scale", epsilon=0.01, embeddings_root="embeddings",
                 dataset_root="datasets", use_cache=True):
    """Return ``(results, orders)`` where results maps dataset label -> [R^2 per order].

    For each dataset, build the ``fourier_rbf`` embedding at each ``order`` through
    the embedding stage (cached), then regress the teacher's ``soft[:, 0]`` with an
    RBF-SVR on those features (same train/test split as the grid).
    """
    from embedding import build_embeddings_for

    dataset_map = discover_configs(configs_dir)
    orders = list(orders)
    results = {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                    # ensure the artifact exists
        soft = load_raw(artifact_path(dcfg, dataset_root))[1]
        t = soft[:, 0]
        tr, te = _split_indices(soft, test_fraction=dcfg.split.test_fraction,
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


def _lineplot(results, orders, axis_label, save_path, *, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, r2s in results.items():
        ax.plot(orders, r2s, marker="o", label=label)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")               # R^2 = 0 -> predicting the mean
    ax.set_xlabel("Fourier order   (kernel dim = 2 · order · n_features)")
    ax.set_ylabel("Test R²")
    ax.set_title(f"Fourier-kernel capacity across {axis_label}")
    ax.set_xticks(orders)
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
    ap.add_argument("--axis-label", default=None, help="legend/title label (default: folder name)")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation (smaller = more regularized)")
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--save-dir", default="img")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    axis_label = args.axis_label or Path(args.configs_dir).name
    results, orders = run_capacity(args.configs_dir, orders=args.orders, n_train=args.n_train,
                                   n_test=args.n_test, C=args.C, gamma=gamma, epsilon=args.epsilon,
                                   use_cache=not args.force)

    w = max(len(lbl) for lbl in results)
    print("  " + f"{'dataset':<{w}}  " + "  ".join(f"o={o:>2}" for o in orders))
    for label, r2s in results.items():
        print(f"  {label:<{w}}  " + "  ".join(f"{v:>5.2f}" for v in r2s))

    _lineplot(results, orders, axis_label, Path(args.save_dir) / f"capacity_{axis_label}.png",
              show=args.show)


if __name__ == "__main__":
    main()
