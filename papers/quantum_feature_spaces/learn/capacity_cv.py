"""Validated Fourier-kernel capacity curve: per-order best-(C, gamma) test R^2.

Like :mod:`learn.capacity`, but at each Fourier order it **selects (C, gamma)** by a
validation split carved from the training set, then reports the test R^2 of that
choice.  This turns the curve into "best achievable R^2 vs kernel dimension" --
separating the kernel's *representational capacity* from the overfitting you see at
a fixed C (which makes the plain capacity curve decay at high order).

    python -m learn.capacity_cv --configs-dir configs/datasets --orders 1 2 3 4 5

Slower than learn.capacity (it fits ``|C_values| x |gamma_values| + 1`` SVRs per
order per dataset), so keep n_train modest.
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.capacity_cv` and `python learn/capacity_cv.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

from Generator import generate, load_config

from .grid import discover_configs
from .svm import _fit_score, _split_indices, load_target

C_VALUES = [0.1, 1.0, 10.0, 100.0]
GAMMA_VALUES = ["scale", "auto"]


def run_capacity_cv(configs_dir, *, orders=(1, 2, 3, 4, 5, 6), n_train=2000, n_test=1000,
                    C_values=C_VALUES, gamma_values=GAMMA_VALUES, epsilon=0.01,
                    val_fraction=0.25, embeddings_root="embeddings", dataset_root="datasets",
                    use_cache=True, observable=None):
    """Return ``(results, orders, best)``: per dataset, [best-CV test R^2 per order]
    and the chosen ``(C, gamma)`` per order.

    At each order, ``(C, gamma)`` is picked by R^2 on a validation slice of the train
    set; the winner is then refit on the full train set and scored on test.
    """
    from embedding import build_embeddings_for

    dataset_map = discover_configs(configs_dir)
    orders = list(orders)
    results, best = {}, {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)
        t = load_target(dcfg, dataset_root, observable=observable)
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_train], te[:n_test]
        n_fit = int(len(tr) * (1 - val_fraction))
        fit_i, val_i = tr[:n_fit], tr[n_fit:]                    # carve validation from train
        t_fit, t_val = t[fit_i].numpy(), t[val_i].numpy()
        t_tr, t_te = t[tr].numpy(), t[te].numpy()

        r2s, picks = [], []
        for order in orders:
            res, _, _ = build_embeddings_for(
                dcfg, [{"type": "fourier_rbf", "fourier_order": int(order)}],
                embeddings_root=embeddings_root, dataset_root=dataset_root, use_cache=use_cache,
            )
            F = res[0]["blob"]["data"]
            F_fit, F_val = F[fit_i].numpy(), F[val_i].numpy()

            # select (C, gamma) by validation R^2
            best_val, best_cg = float("-inf"), (C_values[0], gamma_values[0])
            for C in C_values:
                for g in gamma_values:
                    val_r2 = _fit_score(F_fit, t_fit, F_val, t_val, C=C, gamma=g, epsilon=epsilon)[1]
                    if val_r2 > best_val:
                        best_val, best_cg = val_r2, (C, g)

            # refit the winner on the full train set, score on test
            C, g = best_cg
            test_r2 = _fit_score(F[tr].numpy(), t_tr, F[te].numpy(), t_te,
                                 C=C, gamma=g, epsilon=epsilon)[1]
            r2s.append(test_r2)
            picks.append(best_cg)
        results[label] = r2s
        best[label] = picks
    return results, orders, best


def _lineplot(results, orders, axis_label, save_path, *, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, r2s in results.items():
        ax.plot(orders, r2s, marker="o", label=label)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Fourier order   (kernel dim = 2 · order · n_features)")
    ax.set_ylabel("Test R²  (best C, γ per order)")
    ax.set_title(f"Validated Fourier-kernel capacity across {axis_label}")
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)
    ax.legend(title=axis_label)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[capacity_cv] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.capacity_cv",
        description="Validated Fourier-kernel capacity (best C,gamma per order), one line per dataset.")
    ap.add_argument("--configs-dir", default="configs/datasets")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--axis-label", default=None)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--C-values", type=float, nargs="+", default=C_VALUES)
    ap.add_argument("--gamma-values", nargs="+", default=list(GAMMA_VALUES),
                    help="'scale', 'auto', or floats")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable instead of "
                         "the stored soft (spoqc_magic + generation.save_dist only)")
    ap.add_argument("--save-dir", default="img")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    gamma_values = [float(g) if g.replace(".", "", 1).isdigit() else g for g in args.gamma_values]
    axis_label = args.axis_label or Path(args.configs_dir).name
    obs_tag = f"_{args.observable}" if args.observable else ""
    results, orders, best = run_capacity_cv(
        args.configs_dir, orders=args.orders, n_train=args.n_train, n_test=args.n_test,
        C_values=args.C_values, gamma_values=gamma_values, epsilon=args.epsilon,
        val_fraction=args.val_fraction, use_cache=not args.force, observable=args.observable)

    w = max(len(lbl) for lbl in results)
    print("  " + f"{'dataset':<{w}}  " + "  ".join(f"o={o:>2}" for o in orders))
    for label, r2s in results.items():
        print(f"  {label:<{w}}  " + "  ".join(f"{v:>5.2f}" for v in r2s))
    print("\n  best (C, gamma) per order:")
    for label, picks in best.items():
        print(f"  {label:<{w}}  " + "  ".join(f"{C:g}/{g}" for C, g in picks))

    _lineplot(results, orders, axis_label, Path(args.save_dir) / f"capacity_cv_{axis_label}{obs_tag}.png",
              show=args.show)


if __name__ == "__main__":
    main()
