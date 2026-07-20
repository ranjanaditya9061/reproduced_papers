"""Sweep a folder of data configs ordered by PROBLEM SIZE ``(m, k)``, plotted like :mod:`learn.poly_sweep`.

The size analogue of :mod:`learn.poly_sweep`: instead of many polynomials on one dataset, it
takes many datasets (one per problem size) at a fixed observable.  Point it at a folder of data
configs; every ``*.yaml`` is read, **ordered by ``(m, k)``** (then k), generated, and regressed
with the classical Fourier-RBF kernel.

Two stacked panels (poly_sweep style, x-axis = problem size):

- **top** -- the Fourier-RBF kernel's test **R²** (usual ``1 - Σ(y-ŷ)²/Σ(y-ȳ)²``, left axis) and
  its raw **difference** = mean squared prediction error ``mean((y-ŷ)²)`` (right axis).
- **bottom** -- a ggplot-style violin per size of the teacher target ``y`` distribution (its
  spread is the data variance; solid bar = median, dashed = mean).

Keep the observable the same across the folder; ``prod_parity_consecutive`` is a natural fit --
it derives its monomials from ``(m, k)``, so it stays "the same observable" while scaling with
the problem.  ``--observable`` re-scores the saved distribution offline at every size (needs
``generation.save_dist``), so one generated pool can be measured under any observable without
regenerating.

    # each config's own (stored) observable
    python -m learn.grid_size --configs-dir configs/size_sweep --save-dir img
    # re-score every size under one observable, computed offline from the saved distributions
    python -m learn.grid_size --configs-dir configs/size_sweep --observable prod_parity_consecutive
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.grid_size` and `python learn/grid_size.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

from Generator import generate, load_config

from .poly_sweep import FOURIER_RBF, _fit_r2_and_diff
from .svm import _split_indices, load_target

DEFAULT_CONFIGS_DIR = "configs/size_sweep"
DATASET_ROOT = "datasets"


def discover_configs_by_size(configs_dir):
    """Ordered ``(dataset_map, observables)`` for every data ``*.yaml`` in a folder.

    ``dataset_map`` is ``{label: config_path}`` **sorted by problem size ``(m, k)``**; the label
    is ``"m=<m>\\nk=<k>"`` (disambiguated with the config ``name``/stem if two configs share a
    size).  ``observables`` is the set of distinct ``problem.observable`` values seen (so the
    caller can flag a folder that isn't actually a fixed-observable sweep).  Files without a
    ``problem.m`` are skipped (not data configs).
    """
    import yaml

    entries = []
    for path in Path(configs_dir).glob("*.yaml"):
        raw = yaml.safe_load(path.read_text()) or {}
        prob = raw.get("problem") or {}
        if "m" not in prob:
            continue                                         # not a data config -> skip
        m, k = int(prob["m"]), int(prob.get("k", 0))
        entries.append((m, k, str(path), raw.get("name") or path.stem, prob.get("observable")))
    if not entries:
        raise SystemExit(f"no data configs (with problem.m) found in {configs_dir}")
    entries.sort(key=lambda e: (e[0], e[1]))

    dataset_map, observables = {}, set()
    for m, k, path, name, obs in entries:
        label = f"m={m}\nk={k}"
        if label in dataset_map:                             # duplicate size -> disambiguate
            label = f"m={m},k={k}\n{name}"
        dataset_map[label] = path
        observables.add(obs)
    return dataset_map, observables


def run_size_sweep(dataset_map, *, n_train=8000, n_test=2000, C=1.0, gamma="scale", epsilon=0.01,
                   fourier_order=3, observable=None, embeddings_root="embeddings",
                   dataset_root=DATASET_ROOT, use_cache=True):
    """Fit the Fourier-RBF kernel on each size's dataset; return ``(per_size, targets)``.

    ``per_size`` maps each label to ``{test_r2, test_diff, var}`` (R², raw MSE difference, and the
    teacher target variance ``Var(y)``); ``targets`` maps it to the full target vector (numpy).
    Each dataset is generated (cache-keyed) and its Fourier-RBF features built once.  With
    ``observable`` set, the target is the saved distribution **re-scored offline** under that
    observable (needs ``generation.save_dist``) instead of the stored soft -- so one generated
    pool can be measured under any observable without regenerating.  ``prod_parity_consecutive``
    re-scores per size from the persisted ``(m, k)``.
    """
    from embedding import build_embeddings_for

    spec = dict(FOURIER_RBF, fourier_order=fourier_order)
    per_size, targets = {}, {}
    for label, path in dataset_map.items():
        dcfg = load_config(path)
        generate(dcfg, out_root=dataset_root)                # ensure the artifact exists
        results, _, _ = build_embeddings_for(
            dcfg, [spec], embeddings_root=embeddings_root,
            dataset_root=dataset_root, use_cache=use_cache)
        F = results[0]["blob"]["data"]                       # (N, d) Fourier-RBF features
        t = load_target(dcfg, dataset_root, observable=observable)  # None -> stored soft[:, 0]
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_train], te[:n_test]
        r2, diff = _fit_r2_and_diff(F[tr].numpy(), t[tr].numpy(), F[te].numpy(), t[te].numpy(),
                                    C=C, gamma=gamma, epsilon=epsilon)
        tv = t.numpy()
        per_size[label] = {"test_r2": r2, "test_diff": diff, "var": float(tv.var())}
        targets[label] = tv
    return per_size, targets


def _plot(labels, per_size, targets, save_path, *, axis_label, obs, show=False):
    """Two stacked panels: (top) Fourier-RBF R² + raw difference vs size; (bottom) a ggplot-style
    violin of the teacher target ``y`` distribution per size."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = range(len(labels))
    r2 = [per_size[l]["test_r2"] for l in labels]
    diff = [per_size[l]["test_diff"] for l in labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 1.7 * len(labels)), 9))

    # --- top: R² (left axis) and raw difference / MSE (right axis) --------------- #
    c_r2, c_diff = "#1f77b4", "#d62728"
    l1 = ax1.plot(xs, r2, marker="o", color=c_r2, label="Fourier-RBF  test R²")[0]
    ax1.set_ylabel("test R²", color=c_r2)
    ax1.tick_params(axis="y", labelcolor=c_r2)
    ax1.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax1.set_xticks(list(xs), labels)
    ax1.grid(True, axis="y", alpha=0.3)

    axr = ax1.twinx()
    l2 = axr.plot(xs, diff, marker="s", color=c_diff,
                  label="Fourier-RBF  difference  (mean (y-ŷ)²)")[0]
    axr.set_ylabel("raw difference  mean (y-ŷ)²", color=c_diff)
    axr.tick_params(axis="y", labelcolor=c_diff)
    ax1.legend(handles=[l1, l2], loc="best", fontsize=9)
    ax1.set_title(f"Fourier-RBF R² and raw difference vs {axis_label}  [{obs}]")

    # --- bottom: teacher target distribution per size (ggplot-style violins) ----- #
    data = [targets[l] for l in labels]
    parts = ax2.violinplot(data, positions=list(xs), showmeans=True, showmedians=True,
                           showextrema=True, widths=0.8)
    cmap = plt.get_cmap("viridis")
    denom = max(len(labels) - 1, 1)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(cmap(i / denom))
        body.set_edgecolor("black")
        body.set_alpha(0.75)
    for key in ("cmeans", "cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    if "cmeans" in parts:
        parts["cmeans"].set_linestyle("--")            # dashed = mean, solid = median
    ax2.set_xticks(list(xs), labels)
    ax2.set_xlabel(axis_label)
    ax2.set_ylabel("teacher target  y  (soft[:, 0])")
    ax2.set_title("Teacher target distribution per problem size  (solid = median, dashed = mean)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[grid_size] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.grid_size",
        description="Sweep a folder of data configs ordered by problem size (m, k): Fourier-RBF "
                    "R² + raw difference, and the teacher target variance, per size (poly_sweep style).")
    ap.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR,
                    help="folder of data configs; each *.yaml -> one x-axis column, ordered by (m, k)")
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable at every size "
                         "(needs generation.save_dist); default: each config's stored soft. "
                         "e.g. prod_parity_consecutive, prod_parity__M0-1, loop_path_parity__L0-1")
    ap.add_argument("--axis-label", default="problem size", help="x-axis title for the sweep")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation")
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--fourier-order", type=int, default=3)
    ap.add_argument("--save-dir", default="img", help="directory for the output PNG")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    dataset_map, observables = discover_configs_by_size(args.configs_dir)
    if args.observable:                                      # re-score offline: this IS the observable
        obs = args.observable
    else:
        obs = next(iter(observables)) if len(observables) == 1 else "mixed"
        if len(observables) > 1:
            print(f"[grid_size] WARNING: folder mixes observables {sorted(map(str, observables))}; "
                  "hold it constant, or pass --observable to re-score every size the same way")

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    per_size, targets = run_size_sweep(dataset_map, n_train=args.n_train, n_test=args.n_test,
                                       C=args.C, gamma=gamma, epsilon=args.epsilon,
                                       fourier_order=args.fourier_order, observable=args.observable,
                                       use_cache=not args.force)

    labels = list(dataset_map)
    wl = max(len(lbl.replace("\n", ",")) for lbl in labels)
    print(f"\n=== Fourier-RBF per problem size  (observable={obs}) ===")
    print(f"  {'size':>{wl}}  {'test R2':>9}  {'difference':>11}  {'Var(y)':>9}")
    for lbl in labels:
        p = per_size[lbl]
        print(f"  {lbl.replace(chr(10), ','):>{wl}}  {p['test_r2']:>9.3f}  "
              f"{p['test_diff']:>11.4g}  {p['var']:>9.4g}")

    save = None if args.show else str(Path(args.save_dir) / f"grid_size_{obs}.png")
    _plot(labels, per_size, targets, save, axis_label=args.axis_label, obs=obs, show=args.show)


if __name__ == "__main__":
    main()
