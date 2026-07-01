"""Cross-evaluation grid: every learner (embedding) on every dataset (teacher).

Produces two complementary views:

- **test R²** (datasets x learners) — regress the row teacher's continuous output
  ``soft[:, 0]`` with the column's kernel (RBF-SVR).  No labelling, no balancing.
- **pairwise geometric difference** ``g(K_row || K_col)`` (learners x learners) —
  a directional, label-free, teacher-free comparison of the kernels themselves
  (the geometric-difference analogue of centered kernel alignment).  Diagonal = 1;
  a small entry means the column kernel adds little expressivity over the row
  kernel.  Computed on the shared X, so it is the same for every dataset.

Read them together: a learner does well on a teacher when its kernel is close
(small g) to the teacher's own kernel — for the quantum teachers, that teacher
kernel is one of the rows/cols of the g matrix (the matched embedding).

    python -m learn.grid --datasets photonic qubit analytical mlp --save-dir img
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.grid` and `python learn/grid.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import torch

from analyzer.metrics import geometric_difference
from Generator import artifact_path, generate, load_config, load_raw
from kernel import gram_from_cache
from sklearn.svm import SVR

from .svm import _split_indices, _tag

#: the learner list — authored ONCE and swept across every dataset (matched seed 42
#: == teacher_seed; 7 = a random/unmatched circuit).  No per-dataset embed config.
DEFAULT_EMBEDDINGS = [
    {"type": "rbf"},
    {"type": "fourier_rbf", "fourier_order": 3},
    {"type": "qubit_projected", "seed": 42},      # same params  -> matched
    {"type": "qubit_projected", "seed": 7},       # random params
    {"type": "photonic_projected", "seed": 42},   # same params  -> matched
    {"type": "photonic_projected", "seed": 7},    # random params
]

#: canonical learner (column) order
FULL_ORDER = [
    "rbf", "fourier_rbf",
    "qubit_projected@42*", "qubit_projected@7",
    "photonic_projected@42*", "photonic_projected@7",
]
FULL_ORDER_NAME = ["RBF Kernel", f"Fourier\nRBF Kernel", f"Matched\nQubit Kernel", f"Random\nQubit Kernel", f"Matched\nPhotonic Kernel", f"Random\nPhotonic Kernel"]

#: the random-quantum 4x4 subset (matched columns dropped)
RANDOM_ORDER = [
    "rbf", "fourier_rbf", "qubit_projected@7",
    "photonic_projected@7",
]
RANDOM_ORDER_NAME = ["RBF Kernel", f"Fourier\nRBF Kernel", f"Random\nQubit Kernel", f"Random\nPhotonic Kernel"]

#: default sweep folder: every *.yaml here becomes one column of the grid, labelled
#: by its config's ``name:`` field.  Override with ``--configs-dir configs/layers`` etc.
DEFAULT_CONFIGS_DIR = "configs/datasets"


def _one_dataset(data_cfg, *, embeddings, n_train, n_test, C, gamma, epsilon,
                 embeddings_root, dataset_root, use_cache):
    """Return ``{learner_name: {test_r2}}`` for one dataset (one data config).

    Regresses each embedding onto the teacher's continuous ``soft[:, 0]`` with an
    RBF-SVR (no labelling, no balancing); reports R^2.
    """
    from embedding import build_embeddings_for

    dcfg = load_config(data_cfg)
    generate(dcfg, out_root=dataset_root)            # ensure the artifact exists

    results, _, _ = build_embeddings_for(
        dcfg, embeddings, embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )
    soft = load_raw(artifact_path(dcfg, dataset_root))[1]
    t = soft[:, 0]                                   # continuous target (no labelling)

    tr, te = _split_indices(soft, test_fraction=dcfg.split.test_fraction,
                            split_seed=dcfg.split.split_seed)
    tr, te = tr[:n_train], te[:n_test]
    t_tr, t_te = t[tr].numpy(), t[te].numpy()

    out = {}
    for r in results:
        F = r["blob"]["data"]
        reg = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=epsilon).fit(F[tr].numpy(), t_tr)
        out[_tag(r["blob"])] = {"test_r2": float(reg.score(F[te].numpy(), t_te))}
    return out


def kernel_g_matrix(data_cfg, embeddings, *, n_gram=600, reg=1e-6,
                    embeddings_root="embeddings", dataset_root="datasets", use_cache=True):
    """Directional kernel x kernel geometric-difference matrix ``g(K_row || K_col)``.

    Purely geometric (no labels, no teacher): every embedding's RBF Gram is built on
    the shared ``X`` and differenced pairwise.  Because ``X`` is the same for every
    dataset (shared sample_seed), this matrix is identical regardless of which dataset
    is passed.  The diagonal is ``g(K, K) = 1``; small off-diagonal => the column
    kernel adds little expressivity over the row kernel.
    """
    import numpy as np
    from embedding import build_embeddings_for

    dcfg = load_config(data_cfg)
    generate(dcfg, out_root=dataset_root)
    results, _, _ = build_embeddings_for(
        dcfg, embeddings, embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )
    soft = load_raw(artifact_path(dcfg, dataset_root))[1]
    tr, _ = _split_indices(soft, test_fraction=dcfg.split.test_fraction,
                           split_seed=dcfg.split.split_seed)
    g_idx = tr[:n_gram]

    blobs = {_tag(r["blob"]): r["blob"] for r in results}
    names = [FULL_ORDER_NAME[i] for i,n in enumerate(FULL_ORDER) if n in blobs]
    ugly_names = [n for n in FULL_ORDER if n in blobs]
    grams = {n: gram_from_cache(blobs[n], idx_a=g_idx) for n in ugly_names}

    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(ugly_names):
        for j, b in enumerate(ugly_names):
            M[i, j] = geometric_difference(grams[a], grams[b], reg=reg)  # g(K_a || K_b)
    return names, M


def discover_configs(configs_dir) -> dict:
    """Ordered ``{printable_label: config_path}`` for every ``*.yaml`` in a folder.

    The label is the config's own ``name:`` field (falls back to the filename stem).
    Files are swept in sorted-filename order, so prefix them (e.g. ``0_...``,
    ``1_...``) to control the sweep order along the axis.
    """
    import yaml

    out = {}
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        label = raw.get("name") or path.stem
        out[label] = str(path)
    return out


def run_grid(dataset_map=None, *, embeddings=None, n_train=1500, n_test=500,
             C=1.0, gamma="scale", epsilon=0.01, embeddings_root="embeddings",
             dataset_root="datasets", use_cache=True):
    """Run each dataset in ``dataset_map`` (``{label: config_path}``) against the
    shared ``embeddings``; return ``(labels, per)`` where per maps label -> {learner
    -> {test_r2}} (RBF-SVR R^2 regressing the teacher's soft output)."""
    dataset_map = discover_configs(DEFAULT_CONFIGS_DIR) if dataset_map is None else dataset_map
    embeddings = embeddings or DEFAULT_EMBEDDINGS
    per_dataset = {}
    for label, path in dataset_map.items():
        per_dataset[label] = _one_dataset(
            path, embeddings=embeddings, n_train=n_train, n_test=n_test,
            C=C, gamma=gamma, epsilon=epsilon, embeddings_root=embeddings_root,
            dataset_root=dataset_root, use_cache=use_cache,
        )
    return list(per_dataset), per_dataset


def _matrix(per_dataset, rows, cols, field):
    """Assemble a (rows x cols) numpy matrix pulling ``field`` from per_dataset."""
    import numpy as np

    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = per_dataset[r].get(c, {}).get(field)
            if v is not None:
                M[i, j] = v
    return M


def _heatmap(M, rows, cols, col_name, title, save_path, *, vmin=None, vmax=None, cmap="viridis",
             fmt="{:.2f}", xlabel=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(cols), 1.5 + 0.7 * len(rows)))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), col_name, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    if xlabel:
        ax.set_xlabel(xlabel)

    # pick black/white text per cell by the cell colour's luminance (works for any cmap)
    cmap_obj = plt.get_cmap(cmap)
    lo = vmin if vmin is not None else np.nanmin(M)
    hi = vmax if vmax is not None else np.nanmax(M)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if v == v:  # not NaN
                r, g, b, _ = cmap_obj((v - lo) / (hi - lo + 1e-12))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="black" if lum > 0.5 else "white", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[grid] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _print_matrix(M, rows, cols, fmt="{:6.3f}"):
    w = max(len(r) for r in rows)
    print(" " * (w + 2) + "  ".join(f"{c[:10]:>10}" for c in cols))
    for i, r in enumerate(rows):
        cells = "  ".join((fmt.format(M[i, j]) if M[i, j] == M[i, j] else f"{'-':>6}").rjust(10)
                          for j in range(len(cols)))
        print(f"{r:>{w}}  {cells}")


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="learn.grid",
                                 description="Sweep a folder of data configs x learners (R^2) + geometric-difference matrix.")
    ap.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR,
                    help="folder of data configs to sweep (each *.yaml -> one x-axis column, labelled by its name:)")
    ap.add_argument("--axis-label", default=None,
                    help="x-axis title for the sweep (default: the folder name)")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-gram", type=int, default=1000, help="rows for the O(N^3) geometric difference")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--reg", type=float, default=1e-3,
                    help="ridge for geometric difference; raise if g(rbf||rbf) drifts above 1")
    ap.add_argument("--save-dir", default="img", help="directory for the heatmap PNGs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    dataset_map = discover_configs(args.configs_dir)
    if not dataset_map:
        raise SystemExit(f"no *.yaml configs found in {args.configs_dir}")
    axis_label = args.axis_label or Path(args.configs_dir).name

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    labels, per = run_grid(dataset_map, n_train=args.n_train, n_test=args.n_test,
                           C=args.C, gamma=gamma, epsilon=args.epsilon, use_cache=not args.force)

    # learners present (keep parallel column name lists)
    full = [(c, nm) for c, nm in zip(FULL_ORDER, FULL_ORDER_NAME) if any(c in per[l] for l in labels)]
    rand = [(c, nm) for c, nm in zip(RANDOM_ORDER, RANDOM_ORDER_NAME) if any(c in per[l] for l in labels)]
    full_cols, full_names = [c for c, _ in full], [nm for _, nm in full]
    rand_cols, rand_names = [c for c, _ in rand], [nm for _, nm in rand]

    r2_full = _matrix(per, labels, full_cols, "test_r2")     # (configs x learners)
    r2_rand = _matrix(per, labels, rand_cols, "test_r2")

    # kernel x kernel geometric difference (label-free; computed once on the first config)
    first_cfg = next(iter(dataset_map.values()))
    knames, g_kernel = kernel_g_matrix(first_cfg, DEFAULT_EMBEDDINGS, n_gram=args.n_gram,
                                       reg=args.reg, use_cache=not args.force)

    print(f"\n=== test R^2 (rows = {axis_label}, cols = learners) ===")
    _print_matrix(r2_full, labels, full_cols)
    print("\n=== pairwise geometric difference  g(K_row || K_col)  [diag = 1] ===")
    _print_matrix(g_kernel, knames, knames)

    sd = Path(args.save_dir)
    # Transposed so the swept configs are the X-AXIS (= axis_label), learners on Y.
    _heatmap(r2_full.T, full_names, labels, labels,
             f"Test R² across {axis_label}", sd / f"grid_r2_{axis_label}_full.png",
             vmin=0.0, vmax=1.0, cmap="RdYlGn", xlabel=axis_label, show=args.show)
    _heatmap(r2_rand.T, rand_names, labels, labels,
             f"Test R² across {axis_label} (random quantum)", sd / f"grid_r2_{axis_label}_random.png",
             vmin=0.0, vmax=1.0, cmap="RdYlGn", xlabel=axis_label, show=args.show)
    _heatmap(g_kernel, knames, knames, knames, "Pairwise geometric difference  g(K_row || K_col)",
             sd / f"grid_geomdiff_{axis_label}.png", vmin=1.0, cmap="RdYlGn_r", fmt="{:.1f}",
             show=args.show)


if __name__ == "__main__":
    main()
