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

#: dataset short-name -> data config (the only thing that varies across the grid)
DEFAULT_DATASETS = {
    "photonic": "configs/data_photonic.yaml",
    "photonic_parity": "configs/data_photonic_parity.yaml",
    "qubit": "configs/data_qubit.yaml",
    "analytical": "configs/data_analytical.yaml",
    "mlp": "configs/data_mlp.yaml",
    "spoqc_photonic": "configs/data_spoqc_photonic.yaml",
    "spoqc_photonic_parity": "configs/data_spoqc_photonic_parity.yaml",
    "spoqc_photonic_one": "configs/data_spoqc_photonic_one.yaml",
    "spoqc_photonic_two": "configs/data_spoqc_photonic_two.yaml",
    "spoqc_photonic_three": "configs/data_spoqc_photonic_three.yaml",
    "spoqc_photonic_four": "configs/data_spoqc_photonic_four.yaml",
    "spoqc_photonic_layer_2": "configs/data_spoqc_photonic_layer_2.yaml",
    "spoqc_photonic_layer_3": "configs/data_spoqc_photonic_layer_3.yaml",
    "spoqc_photonic_layer_4": "configs/data_spoqc_photonic_layer_4.yaml"
}


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


def run_grid(datasets=None, *, embeddings=None, n_train=1500, n_test=500,
             C=1.0, gamma="scale", epsilon=0.01, embeddings_root="embeddings",
             dataset_root="datasets", use_cache=True):
    """Run every dataset against the shared ``embeddings`` list; return
    ``(row_labels, per_dataset)`` where per_dataset maps dataset -> {learner ->
    {test_r2}} (RBF-SVR R^2 regressing the teacher's soft output)."""
    datasets = datasets or list(DEFAULT_DATASETS)
    embeddings = embeddings or DEFAULT_EMBEDDINGS
    per_dataset = {}
    for name in datasets:
        cfg = DEFAULT_DATASETS[name] if name in DEFAULT_DATASETS else name
        per_dataset[name] = _one_dataset(
            cfg, embeddings=embeddings, n_train=n_train, n_test=n_test,
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
             fmt="{:.2f}", show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(cols), 1.5 + 0.7 * len(rows)))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), col_name, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)

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
                                 description="Datasets x learners R^2 grid + pairwise geometric-difference matrix.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    help="dataset short-names (or embed-config paths)")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-gram", type=int, default=1000, help="rows for the O(N^3) geometric difference")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--reg", type=float, default=1e-6,
                    help="ridge for geometric difference; raise if g(rbf||rbf) drifts from 1")
    ap.add_argument("--save-dir", default="img", help="directory for the heatmap PNGs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    rows, per = run_grid(args.datasets, n_train=args.n_train, n_test=args.n_test,
                         C=args.C, gamma=gamma, epsilon=args.epsilon, use_cache=not args.force)

    full_cols = [c for c in FULL_ORDER if any(c in per[r] for r in rows)]
    rand_cols = [c for c in RANDOM_ORDER if any(c in per[r] for r in rows)]

    r2_full = _matrix(per, rows, full_cols, "test_r2")
    r2_rand = _matrix(per, rows, rand_cols, "test_r2")

    # kernel x kernel pairwise geometric difference (label-free; same for every dataset
    # since X is shared) -- computed once on the first dataset.
    first_cfg = DEFAULT_DATASETS.get(args.datasets[0], args.datasets[0])
    knames, g_kernel = kernel_g_matrix(first_cfg, DEFAULT_EMBEDDINGS, n_gram=args.n_gram,
                                       reg=args.reg, use_cache=not args.force)

    print("\n=== test R^2 (full grid) ===")
    _print_matrix(r2_full, rows, full_cols)
    print("\n=== test R^2 (random-quantum 4x4) ===")
    _print_matrix(r2_rand, rows, rand_cols)
    print("\n=== pairwise geometric difference  g(K_row || K_col)  [diag = 1] ===")
    _print_matrix(g_kernel, knames, knames)

    sd = Path(args.save_dir)
    # R^2: green = high (good, ->1).  g: green = LOW (kernels close), so reverse the cmap.
    _heatmap(r2_full, rows, full_cols, FULL_ORDER_NAME, "Test R² — full grid (regress soft[:,0])",
             sd / "grid_r2_full.png", vmin=0.0, vmax=1.0, cmap="RdYlGn", show=args.show)
    _heatmap(r2_rand, rows, rand_cols, RANDOM_ORDER_NAME, "Test R² — random quantum (4x4)",
             sd / "grid_r2_random.png", vmin=0.0, vmax=1.0, cmap="RdYlGn", show=args.show)
    _heatmap(g_kernel, knames, knames, knames, "Pairwise geometric difference  g(K_row || K_col)",
             sd / "grid_geomdiff_kernels.png", vmin=1.0, cmap="RdYlGn_r", fmt="{:.1f}", show=args.show)


if __name__ == "__main__":
    main()
