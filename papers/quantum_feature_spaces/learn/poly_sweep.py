"""Sweep ONE photonic dataset across a list of polynomials (``prod_parity`` observables).

The dataset is generated once (with ``generation.save_dist``, so the full Fock distribution
is persisted); every polynomial then re-scores that saved distribution offline -- no boson
sampler rerun.  The Fourier-RBF embedding is a function of ``X`` only, so it is built once and
reused; only the *target* changes from polynomial to polynomial.

Two plots (one figure, stacked):

- **top** -- per polynomial (x-axis, labelled by the sweep config's names), the classical
  Fourier-RBF kernel's test **R²** (usual ``1 - Σ(y-ŷ)²/Σ(y-ȳ)²``) and its raw **difference**
  = mean squared prediction error ``mean((y-ŷ)²)`` -- the R² residual with NO division by the
  target variance (so it does not collapse when the ``(-1)^P(n)`` target is mean-biased).
- **bottom** -- a ggplot-style violin per polynomial (polynomials on the x-axis) showing the
  distribution of the teacher target ``y`` values, so you can see how each polynomial reshapes
  the label spread (solid bar = median, dashed = mean).

The polynomial set is a **readable sweep config** living in ``OUTPUT_DIR`` (``polytest/``): a
``name`` plus a ``polynomials: {label: prod_parity observable}`` mapping.  Add / edit one there
to try a different set -- it stays saved next to its own outputs, so a run is reproducible from
the config alone.  Each run writes, alongside the ``poly_sweep_<name>.png`` figure, a
``poly_sweep_<name>.txt`` recording each label, its polynomial in readable form (e.g.
``P2: n0*n1``; multi-term monomials render as a sum ``n0*n1 + n2*n3``), the observable string,
and the Fourier-RBF R² / difference.

    # default sweep = polytest/consecutive.yaml
    python -m learn.poly_sweep --config configs/Datasets/13_photonic_prod_parity.yaml
    # a different set
    python -m learn.poly_sweep --config configs/Datasets/13_photonic_prod_parity.yaml --sweep polytest/pairs.yaml
"""

from __future__ import annotations

from pathlib import Path

# Support `python -m learn.poly_sweep` and `python learn/poly_sweep.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

from sklearn.svm import SVR

from Generator import generate, load_config

from .svm import _split_indices, load_target

#: the single classical learner this sweep reports on (matches learn.grid's column)
FOURIER_RBF = {"type": "fourier_rbf", "fourier_order": 3}

#: folder holding the readable sweep configs AND the outputs (figure + summary) they produce.
OUTPUT_DIR = "polytest"
#: sweep config used when ``--sweep`` is not given (``<OUTPUT_DIR>/<DEFAULT_SWEEP>.yaml``).
DEFAULT_SWEEP = "consecutive"


def readable_poly(observable: str, m: int) -> str:
    """Human-readable, **signed** polynomial for an observable, mode indices matching ``M<...>``.

    e.g. ``prod_parity__M0`` -> ``n0``; ``prod_parity__M0-1__M2-3`` -> ``n0*n1 + n2*n3``;
    ``prod_parity__M0-1__N2-3-4`` -> ``n0*n1 - n2*n3*n4`` (``N`` = subtracted term).  This is
    the polynomial *as specified* (coefficients summed per monomial, signs kept) -- note the
    label itself is ``(-1)^P`` so the sign does not change it, but the record preserves what you
    typed.  Needs ``m`` to expand presets such as ``lo1``.
    """
    from model.photonic import _parse_prod_segment, is_prod_parity_observable

    if not is_prod_parity_observable(observable):
        return observable                                    # e.g. prod_parity_consecutive (needs k)

    coeffs: dict = {}
    for seg in observable.split("__")[1:] or ["full"]:
        for c, mono in _parse_prod_segment(seg, m):
            key = tuple(sorted(mono))
            coeffs[key] = coeffs.get(key, 0) + c

    out = ""
    for j, mono in enumerate(k for k in sorted(coeffs) if coeffs[k] != 0):
        c = coeffs[mono]
        body = "*".join(f"n{i}" for i in mono)
        term = (body if abs(c) == 1 else f"{abs(c)}*{body}")
        if j == 0:
            out = ("-" if c < 0 else "") + term
        else:
            out += (" - " if c < 0 else " + ") + term
    return out or "0"


def load_sweep(path):
    """Read a sweep config; return ``(name, observables, labels)`` in the config's own order.

    The YAML has a ``name`` and a ``polynomials`` mapping ``{label: prod_parity observable}``,
    e.g.::

        name: consecutive
        polynomials:
          P1: prod_parity__M0
          P2: prod_parity__M0-1
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    polys = raw.get("polynomials") or {}
    if not polys:
        raise ValueError(f"sweep config {str(path)!r} needs a non-empty 'polynomials' mapping "
                         "{label: prod_parity observable}")
    labels = list(polys)                                     # preserve insertion order
    return (raw.get("name") or Path(path).stem), [polys[lbl] for lbl in labels], labels

def _fit_r2_and_diff(F_tr, t_tr, F_te, t_te, *, C, gamma, epsilon):
    """Fit an RBF-SVR (train-standardized features + target) and return ``(test_r2, test_diff)``
    on the RAW target scale:

      ``test_r2``   = ``1 - Σ(y-ŷ)² / Σ(y-ȳ)²``   -- the usual R² (matches learn.svm / learn.grid)
      ``test_diff`` = ``mean((y-ŷ)²)``             -- the RAW mean squared prediction error: R²'s
                      residual with NO division by the target variance (the "difference").

    Features/target are standardized on TRAIN stats for a stable fit, then predictions are
    mapped back to raw target units so ``test_diff`` is a genuine raw error.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(F_tr)
    Ftr, Fte = xs.transform(F_tr), xs.transform(F_te)
    mu, sd = float(t_tr.mean()), (float(t_tr.std()) or 1.0)
    ytr = (t_tr - mu) / sd
    reg = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=epsilon).fit(Ftr, ytr)
    pred = reg.predict(Fte) * sd + mu                        # back to raw target units
    resid = t_te - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t_te - t_te.mean()) ** 2))
    test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    test_diff = float(np.mean(resid ** 2))                   # raw MSE, no normalization
    return test_r2, test_diff


def run_poly_sweep(config, polynomials, *, n_train=8000, n_test=2000, C=1.0, gamma="scale",
                   epsilon=0.01, fourier_order=3, embeddings_root="embeddings",
                   dataset_root="datasets", use_cache=True):
    """Re-score one dataset under each polynomial; return ``(per_poly, targets, m)``.

    ``per_poly`` maps each observable string to ``{test_r2, test_diff, poly}`` for the
    Fourier-RBF kernel (``poly`` = the readable polynomial); ``targets`` maps it to the full
    teacher target vector (numpy) for the distribution plot; ``m`` is the dataset's mode count.
    The dataset is generated once and the Fourier-RBF features are built once (they do not
    depend on the observable); only the re-scored target changes per polynomial.
    """
    from embedding import build_embeddings_for

    dcfg = load_config(config)
    generate(dcfg, out_root=dataset_root)                    # ensure artifact + distributions.npz
    m = dcfg.problem.m

    spec = dict(FOURIER_RBF, fourier_order=fourier_order)
    results, _, _ = build_embeddings_for(
        dcfg, [spec], embeddings_root=embeddings_root,
        dataset_root=dataset_root, use_cache=use_cache,
    )
    F = results[0]["blob"]["data"]                           # (N, d) Fourier-RBF features

    per_poly, targets = {}, {}
    for obs in polynomials:
        t = load_target(dcfg, dataset_root, observable=obs)  # re-score the saved distribution
        tr, te = _split_indices(t, test_fraction=dcfg.split.test_fraction,
                                split_seed=dcfg.split.split_seed)
        tr, te = tr[:n_train], te[:n_test]
        t_tr, t_te = t[tr].numpy(), t[te].numpy()
        test_r2, test_diff = _fit_r2_and_diff(
            F[tr].numpy(), t_tr, F[te].numpy(), t_te, C=C, gamma=gamma, epsilon=epsilon)
        per_poly[obs] = {"test_r2": test_r2, "test_diff": test_diff,
                         "poly": readable_poly(obs, m)}
        targets[obs] = t.numpy()
    return per_poly, targets, m


def _plot(per_poly, targets, polynomials, names, save_path, *, title=None, show=False):
    """Two stacked panels: (top) Fourier-RBF R² + raw difference vs polynomial;
    (bottom) the teacher target distribution per polynomial (violin plot)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r2 = [per_poly[o]["test_r2"] for o in polynomials]
    diff = [per_poly[o]["test_diff"] for o in polynomials]
    xs = range(len(polynomials))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 1.6 * len(polynomials)), 9))

    # --- top: R² (left axis) and raw difference / MSE (right axis) ---------------- #
    c_r2, c_diff = "#1f77b4", "#d62728"
    l1 = ax1.plot(xs, r2, marker="o", color=c_r2, label="Fourier-RBF  test R²")[0]
    ax1.set_ylabel("test R²", color=c_r2)
    ax1.tick_params(axis="y", labelcolor=c_r2)
    ax1.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax1.set_xticks(list(xs), names, rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0,1)

    axr = ax1.twinx()
    l2 = axr.plot(xs, diff, marker="s", color=c_diff,
                  label="Fourier-RBF  difference  (mean (y-ŷ)²)")[0]
    axr.set_ylabel("raw difference  mean (y-ŷ)²", color=c_diff)
    axr.set_ylim(0,0.03)
    axr.tick_params(axis="y", labelcolor=c_diff)
    ax1.legend(handles=[l1, l2], loc="best", fontsize=9)
    ax1.set_title(title or "Fourier-RBF R² and raw difference per polynomial")

    # --- bottom: teacher target distribution per polynomial (ggplot-style violins) - #
    data = [targets[o] for o in polynomials]
    parts = ax2.violinplot(data, positions=list(xs), showmeans=True, showmedians=True,
                           showextrema=True, widths=0.8)
    cmap = plt.get_cmap("viridis")
    denom = max(len(polynomials) - 1, 1)
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
    ax2.set_xticks(list(xs), names, rotation=30, ha="right")
    ax2.set_xlabel("polynomial")
    ax2.set_ylabel("teacher target  y  (E[(−1)^P(n)])")
    ax2.set_title("Teacher target distribution per polynomial  (solid = median, dashed = mean)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(-1,1)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[poly_sweep] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _write_summary(path, sweep_name, polynomials, names, per_poly, *, data_stem, m):
    """Save a readable record: each label, its polynomial (``P2 : n0*n1``), the observable
    string, and the Fourier-RBF R² / raw difference -- so a run is legible after the fact."""
    wl = max(len(nm) for nm in names)
    wo = max(len(o) for o in polynomials)
    lines = [f"# poly sweep '{sweep_name}'   (data config: {data_stem}, m={m})",
             "# label : [observable]   Fourier-RBF test R2 / raw difference   =  polynomial", ""]
    for o, nm in zip(polynomials, names):
        p = per_poly[o]
        lines.append(f"{nm:<{wl}} : [{o:<{wo}}]   R2={p['test_r2']:+.3f}  "
                     f"diff={p['test_diff']:.4g}   =  {p['poly']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[poly_sweep] saved {path}")


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.poly_sweep",
        description="Sweep one photonic dataset across a list of prod_parity polynomials: "
                    "Fourier-RBF R² + raw difference, and the target distribution, per polynomial.")
    ap.add_argument("--config", required=True, help="data config (needs generation.save_dist)")
    ap.add_argument("--sweep", default=None,
                    help="sweep config YAML (name + {label: observable}); default "
                         f"{OUTPUT_DIR}/{DEFAULT_SWEEP}.yaml")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0, help="SVR regularisation")
    ap.add_argument("--gamma", default="scale", help="RBF gamma ('scale', 'auto', or a float)")
    ap.add_argument("--epsilon", type=float, default=0.01, help="SVR epsilon-insensitive tube")
    ap.add_argument("--fourier-order", type=int, default=3)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute embeddings (skip cache)")
    args = ap.parse_args(argv)

    sweep_path = args.sweep or str(Path(OUTPUT_DIR) / f"{DEFAULT_SWEEP}.yaml")
    sweep_name, polynomials, names = load_sweep(sweep_path)

    gamma = float(args.gamma) if args.gamma.replace(".", "", 1).isdigit() else args.gamma
    per_poly, targets, m = run_poly_sweep(
        args.config, polynomials, n_train=args.n_train, n_test=args.n_test,
        C=args.C, gamma=gamma, epsilon=args.epsilon, fourier_order=args.fourier_order,
        use_cache=not args.force,
    )

    wl = max(len(nm) for nm in names)
    wo = max(len(o) for o in polynomials)
    print(f"\n=== Fourier-RBF per polynomial  (sweep '{sweep_name}', data {Path(args.config).stem}) ===")
    for o, nm in zip(polynomials, names):
        p = per_poly[o]
        print(f"  {nm:>{wl}}  {o:<{wo}}  R2={p['test_r2']:+.3f}  "
              f"diff={p['test_diff']:>10.4g}   =  {p['poly']}")

    _write_summary(Path(OUTPUT_DIR) / f"poly_sweep_{sweep_name}.txt", sweep_name,
                   polynomials, names, per_poly, data_stem=Path(args.config).stem, m=m)
    save = None if args.show else str(Path(OUTPUT_DIR) / f"poly_sweep_{sweep_name}.png")
    _plot(per_poly, targets, polynomials, names, save,
          title=f"Fourier-RBF R² and raw difference per polynomial  ({sweep_name})",
          show=args.show)


if __name__ == "__main__":
    main()