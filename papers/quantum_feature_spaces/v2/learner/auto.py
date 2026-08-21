"""Thin dispatcher: config + observable + learner name -> fitted learner -> held-out stats.

    res = run_config("configs/photonic.yaml", "parity", "ridge", basis="fourier", order=3)
    res["r2"]

Loads the saved artifact for ``cfg_path``, builds ``learner_name`` from the registry
(:func:`~learner.base.build_learner`) with whatever ``**learner_kwargs`` were passed, fits it on the
config's own train split, and returns :func:`~learner.base.evaluate`'s stats (``r2``,
``log_likelihood``, ``rmse``, ...).  Mirrors :func:`~.compare.run_arm`'s loading exactly (same
artifact lookup, same :func:`~pipeline.split.split_indices` convention) but is not part of the
paired ``det``/``perm`` protocol -- this is the single-config, single-learner entry point.

**Also has :func:`sweep_degree_grid`, a port of ``learn.linreg.feature_grid`` from the v1 tree.**
``ridge`` is additive across coordinates (no cross terms), which is a structural ceiling for a
multilinear label like the photonic permanent -- confirmed empirically: an earlier
correlation-selected adaptive-basis learner (``topk_fourier``, since removed) was tried with its
frequency search widened well past the model's photon number and showed no improvement over plain
``fourier``, because the missing ingredient is interaction terms, not which harmonics get picked.
``sweep_degree_grid`` is the one place in this module that actually adds cross terms: it expands the
Fourier base with ``sklearn.preprocessing.PolynomialFeatures(degree=d)`` before the ridge fit, so
``d=2`` includes every pairwise product of Fourier features and ``d=1`` reproduces plain ``fourier``.
Feature count is ``C(n_in + d, d)`` in the Fourier base width ``n_in``, so it is guarded by
``max_feat`` the same way v1's sweep is -- this is the tool for "does the gap close once
interactions are allowed", not a default to run at large ``order``/``degree`` casually.  Measured on
``photonic``/``parity``: ``R^2`` moves from ``~0.07`` at ``order=1, degree=1`` to ``~0.71`` at
``order=2, degree=3``, competitive with ``svr``/``mlp``.
"""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):                    # allow `python learner/auto.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learner"

from .base import build_learner, evaluate
from . import embedding, kernel, nn  # noqa: F401  -- registration side effects


def run_config(cfg_path: str | Path, observable: str, learner_name: str, *,
               out_root: str = "datasets", scores_root: str = "scores",
               n_train: int | None = None, graph_density: float = 0.5,
               split_seed: int | None = None, force: bool = False, **learner_kwargs) -> dict:
    """Config -> cached-or-fitted ``learner_name`` -> held-out stats, in one call.

    Thin wrapper over :func:`~learner.cache.cached_fit` -- the actual load/split/fit/evaluate/cache
    logic lives there, shared with :func:`~learner.compare.run_arm` and every other learner-fitting
    call site, so there is exactly one place a fit is ever paid for a given (config, observable,
    learner, hyperparameters, split) combination.

    Raises if the artifact for ``cfg_path`` has not been generated yet.

    ``split_seed``, if given, overrides ``cfg.split.split_seed`` for this call only -- the config
    on disk is untouched.  Needed by any caller that wants to average a score over several
    train/test partitions (e.g. :mod:`eval.best_of_grid`'s seed-averaging): reseeding a learner's
    own ``seed`` kwarg alone does nothing for ``ridge``/``svr`` at their default (deterministic)
    settings, since only the split itself -- not the learner's internal randomness -- varies their
    score in that regime.  ``None`` (the default) keeps today's behaviour exactly, reading the
    config's own ``split_seed``.

    Reads the **exact** branch when ``cfg.generation.shots == 0`` (the default), else the shots
    branch -- via :func:`~pipeline.score.load_dataset`, which is what makes this work at all past
    the point where the exact branch cannot be generated (:meth:`~model.fermion.FermionModel.
    shot_counts`'s MH sampler exists precisely for that regime).
    """
    from .cache import cached_fit

    res = cached_fit(cfg_path, observable, learner_name, out_root=out_root,
                     scores_root=scores_root, n_train=n_train, graph_density=graph_density,
                     split_seed=split_seed, force=force, **learner_kwargs)
    res["learner"] = learner_name
    return res


def _standardize(Ftr, Fte):
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit(Ftr)
    return xs.transform(Ftr), xs.transform(Fte)


def _expanded_dim(n_in: int, degree: int) -> int:
    """Column count of degree-``1..degree`` monomials over ``n_in`` inputs (no bias term)."""
    import math
    return math.comb(n_in + degree, degree) - 1


def sweep_degree_grid(cfg_path: str | Path, observable: str, *, orders=(1, 2, 3),
                      degrees=(1, 2, 3), out_root: str = "datasets", scores_root: str = "scores",
                      n_train: int | None = None, graph_density: float = 0.5,
                      max_feat: int = 20_000) -> list[dict]:
    """``(fourier_order x interaction_degree)`` grid: the one place here that adds cross terms.

    For each ``order`` in ``orders``, builds :func:`~learner.features.fourier_features` at that
    order as the base map, then for each ``degree`` in ``degrees`` expands it with
    ``sklearn.preprocessing.PolynomialFeatures(degree)`` -- so ``degree=1`` reproduces plain
    ``fourier``, and ``degree=2`` adds every pairwise product of Fourier features (the interaction
    structure ``ridge`` cannot represent on its own, per this module's docstring).  Ridge
    lambda is picked per cell by :class:`sklearn.linear_model.RidgeCV` (closed-form LOO-GCV), and
    both train and test ``R^2`` are reported -- a widening train/test gap *is* the overfitting that
    shows up once ``C(n_in + degree, degree)`` approaches ``n_train``, shown rather than assumed
    away.  A cell wider than ``max_feat`` is skipped and logged (never silently OOM'd); width is
    monotone in ``degree``, so the remaining degrees at that order are skipped too.

    Returns a list of ``{order, degree, n_feat, train_r2, test_r2, lam}`` rows, ascending in
    ``(order, degree)``.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import PolynomialFeatures

    from config import load_config
    from pipeline.score import load_dataset
    from pipeline.split import split_indices
    from .features import fourier_features

    lambdas = np.geomspace(1e2, 1e5, 25)

    cfg = load_config(cfg_path)
    X, soft, _ = load_dataset(cfg, observable, out_root=out_root, scores_root=scores_root,
                              graph_density=graph_density)
    tr, te = split_indices(len(X), test_fraction=cfg.split.test_fraction,
                           split_seed=cfg.split.split_seed)
    if n_train:
        tr = tr[:int(n_train)]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = soft[tr].double().numpy(), soft[te].double().numpy()

    rows = []
    for order in sorted(set(int(o) for o in orders)):
        Btr = fourier_features(Xtr, order).double().numpy()
        Bte = fourier_features(Xte, order).double().numpy()
        n_in = Btr.shape[1]
        for degree in sorted(set(int(d) for d in degrees)):
            dim = _expanded_dim(n_in, degree)
            if dim > max_feat:
                print(f"[auto] order={order}: degree={degree} needs {dim} features "
                      f"> max_feat={max_feat}; stopping this order")
                break
            poly = PolynomialFeatures(degree=degree, include_bias=False).fit(Btr)
            Ftr, Fte = poly.transform(Btr), poly.transform(Bte)
            Ftr, Fte = _standardize(Ftr, Fte)
            reg = RidgeCV(alphas=lambdas).fit(Ftr, ytr)
            rows.append({"order": order, "degree": degree, "n_feat": int(Ftr.shape[1]),
                        "train_r2": float(reg.score(Ftr, ytr)), "test_r2": float(reg.score(Fte, yte)),
                        "lam": float(reg.alpha_)})
    return rows


#: (learner name, kwargs) tried by default in :func:`sweep_heatmap` -- one row per learner.
DEFAULT_SWEEP_LEARNERS = (
    ("ridge", {}),
    ("svr", {}),
    ("mlp", {}),
)


#: Fourier orders / interaction degrees tried by the ``fourier_grid`` row when
#: ``sweep_heatmap(..., include_degree_grid=True)`` -- kept small by default since the grid's cost
#: grows fast in both knobs (see :func:`sweep_degree_grid`).
DEFAULT_GRID_ORDERS = (1, 2)
DEFAULT_GRID_DEGREES = (1, 2, 3)


def sweep_heatmap(cfg_path: str | Path, observables: list[str], *,
                  learners: tuple[tuple[str, dict], ...] = DEFAULT_SWEEP_LEARNERS,
                  out_root: str = "datasets", scores_root: str = "scores",
                  n_train: int | None = None, graph_density: float = 0.5,
                  include_degree_grid: bool = False, grid_orders=DEFAULT_GRID_ORDERS,
                  grid_degrees=DEFAULT_GRID_DEGREES, grid_max_feat: int = 20_000) -> dict:
    """Fit every ``learner`` on every ``observable`` for one config; return the ``R^2`` grid.

    A thin loop over :func:`run_config`, reshaped for plotting: rows are learner names, columns are
    observables.  Returns ``{"observables": [...], "learners": [...], "r2": [[...]]}`` --
    ``r2[i][j]`` is learner ``i``'s held-out ``R^2`` on observable ``j``, plain nested lists so the
    result is JSON-serialisable without depending on this function's caller having numpy.  Any
    ``(learner, observable)`` cell whose fit raises is recorded as ``nan`` rather than aborting the
    whole grid, and printed as a warning -- one bad cell (e.g. an observable a given prep cannot
    score) should not lose every other cell's result.

    ``include_degree_grid=True`` adds one more row, ``"fourier_grid"``: for each observable, runs
    :func:`sweep_degree_grid` over ``grid_orders x grid_degrees`` and takes the best **test** R^2
    cell.  This is the only row here with cross terms (``ridge``/``svr``/``mlp`` are otherwise the
    default trio), so it answers "how far does interaction-aware ridge get" in the same picture as
    the plain learners -- at real extra cost, since it is ``len(observables)`` separate grid
    sweeps.  Off by default for that reason.
    """
    import math

    names = [name for name, _ in learners]
    grid = []
    for name, kwargs in learners:
        row = []
        for obs in observables:
            try:
                res = run_config(cfg_path, obs, name, out_root=out_root, scores_root=scores_root,
                                 n_train=n_train, graph_density=graph_density, **kwargs)
                row.append(res["r2"])
            except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad cell must not
                print(f"[auto] {name}/{obs} failed: {exc}")  # abort the rest of the grid
                row.append(math.nan)
        grid.append(row)

    if include_degree_grid:
        names.append("fourier_grid")
        row = []
        for obs in observables:
            try:
                cells = sweep_degree_grid(cfg_path, obs, orders=grid_orders, degrees=grid_degrees,
                                          out_root=out_root, scores_root=scores_root,
                                          n_train=n_train, graph_density=graph_density,
                                          max_feat=grid_max_feat)
                row.append(max((c["test_r2"] for c in cells), default=math.nan))
            except (Exception, SystemExit) as exc:         # noqa: BLE001 -- see above
                print(f"[auto] fourier_grid/{obs} failed: {exc}")
                row.append(math.nan)
        grid.append(row)

    return {"observables": list(observables), "learners": names, "r2": grid}


def plot_heatmap(result: dict, *, save_path: str | Path | None = None, show: bool = False):
    """``R^2`` heatmap from :func:`sweep_heatmap`'s output -- observables on x, learners on y.

    Returns the ``matplotlib`` figure.  Cell text is the ``R^2`` value to two decimals, ``nan``
    cells (a failed fit) render as a distinct grey rather than a colour-scale value.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    obs, learners, r2 = result["observables"], result["learners"], np.asarray(result["r2"], dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(obs) + 2), max(3, 0.7 * len(learners) + 1.5)))
    masked = np.ma.masked_invalid(r2)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(obs)))
    ax.set_xticklabels(obs, rotation=45, ha="right")
    ax.set_yticks(range(len(learners)))
    ax.set_yticklabels(learners)
    ax.set_xlabel("observable")
    ax.set_ylabel("learner")
    ax.set_title("Held-out R^2: learner x observable")

    for i in range(len(learners)):
        for j in range(len(obs)):
            v = r2[i][j]
            text = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            colour = "black" if not np.isfinite(v) or -0.4 < v < 0.75 else "white"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=9)

    fig.colorbar(im, ax=ax, label="R^2")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _plot_r2_grid(row_labels: list[str], col_labels: list[str], r2, *,
                  row_axis: str, col_axis: str, title: str,
                  save_path: str | Path | None = None, show: bool = False):
    """Shared cell-text/colour-scale heatmap renderer behind :func:`plot_heatmap`,
    :func:`plot_variant_observable_grid` and :func:`plot_variant_learner_grid` -- only the axis
    labels, title and the ``r2`` matrix's orientation differ between the three call sites.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r2 = np.asarray(r2, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(col_labels) + 2),
                                    max(3, 0.7 * len(row_labels) + 1.5)))
    masked = np.ma.masked_invalid(r2)
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_axis)
    ax.set_ylabel(row_axis)
    ax.set_title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = r2[i][j]
            text = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            colour = "black" if not np.isfinite(v) or -0.4 < v < 0.75 else "white"
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=9)

    fig.colorbar(im, ax=ax, label="R^2")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


#: (variant name, config path) pairs -- the unit both grid sweeps below iterate over instead of
#: :func:`sweep_heatmap`'s single ``cfg_path``, so a whole folder of circuit-variant configs (e.g.
#: ``configs/eval/photonic_encoding/*.yaml``) becomes one axis of the grid.
Variant = tuple[str, "str | Path"]


def sweep_variant_observable_grid(variants: list[Variant], observables: list[str],
                                  learner_name: str = "ridge", *, learner_kwargs: dict | None = None,
                                  out_root: str = "datasets", scores_root: str = "scores",
                                  n_train: int | None = None, graph_density: float = 0.5) -> dict:
    """Fit ``learner_name`` on every ``(variant, observable)`` pair; return the ``R^2`` grid.

    Rows are ``variants`` (circuit configs -- e.g. the encodings in one ``configs/eval/`` subfolder),
    columns are ``observables``, for one fixed learner -- the mode-A counterpart to
    :func:`sweep_heatmap`'s learner x observable grid, with the circuit swapped in as the varying
    axis instead of held fixed.  Same per-cell failure handling as :func:`sweep_heatmap`: a raising
    cell becomes ``nan`` and is printed, not fatal to the rest of the grid.
    """
    import math

    names = [name for name, _ in variants]
    grid = []
    for name, cfg_path in variants:
        row = []
        for obs in observables:
            try:
                res = run_config(cfg_path, obs, learner_name, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train,
                                 graph_density=graph_density, **(learner_kwargs or {}))
                row.append(res["r2"])
            except (Exception, SystemExit) as exc:         # noqa: BLE001 -- one bad cell must not
                print(f"[auto] {name}/{obs} failed: {exc}")  # abort the rest of the grid
                row.append(math.nan)
        grid.append(row)

    return {"variants": names, "observables": list(observables), "learner": learner_name, "r2": grid}


def sweep_variant_learner_grid(variants: list[Variant], observable: str, *,
                               learners: tuple[tuple[str, dict], ...] = DEFAULT_SWEEP_LEARNERS,
                               out_root: str = "datasets", scores_root: str = "scores",
                               n_train: int | None = None, graph_density: float = 0.5) -> dict:
    """Fit every ``learner`` in ``learners`` on every ``variant``, at one fixed ``observable``.

    Rows are ``variants``, columns are ``learners`` -- the mode-B counterpart to
    :func:`sweep_variant_observable_grid`: same circuit axis, but the varying column is which
    learner fits it rather than which observable it is scored against.
    """
    import math

    names = [name for name, _ in variants]
    learner_names = [name for name, _ in learners]
    grid = []
    for name, cfg_path in variants:
        row = []
        for learner_name, kwargs in learners:
            try:
                res = run_config(cfg_path, observable, learner_name, out_root=out_root,
                                 scores_root=scores_root, n_train=n_train,
                                 graph_density=graph_density, **kwargs)
                row.append(res["r2"])
            except (Exception, SystemExit) as exc:         # noqa: BLE001 -- see above
                print(f"[auto] {name}/{learner_name} failed: {exc}")
                row.append(math.nan)
        grid.append(row)

    return {"variants": names, "learners": learner_names, "observable": observable, "r2": grid}


def plot_variant_observable_grid(result: dict, *, save_path: str | Path | None = None,
                                 show: bool = False):
    """``R^2`` heatmap from :func:`sweep_variant_observable_grid`'s output: variants on y,
    observables on x, one fixed learner named in the title."""
    return _plot_r2_grid(result["variants"], result["observables"], result["r2"],
                         row_axis="variant", col_axis="observable",
                         title=f"Held-out R^2: variant x observable (learner={result['learner']})",
                         save_path=save_path, show=show)


def plot_variant_learner_grid(result: dict, *, save_path: str | Path | None = None,
                              show: bool = False):
    """``R^2`` heatmap from :func:`sweep_variant_learner_grid`'s output: variants on y,
    learners on x, one fixed observable named in the title."""
    return _plot_r2_grid(result["variants"], result["learners"], result["r2"],
                         row_axis="variant", col_axis="learner",
                         title=f"Held-out R^2: variant x learner (observable={result['observable']})",
                         save_path=save_path, show=show)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Config + observable + learner -> held-out R^2")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", default=["parity", "majority", "ent", "osc"])
    ap.add_argument("--learner", default="ridge", choices=["ridge", "svr", "mlp"])
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--graph-density", type=float, default=0.5)
    ap.add_argument("--degree-grid", action="store_true",
                    help="run sweep_degree_grid instead of --learner: fourier_order x "
                    "interaction_degree, with cross terms via PolynomialFeatures")
    ap.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--degrees", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--max-feat", type=int, default=20_000)
    ap.add_argument("--heatmap", action="store_true",
                    help="run sweep_heatmap over ridge/svr/mlp x --observables and save a PNG "
                    "instead of --learner/--degree-grid")
    ap.add_argument("--heatmap-out", default="learner_heatmap.png")
    ap.add_argument("--heatmap-degree-grid", action="store_true",
                    help="with --heatmap, add a fourier_grid row: best test R^2 from "
                    "sweep_degree_grid over --orders x --degrees, per observable")
    ap.add_argument("--force", action="store_true", help="ignore any cached fit for this exact "
                    "(config, observable, learner, hyperparameters, split) combination and refit, "
                    "overwriting the cache -- only applies to the plain --learner path, not "
                    "--degree-grid/--heatmap")
    args = ap.parse_args(argv)

    if args.heatmap:
        result = sweep_heatmap(args.config, args.observables, n_train=args.n_train,
                               graph_density=args.graph_density,
                               include_degree_grid=args.heatmap_degree_grid,
                               grid_orders=args.orders, grid_degrees=args.degrees,
                               grid_max_feat=args.max_feat)
        w = max(10, max(len(l) for l in result["learners"]) + 1)
        print(f"{'observable':<26}" + "".join(f"{l:>{w}}" for l in result["learners"]))
        for j, obs in enumerate(result["observables"]):
            print(f"{obs:<26}" + "".join(f"{result['r2'][i][j]:>{w}.4f}"
                                         for i in range(len(result["learners"]))))
        plot_heatmap(result, save_path=args.heatmap_out)
        print(f"\nsaved {args.heatmap_out}")
        return

    for obs in args.observables:
        if args.degree_grid:
            rows = sweep_degree_grid(args.config, obs, orders=args.orders, degrees=args.degrees,
                                     n_train=args.n_train, graph_density=args.graph_density,
                                     max_feat=args.max_feat)
            print(f"--- {obs}")
            for r in rows:
                print(f"  order={r['order']:<3} degree={r['degree']:<3} n_feat={r['n_feat']:>6} "
                      f"train_r2={r['train_r2']:>8.4f}  test_r2={r['test_r2']:>8.4f}  "
                      f"lam={r['lam']:.3g}")
        else:
            res = run_config(args.config, obs, args.learner, n_train=args.n_train,
                             graph_density=args.graph_density, force=args.force)
            print(f"{obs:<26} learner={args.learner:<12} r2={res['r2']:>8.4f}  "
                  f"logL={res['log_likelihood']:>9.4f}")


if __name__ == "__main__":
    main()
