"""How the interaction degree needed to *learn* scales with a config parameter.

A config-driven experiment on top of :func:`learn.linreg.sweep_config`.  It takes one base
config and two groups of **arbitrary** dotted config fields (nothing is hardcoded to m/k/seed):

- **sweep** group -- the x-axis.  One or more fields whose value lists co-vary (zip) into an
  ordered list of sweep points, e.g. ``problem.m=4,6,8 problem.k=2,3,4`` -> ``(m=4,k=2),
  (m=6,k=3),(m=8,k=4)``; or a single field like ``problem.n_vertices=6,8,10``.
- **average** group -- repeats averaged over at each sweep point, e.g.
  ``problem.graph_seed=1,2,3`` or ``seeds.teacher_seed=1,2,3``.

For every (sweep point x average point) the fields are written into a copy of the base config,
and for each basis the ``(fourier_order x degree)`` grid is searched, recording the **minimum
number of explicit features** ``n_feat`` at which test R^2 reaches ``--threshold`` (both knobs
inflate ``n_feat``, so this is the smallest feature budget that learns).  Everything (incl. the
full grid curve) is saved to JSON.  The plot then shows, per basis, the **mean min-``n_feat``
over the average group vs the swept x-field** (log y) -- the classical feature-budget scaling.

    # sweep m,k together; average over the teacher seed
    python -m learn.scaling --config configs/Qubits/1_qubit.yaml \\
        --sweep problem.m=4,6,8 problem.k=2,3,4 --average seeds.teacher_seed=1,2,3 \\
        --threshold 0.9 --save-data img/scaling_qubit.json

    # fix m, sweep n_vertices; average over the graph seed (photonic loop_path)
    python -m learn.scaling --config configs/tmp/10_photonic_loop_path.yaml \\
        --sweep problem.n_vertices=6,8,10 --average problem.graph_seed=1,2,3

    # replot from saved data
    python -m learn.scaling --config <cfg> --from-file img/scaling_qubit.json
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

# Support `python -m learn.scaling` and `python learn/scaling.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

import numpy as np

from Generator import load_config

from .linreg import FEATURE_BASES, feature_grid


# --- config-field overriding + group parsing -------------------------------- #

def _set_field(cfg, dotted, value):
    """Set a dotted field (e.g. ``problem.n_vertices``) on a nested config dataclass."""
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"unknown config field {dotted!r}")
    setattr(obj, parts[-1], value)


def _parse_val(v: str):
    """``'4'`` -> 4, ``'0.9'`` -> 0.9, else the trimmed string (e.g. an observable name)."""
    v = v.strip()
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_group(specs):
    """``['problem.m=4,6,8', 'problem.k=2,3,4']`` -> ``(points, field_names)``.

    Fields in a group co-vary (zip), so their value lists must share one length; the result is
    that many override dicts.  ``None``/empty -> ``([{}], [])`` (a single no-override point).
    """
    if not specs:
        return [{}], []
    fields, length = {}, None
    for spec in specs:
        field, sep, raw = spec.partition("=")
        if not sep:
            raise ValueError(f"bad group spec {spec!r}; expected FIELD=v1,v2,...")
        vals = [_parse_val(v) for v in raw.split(",")]
        if length is None:
            length = len(vals)
        elif len(vals) != length:
            raise ValueError(f"group fields must have equal-length value lists; {field!r} "
                             f"has {len(vals)} != {length}")
        fields[field.strip()] = vals
    field_names = list(fields)
    points = [{f: fields[f][i] for f in field_names} for i in range(length)]
    return points, field_names


def _label(overrides):
    return ",".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())


# --- per-run summary + compute ---------------------------------------------- #

def _clean(x):
    """nan -> None (valid JSON) and round for compactness."""
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(float(x), 4)


def _summarize_basis(rows, threshold):
    """Smallest ``n_feat`` over the (order x degree) grid whose test R^2 >= ``threshold``.

    ``rows`` are :func:`learn.linreg.feature_grid` points.  Returns the minimum feature count
    that reaches the threshold and the ``(fourier_order, degree)`` that achieved it, or
    ``reached=False`` if no computed grid point (within ``max_feat``) cleared it.
    """
    best = None
    for r in rows:
        te = r["test_r2"]
        if te is None or math.isnan(te) or te < threshold:
            continue
        if best is None or r["n_feat"] < best["n_feat"]:
            best = r
    if best is None:
        return {"min_n_feat": None, "fourier_order": None, "degree": None,
                "r2_at_min": None, "reached": False}
    return {"min_n_feat": int(best["n_feat"]), "fourier_order": int(best["fourier_order"]),
            "degree": int(best["degree"]), "r2_at_min": _clean(best["test_r2"]), "reached": True}


def _curve(rows):
    return [{"fourier_order": int(r["fourier_order"]), "degree": int(r["degree"]),
             "n_feat": int(r["n_feat"]), "test_r2": _clean(r["test_r2"]),
             "train_r2": _clean(r["train_r2"])} for r in rows]


def run_scaling(base_config, sweep, average=None, *, x_field=None, bases=("monomial", "fourier"),
                threshold=0.5, max_feat=100000, max_degree=20, max_fourier_order=3,
                n_fit=4000, n_test=2000, lambdas=None, dataset_root="datasets", observable=None):
    """Sweep points x average points; return the JSON-serialisable results payload.

    ``sweep`` and ``average`` are lists of override dicts ``{dotted_field: value}`` (as produced
    by :func:`parse_group`; ``average`` defaults to a single no-override repeat).  ``x_field``
    (default: first field of the first sweep point) is the field plotted on the x-axis.  At each
    (sweep x average) point the ``(fourier_order x degree)`` grid is searched and the **minimum
    feature count** reaching ``threshold`` is recorded.
    """
    if not sweep:
        raise ValueError("sweep must contain at least one point")
    average = average or [{}]
    x_field = x_field or next(iter(sweep[0]))
    if any(x_field not in sp for sp in sweep):
        raise ValueError(f"x_field {x_field!r} must appear in every sweep point")

    base = load_config(base_config)
    degrees = list(range(1, int(max_degree) + 1))
    fourier_orders = list(range(1, int(max_fourier_order) + 1))
    runs = []
    for si, sp in enumerate(sweep):
        for ap in average:
            overrides = {**sp, **ap}
            dcfg = deepcopy(base)
            try:
                for f, v in overrides.items():
                    _set_field(dcfg, f, v)
                if "problem.m" in overrides:
                    dcfg.problem.n_features = None           # -> m-1, tracks a swept m
                dcfg.validate()
            except ValueError as exc:
                print(f"[scaling] skip {_label(overrides)}: {exc}")
                continue
            per_rows = feature_grid(
                dcfg, bases=bases, degrees=degrees, fourier_orders=fourier_orders, n_fit=n_fit,
                n_test=n_test, lambdas=lambdas, max_feat=max_feat, dataset_root=dataset_root,
                observable=observable, label=_label(overrides))
            per_basis = {b: {**_summarize_basis(per_rows[b], threshold), "curve": _curve(per_rows[b])}
                         for b in bases}
            runs.append({"sweep_idx": si, "x": sp.get(x_field), "sweep": sp, "average": ap,
                         "per_basis": per_basis})
            for b in bases:
                s = per_basis[b]
                print(f"[scaling] {_label(overrides)} [{b}]: min_n_feat={s['min_n_feat']} "
                      f"(order={s['fourier_order']}, degree={s['degree']}, r2={s['r2_at_min']})")
    return {
        "meta": {"base_config": str(base_config), "bases": list(bases), "threshold": threshold,
                 "max_feat": max_feat, "max_degree": max_degree, "max_fourier_order": max_fourier_order,
                 "n_fit": n_fit, "n_test": n_test, "observable": observable, "sweep": list(sweep),
                 "average": list(average), "x_field": x_field, "x_label": x_field.split(".")[-1]},
        "runs": runs,
    }


# --- persistence + aggregation ---------------------------------------------- #

def save_results(payload, path):
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[scaling] saved {path}")
    return path


def load_results(path):
    import json

    with open(path) as f:
        return json.load(f)


def aggregate(payload):
    """``{basis: {sweep_idx: {x, mean, std, n_reached, n_total}}}`` of min-``n_feat`` over the
    average group (nan-safe)."""
    from collections import defaultdict

    bases = payload["meta"]["bases"]
    sweep = payload["meta"]["sweep"]
    xf = payload["meta"]["x_field"]
    vals = defaultdict(list)                                  # (basis, sweep_idx) -> [min_n_feat]
    totals = defaultdict(int)                                 # (basis, sweep_idx) -> repeats run
    for run in payload["runs"]:
        si = run["sweep_idx"]
        for b in bases:
            totals[(b, si)] += 1
            md = run["per_basis"][b]["min_n_feat"]
            if md is not None:
                vals[(b, si)].append(md)
    agg = {}
    for b in bases:
        agg[b] = {}
        for si, sp in enumerate(sweep):
            v = vals[(b, si)]
            agg[b][si] = {"x": sp.get(xf),
                          "mean": float(np.mean(v)) if v else float("nan"),
                          "std": float(np.std(v)) if v else float("nan"),
                          "n_reached": len(v), "n_total": totals[(b, si)]}
    return agg


# --- plot ------------------------------------------------------------------- #

def _plot(payload, save_path, *, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate(payload)
    sweep = payload["meta"]["sweep"]
    idxs = list(range(len(sweep)))
    xvals = [agg[payload["meta"]["bases"][0]][i]["x"] for i in idxs]
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xvals)
    xpos = xvals if numeric else idxs
    thr, xl = payload["meta"]["threshold"], payload["meta"]["x_label"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for b in payload["meta"]["bases"]:
        means = [agg[b][i]["mean"] for i in idxs]
        stds = [agg[b][i]["std"] for i in idxs]
        line, = ax.plot(xpos, means, marker="o", label=b)
        lo = [max(m - s, 1) for m, s in zip(means, stds)]    # clamp for the log axis
        ax.fill_between(xpos, lo, [m + s for m, s in zip(means, stds)],
                        color=line.get_color(), alpha=0.15)
        for i in idxs:                                       # flag partial-coverage points
            a = agg[b][i]
            if 0 < a["n_reached"] < a["n_total"]:
                ax.annotate(f"{a['n_reached']}/{a['n_total']}", (xpos[i], a["mean"]),
                            fontsize=7, color="grey", xytext=(3, 3), textcoords="offset points")
    if not numeric:
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(x) for x in xvals])
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel(f"min # features to reach test R² ≥ {thr}  (mean ± std over averages)")
    ax.set_title(f"Feature budget needed to learn vs {xl}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(title="basis")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        print(f"[scaling] saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _print_table(payload):
    agg = aggregate(payload)
    sweep = payload["meta"]["sweep"]
    xl = payload["meta"]["x_label"]
    print(f"\n  min # features to reach R^2 >= {payload['meta']['threshold']:.2f} "
          f"(mean over averages; * = partial, - = none reached):")
    for b in payload["meta"]["bases"]:
        cells = []
        for i in range(len(sweep)):
            a = agg[b][i]
            if a["n_reached"] == 0:
                cells.append(f"{xl}={a['x']}: -")
            else:
                mark = "" if a["n_reached"] == a["n_total"] else "*"
                cells.append(f"{xl}={a['x']}: {a['mean']:.0f}{mark}")
        print(f"    {b:<10} " + "   ".join(cells))


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="learn.scaling",
        description="Min # features to learn (test R^2 >= threshold, searching order x degree) "
                    "vs a swept config field.")
    ap.add_argument("--config", required=True,
                    help="base experiment config; swept/averaged fields are overridden on a copy")
    ap.add_argument("--sweep", nargs="+",
                    help="x-axis group: FIELD=v1,v2,... (dotted, co-varying fields zip). "
                         "e.g. problem.m=4,6,8 problem.k=2,3,4  (required unless --from-file)")
    ap.add_argument("--average", nargs="+", default=None,
                    help="averaging group: FIELD=v1,v2,... e.g. problem.graph_seed=1,2,3")
    ap.add_argument("--x-field", default=None,
                    help="which swept field is the x-axis (default: first --sweep field)")
    ap.add_argument("--bases", nargs="+", default=["fourier"],
                    choices=sorted(FEATURE_BASES))
    ap.add_argument("--threshold", type=float, default=0.5, help="target test R^2")
    ap.add_argument("--max-feat", type=int, default=100000, help="stop a basis once wider than this")
    ap.add_argument("--max-degree", type=int, default=10, help="largest interaction degree to try")
    ap.add_argument("--n-fit", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--max-fourier-order", type=int, default=10,
                    help="search Fourier orders 1..this alongside degree (both inflate n_feat)")
    ap.add_argument("--lambdas", type=float, nargs="+", default=None)
    ap.add_argument("--observable", default=None,
                    help="re-score the saved distribution under this observable (needs save_dist); "
                         "photonic graph observables may encode the selection, e.g. "
                         "loop_path_parity__L0-1__P2-3")
    ap.add_argument("--dataset-root", default="datasets")
    ap.add_argument("--save-data", default="scaling/", help="write the raw results JSON here")
    ap.add_argument("--save-dir", default="scalinf")
    ap.add_argument("--from-file", default=None,
                    help="skip compute; load a saved results JSON and just (re)plot")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    if args.from_file:
        payload = load_results(args.from_file)
        base = Path(args.config).stem
        xl = payload["meta"]["x_label"]
        obs_tag = f"_{args.observable}" if args.observable else ""
    else:
        if not args.sweep:
            ap.error("--sweep is required unless --from-file is given")
        sweep, sweep_fields = parse_group(args.sweep)
        average, _ = parse_group(args.average)
        payload = run_scaling(
            args.config, sweep, average, x_field=args.x_field or sweep_fields[0], bases=args.bases,
            threshold=args.threshold, max_feat=args.max_feat, max_degree=args.max_degree,
            max_fourier_order=args.max_fourier_order, n_fit=args.n_fit, n_test=args.n_test,
            lambdas=args.lambdas, dataset_root=args.dataset_root, observable=args.observable)
        
        base = Path(args.config).stem
        xl = payload["meta"]["x_label"]
        obs_tag = f"_{args.observable}" if args.observable else ""

        if args.save_data:
            save_results(payload, Path(args.save_data) / f"scaling_{base}_{xl}{obs_tag}.json")

    _print_table(payload)
    _plot(payload, Path(args.save_dir) / f"scaling_{base}_{xl}{obs_tag}.png", show=args.show)


if __name__ == "__main__":
    main()
