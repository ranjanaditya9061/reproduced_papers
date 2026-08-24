"""Plot: ``min_delta`` (actual nearest-neighbour gap) vs ``eps_local(N)`` (resolvable gap at
shot budget ``N``), one scatter series each, same x-axis -- points below the ``eps_local`` series
are the ones NOT resolvable at that ``N``.  A third, flat reference line shows
``eps_global(N)`` (Makarovskiy et al.'s Proposition 1, :mod:`metrics.global_ratio`) at the same
``N`` -- the arbitrary-pair bound the local, neighbour-restricted one should sit below, showing how
much slack the local question buys.

    python eval/local_ratio_plot.py --config configs/photonic.yaml --observable parity --N 10000

**Unit conversion for the global line.**  ``eps_local`` and ``min_delta`` are both in raw observable
score units (a value gap ``Delta(x,g)``); Makarovskiy et al.'s ``eps_global`` is in units of
``sqrt(Var_circ)`` (Proposition 1's own convention, resolving each point to within
``epsilon * sqrt(Var_circ(O))``).  The global line plotted here is ``eps_global(N) *
sqrt(Var_circ)``, converted into the same raw-gap units as the other two series -- comparing the
three directly would otherwise be comparing different units.

**Plotting only.**  Every number here comes from :func:`metrics.local_ratio.cached_local_ratio`,
:func:`metrics.local_ratio.eps_local_of_N`, and :func:`metrics.global_ratio.cached_global_ratio`/
:func:`metrics.global_ratio.eps_global_of_N` (all cache-hit-or-compute, or cheap arithmetic on a
cached result) -- this script does no simulation, no observable scoring, no neighbour search of its
own.
"""

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/local_ratio_plot.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def local_margin(result: dict, *, N: int, delta: float = 0.1):
    """``min_delta - eps_local(N)``, ``(N,)`` list -- the resolvability margin.

    Positive: the point's hardest neighbour is resolvable at this ``N``, by that much headroom.
    Negative: not resolvable, by that much shortfall.  This is the quantity expected to track a
    finite-shot learner's R^2 degradation directly -- a large positive margin should predict
    robust learnability in that neighbourhood, a margin near or below zero should predict R^2
    collapsing there, since ``min_delta`` and ``eps_local`` alone only say resolvable-or-not, not
    by how much.
    """
    from metrics.local_ratio import eps_local_of_N

    eps = eps_local_of_N(result, N=N, delta=delta)
    return [md - e for md, e in zip(result["min_delta"], eps)]


def plot_local_ratio(result: dict, global_result: dict | None = None, *, N: int,
                     delta: float = 0.1, t: float = 1.0,
                     save_path: str | Path | None = None, show: bool = False):
    """``min_delta``, ``eps_local(N)``, and their difference (the resolvability margin) -- two
    stacked panels, x-axis = pool row index (sorted by ``min_delta`` ascending, so the
    hardest-to-resolve points sit on the left and the crossover between the two series, and the
    margin's zero-crossing, are easy to read).  ``global_result``, if given
    (:func:`metrics.global_ratio.cached_global_ratio`'s output), adds a flat reference line at
    ``eps_global(N) * sqrt(Var_circ)`` -- the arbitrary-pair bound, in the same raw-gap units as
    the other two series (see the module docstring's unit-conversion note).  Returns the
    ``matplotlib`` figure.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from metrics.local_ratio import eps_local_of_N

    min_delta = np.asarray(result["min_delta"])
    eps = np.asarray(eps_local_of_N(result, N=N, delta=delta))
    margin = min_delta - eps

    order = np.argsort(min_delta)
    x = np.arange(len(min_delta))
    resolvable = eps[order] < min_delta[order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.scatter(x, min_delta[order], s=8, alpha=0.5, label="min_delta (actual gap)",
               color="tab:blue")
    ax1.scatter(x, eps[order], s=8, alpha=0.5,
               label=f"eps_local(N={N}, delta={delta})", color="tab:orange")
    if global_result is not None:
        from metrics.global_ratio import eps_global_of_N

        eps_g = eps_global_of_N(global_result, N=N, t=t, delta=delta)
        eps_g_raw = eps_g * (global_result["var_circ"] ** 0.5)
        ax1.axhline(eps_g_raw, color="tab:red", linewidth=1.2, linestyle="--",
                   label=f"eps_global(N={N}) [arbitrary pair]")
    ax1.fill_between(x, 0, np.where(resolvable, min_delta[order], 0),
                     alpha=0.08, color="tab:green", step=None)
    n_resolvable = int(resolvable.sum())
    ax1.set_ylabel("observable value gap")
    ax1.set_title(f"{result['observable']}: local resolvability at N={N} shots\n"
                 f"{n_resolvable}/{len(min_delta)} points resolvable "
                 f"(min_delta > eps_local)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.scatter(x, margin[order], s=8, alpha=0.5, color="tab:purple")
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel(f"pool points, sorted by min_delta (n={len(min_delta)})")
    ax2.set_ylabel("margin\n(min_delta - eps_local)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Plot min_delta vs eps_local(N) for one config's "
                                             "input pool.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--t", type=float, default=1.0, help="Proposition 1's t parameter for the "
                                                         "global reference line")
    ap.add_argument("--no-global", action="store_true", help="skip the eps_global reference line")
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--out", default=None, help="save path; defaults to "
                                                "local_ratio__<observable>__N<N>.png")
    args = ap.parse_args(argv)

    import statistics

    from metrics.global_ratio import cached_global_ratio
    from metrics.local_ratio import cached_local_ratio

    result = cached_local_ratio(args.config, args.observable, k=args.k, out_root=args.out_root,
                                scores_root=args.scores_root)
    margin = local_margin(result, N=args.N, delta=args.delta)
    print(f"{args.observable}: median margin (min_delta - eps_local) = "
         f"{statistics.median(margin):.4g}  "
         f"({sum(m > 0 for m in margin)}/{len(margin)} resolvable)")

    global_result = None
    if not args.no_global:
        global_result = cached_global_ratio(args.config, args.observable, out_root=args.out_root,
                                            scores_root=args.scores_root)

    out = args.out or f"local_ratio__{args.observable}__N{args.N}.png"
    plot_local_ratio(result, global_result, N=args.N, delta=args.delta, t=args.t, save_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
