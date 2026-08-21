"""The paired ``perm``-vs-``det`` protocol, with the decision logic in code rather than in prose.

    python -m learner.compare --perm configs/photonic.yaml --det configs/fermion.yaml \
        --observables parity ent osc --learner ridge

**Why this file exists.**  A ``perm``-arm failure on its own is confounded by architecture and
optimiser -- that is the standard objection that kills training-based evidence.  So the fermion arm
is not a second result, it **calibrates the learner**: it is the same circuit, same seed, same input
state, same Fock basis, with ``|det|^2`` swapped for ``|Perm|^2``.  If the learner cannot fit the
*easy* arm, it cannot be used to make a claim about the hard one.

**Paired design.**  Both arms share everything -- architecture, hyperparameters, ``n_train``,
seeds, and split indices -- so the only difference is the label set, and the reported statistic is
the **paired difference**.

======== ======== ================================================================
``det``  ``perm``  reading
======== ======== ================================================================
succeeds fails    the informative cell
succeeds succeeds no separation at this size
fails    fails    **learner inadequate -- comparison VOID**, not a hardness signal
fails    succeeds investigate; likely a bug or a mismatched control
======== ======== ================================================================

Row 3 is emitted loudly, because it is the failure mode that otherwise gets written up as a result.

**Multiple comparisons.**  With ``R`` observables x ``M`` grid points you are implicitly selecting a
maximum, and bootstrap bands do not cover that selection.  ``--confirm-split`` divides the
observable pool into a selection half and a confirmation half, and only the confirmation half's
numbers may be quoted as a result.

**What a separation here would and would not mean.**  This is a learnability comparison of two
labelling functions.  Per :mod:`metrics.distribution`, it is *not* evidence about the
``Perm``/``det`` complexity separation: the boson family is indexed by ``O(m^2)`` numbers and is
maximally compressible even though each ``p(n)`` is ``#P``-hard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learner"

from . import embedding, kernel, nn  # noqa: F401  -- registration side effects

#: Held-out ``R^2`` above which an arm counts as "succeeds".  A threshold has to be fixed *before*
#: looking at the numbers for the decision table to mean anything; 0.5 is the repo's existing
#: convention for "the map was substantially learned".
SUCCESS_R2 = 0.5


def run_arm(cfg_path: str | Path, observable: str, *, learner: str, hparams: dict,
            out_root: str = "datasets", scores_root: str = "scores",
            n_train: int | None = None, graph_density: float = 0.5) -> dict:
    """Fit one arm and return its held-out statistics.  Split indices depend only on the pool size,
    so both arms see identical rows by construction.

    Thin wrapper over :func:`~learner.cache.cached_fit`, fixing ``split_seed`` to the config's own
    (unlike :func:`~learner.auto.run_config`, which allows an override) -- the paired protocol's
    "both arms see identical rows by construction" guarantee depends on that being unconditional.

    Reads the exact branch when ``cfg.generation.shots == 0``, else the shots branch -- see
    :func:`~pipeline.score.load_dataset`. An arm whose config has ``shots > 0`` while the other
    arm's does not is still a valid *paired* comparison as long as both see the same row count
    and split (``split_indices`` depends only on pool size), but the label noise differs between
    arms in that case -- not this function's concern, since it fits exactly what its own config
    says to.
    """
    from .cache import cached_fit

    return cached_fit(cfg_path, observable, learner, out_root=out_root, scores_root=scores_root,
                      n_train=n_train, graph_density=graph_density, split_seed=None, **hparams)


def compare(perm_cfg, det_cfg, observables, *, learner: str = "ridge", hparams: dict | None = None,
            **kw) -> list[dict]:
    """Run both arms on every observable with everything else held fixed."""
    hparams = hparams or {}
    rows = []
    for obs in observables:
        det = run_arm(det_cfg, obs, learner=learner, hparams=hparams, **kw)
        perm = run_arm(perm_cfg, obs, learner=learner, hparams=hparams, **kw)
        rows.append({
            "observable": obs, "det": det, "perm": perm,
            "d_log_likelihood": perm["log_likelihood"] - det["log_likelihood"],
            "d_r2": perm["r2"] - det["r2"],
            "verdict": verdict(det["r2"], perm["r2"]),
        })
    return rows


def verdict(det_r2: float, perm_r2: float, threshold: float = SUCCESS_R2) -> str:
    """The four-row decision table, in code."""
    det_ok, perm_ok = det_r2 >= threshold, perm_r2 >= threshold
    if det_ok and not perm_ok:
        return "INFORMATIVE"
    if det_ok and perm_ok:
        return "no separation at this size"
    if not det_ok and not perm_ok:
        return "VOID: learner inadequate"
    return "INVESTIGATE: likely a bug or mismatched control"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Paired perm-vs-det learner comparison")
    ap.add_argument("--perm", default="configs/photonic.yaml", help="the |Perm|^2 arm")
    ap.add_argument("--det", default="configs/fermion.yaml", help="the |det|^2 control arm")
    ap.add_argument("--observables", nargs="+", default=["parity", "majority", "ent", "osc"])
    ap.add_argument("--learner", default="ridge", choices=["ridge", "svr", "mlp"],
                    help="ridge=embedding-based, svr=kernel-based, mlp=nn-based")
    ap.add_argument("--basis", default=None, help="ridge/svr feature map; unset keeps each "
                    "learner's own default (fourier for ridge, raw for svr -- the clean kernel arm)")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--graph-density", type=float, default=0.5)
    ap.add_argument("--confirm-split", action="store_true",
                    help="split the observable pool into selection / confirmation halves")
    args = ap.parse_args(argv)

    if args.learner in ("ridge", "svr"):
        hparams = {"order": args.order, "alpha": args.alpha}
        if args.basis is not None:
            hparams["basis"] = args.basis
    else:
        hparams = {}
    if args.learner == "svr":
        hparams.pop("alpha", None)

    obs = list(args.observables)
    halves = [("confirmation", obs[::2]), ("selection", obs[1::2])] if args.confirm_split \
        else [("all", obs)]

    print(f"Paired comparison   perm={Path(args.perm).stem}  det={Path(args.det).stem}  "
          f"learner={args.learner} {hparams}")
    print(f"Success threshold fixed in advance: held-out R^2 >= {SUCCESS_R2}")
    print("Verdicts use R^2: with noiseless labels the Gaussian logL is dominated by label scale "
          "(see learner.base).\nPromote logL to primary only at generation.shots > 0.\n")

    for label, names in halves:
        if not names:
            continue
        rows = compare(args.perm, args.det, names, learner=args.learner, hparams=hparams,
                       n_train=args.n_train, graph_density=args.graph_density)
        if args.confirm_split:
            print(f"### {label.upper()} half"
                  + ("   <- only these may be quoted as a result" if label == "confirmation"
                     else "   (used to pick, not to report)"))
        hdr = (f"{'observable':<26}{'det R2':>9}{'perm R2':>9}{'dR2':>9}"
               f"{'det logL':>11}{'perm logL':>11}{'d logL':>9}   verdict")
        print(hdr)
        print("-" * (len(hdr) + 12))
        for r in rows:
            print(f"{r['observable']:<26}{r['det']['r2']:>9.4f}{r['perm']['r2']:>9.4f}"
                  f"{r['d_r2']:>+9.4f}{r['det']['log_likelihood']:>11.4f}"
                  f"{r['perm']['log_likelihood']:>11.4f}{r['d_log_likelihood']:>+9.4f}   "
                  f"{r['verdict']}")
        void = [r["observable"] for r in rows if r["verdict"].startswith("VOID")]
        if void:
            print(f"\n  !! COMPARISON VOID for {void}: the det control arm also failed, so the "
                  f"learner is inadequate at this size.\n     This is NOT a hardness signal -- "
                  f"raise n_train / capacity, or widen the feature map, before reading anything "
                  f"into the perm arm.")
        print(f"  n_train={rows[0]['det']['n_train']}, n_test={rows[0]['det']['n_test']}\n")


if __name__ == "__main__":
    main()
