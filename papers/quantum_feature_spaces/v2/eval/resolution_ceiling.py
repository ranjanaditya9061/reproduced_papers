"""Shot-noise resolution ceiling: eps_x(x), N_eff, R^2_max -- Pipeline B (direct empirical).

    python eval/resolution_ceiling.py --config configs/photonic_shots.yaml --observable parity
    python eval/resolution_ceiling.py --config configs/size/size_photonic_fock/m16k08.yaml \
        --observables parity n_first ent osc sq_parity xent_parity

No learner can resolve two inputs closer together than shot noise allows: at a finite shot budget,
the *measured* score at x_i and at a nearby x_j are statistically indistinguishable once their gap
is smaller than the combined shot noise, so any learner fit on these scores is structurally unable
to separate them. This bounds achievable R^2 before any learner is even fit. Three steps, all
"Pipeline B": direct empirical, no gradient/Jacobian anywhere -- distinguishability is read straight
off pairs of measured scores and the shot noise on each, never off ``d<O>/dx``.

1. :func:`estimate_epsilon_x` -- for each training point, walk its nearest neighbours outward (by
   x-distance) and find the crossover: the farthest neighbour whose measured score is still
   statistically indistinguishable from this point's own, i.e. ``z = |y_i - y_j| /
   sqrt(sigma_i^2 + sigma_j^2) < z_thresh``. That's ``eps_x(x_i)`` -- the same pairwise gate is
   reused directly (never proxied through a radius) in steps 2 and 3.
2. :func:`cluster_by_resolution` -- merge training points that are directly mutually indistinguish-
   able by the same z-test (restricted to x-neighbours within the local eps_x radius, so the
   candidate set stays small), and count the resulting clusters: ``N_eff``. This re-runs the actual
   pairwise test on every candidate pair rather than trusting a scalar eps_x radius as a proxy for
   it -- a pair can be close in x yet still genuinely distinguishable if the score is non-smooth
   between them, and only a direct z-test on that specific pair catches that.
3. :func:`estimate_r2_max` -- per test point, the *same* pairwise gate as step 1, applied directly:
   is this test point's own measured score within combined shot noise of its nearest training
   point's? Covered -> predict that neighbour's resolution-cluster mean (the MSE-optimal constant a
   memorization-only learner could output for that cell); uncovered -> the optimistic/strict
   convention split (nearest-cluster-mean-anyway / global-mean), reported both ways since which
   convention you pick changes the final number materially.

This is the regression-only path: the observables scored throughout this repo are continuous
(``⟨O⟩_x``), so the ceiling that matters is R^2, not a classification accuracy over some ad hoc
sign/median discretization of a score that was never a label to begin with.

``--observables`` (:func:`compare_observables`) sweeps several observables in one run and, for
each, also fits :data:`learner.auto.DEFAULT_SWEEP_LEARNERS` (ridge/svr/mlp) and reports the best
learner's actual held-out ``R^2`` next to the ceiling -- the comparison this module exists to make.
Note a real caveat this comparison surfaces rather than hides: the ceiling here bounds a
memorization/nearest-neighbour-shaped learner; a global smooth model (``mlp``) pools information
across the whole training set, not just one local pair, so it can legitimately score *above* this
ceiling (flagged ``above_r2_ceiling`` per row when it does).

Per-point shot noise (``sigma``) comes from :func:`bootstrap_shot_sigmas`: resample each row's own
recorded shots with replacement and rescore -- "how much would this row's measured score move under
an independent draw of the same budget", with no assumption about the observable's functional form
(linear/quadratic/...), so it is exact for every :mod:`observable` family, not just the ones with a
closed-form plug-in variance. Cost is ``O(n_boot * N * shots)`` (one rescore of the whole dataset per
bootstrap replicate via :func:`pipeline.score.score_shots_streaming`) -- lower ``--n-boot`` or point
at a config with fewer rows/shots if this is too slow to iterate on.

Needs a shots-based config (``generation.shots > 0``) with its shots already generated
(``pipeline.generate``) -- eps_x is meaningless without shot noise to measure it against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):                    # allow `python eval/resolution_ceiling.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def bootstrap_shot_sigmas(obs, seq: np.ndarray, shots: int, *, n_boot: int = 32,
                          seed: int = 0) -> np.ndarray:
    """``(N,)`` empirical std of the measured score, under resampling each row's own recorded shots.

    """
    from pipeline.score import score_shots_streaming

    rng = np.random.default_rng(seed)
    N, S = seq.shape
    boots = np.empty((n_boot, N), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, S, size=(N, S))
        seq_b = np.take_along_axis(seq, idx, axis=1)
        boots[b] = score_shots_streaming(obs, seq_b, shots).double().numpy()
    return boots.std(axis=0, ddof=1)


def _z_gate(y_a: np.ndarray, sigma_a: np.ndarray, y_b: np.ndarray,
           sigma_b: np.ndarray) -> np.ndarray:
    """``z = |y_a - y_b| / sqrt(sigma_a^2 + sigma_b^2)`` -- the indistinguishability test"""

    return np.abs(y_a - y_b) / np.sqrt(sigma_a ** 2 + sigma_b ** 2)


def estimate_epsilon_x(X: np.ndarray, y: np.ndarray, sigma: np.ndarray, *, k: int = 10,
                       z_thresh: float = 2.0):
    """``(eps_x, censored)``, both ``(N,)`` -- Pipeline B, direct empirical, no gradient.

    For each point ``i``, its ``k`` nearest neighbours (by x-distance, ascending) are walked
    outward; ``eps_x[i]`` is the crossover distance -- the farthest neighbour in the longest
    indistinguishable *prefix* (:func:`_z_gate` holds for every closer neighbour too). A prefix, not
    "any indistinguishable neighbour among the k", because distinguishability should only grow with
    distance on average; taking the plain max over a non-contiguous set would let one noisy z-score
    downstream of a genuine crossover reopen the radius.

    ``eps_x[i] = 0`` means even the nearest neighbour is already distinguishable -- this point's own
    location already has full resolution. ``censored[i]`` is set when *all* ``k`` neighbours stayed
    indistinguishable: ``eps_x[i]`` is then a lower bound, right-censored by ``k`` -- raise ``k`` for
    a tighter estimate at that point.
    """
    from scipy.spatial import cKDTree

    N = len(X)
    tree = cKDTree(X)
    k_eff = min(k + 1, N)
    dists, idxs = tree.query(X, k=k_eff)          # column 0 is self, at distance 0
    if k_eff == 1:                                  # only one point in the dataset
        return np.zeros(N), np.zeros(N, dtype=bool)
    dists, idxs = dists[:, 1:], idxs[:, 1:]

    eps = np.zeros(N)
    censored = np.zeros(N, dtype=bool)
    for i in range(N):
        j_row = idxs[i]
        z = _z_gate(y[i], sigma[i], y[j_row], sigma[j_row])
        indist = z < z_thresh
        run = np.cumprod(indist.astype(np.int64)).astype(bool)   # longest indistinguishable prefix
        if run.any():
            eps[i] = dists[i][run].max()
            censored[i] = bool(run.all())
    return eps, censored


def cluster_by_resolution(X: np.ndarray, y: np.ndarray, sigma: np.ndarray, eps_x: np.ndarray, *,
                          z_thresh: float = 2.0):
    """``(n_eff, cluster_labels)`` -- merge training points that are **directly, pairwise**
    indistinguishable by :func:`_z_gate`, and count the resulting connected components, rather than
    dividing volume by one global ``eps_x`` (which cannot see the field's non-uniformity at all).

    ``eps_x`` is used only to bound the *candidate* pairs a KD-tree radius join proposes (``d <=
    min(eps_x_i, eps_x_j)``, so the search stays local and cheap) -- it is never trusted as a proxy
    for whether a given pair actually merges. Every candidate pair is re-tested directly with the
    same z-gate :func:`estimate_epsilon_x` used to build ``eps_x`` in the first place. This matters
    because ``eps_x[i]`` was computed against ``i``'s own k-nearest-neighbour set specifically; a
    *different* point ``j`` merely close enough in x to fall inside that radius was never itself
    z-tested against ``i``, and the score need not vary smoothly/monotonically between them -- two
    points closer than ``eps_x_i`` can still be genuinely distinguishable if there is a ridge or
    oscillation in that particular direction, and only a direct pairwise test catches that rather
    than assuming "closer than a point we already found indistinguishable" implies indistinguishable.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    N = len(X)
    max_eps = float(eps_x.max()) if N else 0.0
    rows, cols = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if max_eps > 0:
        tree = cKDTree(X)
        pairs = tree.query_pairs(r=max_eps, output_type="ndarray")
        if len(pairs):
            d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
            within_radius = d <= np.minimum(eps_x[pairs[:, 0]], eps_x[pairs[:, 1]])
            pairs = pairs[within_radius]
        if len(pairs):
            z = _z_gate(y[pairs[:, 0]], sigma[pairs[:, 0]], y[pairs[:, 1]], sigma[pairs[:, 1]])
            keep = z < z_thresh                      # the direct pairwise re-test, not a proxy
            pairs = pairs[keep]
            rows, cols = pairs[:, 0], pairs[:, 1]

    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    n_eff, cluster_labels = connected_components(graph, directed=False)
    return int(n_eff), cluster_labels


def estimate_r2_max(X_train: np.ndarray, y_train: np.ndarray, sigma_train: np.ndarray,
                    cluster_labels: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                    sigma_test: np.ndarray, *, z_thresh: float = 2.0) -> dict:
    """The ``R^2`` ceiling for a learner restricted to predicting one constant per resolution
    cluster (the best it can do once two points are indistinguishable is output the same value for
    both). Coverage is the direct pairwise gate, not a geometric radius lookup: for each test
    point's nearest training neighbour (by x-distance, to pick WHICH pair to test -- the gate itself
    never looks at x again), is the test point's own measured score within combined shot noise of
    that neighbour's (:func:`_z_gate`)? That is the same indistinguishability test
    :func:`estimate_epsilon_x` used to build ``eps_x`` in the first place, just spent directly on
    the (test, nearest-train) pair.

    Covered -> predict the nearest neighbour's resolution-cluster mean (the MSE-optimal constant
    for that cell). Uncovered -> reported **two ways**, since the choice materially changes the
    final number and there is no way to pick one that is not a convention: ``optimistic`` still
    predicts the nearest cluster's mean (as if the resolution constraint did not apply out there,
    the best case); ``strict`` falls back to the global training mean (the zero-skill constant a
    memorization-only learner is left with once nothing nearby is resolved). ``R^2`` is computed
    the same way as :func:`learner.base.r2_score` (``1 - SSE/SST``) so it is directly comparable to
    any learner's reported held-out ``R^2`` on the same split.
    """
    from scipy.spatial import cKDTree

    cluster_mean = {int(c): float(y_train[cluster_labels == c].mean())
                    for c in np.unique(cluster_labels)}
    global_mean = float(y_train.mean())

    tree = cKDTree(X_train)
    _, nn = tree.query(X_test, k=1)

    z = _z_gate(y_test, sigma_test, y_train[nn], sigma_train[nn])
    covered = z < z_thresh

    nn_mean = np.array([cluster_mean[int(cluster_labels[j])] for j in nn])
    pred_optimistic = nn_mean
    pred_strict = np.where(covered, nn_mean, global_mean)

    sst = float(((y_test - y_test.mean()) ** 2).sum())
    sst = max(sst, 1e-30)
    r2_optimistic = 1.0 - float(((y_test - pred_optimistic) ** 2).sum()) / sst
    r2_strict = 1.0 - float(((y_test - pred_strict) ** 2).sum()) / sst

    return {
        "r2_max_optimistic": r2_optimistic,
        "r2_max_strict": r2_strict,
        "r2_coverage_fraction": float(covered.mean()),
        "per_test_covered": covered,
        "per_test_nearest_train": nn,
    }


# --- wiring into the actual pipeline (shots branch) ------------------------------------------- #


def load_shots_dataset(cfg_path: str | Path, observable: str, *, out_root: str = "datasets"):
    """``(X, y, obs, seq, shots, cfg)`` for a shots-based config -- ``y`` is the same cached
    :func:`pipeline.score.load_soft` score every other eval module reads; ``seq``/``obs`` are the
    raw shot sequence and built observable :func:`bootstrap_shot_sigmas` needs, which
    :func:`pipeline.score.load_dataset` does not expose (it returns only the aggregated score).
    """
    from config import load_config
    from model import build_model
    from model.sampler import sample_X
    from observable import observable_on_keys
    from pipeline.score import context_for, load_soft
    from pipeline.shots import load_shots, shots_path

    cfg = load_config(cfg_path)
    if not cfg.generation.shots:
        raise SystemExit("resolution_ceiling needs a shots-based config (generation.shots > 0)")

    model = build_model(cfg)
    sdir = shots_path(cfg, model, root=out_root)
    if not sdir.exists():
        raise SystemExit(f"no shots at {sdir}; set generation.shots > 0 and run "
                         f"pipeline.generate first")

    keys, seq, meta = load_shots(sdir, cfg.generation.shots)
    shots = int(meta["shots"])
    ctx = context_for(meta, keys)
    obs = observable_on_keys(observable, ctx, keys)
    X = sample_X(int(meta["size"]), int(meta["n_features"]), int(meta["sample_seed"]))
    y = load_soft(None, observable, shots_dir=sdir, num_shots=shots)

    return X.double().numpy(), y.double().numpy(), obs, seq, shots, cfg


def run(cfg_path: str | Path, observable: str, *, out_root: str = "datasets", k: int = 10,
       z_thresh: float = 2.0, n_boot: int = 32, split_seed: int | None = None,
       boot_seed: int = 0) -> dict:
    from pipeline.split import split_indices

    X, y, obs, seq, shots, cfg = load_shots_dataset(cfg_path, observable, out_root=out_root)
    sigma = bootstrap_shot_sigmas(obs, seq, shots, n_boot=n_boot, seed=boot_seed)

    tr, te = split_indices(len(X), test_fraction=cfg.split.test_fraction,
                           split_seed=cfg.split.split_seed if split_seed is None else split_seed)
    tr, te = tr.numpy(), te.numpy()

    Xtr, ytr, sig_tr = X[tr], y[tr], sigma[tr]
    Xte, yte, sig_te = X[te], y[te], sigma[te]

    eps_x, censored = estimate_epsilon_x(Xtr, ytr, sig_tr, k=k, z_thresh=z_thresh)
    n_eff, cluster_labels = cluster_by_resolution(Xtr, ytr, sig_tr, eps_x, z_thresh=z_thresh)
    r2 = estimate_r2_max(Xtr, ytr, sig_tr, cluster_labels, Xte, yte, sig_te, z_thresh=z_thresh)

    result = {
        "config": str(cfg_path), "observable": observable, "shots": shots,
        "n_train": len(tr), "n_test": len(te), "k": k, "z_thresh": z_thresh, "n_boot": n_boot,
        "epsilon_x_mean": float(eps_x.mean()), "epsilon_x_median": float(np.median(eps_x)),
        "epsilon_x_censored_frac": float(censored.mean()),
        "n_eff": n_eff, "n_train_raw": len(tr),
        **r2,
    }
    return result


#: The observable set :mod:`eval.gradient_vs_hardness`/:mod:`eval.gradient_vs_r2_observable` sweep
#: -- reused here as the default for :func:`compare_observables` so a cross-observable ceiling run
#: lines up with the other per-observable sweeps already in this package, rather than picking an
#: independent default set.
DEFAULT_OBSERVABLES = ["parity", "n_first", "ent", "osc", "sq_parity", "xent_parity"]


def fit_best_learner(cfg, observable: str, *, out_root: str = "datasets",
                     scores_root: str = "scores", split_seed: int | None = None) -> dict:
    """Best-of-``ridge``/``svr``/``mlp`` (:data:`learner.auto.DEFAULT_SWEEP_LEARNERS`) held-out
    ``R^2`` on ``observable`` -- the number the ceiling in this module is meant to be compared
    against.

    Fits all three (not just the eventual winner) since which learner wins the ``R^2`` race is not
    knowable up front, and reports every arm so a caller can see how close the runner-up was.
    """
    from learner import kernel, nn  # noqa: F401 -- registration side effects
    from learner.auto import DEFAULT_SWEEP_LEARNERS
    from learner.base import build_learner, evaluate
    from pipeline.score import load_dataset
    from pipeline.split import split_indices

    X, y, _ = load_dataset(cfg, observable, out_root=out_root, scores_root=scores_root)
    tr, te = split_indices(len(X), test_fraction=cfg.split.test_fraction,
                           split_seed=cfg.split.split_seed if split_seed is None else split_seed)

    per_learner = {}
    for name, kwargs in DEFAULT_SWEEP_LEARNERS:
        learner = build_learner(name, **kwargs).fit(X[tr], y[tr])
        res = evaluate(learner, X[te], y[te])
        per_learner[name] = {"r2": res["r2"]}

    best = max(per_learner, key=lambda n: per_learner[n]["r2"])
    return {"best_learner": best, "best_r2": per_learner[best]["r2"], "per_learner": per_learner}


def compare_observables(cfg_path: str | Path, observables: list[str] = DEFAULT_OBSERVABLES, *,
                        out_root: str = "datasets", scores_root: str = "scores", k: int = 10,
                        z_thresh: float = 2.0, n_boot: int = 32,
                        split_seed: int | None = None) -> list[dict]:
    """Ceiling (this module) vs. actual best-of-3-learner performance (:func:`fit_best_learner`),
    one row per observable -- the comparison this module exists to make: is the achieved ``R^2``
    above or below what the shot-noise resolution limit alone predicts as the ceiling.

    A ceiling this module computes is only valid for a memorization/nearest-neighbour-shaped
    learner (see the module docstring): a global smooth model like ``mlp`` pools information across
    the *whole* training set rather than any one local pair, so it can legitimately clear this
    ceiling. When it does, that is reported, not hidden -- ``above_r2_ceiling`` flags it per row.

    One bad observable (a missing artifact, an unsupported combination) is printed and skipped, not
    fatal to the rest -- same per-cell resilience convention as the other ``eval/`` sweeps.
    """
    rows = []
    for obs in observables:
        try:
            ceiling = run(cfg_path, obs, out_root=out_root, k=k, z_thresh=z_thresh, n_boot=n_boot,
                          split_seed=split_seed)
            learned = fit_best_learner(load_config_cached(cfg_path), obs, out_root=out_root,
                                       scores_root=scores_root, split_seed=split_seed)
        except (Exception, SystemExit) as exc:          # noqa: BLE001 -- one bad observable must
            print(f"[resolution_ceiling] {obs} failed: {exc}")  # not abort the rest of the sweep
            continue

        row = {"observable": obs, "n_eff": ceiling["n_eff"], "n_train": ceiling["n_train"],
              "epsilon_x_median": ceiling["epsilon_x_median"],
              "epsilon_x_censored_frac": ceiling["epsilon_x_censored_frac"],
              "r2_max_optimistic": ceiling["r2_max_optimistic"],
              "r2_max_strict": ceiling["r2_max_strict"],
              "best_learner": learned["best_learner"], "achieved_r2": learned["best_r2"],
              "per_learner": learned["per_learner"],
              "above_r2_ceiling": learned["best_r2"] > ceiling["r2_max_optimistic"]}
        rows.append(row)
        print(f"{obs:<20} N_eff={row['n_eff']:>5}/{row['n_train']:<5}  "
              f"R2_max(opt/strict)={row['r2_max_optimistic']:.3f}/{row['r2_max_strict']:.3f}  "
              f"achieved_R2({row['best_learner']})={row['achieved_r2']:.3f} "
              f"{'ABOVE CEILING' if row['above_r2_ceiling'] else ''}", flush=True)
    return rows


def load_config_cached(cfg_path: str | Path):
    from config import load_config

    return load_config(cfg_path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--observable", default=None, help="single observable; mutually exclusive "
                    "with --observables")
    ap.add_argument("--observables", nargs="+", default=None, help="sweep several observables and "
                    "compare each to its best-of-ridge/svr/mlp learner (default: "
                    f"{DEFAULT_OBSERVABLES})")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--k", type=int, default=10, help="neighbours walked per training point when "
                    "crossing-over eps_x(x)")
    ap.add_argument("--z-thresh", type=float, default=2.0, help="indistinguishability gate: "
                    "|y_i - y_j| / sqrt(sigma_i^2 + sigma_j^2) < z-thresh")
    ap.add_argument("--n-boot", type=int, default=32, help="bootstrap replicates for the "
                    "per-row shot-noise sigma (O(n_boot * N * shots) -- lower this if slow)")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.observables or args.observable is None:
        observables = args.observables or DEFAULT_OBSERVABLES
        rows = compare_observables(args.config, observables, out_root=args.root,
                                   scores_root=args.scores_root, k=args.k, z_thresh=args.z_thresh,
                                   n_boot=args.n_boot, split_seed=args.split_seed)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2))
            print(f"wrote {args.out}")
        return

    result = run(args.config, args.observable, out_root=args.root, k=args.k,
                z_thresh=args.z_thresh, n_boot=args.n_boot, split_seed=args.split_seed)

    printable = {key: val for key, val in result.items() if not isinstance(val, np.ndarray)}
    print(json.dumps(printable, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(printable, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
