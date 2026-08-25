"""``dT/dx`` and the normalized gradient ``dT/dx / sqrt(Var_circ(O))``, from scratch, exact-only.

    from metrics.gradient import cached_gradient
    result = cached_gradient("configs/photonic.yaml", "parity")
    result["g_norm"]            # (N,) list, ||dT/dx|| at every point in the pool (or a subsample)
    result["g_norm_normalized"] # (N,) list, the same divided by sqrt(Var_circ(O))

**Applies to every observable except ``max_prob``.**  ``dT/dx_i = Cov_p(psi, s_i) = sum_n psi_n .
d_i p_n``, needing only the observable's influence function ``psi_n = dT/dp_n`` (defined for every
class -- ``Expectation``, ``Quadratic``, ``ProbFunction`` -- per :mod:`observable.base`'s own
docstring) and the input Jacobian ``dp``. The one exclusion is ``max_prob``
(``is_differentiable = False``, the sole class in the registry without a defined ``influence``) --
this module raises rather than silently returning a meaningless number for it.

**Finite differences, not autograd -- built here directly, no legacy machinery imported.**
Autograd (``torch.func.jacrev``) through a merlin-backed model OOMs past ``m~12`` on ordinary
hardware, because the backward graph retained scales with ``n_outcomes`` (77520 at ``m=14``), not
with the eventual ``(n_outcomes, n_features)`` Jacobian shape -- central differences
(``2*n_features`` plain forward calls to ``model.probs(x, grad=False)``, no graph at all) sidestep
this entirely and are the only Jacobian path used here, at every size.

**No shots.**  Every ``p``/``dp`` here comes from the exact distribution -- do not add a ``shots=``
kwarg; see :mod:`metrics.shot_variance`'s module docstring for why an exact-regime metric should
stay that way rather than growing a shots branch that silently changes what it measures.

**``spin_magic`` readout conditioning (``readout_mu_zero``).**  ``model.probs(x, grad=False)`` on a
live model returns the *raw*, unconditioned basis -- for ``spin_magic`` that includes its two
readout modes (:meth:`~model.photonic.PhotonicModel.readout_modes`), doubling the outcome count
against :func:`~pipeline.distribution.load_dist`'s default (``load_full=False``), which applies the
``mu = 0`` post-selection at load time.  A saved dataset's ``dist.keys`` is therefore
post-selection-sized while a fresh finite-difference ``model.probs()`` call is not -- mismatched
outcome counts across the two, which crashes :func:`dT_dx`'s ``dp.T @ psi`` on a shape mismatch
whenever the observable was built (via ``resolve_observable``) against the saved, conditioned
``dist.keys``.  ``readout_mu_zero=True`` (the default) fixes this by conditioning every raw
``probs()`` call the same way :func:`~pipeline.distribution.load_dist` conditions the saved
artifact -- via :func:`~pipeline.distribution.readout_condition`, the same function both already
share (see :func:`~eval_legacy.sweep_delta.readout_zero_probs`, an existing live-model use of it) --
so the live and saved bases always agree.  A no-op for every model without readout modes (every prep
except ``spin_magic``).  Pass ``readout_mu_zero=False`` to get the raw, unconditioned basis instead
(only meaningful together with an observable resolved against the *unconditioned* basis, e.g. via
``load_dist(path, load_full=True)`` -- mixing conditioned and unconditioned bases is exactly the bug
this flag exists to prevent).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

if __package__ in (None, ""):                    # allow `python metrics/gradient.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "metrics"

#: Central-difference step -- same value and same rationale as the rest of this repo's FD paths
#: (large deliberately: truncation error is O(eps^2), round-off is O(float_eps/eps), these models
#: are float32-rooted so shrinking eps makes it worse, not better).
FD_EPS = 1e-2


def _condition_probs(model, P: torch.Tensor, *, readout_mu_zero: bool) -> torch.Tensor:
    """``P`` (``(1, n_out_raw)``, aligned to ``model.outcome_keys()``), conditioned on the readout
    modes to match :func:`~pipeline.distribution.load_dist`'s default-loaded basis -- see the
    module docstring's ``readout_mu_zero`` note.  A no-op when ``readout_mu_zero=False`` or the
    model has no readout modes (every prep except ``spin_magic``)."""
    if not readout_mu_zero:
        return P
    readout = model.readout_modes()
    if not readout:
        return P
    from pipeline.distribution import readout_condition

    keys = model.outcome_keys()
    structure = getattr(model.prep, "structure", None)
    (P,), _ = readout_condition((P,), keys, readout, structure=structure)
    return P


def probs_and_jacobian_fd(model, x: torch.Tensor, *, eps: float = FD_EPS,
                          readout_mu_zero: bool = True):
    """``(p, dp)`` at one input ``x (n_f,)``: ``p (n_out,)``, ``dp (n_out, n_f)`` by central
    differences -- ``2 * n_f`` calls to ``model.probs(., grad=False)``, no autograd graph.

    ``readout_mu_zero`` (default ``True``) conditions every raw ``probs()`` call on the readout
    modes before differencing -- see the module docstring; matches the basis a saved-and-reloaded
    ``spin_magic`` distribution has by default, and is a no-op for every other prep.
    """
    p = _condition_probs(model, model.probs(x.unsqueeze(0), grad=False),
                        readout_mu_zero=readout_mu_zero)[0].double()
    cols = []
    for i in range(int(x.shape[0])):
        h = torch.zeros_like(x)
        h[i] = float(eps)
        p_plus = _condition_probs(model, model.probs((x + h).unsqueeze(0), grad=False),
                                 readout_mu_zero=readout_mu_zero)[0].double()
        p_minus = _condition_probs(model, model.probs((x - h).unsqueeze(0), grad=False),
                                  readout_mu_zero=readout_mu_zero)[0].double()
        cols.append((p_plus - p_minus) / (2.0 * float(eps)))
    dp = torch.stack(cols, dim=1)
    return p, dp


def dT_dx(obs, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
    """``g_i = dT/dx_i = sum_n psi_n . d_i p_n``, ``(n_f,)`` -- the exact gradient of the label map.

    ``psi`` is defined only up to an additive constant (``sum_n d_i p_n = 0`` for every direction,
    since probabilities sum to 1), so no projection/centering is needed here beyond what
    ``obs.influence`` itself returns.
    """
    psi = obs.influence(p.unsqueeze(0))[0].double()
    return dp.double().T @ psi


#: Each point costs 2 * n_features full circuit forward passes (:func:`probs_and_jacobian_fd`) --
#: the full pool (typically 10_000 rows) is too expensive to default to, unlike
#: :mod:`metrics.shot_variance`/:mod:`metrics.circuit_variance`, which only need one forward pass
#: per point and so use the whole pool. 100 matches this project's own convention for gradient
#: sweeps elsewhere (e.g. the normalized-gradient-minimum construction discussed for the paper).
DEFAULT_N_X = 100


def cached_gradient(cfg_path: str | Path, observable: str, *, n_x: int | None = DEFAULT_N_X,
                    fd_eps: float = FD_EPS, readout_mu_zero: bool = True,
                    out_root: str | Path = "datasets", scores_root: str | Path = "scores",
                    graph_density: float = 0.5, force: bool = False) -> dict:
    """``||dT/dx||`` and its ``Var_circ``-normalized form at the first ``n_x`` rows (default 100,
    not the full pool -- see :data:`DEFAULT_N_X`) of ``cfg_path``'s input pool, or a prior run's
    cache.  Pass ``n_x=None`` explicitly to use every row instead.

    ``readout_mu_zero`` (default ``True``) -- see the module docstring -- fixes a basis mismatch on
    ``spin_magic`` configs: without it, the finite-difference ``model.probs()`` calls return the
    raw, unconditioned (readout-inclusive) basis while the observable is resolved against the
    saved, ``mu=0``-conditioned ``dist.keys``, crashing on a shape mismatch.  Folded into the cache
    path (``__mu0`` / ``__full`` suffix) so a cache entry written by a version of this function
    before this flag existed is never returned for a call that now expects the fix -- that stale
    entry (from a run that either crashed before writing one, or, on a non-``spin_magic`` config
    where conditioning is a no-op, wrote an identical result) simply misses and recomputes; pass
    ``force=True`` to also ignore a same-flag cache entry and refit from scratch.

    Returns ``{"g_norm": [...], "g_norm_normalized": [...], "n": N, "artifact": name}``.  Raises for
    ``max_prob`` (or any other ``is_differentiable=False`` observable) -- there is no gradient to
    report for it.
    """
    from config import load_config
    from model import build_model
    from observable import resolve_observable
    from pipeline.artifact import exact_path
    from pipeline.distribution import load_dist
    from pipeline.score import context_for

    from .circuit_variance import cached_circuit_variance

    cfg = load_config(cfg_path)
    if cfg.generation.shots:
        raise ValueError(f"{cfg_path!r} has generation.shots={cfg.generation.shots}; "
                         "the gradient is exact-only -- point this at an exact (shots=0) config")

    model = build_model(cfg)
    path = exact_path(cfg, model, out_root)
    if not path.exists():
        raise SystemExit(f"no artifact at {path}; run `python -m pipeline.generate --config "
                         f"{cfg_path}` first")

    dist = load_dist(path)
    artifact_name = str(dist.meta["hash"])

    mu_tag = "mu0" if readout_mu_zero else "full"
    cache_path = (Path(scores_root) / artifact_name / "exact" / "gradient"
                 / f"{observable}__n{n_x or 'all'}__{mu_tag}.json")
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    ctx = context_for(dist.meta, dist.keys, dist.probs_at_zero.numpy(), graph_density=graph_density)
    obs = resolve_observable(observable, ctx)
    if not obs.is_differentiable:
        raise ValueError(f"{observable!r} is not differentiable in p (is_differentiable=False); "
                         f"dT/dx is undefined for it")

    circ = cached_circuit_variance(cfg_path, observable, out_root=out_root,
                                   scores_root=scores_root, graph_density=graph_density,
                                   force=force)
    sqrt_var_circ = max(circ["var_circ"], 1e-300) ** 0.5

    n_rows = len(dist) if n_x is None else min(int(n_x), len(dist))
    g_norm = []
    for i in range(n_rows):
        p, dp = probs_and_jacobian_fd(model, dist.X[i], eps=fd_eps,
                                      readout_mu_zero=readout_mu_zero)
        g = dT_dx(obs, p, dp)
        g_norm.append(float(g.norm()))

    result = {
        "g_norm": g_norm,
        "g_norm_normalized": [g / sqrt_var_circ for g in g_norm],
        "n": n_rows,
        "artifact": artifact_name,
        "observable": observable,
        "fd_eps": fd_eps,
        "readout_mu_zero": readout_mu_zero,
        "sqrt_var_circ": sqrt_var_circ,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))
    return result


def main(argv=None) -> None:
    import argparse
    import statistics

    ap = argparse.ArgumentParser(description="||dT/dx|| and its normalized form over a config's "
                                             "input pool, cached.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--observables", nargs="+", required=True)
    ap.add_argument("--n-x", type=int, default=DEFAULT_N_X, help="subsample this many rows "
                                                                "(default 100; pass 0 for all "
                                                                "rows in the pool)")
    ap.add_argument("--out-root", default="datasets")
    ap.add_argument("--scores-root", default="scores")
    ap.add_argument("--force", action="store_true", help="ignore any cached result and recompute")
    ap.add_argument("--no-readout-mu-zero", action="store_true",
                    help="use the raw, unconditioned basis instead of the default mu=0 "
                         "readout post-selection (spin_magic only; no effect on other preps)")
    args = ap.parse_args(argv)

    n_x = None if args.n_x == 0 else args.n_x
    for obs in args.observables:
        res = cached_gradient(args.config, obs, n_x=n_x, out_root=args.out_root,
                              scores_root=args.scores_root, force=args.force,
                              readout_mu_zero=not args.no_readout_mu_zero)
        print(f"{obs:32s} min||g||={min(res['g_norm']):.4g}  "
             f"min||g||_norm={min(res['g_norm_normalized']):.4g}  n={res['n']}")


if __name__ == "__main__":
    main()
