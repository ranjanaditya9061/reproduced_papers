"""Generate a random ``prod_parity`` polynomial sweep config for :mod:`learn.poly_sweep`.

Emits a YAML with a ``name`` and a ``polynomials: {label: observable}`` mapping -- the exact
format :func:`learn.poly_sweep.load_sweep` reads.  Each polynomial is a random sum of
square-free monomials in the photon counts: pick ``n_terms`` distinct monomials, each of a
degree ("order") drawn from ``--degrees``, over ``m`` modes; a term may be ``N``-subtracted
with probability ``--subtract-prob``.  Every observable is validated (and any that cancels to
an empty polynomial mod 2 is rejected), and each line is annotated with its readable form.

    # 6 polynomials, monomial degrees in {1,2,3} over m=6 modes, up to 2 terms each
    python -m learn.make_polys --m 6 --degrees 1 2 3 --count 6 --name "Random low order" --seed 0

    # one polynomial per degree (a clean order progression), single monomial each
    python -m learn.make_polys --m 6 --degrees 1 2 3 4 5 6 --one-per-degree --max-terms 1 \\
        --name "Consecutive random"

Then run it:  ``python -m learn.poly_sweep --config <data cfg> --sweep polytest/<slug>.yaml``
"""

from __future__ import annotations

import re
from pathlib import Path

# Support `python -m learn.make_polys` and `python learn/make_polys.py`.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "learn"

#: where sweep configs live (matches learn.poly_sweep.OUTPUT_DIR)
OUTPUT_DIR = "polytest"


def _slug(name: str) -> str:
    """Filesystem-safe stem from a sweep name (``Random Third Order`` -> ``random_third_order``)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "random_polys"


def random_polynomial(m, degrees, n_terms, rng, *, subtract_prob=0.0):
    """A random ``prod_parity`` observable: ``n_terms`` distinct monomials over ``m`` modes.

    Each monomial's degree is sampled from ``degrees`` (clamped to ``m``); its modes are a
    random distinct subset.  A term is ``N``-subtracted with probability ``subtract_prob``
    (else ``M``, added).  Duplicate monomials within the polynomial are skipped so terms do
    not silently cancel.
    """
    monos, seen = [], set()
    guard = 0
    while len(monos) < n_terms and guard < 200:
        guard += 1
        d = min(rng.choice(degrees), m)
        if d < 1:
            continue
        mono = tuple(sorted(rng.sample(range(m), d)))
        if mono in seen:
            continue
        seen.add(mono)
        sign = "N" if rng.random() < subtract_prob else "M"
        monos.append(sign + "-".join(str(i) for i in mono))
    return "prod_parity__" + "__".join(monos)


def build_polynomials(m, *, count, degrees, max_terms, rng, subtract_prob=0.0,
                      one_per_degree=False, labels=None):
    """Ordered ``{label: observable}`` of ``count`` distinct, valid random polynomials.

    With ``one_per_degree`` the i-th polynomial's monomials are all of degree ``degrees[i]``
    (a clean order progression, ``count`` defaults to ``len(degrees)``); otherwise each term's
    degree is drawn independently from ``degrees``.  Observables are deduplicated and validated
    via :func:`model.photonic.parse_prod_parity` (rejecting any that cancels to empty).
    """
    from model.photonic import parse_prod_parity

    out, used = {}, set()
    guard = 0
    while len(out) < count and guard < 2000:
        guard += 1
        i = len(out)
        degs = [degrees[i % len(degrees)]] if one_per_degree else degrees
        n_terms = 1 if one_per_degree and max_terms == 1 else rng.randint(1, max_terms)
        obs = random_polynomial(m, degs, n_terms, rng, subtract_prob=subtract_prob)
        if obs in used:
            continue
        try:
            parse_prod_parity(obs, m)                        # valid + non-cancelling?
        except ValueError:
            continue
        used.add(obs)
        out[labels[i] if labels and i < len(labels) else f"P{i + 1}"] = obs
    if len(out) < count:
        raise RuntimeError(f"could only build {len(out)}/{count} distinct polynomials "
                           f"(m={m}, degrees={degrees}); widen --degrees or raise --max-terms")
    return out


def write_config(path, name, polynomials, m):
    """Write the sweep config YAML, each observable annotated with its readable polynomial."""
    from .poly_sweep import readable_poly

    wl = max(len(lbl) for lbl in polynomials)
    wo = max(len(o) for o in polynomials.values())
    lines = [
        f"# Auto-generated random prod_parity polynomials (m={m}).  Edit freely, or regenerate",
        "# with learn.make_polys.  Run:  python -m learn.poly_sweep --config <data cfg> "
        f"--sweep {path}",
        f"name: {name}",
        "polynomials:",
    ]
    for lbl, obs in polynomials.items():
        lines.append(f"  {lbl + ':':<{wl + 1}} {obs:<{wo}}   # {readable_poly(obs, m)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> None:
    import argparse
    import random

    ap = argparse.ArgumentParser(
        prog="learn.make_polys",
        description="Generate a random prod_parity polynomial sweep config for learn.poly_sweep.")
    ap.add_argument("--m", type=int, default=6, help="number of modes (mode indices live in [0, m))")
    ap.add_argument("--degrees", nargs="+", type=int, default=[1, 2, 3],
                    help="pool of monomial degrees ('orders') to draw from (clamped to m)")
    ap.add_argument("--count", type=int, default=10,
                    help="number of polynomials (default: 6, or len(--degrees) with --one-per-degree)")
    ap.add_argument("--max-terms", type=int, default=10,
                    help="max monomials per polynomial (each poly draws 1..max-terms)")
    ap.add_argument("--subtract-prob", type=float, default=0.0,
                    help="probability a monomial is N-subtracted (else M-added)")
    ap.add_argument("--one-per-degree", action="store_true",
                    help="poly i uses only degree --degrees[i] (a clean order progression)")
    ap.add_argument("--name", default="Random polynomials", help="sweep name (also the file slug)")
    ap.add_argument("--labels", nargs="*", default=None, help="explicit labels (default P1, P2, ...)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible configs)")
    ap.add_argument("--out", default=None,
                    help=f"output path (default {OUTPUT_DIR}/<slug of --name>.yaml)")
    args = ap.parse_args(argv)

    degrees = [d for d in args.degrees if 1 <= d <= args.m]
    if not degrees:
        ap.error(f"no valid degrees in {args.degrees} for m={args.m} (need 1 <= d <= m)")
    count = args.count if args.count is not None else (len(degrees) if args.one_per_degree else 6)

    rng = random.Random(args.seed)
    polynomials = build_polynomials(
        args.m, count=count, degrees=degrees, max_terms=args.max_terms, rng=rng,
        subtract_prob=args.subtract_prob, one_per_degree=args.one_per_degree, labels=args.labels)

    name = f"{args.name} (seed {args.seed})"                  # seed in the name -> distinct slug/config
    out = args.out or str(Path(OUTPUT_DIR) / f"{_slug(name)}.yaml")
    write_config(out, name, polynomials, args.m)

    from .poly_sweep import readable_poly
    print(f"[make_polys] wrote {out}  (name '{name}', {len(polynomials)} polynomials, m={args.m})")
    wl = max(len(lbl) for lbl in polynomials)
    for lbl, obs in polynomials.items():
        print(f"  {lbl:>{wl}}  {obs}   =  {readable_poly(obs, args.m)}")


if __name__ == "__main__":
    main()