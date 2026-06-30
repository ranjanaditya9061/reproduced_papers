"""Sweep one focused parameter of a config over a list of values.

Loads a base config, overrides a single ``<section>.<field>`` for each value in
``--values``, (re)generates the dataset, and prints a one-line summary per value.

Because artifacts are content-hashed:
- sweeping a **generation** field (e.g. ``problem.k``, ``generation.size``,
  ``seeds.data_seed``) produces a distinct dataset per value;
- sweeping a **prepare** field (e.g. ``prepare.min_margin``) reuses the *same*
  cached pool and just re-labels/filters it — so you see yield-vs-margin for free.

Usage (run from the paper root)::

    python sweep.py --config configs/example_photonic.yaml \
        --param prepare.min_margin --values 0.0 0.05 0.10 0.20

    python sweep.py --config configs/example_photonic.yaml \
        --param problem.k --values 2 3 4
"""

from __future__ import annotations

import argparse

from Generator import artifact_path, generate, load_config, load_raw
from Generator.prepare import derive_confidence, derive_labels

SECTIONS = ("problem", "generation", "prepare", "seeds")


def _coerce(raw: str, current):
    """Cast the CLI string to the type of the field's current value."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None:
        if raw.lower() in ("null", "none"):
            return None
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                pass
    return raw  # str


def set_param(cfg, dotted: str, raw: str) -> None:
    """Override ``cfg.<section>.<field>`` in place from a ``section.field`` path."""
    section, _, field = dotted.partition(".")
    if section not in SECTIONS or not field:
        raise SystemExit(
            f"--param must be '<section>.<field>' with section in {list(SECTIONS)}; "
            f"got {dotted!r}"
        )
    obj = getattr(cfg, section)
    if not hasattr(obj, field):
        raise SystemExit(f"unknown field {field!r} in section {section!r}")
    setattr(obj, field, _coerce(raw, getattr(obj, field)))


def sweep(config_path, param, values, *, out_root="datasets", force=False):
    """Generate + summarize one dataset per value of ``param``."""
    print(f"Sweeping {param} over {list(values)}  (base config: {config_path})")
    for raw in values:
        cfg = load_config(config_path)          # fresh each iteration
        set_param(cfg, param, raw)
        cfg.validate()

        path = generate(cfg, out_root=out_root, force=force)
        X, soft, _ = load_raw(path)
        y = derive_labels(soft)
        mm = cfg.prepare.min_margin
        surv = int((derive_confidence(soft) >= mm).sum())
        n = X.shape[0]
        print(
            f"  {param}={str(raw):>8}  ->  {path.name}  "
            f"pool={n}  class1={int((y == 1).sum())}  "
            f"survive@{mm}={surv} ({surv / max(n, 1):.0%})"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sweep one config parameter over a list of values.")
    ap.add_argument("--config", required=True, help="base YAML config")
    ap.add_argument("--param", required=True, help="section.field to vary, e.g. prepare.min_margin")
    ap.add_argument("--values", nargs="+", required=True, help="values to sweep over")
    ap.add_argument("--out-root", default="datasets", help="root dir for artifacts")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    args = ap.parse_args(argv)
    sweep(args.config, args.param, args.values, out_root=args.out_root, force=args.force)


if __name__ == "__main__":
    main()