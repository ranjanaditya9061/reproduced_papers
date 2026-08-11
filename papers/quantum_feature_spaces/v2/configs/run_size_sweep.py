"""Drive data generation over every ``(m, k)`` size tier from the SAME ``(6, 3)`` config templates
in ``configs/size_sweep_full/`` -- ``m=6`` completes before ``m=8`` starts, and so on -- without
writing a second batch of config files for each size.

    python configs/run_size_sweep.py
    python configs/run_size_sweep.py --sizes 6 8 10 12    # only these m values, in this order
    python configs/run_size_sweep.py --root datasets --force

**No per-size config files.**  ``configs/size_sweep_full/*.yaml`` are written once, at ``(6, 3)``,
by ``configs/generate_size_sweep.py``.  This script loads each of them, overrides
``problem.m``/``problem.k`` in memory for the target size (``k = m // 2``, the ratio every config
here already uses), re-validates, and generates from the mutated config directly -- ``m=8`` is
"the same (6,3) config with m,k swapped," not a new file.  ``cx_pairs: "chain"`` already re-resolves
itself from whatever ``k`` it sees (:func:`circuit.spin.normalize_cx_pairs`), so nothing else needs
adjusting when the size changes.

**Only ``photonic``/``fock`` continues past ``m=12``.**  Every other kind hits the exact-``probs``
memory wall there (:mod:`pipeline.distribution`'s ``check_size``); from ``m=14`` on, only the
``photonic_fock*`` configs run, switched onto the shots branch (``generation.shots`` set instead of
relying on exact ``probs``).

**One bad config does not stop the run.**  A single failure (e.g. a size that blows the memory
guard, or a prep that cannot run in this environment) is caught, printed, and recorded -- every
other config at every size still gets a chance to generate.  The failures are summarised at the end
so nothing silently goes missing.

**Every print is flushed immediately.**  ``spin``/``spin_magic`` configs build one perceval
processor per row and can take a long time even with ``generation.n_jobs`` raised (see
``configs/generate_size_sweep.py``'s ``SPIN_N_JOBS``), and Python line-buffers stdout only when it
is a TTY -- redirected to a file or piped (e.g. run in the background), it block-buffers by default,
so nothing appears for the whole run and it looks hung even though it is working.  Every ``print``
here passes ``flush=True`` for that reason, and each config reports its own elapsed time on
completion so a slow one is visibly progressing, not silent.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

if __package__ in (None, ""):                    # allow `python configs/run_size_sweep.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from pipeline.generate import generate_exact, generate_shots

CONFIG_DIR = Path(__file__).parent / "size_sweep_full"
#: (m, k) tiers with exact probs -- every template runs here.
EXACT_SIZES = [(6, 3), (8, 4), (10, 5), (12, 6)]
#: (m, k) tiers past the exact-probs memory wall -- photonic/fock only, via the shots branch.
SHOTS_SIZES = [(14, 7), (16, 8), (18, 9)]
#: Shots to draw per row at the SHOTS_SIZES tier.
SHOTS_BUDGET = 10_000


#: The template's own size -- n_jobs scaling below is relative to what configs/generate_size_sweep.py
#: measured RAM against when it wrote generation.n_jobs into each template.
_TEMPLATE_M, _TEMPLATE_K = 6, 3
#: Mirrors configs/generate_size_sweep.py's PER_WORKER_RAM_GB -- kept in sync manually since the
#: two scripts don't share that constant across the process boundary a fresh interpreter run implies.
_PER_WORKER_RAM_GB = 1.5


#: Preps that actually consume generation.n_jobs (circuit.spin.parallel_row_map's process pool,
#: via SpinPrep.probs / SpinMagicPrep.probs).  Every other kind is batched (merlin) or pure-torch
#: and a process pool would be a pure no-op for it -- re-deriving n_jobs there wastes a psutil call
#: for nothing, so this is what _for_size checks, not the template's numeric n_jobs value (which
#: is ambiguous: "1" means "this kind ignores it" for those kinds, but for spin/spin_magic it just
#: means whatever configs/generate_size_sweep.py measured on ITS machine at write time -- treating
#: that "1" as "leave it alone" was the bug: it silently pinned spin/spin_magic to 1 job forever
#: even on a machine with cores and RAM to spare).
_POOLED_PREPS = ("spin", "spin_magic")


def _n_jobs_for_size(m: int, k: int) -> int:
    """``n_jobs`` for ``(m, k)`` from *live* available RAM and cores, not a value baked into a
    template at write time -- so a size bump both shrinks the worker count when memory is tight
    (a bigger circuit needs more per-row state) and grows it when there is real headroom (a beefier
    machine, or RAM freed up since the template was written).  Falls back to 1 (serial) if
    ``psutil`` is not installed -- correct, just slow, never silently wrong.
    """
    try:
        import math
        import os
        import psutil
        outcomes = math.comb(m + k - 1, k)
        base_outcomes = math.comb(_TEMPLATE_M + _TEMPLATE_K - 1, _TEMPLATE_K)
        per_worker_gb = _PER_WORKER_RAM_GB * max(1.0, outcomes / base_outcomes)
        cpu_cap = max(1, (os.cpu_count() or 4) - 1)
        ram_cap = max(1, int(psutil.virtual_memory().available / (per_worker_gb * 1024 ** 3)))
        return min(cpu_cap, ram_cap)
    except ImportError:
        return 1


def _for_size(cfg, m: int, k: int, *, shots: int = 0):
    """A copy of ``cfg`` overridden to ``(m, k)`` (and ``generation.shots`` if given), re-validated.

    Every knob that depends on ``k`` (``cx_pairs: "chain"``, the prep's own geometry checks) is
    re-resolved from the mutated ``problem.k`` at ``build_prep`` time -- nothing here needs to know
    which prep it is dealing with.  ``generation.n_jobs`` is re-derived for ``spin``/``spin_magic``
    (the only preps that use it) from live resources -- see :func:`_n_jobs_for_size`; every other
    kind keeps the template's value (it is unused there regardless of what it says).
    """
    import copy
    c = copy.deepcopy(cfg)
    c.problem.m, c.problem.k = m, k
    if c.model.prep in _POOLED_PREPS:
        c.generation.n_jobs = _n_jobs_for_size(m, k)
    if shots:
        c.generation.shots = shots
    c.validate()
    return c


def run(*, config_dir: Path = CONFIG_DIR, sizes: list[tuple[int, int]] | None = None,
       root: str = "datasets", force: bool = False) -> list[tuple[Path, tuple[int, int], Exception]]:
    """Generate every ``(6, 3)`` template in ``config_dir`` at every size in ``sizes``, one size
    tier at a time, ascending.  ``photonic_fock*`` templates also run the ``SHOTS_SIZES`` tier.

    Returns the ``(path, (m, k), exception)`` triples that failed -- empty if everything succeeded.
    """
    templates = sorted(config_dir.glob("*.yaml"))
    if not templates:
        raise SystemExit(f"no config templates found in {config_dir}")

    order = sizes if sizes else EXACT_SIZES + SHOTS_SIZES
    failures = []

    for m, k in order:
        shots_tier = (m, k) in SHOTS_SIZES
        paths = [p for p in templates if not shots_tier or p.stem.startswith("photonic_fock")]
        if not paths:
            print(f"=== (m={m}, k={k}): no applicable templates, skipping", flush=True)
            continue
        print(f"=== (m={m}, k={k}): {len(paths)} configs"
              + (" (shots branch)" if shots_tier else ""), flush=True)
        for i, p in enumerate(paths, 1):
            t0 = time.monotonic()
            try:
                base = load_config(p)
                cfg = _for_size(base, m, k, shots=SHOTS_BUDGET if shots_tier else 0)
                print(f"--- [{i}/{len(paths)}] {p.name} (n_jobs={cfg.generation.n_jobs})",
                      flush=True)
                saved = (generate_shots(cfg, root=root, force=force) if cfg.generation.shots
                         else generate_exact(cfg, root=root, force=force))
                print(f"    saved: {saved}  ({time.monotonic() - t0:.1f}s)", flush=True)
            except Exception as exc:                       # noqa: BLE001 -- one bad config must
                print(f"    FAILED after {time.monotonic() - t0:.1f}s: {exc}", flush=True)
                failures.append((p, (m, k), exc))          # not stop the rest of the sweep
        print(flush=True)

    return failures


def _parse_size(s: str) -> tuple[int, int]:
    m = int(s)
    if m % 2:
        raise argparse.ArgumentTypeError(f"m must be even for k=m//2 (got {m})")
    return m, m // 2


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the size sweep from the (6,3) config templates, "
                                              "overriding m/k per size rather than writing new "
                                              "config files.")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--sizes", nargs="+", type=_parse_size, default=None,
                    help="m values (k = m // 2), in the order given -- e.g. '6 8 10 12'. "
                    "Default: EXACT_SIZES then SHOTS_SIZES, ascending.")
    ap.add_argument("--root", default="datasets", help="store root; both branches live under it")
    ap.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = ap.parse_args(argv)

    failures = run(config_dir=Path(args.config_dir), sizes=args.sizes, root=args.root,
                  force=args.force)

    if failures:
        print(f"\n{len(failures)} config(s) failed:", flush=True)
        for p, size, exc in failures:
            print(f"  {p.name} @ {size}: {exc}", flush=True)
        raise SystemExit(1)
    print("all configs generated successfully", flush=True)


if __name__ == "__main__":
    main()
