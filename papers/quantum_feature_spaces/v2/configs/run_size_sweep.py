"""Drive data generation over ``configs/size_sweep_full/`` in size order: every ``m=6`` config
completes before any ``m=8`` config starts, and so on.

    python configs/run_size_sweep.py
    python configs/run_size_sweep.py --sizes 6 8          # only these m values
    python configs/run_size_sweep.py --root datasets --force

**Why not just ``pipeline.generate --configs-dir``.**  That sorts filenames alphabetically, which
is size order only while every ``m`` is a single digit -- ``m10`` sorts before ``m6``.  This script
parses ``mMkK`` out of each filename and groups/sorts on the integer ``m``, so the ordering holds
once the sweep is widened past ``m=9``.

**One bad config does not stop the run.**  A single failure (e.g. a size that blows the memory
guard, or a prep that cannot run in this environment) is caught, printed, and recorded -- every
other config at every size still gets a chance to generate.  The failures are summarised at the end
so nothing silently goes missing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

if __package__ in (None, ""):                    # allow `python configs/run_size_sweep.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from pipeline.generate import generate_exact, generate_shots

CONFIG_DIR = Path(__file__).parent / "size_sweep_full"
#: Matches "m<M>k<K>" anywhere in the stem -- generate_size_sweep.py's tag sits mid-filename
#: (e.g. "spin_m6k3_encboth_l1"), not necessarily at the end.
_SIZE_RE = re.compile(r"m(\d+)k(\d+)")


def _size_of(path: Path) -> tuple[int, int]:
    m = _SIZE_RE.search(path.stem)
    if not m:
        raise ValueError(f"{path.name} does not match the *m<M>k<K>* naming convention")
    return int(m.group(1)), int(m.group(2))


def grouped_by_size(config_dir: Path = CONFIG_DIR) -> dict[int, list[Path]]:
    """``{m: [config paths]}``, each list sorted by name for a stable, repeatable run order."""
    groups: dict[int, list[Path]] = {}
    for p in sorted(config_dir.glob("*.yaml")):
        m, _ = _size_of(p)
        groups.setdefault(m, []).append(p)
    return groups


def run(*, config_dir: Path = CONFIG_DIR, sizes: list[int] | None = None,
       root: str = "datasets", force: bool = False) -> list[tuple[Path, Exception]]:
    """Generate every config in ``config_dir``, one ``m`` tier at a time, ascending.

    Returns the ``(path, exception)`` pairs that failed -- empty if everything succeeded.
    """
    groups = grouped_by_size(config_dir)
    order = sorted(sizes) if sizes else sorted(groups)
    failures = []

    for m in order:
        paths = groups.get(m, [])
        if not paths:
            print(f"=== m={m}: no configs found, skipping")
            continue
        print(f"=== m={m}: {len(paths)} configs")
        for p in paths:
            print(f"--- {p.name}")
            try:
                cfg = load_config(p)
                saved = (generate_shots(cfg, root=root, force=force) if cfg.generation.shots
                         else generate_exact(cfg, root=root, force=force))
                print("    saved:", saved)
            except Exception as exc:                       # noqa: BLE001 -- one bad config must
                print(f"    FAILED: {exc}")                # not stop the rest of the sweep
                failures.append((p, exc))
        print()

    return failures


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--sizes", nargs="+", type=int, default=None,
                    help="only these m values, in the order given (default: every m present, "
                    "ascending)")
    ap.add_argument("--root", default="datasets", help="store root; both branches live under it")
    ap.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = ap.parse_args(argv)

    failures = run(config_dir=Path(args.config_dir), sizes=args.sizes, root=args.root,
                  force=args.force)

    if failures:
        print(f"\n{len(failures)} config(s) failed:")
        for p, exc in failures:
            print(f"  {p.name}: {exc}")
        raise SystemExit(1)
    print("all configs generated successfully")


if __name__ == "__main__":
    main()
