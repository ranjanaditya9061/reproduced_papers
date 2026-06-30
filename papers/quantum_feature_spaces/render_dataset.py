"""Render a dataset's teacher circuit, if the teacher is circuit-based.

    python render_dataset.py --config configs/data_spoqc.yaml [--out circuit.png]

Builds the teacher named by the data config and calls its ``render`` method.
Circuit-based teachers (``spoqc_photonic``, …) save a diagram; non-circuit
teachers (``analytical``, ``mlp``) report that rendering is unavailable instead
of erroring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Make the paper packages importable regardless of how this is invoked.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generator import load_config
from model import build_teacher


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Render a data config's teacher circuit if available.")
    ap.add_argument("--config", required=True, help="path to a data (generation) config YAML")
    ap.add_argument("--out", default=None, help="output image path (default: <generator>.png)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = args.out or f"{cfg.generation.generator}.png"
    teacher = build_teacher(cfg)

    try:
        path = teacher.render(out)
    except NotImplementedError as e:
        print(f"[render] unavailable for generator {cfg.generation.generator!r}: {e}")
        return
    except ImportError as e:
        print(f"[render] cannot render (missing dependency): {e}")
        return

    print(f"[render] {cfg.generation.generator} circuit -> {path}")


if __name__ == "__main__":
    main()