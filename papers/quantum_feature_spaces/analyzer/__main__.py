"""Analyzer CLI with two subcommands::

    python -m analyzer preview --config <data.yaml>  [-n N]   # inspect a dataset
    python -m analyzer compare --config <embed.yaml> [--target] [--force]

For backward compatibility, a bare ``--config`` (no subcommand) runs ``preview``::

    python -m analyzer --config <data.yaml>
"""

from __future__ import annotations

import sys

from .compare import main as compare_main
from .loader import main as preview_main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        compare_main(argv[1:])
    elif argv and argv[0] == "preview":
        preview_main(argv[1:])
    else:
        preview_main(argv)  # backward compat: bare `--config ...` previews


if __name__ == "__main__":
    main()