"""Ensure the paper directory is importable so `Generator` and `data` resolve
regardless of how pytest is invoked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))