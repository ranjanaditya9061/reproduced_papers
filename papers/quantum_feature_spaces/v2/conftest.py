"""Keep this directory ahead of the legacy top-level package on sys.path.

The parent directory's own conftest.py (``quantum_feature_spaces/conftest.py``, pre-dating this
rewrite) inserts itself at position 0 so its legacy ``model``/``learner``/etc. packages resolve --
which shadows this directory's same-named packages when pytest walks up and loads both. Loading
after the parent (child conftests run later) and inserting at 0 again puts this directory back in
front, so ``import model`` etc. resolve to ``v2/model``, not the legacy top-level one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
