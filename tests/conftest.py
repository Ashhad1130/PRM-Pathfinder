import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Configs use paths relative to the project root, so tests must run from there.
os.chdir(PROJECT_ROOT)

# Keep a broken-but-installed pyarrow from taking sklearn/transformers down with it.
# No-op when pyarrow is healthy or genuinely absent. See src/rapfprm/compat.py.
from rapfprm.compat import neutralise_broken_pyarrow  # noqa: E402

neutralise_broken_pyarrow()
