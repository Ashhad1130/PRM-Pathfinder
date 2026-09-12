"""Make `src/` importable when the scripts are run without `pip install -e .`.

Keeps the quickstart honest: `python scripts/run_eval.py ...` works straight from a fresh
clone, before anyone has installed the package.

All paths in the config files (``data/pool``, ``artifacts/index``, ``runs/``) are relative
to the project root, so the scripts chdir there. A config then means the same thing no
matter which directory you invoke the script from.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(PROJECT_ROOT)

# Must run before anything imports sklearn/transformers. No-op on a healthy machine.
from rapfprm.compat import neutralise_broken_pyarrow  # noqa: E402

neutralise_broken_pyarrow()


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
