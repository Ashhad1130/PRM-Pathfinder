"""Environment compatibility shims.

Currently one: neutralising a pyarrow installation whose native library cannot load.

Why this is needed. Libraries that only *optionally* use pyarrow guard the import with
``except ModuleNotFoundError`` — scikit-learn does exactly this in ``utils/fixes.py``:

    try:
        import pyarrow
        ...
    except ModuleNotFoundError:
        pass

That guard assumes the only failure mode is "not installed". But a pyarrow that is
installed while its compiled extension is unloadable raises plain ``ImportError``
(``DLL load failed ...``), which sails straight through the guard. sklearn then fails to
import, and because ``transformers.generation.candidate_generator`` does
``from sklearn.metrics import roc_curve``, so does transformers — which means no model
loads at all. One blocked DLL takes down the entire stack.

Observed on Windows with an Application Control policy blocking the pyarrow DLL, and the
same shape appears with mismatched CRT/glibc builds and partially-installed wheels.

The fix is to make a broken pyarrow look *absent*, which is what those guards already
handle correctly. This is a no-op when pyarrow imports fine, so it is safe to call
unconditionally and safe to ship — on a healthy machine (a cluster, CI) nothing happens.

What it does NOT do: rescue anything that genuinely needs pyarrow. ``datasets`` really
does require it, so `scripts/build_pool.py` still cannot run on such a machine. Evaluation
data is unaffected — `data/processbench.py` reads ProcessBench's plain JSON directly.

Proper fixes, in preference order:
  1. Allow the pyarrow DLL in the machine's security policy.
  2. Reinstall pyarrow with a build that loads (`pip install --force-reinstall pyarrow`).
  3. Uninstall pyarrow, if nothing on the machine needs `datasets`.
"""

from __future__ import annotations

import importlib
import logging
import sys
from importlib.abc import MetaPathFinder

logger = logging.getLogger(__name__)


class _BlockedModuleFinder(MetaPathFinder):
    """Meta-path hook that reports a module as genuinely missing."""

    def __init__(self, blocked: str, reason: str) -> None:
        self.blocked = blocked
        self.reason = reason

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if fullname == self.blocked or fullname.startswith(self.blocked + "."):
            raise ModuleNotFoundError(
                f"No module named {fullname!r} ({self.reason})", name=fullname
            )
        return None


def neutralise_broken_pyarrow(force: bool = False) -> bool:
    """Present an unloadable pyarrow as absent. Returns True if the shim was installed.

    ``force`` skips the probe and installs the shim regardless — useful for testing the
    shim itself, never needed in normal use.
    """
    already_shimmed = any(
        isinstance(finder, _BlockedModuleFinder) and finder.blocked == "pyarrow"
        for finder in sys.meta_path
    )
    if already_shimmed:
        return False

    if not force:
        if "pyarrow" in sys.modules:
            return False  # already imported successfully
        try:
            importlib.import_module("pyarrow")
            return False  # healthy
        except ModuleNotFoundError:
            return False  # genuinely absent; the guards already work
        except ImportError as exc:
            reason = f"installed but unusable: {exc}"
        except Exception as exc:  # noqa: BLE001 - defensive: any failure means unusable
            reason = f"installed but unusable: {exc!r}"
    else:
        reason = "blocked for testing"

    # A failed import can leave a half-initialised entry behind; clear it first.
    for name in [n for n in sys.modules if n == "pyarrow" or n.startswith("pyarrow.")]:
        del sys.modules[name]

    sys.meta_path.insert(0, _BlockedModuleFinder("pyarrow", reason))
    logger.warning(
        "pyarrow is %s. Treating it as absent so scikit-learn and transformers still "
        "import; `datasets` will not work on this machine. See src/rapfprm/compat.py.",
        reason,
    )
    return True
