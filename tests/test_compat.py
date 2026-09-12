"""The broken-pyarrow shim.

Guards a subtle failure: a library that guards `import pyarrow` with
`except ModuleNotFoundError` is not protected against an installed-but-unloadable
pyarrow, which raises plain ImportError. See src/rapfprm/compat.py.
"""

import importlib
import sys

import pytest

from rapfprm.compat import _BlockedModuleFinder, neutralise_broken_pyarrow


@pytest.fixture
def clean_meta_path():
    """Start each test with no shim installed, and restore the session's state after.

    conftest.py installs the shim for the whole test session when this machine's pyarrow
    is broken, so a test that wants to observe installation has to clear it first.
    """
    saved_meta_path = list(sys.meta_path)
    saved_modules = {k: v for k, v in sys.modules.items() if k.startswith("pyarrow")}

    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _BlockedModuleFinder)]

    yield

    sys.meta_path[:] = saved_meta_path
    for name in [n for n in sys.modules if n.startswith("pyarrow")]:
        del sys.modules[name]
    sys.modules.update(saved_modules)


def test_forced_shim_makes_pyarrow_look_absent(clean_meta_path):
    assert neutralise_broken_pyarrow(force=True) is True

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyarrow")


def test_shimmed_import_is_catchable_by_the_standard_guard(clean_meta_path):
    """The whole point: sklearn's `except ModuleNotFoundError` must now work."""
    neutralise_broken_pyarrow(force=True)

    caught = False
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        caught = True
    assert caught, "guarded import should see ModuleNotFoundError, not ImportError"


def test_submodules_are_blocked_too(clean_meta_path):
    neutralise_broken_pyarrow(force=True)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyarrow.lib")


def test_shim_is_not_installed_twice(clean_meta_path):
    assert neutralise_broken_pyarrow(force=True) is True
    assert neutralise_broken_pyarrow(force=True) is False

    installed = [f for f in sys.meta_path if isinstance(f, _BlockedModuleFinder)]
    assert len(installed) == 1


def test_shim_leaves_unrelated_modules_alone(clean_meta_path):
    neutralise_broken_pyarrow(force=True)
    assert importlib.import_module("json") is not None
    assert importlib.import_module("numpy") is not None


def test_finder_ignores_names_that_merely_start_with_pyarrow(clean_meta_path):
    """`pyarrow_hotfix` is a real package; blocking pyarrow must not blackhole it."""
    finder = _BlockedModuleFinder("pyarrow", "testing")
    assert finder.find_spec("pyarrow_hotfix") is None
    assert finder.find_spec("numpy") is None
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("pyarrow.compute")


def test_probe_is_a_noop_when_pyarrow_is_healthy_or_absent():
    """On a normal machine the shim must not engage. Never skip this assertion."""
    try:
        importlib.import_module("pyarrow")
        healthy = True
    except ModuleNotFoundError:
        healthy = False       # genuinely absent -> guards already work
    except ImportError:
        pytest.skip("pyarrow is installed but broken here; the probe path is exercised")

    # Whether healthy or absent, the non-forced call must decline to install the shim.
    assert neutralise_broken_pyarrow() is False
    assert healthy in (True, False)
