"""Public module names, the experimental package, and the 0.6 move shims.

The documentation imports ``tinytt.backend``, ``tinytt.functional`` and
``tinytt.linesearch``; before 0.6 it imported the underscore modules, which a
user reads as "not for me".  The public names are aliases of the same module
objects, and that identity is what these tests pin: a copy would split state
such as the default device between the two.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

import tinytt as tt


@pytest.mark.parametrize(
    "public,private",
    [("backend", "_backend"), ("functional", "_functional"), ("linesearch", "_linesearch")],
)
def test_the_public_name_is_the_private_module_itself(public, private):
    module = importlib.import_module(f"tinytt.{public}")
    assert module is importlib.import_module(f"tinytt.{private}")
    assert getattr(tt, public) is module
    assert sys.modules[f"tinytt.{public}"] is module


def test_documented_solver_names_exist():
    """The README listed tt.solvers.gmres_restart and BiCGSTAB_reset before
    either was importable from there."""
    from tinytt._iterative_solvers import BiCGSTAB_reset, cg, gmres, gmres_restart

    assert tt.solvers.gmres_restart is gmres_restart
    assert tt.solvers.BiCGSTAB_reset is BiCGSTAB_reset
    assert tt.solvers.gmres is gmres
    assert tt.solvers.cg is cg
    assert tt.round_tt is importlib.import_module("tinytt._decomposition").round_tt


def test_importing_tinytt_does_not_import_experimental():
    """The core package stays independent of the experimental one."""
    code = (
        "import sys, tinytt, tinytt.manifold, tinytt.bug\n"
        "assert not any(m.startswith('tinytt.experimental') for m in sys.modules), "
        "sorted(m for m in sys.modules if 'experimental' in m)\n"
    )
    import subprocess

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_experimental_exports():
    import tinytt.experimental as ex
    from tinytt.experimental.clora import CLoRAModel
    from tinytt.experimental.momentum import DFIMomentum, DFOMomentum, bug_with_momentum

    assert ex.CLoRAModel is CLoRAModel
    assert ex.DFIMomentum is DFIMomentum and ex.DFOMomentum is DFOMomentum
    assert ex.bug_with_momentum is bug_with_momentum


# ---------------------------------------------------------------------------
# the pre-0.6 locations: they work, they warn, and they name the new home
# ---------------------------------------------------------------------------

def _fresh(name):
    sys.modules.pop(name, None)
    with pytest.warns(DeprecationWarning, match="experimental"):
        return importlib.import_module(name)


def test_the_old_clora_module_warns_and_re_exports():
    from tinytt.experimental import clora

    old = _fresh("tinytt.clora")
    assert old.CLoRAModel is clora.CLoRAModel
    assert old._factorize_core is clora._factorize_core
    assert old._merge_factors is clora._merge_factors


def test_the_old_momentum_module_warns_and_re_exports():
    from tinytt.experimental import momentum

    old = _fresh("tinytt.manifold.momentum")
    assert old.DFIMomentum is momentum.DFIMomentum
    assert old.DFOMomentum is momentum.DFOMomentum


@pytest.mark.parametrize("name", ["DFIMomentum", "DFOMomentum"])
def test_the_old_manifold_attribute_warns_and_resolves(name):
    import tinytt.manifold as manifold
    from tinytt.experimental import momentum

    with pytest.warns(DeprecationWarning, match=f"experimental.{name}"):
        value = getattr(manifold, name)
    assert value is getattr(momentum, name)


def test_the_old_bug_with_momentum_warns_and_resolves():
    import tinytt.bug as bug
    from tinytt.experimental.momentum import bug_with_momentum

    with pytest.warns(DeprecationWarning, match="experimental.momentum"):
        assert bug.bug_with_momentum is bug_with_momentum


@pytest.mark.parametrize("module", ["tinytt.manifold", "tinytt.bug"])
def test_unknown_attributes_still_raise(module):
    mod = importlib.import_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="no_such_name"):
            _ = mod.no_such_name
