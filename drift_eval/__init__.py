"""Gradual-shift (§8b) evaluation port for cRSSM checkpoints.

The protocol, the schedule generator and the factor semantics belong to CausalWorldModel (CWM).
This package only moves CWM's schedules into the cRSSM env chain and writes CWM's output schema.
See drift_eval/README.md.

Import-time path bootstrap. The shared /home/jovyan/envs/dali venv carries setuptools editable finders
(__editable__.dreamerv3-1.5.0.pth, __editable__.contextual_mbrl-0.0.0.pth) that map `dreamerv3` and
`contextual_mbrl` to the DALI repo. Whenever THIS repo's copies are not ahead of them on sys.path
(no PYTHONPATH, any cwd), `import dreamerv3` silently loads DALI's agent. So this package puts this
repo's `dreamerv3_compat` and root first on sys.path before anything imports them, and refuses to
continue if a foreign copy was already imported. `drift_eval.stack.verify_stack()` re-checks after the
real imports.
"""

import os as _os
import sys as _sys

REPO_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
COMPAT_DIR = _os.path.join(REPO_DIR, "dreamerv3_compat")
STACK_MODULES = ("dreamerv3", "embodied", "contextual_mbrl")


def module_location(mod):
    f = getattr(mod, "__file__", None)
    if f:
        return _os.path.abspath(f)
    paths = list(getattr(mod, "__path__", []) or [])
    return _os.path.abspath(paths[0]) if paths else ""


def _to_front(path):
    while path in _sys.path:
        _sys.path.remove(path)
    _sys.path.insert(0, path)


_to_front(REPO_DIR)
_to_front(COMPAT_DIR)

for _name in STACK_MODULES:
    _m = _sys.modules.get(_name)
    if _m is not None and not module_location(_m).startswith(REPO_DIR + _os.sep):
        raise ImportError(
            f"[drift_eval] {_name} was already imported from {module_location(_m)} (real path "
            f"{_os.path.realpath(module_location(_m))}), not from this repo {REPO_DIR}. Import "
            f"drift_eval before dreamerv3/embodied/contextual_mbrl.")
