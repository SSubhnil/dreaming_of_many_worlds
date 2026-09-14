"""Proof that the cRSSM stack (this repo's dreamerv3 / embodied / contextual_mbrl) is what got loaded.

Called before any env or agent is built and recorded in every output. Raises with the resolved path
and the expected repo if any of the three came from elsewhere (e.g. DALI's editable install).
"""

from __future__ import annotations

import os

from drift_eval import REPO_DIR, module_location


def verify_stack():
    import contextual_mbrl
    import dreamerv3
    import dreamerv3.agent
    import embodied

    found = {"dreamerv3": module_location(dreamerv3), "embodied": module_location(embodied),
             "contextual_mbrl": module_location(contextual_mbrl)}
    bad = {k: f"{v} (real path {os.path.realpath(v)})" for k, v in found.items()
           if not v.startswith(REPO_DIR + os.sep)}
    if bad:
        raise RuntimeError(f"[drift_eval] cRSSM stack resolved outside this repo {REPO_DIR}: {bad}. "
                           f"A foreign copy (the dali venv's editable DALI install) won the import.")
    from drift_eval.cwm import sha256_file
    agent_py = os.path.abspath(dreamerv3.agent.__file__)
    return {"dreamerv3_file": found["dreamerv3"], "embodied_file": found["embodied"],
            "contextual_mbrl_file": found["contextual_mbrl"], "agent_py": agent_py,
            "agent_py_sha256": sha256_file(agent_py)}
