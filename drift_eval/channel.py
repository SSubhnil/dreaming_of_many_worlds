"""Schedule payloads into the process that steps the env; fail closed otherwise.

Pattern copied from CWM envs/schedule_channel.py @45e8d5d (commit 1a30c23), which repaired the DMC
spawn defect: a schedule installed as a module global in the PARENT never reaches a spawned
worker, which re-imports the wrapper module with the global None and silently runs every
"scheduled" episode as an unscheduled draw.

Here the payload travels INSIDE the env constructor's arguments (`envbuild.DriftEnvSpec`), and
`install_in_process` runs at the top of that constructor, i.e. in whichever process builds and
steps the env (the main process for the fork's `envs.parallel: none`; the child for
`embodied.Parallel(..., 'process')`, which spawns: embodied/core/worker.py ProcessPipeWorker).

The CARL wrappers have no `switch_mode` field: they run the schedule if a global payload is set
and Bernoulli otherwise. So the port carries the mode explicitly and enforces both directions:
  scheduled + no payload  -> RuntimeError (the silent-unscheduled failure)
  bernoulli + a payload   -> ValueError
  bernoulli               -> any leftover global from an earlier condition in this process is cleared
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

SWITCH_MODES = ("bernoulli", "scheduled")


def _hash(p):
    if p is None:
        return None
    return p.get("carl_payload_hash") or p.get("schedule_hash") or "<unhashed>"


def process_state():
    from drift_eval.cwm import carl_modules
    ci, cr = carl_modules()
    return {
        "pid": os.getpid(),
        "factor_schedule_hash": _hash(ci._ACTIVE_FACTOR_SCHEDULE),
        "reward_schedule_hash": _hash(cr._ACTIVE_REWARD_SCHEDULE),
    }


def clear_process():
    from drift_eval.cwm import carl_modules
    ci, cr = carl_modules()
    ci.clear_instrumentation()
    cr.clear_instrumentation()


def install_in_process(switch_mode, factor_schedule=None, reward_schedule=None, mixed=False):
    """Install (or clear) the CARL wrapper payloads in THIS process and verify what took effect."""
    from drift_eval.cwm import carl_modules
    if switch_mode not in SWITCH_MODES:
        raise ValueError(f"switch_mode must be one of {SWITCH_MODES}, got {switch_mode!r}")
    ci, cr = carl_modules()
    if switch_mode == "scheduled":
        if factor_schedule is None:
            raise RuntimeError(
                f"[drift_eval] switch_mode='scheduled' but no factor schedule payload reached this "
                f"process (pid {os.getpid()}). Pass it in the env constructor; a module global set "
                f"in another process does not cross a spawn.")
        if mixed and reward_schedule is None:
            raise RuntimeError(
                f"[drift_eval] mixed bench in switch_mode='scheduled' but no reward label payload "
                f"reached this process (pid {os.getpid()}).")
        if not mixed and reward_schedule is not None:
            raise ValueError("[drift_eval] reward schedule given for a physics-only bench")
        ci.set_factor_schedule(factor_schedule)
        cr.set_reward_schedule(reward_schedule if mixed else None)
    else:
        if factor_schedule is not None or reward_schedule is not None:
            raise ValueError("[drift_eval] switch_mode='bernoulli' was given a schedule payload")
        ci.clear_instrumentation()
        cr.clear_instrumentation()

    state = process_state()
    want_f = _hash(factor_schedule) if switch_mode == "scheduled" else None
    want_r = _hash(reward_schedule) if (switch_mode == "scheduled" and mixed) else None
    if state["factor_schedule_hash"] != want_f or state["reward_schedule_hash"] != want_r:
        raise RuntimeError(
            f"[drift_eval] payload install did not take effect in pid {os.getpid()}: wanted "
            f"factor={want_f} reward={want_r}, state {state}")
    return state


def env_seed_base(trained_seed):
    """CWM's eval env seed convention (eval.py:45-48 @45e8d5d): BasicSettings.Seed * 1000 + 7."""
    return int(trained_seed) * 1000 + 7


def crn_episode_plan(base_seed, n_episodes, amount):
    """Episode index k -> (env/worker, env seed, episode within env); the layout of CWM's
    envs/schedule_channel.crn_episode_plan: env i gets seed base+i, finishes its j-th episode as
    index k = j*amount + i, and only the first n_episodes are kept (CWM eval.py stops at
    Evaluate.EpisodeNum, so n need not be a multiple of the env count). Two conditions with the same
    base seed, amount and episode count pair episode-by-episode on k."""
    return [{"episode_index": k, "worker": k % amount, "worker_seed": int(base_seed) + k % amount,
             "env_seed": int(base_seed) + k % amount, "episode_in_worker": k // amount}
            for k in range(n_episodes)]


def drift_label_plan(labels, base_seed, n_episodes, amount):
    """Reward label per retained episode, as CWM's run_eval_benchmark._drift_label_plan (:476 @befabde):
    env i's j-th episode reads schedule episode (base_seed + i + j) % len(labels)."""
    n = len(labels)
    return [labels[(p["worker_seed"] + p["episode_in_worker"]) % n]
            for p in crn_episode_plan(base_seed, n_episodes, amount)]


# ── per-episode trace channel (CWM envs/schedule_channel.py DirTraceSink / collect_traces) ────────

_TRACE_FILE_RE = re.compile(r"^trace_(?P<tag>[A-Za-z0-9]+)_(?P<ordinal>\d+)\.npz$")


class DirTraceSink:
    """One npz per finished episode, so traces cross a process boundary."""

    def __init__(self, out_dir, worker_tag="w0"):
        self.out_dir = str(out_dir)
        self.worker_tag = str(worker_tag)
        self.count = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def append(self, trace):
        path = os.path.join(self.out_dir, f"trace_{self.worker_tag}_{self.count:05d}.npz")
        payload = {k: np.asarray(v) for k, v in trace.items()}
        payload["worker_tag"] = np.asarray(self.worker_tag)
        payload["episode_ordinal"] = np.asarray(self.count)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp, path)
        self.count += 1

    def __len__(self):
        return self.count


def collect_traces(trace_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "trace_*.npz"))):
        if _TRACE_FILE_RE.match(os.path.basename(path)) is None:
            continue
        with np.load(path, allow_pickle=False) as z:
            rec = {k: z[k] for k in z.files}
        rec["worker_tag"] = str(rec["worker_tag"])
        rec["episode_ordinal"] = int(rec["episode_ordinal"])
        out.append(rec)
    out.sort(key=lambda r: (r["worker_tag"], r["episode_ordinal"]))
    return out
