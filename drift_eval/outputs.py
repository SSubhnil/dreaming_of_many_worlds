"""Output files in CWM's runner schema (benchmark/run_eval_benchmark.py @45e8d5d).

* raw JSON  <results>/raw/<row_id>__<condition_id>.json — exactly the keys of
  run_eval_benchmark._write_raw_json (:111-133): row_id, condition_id, returns,
  reward_mse_per_episode, n_episodes, metadata{seed, env, git_sha, eval_timestamp_utc}.
  Port provenance (action mode, generator sha256, schedule hash, checkpoint step/sha, ...) is added
  INSIDE `metadata` only, so the top level is identical.
  NB: CWM's older embodied runner (benchmark/eval_shared.write_raw_json) writes a DIFFERENT schema
  (`raw_returns`, flat metadata); this port follows the PyTorch runner, not that one.
* CSV       <results>/main_results.csv — run_eval_benchmark.CSV_COLUMNS (:44-49).
* D1 trace  <results>/traces/<row_id>__<condition_id>__trace.npz + .meta.json — the cell
  run_eval_benchmark writes (:435-452) through switch_recovery.write_trace_cell (imported from the
  CWM root, not copied), plus additive per-step arrays.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import numpy as np

CSV_COLUMNS = [
    "row_id", "condition_id", "method", "domain", "env", "experiment", "seed",
    "axis", "n_episodes", "mean_return", "std_return",
    "reward_mse_mean", "reward_mse_std",
    "raw_returns_path", "training_wandb_run_id", "eval_wandb_run_id",
    "eval_timestamp_utc", "git_sha", "notes",
]
RAW_JSON_KEYS = ("row_id", "condition_id", "returns", "reward_mse_per_episode", "n_episodes",
                 "episode_keys", "metadata")
# Keys CWM's runner writes at befabde beyond 45e8d5d (run_eval_benchmark.py:_write_raw_json :112-138):
# a 45e8d5d-era CWM raw JSON lacks exactly these.
RAW_JSON_KEYS_ADDED_BEFABDE = ("episode_keys",)
RAW_JSON_METADATA_KEYS = ("seed", "env", "git_sha", "paired_streams", "paired_streams_reason",
                          "eval_timestamp_utc")
# Advisor ruling 2026-09-14: port outputs keep paired_streams null, never true/false, with this reason.
PAIRED_STREAMS_REASON = "not_applicable: port has no first-action sampler"
# Written by EVERY output of this port (lead requirement: action mode + generator sha256 in every file).
PORT_METADATA_KEYS = ("action_mode", "generator_sha256", "generator_path", "schedule_hash",
                      "method", "bench", "ckpt_step", "ckpt_sha256",
                      "cwm_root", "cwm_commit", "cwm_dirty_count", "cwm_config_sha256",
                      "dreamerv3_file", "agent_py_sha256")
# The D1 cell CWM's runner writes (run_eval_benchmark.py:435-450) and its sidecar meta keys (:445-450).
TRACE_CELL_SPEC = {
    "reward": (np.float32, 2),
    "reward_mode_id": (np.int16, 2),
    "factor_values": (np.float32, 3),
    "factor_active": (np.bool_, 1),
    "lambda_sampled": (np.float32, 1),
    "schedule_id": (np.int32, 1),
    # befabde run_eval_benchmark.py:1218-1223: row i IS returns[i]; the join key and its return.
    "episode_worker": (np.int32, 1),
    "episode_ordinal": (np.int32, 1),
    "episode_return": (np.float32, 1),
}
TRACE_META_KEYS = ("method", "env", "experiment", "seed", "axis", "condition_id", "row_id",
                   "lambda", "schedule_hash", "git_sha", "paired_streams", "paired_streams_reason",
                   "row_join",
                   "unjoined_worker_traces")


def repo_git_sha(repo_dir):
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_raw_json(raw_dir, row_id, condition_id, returns, mse, seed, env, git_sha, extra_metadata,
                   paired_streams=None, episode_keys=None):
    missing = [k for k in PORT_METADATA_KEYS if k not in extra_metadata]
    if missing:
        raise ValueError(f"[drift_eval] raw JSON metadata lacks {missing}")
    os.makedirs(raw_dir, exist_ok=True)
    filename = f"{row_id}__{condition_id}.json"
    returns = np.asarray(returns, dtype=np.float64)
    metadata = {"seed": seed, "env": env, "git_sha": git_sha, "paired_streams": paired_streams,
                "paired_streams_reason": PAIRED_STREAMS_REASON,
                "eval_timestamp_utc": datetime.now(timezone.utc).isoformat()}
    for k, v in extra_metadata.items():
        if k in metadata:
            raise ValueError(f"[drift_eval] extra metadata would overwrite {k!r}")
        metadata[k] = v
    payload = {
        "row_id": row_id,
        "condition_id": condition_id,
        "returns": returns.tolist(),
        "reward_mse_per_episode": (np.asarray(mse, dtype=np.float64).tolist()
                                   if mse is not None else None),
        "n_episodes": int(len(returns)),
        # (worker, episode ordinal) of returns[i]; the trace join key (CWM run_eval_benchmark.py:125 @befabde).
        "episode_keys": ([[int(w), int(j)] for w, j in episode_keys]
                         if episode_keys is not None else None),
        "metadata": metadata,
    }
    with open(os.path.join(raw_dir, filename), "w") as f:
        json.dump(payload, f)
    return os.path.join("raw", filename)


def validate_raw_json(payload, reference=None):
    """Problems (empty = valid) against CWM's raw-JSON schema, and against a real CWM file."""
    probs = []
    if tuple(sorted(payload)) != tuple(sorted(RAW_JSON_KEYS)):
        probs.append(f"top-level keys {sorted(payload)} != {sorted(RAW_JSON_KEYS)}")
    if not isinstance(payload.get("returns"), list) or not all(
            isinstance(x, float) for x in payload.get("returns", [None])):
        probs.append("returns is not a list of floats")
    if payload.get("n_episodes") != len(payload.get("returns") or []):
        probs.append("n_episodes != len(returns)")
    mse = payload.get("reward_mse_per_episode")
    if mse is not None and (not isinstance(mse, list) or len(mse) != len(payload["returns"])):
        probs.append("reward_mse_per_episode is neither null nor a per-episode list")
    md = payload.get("metadata") or {}
    for k in RAW_JSON_METADATA_KEYS + PORT_METADATA_KEYS:
        if k not in md:
            probs.append(f"metadata lacks {k!r}")
    keys = payload.get("episode_keys")
    if not (isinstance(keys, list) and len(keys) == len(payload.get("returns") or [])
            and all(isinstance(k, list) and len(k) == 2 and all(isinstance(x, int) for x in k)
                    for k in keys)):
        probs.append("episode_keys is not a per-return list of [worker, ordinal] ints")
    if reference is not None:
        extra = set(payload) - set(reference)
        if not set(reference) <= set(payload) or not extra <= set(RAW_JSON_KEYS_ADDED_BEFABDE):
            probs.append(f"top-level keys {sorted(payload)} vs the CWM reference file "
                         f"{sorted(reference)} (+ befabde additions {RAW_JSON_KEYS_ADDED_BEFABDE})")
        for k in reference.get("metadata", {}):
            if k not in md:
                probs.append(f"metadata lacks reference key {k!r}")
        for k in ("row_id", "condition_id"):
            if type(reference[k]) is not type(payload[k]):
                probs.append(f"{k} type differs from reference")
        if type(reference["n_episodes"]) is not type(payload["n_episodes"]):
            probs.append("n_episodes type differs from reference")
    return probs


def append_csv_row(csv_path, row):
    needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if needs_header:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def done_pairs(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {(r["row_id"], r["condition_id"]) for r in csv.DictReader(f)}


def _import_switch_recovery():
    from drift_eval.cwm import cwm_root
    import sys
    bdir = os.path.join(cwm_root(), "benchmark")
    if bdir not in sys.path:
        sys.path.append(bdir)
    import switch_recovery
    if not os.path.abspath(switch_recovery.__file__).startswith(os.path.abspath(cwm_root())):
        raise RuntimeError(f"[drift_eval] switch_recovery resolved outside the CWM root: "
                           f"{switch_recovery.__file__}")
    return switch_recovery


def write_trace_cell(path, cell, meta):
    missing = [k for k in TRACE_META_KEYS + PORT_METADATA_KEYS if k not in meta]
    if missing:
        raise ValueError(f"[drift_eval] trace meta lacks {missing}")
    return _import_switch_recovery().write_trace_cell(path, cell, meta)


def join_traces_to_returns(traces, return_keys, suffix, cid="?"):
    """Order worker traces to match raw returns BY (worker, episode ordinal), never by position.

    Exact mirror of CWM benchmark/run_eval_benchmark.py:659 `_join_traces_to_returns` @befabde (that
    module imports torch, which this interpreter lacks); a test asserts the AST is identical to the
    function at that commit and runs the original on a port output. Port worker tags are `w<i>`, so
    the port calls it with suffix "". Returns (joined, n_unjoined).
    """
    if return_keys is None:
        raise SystemExit(f"[D1] condition {cid!r}: eval_episodes returned no per-episode keys; "
                         f"traces cannot be joined to returns.")
    keys = [(int(w), int(j)) for w, j in return_keys]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"[D1] condition {cid!r}: duplicate episode keys in the returns {keys}.")
    pattern = re.compile(rf"^w(\d+){re.escape(suffix)}$")
    by_key = {}
    for t in traces:
        m = pattern.match(str(t.get("worker_tag", "")))
        if m is None or "episode_ordinal" not in t:
            raise SystemExit(f"[D1] condition {cid!r}: trace without a (worker, ordinal) key "
                             f"(worker_tag={t.get('worker_tag')!r}).")
        key = (int(m.group(1)), int(t["episode_ordinal"]))
        if key in by_key:
            raise SystemExit(f"[D1] condition {cid!r}: two traces for episode {key}.")
        by_key[key] = t
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"[D1] condition {cid!r}: no trace for returned episodes {missing[:8]}.")
    return [by_key[k] for k in keys], len(by_key) - len(keys)


def load_trace_cell(path):
    return _import_switch_recovery().load_trace_cell(path)


def validate_trace_cell(cell):
    probs = []
    for k, (dtype, ndim) in TRACE_CELL_SPEC.items():
        if k not in cell:
            probs.append(f"trace cell lacks {k!r}")
            continue
        a = np.asarray(cell[k])
        if a.dtype != dtype or a.ndim != ndim:
            probs.append(f"{k}: dtype/ndim {a.dtype}/{a.ndim} != {np.dtype(dtype)}/{ndim}")
    if not probs:
        N, T = cell["reward"].shape
        if cell["reward_mode_id"].shape != (N, T) or cell["factor_values"].shape != (N, T, 4) \
                or cell["factor_active"].shape != (4,) or cell["lambda_sampled"].shape != (N,) \
                or cell["schedule_id"].shape != (N,) or cell["episode_worker"].shape != (N,) \
                or cell["episode_ordinal"].shape != (N,) or cell["episode_return"].shape != (N,):
            probs.append("trace cell shapes are not [N,T], [N,T,4], [4], [N] x5")
    meta = cell.get("_meta", {})
    for k in TRACE_META_KEYS + PORT_METADATA_KEYS:
        if k not in meta:
            probs.append(f"trace meta lacks {k!r}")
    return probs
