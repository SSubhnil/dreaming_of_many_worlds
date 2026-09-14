"""Evaluate one cRSSM checkpoint on the §8b drift conditions. See drift_eval/README.md.

    python -m drift_eval.run --bench walker_3f_phys --seed 1 \
        --schedules-dir <dir with bench_spec.json, ir_s.json, ...> --results-root <dir>

One process, one checkpoint; conditions run in sequence with the agent loaded once.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
EXPECTED_T = 500

POLICY_MODES = {"native": "eval", "greedy": "eval_greedy"}
ACTION_MODE_DETAIL = {
    "native": "sampled: agent.policy(mode='eval') -> actor distribution .sample() "
              "(dreamerv3_compat/dreamerv3/agent.py eval branch; the fork's own eval mode)",
    "greedy": "deterministic: agent.policy(mode='eval_greedy') -> actor distribution .mode() "
              "(tanh(mean) for the continuous 'normal' actor); added by this port",
}


def load_port_yaml():
    with open(HERE / "drift_eval.yaml") as f:
        return yaml.safe_load(f)


def _sha(path):
    from drift_eval.cwm import sha256_file
    return sha256_file(path)


def load_bench_spec(schedules_dir, bench_name, gen, gen_sha, cfg, values_table):
    """Read the generator's bench_spec.json and refuse anything that does not match this bench,
    this generator file, CWM's value table, or the checkpoint's training config."""
    from drift_eval import benches as B
    b = B.bench(bench_name)
    with open(os.path.join(schedules_dir, "bench_spec.json")) as f:
        spec = json.load(f)
    probs = []
    if spec.get("bench") != bench_name:
        probs.append(f"bench_spec is for {spec.get('bench')!r}")
    if spec.get("generator_sha256") != gen_sha:
        probs.append(f"bench_spec generator_sha256 {spec.get('generator_sha256')} != loaded "
                     f"generator {gen_sha}")
    if spec.get("generator_version") != gen.GENERATOR_VERSION:
        probs.append("bench_spec generator_version differs from the loaded generator")
    if spec.get("domain") != b["domain"]:
        probs.append(f"domain {spec.get('domain')} != {b['domain']}")
    fids = sorted(int(k) for k in spec.get("factors", {}))
    if fids != sorted(b["factor_ids"]):
        probs.append(f"factor ids {fids} != {sorted(b['factor_ids'])}")
    for k, vals in spec.get("factors", {}).items():
        key = B.value_key(b["domain"], int(k))
        if spec["value_keys"][k] != key:
            probs.append(f"factor {k} value key {spec['value_keys'][k]} != {key}")
        if [float(v) for v in vals] != [float(v) for v in values_table[key]["train_values"]]:
            probs.append(f"factor {k} train values differ from CWM latent_factor_values.yaml")
    carl_lr = [float(x) for x in cfg.env.carl.regime_b_lambda_range]
    if [float(x) for x in spec.get("lambda_range", [])] != carl_lr:
        probs.append(f"generator lambda_range {spec.get('lambda_range')} != checkpoint config "
                     f"regime_b_lambda_range {carl_lr}")
    if b["mixed"]:
        carl_modes = [str(m) for m in cfg.env.carl.regime_b_reward_modes]
        if spec.get("reward_modes") != carl_modes:
            probs.append(f"generator reward modes {spec.get('reward_modes')} != checkpoint config "
                         f"regime_b_reward_modes {carl_modes}")
    elif spec.get("reward_modes") is not None:
        probs.append("physics-only bench has reward modes in bench_spec")
    if probs:
        raise RuntimeError(f"[drift_eval] bench_spec.json in {schedules_dir} refused: {probs}")
    return spec


def load_condition_payloads(schedules_dir, cid, bench_name, bench_spec, gen, conditions, wind_clip):
    """(CARL-unit physics payload | None, reward label payload | None, provenance dict)."""
    from drift_eval import benches as B
    b = B.bench(bench_name)
    name = conditions[cid]["schedule"]
    if name is None:
        return None, None, {"condition": "training_protocol_bernoulli", "schedule_hash": None}
    path = os.path.join(schedules_dir, f"{name}.json")
    with open(path) as f:
        payload = json.load(f)
    probs = []
    if payload.get("generator_version") != gen.GENERATOR_VERSION:
        probs.append("generator_version differs from the loaded generator")
    want_cond = "tau" if name.startswith("tau_") else name
    if payload.get("condition") != want_cond:
        probs.append(f"payload condition {payload.get('condition')!r} != {want_cond!r}")
    want_keys = sorted(f"physics_f{f}" for f in b["factor_ids"])
    if sorted(payload.get("processes", {})) != want_keys:
        probs.append(f"processes {sorted(payload.get('processes', {}))} != {want_keys}")
    for k, vals in payload.get("processes", {}).items():
        if [float(v) for v in vals] != [float(v) for v in bench_spec["factors"][k[len('physics_f'):]]]:
            probs.append(f"{k} value set differs from bench_spec")
    if int(payload.get("T", -1)) != EXPECTED_T:
        probs.append(f"T={payload.get('T')} != {EXPECTED_T}")
    for ep in payload.get("episodes", []):
        if sorted(ep) != want_keys:
            probs.append("an episode lacks a physics process")
            break
    reward, reward_info = None, {}
    if b["mixed"]:
        rpath = os.path.join(schedules_dir, "reward_fixed.json")
        with open(rpath) as f:
            reward = json.load(f)
        if reward.get("generator_version") != gen.GENERATOR_VERSION:
            probs.append("reward_fixed generator_version differs")
        if reward.get("processes", {}).get("reward") != bench_spec["reward_modes"]:
            probs.append("reward_fixed modes differ from bench_spec")
        if any(len(ep["reward"]) != 1 or int(ep["reward"][0][0]) != 0 for ep in reward["episodes"]):
            probs.append("reward_fixed has an entry other than one step-0 label per episode")
        reward_info = {"reward_schedule_hash": reward.get("schedule_hash"),
                       "reward_schedule_file_sha256": _sha(rpath)}
    if probs:
        raise RuntimeError(f"[drift_eval] schedule {path} refused: {probs}")
    carl = B.schedule_to_carl_units(payload, wind_clip)
    n_clipped = {k: int(sum(B.cwm_clip_value(int(k[len("physics_f"):]), v, wind_clip) != float(v)
                            for _, v in events))
                 for k, events in payload["episodes"][0].items()}
    info = {"condition": name, "schedule_hash": payload.get("schedule_hash"),
            "cwm_clip_changed_values_episode0": n_clipped,
            "schedule_file": path, "schedule_file_sha256": _sha(path),
            "carl_payload_hash": carl["carl_payload_hash"], "schedule_stats": payload.get("stats"),
            **reward_info}
    return carl, reward, info


def load_agent(cfg, env, ckpt_file):
    """As CWM benchmark/eval_embodied.py:_load_embodied_agent_and_env (step first, then agent)."""
    from drift_eval.stack import verify_stack
    verify_stack()
    import dreamerv3
    from dreamerv3 import embodied
    path = embodied.Path(ckpt_file)
    sc = embodied.Checkpoint()
    sc.step = embodied.Counter()
    sc.load(path, keys=["step"])
    step = sc._values["step"]
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, cfg)
    ac = embodied.Checkpoint()
    ac.agent = agent
    ac.load(path, keys=["agent"])
    return agent, int(step)


def run_episodes(env, policy, n_episodes, amount):
    """Driver rollout; per env, the list of finished episodes (row 0 of each is the reset obs).

    All envs finish in lockstep (fixed-length DMC episodes), so ceil(n/amount) full rounds run; the
    first n in plan order are kept, as CWM's eval loop keeps the first Evaluate.EpisodeNum."""
    from dreamerv3 import embodied
    rounds = -(-n_episodes // amount)
    per_worker = [[] for _ in range(amount)]

    def on_episode(ep, worker):
        rew = np.asarray(ep["reward"], dtype=np.float32)
        rec = {"return": float(np.asarray(ep["reward"], dtype=np.float64).sum()),
               "reward": rew[1:],
               "action": np.asarray(ep["action"], dtype=np.float32)[:-1]}
        if "reward_hat" in ep:
            rec["reward_hat"] = np.asarray(ep["reward_hat"], dtype=np.float32)[1:]
        per_worker[worker].append(rec)

    driver = embodied.Driver(env)
    driver.on_episode(on_episode)
    driver(policy, episodes=rounds * amount)
    if any(len(w) != rounds for w in per_worker):
        raise RuntimeError(f"[drift_eval] expected {rounds} episodes per env, got "
                           f"{[len(w) for w in per_worker]}")
    return per_worker


def assemble(per_worker, traces, plan, cid="?"):
    """Join recorder traces to the retained returns BY (worker, episode ordinal) with CWM's join
    (outputs.join_traces_to_returns = run_eval_benchmark._join_traces_to_returns :659 @befabde); traces
    of episodes past the retained count are dropped and counted. Returns (rows, n_unjoined)."""
    from drift_eval import outputs
    keys = [(row["worker"], row["episode_in_worker"]) for row in plan]
    joined, unjoined = outputs.join_traces_to_returns(traces, keys, "", cid)
    rows = []
    for row, tr in zip(plan, joined):
        i, j = row["worker"], row["episode_in_worker"]
        ep = per_worker[i][j]
        if int(tr["episode_ordinal"]) != j or int(tr["env_seed"]) != row["env_seed"]:
            raise RuntimeError(f"[drift_eval] trace for env {i} episode {j} is misfiled")
        if not np.array_equal(ep["reward"], tr["reward"]):
            raise RuntimeError(f"[drift_eval] env {i} episode {j}: driver and recorder rewards "
                               f"differ; per-step records are misaligned")
        if not np.array_equal(tr["decision_step"], np.arange(1, len(tr["reward"]) + 1)):
            raise RuntimeError(f"[drift_eval] env {i} episode {j}: decision steps are not 1..T")
        rows.append((row, ep, tr))
    if len({len(tr["reward"]) for _, _, tr in rows}) != 1:
        raise RuntimeError("[drift_eval] episodes of one condition have unequal lengths; refusing "
                           "to crop")
    return rows, unjoined


def check_reward_labels(rows, reward_payload, mode_ids):
    """Every step of each episode runs the fixed label of schedule episode (env_seed + ordinal) % n,
    recorded as CWM's reward_mode_id."""
    n = len(reward_payload["episodes"])
    for row, _, tr in rows:
        j = int(tr["wrapper_episode_ordinal"])
        want = mode_ids[reward_payload["episodes"][(int(tr["env_seed"]) + j) % n]["reward"][0][1]]
        ids = np.concatenate([[int(tr["reset_reward_mode_id"])], tr["reward_mode_id"].astype(int)])
        if not np.all(ids == want):
            raise RuntimeError(f"[drift_eval] episode {row['episode_index']}: reward mode ids "
                               f"{sorted(set(ids.tolist()))} != fixed label {want}")


def build_cell(rows):
    N = len(rows)
    tr0 = rows[0][2]
    cell = {
        "reward": np.stack([tr["reward"] for _, _, tr in rows]).astype(np.float32),
        "reward_mode_id": np.stack([tr["reward_mode_id"] for _, _, tr in rows]).astype(np.int16),
        "factor_values": np.stack([tr["factor_values"] for _, _, tr in rows]).astype(np.float32),
        "factor_active": np.asarray(tr0["factor_active"], dtype=bool),
        "lambda_sampled": np.full(N, np.nan, dtype=np.float32),
        "schedule_id": np.arange(N, dtype=np.int32),
        # additive (not in CWM's cell)
        "decision_step": np.stack([tr["decision_step"] for _, _, tr in rows]).astype(np.int32),
        "reset_factor_values": np.stack([tr["reset_factor_values"] for _, _, tr in rows]).astype(np.float32),
        "reset_reward_mode_id": np.array([int(tr["reset_reward_mode_id"]) for _, _, tr in rows], np.int16),
        "actions": np.stack([ep["action"] for _, ep, _ in rows]).astype(np.float32),
        "returns": np.array([ep["return"] for _, ep, _ in rows], dtype=np.float64),
        "episode_index": np.array([r["episode_index"] for r, _, _ in rows], dtype=np.int32),
        # CWM names (run_eval_benchmark.py:1218-1223 @befabde): row i IS returns[i]; key and return.
        "episode_worker": np.array([r["worker"] for r, _, _ in rows], dtype=np.int32),
        "episode_ordinal": np.array([r["episode_in_worker"] for r, _, _ in rows], dtype=np.int32),
        "episode_return": np.array([ep["return"] for _, ep, _ in rows], dtype=np.float32),
        "env_seed": np.array([r["env_seed"] for r, _, _ in rows], dtype=np.int64),
    }
    if all("reward_hat" in ep for _, ep, _ in rows):
        cell["reward_hat"] = np.stack([ep["reward_hat"] for _, ep, _ in rows]).astype(np.float32)
    return cell


def run_condition(*, cfg, bench_name, cid, conditions, schedules_dir, gen, gen_sha, bench_spec,
                  wind_clip, trained_seed, n_episodes, amount, policy_factory, action_mode,
                  results_root, ckpt_info, cfg_source, git_sha, cwm_hashes, env_parallel="none",
                  seed_task_rng=True, render=True):
    from drift_eval import benches as B, channel, envbuild, outputs
    from drift_eval.cwm import cwm_provenance

    from drift_eval.stack import verify_stack
    b = B.bench(bench_name)
    cwm_prov = cwm_provenance()
    stack_prov = verify_stack()
    base = channel.env_seed_base(trained_seed)
    plan = channel.crn_episode_plan(base, n_episodes, amount)
    row_id = f"crssm_{bench_name}_s{trained_seed}"
    carl, reward, sched_info = load_condition_payloads(schedules_dir, cid, bench_name, bench_spec,
                                                       gen, conditions, wind_clip)
    mode = "bernoulli" if carl is None else "scheduled"
    env_name = B.env_name(bench_name)
    label_plan = None
    if mode == "scheduled" and b["mixed"]:
        labels = [str(ep["reward"][0][1]) for ep in reward["episodes"]]
        label_plan = channel.drift_label_plan(labels, base, n_episodes, amount)
        counts = {m: label_plan.count(m) for m in bench_spec["reward_modes"]}
        if len(set(counts.values())) != 1:
            raise RuntimeError(
                f"[drift_eval] condition {cid!r}: the {n_episodes} retained episodes (amount={amount}) "
                f"would run the objectives {counts}, not balanced across the train modes (CWM "
                f"run_eval_benchmark._resolve_schedule_modes refuses the same). Mixed smoke: "
                f"--n-episodes 10 --amount 5.")
    trace_root = os.path.join(results_root, "traces")
    wdir = os.path.join(trace_root, "_workers", f"{row_id}__{cid}")
    shutil.rmtree(wdir, ignore_errors=True)
    specs = [envbuild.DriftEnvSpec(
        bench=bench_name, switch_mode=mode, env_seed=base + i, wind_clip=wind_clip,
        factor_schedule_carl=carl, reward_schedule=reward, trace_dir=wdir, worker_tag=f"w{i}",
        seed_task_rng=seed_task_rng, expected_T=EXPECTED_T, render=render)
        for i in range(amount)]

    channel.clear_process()
    t0 = time.time()
    env = envbuild.make_drift_batch_env(cfg, specs, parallel=env_parallel)
    try:
        policy = policy_factory(env, base)
        per_worker = run_episodes(env, policy, n_episodes, amount)
    finally:
        try:
            env.close()
        except Exception:
            pass
        channel.clear_process()
    elapsed = time.time() - t0

    rows, unjoined = assemble(per_worker, channel.collect_traces(wdir), plan, cid)
    episode_keys = [(r["worker"], r["episode_in_worker"]) for r, _, _ in rows]
    # CWM records Evaluate.PairedStreams (it seeds the parent's action_space.sample stream). This
    # stack has no such stream, so the value is None, not a claim either way.
    paired_streams = None
    if mode == "scheduled" and b["mixed"]:
        check_reward_labels(rows, reward, B.cwm_reward_mode_ids(b["domain"]))
    rounds = len(per_worker[0])
    # CWM's trace meta: a mixed drift cell's schedule_hash is the REWARD label schedule and the physics
    # hash sits in physics_schedule_hash (run_eval_benchmark.py:1224-1237 @befabde).
    if label_plan is not None:
        schedule_hash = sched_info.get("reward_schedule_hash")
        extra_hash = {"physics_schedule_hash": sched_info.get("schedule_hash"),
                      "reward_label_plan": label_plan}
    else:
        schedule_hash = sched_info.get("schedule_hash")
        extra_hash = {}
    returns = [ep["return"] for _, ep, _ in rows]
    mse = None
    if all("reward_hat" in ep for _, ep, _ in rows):
        mse = [float(np.mean((ep["reward_hat"].astype(np.float64) - ep["reward"]) ** 2))
               for _, ep, _ in rows]

    meta = {
        "method": "crssm", "bench": bench_name, "bench_short": b["short"],
        "action_mode": action_mode, "action_mode_detail": ACTION_MODE_DETAIL[action_mode],
        "policy_mode": POLICY_MODES[action_mode],
        "generator_path": gen.__drift_eval_path__, "generator_sha256": gen_sha,
        "generator_version": gen.GENERATOR_VERSION,
        "schedule_hash": schedule_hash, **extra_hash,
        **{k: v for k, v in sched_info.items() if k not in ("schedule_hash", "schedule_stats")},
        "reward_mode_id_convention": ("CWM: index in sorted domain reward registry "
                                      "(dmc_reward_factors.py:359/:432)" if b["mixed"] else "-1 (no reward axis)"),
        "switch_mode": mode,
        "clip": ("scheduled: CWM clip (dmc_latent_factors.py:_apply_factor bounds and order) applied "
                 "in CWM units before conversion; CARL wrapper clip_range set to the same bounds. "
                 "bernoulli: CARL native clip" if mode == "scheduled" else "CARL native clip"),
        "wind_clip": wind_clip,
        "ckpt_path": ckpt_info["path"], "ckpt_sha256": ckpt_info["sha256"],
        "ckpt_step": ckpt_info["step"], "config_source": cfg_source,
        "trained_seed": int(trained_seed), "env_seed_base": base,
        "env_seeds": [base + i for i in range(amount)], "amount": amount, "rounds_run": rounds,
        "episode_index_layout": ("k = j*amount + i (env i, its j-th episode); the first n_episodes kept; "
                                 "schedule episode read = (env_seed + j) % n"),
        "factor_timing": "CWM: factor written, physics.step with no mj_forward (carl_intra_episode.py:631 masked)",
        "seed_task_rng": bool(seed_task_rng),
        "policy_rng": "JAXAgent.rng = numpy default_rng(env_seed_base) at the start of every condition",
        "renderer": os.environ.get("MUJOCO_GL", ""), "render": bool(render),
        "units": "CWM (factor 1 gravity = multiplier of 9.81 m/s^2)", "T": EXPECTED_T,
        **cwm_prov, **stack_prov, "cwm_file_sha256": cwm_hashes, "crssm_git_sha": git_sha,
        "elapsed_s": elapsed, "wallclock_per_episode_s": elapsed / (rounds * amount),
        "protocol_status": ("walker: ICLR plan §8b design of record" if b["domain"] == "walker" else
                            "quadruped: mirror of the walker design ruled by trajd-advisor "
                            "2026-09-14, pending PI ratification; not in §8b"),
        "stack_reference": ("none: CWM runner refuses drift on reward-factor families at 45e8d5d "
                            "(benchmark/run_eval_benchmark.py:799)" if b["mixed"] else
                            "CWM benchmark/run_eval_benchmark.py @45e8d5d"),
    }
    raw_dir = os.path.join(results_root, "raw")
    raw_rel = outputs.write_raw_json(raw_dir, row_id, cid, returns, mse, int(trained_seed),
                                     env_name, git_sha, meta, paired_streams=paired_streams,
                                     episode_keys=episode_keys)
    arr = np.asarray(returns, dtype=np.float64)
    marr = np.asarray(mse, dtype=np.float64) if mse is not None else None
    from datetime import datetime, timezone
    outputs.append_csv_row(os.path.join(results_root, "main_results.csv"), {
        "row_id": row_id, "condition_id": cid, "method": "crssm", "domain": "dmc",
        "env": env_name, "experiment": bench_name, "seed": int(trained_seed),
        "axis": conditions[cid]["axis"], "n_episodes": len(arr),
        "mean_return": float(arr.mean()), "std_return": float(arr.std()),
        "reward_mse_mean": float(marr.mean()) if marr is not None else float("nan"),
        "reward_mse_std": float(marr.std()) if marr is not None else float("nan"),
        "raw_returns_path": raw_rel, "training_wandb_run_id": "", "eval_wandb_run_id": "offline",
        "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha,
        "notes": (f"action_mode={action_mode}; generator_sha256={gen_sha}; "
                  f"schedule_hash={schedule_hash}; ckpt_step={ckpt_info['step']}; "
                  f"bench={bench_name}; cwm_commit={cwm_prov['cwm_commit']}; "
                  f"cwm_dirty_count={cwm_prov['cwm_dirty_count']}; paired_streams={paired_streams}; "
                  f"paired_streams_reason={outputs.PAIRED_STREAMS_REASON}"),
    })
    cell = build_cell(rows)
    tmeta = {"env": env_name, "experiment": bench_name, "seed": int(trained_seed),
             "axis": conditions[cid]["axis"], "condition_id": cid, "row_id": row_id,
             "lambda": None, "git_sha": git_sha, **meta,
             "paired_streams": paired_streams,
             "paired_streams_reason": outputs.PAIRED_STREAMS_REASON,
             "row_join": "by (episode_worker, episode_ordinal) to raw returns",
             "unjoined_worker_traces": unjoined}
    tpath = os.path.join(trace_root, f"{row_id}__{cid}__trace.npz")
    outputs.write_trace_cell(tpath, cell, tmeta)
    shutil.rmtree(wdir, ignore_errors=True)
    return {"row_id": row_id, "condition_id": cid, "mean_return": float(arr.mean()),
            "n": len(arr), "elapsed_s": elapsed, "raw": os.path.join(results_root, raw_rel),
            "trace": tpath}


def parse_args(argv=None):
    from drift_eval import benches as B
    from drift_eval.cwm import DEFAULT_GENERATOR, VENDORED_GENERATOR_SHA256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, help=f"one of {sorted(B.BENCHES)}")
    ap.add_argument("--seed", type=int, required=True, help="trained seed (checkpoint directory)")
    ap.add_argument("--ckpt-root", default="/home/jovyan/baseline_ckpts/crssm")
    ap.add_argument("--ckpt-dir", default=None, help="overrides <ckpt-root>/<slug>/seed<N>")
    ap.add_argument("--schedules-dir", required=True,
                    help="output dir of the CWM generator CLI run with --bench <bench>")
    ap.add_argument("--generator", default=None,
                    help=f"optional override; default the vendored copy {DEFAULT_GENERATOR}")
    ap.add_argument("--generator-sha256", default=VENDORED_GENERATOR_SHA256,
                    help="asserted at load; default the vendored copy's sha256")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--amount", type=int, default=5,
                    help="envs per batch, stepped in lockstep; 5 = CWM's Evaluate.NumEnvs, so the CRN "
                         "episode plan is CWM's")
    ap.add_argument("--action-mode", choices=sorted(POLICY_MODES), default="native")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--env-parallel", choices=["none", "process"], default="none")
    ap.add_argument("--allow-partial-ckpt", action="store_true")
    ap.add_argument("--no-seed-task-rng", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run conditions already in the CSV")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from drift_eval import benches as B, channel, cwm, outputs
    from drift_eval import config as dcfg

    bench_name = B.resolve_name(args.bench)
    b = B.bench(bench_name)
    port = load_port_yaml()
    conditions = port["conditions"]
    cids = args.conditions or port["default_conditions"]
    unknown = [c for c in cids if c not in conditions]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; known {sorted(conditions)}")

    cwm_hashes = cwm.ensure_cwm(verify=True)
    gen, gen_sha = cwm.load_generator(args.generator, expected_sha256=args.generator_sha256)

    ckpt_dir = args.ckpt_dir or os.path.join(args.ckpt_root, b["slug"], f"seed{args.seed}")
    ckpt_file = os.path.join(ckpt_dir, "checkpoint.ckpt")
    listed = (port["checkpoints"].get(bench_name) or {}).get(args.seed)
    ckpt_sha = cwm.sha256_file(ckpt_file)
    if listed is None or listed["sha256"] != ckpt_sha:
        raise SystemExit(f"{ckpt_file} sha256 {ckpt_sha} is not the inventory entry {listed}")
    sig = dcfg.verify_ckpt(bench_name, ckpt_file)
    if sig["step"] != 1_000_000 and not args.allow_partial_ckpt:
        raise SystemExit(f"{ckpt_file} is at step {sig['step']}, not 1000000 "
                         f"(pass --allow-partial-ckpt to evaluate it anyway)")
    cfg, cfg_source = dcfg.load_config(bench_name, ckpt_dir, args.seed)
    values = cwm.load_values_table()
    bench_spec = load_bench_spec(args.schedules_dir, bench_name, gen, gen_sha, cfg, values)
    wind_clip = B.cwm_wind_clip(values, b["domain"])

    git_sha = outputs.repo_git_sha(REPO)
    row_id = f"crssm_{bench_name}_s{args.seed}"
    done = outputs.done_pairs(os.path.join(args.results_root, "main_results.csv"))
    cache = {}

    def policy_factory(env, base):
        if "agent" not in cache:
            cache["agent"], cache["step"] = load_agent(cfg.update({"seed": int(base)}), env, ckpt_file)
            if cache["step"] != sig["step"]:
                raise RuntimeError("checkpoint step counter changed between reads")
        agent = cache["agent"]
        agent.rng = np.random.default_rng(int(base))
        mode = POLICY_MODES[args.action_mode]

        def policy(obs, state):
            return agent.policy(obs, state, mode=mode)
        return policy

    for cid in cids:
        if (row_id, cid) in done and not args.force:
            print(f"[drift_eval] {row_id} {cid}: already in CSV, skipping")
            continue
        print(f"[drift_eval] {row_id} {cid} action_mode={args.action_mode} "
              f"generator_sha256={gen_sha[:12]}", flush=True)
        out = run_condition(
            cfg=cfg, bench_name=bench_name, cid=cid, conditions=conditions,
            schedules_dir=args.schedules_dir, gen=gen, gen_sha=gen_sha, bench_spec=bench_spec,
            wind_clip=wind_clip, trained_seed=args.seed, n_episodes=args.n_episodes,
            amount=args.amount, policy_factory=policy_factory, action_mode=args.action_mode,
            results_root=args.results_root,
            ckpt_info={"path": ckpt_file, "sha256": ckpt_sha, "step": sig["step"]},
            cfg_source=cfg_source, git_sha=git_sha, cwm_hashes=cwm_hashes,
            env_parallel=args.env_parallel, seed_task_rng=not args.no_seed_task_rng)
        print(f"[drift_eval] {cid}: mean_return={out['mean_return']:.2f} n={out['n']} "
              f"elapsed={out['elapsed_s']:.1f}s -> {out['raw']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
