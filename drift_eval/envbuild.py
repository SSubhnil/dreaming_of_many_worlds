"""Env construction for one drift condition: the fork's own env chain plus four port hooks.

`build_drift_env(config, spec)` is a picklable constructor. In order:
  1. verify the CWM root bytes and install/clear the payloads IN THIS PROCESS (channel.py);
  2. build the env with the fork's `contextual_mbrl.dreamer.envs.make_env` (unchanged chain:
     CARL env -> NormalizeContextWrapper -> CARL regime-B wrapper [-> reward switcher] ->
     FromGymnasium -> RenderImage -> ResizeImage -> dreamerv3.wrap_env), with `seed = env_seed`;
  3. seed the dm_control task RNG from `env_seed` (optional, default on). CARL rebuilds the
     dm_control env on context change without a task seed (carl/envs/dmc/carl_dmcontrol.py:70-76,
     loader passes random=None), so `reset(seed=)` does NOT fix initial states: measured on this box,
     two CARL walker envs reset with seed 4007 gave different initial observations. The hook passes
     one persistent RandomState into every rebuild, so env i's j-th episode starts from the j-th
     draw of the stream seeded env_seed, whatever the factor schedule;
  4. in scheduled conditions, set the regime-B wrapper's clip ranges to CWM's (benches.py);
  5. insert `DriftRecorder` inside FromGymnasium.
"""

from __future__ import annotations

import contextlib
import dataclasses
import types
from typing import Optional

import numpy as np

from drift_eval import benches as B
from drift_eval import channel


@dataclasses.dataclass
class DriftEnvSpec:
    bench: str
    switch_mode: str                          # "bernoulli" | "scheduled"
    env_seed: int
    wind_clip: float
    factor_schedule_carl: Optional[dict] = None
    reward_schedule: Optional[dict] = None
    trace_dir: Optional[str] = None
    worker_tag: str = "w0"
    seed_task_rng: bool = True
    expected_T: int = 500
    render: bool = True                       # False only for CPU tests without a checkpoint
    strict: bool = True


def _children(obj):
    d = getattr(obj, "__dict__", {})
    for key in ("env", "_env"):
        if key in d and d[key] is not None:
            yield key, d[key]


def walk_chain(env, max_depth=40):
    """Outermost-first list of (parent, attribute, object) along .env/._env, without __getattr__."""
    out = [(None, None, env)]
    cur = env
    for _ in range(max_depth):
        nxt = list(_children(cur))
        if not nxt:
            break
        key, child = nxt[0]
        out.append((cur, key, child))
        cur = child
    return out


def _find(chain, class_name):
    hits = [(p, k, o) for p, k, o in chain if type(o).__name__ == class_name]
    return hits[0] if hits else (None, None, None)


def _seeded_update_context(self):
    from carl.envs.dmc import carl_dmcontrol
    from carl.envs.dmc.wrappers import MujocoToGymWrapper
    env = carl_dmcontrol.load_dmc_env(
        domain_name=self.domain, task_name=self.task, context=self.context,
        task_kwargs={"random": self._drift_task_random},
        environment_kwargs={"flat_observation": True})
    self.env = MujocoToGymWrapper(env)


def match_cwm_semantics(rb, rs, cfg, spec):
    """Make the CARL wrappers draw and index as CWM's DMControlLatentFactors / DMControlRewardFactors.
    Each item was confirmed bitwise against CWM's stack by the verifier (2026-09-14, PARITY_TABLE.md,
    table `crssmfix`).

    (a) Gravity value set = CWM table x 9.81, computed as CWM computes it (`v * (-9.81)`,
        dmc_latent_factors.py:266), not CARL's literal m/s^2 table (3.924 vs 0.4*9.81: 1 ulp).
    (b) Bernoulli draw order by CWM factor id (dmc_latent_factors.py:367/:407). CARL sorts factors by
        NAME (carl_intra_episode.py:175), so quadruped drew gravity before joint_damping. Every
        per-factor array indexed by that order is permuted with the names.
    (c) Reward-switch RNG seeded with env_seed, as CWM's eval.py:118 seeds DMControlRewardFactors,
        not seed+1000 (fork envs.py:916).
    (d) Schedule episode index (env_seed + ordinal) % n in both wrappers, as CWM's `_sched_offset`
        (dmc_latent_factors.py:221, dmc_reward_factors.py); CARL uses ordinal % n
        (carl_intra_episode.py:427, carl_reward_switch.py:155).
    """
    from drift_eval.cwm import load_values_table
    values = load_values_table()
    if "gravity" in rb._factor_configs:                                            # (a)
        overrides = dict(e.split("=", 1) for e in (cfg.env.carl.regime_b_factor_splits or [])
                         if e and "=" in e)
        split = overrides.get("gravity", cfg.env.carl.regime_b_split)
        rb._factor_configs["gravity"].values = [B.to_carl_units(1, v)
                                                for v in values["gravity"][f"{split}_values"]]
    order = sorted(rb._factor_names, key=B.canonical_fid)                         # (b)
    if order != rb._factor_names:
        idx = [rb._factor_names.index(n) for n in order]
        rb._factor_names = order
        for attr in ("_train_low", "_train_high", "_current_lambdas", "_factor_values",
                     "_factor_changed", "_segment_ids", "_steps_since_switch"):
            setattr(rb, attr, getattr(rb, attr)[idx])
        rb._wind_factor_idx = order.index("wind_x") if "wind_x" in order else None
    if rs is not None:                                                             # (c)
        rs._rng = np.random.RandomState(int(spec.env_seed))
    _offset = int(spec.env_seed)                                                   # (d)
    rb._drift_sched_offset = _offset
    ci = __import__(type(rb).__module__, fromlist=["_"])

    def _phys_sched(self, _ci=ci):
        eps = (_ci._ACTIVE_FACTOR_SCHEDULE or {}).get("episodes", [])
        return eps[(self._drift_sched_offset + self._episode_ordinal) % len(eps)] if eps else None
    rb._get_scheduled_episode = types.MethodType(_phys_sched, rb)
    if rs is not None:
        rs._drift_sched_offset = _offset
        cr = __import__(type(rs).__module__, fromlist=["_"])

        def _rew_sched(self, _cr=cr):
            eps = (_cr._ACTIVE_REWARD_SCHEDULE or {}).get("episodes", [])
            if not eps:
                return None
            return eps[(self._drift_sched_offset + self._episode_ordinal) % len(eps)].get("reward") or None
        rs._get_scheduled_episode = types.MethodType(_rew_sched, rs)


@contextlib.contextmanager
def _no_render():
    """CPU tests only: RenderImage renders every step and needs GL. The fork's cRSSM encoder and
    decoder ignore images (cnn_keys '$^'), but the obs space differs, so never use with a ckpt."""
    from dreamerv3 import embodied
    orig = embodied.core.wrappers.RenderImage
    embodied.core.wrappers.RenderImage = lambda env, key="image": env
    try:
        yield
    finally:
        embodied.core.wrappers.RenderImage = orig


def build_drift_env(config, spec: DriftEnvSpec):
    from drift_eval.cwm import ensure_cwm
    from drift_eval.recorder import DriftRecorder

    from drift_eval.stack import verify_stack
    bench = B.bench(spec.bench)
    mixed = bool(bench["mixed"])
    verify_stack()
    ensure_cwm(verify=True)
    # (1) payload in THIS process, before the env (and so before its first reset) exists.
    channel.install_in_process(spec.switch_mode, spec.factor_schedule_carl, spec.reward_schedule,
                               mixed=mixed)

    from contextual_mbrl.dreamer.envs import make_env
    cfg = config.update({"seed": int(spec.env_seed), "envs.parallel": "none"})
    if cfg.task != bench["task"] or cfg.env.carl.regime_b_factors != bench["regime_b_factors"]:
        raise RuntimeError(f"[drift_eval] config task/factors {cfg.task}/"
                           f"{cfg.env.carl.regime_b_factors} do not match bench {spec.bench}")
    with (contextlib.nullcontext() if spec.render else _no_render()):
        env = make_env(cfg)                                                       # (2)

    chain = walk_chain(env)
    _, _, fg = _find(chain, "FromGymnasium")
    _, _, rb = _find(chain, "CARLIntraEpisodeSwitching")
    _, _, rs = _find(chain, "CARLRewardModeSwitcher")
    carl_env = next((o for _, _, o in chain
                     if hasattr(type(o), "_update_context") and hasattr(type(o), "domain")), None)
    if fg is None or rb is None or carl_env is None:
        raise RuntimeError(f"[drift_eval] env chain missing a piece: FromGymnasium={fg is not None} "
                           f"regime-B={rb is not None} CARL={carl_env is not None}")
    if mixed != (rs is not None):
        raise RuntimeError(f"[drift_eval] bench {spec.bench} mixed={mixed} but reward switcher "
                           f"present={rs is not None}")
    want = sorted(B.carl_name(bench["domain"], f) for f in bench["factor_ids"])
    if sorted(rb._factor_names) != want:
        raise RuntimeError(f"[drift_eval] regime-B wrapper manages {rb._factor_names}, bench "
                           f"{spec.bench} expects {want}")

    if spec.seed_task_rng:                                                        # (3)
        carl_env._drift_task_random = np.random.RandomState(int(spec.env_seed))
        carl_env._update_context = types.MethodType(_seeded_update_context, carl_env)
        carl_env._update_context()   # the current dm_control env already draws from the stream

    match_cwm_semantics(rb, rs, cfg, spec)                                        # (3b)

    if spec.switch_mode == "scheduled":                                           # (4)
        for name, fc in rb._factor_configs.items():
            fc.clip_range = B.cwm_clip_in_carl_units(B.canonical_fid(name), spec.wind_clip)

    sink = channel.DirTraceSink(spec.trace_dir, spec.worker_tag) if spec.trace_dir else None
    fg._env = DriftRecorder(fg._env, rb, rs, spec, sink, strict=spec.strict)      # (5)
    env._drift_parts = {"regime_b": rb, "reward_switch": rs, "carl_env": carl_env}
    return env


def make_drift_batch_env(config, specs, parallel="none"):
    """BatchEnv over one env per spec. `parallel='process'` builds each env in a spawned worker;
    the payload travels in the pickled constructor either way. No RestartOnException: a crashed
    env is a failed condition, not a silently re-seeded one."""
    from functools import partial
    from dreamerv3 import embodied
    if parallel == "none":
        envs = [build_drift_env(config, s) for s in specs]
    elif parallel == "process":
        envs = [embodied.Parallel(partial(build_drift_env, config, s), "process") for s in specs]
    else:
        raise ValueError(parallel)
    return embodied.BatchEnv(envs, parallel=(parallel != "none"))
