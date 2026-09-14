"""Bench table and factor semantics: CWM factor ids <-> the CARL regime-B wrapper, in ONE place.

CWM (envs/dmc_latent_factors.py @45e8d5d) and the CARL regime-B wrapper
(benchmark/wrappers/carl_intra_episode.py @45e8d5d) apply the same physical factors in different
units and with different clip ranges:

  id  CWM name / unit                                CARL wrapper name / unit
  0   actuator_strength (walker), x actuator_gear    actuator_strength, x actuator_gear   (same)
  0   joint_damping (quadruped), x dof_damping       joint_damping, x dof_damping         (same)
  1   gravity, multiplier of 9.81                    gravity, ABSOLUTE m/s^2              (x 9.81)
  3   wind, additive N on torso x                    wind_x, additive N on torso x        (same)

  CWM clips (dmc_latent_factors.py:257-274): ids 0-2 to [0.1, 10.0]; wind to +-max|v| over the
  domain's four wind value sets. CARL clips: gravity (0.98, 39.24) m/s^2 = (0.0999, 4.0) x;
  actuator/damping (0.05, 15); wind +-150 (walker) / +-550 (quadruped).

The CARL gravity clip binds on the H conditions (the generator's H-S gravity path reaches
5.126 x 9.81 = 50.3 m/s^2 > 39.24), so in scheduled conditions the port sets the wrapper's clip
ranges to CWM's, expressed in CARL units. The unit conversion and the clip mapping are defined here
and nowhere else.

Bench names are the CWM generator's (`drift_generator.BENCH_DOMAINS`, generator at CWM befabde).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

GRAVITY_MS2 = 9.81            # CWM: opt.gravity[2] = v * (-9.81); CARL wrapper: opt.gravity[2] = -value
CWM_MULT_CLIP = (0.1, 10.0)   # CWM dmc_latent_factors.py:258/265/269
NUM_FACTORS = 4
SPLITS = ("train_values", "test_iid_values", "test_ood_mild_values", "test_ood_hard_values")

# CWM value-table key per (domain, factor id) in config_files/latent_factor_values.yaml.
_VALUE_KEY = {
    ("walker", 0): "actuator_strength",
    ("walker", 1): "gravity",
    ("walker", 3): "wind_walker",
    ("quadruped", 0): "joint_damping",
    ("quadruped", 1): "gravity",
    ("quadruped", 3): "wind_quadruped",
}
# CARL regime-B wrapper feature name per (domain, factor id).
_CARL_NAME = {
    ("walker", 0): "actuator_strength",
    ("quadruped", 0): "joint_damping",
    ("walker", 1): "gravity",
    ("quadruped", 1): "gravity",
    ("walker", 3): "wind_x",
    ("quadruped", 3): "wind_x",
}

# `slug`: checkpoint directory on HF `ssubhnil/cwm` crssm/ and under /home/jovyan/baseline_ckpts/crssm.
# `preset`: the fork's configs.yaml preset (CWM benchmark/reconstruct_crssm_config.py:SLUG_SPEC, whose
# K/act/obs are asserted against checkpoint tensor shapes). K/act/obs repeated here and re-checked.
BENCHES = {
    "walker_3f_phys": dict(
        short="w3f", domain="walker", slug="dmc_walker_k3", preset="benchmark_regime_b_K3",
        task="carl_dmc_walker", regime_b_factors="K3", factor_ids=(0, 1, 3), mixed=False,
        ctx_dim=3, act_dim=6, obs_dim=24),
    "walker_2f_phys": dict(
        short="w2fphys", domain="walker", slug="dmc_walker_k2", preset="benchmark_regime_b",
        task="carl_dmc_walker", regime_b_factors="K2", factor_ids=(0, 1), mixed=False,
        ctx_dim=2, act_dim=6, obs_dim=24),
    "walker_2f_mixed": dict(
        short="w2fmix", domain="walker", slug="dmc_walker_2f_mixed",
        preset="benchmark_regime_b_mixed", task="carl_dmc_walker", regime_b_factors="K3_mixed",
        factor_ids=(1,), mixed=True, ctx_dim=2, act_dim=6, obs_dim=24),
    # Quadruped: a MIRROR of the walker design ruled by trajd-advisor 2026-09-14, pending PI
    # ratification; not in ICLR plan §8b.
    "quadruped_2f_phys": dict(
        short="q2fphys", domain="quadruped", slug="dmc_quadruped_k2",
        preset="benchmark_regime_b_quadruped", task="carl_dmc_quadruped", regime_b_factors="K2",
        factor_ids=(0, 1), mixed=False, ctx_dim=2, act_dim=12, obs_dim=78),
    "quadruped_2f_mixed": dict(
        short="q2fmix", domain="quadruped", slug="dmc_quadruped_2f_mixed",
        preset="benchmark_regime_b_mixed_quadruped", task="carl_dmc_quadruped",
        regime_b_factors="K3_mixed", factor_ids=(1,), mixed=True, ctx_dim=2, act_dim=12,
        obs_dim=78),
}
ALIASES = {"w3f": "walker_3f_phys", "w2fphys": "walker_2f_phys", "w2fmix": "walker_2f_mixed"}
# CWM's env name per domain (the `env` field of its CSV rows and raw JSON metadata).
ENV_NAME = {"walker": "dm_walker_walk", "quadruped": "dm_quadruped_walk"}
_REWARD_MODE_IDS = {}


def env_name(name):
    return ENV_NAME[bench(name)["domain"]]


def cwm_reward_mode_ids(domain):
    """CWM's reward_mode_id: the index of the mode in the SORTED domain reward registry
    (envs/dmc_reward_factors.py:359 `_all_mode_names = sorted(...)`, :432 `_mode_name_to_id`).
    walker: walk_backward 5, walk_forward 6; quadruped: go_north 3, go_south 4."""
    if domain not in _REWARD_MODE_IDS:
        from drift_eval.cwm import ensure_cwm
        ensure_cwm(verify=True)
        from envs.dmc_reward_factors import DOMAIN_REWARD_REGISTRIES
        names = sorted(DOMAIN_REWARD_REGISTRIES[domain]().keys())
        _REWARD_MODE_IDS[domain] = {n: i for i, n in enumerate(names)}
    return _REWARD_MODE_IDS[domain]


def resolve_name(name):
    name = ALIASES.get(name, name)
    if name not in BENCHES:
        raise KeyError(f"unknown bench {name!r}; known: {sorted(BENCHES)} (+aliases {sorted(ALIASES)})")
    return name


def bench(name):
    return BENCHES[resolve_name(name)]


def value_key(domain, fid):
    return _VALUE_KEY[(domain, fid)]


def carl_name(domain, fid):
    return _CARL_NAME[(domain, fid)]


def canonical_fid(carl_feature):
    """CARL feature name -> CWM factor id (as carl_intra_episode._CANONICAL_FACTOR_INDEX)."""
    return {"actuator_strength": 0, "joint_damping": 0, "gravity": 1, "friction": 2,
            "wind_x": 3}[carl_feature]


def to_carl_units(fid, v):
    return float(v) * GRAVITY_MS2 if fid == 1 else float(v)


def to_cwm_units(fid, x):
    return float(x) / GRAVITY_MS2 if fid == 1 else float(x)


def cwm_wind_clip(values_table, domain):
    """CWM's wind clip: max |v| over the domain's four wind value sets (dmc_latent_factors.py:183-193).
    It only matters where wind is active (walker_3f_phys)."""
    vd = values_table[value_key(domain, 3)]
    return max(abs(float(v)) for s in SPLITS for v in vd[s])


def cwm_clip_in_carl_units(fid, wind_clip):
    """CWM's clip range for factor `fid`, in the CARL wrapper's units."""
    if fid == 3:
        return (-float(wind_clip), float(wind_clip))
    lo, hi = CWM_MULT_CLIP
    return (to_carl_units(fid, lo), to_carl_units(fid, hi))


def u_of(fid, v):
    """The generator's natural coordinate: log for multiplicative factors, identity for wind."""
    return float(v) if fid == 3 else math.log(float(v))


def cwm_clip_value(fid, v, wind_clip):
    """CWM's applier, same bounds and same operation (dmc_latent_factors.py:_apply_factor @45e8d5d:
    `value = float(np.clip(value, 0.1, 10.0))` for ids 0-2, `float(np.clip(value, -wc, wc))` for
    wind), in CWM units, BEFORE any unit conversion. E.g. the generator's quadruped H-S joint_damping
    10.000000000000002 becomes 10.0, as it does in CWM."""
    if fid == 3:
        return float(np.clip(v, -float(wind_clip), float(wind_clip)))
    lo, hi = CWM_MULT_CLIP
    return float(np.clip(v, lo, hi))


def schedule_to_carl_units(payload, wind_clip):
    """Deep copy of a CWM generator physics payload, each value clipped as CWM clips it and then
    converted to CARL units.

    The CARL wrapper applies `episodes[k]['physics_fK']` values directly
    (carl_intra_episode.py:548-550, 604-607), so gravity must arrive in m/s^2. CWM clips in CWM units
    and then scales (gravity: `value * (-9.81)`), so the clip comes first here too. Step indices are
    carried unchanged; `source_schedule_hash` keeps the generator's hash and `carl_payload_hash`
    identifies the converted payload.
    """
    out = copy.deepcopy(payload)
    for ep in out["episodes"]:
        for key, events in ep.items():
            if key.startswith("physics_f"):
                fid = int(key[len("physics_f"):])
                ep[key] = [[int(s), to_carl_units(fid, cwm_clip_value(fid, v, wind_clip))]
                           for s, v in events]
    out["processes"] = {k: [to_carl_units(int(k[len("physics_f"):]), v) for v in vals]
                        for k, vals in payload["processes"].items()}
    out["source_schedule_hash"] = payload.get("schedule_hash")
    out["units"] = "carl (gravity m/s^2)"
    out["carl_payload_hash"] = hashlib.sha256(json.dumps(
        {"episodes": out["episodes"], "processes": out["processes"],
         "source": out["source_schedule_hash"]}, sort_keys=True).encode()).hexdigest()
    return out
