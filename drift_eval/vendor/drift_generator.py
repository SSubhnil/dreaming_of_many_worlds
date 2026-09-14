#!/usr/bin/env python3
"""§8b drift generator: calibrated in-range sine (IR-S), its value-matched abrupt permutation
(IR-A), the hard-excursion pair (H-S / H-A), and the τ-dial ramps — as schedule-CRN payloads the
eval runner installs and `eval.build_vec_env` forwards into spawned workers.

What is calibrated, and against what. The training process is Bernoulli with per-step probability
λ̄ and i.i.d. draws from the factor's train value set V, so over the 499 transitions of a 500-step
episode its expected total variation in the factor's natural coordinate u is

    B_f = 499 · λ̄ · E|u(v_i) − u(v_j)|,   v_i, v_j ~ Uniform(V) independent,

with u = log for the multiplicative factors (0,1,2) and u = identity for wind (3, additive
Newtons). Every in-range condition is generated to that TV, so the gradual and abrupt arms differ
in HOW the variation is spent, not in how much.

Two things cannot both be exact, and this generator says which it holds. With the sample multiset
fixed, a path's TV is the sum of its monotone runs' extents; matching the sine's TV exactly forces
the same number of runs, so the permutation's increments are bounded by the value grid's gaps.
IR-A therefore relaxes the multiset to BAND resolution
(occupancy held exactly, TV held exactly, jump count ≈ the number of band transitions), which is
what makes a matched abrupt comparator possible at all; `abrupt_band_path` states the trade in
full.

Conventions (§8b item 2, action-time indexing). An episode's event list is DENSE and indexed by
decision step: entry 0 is the value in force at reset, entries 1…T the value in force while action
t is applied. `envs/dmc_latent_factors.py` applies entry 0 at reset and entry t when its decision
counter reaches t, and its trace records `decision_step`, so a verifier can check the parameter
used BY EACH TRANSITION rather than inferring an offset.

Quadruped benches — a MIRROR, not the design of record. `quadruped_2f_phys` (joint_damping +
gravity) and `quadruped_2f_mixed` (gravity + compass_direction reward) get the same four conditions,
the τ-dial and the same TV-matching rule as walker. The mirror was ruled by trajd-advisor on
2026-09-14. It is NOT in ICLR plan §8b and is pending PI ratification. `resolve_bench` reads each
bench's active factors and reward train modes from its preset (`config_files/experiments/dmc.yaml`)
and its value sets from `config_files/latent_factor_values.yaml` under the config name, as
`envs/dmc_latent_factors.build_latent_factor_config` does: `joint_damping` is factor id 0 (u = log)
with its own table row, and `wind` reads `wind_<domain>`. λ̄ is the midpoint of the DMC
`LambdaSwitchRange` (the preset's, else configure.yaml's). On a mixed bench the reward objective is
FIXED within each episode and balanced across episodes (`build_reward_fixed_schedule`), which is the
w2fmix rule. `build_condition` and the legacy CLI path are untouched, and walker outputs are
hash-guarded against 45e8d5d in tests/test_drift_generator.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter

import numpy as np

GENERATOR_VERSION = "8b-1.0.0"

# Factor id ↔ name, mirroring envs/dmc_latent_factors.FACTOR_NAMES (the index IS the id).
FACTOR_NAMES = ["actuator_strength", "gravity", "friction", "wind"]
# Config-name aliases, mirroring envs/dmc_latent_factors.FACTOR_ALIASES: joint_damping IS factor 0
# (on quadruped it scales dof_damping), and its values live under its own table key.
FACTOR_ALIASES = {"joint_damping": "actuator_strength"}
_ADDITIVE = {3}                      # wind is an additive force in N; the rest are multiplicative
_JUMP_EPS = 1e-9

# Bench key -> domain. The preset of the same name supplies the active factors and reward modes.
# The walker keys are the reference the quadruped keys mirror.
BENCH_DOMAINS = {
    "walker_2f_phys": "walker",
    "walker_3f_phys": "walker",
    "walker_2f_mixed": "walker",
    "quadruped_2f_phys": "quadruped",
    "quadruped_2f_mixed": "quadruped",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def factor_ref(name, domain):
    """(factor id, value-table key) for a config factor NAME on `domain`."""
    canonical = FACTOR_ALIASES.get(name, name)
    fid = FACTOR_NAMES.index(canonical)
    key = f"wind_{domain}" if canonical == "wind" else name
    return fid, key


def u_of(factor_id, v):
    """Natural coordinate: log for multiplicative factors, identity for the additive one."""
    return float(v) if factor_id in _ADDITIVE else math.log(float(v))


def v_of(factor_id, u):
    return float(u) if factor_id in _ADDITIVE else math.exp(float(u))


def tv(path):
    """Total variation of a sampled path, in the coordinate it is given in."""
    a = np.asarray(path, dtype=np.float64)
    return float(np.abs(np.diff(a)).sum())


def training_tv_budget(factor_id, values, T=500, lam=0.0055):
    """B_f = (T−1)·λ̄·E|u(v_i) − u(v_j)| — the training process's expected TV over one episode."""
    us = [u_of(factor_id, v) for v in values]
    pairs = [abs(a - b) for a in us for b in us]
    return (T - 1) * lam * (sum(pairs) / len(pairs))


def band_quantize(u_path, n_bands=4):
    """Group a path's samples into `n_bands` equal-width bands and replace each by its band's
    occupancy-weighted centre. Occupancy (time spent in each band) is preserved exactly; the value
    multiset is preserved only at band resolution. This is what makes an abrupt comparator
    possible at all — see `abrupt_band_path`."""
    p = np.asarray(u_path, dtype=np.float64)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 1e-15:
        return p.copy(), np.zeros(len(p), dtype=int), np.array([lo])
    edges = np.linspace(lo, hi, n_bands + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bands - 1)
    centres = np.array([p[idx == b].mean() if np.any(idx == b) else 0.5 * (edges[b] + edges[b + 1])
                        for b in range(n_bands)])
    return centres[idx], idx, centres


def sine_path(amplitude, cycles, T, phase=0.0):
    """Common-phase sine sampled at decision steps 0…T (neutral endpoints for integer cycles)."""
    t = np.arange(T + 1, dtype=np.float64) / T
    return amplitude * np.sin(2 * math.pi * cycles * t + phase)


def calibrate_sine(factor_id, values, budget, T=500, cycles=1, amp_cap=None):
    """Amplitude whose sine path has TV EXACTLY `budget`, or the in-range maximum if the range
    binds. TV is linear in amplitude (TV(A) = A·TV(1)), so this is one division, not a search;
    the achieved TV and whether the range bound bound are both returned.
    """
    cap = amp_cap if amp_cap is not None else max(abs(u_of(factor_id, v)) for v in values)
    unit_tv = tv(sine_path(1.0, cycles, T))
    amp = budget / unit_tv
    capped = amp > cap
    if capped:
        amp = cap
    path = sine_path(amp, cycles, T)
    return amp, tv(path), path, capped


def abrupt_band_path(path, n_bands=4):
    """The value-matched ABRUPT comparator for `path`, and the reason it is built this way.

    A permutation of a dense sine's samples CANNOT be both TV-matched and abrupt. For a fixed
    multiset, TV is the sum of the monotone runs' extents, so matching the sine's TV pins the
    number of runs; within a run the increments are the multiset's own gaps, which for a dense
    sine are the sine's own step sizes. Exact-multiset matching therefore forces a staircase of
    small steps — provably not the abrupt regime. (Checked numerically in
    tests/test_drift_generator.py::test_exact_multiset_permutation_cannot_be_abrupt.)

    So the contract is relaxed in one stated place: samples are grouped into `n_bands` bands, each
    band's occupancy (time spent) is preserved EXACTLY, the path holds each band's value as a
    constant block, and the block values are rescaled so the TV matches the source path's exactly.
    What is held: occupancy, endpoints, TV, range. What is not: the value multiset below band
    resolution. Jump count = number of block transitions, which is what the training process's
    ≈ (T−1)·λ̄ events should be compared against.
    """
    p = np.asarray(path, dtype=np.float64)
    _, idx, centres = band_quantize(p, n_bands=n_bands)
    blocks = [idx[0]]
    bounds = [0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1]:
            blocks.append(idx[i])
            bounds.append(i)
    bounds.append(len(idx))

    out = np.empty_like(p)
    for j, b in enumerate(blocks):
        out[bounds[j]:bounds[j + 1]] = centres[b]
    # Rescale the block values about their mean so the TV matches the source exactly. Scaling a
    # piecewise-constant path scales its TV linearly, so this is exact, and it preserves both the
    # block structure and the occupancy.
    src, cur = tv(p), tv(out)
    if cur > 0:
        m = out.mean()
        out = m + (out - m) * (src / cur)
    return out


def tau_ramp_path(u_lo, u_hi, T, tau, event_step=101):
    """Single event at `event_step`, executed as a ramp of length τ. TV = |u_hi − u_lo| for every
    τ, so the τ-dial varies smoothness at fixed total variation (τ = 1 is the abrupt anchor)."""
    p = np.full(T + 1, float(u_lo), dtype=np.float64)
    for j in range(tau):
        s = event_step + j
        if s <= T:
            p[s:] = u_lo + (u_hi - u_lo) * (j + 1) / tau
    return p


def path_stats(factor_id, u_path, grid_u):
    """Per-episode quantities the appendix figures rest on (TV, jump count, extrema, occupancy)."""
    d = np.abs(np.diff(np.asarray(u_path, dtype=np.float64)))
    occ = Counter(float(x) for x in np.asarray(u_path))
    return {
        "tv": tv(u_path),
        "jump_count": int((d > _JUMP_EPS).sum()),
        "max_increment": float(d.max()) if len(d) else 0.0,
        "u_min": float(np.min(u_path)),
        "u_max": float(np.max(u_path)),
        "v_min": v_of(factor_id, float(np.min(u_path))),
        "v_max": v_of(factor_id, float(np.max(u_path))),
        "occupancy": {f"{k:.6f}": occ[k] for k in sorted(occ)},
    }


def _episode_events(factor_id, u_path):
    """Dense [[step, value], …] over 0…T in the factor's PHYSICAL units (entry 0 = reset value)."""
    return [[int(t), v_of(factor_id, float(u))] for t, u in enumerate(u_path)]


def build_condition(condition, factors, T=500, lam=0.0055, n_episodes=64, cycles=1,
                    hard_values=None, tau=None, seed=0):
    """One schedule-CRN payload.

    `factors` maps factor id -> train value list. `condition` is one of
    ir_s / ir_a / h_s / h_a / tau.
    """
    rng = np.random.RandomState(seed)
    episodes, stats = [], {}
    per_factor_paths = {}

    for fid, values in factors.items():
        grid_u = sorted(u_of(fid, v) for v in values)
        budget = training_tv_budget(fid, values, T=T, lam=lam)
        capped = False
        if condition in ("ir_s", "ir_a"):
            _, achieved, base, capped = calibrate_sine(fid, values, budget, T=T, cycles=cycles)
        elif condition in ("h_s", "h_a"):
            hv = (hard_values or {}).get(fid)
            if hv is None:
                raise ValueError(f"condition {condition} needs hard_values for factor {fid}")
            cap = max(abs(u_of(fid, v)) for v in hv)
            # Half-cycle excursion to the hard endpoint: TV lands in the §8b 1.36–2.28×B band by
            # construction (the target is 1.8×B) unless the hard range itself binds.
            _, achieved, base, capped = calibrate_sine(fid, list(values) + list(hv), budget * 1.8,
                                                       T=T, cycles=0.5, amp_cap=cap)
        elif condition == "tau":
            if tau is None:
                raise ValueError("condition 'tau' needs tau")
            base = tau_ramp_path(grid_u[len(grid_u) // 2], grid_u[-1], T, tau)
            achieved = tv(base)
        else:
            raise ValueError(f"unknown condition {condition!r}")

        if condition in ("ir_a", "h_a"):
            path = abrupt_band_path(base)
        else:
            path = base

        per_factor_paths[fid] = path
        stats[f"physics_f{fid}"] = {"budget_tv": budget, "achieved_tv_base": achieved,
                                    "amplitude_capped_by_range": bool(capped),
                                    **path_stats(fid, path, grid_u)}

    # CRN across conditions: every episode of every condition uses the SAME path, so an episode
    # index pairs across conditions and only the schedule shape differs. Episode-to-episode
    # variation comes from the env seed, which the battery pins per episode index.
    for _ in range(n_episodes):
        episodes.append({f"physics_f{fid}": _episode_events(fid, p)
                         for fid, p in per_factor_paths.items()})

    payload = {
        "generator_version": GENERATOR_VERSION,
        "condition": condition,
        "processes": {f"physics_f{fid}": [float(v) for v in vals]
                      for fid, vals in factors.items()},
        "lambda": lam,
        "n_episodes": n_episodes,
        "T": T,
        "master_seed": seed,
        "episodes": episodes,
        "stats": stats,
    }
    payload["schedule_hash"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("generator_version", "condition", "processes",
                                            "lambda", "T", "episodes")},
                   sort_keys=True).encode()).hexdigest()
    return payload


def resolve_bench(bench, values_path=None, presets_path=None, base_config_path=None):
    """Active factors, value sets, λ̄ and reward train modes for a bench, read from the repo configs.

    Nothing is copied here. The factor list comes from the bench preset. The values come from the
    latent-factor table under the CONFIG name (so `joint_damping` reads its own row, not
    actuator_strength's). λ̄ = midpoint of LambdaSwitchRange: per-episode λ_k ~ U[lo, hi], and the
    expected TV is linear in λ. Inline preset value lists are refused rather than silently mirrored.
    """
    import yaml

    if bench not in BENCH_DOMAINS:
        raise ValueError(f"unknown bench {bench!r}; known: {sorted(BENCH_DOMAINS)}")
    domain = BENCH_DOMAINS[bench]

    def _load(path, rel):
        with open(path or os.path.join(_REPO_ROOT, "config_files", rel)) as f:
            return yaml.safe_load(f)

    table = _load(values_path, "latent_factor_values.yaml")
    dmc = _load(presets_path, os.path.join("experiments", "dmc.yaml"))[bench]["Environment"]["DMC"]
    base_lf = _load(base_config_path, "configure.yaml")["Environment"]["DMC"]["LatentFactors"]

    lf = dmc["LatentFactors"]
    factors, hard, keys = {}, {}, {}
    for name, fc in lf["Factors"].items():
        fc = fc or {}
        if not fc.get("active", False):
            continue
        fid, key = factor_ref(name, domain)
        if fid in factors:
            raise ValueError(f"{bench}: factor id {fid} declared twice ({keys[fid]!r}, {name!r})")
        inline = [k for k in fc if k.endswith("_values")]
        if inline:
            raise ValueError(f"{bench}: preset declares {name}.{inline} inline; "
                             f"this generator reads latent_factor_values.yaml only")
        factors[fid] = [float(v) for v in table[key]["train_values"]]
        hard[fid] = [float(v) for v in table[key]["test_ood_hard_values"]]
        keys[fid] = key

    lsr = lf.get("LambdaSwitchRange", base_lf.get("LambdaSwitchRange"))
    if lsr is None:
        raise ValueError(f"{bench}: no LambdaSwitchRange; the per-factor lambda_switch path is "
                         f"not what the walker budget was calibrated against")
    lo, hi = float(lsr[0]), float(lsr[1])

    rf = dmc.get("RewardFactors") or {}
    reward_modes = None
    if rf.get("Enabled", False):
        if "train_modes" not in rf:
            raise ValueError(f"{bench}: RewardFactors enabled without train_modes in the preset")
        reward_modes = [str(m) for m in rf["train_modes"]]

    order = sorted(factors)
    return {
        "bench": bench,
        "domain": domain,
        "factors": {fid: factors[fid] for fid in order},
        "hard_values": {fid: hard[fid] for fid in order},
        "value_keys": {fid: keys[fid] for fid in order},
        "lambda_range": [lo, hi],
        "lam": (lo + hi) / 2,
        "reward_modes": reward_modes,
    }


def build_reward_fixed_schedule(modes, n_episodes=64, T=500):
    """Reward-label schedule-CRN for a MIXED bench under a drift condition (the w2fmix rule).

    The objective is FIXED within each episode, and episodes are balanced across modes: episode k
    gets modes[k % len(modes)]. The same payload serves every condition, so episode index still
    pairs across conditions. Each episode has ONE entry, at step 0, which is consumed at reset:
    `DMControlRewardFactors.step` increments the decision counter before matching entries, so a
    step-0 entry never fires again. A dense repeat of the label would log an identity "switch"
    every step. The wrappers offset the episode ordinal by seed (`_sched_offset`), so the balance
    holds over any contiguous window whose length is a multiple of len(modes).
    """
    modes = [str(m) for m in modes]
    if not modes:
        raise ValueError("no reward modes")
    if n_episodes % len(modes):
        raise ValueError(f"n_episodes={n_episodes} is not a multiple of {len(modes)} modes; "
                         f"the objective would not be balanced across episodes")
    episodes = [{"reward": [[0, modes[k % len(modes)]]]} for k in range(n_episodes)]
    payload = {
        "generator_version": GENERATOR_VERSION,
        "condition": "reward_fixed",
        "processes": {"reward": modes},
        "lambda": 0.0,
        "n_episodes": n_episodes,
        "T": T,
        "master_seed": 0,
        "episodes": episodes,
        "stats": {"reward": {
            "episodes_per_mode": dict(Counter(ep["reward"][0][1] for ep in episodes)),
            "max_distinct_labels_in_an_episode": max(len({m for _, m in ep["reward"]})
                                                     for ep in episodes),
        }},
    }
    payload["schedule_hash"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("generator_version", "condition", "processes",
                                            "lambda", "T", "episodes")},
                   sort_keys=True).encode()).hexdigest()
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--values", default="config_files/latent_factor_values.yaml")
    ap.add_argument("--bench", default=None, choices=sorted(BENCH_DOMAINS),
                    help="resolve factors, value sets, λ̄ and reward modes from this bench's "
                         "preset; excludes --factors/--domain/--lam")
    ap.add_argument("--factors", default=None,
                    help="comma-separated factor NAMES, as the bench's config activates them "
                         "(default actuator_strength,gravity,wind)")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--lam", type=float, default=None, help="default 0.0055")
    ap.add_argument("--taus", default="1,8,16,64,256")
    ap.add_argument("--domain", default=None, help="default walker")
    args = ap.parse_args()

    reward_modes, spec = None, None
    if args.bench is not None:
        if any(x is not None for x in (args.factors, args.domain, args.lam)):
            ap.error("--bench resolves factors, domain and λ̄ from the bench preset; "
                     "do not pass --factors/--domain/--lam with it")
        spec = resolve_bench(args.bench, values_path=args.values)
        factors, hard, lam = spec["factors"], spec["hard_values"], spec["lam"]
        reward_modes = spec["reward_modes"]
    else:
        import yaml
        with open(args.values) as f:
            vals = yaml.safe_load(f)
        domain = args.domain if args.domain is not None else "walker"
        lam = args.lam if args.lam is not None else 0.0055
        names = args.factors if args.factors is not None else "actuator_strength,gravity,wind"
        factors, hard = {}, {}
        for name in names.split(","):
            fid, key = factor_ref(name, domain)
            factors[fid] = list(vals[key]["train_values"])
            hard[fid] = list(vals[key]["test_ood_hard_values"])

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    conds = [("ir_s", {}), ("ir_a", {}),
             ("h_s", {"hard_values": hard}), ("h_a", {"hard_values": hard})]
    for cond, kw in conds:
        p = build_condition(cond, factors, T=args.T, lam=lam,
                            n_episodes=args.episodes, **kw)
        path = os.path.join(args.out_dir, f"{cond}.json")
        with open(path, "w") as f:
            json.dump(p, f)
        written.append((cond, path, p["schedule_hash"][:12], p["stats"]))
    for tau in [int(x) for x in args.taus.split(",")]:
        p = build_condition("tau", factors, T=args.T, lam=lam,
                            n_episodes=args.episodes, tau=tau)
        path = os.path.join(args.out_dir, f"tau_{tau}.json")
        with open(path, "w") as f:
            json.dump(p, f)
        written.append((f"tau_{tau}", path, p["schedule_hash"][:12], p["stats"]))

    for cond, path, h, stats in written:
        line = ", ".join(f"{k} TV={v['tv']:.4f} (budget {v['budget_tv']:.4f}) "
                         f"jumps={v['jump_count']} max_inc={v['max_increment']:.4f}"
                         for k, v in stats.items())
        print(f"{cond:14s} hash={h}  {line}")

    if reward_modes is not None:
        r = build_reward_fixed_schedule(reward_modes, n_episodes=args.episodes, T=args.T)
        with open(os.path.join(args.out_dir, "reward_fixed.json"), "w") as f:
            json.dump(r, f)
        print(f"{'reward_fixed':14s} hash={r['schedule_hash'][:12]}  {r['stats']['reward']}")
    if spec is not None:
        with open(os.path.abspath(__file__), "rb") as fh:
            gen_sha = hashlib.sha256(fh.read()).hexdigest()
        with open(os.path.join(args.out_dir, "bench_spec.json"), "w") as f:
            json.dump({**spec, "generator_version": GENERATOR_VERSION,
                       "generator_sha256": gen_sha}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
