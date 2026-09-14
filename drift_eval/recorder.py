"""Per-step record of the factor values the physics ACTUALLY used, checked against the schedule.

`DriftRecorder` sits directly inside embodied's FromGymnasium, outside the CARL regime-B wrapper
(and the reward switcher on mixed benches). After every reset and step it reads the parameters
back from MuJoCo (the read-back does not touch MjData; a test hashes the full state around it):

  factor 0  walker: model.actuator_gear / nominal;   quadruped: model.dof_damping / nominal
  factor 1  -model.opt.gravity[2]                    (m/s^2; recorded /9.81 in CWM units)
  factor 3  data.xfrc_applied[torso, 0]              (N)

Timing. CWM's DMControlLatentFactors writes the factor and then steps the physics with NO
mj_forward (envs/dmc_latent_factors.py:440/:451). The CARL wrapper calls `physics.forward()` after
every switch (carl_intra_episode.py:630-631). `step()` therefore masks `physics.forward` for the
duration of the inner step only (reset keeps its forward, as CWM's reset does at :385).

Indexing follows CWM §8b item 2: row t is the value in force while action t was applied
(decision step t, 1..T); the reset value (step 0) is kept separately. The schedule episode an env
reads is (env_seed + ordinal) % n, as CWM's wrappers read it. In scheduled mode every readback is
compared EXACTLY with the schedule entry for that decision step (after CWM's clip) and a mismatch
raises; so does a payload in this process whose hash is not the one this env was built for, a
reset/step PhysicsError retry (it re-draws, which unpairs episodes), and an episode whose length is
not the schedule's T. `reward_mode_id` is CWM's id (index in the sorted domain reward registry).
"""

from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from drift_eval import benches as B


def _no_forward(*args, **kwargs):
    return None


def dense_schedule(events, T):
    """[[step, value], ...] -> float64 array over decision steps 0..T (forward-filled)."""
    out = np.full(T + 1, np.nan, dtype=np.float64)
    ev = sorted((int(s), float(v)) for s, v in events)
    if not ev or ev[0][0] != 0:
        raise ValueError("schedule has no step-0 (reset) entry")
    j = 0
    cur = ev[0][1]
    for t in range(T + 1):
        while j < len(ev) and ev[j][0] == t:
            cur = ev[j][1]
            j += 1
        out[t] = cur
    return out


def readback_native(regime_b, domain, fid):
    """Parameter currently in the physics for factor `fid`, in CARL units (scalar)."""
    ph = regime_b._physics
    if fid == 0:
        if domain == "quadruped":
            cur, nom = ph.model.dof_damping, regime_b._default_dof_damping
        else:
            cur, nom = ph.model.actuator_gear, regime_b._default_actuator_gear
        mask = nom != 0
        return float(np.median(np.asarray(cur)[mask] / np.asarray(nom)[mask]))
    if fid == 1:
        return float(-ph.model.opt.gravity[2])
    if fid == 3:
        return float(ph.data.xfrc_applied[regime_b._torso_body_id, 0])
    raise ValueError(f"factor {fid} not supported by the CARL regime-B wrapper")


def matches_native(regime_b, domain, fid, expected_native):
    """Exact comparison with what the CARL wrapper writes for `expected_native` (CARL units)."""
    ph = regime_b._physics
    if fid == 0:
        if domain == "quadruped":
            return bool(np.array_equal(ph.model.dof_damping,
                                       regime_b._default_dof_damping * expected_native))
        return bool(np.array_equal(ph.model.actuator_gear,
                                   regime_b._default_actuator_gear * expected_native))
    if fid == 1:
        return bool(ph.model.opt.gravity[2] == -expected_native)
    if fid == 3:
        return bool(ph.data.xfrc_applied[regime_b._torso_body_id, 0] == expected_native)
    raise ValueError(fid)


class DriftRecorder(gym.Wrapper):

    def __init__(self, env, regime_b, reward_switch, spec, sink, strict=True):
        super().__init__(env)
        self._rb = regime_b
        self._rs = reward_switch
        self._spec = spec
        self._sink = sink
        self._strict = bool(strict)
        self._bench = B.bench(spec.bench)
        self._domain = self._bench["domain"]
        self._fids = tuple(self._bench["factor_ids"])
        self._rb_mod = sys.modules[type(regime_b).__module__]
        self._rs_mod = sys.modules[type(reward_switch).__module__] if reward_switch is not None else None
        self._mode_ids = B.cwm_reward_mode_ids(self._domain) if reward_switch is not None else None
        self._failures0 = int(getattr(regime_b, "_physics_failures", 0))
        self._T = int(spec.expected_T)
        self._expected = None
        if spec.switch_mode == "scheduled":
            p = spec.factor_schedule_carl
            self._expected = []
            for ep in p["episodes"]:
                row = {}
                for fid in self._fids:
                    raw = dense_schedule(ep[f"physics_f{fid}"], self._T)
                    lo, hi = B.cwm_clip_in_carl_units(fid, spec.wind_clip)
                    row[fid] = np.array([float(np.clip(x, lo, hi)) for x in raw])
                self._expected.append(row)
        self._acc = None

    # -- checks --

    def _check_payload_identity(self):
        f = self._rb_mod._ACTIVE_FACTOR_SCHEDULE
        r = self._rs_mod._ACTIVE_REWARD_SCHEDULE if self._rs_mod is not None else None
        if self._spec.switch_mode == "scheduled":
            want = self._spec.factor_schedule_carl["carl_payload_hash"]
            got = None if f is None else f.get("carl_payload_hash")
            if got != want:
                raise RuntimeError(
                    f"[drift_eval] scheduled env reset but the CARL wrapper module "
                    f"{self._rb_mod.__name__} holds payload {got!r}, not {want!r}. The schedule did "
                    f"not reach the process that steps this env.")
            if self._rs_mod is not None:
                want_r = self._spec.reward_schedule["schedule_hash"]
                got_r = None if r is None else r.get("schedule_hash")
                if got_r != want_r:
                    raise RuntimeError(
                        f"[drift_eval] mixed scheduled env reset but the reward switcher holds "
                        f"{got_r!r}, not {want_r!r}.")
        else:
            if f is not None or r is not None:
                raise RuntimeError(
                    "[drift_eval] bernoulli (training-protocol) env reset with a schedule payload "
                    "still installed in this process; it would silently run scheduled.")

    def _check_physics_failures(self):
        n = int(getattr(self._rb, "_physics_failures", 0))
        if n != self._failures0:
            raise RuntimeError(
                f"[drift_eval] CARL wrapper retried a PhysicsError ({n - self._failures0} new). A "
                f"retry re-draws the episode and unpairs it from the other conditions; refusing.")

    def _read_and_check(self, t):
        vals = np.full(B.NUM_FACTORS, np.nan, dtype=np.float64)
        for fid in self._fids:
            native = readback_native(self._rb, self._domain, fid)
            vals[fid] = B.to_cwm_units(fid, native)
            if self._expected is not None and self._strict:
                n = len(self._expected)
                exp = self._expected[(int(self._spec.env_seed) + self._rb._episode_ordinal) % n][fid][t]
                if not matches_native(self._rb, self._domain, fid, exp):
                    raise RuntimeError(
                        f"[drift_eval] factor {fid} at decision step {t}: physics holds "
                        f"{native!r} (CARL units), schedule says {exp!r}. Applied path != schedule.")
        return vals

    def _mode_id(self, info):
        if self._mode_ids is None:
            return -1
        return int(self._mode_ids[info["reward_mode"]])

    # -- env interface --

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._check_payload_identity()
        self._check_physics_failures()
        vals = self._read_and_check(0)
        self._acc = {"reward": [], "reward_mode_id": [], "factor_values": [], "decision_step": [],
                     "reset_factor_values": vals.astype(np.float32),
                     "reset_reward_mode_id": self._mode_id(info)}
        return obs, info

    def step(self, action):
        ph = self._rb._physics
        ph.forward = _no_forward          # no mj_forward between factor write and physics.step
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        finally:
            if "forward" in vars(ph):
                del ph.forward
        self._check_physics_failures()
        t = int(self._rb._decision_step)
        vals = self._read_and_check(t)
        if self._acc is not None:
            self._acc["reward"].append(float(reward))
            self._acc["reward_mode_id"].append(self._mode_id(info))
            self._acc["factor_values"].append(vals.astype(np.float32))
            self._acc["decision_step"].append(t)
            if terminated or truncated:
                self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self):
        acc, self._acc = self._acc, None
        T = len(acc["reward"])
        if self._spec.switch_mode == "scheduled" and T != self._T:
            raise RuntimeError(
                f"[drift_eval] scheduled episode ended after {T} decision steps, schedule T is "
                f"{self._T}; the late part of the schedule was not executed.")
        active = np.array([i in self._fids for i in range(B.NUM_FACTORS)], dtype=bool)
        if self._sink is not None:
            self._sink.append({
                "reward": np.asarray(acc["reward"], dtype=np.float32),
                "reward_mode_id": np.asarray(acc["reward_mode_id"], dtype=np.int16),
                "factor_values": np.stack(acc["factor_values"]).astype(np.float32),
                "factor_active": active,
                "decision_step": np.asarray(acc["decision_step"], dtype=np.int32),
                "reset_factor_values": acc["reset_factor_values"],
                "reset_reward_mode_id": np.asarray(acc["reset_reward_mode_id"], dtype=np.int16),
                "wrapper_episode_ordinal": np.asarray(int(self._rb._episode_ordinal)),
                "env_seed": np.asarray(int(self._spec.env_seed)),
            })
