"""CPU tests for the cRSSM gradual-shift (§8b) port. No GPU, no rendering.

The fork's interpreter has no pytest, so these are unittest tests:

    cd /home/jovyan/projects/dreaming_of_many_worlds
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    XLA_FLAGS=--xla_force_host_platform_device_count=1 \
    PYTHONPATH=$PWD:$PWD/dreamerv3_compat \
    /home/jovyan/envs/dali/bin/python -m unittest drift_eval.tests.test_drift_eval -v

Every env is the fork's real chain (contextual_mbrl.dreamer.envs.make_env: CARL walker/quadruped
-> NormalizeContextWrapper -> CWM's CARL regime-B wrapper [-> reward switcher] -> FromGymnasium ->
dreamerv3.wrap_env) with RenderImage bypassed. Factor checks read MuJoCo directly with code in THIS
file, not the port's recorder, and run the recorder with strict=False so the test itself must catch
a defect.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "disable"
for _k, _v in (("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
               ("XLA_FLAGS", "--xla_force_host_platform_device_count=1"),
               ("TF_CPP_MIN_LOG_LEVEL", "3"), ("DISABLE_TENSORBOARD", "1"),
               ("CAUSAL_WORLD_MODEL_ROOT", "/home/jovyan/m10_code/45e8d5d")):
    os.environ.setdefault(_k, _v)

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402

VENDORED_GENERATOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "vendor", "drift_generator.py")
GENERATOR = os.environ.get("DRIFT_GENERATOR", VENDORED_GENERATOR)
GENERATOR_SHA256 = "39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6"
CWM_ROOT = os.environ["CAUSAL_WORLD_MODEL_ROOT"]
VALUES = os.path.join(CWM_ROOT, "config_files", "latent_factor_values.yaml")
# A real raw JSON written by CWM's runner (benchmark/run_eval_benchmark.py:_write_raw_json).
REFERENCE_RAW_JSON = ("/home/jovyan/projects/CausalWorldModel/docs/trajd/a2c_reads_2026-09-11/"
                      "ood_rec_c2wm_d1_950k_w2fmix/raw/d1_w2fmix_c2wm_s1__lambda_0.05.json")
REFERENCE_RAW_JSON_SHA256 = "384006370d6e40f5037eb1f5ff8f68f31c6c337d0f97967f5c22fdfa0a977125"
BENCH_NAMES = ("walker_3f_phys", "walker_2f_phys", "walker_2f_mixed",
               "quadruped_2f_phys", "quadruped_2f_mixed")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DALI_REPO = "/home/jovyan/projects/DALI"
_TMP = None


def setUpModule():
    global _TMP
    if not os.path.exists(GENERATOR):
        raise RuntimeError(f"generator not found: {GENERATOR}")
    _TMP = tempfile.mkdtemp(prefix="drift_eval_tests_")
    for b in BENCH_NAMES:
        subprocess.run([sys.executable, "-m", "drift_eval.make_schedules", "--bench", b,
                        "--out-dir", os.path.join(_TMP, "sched", b), "--episodes", "4",
                        "--taus", "8", "--generator", GENERATOR], check=True, capture_output=True,
                       cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))


def tearDownModule():
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)


# ── helpers ────────────────────────────────────────────────────────────────────────────────────

def sched_dir(bench):
    return os.path.join(_TMP, "sched", bench)


def sched(bench, name):
    with open(os.path.join(sched_dir(bench), f"{name}.json")) as f:
        return json.load(f)


def make_cfg(bench):
    from drift_eval import config as dcfg
    return dcfg.reconstruct(bench, logdir="/nonexistent/drift_eval_test_logdir", seed=1)


def make_spec(bench, cond, seed=4007, trace_dir=None, strict=False, seed_task_rng=True,
              drop_payload=False, payloads=None):
    from drift_eval import benches as B, cwm, envbuild
    b = B.bench(bench)
    wind = B.cwm_wind_clip(cwm.load_values_table(), b["domain"])
    carl = reward = None
    mode = "bernoulli"
    if cond is not None:
        mode = "scheduled"
        phys, rew = payloads if payloads is not None else (
            sched(bench, cond), sched(bench, "reward_fixed") if b["mixed"] else None)
        carl = B.schedule_to_carl_units(phys, wind)
        reward = rew
    if drop_payload:
        carl = reward = None
    return envbuild.DriftEnvSpec(bench=bench, switch_mode=mode, env_seed=seed, wind_clip=wind,
                                 factor_schedule_carl=carl, reward_schedule=reward,
                                 trace_dir=trace_dir, worker_tag="w0",
                                 seed_task_rng=seed_task_rng, render=False, strict=strict)


def build(bench, spec):
    from drift_eval import envbuild
    return envbuild.build_drift_env(make_cfg(bench), spec)


def close(env):
    from drift_eval import channel
    try:
        env.close()
    finally:
        channel.clear_process()


def drive_episode(env, on_obs=None, max_steps=None):
    """One episode through the embodied API with zero actions; on_obs() after reset and each step."""
    act = {"action": np.zeros(env.act_space["action"].shape, np.float32), "reset": True}
    obs = env.step(act)
    if on_obs:
        on_obs()
    act["reset"] = False
    n = 0
    while not bool(obs["is_last"]):
        obs = env.step(act)
        n += 1
        if on_obs:
            on_obs()
        if max_steps is not None and n >= max_steps:
            break
    return n


def cwm_wind_clip_literal(domain):
    import yaml
    with open(VALUES) as f:
        vd = yaml.safe_load(f)[f"wind_{domain}"]
    return max(abs(v) for s in ("train_values", "test_iid_values", "test_ood_mild_values",
                                "test_ood_hard_values") for v in vd[s])


def expected_native(payload, fid, t, domain):
    """What the physics must hold at decision step t, in CARL units, computed from the generator's
    payload with CWM's clip and CWM's gravity convention (v * 9.81)."""
    events = payload["episodes"][0][f"physics_f{fid}"]
    assert [s for s, _ in events] == list(range(len(events))), "generator events are not dense"
    v = float(events[t][1])
    if fid == 3:
        wc = cwm_wind_clip_literal(domain)
        return min(max(v, -wc), wc)
    v = min(max(v, 0.1), 10.0)
    return v * 9.81 if fid == 1 else v


def physics_snapshot(rb, domain, fid):
    ph = rb._physics
    if fid == 0:
        if domain == "quadruped":
            return (np.array(ph.model.dof_damping), np.array(rb._default_dof_damping))
        return (np.array(ph.model.actuator_gear), np.array(rb._default_actuator_gear))
    if fid == 1:
        return float(ph.model.opt.gravity[2])
    return float(ph.data.xfrc_applied[rb._torso_body_id, 0])


def snapshot_matches(snap, fid, exp):
    if fid == 0:
        cur, nom = snap
        return bool(np.array_equal(cur, nom * exp))
    if fid == 1:
        return snap == -exp
    return snap == exp


def snapshot_to_u(snap, fid):
    if fid == 0:
        cur, nom = snap
        m = nom != 0
        return float(np.log(np.median(cur[m] / nom[m])))
    if fid == 1:
        return float(np.log(-snap / 9.81))
    return float(snap)


def _run_child(target, *args, timeout=300):
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(send, *args))
    proc.start()
    send.close()
    if not recv.poll(timeout):
        proc.terminate()
        raise AssertionError(f"child produced nothing within {timeout}s")
    out = recv.recv()
    proc.join(60)
    return out


# ── spawned children (module level: spawn pickles by reference) ────────────────────────────────

def _child_build_and_step(conn, bench, spec, n_steps):
    try:
        from drift_eval import benches as B
        env = build(bench, spec)
        rb = env._drift_parts["regime_b"]
        b = B.bench(bench)
        vals = []
        drive_episode(env, lambda: vals.append(
            (int(rb._decision_step), {f: physics_snapshot(rb, b["domain"], f)
                                      for f in b["factor_ids"]})), max_steps=n_steps)
        held = sys.modules[type(rb).__module__]._ACTIVE_FACTOR_SCHEDULE
        conn.send({"ok": True, "pid": os.getpid(), "vals": vals,
                   "hash": None if held is None else held.get("carl_payload_hash")})
        env.close()
    except Exception as exc:
        conn.send({"ok": False, "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}"[:600]})
    finally:
        conn.close()


def _child_bare_state(conn):
    from drift_eval.cwm import carl_modules
    ci, cr = carl_modules()
    conn.send({"factor_none": ci._ACTIVE_FACTOR_SCHEDULE is None,
               "reward_none": cr._ACTIVE_REWARD_SCHEDULE is None})
    conn.close()


def _child_scheduled_without_payload(conn, bench, spec):
    try:
        build(bench, spec)
        conn.send({"raised": False, "msg": "built", "env_module_imported": True})
    except Exception as exc:
        conn.send({"raised": isinstance(exc, RuntimeError)
                   and "no factor schedule payload reached this process" in str(exc),
                   "msg": f"{type(exc).__name__}: {exc}"[:400],
                   "env_module_imported": "contextual_mbrl.dreamer.envs" in sys.modules})
    finally:
        conn.close()


def _child_ensure_cwm(conn, root):
    os.environ["CAUSAL_WORLD_MODEL_ROOT"] = root
    try:
        from drift_eval import cwm
        cwm.ensure_cwm(verify=True)
        conn.send({"raised": False, "msg": ""})
    except Exception as exc:
        conn.send({"raised": isinstance(exc, RuntimeError)
                   and "expected the 45e8d5d bytes" in str(exc), "msg": str(exc)[:300]})
    finally:
        conn.close()


def _fake_policy_factory(env, base):
    shape = env.act_space["action"].shape

    def policy(obs, state):
        n = len(obs["is_first"])
        return {"action": np.zeros((n,) + shape, np.float32),
                "reward_hat": np.zeros(n, np.float32)}, state
    return policy


# ── tests ──────────────────────────────────────────────────────────────────────────────────────

class TestScheduleApplication(unittest.TestCase):

    def test_applied_factor_equals_schedule_value_exact(self):
        """Schedule file -> physics at decision step t holds the schedule value, exactly, for every
        t in 0..T, on walker and quadruped, IR and H (where CARL's own gravity clip would bind), and
        the mixed benches' reward label is the generator's label for that episode."""
        from drift_eval import benches as B
        cases = (("walker_3f_phys", "h_s"), ("walker_3f_phys", "ir_s"),
                 ("walker_2f_mixed", "ir_s"), ("quadruped_2f_phys", "h_s"),
                 ("quadruped_2f_mixed", "h_s"))
        for bench, cond in cases:
            with self.subTest(bench=bench, cond=cond):
                b = B.bench(bench)
                p = sched(bench, cond)
                for ep in p["episodes"]:
                    self.assertEqual(ep, p["episodes"][0])
                rp = sched(bench, "reward_fixed") if b["mixed"] else None
                env = build(bench, make_spec(bench, cond, strict=False))
                rb, rs = env._drift_parts["regime_b"], env._drift_parts["reward_switch"]
                bad, seen = [], []

                def on_obs():
                    t = int(rb._decision_step)
                    seen.append(t)
                    for fid in b["factor_ids"]:
                        exp = expected_native(p, fid, t, b["domain"])
                        snap = physics_snapshot(rb, b["domain"], fid)
                        if not snapshot_matches(snap, fid, exp) and len(bad) < 5:
                            bad.append((fid, t, snap if fid else "array", exp))
                    if rp is not None:
                        lab = rp["episodes"][(4007 + rb._episode_ordinal) % len(rp["episodes"])]["reward"][0][1]
                        if rs._current_mode != lab and len(bad) < 5:
                            bad.append(("reward", t, rs._current_mode, lab))
                try:
                    n = drive_episode(env, on_obs)
                finally:
                    close(env)
                self.assertEqual(bad, [])
                self.assertEqual(seen, list(range(p["T"] + 1)))
                self.assertEqual(n, p["T"])

    def test_scheduled_values_pass_through_cwm_clip_bit_for_bit(self):
        """CWM's own applier (envs/dmc_latent_factors.py:DMControlLatentFactors._apply_factor @45e8d5d,
        called on a stub physics) and the port must yield the identical clipped factor value and the
        identical gravity / wind parameter at every decision step of H-S. The quadruped payload holds
        joint_damping 10.000000000000002, which CWM applies as 10.0."""
        import types
        from drift_eval import benches as B
        from drift_eval.cwm import ensure_cwm
        ensure_cwm(verify=True)
        from envs.dmc_latent_factors import DMControlLatentFactors
        self.assertTrue(DMControlLatentFactors.__module__ == "envs.dmc_latent_factors"
                        and sys.modules["envs.dmc_latent_factors"].__file__.startswith(CWM_ROOT))

        def cwm_apply(domain, fid, v, wc):
            ph = types.SimpleNamespace(
                model=types.SimpleNamespace(actuator_gear=np.ones(1), dof_damping=np.ones(1),
                                            opt=types.SimpleNamespace(gravity=np.zeros(3)),
                                            geom_friction=np.ones((1, 3))),
                data=types.SimpleNamespace(xfrc_applied=np.zeros((2, 6))))
            stub = types.SimpleNamespace(
                env=types.SimpleNamespace(_env=types.SimpleNamespace(physics=ph)),
                _use_joint_damping=(domain == "quadruped"), _default_actuator_gear=np.ones(1),
                _default_dof_damping=np.ones(1), _default_geom_friction=np.ones((1, 3)),
                _wind_clip=wc, _torso_body_id=1, _factor_values=np.array([1.0, 1.0, 1.0, 0.0]))
            DMControlLatentFactors._apply_factor(stub, fid, v)
            return float(stub._factor_values[fid]), ph

        for bench in ("quadruped_2f_phys", "walker_3f_phys"):
            with self.subTest(bench=bench):
                b = B.bench(bench)
                wc = cwm_wind_clip_literal(b["domain"])
                p = sched(bench, "h_s")
                if bench == "quadruped_2f_phys":
                    self.assertTrue(any(v > 10.0 for _, v in p["episodes"][0]["physics_f0"]),
                                    "payload no longer exercises the upper clip")
                env = build(bench, make_spec(bench, "h_s"))
                rb = env._drift_parts["regime_b"]
                bad = []

                def on_obs():
                    t = int(rb._decision_step)
                    for fid in b["factor_ids"]:
                        v = p["episodes"][0][f"physics_f{fid}"][t][1]
                        val, ph = cwm_apply(b["domain"], fid, v, wc)
                        port = float(rb._factor_values[rb._factor_names.index(B.carl_name(b["domain"], fid))])
                        if fid == 1:
                            ok = rb._physics.model.opt.gravity[2] == ph.model.opt.gravity[2]
                        elif fid == 3:
                            ok = port == val and rb._physics.data.xfrc_applied[rb._torso_body_id, 0] \
                                == ph.data.xfrc_applied[1, 0]
                        else:
                            ok = port == val
                        if not ok and len(bad) < 5:
                            bad.append((fid, t, v, val, port))
                try:
                    drive_episode(env, on_obs)
                finally:
                    close(env)
                self.assertEqual(bad, [])

    def test_ir_a_jump_count_on_applied_path_matches_generator(self):
        from drift_eval import benches as B, channel
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                b = B.bench(bench)
                tdir = tempfile.mkdtemp(dir=_TMP)
                env = build(bench, make_spec(bench, "ir_a", trace_dir=tdir))
                try:
                    drive_episode(env)
                finally:
                    close(env)
                traces = channel.collect_traces(tdir)
                self.assertEqual(len(traces), 1)
                tr = traces[0]
                p = sched(bench, "ir_a")
                for fid in b["factor_ids"]:
                    path = np.concatenate([[tr["reset_factor_values"][fid]],
                                           tr["factor_values"][:, fid]]).astype(np.float64)
                    u = path if fid == 3 else np.log(path)
                    jumps = int((np.abs(np.diff(u)) > 1e-9).sum())
                    self.assertEqual(jumps, p["stats"][f"physics_f{fid}"]["jump_count"],
                                     f"factor {fid}")

    def test_tv_of_gradual_equals_matched_abrupt_on_applied_paths(self):
        """TV(IR-S) == TV(IR-A) and TV(H-S) == TV(H-A) to 1e-9, measured on what the physics used,
        and equal to the generator's own TV."""
        from drift_eval import benches as B
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            b = B.bench(bench)
            for s_cond, a_cond in (("ir_s", "ir_a"), ("h_s", "h_a")):
                with self.subTest(bench=bench, pair=(s_cond, a_cond)):
                    tvs = {}
                    for cond in (s_cond, a_cond):
                        env = build(bench, make_spec(bench, cond))
                        rb = env._drift_parts["regime_b"]
                        paths = {f: [] for f in b["factor_ids"]}
                        try:
                            drive_episode(env, lambda: [paths[f].append(snapshot_to_u(
                                physics_snapshot(rb, b["domain"], f), f)) for f in b["factor_ids"]])
                        finally:
                            close(env)
                        tvs[cond] = {f: float(np.abs(np.diff(paths[f])).sum()) for f in paths}
                    gen_stats = sched(bench, s_cond)["stats"]
                    for f in b["factor_ids"]:
                        self.assertLessEqual(abs(tvs[s_cond][f] - tvs[a_cond][f]), 1e-9, f"factor {f}")
                        self.assertLessEqual(abs(tvs[s_cond][f] - gen_stats[f"physics_f{f}"]["tv"]),
                                             1e-9, f"factor {f}")


class TestProcessPropagation(unittest.TestCase):

    def test_schedule_reaches_the_spawned_process_that_steps_the_env(self):
        """The env is built AND stepped in a spawned child; the payload travels in the constructor
        spec, the child's wrapper module holds it, and the child's physics follows it."""
        from drift_eval import benches as B
        for bench, cond in (("walker_3f_phys", "ir_s"), ("quadruped_2f_phys", "h_s")):
            with self.subTest(bench=bench):
                b = B.bench(bench)
                spec = make_spec(bench, cond, strict=True)
                out = _run_child(_child_build_and_step, bench, spec, 3)
                self.assertTrue(out["ok"], out.get("error"))
                self.assertNotEqual(out["pid"], os.getpid())
                self.assertEqual(out["hash"], spec.factor_schedule_carl["carl_payload_hash"])
                p = sched(bench, cond)
                self.assertEqual([t for t, _ in out["vals"]], [0, 1, 2, 3])
                for t, snaps in out["vals"]:
                    for fid, snap in snaps.items():
                        self.assertTrue(snapshot_matches(snap, fid,
                                                         expected_native(p, fid, t, b["domain"])),
                                        f"factor {fid} step {t}")

    def test_parent_module_global_does_not_cross_spawn(self):
        """Standing fixture of the defect class (CWM ledger DB-260912-03): a schedule set as a
        module global in the parent is None in a spawned child. Passes by asserting the defect."""
        from drift_eval import channel
        from drift_eval.cwm import carl_modules
        ci, cr = carl_modules()
        ci.set_factor_schedule({"episodes": [], "schedule_hash": "parent"})
        cr.set_reward_schedule({"episodes": [], "schedule_hash": "parent"})
        try:
            out = _run_child(_child_bare_state)
        finally:
            channel.clear_process()
        self.assertTrue(out["factor_none"])
        self.assertTrue(out["reward_none"])


class TestFailClosed(unittest.TestCase):

    def test_scheduled_without_payload_raises_before_the_env_exists(self):
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                spec = make_spec(bench, "ir_s", drop_payload=True)
                self.assertEqual(spec.switch_mode, "scheduled")
                out = _run_child(_child_scheduled_without_payload, bench, spec)
                self.assertTrue(out["raised"], out["msg"])
                self.assertFalse(out["env_module_imported"], "env chain was imported before the raise")

    def test_mixed_scheduled_without_reward_label_raises(self):
        from drift_eval import benches as B, channel
        for bench in ("walker_2f_mixed", "quadruped_2f_mixed"):
            with self.subTest(bench=bench):
                carl = B.schedule_to_carl_units(sched(bench, "ir_s"),
                                                cwm_wind_clip_literal(B.bench(bench)["domain"]))
                try:
                    with self.assertRaisesRegex(RuntimeError, "no reward label payload"):
                        channel.install_in_process("scheduled", carl, None, mixed=True)
                finally:
                    channel.clear_process()

    def test_bernoulli_refuses_a_payload_and_never_runs_a_leftover_schedule(self):
        from drift_eval import benches as B, channel
        from drift_eval.cwm import carl_modules
        ci, _ = carl_modules()
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                carl = B.schedule_to_carl_units(sched(bench, "ir_s"),
                                                cwm_wind_clip_literal(B.bench(bench)["domain"]))
                with self.assertRaises(ValueError):
                    channel.install_in_process("bernoulli", carl, None)
                ci.set_factor_schedule(carl)                       # leftover from a previous condition
                env = build(bench, make_spec(bench, None))
                try:
                    self.assertIsNone(channel.process_state()["factor_schedule_hash"])
                    ci.set_factor_schedule(carl)                   # re-introduced after construction
                    with self.assertRaisesRegex(RuntimeError, "still installed"):
                        drive_episode(env, max_steps=1)
                finally:
                    close(env)

    def test_cwm_root_with_different_wrapper_bytes_is_refused(self):
        from drift_eval.cwm import EXPECTED_SHA256
        root = tempfile.mkdtemp(dir=_TMP)
        for rel in EXPECTED_SHA256:
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            shutil.copy(os.path.join(CWM_ROOT, rel), os.path.join(root, rel))
        with open(os.path.join(root, "benchmark/wrappers/carl_intra_episode.py"), "a") as f:
            f.write("\n# one extra line\n")
        out = _run_child(_child_ensure_cwm, root)
        self.assertTrue(out["raised"], out["msg"])
        out = _run_child(_child_ensure_cwm, CWM_ROOT)
        self.assertFalse(out["raised"], out["msg"])


class TestPairingAndTables(unittest.TestCase):

    def test_seeded_task_rng_pairs_initial_states_across_conditions(self):
        """Same env seed -> same initial states episode by episode under two different schedules;
        another seed -> different. Natively CARL does not seed the dm_control task RNG."""
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                def initial_qpos(cond, seed):
                    env = build(bench, make_spec(bench, cond, seed=seed))
                    rb = env._drift_parts["regime_b"]
                    qs = []
                    try:
                        for _ in range(2):
                            snap = []
                            drive_episode(env, lambda: snap.append(rb._physics.data.qpos.copy())
                                          if not snap else None, max_steps=1)
                            qs.append(snap[0])
                    finally:
                        close(env)
                    return qs
                s, a, o = initial_qpos("ir_s", 4007), initial_qpos("ir_a", 4007), initial_qpos("ir_s", 4008)
                for k in range(2):
                    np.testing.assert_array_equal(s[k], a[k])
                self.assertFalse(np.array_equal(s[0], s[1]))
                self.assertFalse(np.array_equal(s[0], o[0]))

    def test_carl_train_value_sets_are_the_cwm_tables(self):
        """Condition A runs the CARL wrapper's own train value sets; they must be CWM's table (in
        CARL units), and the generator's bench_spec must read the same rows."""
        from drift_eval import benches as B, cwm
        values = cwm.load_values_table()
        for bench in BENCH_NAMES:
            with self.subTest(bench=bench):
                b = B.bench(bench)
                with open(os.path.join(sched_dir(bench), "bench_spec.json")) as f:
                    spec = json.load(f)
                env = build(bench, make_spec(bench, None))
                rb = env._drift_parts["regime_b"]
                try:
                    for fid in b["factor_ids"]:
                        key = B.value_key(b["domain"], fid)
                        cwm_train = [float(v) for v in values[key]["train_values"]]
                        self.assertEqual([float(v) for v in spec["factors"][str(fid)]], cwm_train)
                        got = [float(v) for v in rb._factor_configs[B.carl_name(b["domain"], fid)].values]
                        np.testing.assert_allclose(got, [B.to_carl_units(fid, v) for v in cwm_train],
                                                   rtol=1e-12, atol=1e-12)
                finally:
                    close(env)


class TestOutputsAndProvenance(unittest.TestCase):

    def test_outputs_match_cwm_runner_schema(self):
        from drift_eval import benches as B, cwm, outputs as O, run as R
        port = R.load_port_yaml()
        gen, gen_sha = cwm.load_generator(GENERATOR)
        self.assertEqual(gen_sha, GENERATOR_SHA256)
        values = cwm.load_values_table()
        ref = None
        if os.path.exists(REFERENCE_RAW_JSON):
            with open(REFERENCE_RAW_JSON, "rb") as f:
                blob = f.read()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), REFERENCE_RAW_JSON_SHA256)
            ref = json.loads(blob)
        else:
            print(f"\n[note] reference raw JSON absent ({REFERENCE_RAW_JSON}); embedded schema only")
        for bench, cid in (("walker_2f_mixed", "drift_ir_s"), ("quadruped_2f_phys", "phys_train")):
            with self.subTest(bench=bench, cid=cid):
                b = B.bench(bench)
                cfg = make_cfg(bench)
                res = os.path.join(_TMP, "results", bench)
                bench_spec = R.load_bench_spec(sched_dir(bench), bench, gen, gen_sha, cfg, values)
                out = R.run_condition(
                    cfg=cfg, bench_name=bench, cid=cid, conditions=port["conditions"],
                    schedules_dir=sched_dir(bench), gen=gen, gen_sha=gen_sha, bench_spec=bench_spec,
                    wind_clip=B.cwm_wind_clip(values, b["domain"]), trained_seed=1, n_episodes=2,
                    amount=2, policy_factory=_fake_policy_factory, action_mode="native",
                    results_root=res, ckpt_info={"path": "none (CPU test)", "sha256": "none", "step": 0},
                    cfg_source="reconstructed", git_sha="test", cwm_hashes={}, render=False)
                with open(out["raw"]) as f:
                    raw = json.load(f)
                self.assertEqual(O.validate_raw_json(raw, reference=ref), [])
                self.assertEqual(raw["n_episodes"], 2)
                self.assertEqual(raw["metadata"]["generator_sha256"], GENERATOR_SHA256)
                self.assertEqual(raw["metadata"]["action_mode"], "native")
                with open(os.path.join(res, "main_results.csv"), newline="") as f:
                    self.assertEqual(next(csv.reader(f)), O.CSV_COLUMNS)
                cell = O.load_trace_cell(out["trace"])
                self.assertEqual(O.validate_trace_cell(cell), [])
                self.assertEqual(cell["reward"].shape, (2, 500))
                self.assertEqual(cell["_meta"]["generator_sha256"], GENERATOR_SHA256)
                self.assertEqual(cell["_meta"]["action_mode"], "native")
                active = [i in b["factor_ids"] for i in range(4)]
                self.assertEqual(cell["factor_active"].tolist(), active)
                for i in range(4):
                    self.assertEqual(bool(np.isnan(cell["factor_values"][:, :, i]).all()), not active[i])

    def test_outputs_record_the_external_cwm_commit_and_config_hashes(self):
        """Raw JSON, CSV notes and trace meta carry the CWM root's commit + dirty count and the sha256
        of dmc.yaml / configure.yaml / latent_factor_values.yaml, and those match the tree."""
        from drift_eval import benches as B, cwm, outputs as O, run as R
        gen, gen_sha = cwm.load_generator(GENERATOR)
        values = cwm.load_values_table()
        bench, cid = "quadruped_2f_phys", "phys_train"
        cfg = make_cfg(bench)
        res = os.path.join(tempfile.mkdtemp(dir=_TMP), "results")
        out = R.run_condition(
            cfg=cfg, bench_name=bench, cid=cid, conditions=R.load_port_yaml()["conditions"],
            schedules_dir=sched_dir(bench), gen=gen, gen_sha=gen_sha,
            bench_spec=R.load_bench_spec(sched_dir(bench), bench, gen, gen_sha, cfg, values),
            wind_clip=B.cwm_wind_clip(values, "quadruped"), trained_seed=1, n_episodes=2, amount=2,
            policy_factory=_fake_policy_factory, action_mode="native", results_root=res,
            ckpt_info={"path": "none (CPU test)", "sha256": "none", "step": 0},
            cfg_source="reconstructed", git_sha="test", cwm_hashes={}, render=False)
        commit = subprocess.check_output(["git", "-C", CWM_ROOT, "rev-parse", "HEAD"]).decode().strip()
        dirty = len([l for l in subprocess.check_output(
            ["git", "-C", CWM_ROOT, "status", "--porcelain"]).decode().splitlines() if l.strip()])
        want_sha = {rel: hashlib.sha256(open(os.path.join(CWM_ROOT, rel), "rb").read()).hexdigest()
                    for rel in ("config_files/experiments/dmc.yaml", "config_files/configure.yaml",
                                "config_files/latent_factor_values.yaml")}
        with open(out["raw"]) as f:
            raw_md = json.load(f)["metadata"]
        meta = O.load_trace_cell(out["trace"])["_meta"]
        agent_py = os.path.join(REPO, "dreamerv3_compat", "dreamerv3", "agent.py")
        agent_sha = hashlib.sha256(open(agent_py, "rb").read()).hexdigest()
        for md in (raw_md, meta):
            self.assertEqual(md["dreamerv3_file"],
                             os.path.join(REPO, "dreamerv3_compat", "dreamerv3", "__init__.py"))
            self.assertEqual(md["agent_py_sha256"], agent_sha)
            self.assertEqual(md["cwm_commit"], commit)
            self.assertEqual(md["cwm_dirty_count"], dirty)
            self.assertEqual(md["cwm_config_sha256"], want_sha)
        with open(os.path.join(res, "main_results.csv"), newline="") as f:
            notes = next(csv.DictReader(f))["notes"]
        self.assertIn(f"cwm_commit={commit}", notes)
        self.assertIn(f"cwm_dirty_count={dirty}", notes)

    def test_runner_refuses_foreign_generator_bytes_or_payloads(self):
        from drift_eval import cwm, run as R
        gen, gen_sha = cwm.load_generator(GENERATOR)
        values = cwm.load_values_table()
        bench = "walker_3f_phys"
        cfg = make_cfg(bench)
        d = tempfile.mkdtemp(dir=_TMP)
        shutil.copytree(sched_dir(bench), os.path.join(d, "s"))
        d = os.path.join(d, "s")
        spec = R.load_bench_spec(d, bench, gen, gen_sha, cfg, values)          # accepted as generated
        with open(os.path.join(d, "bench_spec.json")) as f:
            tampered = json.load(f)
        tampered["generator_sha256"] = "0" * 64
        with open(os.path.join(d, "bench_spec.json"), "w") as f:
            json.dump(tampered, f)
        with self.assertRaisesRegex(RuntimeError, "generator_sha256"):
            R.load_bench_spec(d, bench, gen, gen_sha, cfg, values)
        with open(os.path.join(d, "ir_s.json")) as f:
            p = json.load(f)
        p["generator_version"] = "not-this-generator"
        with open(os.path.join(d, "ir_s.json"), "w") as f:
            json.dump(p, f)
        with self.assertRaisesRegex(RuntimeError, "generator_version"):
            R.load_condition_payloads(d, "drift_ir_s", bench, spec, gen, R.load_port_yaml()["conditions"],
                                      cwm_wind_clip_literal("walker"))

    def test_generator_sha256_is_asserted_at_load(self):
        """The vendored copy is the default and carries the recorded sha256; any other bytes are
        refused at load."""
        from drift_eval import cwm
        gen, sha = cwm.load_generator()
        self.assertEqual(gen.__drift_eval_path__, os.path.abspath(VENDORED_GENERATOR))
        self.assertEqual(sha, GENERATOR_SHA256)
        with open(os.path.join(os.path.dirname(VENDORED_GENERATOR), "VENDORED.md")) as f:
            self.assertIn(GENERATOR_SHA256, f.read())
        other = os.path.join(tempfile.mkdtemp(dir=_TMP), "drift_generator.py")
        shutil.copy(VENDORED_GENERATOR, other)
        with open(other, "a") as f:
            f.write("\n# one extra line\n")
        with self.assertRaisesRegex(RuntimeError, "Refusing to generate or read schedules"):
            cwm.load_generator(other)


class TestStackResolution(unittest.TestCase):
    """The dali venv's editable finders map dreamerv3/contextual_mbrl to DALI. With DALI's copies FIRST
    on PYTHONPATH and cwd outside the repo, importing drift_eval must still yield this repo's stack."""

    def _run(self, code):
        """Child interpreter, cwd outside the repo, with DALI's dreamerv3 and contextual_mbrl FIRST on
        sys.path. DALI's repo root is NOT put on the path: it has its own `drift_eval` package, which
        would shadow this one and test nothing. A shadow dir holds only the two colliding packages
        (symlinks to DALI's copies), then this repo follows."""
        dali_d3 = os.path.join(DALI_REPO, "dreamerv3_compat", "dreamerv3")
        dali_cm = os.path.join(DALI_REPO, "contextual_mbrl")
        if not (os.path.isdir(dali_d3) and os.path.isdir(dali_cm)):
            self.skipTest(f"no DALI checkout at {DALI_REPO} to simulate the collision")
        shadow = tempfile.mkdtemp(dir=_TMP, prefix="dali_first_")
        os.symlink(dali_d3, os.path.join(shadow, "dreamerv3"))
        os.symlink(dali_cm, os.path.join(shadow, "contextual_mbrl"))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([shadow, REPO]))
        pre = subprocess.run([sys.executable, "-c",
                              "import dreamerv3, os; print(os.path.realpath(dreamerv3.__file__))"],
                             cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True,
                             timeout=600)
        self.assertIn(DALI_REPO, pre.stdout, "simulation does not put DALI's dreamerv3 first")
        return subprocess.run([sys.executable, "-c", code], cwd=tempfile.gettempdir(), env=env,
                              capture_output=True, text=True, timeout=600)

    def test_repo_stack_wins_over_dali_first_on_sys_path(self):
        p = self._run("import json, drift_eval\nfrom drift_eval.stack import verify_stack\n"
                      "print('STACK=' + json.dumps(verify_stack()))")
        line = [l for l in p.stdout.splitlines() if l.startswith("STACK=")]
        self.assertTrue(line, p.stderr[-1500:])
        prov = json.loads(line[0][len("STACK="):])
        for key in ("dreamerv3_file", "embodied_file", "contextual_mbrl_file", "agent_py"):
            self.assertTrue(prov[key].startswith(REPO + os.sep), (key, prov[key]))
        agent_py = os.path.join(REPO, "dreamerv3_compat", "dreamerv3", "agent.py")
        self.assertEqual(prov["agent_py_sha256"],
                         hashlib.sha256(open(agent_py, "rb").read()).hexdigest())

    def test_a_foreign_dreamerv3_imported_first_is_refused(self):
        p = self._run("import dreamerv3\nimport drift_eval\nprint('NOT_REFUSED')")
        self.assertNotIn("NOT_REFUSED", p.stdout)
        self.assertIn("was already imported from", p.stderr)
        self.assertIn(DALI_REPO, p.stderr)


def _probe_value(fid, e):
    """Distinct constant per schedule episode e, inside CWM's clip (index probe)."""
    return -40.0 + e if fid == 3 else 0.5 + 0.01 * e


def _mjdata_hash(m, d):
    import mujoco
    h = hashlib.sha256()
    kind = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    buf = np.empty(mujoco.mj_stateSize(m, kind), np.float64)
    mujoco.mj_getState(m, d, buf, kind)
    h.update(buf.tobytes())
    for name in ("qacc", "qacc_warmstart", "qfrc_bias", "qfrc_passive", "qfrc_actuator",
                 "actuator_moment", "xpos", "xquat", "cinert", "cvel", "ctrl", "xfrc_applied"):
        h.update(np.ascontiguousarray(getattr(d, name)).tobytes())
    return h.hexdigest()


class TestParity(unittest.TestCase):
    """Env parity with CWM's stack: items 1-5 of the verifier's cRSSM rows (PARITY_TABLE.md,
    2026-09-14; each fix confirmed bitwise there, table `crssmfix`), the schema items, and the
    read-back no-side-effect check."""

    def test_no_forward_between_factor_write_and_physics_step(self):
        """Item 1. CWM writes the factor and steps with no mj_forward (dmc_latent_factors.py:440/:451).
        Every IR-S step switches; forward() calls made from carl_intra_episode.py during decision steps
        must be 0 (unmasked, the wrapper calls it after every switch, :630-631)."""
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                env = build(bench, make_spec(bench, "ir_s"))
                rb = env._drift_parts["regime_b"]
                P = type(rb._physics)
                had_own = "forward" in P.__dict__
                orig = P.forward
                state = {"armed": False, "n": 0}

                def counting(self_, *a, **k):
                    if state["armed"] and sys._getframe(1).f_code.co_filename.endswith(
                            "carl_intra_episode.py"):
                        state["n"] += 1
                    return orig(self_, *a, **k)
                P.forward = counting
                try:
                    drive_episode(env, lambda: state.__setitem__("armed", True), max_steps=50)
                finally:
                    if had_own:
                        P.forward = orig
                    else:
                        delattr(P, "forward")
                    close(env)
                self.assertEqual(state["n"], 0)

    def test_bernoulli_gravity_values_are_cwm_table_times_981(self):
        """Item 2. Condition A draws gravity from CWM's table times 9.81 as CWM computes it
        (`v * -9.81`), not CARL's literal m/s^2 table (1 ulp apart)."""
        from drift_eval import cwm
        values = cwm.load_values_table()
        want = [float(v) * 9.81 for v in values["gravity"]["train_values"]]
        self.assertNotEqual(want, [3.924, 5.886, 9.810, 14.715, 19.620],
                            "CWM's product and CARL's literal no longer differ; the test is vacuous")
        for bench in ("walker_3f_phys", "quadruped_2f_mixed"):
            with self.subTest(bench=bench):
                env = build(bench, make_spec(bench, None))
                try:
                    got = [float(v) for v in env._drift_parts["regime_b"]._factor_configs["gravity"].values]
                finally:
                    close(env)
                self.assertEqual(got, want)

    def test_bernoulli_draws_follow_cwm_factor_id_order(self):
        """Item 3. CWM's DMControlLatentFactors, seeded with the env seed, draws lambda then the initial
        value per active factor in factor-id order at reset (dmc_latent_factors.py:356-375), and per
        decision step random() then choice() in the same order (:406-415). Replicated here and compared
        with the CARL wrapper's lambdas and applied values for 200 steps."""
        from drift_eval import benches as B, cwm
        values = cwm.load_values_table()
        for bench in ("quadruped_2f_phys", "walker_3f_phys"):
            with self.subTest(bench=bench):
                b = B.bench(bench)
                dom, fids, seed = b["domain"], list(b["factor_ids"]), 1008
                lo, hi = [float(x) for x in make_cfg(bench).env.carl.regime_b_lambda_range]
                names = [B.carl_name(dom, f) for f in fids]
                table = {f: [B.to_carl_units(f, v) for v in values[B.value_key(dom, f)]["train_values"]]
                         for f in fids}
                rng = np.random.RandomState(seed)
                lam = {f: rng.uniform(lo, hi) for f in fids}
                cur = {f: rng.choice(table[f]) for f in fids}
                env = build(bench, make_spec(bench, None, seed=seed))
                rb = env._drift_parts["regime_b"]
                bad, first = [], [True]

                def on_obs():
                    if first[0]:
                        first[0] = False
                        for f, n in zip(fids, names):
                            if rb._current_lambdas[rb._factor_names.index(n)] != lam[f]:
                                bad.append(("lambda", f))
                    else:
                        for f in fids:
                            if rng.random() < lam[f]:
                                cur[f] = rng.choice(table[f])
                    for f, n in zip(fids, names):
                        if float(rb._factor_values[rb._factor_names.index(n)]) != float(cur[f]) \
                                and len(bad) < 5:
                            bad.append(("value", f, int(rb._decision_step)))
                try:
                    self.assertEqual(rb._factor_names, names)
                    drive_episode(env, on_obs, max_steps=200)
                finally:
                    close(env)
                self.assertEqual(bad, [])

    def test_reward_switch_rng_is_seeded_with_env_seed(self):
        """Item 4. CWM seeds DMControlRewardFactors with the env seed (eval.py:118) and at reset draws
        lambda then the initial mode (dmc_reward_factors.py:514-522). The fork seeds seed+1000."""
        for bench in ("walker_2f_mixed", "quadruped_2f_mixed"):
            with self.subTest(bench=bench):
                seed = 1009
                cfg = make_cfg(bench)
                lo, hi = [float(x) for x in cfg.env.carl.regime_b_lambda_range]
                modes = [str(m) for m in cfg.env.carl.regime_b_reward_modes]
                env = build(bench, make_spec(bench, None, seed=seed))
                rs = env._drift_parts["reward_switch"]
                ref = np.random.RandomState(seed)
                got_state, ref_state = rs._rng.get_state(), ref.get_state()
                seen = {}
                try:
                    drive_episode(env, lambda: seen.setdefault(
                        "x", (float(rs._current_lambda), str(rs._current_mode))), max_steps=1)
                finally:
                    close(env)
                self.assertTrue(np.array_equal(got_state[1], ref_state[1]) and got_state[2] == ref_state[2])
                self.assertEqual(seen["x"], (float(ref.uniform(lo, hi)), str(ref.choice(modes))))

    def test_schedule_episode_index_is_env_seed_plus_ordinal(self):
        """Item 5. With a distinct constant path per schedule episode, env seed 1007 must read episode
        (1007 + j) % 64 on its j-th reset (47, 48), in the physics wrapper and the reward switcher."""
        from drift_eval import benches as B
        n, seed = 64, 1007
        for bench in ("walker_3f_phys", "quadruped_2f_mixed"):
            with self.subTest(bench=bench):
                b = B.bench(bench)
                dom, fids = b["domain"], list(b["factor_ids"])
                phys = {"episodes": [{f"physics_f{f}": [[t, _probe_value(f, e)] for t in range(501)]
                                      for f in fids} for e in range(n)],
                        "processes": {f"physics_f{f}": [_probe_value(f, e) for e in range(n)] for f in fids},
                        "schedule_hash": "index-probe", "n_episodes": n, "T": 500}
                rew, modes = None, None
                if b["mixed"]:
                    modes = sched(bench, "bench_spec")["reward_modes"]
                    rew = {"episodes": [{"reward": [[0, modes[e % 2]]]} for e in range(n)],
                           "schedule_hash": "index-probe-reward", "n_episodes": n, "T": 500}
                env = build(bench, make_spec(bench, "probe", seed=seed, strict=True, payloads=(phys, rew)))
                rb, rs = env._drift_parts["regime_b"], env._drift_parts["reward_switch"]
                got = []
                try:
                    for _ in range(2):
                        snap = {}
                        drive_episode(env, lambda: snap.setdefault("s", (
                            {f: physics_snapshot(rb, dom, f) for f in fids},
                            None if rs is None else str(rs._current_mode))), max_steps=1)
                        got.append(snap["s"])
                finally:
                    close(env)
                for j, (snaps, mode) in enumerate(got):
                    e = (seed + j) % n
                    for f in fids:
                        v = _probe_value(f, e)
                        self.assertTrue(snapshot_matches(snaps[f], f, v * 9.81 if f == 1 else v),
                                        (bench, f, j, e))
                    if modes is not None:
                        self.assertEqual(mode, modes[e % 2])

    def test_row_and_trace_fields_match_cwm_runner(self):
        """Schema items: CWM condition ids, env / experiment row fields, 5 envs per batch, the mixed
        trace's schedule_hash = reward schedule hash (physics in physics_schedule_hash), CWM reward-mode
        ids, ddof=0 std, and CWM's refusal of an unbalanced label plan."""
        from drift_eval import benches as B, cwm, outputs as O, run as R
        bench, cid = "walker_2f_mixed", "drift_ir_s"
        port = R.load_port_yaml()
        self.assertTrue({"phys_train", "drift_ir_a", "drift_ir_s", "drift_h_a", "drift_h_s", "drift_tau_1",
                         "drift_tau_8", "drift_tau_16", "drift_tau_64", "drift_tau_256"}
                        <= set(port["conditions"]))
        self.assertEqual(R.parse_args(["--bench", bench, "--seed", "1", "--schedules-dir", "d",
                                       "--results-root", "r"]).amount, 5)
        gen, gen_sha = cwm.load_generator(GENERATOR)
        values = cwm.load_values_table()
        cfg = make_cfg(bench)
        common = dict(cfg=cfg, bench_name=bench, cid=cid, conditions=port["conditions"],
                      schedules_dir=sched_dir(bench), gen=gen, gen_sha=gen_sha,
                      bench_spec=R.load_bench_spec(sched_dir(bench), bench, gen, gen_sha, cfg, values),
                      wind_clip=B.cwm_wind_clip(values, "walker"), trained_seed=1,
                      policy_factory=_fake_policy_factory, action_mode="native",
                      ckpt_info={"path": "none (CPU test)", "sha256": "none", "step": 0},
                      cfg_source="reconstructed", git_sha="test", cwm_hashes={}, render=False)
        with self.assertRaisesRegex(RuntimeError, "not balanced"):
            R.run_condition(n_episodes=5, amount=5, results_root=tempfile.mkdtemp(dir=_TMP), **common)
        res = tempfile.mkdtemp(dir=_TMP)
        out = R.run_condition(n_episodes=10, amount=5, results_root=res, **common)
        with open(out["raw"]) as f:
            raw = json.load(f)
        self.assertEqual(raw["condition_id"], "drift_ir_s")
        self.assertEqual(raw["metadata"]["env"], "dm_walker_walk")
        with open(os.path.join(res, "main_results.csv"), newline="") as f:
            row = next(csv.DictReader(f))
        self.assertEqual((row["env"], row["experiment"], row["condition_id"]),
                         ("dm_walker_walk", "walker_2f_mixed", "drift_ir_s"))
        self.assertEqual(float(row["std_return"]), float(np.std(np.asarray(raw["returns"]))))
        cell = O.load_trace_cell(out["trace"])
        meta = cell["_meta"]
        self.assertEqual(meta["schedule_hash"], sched(bench, "reward_fixed")["schedule_hash"])
        self.assertEqual(meta["physics_schedule_hash"], sched(bench, "ir_s")["schedule_hash"])
        self.assertEqual((meta["env"], meta["experiment"]), ("dm_walker_walk", "walker_2f_mixed"))
        labels = [ep["reward"][0][1] for ep in sched(bench, "reward_fixed")["episodes"]]
        plan = [labels[(1007 + k % 5 + k // 5) % len(labels)] for k in range(10)]   # CWM _drift_label_plan
        self.assertEqual(meta["reward_label_plan"], plan)
        ids = {"walk_backward": 5, "walk_forward": 6}
        self.assertEqual(cell["reward"].shape, (10, 500))
        for k in range(10):
            self.assertTrue(np.all(cell["reward_mode_id"][k] == ids[plan[k]]), (k, plan[k]))

    def test_readback_does_not_mutate_physics(self):
        """The recorder's read-back leaves the full MjData state and the derived arrays a forward()
        would rewrite unchanged; the control (mj_forward on a copy) must change the same hash."""
        import copy
        import mujoco
        from drift_eval import envbuild
        for bench in ("walker_3f_phys", "quadruped_2f_phys"):
            with self.subTest(bench=bench):
                env = build(bench, make_spec(bench, "ir_s", strict=True))
                rb = env._drift_parts["regime_b"]
                _, _, fg = envbuild._find(envbuild.walk_chain(env), "FromGymnasium")
                rec = fg._env
                orig = rec._read_and_check
                st = {"reads": 0, "changed": 0, "control": 0, "detected": 0}

                def checked(t):
                    m, d = rb._physics.model.ptr, rb._physics.data.ptr
                    before = _mjdata_hash(m, d)
                    out = orig(t)
                    st["reads"] += 1
                    st["changed"] += int(_mjdata_hash(m, d) != before)
                    if t >= 1 and st["control"] < 20:
                        d2 = copy.copy(d)
                        h0 = _mjdata_hash(m, d2)
                        mujoco.mj_forward(m, d2)
                        st["control"] += 1
                        st["detected"] += int(_mjdata_hash(m, d2) != h0)
                    return out
                rec._read_and_check = checked
                try:
                    drive_episode(env, max_steps=60)
                finally:
                    close(env)
                self.assertGreaterEqual(st["reads"], 61)
                self.assertEqual(st["changed"], 0)
                self.assertGreater(st["control"], 0)
                self.assertEqual(st["detected"], st["control"])


CWM_JOIN_COMMIT = "befabde38bfea2e9d150fa2da41facf738c49d59"
_CWM_JOIN_FUNCS = ("_notes_with_paired_streams", "_paired_streams_from_record",
                   "_join_raw_json_to_traces", "_join_traces_to_returns")


def _cwm_join_functions():
    """OUR runner's keyed-join path at CWM befabde (benchmark/run_eval_benchmark.py:
    `_join_raw_json_to_traces` :651 -> `_paired_streams_from_record` :632 + `_join_traces_to_returns`
    :659 + `_notes_with_paired_streams` :615). The module imports torch, absent from this interpreter,
    so these functions' own source is executed as written. Source = the first that holds all four,
    both based on CAUSAL_WORLD_MODEL_ROOT only: the file under that root, else `git -C <root> show
    befabde:`. Returns (namespace, ast of _join_traces_to_returns, source) or None."""
    import ast
    import re
    rel = "benchmark/run_eval_benchmark.py"
    candidates = []
    path = os.path.join(CWM_ROOT, rel)
    if os.path.exists(path):
        candidates.append((f"file {path}", lambda: open(path).read()))
    candidates.append((f"git -C {CWM_ROOT} show {CWM_JOIN_COMMIT[:7]}:{rel}",
                       lambda: subprocess.run(
                           ["git", "-C", CWM_ROOT, "show", f"{CWM_JOIN_COMMIT}:{rel}"],
                           capture_output=True, text=True).stdout))
    for desc, get in candidates:
        try:
            tree = ast.parse(get() or "")
        except (OSError, SyntaxError):
            continue
        fns = {n.name: n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name in _CWM_JOIN_FUNCS}
        if len(fns) != len(_CWM_JOIN_FUNCS):
            continue
        ns = {"re": re}
        exec(compile(ast.Module(body=[fns[n] for n in _CWM_JOIN_FUNCS], type_ignores=[]), desc, "exec"), ns)
        return ns, fns["_join_traces_to_returns"], desc
    return None


class TestKeyedJoin(unittest.TestCase):

    def test_trace_rows_join_returns_by_key_with_cwm_join(self):
        """A real port output (walker_3f_phys, drift_ir_s, 7 episodes over 5 envs: 2 rounds, 3 unjoined)
        is joined with CWM's own `_join_raw_json_to_traces` @befabde (which validates paired_streams and
        calls `_join_traces_to_returns`): the trace rows, rebuilt from the cell arrays and shuffled, are
        keyed by (episode_worker, episode_ordinal) against the raw JSON's episode_keys; every joined
        row's reward sum must be its return. Dropping or permuting a key array breaks the join."""
        import ast
        import inspect
        from drift_eval import benches as B, cwm, outputs as O, run as R
        found = _cwm_join_functions()
        if found is None:
            self.skipTest(f"CAUSAL_WORLD_MODEL_ROOT={CWM_ROOT} neither contains nor can `git show` "
                          f"CausalWorldModel commit {CWM_JOIN_COMMIT}'s keyed join "
                          f"(benchmark/run_eval_benchmark.py _join_raw_json_to_traces)")
        cwm_ns, cwm_fn, source = found
        print(f"\n[join] CWM keyed join taken from: {source}")
        port_fn = next(n for n in ast.parse(inspect.getsource(O)).body
                       if isinstance(n, ast.FunctionDef) and n.name == "join_traces_to_returns")
        self.assertEqual(ast.dump(port_fn.args), ast.dump(cwm_fn.args))
        self.assertEqual([ast.dump(s) for s in port_fn.body[1:]], [ast.dump(s) for s in cwm_fn.body[1:]],
                         "the port's join is not CWM's join")

        bench, cid = "walker_3f_phys", "drift_ir_s"
        gen, gen_sha = cwm.load_generator(GENERATOR)
        values = cwm.load_values_table()
        cfg = make_cfg(bench)
        res = tempfile.mkdtemp(dir=_TMP)
        out = R.run_condition(
            cfg=cfg, bench_name=bench, cid=cid, conditions=R.load_port_yaml()["conditions"],
            schedules_dir=sched_dir(bench), gen=gen, gen_sha=gen_sha,
            bench_spec=R.load_bench_spec(sched_dir(bench), bench, gen, gen_sha, cfg, values),
            wind_clip=B.cwm_wind_clip(values, "walker"), trained_seed=1, n_episodes=7, amount=5,
            policy_factory=_fake_policy_factory, action_mode="native", results_root=res,
            ckpt_info={"path": "none (CPU test)", "sha256": "none", "step": 0},
            cfg_source="reconstructed", git_sha="test", cwm_hashes={}, render=False)
        with open(out["raw"]) as f:
            raw = json.load(f)
        cell = O.load_trace_cell(out["trace"])
        meta = cell["_meta"]
        reason = "not_applicable: port has no first-action sampler"
        for md in (raw["metadata"], meta):
            self.assertIn("paired_streams", md)
            self.assertIsNone(md["paired_streams"])                 # null, never true/false
            self.assertEqual(md["paired_streams_reason"], reason)
        self.assertEqual(meta["row_join"], "by (episode_worker, episode_ordinal) to raw returns")
        self.assertEqual(meta["unjoined_worker_traces"], 3)
        returns = raw["returns"]
        self.assertEqual(len(returns), 7)
        gaps = np.diff(np.sort(np.asarray(returns)))
        self.assertTrue(np.all(gaps > 1e-6), "returns are not distinct; a mis-join would go unseen")

        rows = [{"worker_tag": f"w{int(cell['episode_worker'][r])}",
                 "episode_ordinal": int(cell["episode_ordinal"][r]),
                 "reward": cell["reward"][r], "episode_return": float(cell["episode_return"][r])}
                for r in range(len(returns))]
        shuffled = [rows[k] for k in np.random.RandomState(0).permutation(len(rows))]
        try:
            joined, unjoined, notes = cwm_ns["_join_raw_json_to_traces"](raw, shuffled, "", "", cid)
        except SystemExit as exc:
            self.fail(f"CWM's join refused the port output: {exc}")
        self.assertEqual(unjoined, 0)
        self.assertEqual(notes, "paired_streams=null (not_applicable: port has no first-action sampler)")
        for i, tr in enumerate(joined):
            self.assertAlmostEqual(float(np.sum(tr["reward"].astype(np.float64))), returns[i], delta=1e-6,
                                   msg=f"row joined to returns[{i}] carries another episode's rewards")
            self.assertEqual(np.float32(tr["episode_return"]), np.float32(returns[i]))


class TestActionMode(unittest.TestCase):

    def test_greedy_mode_reads_the_actor_mode_not_a_sample(self):
        """JAX on CPU, randomly initialised cRSSM agent (walker_3f_phys config).

        The RSSM posterior latent is itself SAMPLED in obs_step (dreamerv3/nets.py:142, nj.rng()), so
        no mode makes the whole policy deterministic. Under the same policy RNG seed both modes draw
        the same latent (it is the first nj.rng() use), so any action difference is the actor readout:
        'eval_greedy' must return the distribution mode (inside (-1, 1)) and differ from 'eval''s
        sample; a greedy branch that samples would return exactly the 'eval' action."""
        import dreamerv3
        from dreamerv3 import embodied
        bench = "walker_3f_phys"
        cfg = make_cfg(bench).update({"jax.platform": "cpu", "jax.prealloc": False,
                                      "jax.policy_devices": [0], "jax.train_devices": [0]})
        env = build(bench, make_spec(bench, None))
        try:
            agent = dreamerv3.Agent(env.obs_space, env.act_space, embodied.Counter(), cfg)
            obs = env.step({"action": np.zeros(env.act_space["action"].shape, np.float32),
                            "reset": True})
            batch = {k: np.asarray(v)[None] for k, v in obs.items()}

            def act(mode, seed):
                agent.rng = np.random.default_rng(seed)
                outs, state = agent.policy(batch, None, mode=mode)
                latent = state[0][0]
                return np.asarray(outs["action"]), outs, np.asarray(latent["stoch"])
            results = []
            for seed in (1, 2, 3):
                g, go, g_lat = act("eval_greedy", seed)
                s, _, s_lat = act("eval", seed)
                results.append((g, s, g_lat, s_lat))
                self.assertIn("reward_hat", go)
        finally:
            close(env)
        for g, s, g_lat, s_lat in results:
            np.testing.assert_array_equal(g_lat, s_lat)      # same latent under the same seed
            self.assertFalse(np.array_equal(g, s))           # greedy is not the sample
            self.assertTrue(np.all(np.abs(g) < 1.0))         # tanh(mean)


if __name__ == "__main__":
    unittest.main()
