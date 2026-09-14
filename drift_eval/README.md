# drift_eval — gradual-shift (§8b) evaluation of cRSSM checkpoints

Runs CausalWorldModel's (CWM) gradual-shift protocol on the cRSSM baseline checkpoints and writes
CWM's output schema, so CWM's readers consume cRSSM rows unchanged.

- **Protocol.** Walker: ICLR plan §8b (design of record) and §8c (logging), CWM `docs/ICLR27/ICLR_PAPER_PLAN.md:916-1007`.
  **Quadruped:** a mirror of the walker design ruled by trajd-advisor 2026-09-14, pending PI
  ratification. It is not in §8b.
- **Schedules.** Taken only from CWM's generator (`benchmark/drift_generator.py`), **vendored
  unmodified** at `drift_eval/vendor/drift_generator.py` (CWM commit
  `befabde38bfe…`, sha256 `39b72cea…61e09f6`; see `vendor/VENDORED.md`). The vendored copy is the
  default. `--generator` / `DRIFT_GENERATOR` is an optional override. The sha256 is asserted at load
  and written into every output file. `make_schedules.py` runs the generator's own `main()`
  unchanged, after pointing its preset root at the verified CWM tree. The port never re-implements
  the generator.
- **Factor semantics, wrappers and presets: an EXTERNAL CWM tree, not vendored.** It must be at CWM
  commit **45e8d5d or later with the byte-identical files listed below**, pointed to by the environment
  variable `CAUSAL_WORLD_MODEL_ROOT` (default `/home/jovyan/m10_code/45e8d5d`; the launcher also
  pins `CWM_SHA_PIN`). Every output records that root's commit and dirty-file count (`cwm_commit`,
  `cwm_dirty_count`, or "not a git checkout") and the sha256 of `config_files/experiments/dmc.yaml`,
  `config_files/configure.yaml` and `config_files/latent_factor_values.yaml` (`cwm_config_sha256`). `carl_intra_episode.py`, `carl_reward_switch.py`,
  `dmc_reward_factors.py`, `latent_factor_values.yaml` and `switch_recovery.py` are hash-checked
  against their 45e8d5d bytes, and the run fails closed on a mismatch (`cwm.py`).

## Conditions (`drift_eval.yaml`): ids are CWM's

Condition ids are CWM's own (`benchmark/eval_conditions.yaml:143/:154` @befabde, families `dmc_drift_*`, and
`phys_train`), so no consumer code changes. Plan letters map to them:

| plan letter | condition id (CWM) | schedule | what |
|---|---|---|---|
| A | `phys_train` | none | training protocol: regime B, train split, Bernoulli switching, λ ~ U[0.003, 0.008] per episode |
| B | `drift_ir_a` | `ir_a.json` | in-range matched abrupt (band permutation of IR-S, same TV; 5 jumps per factor) |
| C | `drift_ir_s` | `ir_s.json` | in-range gradual log-sine, TV = training budget B_f |
| D | `drift_h_a` | `h_a.json` | hard-excursion matched abrupt |
| E | `drift_h_s` | `h_s.json` | hard-excursion gradual half-cycle |
| F | `drift_tau_<k>`, k ∈ {1,8,16,64,256} | `tau_<k>.json` | τ-dial (optional; not in the default set) |

On mixed benches (`walker_2f_mixed`, `quadruped_2f_mixed`), B–F add the generator's
`reward_fixed.json`: one reward label per episode, fixed within the episode. Env i (seed base+i)
reads schedule episode `(base + i + j) % n` for its j-th episode, in both wrappers, as CWM's
`_sched_offset` does. The retained episodes must be balanced across the modes, or the run refuses,
as CWM's `_resolve_schedule_modes` does. In A, the native Bernoulli reward switching runs.

**Reward-mode ids are CWM's:** each is the index of the mode in the SORTED domain reward registry
(`envs/dmc_reward_factors.py:359/:432`). CARL's switcher index (0/1) is not written.

| domain | mode | reward_mode_id |
|---|---|---|
| walker | walk_backward | 5 |
| walker | walk_forward | 6 |
| quadruped | go_north | 3 |
| quadruped | go_south | 4 |

**Row fields are CWM's:** `env` = `dm_walker_walk` / `dm_quadruped_walk` and `experiment` = the
bench name (e.g. `walker_3f_phys`). On a mixed drift trace, `schedule_hash` is the reward-label
schedule's hash, with the physics hash in `physics_schedule_hash` and the plan in
`reward_label_plan` (`run_eval_benchmark.py:1224-1237` @befabde). `std_return` uses ddof=0.
**Envs per batch: 5** (`--amount` default = CWM `Evaluate.NumEnvs`), so the CRN episode plan
(k = j·5 + i, first N kept) is CWM's.

**Keyed join (CWM befabde fields, same names):** the raw JSON carries top-level `episode_keys`
(`[worker, ordinal]` of each return) and `metadata.paired_streams`. The trace cell carries
`episode_worker`, `episode_ordinal` and `episode_return` (row i IS returns[i]). The trace meta carries
`paired_streams`, `row_join` and `unjoined_worker_traces`. Traces are joined to returns by
(worker, ordinal) with an exact mirror of CWM `run_eval_benchmark._join_traces_to_returns`
(`:659` @befabde; a test checks the AST is identical and runs CWM's `_join_raw_json_to_traces` `:651`,
which also validates `paired_streams`, on a port output).
`paired_streams` is null (never true/false, advisor ruling 2026-09-14), with sibling
`paired_streams_reason: "not_applicable: port has no first-action sampler"` in the raw metadata and
the trace meta: this stack has no parent `action_space.sample` stream for CWM's
PairedStreams to seed.

## Checkpoints (cRSSM, HF `ssubhnil/cwm` `crssm/<slug>/seed<N>/checkpoint.ckpt`, local `/home/jovyan/baseline_ckpts/crssm`)

Step is the pickled step counter inside each file. Each local sha256 equals the HF LFS sha256
(checked 2026-09-14 for every file below; the inventory is in `drift_eval.yaml`).

| bench | slug | seeds at step 1,000,000 | other seeds | config.yaml |
|---|---|---|---|---|
| walker_3f_phys (w3f) | dmc_walker_k3 | 1 2 3 4 5 | — | in logdir |
| walker_2f_phys (w2fphys) | dmc_walker_k2 | 1 2 3 4 5 | — | in logdir |
| walker_2f_mixed (w2fmix) | dmc_walker_2f_mixed | 1 2 3 4 5 | — | in logdir |
| quadruped_2f_phys | dmc_quadruped_k2 | **2 only** | s1 395,300 · s3 296,100 · s4 27,700 · s5 418,600 (refused unless `--allow-partial-ckpt`) | none: reconstructed |
| quadruped_2f_mixed | dmc_quadruped_2f_mixed | 1 2 3 4 5 | — | none: reconstructed |

No cRSSM checkpoint is at 950k. The older `dmc_quadruped_mixed` slug (steps 8,800 to 1M, trained
2026-04-07..09) is not used. The walker `config.yaml` files are dated after the checkpoints and match
CWM `benchmark/reconstruct_crssm_config.py`'s recipe (inferred from the file dates and that script's
docstring). Where no `config.yaml` exists, `config.py` rebuilds the config from the fork's own
`configs.yaml` (defaults + `carl` + the bench preset). Either way, context dim K, action dim and
observation dim are checked against the checkpoint's tensor shapes before anything runs.

## Run

```bash
source /home/jovyan/m10_code/45e8d5d/benchmark/devbox_dali_env.sh   # hardware EGL; OSMesa is refused
BENCH=walker_3f_phys SMOKE=1 bash drift_eval/launch_crssm_drift.sh  # GPU smoke: 1 seed, 10 episodes x 5 envs per condition A-E
BENCH=walker_3f_phys bash drift_eval/launch_crssm_drift.sh          # battery: all 1M seeds, A-E, 64 episodes, 5 envs
```

The launcher refuses to start unless all of these hold: CWM root is a clean 45e8d5d, the generator
sha256 is as expected, `MUJOCO_GL=egl`, and the checkpoint sha256 is in the inventory. It writes the
schedules with the generator CLI (`--bench`), writes `provenance.txt`, and runs
`python -m drift_eval.run` once per seed.
Interpreter: `/home/jovyan/envs/dali/bin/python`, with `PYTHONPATH=<repo>:<repo>/dreamerv3_compat` so
this fork shadows DALI's editable installs.

**GPU need:** one GPU per process, used for the JAX policy (float16) and for EGL rendering of the
fork's `RenderImage` wrapper. The 5 envs run in-process. Memory use is unmeasured.
**Wall-clock:** to be measured on the GPU smoke; `wallclock_per_episode_s` is recorded in every raw
JSON. On CPU without a policy, one 500-step walker episode through the full chain took roughly
0.5–3 s in the tests. That is not an estimate of GPU wall-clock.

## Outputs (`<results_root>/`; `drift_eval_results/` is git-ignored)

- `raw/<row_id>__<cid>.json` has exactly the top-level keys of CWM
  `run_eval_benchmark._write_raw_json` (`benchmark/run_eval_benchmark.py:111-133`): `row_id`,
  `condition_id`, `returns`, `reward_mse_per_episode`, `n_episodes`, and `metadata{seed, env,
  git_sha, eval_timestamp_utc}`. Port provenance is added inside `metadata` only: action mode,
  generator path and sha256, schedule hash and file sha256, checkpoint step and sha256, config
  source, env seeds, renderer, CWM file hashes, and wall-clock. `row_id = crssm_<bench>_s<seed>`.
  CWM's older embodied runner (`benchmark/eval_shared.write_raw_json`) uses a different schema
  (`raw_returns`); this port follows the PyTorch runner.
- `main_results.csv` has CWM's `CSV_COLUMNS`; `notes` carries the action mode, generator sha256,
  schedule hash and checkpoint step.
- `traces/<row_id>__<cid>__trace.npz` + `.meta.json` is CWM's D1 cell (`reward[N,T]`,
  `reward_mode_id[N,T]`, `factor_values[N,T,4]` in CWM units with NaN for inactive factors,
  `factor_active[4]`, `lambda_sampled[N]`, `schedule_id[N]`), written by CWM's
  `switch_recovery.write_trace_cell`. Additive arrays: `decision_step`, `reset_factor_values`,
  `reset_reward_mode_id`, `actions`, `reward_hat`, `returns`, `episode_index`, `worker`,
  `episode_in_worker`, `env_seed`. `factor_values` are read back from MuJoCo after each step, not
  taken from the wrapper's bookkeeping.

## What is enforced (each has a CPU test that fails under the defect it guards)

- The payload travels inside the env constructor and is installed in the process that builds and
  steps the env, before the first reset. This avoids CWM's DMC spawn defect: a module global set in
  the parent is None in a spawned child.
- Scheduled mode with no physics payload (or, on mixed benches, no reward label) raises before the
  env exists. A Bernoulli condition refuses a payload, clears leftovers from earlier conditions in
  the process, and raises if a schedule is present at any reset.
- At every decision step of a scheduled episode, the physics must hold exactly the schedule value
  after CWM's clip. The recorder raises otherwise. The recorder also raises if the wrapper module
  holds a different payload hash, on a PhysicsError retry, when an episode length is not T = 500,
  when driver and recorder rewards are misaligned, or when a mixed-bench reward label differs.
- **Clip, as CWM's applier does it.** In B–F every schedule value first goes through CWM's clip in
  CWM units, with the same bounds and operation (`float(np.clip(v, 0.1, 10.0))` for ids 0–2,
  `float(np.clip(v, -wc, wc))` for wind, `dmc_latent_factors.py:_apply_factor`). Only then is it
  converted multiplier → m/s² for gravity (×9.81; `benches.py` is the only place). Example: the
  generator's quadruped H-S joint_damping 10.000000000000002 is applied as 10.0. A test runs CWM's
  real `_apply_factor` on a stub and requires bitwise equality at every step.
- **Native cRSSM clips the schedule bypasses (B–F only; A keeps them).** The CARL wrapper's
  `clip_range` is set to CWM's bounds in CARL units. Only the gravity clip actually changes an
  applied value: CARL's is (0.98, 39.24) m/s² = (0.0999, 4.0)×. The H-S/H-A gravity path reaches
  5.126× = 50.3 m/s², on every bench (all five drift gravity), so CARL's upper gravity clip is
  bypassed and CWM's 10× applies. CARL's actuator_strength / joint_damping clip (0.05, 15) and walker
  wind clip ±150 N are wider than CWM's (0.1, 10) and ±120 N. The schedule never reaches them after
  CWM's clip.
- The generator's `bench_spec.json` must match the loaded generator sha256, CWM's value table, the
  checkpoint config's `regime_b_lambda_range`, and, on mixed benches, `regime_b_reward_modes`.

## Action mode

- **Native (default):** the fork samples at eval. `agent.policy(mode="eval")` draws
  `outs["action"].sample(seed=nj.rng())` (`dreamerv3_compat/dreamerv3/agent.py:86-88` at 7124ac2);
  CWM's embodied runner calls `mode="eval"` (`benchmark/eval_embodied.py:298` @45e8d5d).
- **Greedy (`--action-mode greedy`):** added in this commit as `mode="eval_greedy"`, which returns the
  actor distribution's `.mode()` (tanh(mean) for the `normal` actor). It is opt-in and leaves `eval`
  unchanged. The mode that ran is recorded in every raw JSON, CSV row and trace meta.
- **Neither mode is a deterministic policy.** The RSSM posterior latent is sampled at every step
  (`dreamerv3_compat/dreamerv3/nets.py:142`, `dist.sample(seed=nj.rng())` in `obs_step`). Greedy
  removes the actor's sampling noise only. Making the latent a mode as well is not a one-line change
  and was not made. Whether CWM's greedy eval samples its own latent was not checked.

## Footing: every axis on which this port differs from CWM's battery

| axis | CWM battery (runner @45e8d5d, §8b/§8c) | cRSSM port |
|---|---|---|
| checkpoint step | terminal 950k (`*_final`) | 1,000,000 (step counter); quadruped_2f_phys: only seed 2 |
| trained seeds | 3 (s1–3) | 5 per bench, except quadruped_2f_phys (1); seed identities do not correspond across methods |
| episodes per condition per seed | 64, CRN-paired, NumEnvs 5 | 64 by default (`--n-episodes`), 5 envs per batch, first 64 of 13 rounds kept (CWM's plan) |
| factor application timing | factor written, then `physics.step`, no `mj_forward` | matched (see the footing line below the table) |
| Bernoulli draws (A) | RNG draws in factor-id order; reward RNG seeded with env seed; gravity values `v*9.81` | matched: CARL's name order, seed+1000 and literal gravity table replaced (`envbuild.match_cwm_semantics`) |
| schedule episode index | `(env_seed + ordinal) % n` | matched (CARL's `ordinal % n` overridden in both wrappers) |
| action selection | greedy (`eval.py:277` default `greedy=True`; runner passes no `greedy`, `run_eval_benchmark.py:851`) | **sampled** by default: the fork's eval mode draws `.sample()` (`agent.py:86-88` at 7124ac2). **Greedy flag added** (`--action-mode greedy` → `mode="eval_greedy"` → `.mode()`, a one-branch change in `agent.py`); the mode that ran is recorded in every output |
| model input | pixels 64×64 | featurized state (walker 24-d, quadruped 78-d) **plus the oracle physics context** in `obs["context"]`, rewritten each step from the applied values, so cRSSM observes the drift path |
| renderer | hardware EGL; pixels are the input | hardware EGL renders `image`, which the cRSSM encoder does not read (`cnn_keys: $^`) |
| decisions per episode | 500 (action repeat 2) | 500: CARL episodes are 1000 control steps (measured, walker and quadruped) and the regime-B wrapper repeats each action twice (`carl_intra_episode.py:709/780` default, not overridden by `envs.py`) |
| gravity unit inside the env | multiplier of 9.81 | m/s²; converted once (`benches.py`); all outputs in CWM units |
| clip | CWM: `np.clip(v, 0.1, 10.0)`, wind ±max\|table\|, in CWM units, then applied | A: CARL native; B–F: CWM's clip, same bounds and order, then unit conversion; CARL's upper gravity clip (4.0×) bypassed |
| factor application | `DMControlLatentFactors` (nominal = XML snapshot) | CWM's CARL regime-B wrapper (nominal = CARL context divided out, D3 fix); wind re-asserted before each inner step |
| env seed | `Seed*1000+7`; worker i gets +i; dm_control task RNG seeded | same base and +i. CARL does not seed the dm_control task RNG natively (measured: two envs reset with seed 4007 differ); the port passes a persistent `RandomState(env_seed)` into CARL's env rebuilds |
| PairedStreams | seeds the vector env's `action_space` streams | not applicable (no `action_space.sample`). The policy's JAX key stream is reset to `default_rng(env_seed_base)` per condition, but sampled actions diverge once observations do; CWM's greedy policy has no action noise |
| env parallelism | `AsyncVectorEnv(context="spawn")`, 5 envs | in-process `BatchEnv` of 5 (`envs.parallel: none`); `--env-parallel process` available |
| reward label on mixed benches (B–F) | **no stack reference: CWM's runner refuses drift on reward-factor families at 45e8d5d (`run_eval_benchmark.py:799-803`)** | the generator's `reward_fixed.json`, one label per episode |
| condition A reward switching (mixed) | CWM training protocol | fork default: reward λ = physics λ range (`envs.py:846-850`); not compared with CWM's A |
| unmanaged physics | CWM nominal | CARL defaults; unmanaged multiplicative factors forced to nominal by the wrapper |
| config provenance | `ckpt/config.json` | walker `config.yaml` in logdir (a reconstruction); quadruped rebuilt at runtime; both shape-checked |
| protocol status | walker §8b design of record | walker same; quadruped mirror pending PI |

Factor application timing matched to CausalWorldModel `envs/dmc_latent_factors.py:440/:451` (factor written, then `physics.step` with no `mj_forward`: gear, gravity and damping lag one physics substep of each decision). The DALI/cRSSM checkpoints were trained through CausalWorldModel's `benchmark/wrappers/carl_intra_episode.py:631`, which calls `forward()` after every switch, so they are evaluated one physics substep off their training timing (measured 7.0e-6 qpos at the first switch, zero actions, w3f, seed 1; return effect on trained policies unmeasured). TrajD/DRAMA were trained on the no-forward timing (`train.py:793` → `envs/dmc_latent_factors.py:440/:451`), so each model family is evaluated on ours; only the comparators are off their training timing.

## Which dreamerv3 gets loaded

The shared venv `/home/jovyan/envs/dali` has setuptools editable finders
(`__editable__.dreamerv3-1.5.0.pth`, `__editable__.contextual_mbrl-0.0.0.pth`) that map `dreamerv3`
and `contextual_mbrl` to `/home/jovyan/projects/DALI`. Without this repo on PYTHONPATH (for example
`python -m unittest discover -s drift_eval/tests -t .`), `import dreamerv3` loaded DALI's agent. That
made the greedy test fail with `AttributeError: use_ctx_encoder`. Importing `drift_eval` now moves this
repo's `dreamerv3_compat` and root to the front of `sys.path`. It refuses to continue if a foreign
copy was already imported. `stack.verify_stack()` then asserts `dreamerv3`, `embodied` and
`contextual_mbrl` resolve under this repo before any env or agent is built. Every output records
`dreamerv3_file` and `agent_py_sha256`.

## Pre-existing defects found while porting (not fixed outside this directory)

- **Module identity.** CWM `benchmark/eval_embodied.py:404` installs `--factor-schedule-file` on
  `wrappers.carl_intra_episode` (benchmark/ on sys.path). The fork's `envs.py:863-904` builds the
  env from `benchmark.wrappers.carl_intra_episode`. These are two module objects with separate
  schedule globals (checked: `A is B` is False under the fork's interpreter). An embodied-runner
  drift row may therefore have run unscheduled (inferred; no such run was checked). This port
  installs and verifies through the module object of the wrapper instance.
- CARL's `reset(seed=)` does not fix initial states (see the footing table).

## Tests (CPU only, no rendering)

```bash
CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
XLA_FLAGS=--xla_force_host_platform_device_count=1 PYTHONPATH=$PWD:$PWD/dreamerv3_compat \
/home/jovyan/envs/dali/bin/python -m unittest drift_eval.tests.test_drift_eval -v
```

The fork's interpreter has no pytest, so the tests use unittest. They build the fork's real env
chain with `RenderImage` bypassed. The one output-schema reference is a real CWM raw JSON
(`CausalWorldModel/docs/trajd/a2c_reads_2026-09-11/ood_rec_c2wm_d1_950k_w2fmix/raw/d1_w2fmix_c2wm_s1__lambda_0.05.json`,
sha256 `384006…977125`), used when present. No CWM per-step trace file exists on the PVC, so the
trace cell is validated against CWM's writer code.

Not run here, GPU smoke only: checkpoint loading and a real policy rollout; the renderer;
wall-clock and memory.
