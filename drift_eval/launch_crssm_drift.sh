#!/usr/bin/env bash
# drift_eval/launch_crssm_drift.sh — gradual-shift (§8b) battery on cRSSM checkpoints, one bench.
#
#   BENCH=walker_3f_phys bash drift_eval/launch_crssm_drift.sh              # every 1M seed, A-E, 64 ep
#   BENCH=walker_3f_phys SMOKE=1 bash drift_eval/launch_crssm_drift.sh      # GPU smoke: 1 seed, 10 ep/cond, 5 envs
#
# BENCH: walker_3f_phys | walker_2f_phys | walker_2f_mixed | quadruped_2f_phys | quadruped_2f_mixed
# Optional: SEEDS="1 2 3", CONDITIONS="phys_train drift_ir_a drift_ir_s drift_h_a drift_h_s drift_tau_1 ...",
#           ACTION_MODE=native|greedy, N_EPISODES=64, AMOUNT=5 (CWM Evaluate.NumEnvs), RESULTS_ROOT=..., CKPT_ROOT=...
#
# Needs ONE GPU (JAX policy + hardware-EGL rendering of the fork's RenderImage wrapper). On cwm-devbox
#   source /home/jovyan/m10_code/45e8d5d/benchmark/devbox_dali_env.sh
# first (MUJOCO_GL=egl, vendored glvnd, XLA no-prealloc). OSMesa is refused: TensorFlow is importable
# in this interpreter and SIGSEGVs next to Mesa llvmpipe. Wall-clock: to be measured on the GPU smoke.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRSSM_DIR="$(dirname "$HERE")"
: "${BENCH:?set BENCH (walker_3f_phys walker_2f_phys walker_2f_mixed quadruped_2f_phys quadruped_2f_mixed)}"

CWM_ROOT="${CAUSAL_WORLD_MODEL_ROOT:-/home/jovyan/m10_code/45e8d5d}"
CWM_SHA_PIN="${CWM_SHA_PIN:-45e8d5d26d43ff41e7c4755b8c3f3d925738bbd0}"
# Generator: the vendored copy (drift_eval/vendor/VENDORED.md) unless DRIFT_GENERATOR overrides it.
GENERATOR="${DRIFT_GENERATOR:-$HERE/vendor/drift_generator.py}"
GENERATOR_SHA256="${DRIFT_GENERATOR_SHA256:-39b72cea67126f781605a2ebf7c658fa316ed380e9e2ee3badec132c561e09f6}"
PY="${CRSSM_PYTHON:-/home/jovyan/envs/dali/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/home/jovyan/baseline_ckpts/crssm}"
RESULTS_ROOT="${RESULTS_ROOT:-$CRSSM_DIR/drift_eval_results/$BENCH}"
CONDITIONS="${CONDITIONS:-phys_train drift_ir_a drift_ir_s drift_h_a drift_h_s}"
ACTION_MODE="${ACTION_MODE:-native}"
N_EPISODES="${N_EPISODES:-64}"
AMOUNT="${AMOUNT:-5}"

case "$BENCH" in
  walker_3f_phys|walker_2f_phys|walker_2f_mixed|quadruped_2f_mixed) DEFAULT_SEEDS="1 2 3 4 5" ;;
  quadruped_2f_phys) DEFAULT_SEEDS="2" ;;   # only seed 2 reached step 1000000 (drift_eval.yaml)
  *) echo "REFUSING: unknown BENCH=$BENCH"; exit 2 ;;
esac
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
if [ "${SMOKE:-0}" = "1" ]; then
  # CWM's smoke rule: 10 episodes (even, a multiple of NumEnvs 5), never 1 or 3 (balanced labels)
  SEEDS="${SEEDS%% *}"; N_EPISODES=10; AMOUNT=5; RESULTS_ROOT="${RESULTS_ROOT}_smoke"
fi

[ "${MUJOCO_GL:-}" = "egl" ] || { echo "REFUSING: MUJOCO_GL=${MUJOCO_GL:-unset}; source devbox_dali_env.sh (hardware EGL) first"; exit 3; }
[ -x "$PY" ] || { echo "REFUSING: no interpreter at $PY"; exit 3; }
( cd "$CWM_ROOT" && [ "$(git rev-parse HEAD)" = "$CWM_SHA_PIN" ] && [ -z "$(git status --porcelain)" ] ) \
  || { echo "REFUSING: $CWM_ROOT is not a clean checkout of $CWM_SHA_PIN"; exit 4; }
got="$(sha256sum "$GENERATOR" | cut -d' ' -f1)"
[ "$got" = "$GENERATOR_SHA256" ] || { echo "REFUSING: generator sha256 $got != $GENERATOR_SHA256"; exit 5; }

export CAUSAL_WORLD_MODEL_ROOT="$CWM_ROOT" DRIFT_GENERATOR="$GENERATOR"
export PYTHONPATH="$CRSSM_DIR:$CRSSM_DIR/dreamerv3_compat:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false DISABLE_TENSORBOARD=1

mkdir -p "$RESULTS_ROOT/logs"
SCHED="$RESULTS_ROOT/schedules"
( cd "$CRSSM_DIR" && "$PY" -m drift_eval.make_schedules --bench "$BENCH" --out-dir "$SCHED" \
    --episodes 64 --generator "$GENERATOR" --generator-sha256 "$GENERATOR_SHA256" ) \
  > "$RESULTS_ROOT/logs/generator.log" 2>&1

{
  echo "start=$(date -u +%FT%TZ) host=$(hostname) bench=$BENCH seeds=$SEEDS conditions=$CONDITIONS"
  echo "action_mode=$ACTION_MODE n_episodes=$N_EPISODES amount=$AMOUNT smoke=${SMOKE:-0}"
  echo "crssm_git=$(git -C "$CRSSM_DIR" rev-parse HEAD) dirty_files=$(git -C "$CRSSM_DIR" status --porcelain | wc -l)"
  echo "cwm_root=$CWM_ROOT cwm_head=$(git -C "$CWM_ROOT" rev-parse HEAD)"
  echo "generator=$GENERATOR sha256=$got"
  echo "python=$PY CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_GL=$MUJOCO_GL EGL_DEVICE_ID=${EGL_DEVICE_ID:-unset}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
} > "$RESULTS_ROOT/provenance.txt"
cat "$RESULTS_ROOT/provenance.txt"

cd "$CRSSM_DIR"
for s in $SEEDS; do
  # shellcheck disable=SC2086
  "$PY" -u -m drift_eval.run --bench "$BENCH" --seed "$s" --ckpt-root "$CKPT_ROOT" \
    --schedules-dir "$SCHED" --generator "$GENERATOR" --generator-sha256 "$GENERATOR_SHA256" \
    --conditions $CONDITIONS --n-episodes "$N_EPISODES" --amount "$AMOUNT" \
    --action-mode "$ACTION_MODE" --results-root "$RESULTS_ROOT" 2>&1 | tee "$RESULTS_ROOT/logs/seed$s.log"
done
echo "done=$(date -u +%FT%TZ)" >> "$RESULTS_ROOT/provenance.txt"
