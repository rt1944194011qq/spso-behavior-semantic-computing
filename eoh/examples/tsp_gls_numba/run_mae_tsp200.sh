#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-/public2/home/rentao/miniconda3/envs/mae_eoh/bin/python}
export MAE_USE_ENVIRONMENT=1
export MAE_RUN_NAME=${MAE_RUN_NAME:-mae_semantic_tsp200_seed2026_run001}
export MAE_TRAIN_DATASET=${MAE_TRAIN_DATASET:-$SCRIPT_DIR/TrainingData/custom_TSP200_seed2026_solved.pkl}
export MAE_SMOKE_TEST=${MAE_SMOKE_TEST:-0}
export MAE_OFFLINE=${MAE_OFFLINE:-0}
export MAE_SEED=${MAE_SEED:-2026}
export MAE_NUM_AGENTS=${MAE_NUM_AGENTS:-10}
export MAE_GENERATIONS=${MAE_GENERATIONS:-20}
export MAE_INITIAL_MULTIPLIER=${MAE_INITIAL_MULTIPLIER:-10}
export MAE_TRAIN_INSTANCES=${MAE_TRAIN_INSTANCES:-200}
export MAE_GLS_TIME_LIMIT=${MAE_GLS_TIME_LIMIT:-60}
export MAE_GLS_ITE_MAX=${MAE_GLS_ITE_MAX:-1000}
export MAE_PERTURBATION_MOVES=${MAE_PERTURBATION_MOVES:-1}
export MAE_TASK_TIMEOUT=${MAE_TASK_TIMEOUT:-900}
export MAE_NUM_LLM_WORKERS=${MAE_NUM_LLM_WORKERS:-16}
export MAE_NUM_EVALUATORS=${MAE_NUM_EVALUATORS:-16}
export MAE_EXPERIENCE_BIRTH_MIN=${MAE_EXPERIENCE_BIRTH_MIN:-0.20}
export MAE_EXPERIENCE_BIRTH_MAX=${MAE_EXPERIENCE_BIRTH_MAX:-0.60}
export MAE_EXPERIENCE_TARGET_WEIGHT=${MAE_EXPERIENCE_TARGET_WEIGHT:-0.60}
export MAE_EXPERIENCE_HALF_LIFE=${MAE_EXPERIENCE_HALF_LIFE:-4.0}
export MAE_EXPERIENCE_DROP_BELOW=${MAE_EXPERIENCE_DROP_BELOW:-0.05}
export MAE_EXPERIENCE_CAPACITY=${MAE_EXPERIENCE_CAPACITY:-24}
export MAE_CUT_TARGET_SIZE=${MAE_CUT_TARGET_SIZE:-4}
export MAE_CUT_MIN_SIZE=${MAE_CUT_MIN_SIZE:-2}
export MAE_NEIGHBOR_RESERVED_SLOTS=${MAE_NEIGHBOR_RESERVED_SLOTS:-1}
export MAE_H1_SELECTION_BOOST=${MAE_H1_SELECTION_BOOST:-1.5}
export MAE_H2_SELECTION_BOOST=${MAE_H2_SELECTION_BOOST:-1.5}
export MAE_OPERATORS=${MAE_OPERATORS:-H1,H2,H3,H4,H5}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MAE_TRAIN_DATASET" ]]; then
  echo "Training dataset is unavailable: $MAE_TRAIN_DATASET" >&2
  exit 1
fi
if [[ "$MAE_OFFLINE" != "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Set DEEPSEEK_API_KEY before starting an online experiment." >&2
  exit 1
fi

RUN_DIR="$SCRIPT_DIR/results_mae/$MAE_RUN_NAME"
if [[ -f "$RUN_DIR/config.json" && -z "${MAE_RESUME_FROM:-}" ]]; then
  echo "Run already exists; choose a new MAE_RUN_NAME: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" validate_tsp_training_data.py \
  --dataset "$MAE_TRAIN_DATASET" \
  --output "$RUN_DIR/dataset_manifest.json"

CONSOLE_LOG="$SCRIPT_DIR/results_mae/${MAE_RUN_NAME}.console.log"
PID_FILE="$SCRIPT_DIR/results_mae/${MAE_RUN_NAME}.pid"
nohup "$PYTHON_BIN" -u runMAEoH.py > "$CONSOLE_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

MONITOR_LOG="$SCRIPT_DIR/results_mae/${MAE_RUN_NAME}.monitor.log"
MONITOR_PID_FILE="$SCRIPT_DIR/results_mae/${MAE_RUN_NAME}.monitor.pid"
nohup "$PYTHON_BIN" -u monitor_mae_run.py \
  --run-dir "$RUN_DIR" --pid "$TRAIN_PID" --interval 60 \
  > "$MONITOR_LOG" 2>&1 < /dev/null &
echo $! > "$MONITOR_PID_FILE"

echo "run directory: $RUN_DIR"
echo "training pid: $TRAIN_PID"
echo "monitor pid: $(cat "$MONITOR_PID_FILE")"
echo "console log: $CONSOLE_LOG"
echo "progress snapshot: $RUN_DIR/monitor/latest_snapshot.json"
