#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/fgcs_heavy}"
CHECKPOINT="${CHECKPOINT:-$ROOT/models/Juggernaut_X_RunDiffusion.safetensors}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/heavy_session_20260715_remaining}"
SCRIPT="${SCRIPT:-$ROOT/gcp_heavy_session_benchmark.py}"

mkdir -p "$OUT_ROOT/logs"

export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

run_one() {
  local model="$1"
  local case_name="$2"
  local target_default="$3"
  local target_long="$4"
  local out_dir="$OUT_ROOT/${model}_${case_name}"
  local log_file="$OUT_ROOT/logs/${model}_${case_name}.log"

  echo "=== $(date -u +%FT%TZ) START $model/$case_name ===" | tee -a "$OUT_ROOT/logs/runner.log"
  rm -rf "$out_dir"
  python "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$out_dir" \
    --models "$model" \
    --cases "$case_name" \
    --target-seconds-default "$target_default" \
    --target-seconds-long "$target_long" \
    --min-variants 12 \
    --max-variants 120 \
    --drift-variants 5 \
    --shirt-models juggernaut_x_bucket,dreamshaper_8 \
    >"$log_file" 2>&1
  echo "=== $(date -u +%FT%TZ) DONE $model/$case_name ===" | tee -a "$OUT_ROOT/logs/runner.log"
}

run_one juggernaut_x_bucket shirt 300 600
run_one sdxl_base_1_0 mug 300 600
run_one dreamshaper_xl_1_0 mug 300 600
run_one dreamshaper_8 mug 300 600
run_one dreamshaper_8 shirt 300 600
run_one openjourney_v4 mug 300 600
