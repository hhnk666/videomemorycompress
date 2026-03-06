#!/bin/bash

# --- 0. Dynamically Get Project Root Directory ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# --- 1. Environment and Resource Configuration ---
export CUDA_VISIBLE_DEVICES=3

# --- 2. Dataset Path Configuration (pointing to data symlinks within the project) ---
TASK_JSON="$PROJECT_ROOT/data/ovobench/ovo_bench_new.json"
VIDEO_DIR="$PROJECT_ROOT/data/ovobench/src_videos"

# --- 3. Model and Output Configuration ---
RUN_NAME="feature_0d5"
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-7B-Instruct"
RESULT_DIR="$SCRIPT_DIR/eval_results/ovobench/${RUN_NAME}"
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.8
echo "=================================================="
echo "Starting OVO-Bench Evaluation"
echo "Project Root: $PROJECT_ROOT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

python "$SCRIPT_DIR/ovobench.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"
