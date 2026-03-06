#!/bin/bash

# --- 0. Dynamically Get Project Root Directory ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# --- 1. Dataset Path Configuration (pointing to data symlinks within the project) ---
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"

# --- 2. Model and Output Configuration ---
RUN_NAME="feature_0d5"
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"
RESULT_DIR="$SCRIPT_DIR/eval_results/videomme/${RUN_NAME}"
DROP_METHOD=feature     
DROP_THRESHOLD=0.5
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_DISABLE=1
N_PROC=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1) 

echo "=================================================="
echo "Starting Video-MME Evaluation"
echo "Project Root: $PROJECT_ROOT"
echo "Master Port:  $MASTER_PORT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/videomme.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"

echo "Evaluation finished."
