#!/bin/bash
# Dynamically Get Paths
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# Core Path Configuration
MODEL_PATH="$PROJECT_ROOT/model/qwen2vl"
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"
PYTHON_SCRIPT="$SCRIPT_DIR/videomme.py"
RESULT_DIR="$PROJECT_ROOT/eval_results/videomme_qwen2vl"
RUN_NAME="qwen2vl_videomme_0.20.556000"
export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:    $PROJECT_ROOT"
echo "Model Path:      $MODEL_PATH"
echo "Task Parquet:    $TASK_PARQUET"
echo "Result Dir:      $RESULT_DIR"
echo "=================================================="
mkdir -p "$RESULT_DIR"
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --video_dir "$VIDEO_DIR" \
    --task_parquet "$TASK_PARQUET" \
    --result_dir "$RESULT_DIR" \
    --run_name "$RUN_NAME"

echo "=================================================="
echo "Evaluation finished."
