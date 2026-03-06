#!/bin/bash
# Dynamically Get Paths
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# Core Path Configuration
MODEL_PATH="$PROJECT_ROOT/model/qwen2vl"
VIDEO_DIR="$PROJECT_ROOT/data"
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"
PYTHON_SCRIPT="$SCRIPT_DIR/mlvu.py"
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/"
RUN_NAME="mlvu0.20.55"
N_FRAMES=768
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:    $PROJECT_ROOT"
echo "Model Path:      $MODEL_PATH"
echo "Task JSON:       $TASK_JSON"
echo "Master Port:     $MASTER_PORT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --video_dir "$VIDEO_DIR" \
    --task_json "$TASK_JSON" \
    --result_dir "$RESULT_DIR" \
    --run_name "$RUN_NAME" \
    --n_frames "$N_FRAMES"

echo "=================================================="
echo "MLVU evaluation finished."
