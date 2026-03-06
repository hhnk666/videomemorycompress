#!/bin/bash
# --- 0. Dynamically Get Paths ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"
VIDEO_DIR="$PROJECT_ROOT" 
RUN_NAME="qwen2.5vl_3b_mlvu_500token"
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/${RUN_NAME}"
DROP_METHOD="feature"     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.30
export PYTHONPATH="$PROJECT_ROOT/qwen2_5vl/demo/qwen_vl_utils/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2
N_PROC=1
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting distributed evaluation for MLVU Benchmark"
echo "Master Port:  $MASTER_PORT"
echo "Run Name:     $RUN_NAME"
echo "CKPT Path:    $CKPT_PATH"
echo "=================================================="

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/mlvu.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"

echo "=================================================="
echo "MLVU evaluation finished."
echo "Results saved in: $RESULT_DIR"
