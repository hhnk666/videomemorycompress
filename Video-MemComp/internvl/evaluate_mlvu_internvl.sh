#!/bin/bash
# --- Dynamically obtain paths ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# Assuming script is under project_root/internvl/scripts/, go up two levels for root directory
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# ==================== Configure MLVU benchmark paths ====================
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"
VIDEO_DIR="$PROJECT_ROOT" 

# ==================== Model and output configuration ====================
RUN_NAME="internvl3.5_mlvu_100f"
CKPT_PATH="$PROJECT_ROOT/model/InternVL3_5"
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/${RUN_NAME}"
NUM_FRAMES=100
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=6  
N_PROC=1                            
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting distributed evaluation for MLVU Benchmark (InternVL)"
echo "Master Port:  $MASTER_PORT"
echo "Run Name:     $RUN_NAME"
echo "CKPT Path:    $CKPT_PATH"
echo "Frames:       $NUM_FRAMES"
echo "=================================================="

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/evaluate_mlvu_internvl.py" \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --num_frames $NUM_FRAMES

echo "=================================================="
echo "MLVU evaluation finished."
echo "Results saved in: $RESULT_DIR"
