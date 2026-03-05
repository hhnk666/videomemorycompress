#!/bin/bash
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# Dataset Path Configuration
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"

# Model & Run Configuration
RUN_NAME="feature_0d5"
CKPT_PATH="$PROJECT_ROOT/model/llava-onevision-qwen2-7b-ov-hf"
RESULT_DIR="$PROJECT_ROOT/llavaov/eval_results/videomme_llava/${RUN_NAME}"

# Environment & Distributed Configuration
export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=0
MASTER_PORT=$(shuf -i 20000-65000 -n 1) 

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Python Script:         $SCRIPT_DIR/videomme.py"
echo "Model Path:            $CKPT_PATH"
echo "Master Port:           $MASTER_PORT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/videomme.py" \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"

echo "=================================================="
echo "Evaluation finished. Results in: $RESULT_DIR"
