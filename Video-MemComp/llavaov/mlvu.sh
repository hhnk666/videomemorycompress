#!/bin/bash
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# Mode Selection
RUN_SUBSET_ONLY="true"

# Core Path Configuration
RUN_NAME="egoschema_llava_onevision" 
CKPT_PATH="$PROJECT_ROOT/model/llava-onevision-qwen2-7b-ov-hf"
PYTHON_SCRIPT="$PROJECT_ROOT/llavaov/egoschema.py"

ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"

# Experiment Parameter Configuration
RESULT_ROOT_DIR="$PROJECT_ROOT/llavaov/eval_results/egoschema_llava_runs"
SAMPLE_FPS=0.5

# Distributed Execution Configuration
export CUDA_VISIBLE_DEVICES=7
N_PROC=1
RESULT_DIR="${RESULT_ROOT_DIR}/${RUN_NAME}"
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Video Dir:             $VIDEO_DIR"
echo "=================================================="

mkdir -p "$RESULT_DIR"

CMD_ARGS=(
    "--run_name" "$RUN_NAME"
    "--ckpt_path" "$CKPT_PATH"
    "--anno_path" "$ANNO_PATH"
    "--video_dir" "$VIDEO_DIR"
    "--result_dir" "$RESULT_DIR"
    "--subset_answers_path" "$SUBSET_ANSWERS_PATH"
    "--sample_fps" "$SAMPLE_FPS"
)

if [ "$RUN_SUBSET_ONLY" = "true" ]; then
    CMD_ARGS+=( "--eval_subset_only" )
fi

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$PYTHON_SCRIPT" \
    "${CMD_ARGS[@]}"
