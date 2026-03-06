#!/bin/bash
# Get Project Root Directory 
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# Mode Selection
RUN_SUBSET_ONLY="true"
RUN_NAME="egoschema_qwen2.5vl" 
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"
PYTHON_SCRIPT="$SCRIPT_DIR/eval_egoschema.py"
ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"
# Experiment Parameter Configuration
RESULT_ROOT_DIR="$SCRIPT_DIR/eval_results/egoschema_runs"
DROP_METHOD="feature" 
DROP_THRESHOLD=0.5
NFRAMES=768 
export CUDA_VISIBLE_DEVICES=6 
N_PROC=1

RESULT_DIR="${RESULT_ROOT_DIR}/${RUN_NAME}"
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Run Name:              $RUN_NAME"
echo "Model Path:            $CKPT_PATH"
echo "=================================================="

mkdir -p "$RESULT_DIR"

CMD_ARGS=(
    "--run_name" "$RUN_NAME"
    "--ckpt_path" "$CKPT_PATH"
    "--anno_path" "$ANNO_PATH"
    "--video_dir" "$VIDEO_DIR"
    "--result_dir" "$RESULT_DIR"
    "--subset_answers_path" "$SUBSET_ANSWERS_PATH"
    "--drop_method" "$DROP_METHOD"
    "--drop_threshold" "$DROP_THRESHOLD"
    "--nframes" "$NFRAMES"
)

if [ "$RUN_SUBSET_ONLY" = "true" ]; then
    CMD_ARGS+=( "--eval_subset_only" )
fi

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$PYTHON_SCRIPT" \
    "${CMD_ARGS[@]}"

echo "=================================================="
echo "Evaluation script finished."
echo "Logs and results are in: $RESULT_DIR"
