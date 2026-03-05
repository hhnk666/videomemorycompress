#!/bin/bash
# --- 0. Dynamically obtain paths ---
# SCRIPT_DIR is project_root/internvl
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# PROJECT_ROOT is project_root/ (go up one level)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. Mode selection ---
RUN_SUBSET_ONLY="true"

# --- 2. Core path configuration ---
RUN_NAME="egoschema_internvl3.5_100" 
CKPT_PATH="$PROJECT_ROOT/model/InternVL3_5"
PYTHON_SCRIPT="$SCRIPT_DIR/egoschema.py"

# --- 3. Dataset path configuration ---
ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"

# --- 4. Experiment parameter configuration ---
RESULT_ROOT_DIR="$PROJECT_ROOT/internvl/eval_results/egoschema_internvl_runs"
NFRAMES=64
export CUDA_VISIBLE_DEVICES=1

RESULT_DIR="${RESULT_ROOT_DIR}/${RUN_NAME}"

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Model Path:            $CKPT_PATH"
echo "Python Script:         $PYTHON_SCRIPT"
echo "Run Name:              $RUN_NAME"
echo "Mode:                  $(if [ "$RUN_SUBSET_ONLY" = "true" ]; then echo "QUICK VALIDATION"; else echo "FULL RUN"; fi)"
echo "=================================================="

mkdir -p "$RESULT_DIR"

CMD_ARGS=(
    "--run_name" "$RUN_NAME"
    "--ckpt_path" "$CKPT_PATH"
    "--anno_path" "$ANNO_PATH"
    "--video_dir" "$VIDEO_DIR"
    "--result_dir" "$RESULT_DIR"
    "--subset_answers_path" "$SUBSET_ANSWERS_PATH"
    "--nframes" "$NFRAMES"
)

if [ "$RUN_SUBSET_ONLY" = "true" ]; then
    CMD_ARGS+=( "--eval_subset_only" )
fi

python "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

echo "=================================================="
echo "Evaluation script finished."
echo "Results are in: $RESULT_DIR"
