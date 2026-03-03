#!/bin/bash

# --- 0. 动态获取路径 (修复版) ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. 模式选择 ---
RUN_SUBSET_ONLY="true"

# --- 2. 核心路径配置 ---
RUN_NAME="egoschema_llava_onevision_0.750.84.0" 
CKPT_PATH="$PROJECT_ROOT/model/llava-onevision-qwen2-7b-ov-hf"
PYTHON_SCRIPT="$PROJECT_ROOT/llavaov/egoschema.py"

# --- 3. 数据集路径配置 ---
ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"

# 【修正点】确保指向 data 目录下的 egoschema
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"

# 确保 subset_answers.json 在 data/egoschema/ 下
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"

# --- 4. 实验参数配置 ---
RESULT_ROOT_DIR="$PROJECT_ROOT/llavaov/eval_results/egoschema_llava_runs"
SAMPLE_FPS=0.5

# --- 5. 分布式执行配置 ---
export CUDA_VISIBLE_DEVICES=7
N_PROC=1

# ==============================================================================
# 脚本执行部分
# ==============================================================================
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