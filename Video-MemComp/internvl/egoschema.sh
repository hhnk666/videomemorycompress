#!/bin/bash

# ==============================================================================
# EgoSchema 评测启动脚本 (GitHub 整理版 - InternVL 3.5)
# ==============================================================================

# --- 0. 动态获取路径 ---
# SCRIPT_DIR 是 project_root/internvl
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# PROJECT_ROOT 是 project_root/ (向上跳一级)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. 模式选择 ---
RUN_SUBSET_ONLY="true"

# --- 2. 核心路径配置 ---
RUN_NAME="egoschema_internvl3.5_100" 

# 修改点：使用项目内的 model 软链接路径
CKPT_PATH="$PROJECT_ROOT/model/InternVL3_5"

# 修改点：指向项目内的 Python 脚本
PYTHON_SCRIPT="$SCRIPT_DIR/egoschema.py"

# --- 3. 数据集路径配置 ---
# 修改点：指向项目内的 data 软链接路径
ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"

# 修改点：确保指向项目内 data 下的 subset_answers.json
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"

# --- 4. 实验参数配置 ---
# 结果保存建议
RESULT_ROOT_DIR="$PROJECT_ROOT/internvl/eval_results/egoschema_internvl_runs"
NFRAMES=64

# --- 5. GPU 配置 ---
export CUDA_VISIBLE_DEVICES=1

# ==============================================================================
# 脚本执行部分
# ==============================================================================

RESULT_DIR="${RESULT_ROOT_DIR}/${RUN_NAME}"

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Model Path:            $CKPT_PATH"
echo "Python Script:         $PYTHON_SCRIPT"
echo "Run Name:              $RUN_NAME"
echo "Mode:                  $(if [ "$RUN_SUBSET_ONLY" = "true" ]; then echo "QUICK VALIDATION"; else echo "FULL RUN"; fi)"
echo "=================================================="

# 确保结果目录存在
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

# 使用 python 直接启动单卡脚本
python "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

echo "=================================================="
echo "Evaluation script finished."
echo "Results are in: $RESULT_DIR"