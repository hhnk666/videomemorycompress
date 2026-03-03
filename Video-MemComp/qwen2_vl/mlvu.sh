#!/bin/bash

# ==============================================================================
# Qwen2-VL MLVU 评测启动脚本 (GitHub 整理版)
# ==============================================================================

# --- 0. 动态获取路径 ---
# 假设此脚本位于 project_root/qwen2vl/ 目录下
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# 向上跳一级到项目根目录 project_root/
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. 核心路径配置 (指向项目内的软链接) ---
# 对应之前建立的软链接：project_root/model/qwen2vl
MODEL_PATH="$PROJECT_ROOT/model/qwen2vl"

# 对应之前建立的软链接：project_root/data/mlvu
# 注意：VIDEO_DIR 通常指向包含 'data/mlvu/videos' 的父目录
VIDEO_DIR="$PROJECT_ROOT/data"

# MLVU 标注文件路径
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"

# 评测 Python 脚本路径
PYTHON_SCRIPT="$SCRIPT_DIR/mlvu.py"

# 结果保存目录
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/"

# --- 2. 运行参数配置 ---
RUN_NAME="mlvu0.20.55"
N_FRAMES=768

# --- 3. 环境与资源配置 ---
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480
# 随机选择端口避免冲突
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:    $PROJECT_ROOT"
echo "Model Path:      $MODEL_PATH"
echo "Task JSON:       $TASK_JSON"
echo "Master Port:     $MASTER_PORT"
echo "=================================================="

# 确保结果目录存在
mkdir -p "$RESULT_DIR"

# --- 4. 使用 torchrun 启动 ---
# 不再硬编码特定的 bin/torchrun 路径，直接调用环境中的 torchrun
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