#!/bin/bash

# ==============================================================================
# Qwen2-VL Video-MME 评测启动脚本 (GitHub 整理版)
# ==============================================================================

# --- 0. 动态获取路径 ---
# SCRIPT_DIR 是 project_root/qwen2vl/
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# PROJECT_ROOT 是 project_root/ (向上跳一级)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. 核心路径配置 (通过 PROJECT_ROOT 引用) ---
# 对应之前建立的软链接：project_root/model/qwen2vl
MODEL_PATH="$PROJECT_ROOT/model/qwen2vl"

# 对应之前建立的软链接：project_root/data/videomme
# 标注文件路径
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"

# 视频目录路径
# 根据你之前建立的链接 /data/videomme 指向 ReKV/videomme
# 通常结构为 VIDEOS_DIR="$PROJECT_ROOT/data/videomme/videos"
# 如果你的视频在 videos 下还有一个 data 目录，请按需微调
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"

# 评测 Python 脚本路径
PYTHON_SCRIPT="$SCRIPT_DIR/videomme.py"

# 结果保存目录
RESULT_DIR="$PROJECT_ROOT/eval_results/videomme_qwen2vl"

# --- 2. 运行参数配置 ---
RUN_NAME="qwen2vl_videomme_0.20.556000"

# --- 3. 环境与资源配置 ---
export CUDA_VISIBLE_DEVICES=0
# 随机选择一个端口防止冲突
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root:    $PROJECT_ROOT"
echo "Model Path:      $MODEL_PATH"
echo "Task Parquet:    $TASK_PARQUET"
echo "Result Dir:      $RESULT_DIR"
echo "=================================================="

# 确保结果目录存在
mkdir -p "$RESULT_DIR"

# --- 4. 使用 torchrun 启动 ---
# 使用环境默认的 torchrun，不再硬编码绝对路径
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --video_dir "$VIDEO_DIR" \
    --task_parquet "$TASK_PARQUET" \
    --result_dir "$RESULT_DIR" \
    --run_name "$RUN_NAME"

echo "=================================================="
echo "Evaluation finished."