#!/bin/bash

# ==============================================================================
# LLaVA-OneVision on Video-MME 评测启动脚本 (GitHub 整理版)
# ==============================================================================

# --- 0. 动态获取路径 (自动适配服务器环境) ---
# SCRIPT_DIR 是 /data1/nyh/Video-MemComp/llavaov
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# PROJECT_ROOT 是 /data1/nyh/Video-MemComp (向上跳一级)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# --- 1. 数据集路径配置 (指向项目内的 data 软链接) ---
# 原路径: /data1/nyh/ReKV/videomme/videomme/test-00000-of-00001.parquet
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"
# 原路径: /data1/nyh/ReKV/videomme/videos
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"

# --- 2. 模型与运行配置 ---
RUN_NAME="feature_0d5"
# 修改点：指向项目内的 model 软链接路径
CKPT_PATH="$PROJECT_ROOT/model/llava-onevision-qwen2-7b-ov-hf"
# 结果保存建议：保存在项目根目录下的 eval_results 中
RESULT_DIR="$PROJECT_ROOT/llavaov/eval_results/videomme_llava/${RUN_NAME}"

# --- 3. 环境与分布式配置 ---
export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=0
MASTER_PORT=$(shuf -i 20000-65000 -n 1) 

echo "=================================================="
echo "Project Root:          $PROJECT_ROOT"
echo "Python Script:         $SCRIPT_DIR/videomme.py"
echo "Model Path:            $CKPT_PATH"
echo "Master Port:           $MASTER_PORT"
echo "=================================================="

# 确保结果目录存在
mkdir -p "$RESULT_DIR"

# --- 4. 启动分布式评估 ---
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