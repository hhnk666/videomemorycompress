#!/bin/bash

# --- 0. 动态获取项目根目录 ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# --- 1. 数据集路径配置 (指向项目内的 data 软链接) ---
# 原路径: /data1/nyh/ReKV/videomme/videomme/test-00000-of-00001.parquet
TASK_PARQUET="$PROJECT_ROOT/data/videomme/videomme/test-00000-of-00001.parquet"
# 原路径: /data1/nyh/ReKV/videomme/videos
VIDEO_DIR="$PROJECT_ROOT/data/videomme/videos"

# --- 2. 模型与结果输出配置 ---
RUN_NAME="feature_0d5"
# 修改点：指向项目内的 model 目录
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"
RESULT_DIR="$SCRIPT_DIR/eval_results/videomme/${RUN_NAME}"

# --- 3. 实验参数 ---
DROP_METHOD=feature     
DROP_THRESHOLD=0.5

# --- 4. 分布式与环境配置 ---
export CUDA_VISIBLE_DEVICES=2
export NCCL_P2P_DISABLE=1
N_PROC=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1) 

echo "=================================================="
echo "Starting Video-MME Evaluation"
echo "Project Root: $PROJECT_ROOT"
echo "Master Port:  $MASTER_PORT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/videomme.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_parquet "$TASK_PARQUET" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"

echo "Evaluation finished."