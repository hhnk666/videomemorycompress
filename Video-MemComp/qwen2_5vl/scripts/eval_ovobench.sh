#!/bin/bash

# --- 0. 动态获取项目根目录 ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# --- 1. 环境与资源配置 ---
export CUDA_VISIBLE_DEVICES=3

# --- 2. 数据集路径配置 (指向项目内的 data 软链接) ---
# 原路径: /data1/nyh/8_29/OVO-Bench/data/ovo_bench_new.json
TASK_JSON="$PROJECT_ROOT/data/ovobench/ovo_bench_new.json"
# 原路径: /data1/nyh/8_29/OVO-Bench/data/src_videos
VIDEO_DIR="$PROJECT_ROOT/data/ovobench/src_videos"

# --- 3. 模型与结果输出配置 ---
RUN_NAME="feature_0d5"
# 修改点：指向项目内的 model 目录
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-7B-Instruct"
RESULT_DIR="$SCRIPT_DIR/eval_results/ovobench/${RUN_NAME}"

# --- 4. 实验参数 (DTD) ---
DROP_METHOD=feature     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.8

# --- 5. 执行评测 ---
echo "=================================================="
echo "Starting OVO-Bench Evaluation"
echo "Project Root: $PROJECT_ROOT"
echo "=================================================="

mkdir -p "$RESULT_DIR"

python "$SCRIPT_DIR/ovobench.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"