#!/bin/bash

# ==============================================================================
# EgoSchema 评测启动脚本 (GitHub 整理版)
# ==============================================================================

# --- 0. 获取项目根目录 ---
# 假设脚本位于 project_root/qwen2_5vl/scripts/ 目录下
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# --- 1. 模式选择 ---
RUN_SUBSET_ONLY="true"

# --- 2. 核心路径配置 (使用相对路径或基于 PROJECT_ROOT) ---
RUN_NAME="egoschema_qwen2.5vl_0.5_0.75" 

# 修改点：指向项目内的 model 目录
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"

# 修改点：指向项目内的脚本路径
PYTHON_SCRIPT="$SCRIPT_DIR/eval_egoschema.py"

# --- 3. 数据集路径配置 (指向项目内的 data 目录) ---
# 这些路径现在都指向你之前通过 ln -s 创建的软链接位置
ANNO_PATH="$PROJECT_ROOT/data/egoschema/full.json"
VIDEO_DIR="$PROJECT_ROOT/data/egoschema/videos"
SUBSET_ANSWERS_PATH="$PROJECT_ROOT/data/egoschema/subset_answers.json"

# --- 4. 实验参数配置 ---
# 结果建议保存在脚本同级或项目根目录下的 eval_results
RESULT_ROOT_DIR="$SCRIPT_DIR/eval_results/egoschema_runs"

# 特征丢弃设置
DROP_METHOD="feature" 
DROP_THRESHOLD=0.5

# 视频处理设置
NFRAMES=768 

# --- 5. 分布式执行配置 ---
export CUDA_VISIBLE_DEVICES=6 
N_PROC=1

# ==============================================================================
# 脚本执行部分
# ==============================================================================

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