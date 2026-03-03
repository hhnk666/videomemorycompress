#!/bin/bash

# ==============================================================================
# MLVU Benchmark InternVL 评测启动脚本 (分布式)
# ==============================================================================

# --- 0. 动态获取路径 ---
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# 假设脚本在 project_root/internvl/scripts/ 下，根目录往上退两级
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.."; pwd)

# ==================== 配置 MLVU benchmark 路径 ====================
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"
VIDEO_DIR="$PROJECT_ROOT" 

# ==================== 模型和结果输出配置 ====================
RUN_NAME="internvl3.5_mlvu_100f"
CKPT_PATH="$PROJECT_ROOT/model/InternVL3_5"
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/${RUN_NAME}"

# ==================== 评测参数 ====================
NUM_FRAMES=100

# ==================== 环境配置 ====================
# 将项目根目录加入PYTHONPATH，确保能找到 InternVL3_5
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ==================== 分布式启动配置 ====================
export CUDA_VISIBLE_DEVICES=6  # 根据你的实际GPU修改
N_PROC=1                             # GPU数量
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting distributed evaluation for MLVU Benchmark (InternVL)"
echo "Master Port:  $MASTER_PORT"
echo "Run Name:     $RUN_NAME"
echo "CKPT Path:    $CKPT_PATH"
echo "Frames:       $NUM_FRAMES"
echo "=================================================="

python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/evaluate_mlvu_internvl.py" \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --num_frames $NUM_FRAMES

echo "=================================================="
echo "MLVU evaluation finished."
echo "Results saved in: $RESULT_DIR"