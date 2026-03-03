#!/bin/bash

# ==============================================================================
# MLVU Benchmark 评测启动脚本 (GitHub 整理版)
# ==============================================================================

# --- 0. 动态获取路径 ---
# 获取当前脚本所在目录：project_root/qwen2_5vl/scripts/
SCRIPT_DIR=$(cd $(dirname "$0"); pwd)
# 获取项目根目录：project_root/
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.."; pwd)

# ==================== 配置 MLVU benchmark 路径 ====================
# 修改点：指向项目内通过 ln -s 链接的路径
# MLVU 的 JSON 标注文件路径
TASK_JSON="$PROJECT_ROOT/data/mlvu/dev_debug_mc.json"
# 存放 MLVU 视频的根目录 (通常指向包含视频文件夹的上级目录)
VIDEO_DIR="$PROJECT_ROOT" 

# ==================== 模型和结果输出配置 ====================
# 定义一个清晰的运行名称
RUN_NAME="qwen2.5vl_3b_mlvu_500token"

# 修改点：指向项目内的 model 目录
CKPT_PATH="$PROJECT_ROOT/model/Qwen2___5-VL-3B-Instruct"

# 结果保存目录 (建议放在脚本目录下的 eval_results，方便管理)
RESULT_DIR="$SCRIPT_DIR/eval_results/mlvu/${RUN_NAME}"

# ==================== DTD 参数 ====================
DROP_METHOD="feature"     # "feature" or "pixel" or "none"
DROP_THRESHOLD=0.30

# ==================== 环境配置 ====================
# 修改点：使用相对路径设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/qwen2_5vl/demo/qwen_vl_utils/src:$PYTHONPATH"

# ==================== 分布式启动配置 ====================
export CUDA_VISIBLE_DEVICES=2
N_PROC=1
export NCCL_P2P_DISABLE=1
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Starting distributed evaluation for MLVU Benchmark"
echo "Master Port:  $MASTER_PORT"
echo "Run Name:     $RUN_NAME"
echo "CKPT Path:    $CKPT_PATH"
echo "=================================================="

# 修改点：Python 脚本路径也使用动态获取的路径
python -m torch.distributed.run \
    --nproc_per_node=$N_PROC \
    --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/mlvu.py" \
    --run_name "$RUN_NAME" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --ckpt_path "$CKPT_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR"

echo "=================================================="
echo "MLVU evaluation finished."
echo "Results saved in: $RESULT_DIR"