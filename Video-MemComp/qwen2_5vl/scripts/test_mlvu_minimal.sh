#!/bin/bash

# Minimal test script for MLVU local benchmark
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=40960

# Test with minimal parameters
TASKS=mlvu_dev_local
NUM_PROCESSES=1
MAIN_PROCESS_PORT=29501

RUN_NAME=test_minimal
CKPT_PATH="/data1/nyh/TimeChat-Online/models/Qwen/Qwen2___5-VL-3B-Instruct"
RESULT_DIR="test_mlvu_minimal"

# DTD arguments  
DROP_METHOD=feature
DROP_TRESHOLD=0.3

echo "Testing MLVU with minimal configuration..."
echo "Tasks: $TASKS"
echo "Model path: $CKPT_PATH"
echo "Result dir: $RESULT_DIR"

# Very minimal test - only 1 sample
cd /data1/nyh/qwen2_5vl/eval
PYTHONPATH=/data1/nyh/qwen2_5vl/eval:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
    --num_processes $NUM_PROCESSES \
    --main_process_port $MAIN_PROCESS_PORT \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${CKPT_PATH},min_pixels=256*256,max_pixels=256*256,max_num_frames=16,fps=1,use_flash_attention_2=True,device_map=cuda:0,drop_method=${DROP_METHOD},drop_threshold=${DROP_TRESHOLD},dr_save_path=${RESULT_DIR}/drop/${RUN_NAME}.jsonl \
    --tasks $TASKS \
    --batch_size 1 \
    --limit 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path ${RESULT_DIR}/log

echo "Minimal test completed. Check ${RESULT_DIR}/log for results."