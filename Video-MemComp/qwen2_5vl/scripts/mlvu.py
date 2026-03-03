#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available
import torch
import torch.distributed as dist
import json
import os
import os.path as osp
from tqdm import tqdm
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import sys

# 确保自定义模型的定义在 Python 路径中
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from qwen_vl_utils import process_vision_info
    from qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLSdpaAttention,  # 确保这个模块存在且可导入
    )
except ImportError:
    print("导入失败。请确保 'qwen_vl_utils' 和 'qwen2_5_vl' 模块在您的 PYTHONPATH 中并且可访问。")
    sys.exit(1)


# --- 默认参数定义 ---
RUN_NAME = "mlvu_eval_run"
DROP_METHOD = 'none'
DROP_THRESHOLD = 0.5
CKPT_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
TASK_JSON = "/path/to/your/mlvu/dev_debug_mc.json"
VIDEO_DIR = "/path/to/your/mlvu/root/"
RESULT_DIR = "eval_results/mlvu"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 1145
MIN_FRAMES = 4

# --- 提示模板 ---
prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, etc.) of the correct option.
Question: {}
Options:
{}
The best answer is:"""


# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)


# --- 辅助函数 ---
def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    if len(s.split()) > 10 and not re.search(r"\b[A-E]\b", s, re.IGNORECASE):
        return ""
    matches = re.search(r"^\(?([A-E])\)?\.?\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    matches = re.search(r"\b([A-E])\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    return ""


def merge_jsonl_files(base_path, world_size):
    merged_data = []
    for rank in range(world_size):
        file_path = f"{base_path}_rank{rank}.jsonl"
        if osp.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        merged_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"无法解析文件 {file_path} 中的行: {line.strip()}")
    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"成功合并 {len(merged_data)} 条记录到 {merged_file_path}")
    return merged_file_path


# --- 主脚本 ---
def main():
    parser = argparse.ArgumentParser(description="在 MLVU 数据集上进行分布式评测。")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    # --- 1. 分布式环境初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    # --- 2. 路径和日志设置 ---
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'drop', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()
    log_path = osp.join(args.result_dir, 'log',
                        f"{args.run_name}_{curr_time}_rank{local_rank}.log")
    output_jsonl_path = osp.join(
        args.result_dir, 'output', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")
    dr_save_path = osp.join(
        args.result_dir, 'drop', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if is_main_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # --- 3. 模型加载与修改 ---
    # <--- 这是解决您报错的关键代码部分 ---
    torch.manual_seed(1234)
    logger.info("设置全局随机种子为 1234")
    if not is_flash_attn_2_available():
        logger.error("Flash Attention 2 不可用，此脚本依赖于注意力替换逻辑，程序将退出。")
        sys.exit(1)

    if not is_main_process:
        dist.barrier()
    logger.info("步骤 1: 在 CPU 上加载模型以准备替换注意力模块...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    model.model.size = 3
    logger.info("模型已加载到 CPU。")
    if is_main_process:
        dist.barrier()

    logger.info(f"[Rank {local_rank}] 步骤 2: 将语言模型的注意力机制替换为 SDPA...")
    llm_config = model.model.config
    for i, layer in enumerate(model.model.layers):
        original_flash_attn_module = layer.self_attn
        original_state_dict = original_flash_attn_module.state_dict()
        new_sdpa_attn_module = Qwen2_5_VLSdpaAttention(
            config=llm_config, layer_idx=i)
        new_sdpa_attn_module.load_state_dict(original_state_dict)
        layer.self_attn = new_sdpa_attn_module.to(dtype=torch.bfloat16)
    logger.info(f"[Rank {local_rank}] 语言模型的注意力模块已全部替换。")

    logger.info(f"[Rank {local_rank}] 步骤 3: 将修改后的模型部署到 cuda:{local_rank}...")
    model.to(f'cuda:{local_rank}')
    logger.info(f"[Rank {local_rank}] 模型已成功部署到 GPU。")
    # --- 关键代码部分结束 ---

    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info(f"[Rank {local_rank}] 预处理器加载完毕。")

    # --- 4. 数据加载与分布式切分 ---
    logger.info(f"[Rank {local_rank}] 正在从 {args.task_json} 加载和处理 MLVU 数据...")
    if not osp.exists(args.task_json):
        logger.error(f"任务文件未找到: {args.task_json}")
        sys.exit(1)
    with open(args.task_json, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    all_questions = []
    for video_info in full_data:
        for i, conv in enumerate(video_info['conversations']):
            question_id = f"{video_info['video_id']}_q{i}"
            all_questions.append({
                'video_id': video_info['video_id'],
                'video_path': video_info['video_path'],
                'duration': video_info.get('duration', -1),
                'question': conv['question'],
                'choices': conv['choices'],
                'answer': conv['answer'],
                'question_type': conv['question_type'],
                'question_id': question_id,
            })

    num_samples = len(all_questions)
    indices = list(range(num_samples))
    rank_indices = indices[local_rank::world_size]
    task_data = [all_questions[i] for i in rank_indices]
    logger.info(
        f"[Rank {local_rank}] 数据扁平化和切分完毕，本进程将处理 {len(task_data)} / {num_samples} 个问题。")

    processed_ids = set()
    if osp.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['question_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
    if processed_ids:
        original_count = len(task_data)
        task_data = [
            item for item in task_data if item['question_id'] not in processed_ids]
        logger.info(
            f"[Rank {local_rank}] 从已有输出文件中恢复进度，跳过 {original_count - len(task_data)} 个已完成任务。")

    # --- 5. 推理循环 ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    progress_bar = tqdm(task_data, total=len(
        task_data), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for item in progress_bar:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'reset_streaming_state'):
                model.model.reset_streaming_state()

            video_path = osp.join(args.video_dir, item['video_path'])
            if not osp.exists(video_path):
                logger.warning(
                    f"视频文件未找到: {video_path}，跳过问题 {item['question_id']}")
                continue

            # duration = item.get('duration', -1)
            # if duration > 0 and duration < 180:
            #     current_fps = 4.0
            # else:
            #     current_fps = 0.5

            options_str = "\n".join(
                [f"{chr(65+i)}) {choice}" for i, choice in enumerate(item['choices'])])
            query = prompt.format(item['question'], options_str)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path,
                 "nframes": 768, "max_frames": args.max_frames},
                {"type": "text", "text": query}
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)

            gen_kwargs = {"max_new_tokens": 128}
            if args.drop_method not in ['none', None]:
                gen_kwargs.update({"drop_method": args.drop_method, "drop_threshold": args.drop_threshold, "drop_absolute": (
                    not args.drop_relative), "dr_save_path": dr_save_path})

            generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(
                inputs.input_ids, generated_ids)]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True)[0]

            try:
                answer_idx = item['choices'].index(item['answer'])
                correct_letter = chr(65 + answer_idx)
            except ValueError:
                logger.warning(
                    f"答案 '{item['answer']}' 不在选项列表 {item['choices']} 中，跳过问题 {item['question_id']}")
                continue

            pred_letter = extract_characters_regex(response)
            is_correct = (pred_letter == correct_letter)
            question_type = item['question_type']
            if is_correct:
                cnt_correct['overall'] += 1
                cnt_correct[question_type] += 1
            cnt_total['overall'] += 1
            cnt_total[question_type] += 1

            output_dict = {
                'question_id': item['question_id'], 'video_id': item['video_id'],
                'question_type': question_type, 'question': item['question'], 'choices': item['choices'], 'answer': item['answer'],
                'correct_letter': correct_letter, 'response': response, 'prediction': pred_letter, 'is_correct': is_correct,
            }
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理 {item.get('question_id', 'N/A')} 时发生严重错误: {e}")
            traceback.print_exc()

    # --- 6. 结果汇总与清理 ---
    dist.barrier()
    logger.info(f"[Rank {local_rank}] 推理完成，等待所有进程并开始汇总...")
    all_question_types = sorted(
        list(set(q['question_type'] for q in all_questions)))
    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    for q_type in all_question_types:
        stats_list.extend([cnt_total[q_type], cnt_correct[q_type]])
    stats_tensor = torch.tensor(
        stats_list, dtype=torch.long, device=f'cuda:{local_rank}')
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    if is_main_process:
        logger.info("--- 所有进程已完成，开始最终结果汇总 ---")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)
        if args.drop_method not in [None, 'none']:
            base_drop_path = osp.join(
                args.result_dir, 'drop', f"{args.run_name}_{curr_time}")
            merge_jsonl_files(base_drop_path, world_size)

        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()
        if total_processed == 0:
            logger.info("所有进程均未成功处理任何问题。")
        else:
            accuracy = 100 * total_correct / total_processed
            logger.info(
                f"【总结果】: 总数={total_processed}, 正确={total_correct}, 准确率={accuracy:.2f}%")

        logger.info("--- 按问题类型分类准确率 ---")
        for i, q_type in enumerate(all_question_types):
            total_cat = stats_tensor[2 + i*2].item()
            correct_cat = stats_tensor[3 + i*2].item()
            if total_cat > 0:
                acc_cat = 100 * correct_cat / total_cat
                logger.info(
                    f"  - {q_type:<18}: {correct_cat}/{total_cat} = {acc_cat:.2f}%")
        cost_time = int(time.time() - start_time)
        logger.info(
            f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (此为单个进程耗时，非总和)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
