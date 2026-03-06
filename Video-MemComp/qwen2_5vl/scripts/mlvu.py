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

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from qwen_vl_utils import process_vision_info
    from qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLSdpaAttention,  
    )
except ImportError:
    print("Import failed. Please ensure 'qwen_vl_utils' and 'qwen2_5_vl' modules are in your PYTHONPATH and accessible.")
    sys.exit(1)

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

prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, etc.) of the correct option.
Question: {}
Options:
{}
The best answer is:"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

def extract_characters_regex():
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
                            f"Failed to parse line in file {file_path}: {line.strip()}")
    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Successfully merged {len(merged_data)} records to {merged_file_path}")
    return merged_file_path


# --- 主脚本 ---
def main():
    parser = argparse.ArgumentParser(description="Distributed evaluation script for MLVU dataset.")
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

    # --- 1. Distributed Environment Initialization ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    # --- 2. Path and Logging Configuration ---
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
    logger.info(f"--- [Rank {local_rank}/{world_size}] Process started ---")
    logger.info(f"Run configuration: {vars(args)}")


    # --- 3. Model Loading and Modification ---
    torch.manual_seed(1234)
    logger.info("Setting global random seed to 1234")
    if not is_flash_attn_2_available():
        logger.error("Flash Attention 2 is not available. This script depends on attention replacement logic. Exiting.")
        sys.exit(1)

    if not is_main_process:
        dist.barrier()
    logger.info("Step 1: Loading model on CPU to prepare for attention module replacement...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    model.model.size = 3
    logger.info("Model loaded to CPU.")
    if is_main_process:
        dist.barrier()

    logger.info(f"[Rank {local_rank}] Step 2: Replacing language model attention mechanism with SDPA...")
    llm_config = model.model.config
    for i, layer in enumerate(model.model.layers):
        original_flash_attn_module = layer.self_attn
        original_state_dict = original_flash_attn_module.state_dict()
        new_sdpa_attn_module = Qwen2_5_VLSdpaAttention(
            config=llm_config, layer_idx=i)
        new_sdpa_attn_module.load_state_dict(original_state_dict)
        layer.self_attn = new_sdpa_attn_module.to(dtype=torch.bfloat16)
    logger.info(f"[Rank {local_rank}] All language model attention modules have been replaced.")

    logger.info(f"[Rank {local_rank}] Step 3: Deploying modified model to cuda:{local_rank}...")
    model.to(f'cuda:{local_rank}')
    logger.info(f"[Rank {local_rank}] Model successfully deployed to GPU.")

    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info(f"[Rank {local_rank}] Preprocessor loaded successfully.")

    # --- 4. Data Loading and Distributed Splitting ---
    logger.info(f"[Rank {local_rank}] Loading and processing MLVU data from {args.task_json}...")
    if not osp.exists(args.task_json):
        logger.error(f"Task file not found: {args.task_json}")
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
        f"[Rank {local_rank}] Data flattening and splitting complete. This process will handle {len(task_data)} / {num_samples} questions.")


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
            f"[Rank {local_rank}] Resuming progress from existing output file. Skipping {original_count - len(task_data)} completed tasks.")

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
                    f"Video file not found: {video_path}, skipping question {item['question_id']}")
                continue

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
                    f"Answer '{item['answer']}' not found in options list {item['choices']}, skipping question {item['question_id']}")
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
                f"[Rank {local_rank}] Critical error processing {item.get('question_id', 'N/A')}: {e}")
            traceback.print_exc()

    dist.barrier()
    logger.info(f"[Rank {local_rank}] Inference complete. Waiting for all processes and starting aggregation...")
    all_question_types = sorted(
        list(set(q['question_type'] for q in all_questions)))
    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    for q_type in all_question_types:
        stats_list.extend([cnt_total[q_type], cnt_correct[q_type]])
    stats_tensor = torch.tensor(
        stats_list, dtype=torch.long, device=f'cuda:{local_rank}')
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    if is_main_process:
        logger.info("--- All processes completed. Starting final result aggregation ---")
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
            logger.info("No questions were successfully processed by any process.")
        else:
            accuracy = 100 * total_correct / total_processed
            logger.info(
                f"[Overall Results]: Total={total_processed}, Correct={total_correct}, Accuracy={accuracy:.2f}%")

        logger.info("--- Accuracy by Question Type ---")
        for i, q_type in enumerate(all_question_types):
            total_cat = stats_tensor[2 + i*2].item()
            correct_cat = stats_tensor[3 + i*2].item()
            if total_cat > 0:
                acc_cat = 100 * correct_cat / total_cat
                logger.info(
                    f"  - {q_type:<18}: {correct_cat}/{total_cat} = {acc_cat:.2f}%")
        cost_time = int(time.time() - start_time)
        logger.info(
            f"Total inference time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (This is per-process time, not cumulative)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
