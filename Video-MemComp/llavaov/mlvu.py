#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
from transformers import LlavaOnevisionProcessor, AutoConfig
from llava_onevision import LlavaOnevisionForConditionalGeneration
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
from decord import VideoReader, cpu
from PIL import Image
import math

# 确保已安装 fuzzywuzzy: pip install fuzzywuzzy python-levenshtein
try:
    from fuzzywuzzy import fuzz
except ImportError:
    print("错误: fuzzywuzzy 库未安装。请运行 'pip install fuzzywuzzy python-levenshtein'")
    sys.exit(1)


# --- 默认参数定义 (已为 MLVU 评测更新) ---
RUN_NAME = "llava_onevision_mlvu_eval"
CKPT_PATH = "/data1/nyh/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf"
# 确保将下面的路径更改为您的 MLVU 数据集路径
TASK_JSON = "/path/to/your/mlvu/dev_debug_mc.json"
VIDEO_DIR = "/path/to/your/mlvu/root/"
RESULT_DIR = "eval/result_mlvu_llava_onevision"
SAMPLE_FPS = 0.2
YARN_FACTOR = 2.0

# --- 提示模板 ---
prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, etc.) of the correct option.
Question: {}
Options: {}
"""

# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

# --- 辅助函数 ---


def extract_directly(s):
    s = s.strip()
    if len(s) == 1 and s.upper() in ['A', 'B', 'C', 'D', 'E']:
        return s.upper()
    if len(s) == 2 and s[1] in [')', ']', '.'] and s[0] in ['A', 'B', 'C', 'D', 'E']:
        return s[0].upper()
    return ""


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:", "The correct option is:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            if s.startswith(":") or s.startswith("."):
                s = s[1:].strip()
    # MLVU 的选项可能超过 D
    matches = re.search(r"^\s*([A-E])", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    matches = re.search(r"[(\[]\s*([A-E])\s*[)\]]", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    matches = re.search(r"\b([A-E])\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    return ""


def find_best_match(response_text, options):
    """
    当正则表达式无法提取选项字母时，此函数通过计算文本相似度来寻找最佳匹配。

    参数:
        response_text (str): 模型的原始文本输出。
        options (list): 包含所有选项文本的列表。

    返回:
        str: 匹配度最高的选项字母 (A, B, C, ...)，如果没有找到则返回空字符串。
    """
    if not response_text or not options:
        return ""

    scores = [fuzz.token_set_ratio(
        response_text.lower(), option.lower()) for option in options]

    if not scores:
        return ""

    max_score = max(scores)

    # 设定一个阈值（例如 75%），以避免在完全不相关时强行匹配
    if max_score > 75:
        best_index = scores.index(max_score)
        return chr(65 + best_index)

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
                        continue
    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"成功合并 {len(merged_data)} 条记录到 {merged_file_path}")
    return merged_file_path

# --- 主脚本 ---


def main():
    parser = argparse.ArgumentParser(
        description="在 MLVU 数据集上使用 LLaVA OneVision 进行流式分布式评测。")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_json", type=str, default=TASK_JSON)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sample_fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--yarn_factor", type=float, default=YARN_FACTOR)
    args = parser.parse_args()

    # --- 1. 分布式环境初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    device = f"cuda:{local_rank}"

    # --- 2. 路径和日志设置 ---
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(osp.join(args.result_dir, 'output'), exist_ok=True)
        os.makedirs(osp.join(args.result_dir, 'log'), exist_ok=True)
    dist.barrier()

    log_path = osp.join(args.result_dir, 'log',
                        f"{args.run_name}_{curr_time}_rank{local_rank}.log")
    output_jsonl_path = osp.join(
        args.result_dir, 'output', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if is_main_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # --- 3. 模型加载 (包含 YaRN 上下文扩展) ---
    torch.manual_seed(1919810)
    logger.info("设置全局随机种子为 1234")
    if not is_main_process:
        dist.barrier()

    logger.info(f"正在从 {args.ckpt_path} 加载处理器...")
    processor = LlavaOnevisionProcessor.from_pretrained(args.ckpt_path)
    logger.info(f"正在从 {args.ckpt_path} 加载模型配置...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)

    text_config = getattr(config, 'text_config', config)
    original_max_len = text_config.max_position_embeddings
    TARGET_MAX_LEN = int(original_max_len * args.yarn_factor)
    logger.info(
        f"正在通过 YaRN 扩展上下文：原始长度={original_max_len}, 因子={args.yarn_factor}, 目标长度={TARGET_MAX_LEN}")
    text_config.original_max_position_embeddings = original_max_len
    text_config.rope_scaling = {
        "type": "yarn", "factor": args.yarn_factor, "original_max_position_embeddings": original_max_len
    }
    text_config.max_position_embeddings = TARGET_MAX_LEN

    logger.info(f"正在从 {args.ckpt_path} 加载模型...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.ckpt_path, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=True,
    )
    if is_main_process:
        dist.barrier()
    model.to(device)
    logger.info(f"[Rank {local_rank}] 模型已成功加载到设备 {device}。")

    # --- 4. 数据加载与分布式切分 (已修改为处理 MLVU JSON) ---
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
                "duration": video_info['duration'],
                'question': conv['question'],
                'options': conv['choices'],
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
    # output_jsonl_path = "/data1/nyh/llava_onevisioncopy/eval_results/mlvu/qwen2.5vl_3b_mlvu2.0/output/qwen2.5vl_3b_mlvu2.0_20251210_101855_rank0.jsonl"
    if osp.exists(output_jsonl_path):
        logger.info(f"[Rank {local_rank}] 从已有输出文件恢复进度: {output_jsonl_path}")
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
            f"[Rank {local_rank}] 跳过 {original_count - len(task_data)} 个已完成任务，剩余 {len(task_data)} 个。")

    # --- 5. 推理循环 (采用流式处理) ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    progress_bar = tqdm(task_data, total=len(
        task_data), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for item in progress_bar:
        try:
            print(item)
            time1 = item['duration']
            # if time1 <= 500:
            #     model.model.inter_frame_threshold = 0.9  # 帧间相似度阈值
            #     model.intra_frame_threshold = 0.95  # 帧内合并阈值
            # if time1 <= 3600:
            #     model.model.inter_frame_threshold = 0.8  # 帧间相似度阈值
            #     model.intra_frame_threshold = 0.95  # 帧内合并阈值
            # else:
            #     model.model.inter_frame_threshold = 0.55  # 帧间相似度阈值
            #     model.intra_frame_threshold = 0.75  # 帧内合并阈值
            # 1533cuda4
            duration_in_seconds = time1
            # 根据视频时长动态调整采样率
            if duration_in_seconds < 1800.0:
                current_sample_fps = 0.5
            else:
                current_sample_fps = 0.2
            if duration_in_seconds <= 250:
                model.model.inter_frame_threshold = 0.95  # 1.1  # 帧间相似度阈值
                model.model.intra_frame_threshold = 0.97  # 0.95  # 帧内合并阈值
            elif duration_in_seconds <= 1800:
                model.model.inter_frame_threshold = 0.8  # 1.1  # 帧间相似度阈值
                model.model.intra_frame_threshold = 0.85  # 0.95  # 帧内合并阈值
            else:
                model.model.inter_frame_threshold = 0.72  # 1.1  # 帧间相似度阈值
                model.model.intra_frame_threshold = 0.82  # 0.95  # 帧内合并阈值

            video_path = osp.join(args.video_dir, item['video_path'])
            if not osp.exists(video_path):
                logger.warning(f"视频文件不存在，跳过: {video_path}")
                continue

            # --- 步骤 A: 准备视频数据 ---
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            sample_fps = 0.5 if time1 < 1800 else 0.2
            frame_interval = math.ceil(
                video_fps / sample_fps) if video_fps > 0 and video_fps is not None else total_frames
            if frame_interval <= 0:
                frame_interval = 1
            frame_indices = list(range(0, total_frames, frame_interval))

            if len(frame_indices) > 44800:  # 限制最大帧数
                indices_to_sample = [
                    i * len(frame_indices) // 44800 for i in range(44800)]
                frame_indices = [frame_indices[i] for i in indices_to_sample]

            video_frames = [Image.fromarray(
                frame) for frame in vr.get_batch(frame_indices).asnumpy()]
            if not video_frames:
                logger.warning(f"从视频 {video_path} 未采样到任何帧，跳过。")
                continue

            # --- 步骤 B: 视频流式预填充 ---
            past_key_values = None
            logical_seq_len = 0
            first_prompt = "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user\n<video>"
            inputs = processor(text=first_prompt, videos=[
                               video_frames[0]], return_tensors="pt").to(device)
            chunk_len = inputs.input_ids.shape[1]
            position_ids = torch.arange(
                logical_seq_len, logical_seq_len + chunk_len, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True,
                                position_ids=position_ids)
                past_key_values = outputs.past_key_values
            logical_seq_len += chunk_len

            if len(video_frames) > 1:
                for frame in video_frames[1:]:
                    inputs = processor(text="<video>", videos=[
                                       frame], return_tensors="pt").to(device)
                    chunk_len = inputs.input_ids.shape[1]
                    position_ids = torch.arange(
                        logical_seq_len, logical_seq_len + chunk_len, device=device).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(input_ids=inputs.input_ids, pixel_values_videos=inputs.pixel_values_videos,
                                        use_cache=True, past_key_values=past_key_values, position_ids=position_ids)
                        past_key_values = outputs.past_key_values
                    logical_seq_len += chunk_len

            # --- 步骤 C: 问题预填充与答案生成 ---
            options_str = '\n'.join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
            question_text = prompt_template.format(
                item['question'], options_str)
            final_prompt = f"{question_text}<|im_end|>\n<|im_start|>assistant\n"
            inputs_question = processor(
                text=final_prompt, return_tensors="pt").to(device)
            question_input_ids = inputs_question.input_ids
            question_attention_mask = inputs_question.attention_mask
            question_len = question_input_ids.shape[1]

            with torch.no_grad():
                if past_key_values is not None:
                    physical_past_len = past_key_values[0][0].shape[-2]
                    batch_size = question_input_ids.shape[0]
                    past_mask = torch.ones(
                        (batch_size, physical_past_len), device=device, dtype=torch.long)
                    attention_mask = torch.cat(
                        [past_mask, question_attention_mask], dim=1)
                    cache_position = torch.arange(
                        logical_seq_len, logical_seq_len + question_len, device=device)

                    generated_ids = model.generate(
                        input_ids=question_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        max_new_tokens=20,  # 稍微增加token以容纳更长的文本答案
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                else:
                    generated_ids = model.generate(
                        input_ids=question_input_ids,
                        attention_mask=question_attention_mask,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

            input_token_len = inputs_question.input_ids.shape[1]
            generated_ids_only_answer = generated_ids[:, input_token_len:]

            # --- 步骤 D: 解码与评估 (已修改) ---
            response = processor.batch_decode(
                generated_ids_only_answer, skip_special_tokens=True)[0].strip()

            try:
                answer_idx = item['options'].index(item['answer'])
                correct_letter = chr(65 + answer_idx)
            except ValueError:
                logger.warning(
                    f"答案 '{item['answer']}' 不在选项列表 {item['options']} 中，跳过问题 {item['question_id']}")
                continue

            # ==========================> 新的提取逻辑开始 <==========================
            pred_letter = extract_directly(response)

            # 如果正则表达式没有找到有效的答案字母...
            if not pred_letter:
                # ...则尝试通过文本匹配来寻找最佳选项
                logger.info(f"正则提取失败，模型原始输出为: '{response}'. 尝试文本匹配...")
                pred_letter = find_best_match(response, item['options'])
                if pred_letter:
                    logger.info(f"文本匹配成功，找到最相似选项: {pred_letter}")
                else:
                    # 确保 pred_letter 是一个空字符串而不是 None
                    pred_letter = extract_characters_regex(response)
                    logger.warning(f"文本匹配也失败了，无法从输出 '{response}' 中确定答案。")
            # ==========================> 新的提取逻辑结束 <==========================

            is_correct = (pred_letter == correct_letter)
            question_type = item['question_type']

            if is_correct:
                cnt_correct['overall'] += 1
                cnt_correct[question_type] += 1
            cnt_total['overall'] += 1
            cnt_total[question_type] += 1

            output_dict = {
                'question_id': item['question_id'], 'video_id': item['video_id'], 'question_type': question_type,
                'question': item['question'], 'options': item['options'], 'answer': item['answer'],
                'correct_letter': correct_letter, 'response': response, 'prediction': pred_letter, 'is_correct': is_correct,
            }
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理 {item.get('question_id', 'N/A')} 时发生错误: {e}")
            traceback.print_exc()

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"[Rank {local_rank}] 推理完成，耗时: {cost_time} 秒。")

    # --- 6. 结果汇总与清理 ---
    dist.barrier()
    all_question_types = sorted(
        list(set(q['question_type'] for q in all_questions)))
    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    for q_type in all_question_types:
        stats_list.extend([cnt_total[q_type], cnt_correct[q_type]])

    stats_tensor = torch.tensor(stats_list, dtype=torch.long, device=device)
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    if is_main_process:
        logger.info("--- 所有进程已完成，开始汇总结果 ---")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)

        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()
        if total_processed == 0:
            logger.info("所有进程均未处理任何问题。")
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
                    f"  - {q_type:<25}: {correct_cat}/{total_cat} = {acc_cat:.2f}%")

        logger.info(
            f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (此为单个进程耗时)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
