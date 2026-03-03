#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
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
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from transformers import AutoTokenizer

try:
    from InternVL3_5 import InternVLChatModel
except ImportError:
    print("导入失败。尝试从 transformers 导入...")
    from transformers import AutoModelForCausalLM as InternVLChatModel

# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

# --- 提示模板 (MLVU Format) ---
prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, etc.) of the correct option.
Question: {}
Options:
{}
The best answer is:"""

# ==============================================================================
# InternVL 视频预处理函数 (完全继承你的代码)
# ==============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = 0, max_frame / fps
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, max_frame).astype(int)
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img_tiles = dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img_tiles]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None, None

# ==============================================================================
# 辅助函数
# ==============================================================================


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()

    # MLVU 选项可能到 E
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


def main():
    parser = argparse.ArgumentParser(
        description="在 MLVU 数据集上进行 InternVL 分布式评测。")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--task_json", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=100)
    args = parser.parse_args()

    # --- 1. 分布式环境初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    torch.manual_seed(1919810)

    # --- 2. 路径和日志设置 ---
    curr_time = datetime.now().strftime("%Y%m%d")
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
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

    logger.info(f"---[Rank {local_rank}/{world_size}] 进程启动 ---")

    # --- 3. 模型和 Tokenizer 加载 ---
    logger.info(f"[Rank {local_rank}] 正在加载 InternVL 模型...")
    # 注意：在分布式环境中，务必使用 device_map={"": local_rank} 替代 auto
    model = InternVLChatModel.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        device_map={"": local_rank}
    ).eval()
    model.inter_frame_threshold = 0.5
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=128, do_sample=False)
    logger.info(f"[Rank {local_rank}] 模型加载完成。")

    # --- 4. 数据加载与分布式切分 ---
    logger.info(f"[Rank {local_rank}] 正在加载 MLVU 数据...")
    with open(args.task_json, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    all_questions = []
    for video_info in full_data:
        for i, conv in enumerate(video_info['conversations']):
            question_id = f"{video_info['video_id']}_q{i}"
            all_questions.append({
                'video_id': video_info['video_id'],
                'video_path': video_info['video_path'],
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

    # 断点续传逻辑
    processed_ids = set()
    if osp.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['question_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
    if processed_ids:
        task_data = [
            item for item in task_data if item['question_id'] not in processed_ids]
        logger.info(
            f"[Rank {local_rank}] 恢复进度，跳过已完成的任务。剩余任务: {len(task_data)}")

    # --- 5. 推理循环 ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    progress_bar = tqdm(task_data, total=len(task_data),
                        disable=not is_main_process, desc=f"Inference")

    for item in progress_bar:
        try:
            video_path = osp.join(args.video_dir, item['video_path'])
            if not osp.exists(video_path):
                logger.warning(f"视频文件未找到: {video_path}")
                continue

            # 加载视频并设置帧数
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=args.num_frames, max_num=1)
            if pixel_values is None:
                continue

            pixel_values = pixel_values.to(torch.bfloat16).cuda(local_rank)

            # 组装 Prompt
            options_str = "\n".join(
                [f"{chr(65+i)}) {choice}" for i, choice in enumerate(item['choices'])])
            question_text = prompt_template.format(
                item['question'], options_str)
            video_prefix = ''.join(
                [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            full_question = video_prefix + question_text

            # 模型推理
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                full_question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )

            # 清理 response
            if 'assistant\n' in response:
                response = response.split('assistant\n')[-1].strip()

            # 校验答案
            try:
                answer_idx = item['choices'].index(item['answer'])
                correct_letter = chr(65 + answer_idx)
            except ValueError:
                logger.warning(f"答案 '{item['answer']}' 不在选项列表中，跳过。")
                continue

            pred_letter = extract_characters_regex(response)
            is_correct = (pred_letter == correct_letter)
            question_type = item['question_type']

            if is_correct:
                cnt_correct['overall'] += 1
                cnt_correct[question_type] += 1
            cnt_total['overall'] += 1
            cnt_total[question_type] += 1

            # 记录结果
            output_dict = {
                'question_id': item['question_id'],
                'video_id': item['video_id'],
                'question_type': question_type,
                'question': item['question'],
                'choices': item['choices'],
                'answer': item['answer'],
                'correct_letter': correct_letter,
                'response': response,
                'prediction': pred_letter,
                'is_correct': is_correct,
            }
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理报错 {item.get('question_id')}: {e}")

    # --- 6. 结果汇总与指标统计 ---
    dist.barrier()
    logger.info(f"[Rank {local_rank}] 推理完成，等待汇总...")

    # 动态获取所有问题类型，确保各 GPU 对齐
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

        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()

        if total_processed > 0:
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
            f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
