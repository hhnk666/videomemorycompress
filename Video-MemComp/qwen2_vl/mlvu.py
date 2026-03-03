#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import torch
import torch.distributed as dist
from decord import VideoReader
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

# 确保自定义模型和工具脚本在 Python 路径中
# 假设 modeling_qwen2_vl.py 和 qwen_vl_utils.py 与此脚本在同一目录或上层目录
try:
    # 这里的路径可能需要根据您的项目结构进行调整
    sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
    from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保 'modeling_qwen2_vl.py' 和 'qwen_vl_utils.py' 脚本在您的PYTHONPATH中，并且相关依赖已安装。")
    sys.exit(1)


# --- 默认参数定义 ---
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct" # 默认模型路径
TASK_JSON = "/path/to/your/mlvu/dev_debug_mc.json"
VIDEO_DIR = "/path/to/your/mlvu/root/"
RESULT_DIR = "eval_results/mlvu"
N_FRAMES = 768  # 默认采样帧数
MIN_FRAMES = 8
MAX_FRAMES = 768 # 允许的最大帧数

# --- 提示模板 ---
PROMPT_TEMPLATE = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, etc.) of the correct option.
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
def extract_answer_letter(s: str) -> str:
    """从模型响应中稳健地提取答案选项字母"""
    s = s.strip()
    # 优先匹配 "The answer is A" 这种结构
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            break

    # 查找开头的 (A) 或 A. 这种格式
    matches = re.search(r"^\(?([A-E])\)?\.?\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    # 如果开头没有，则查找文本中出现的第一个字母选项
    matches = re.search(r"\b([A-E])\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
        
    return ""


def merge_jsonl_files(base_path, world_size):
    """合并所有 rank 生成的 jsonl 文件"""
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
            # 可选：合并后删除分片文件
            # os.remove(file_path)

    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"成功合并 {len(merged_data)} 条记录到 {merged_file_path}")
    return merged_file_path


# --- 主脚本 ---
def main():
    parser = argparse.ArgumentParser(description="在 MLVU 数据集上进行分布式评测。")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型权重路径 (本地或Hugging Face Hub)")
    parser.add_argument("--task_json", type=str, default=TASK_JSON, help="MLVU 任务的JSON文件路径")
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR, help="存放视频文件的根目录")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR, help="存放结果、日志和输出的目录")
    parser.add_argument("--run_name", type=str, default="mlvu_eval_run", help="本次运行的名称，用于生成文件名")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES, help="为每个视频采样的目标帧数")
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES, help="视频处理的最小帧数")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="视频处理的最大帧数")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="模型生成新token的最大数量")
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
        os.makedirs(osp.join(args.result_dir, 'output'), exist_ok=True)
        os.makedirs(osp.join(args.result_dir, 'log'), exist_ok=True)
    dist.barrier()
    
    log_path = osp.join(args.result_dir, 'log', f"{args.run_name}_{curr_time}_rank{local_rank}.log")
    output_jsonl_path = osp.join(args.result_dir, 'output', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if is_main_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    
    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # --- 3. 模型和预处理器加载 ---
    torch.manual_seed(1234)
    if not is_flash_attn_2_available():
        logger.warning("Flash Attention 2 不可用，将回退到 SDPA。性能可能会受影响。")
        attn_implementation = "sdpa"
    else:
        attn_implementation = "flash_attention_2"
        logger.info("Flash Attention 2 可用，将用于加速。")
    
    if not is_main_process:
        dist.barrier()  # 等待主进程下载模型
    
    logger.info(f"[Rank {local_rank}] 开始加载模型: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    if is_main_process:
        dist.barrier() # 主进程加载完毕，通知其他进程
    logger.info(f"[Rank {local_rank}] 模型和预处理器加载完毕。")

    # --- 4. 数据加载与分布式切分 ---
    if not osp.exists(args.task_json):
        logger.error(f"任务文件未找到: {args.task_json}")
        sys.exit(1)
        
    with open(args.task_json, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    # 扁平化数据结构，方便处理
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

    # 分布式数据切分
    num_samples = len(all_questions)
    rank_indices = list(range(num_samples))[local_rank::world_size]
    task_data = [all_questions[i] for i in rank_indices]
    logger.info(f"[Rank {local_rank}] 数据扁平化和切分完毕，本进程将处理 {len(task_data)} / {num_samples} 个问题。")

    # 进度恢复
    # output_jsonl_path="/root/bayes-gpfs-c9f955b46c074f048c960ec71693fa58/nyh/qwen2vl/eval_results/mlvu/output/mlvu0.250.55_20251112_150744_rank0.jsonl"
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
        task_data = [item for item in task_data if item['question_id'] not in processed_ids]
        logger.info(f"[Rank {local_rank}] 从已有输出文件中恢复进度，跳过 {original_count - len(task_data)} 个已完成任务。")

    # --- 5. 推理循环 ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)
    progress_bar = tqdm(task_data, total=len(task_data), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for item in progress_bar:
        try:
            video_path = osp.join(args.video_dir, item['video_path'])
            if not osp.exists(video_path):
                logger.warning(f"视频文件未找到: {video_path}，跳过问题 {item['question_id']}")
                continue

            # 动态调整采样帧数
            try:
                vr = VideoReader(video_path)
                video_frame_count = len(vr)
            except Exception as e:
                logger.error(f"无法用decord读取视频 {video_path}, 跳过. 错误: {e}")
                traceback.print_exc()
                continue
            
            final_nframes = max(args.min_frames, min(args.n_frames, video_frame_count))

            # 构建输入
            options_str = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(item['choices'])])
            query = PROMPT_TEMPLATE.format(item['question'], options_str)
            
            messages = [{"role": "user", "content": [
                {"type": "video", "video": f"file://{osp.abspath(video_path)}", "nframes": final_nframes, "max_frames": args.max_frames},
                {"type": "text", "text": query}
            ]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            patch_size = model.config.vision_config.patch_size
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=patch_size)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # 推理
            gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
            generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            # 结果评估
            try:
                answer_idx = item['choices'].index(item['answer'])
                correct_letter = chr(65 + answer_idx)
            except ValueError:
                logger.warning(f"答案 '{item['answer']}' 不在选项列表 {item['choices']} 中，跳过问题 {item['question_id']}")
                continue

            pred_letter = extract_answer_letter(response)
            is_correct = (pred_letter == correct_letter)
            question_type = item['question_type']
            
            if is_correct:
                cnt_correct['overall'] += 1
                cnt_correct[question_type] += 1
            cnt_total['overall'] += 1
            cnt_total[question_type] += 1

            # 保存单条结果
            output_dict = {
                'question_id': item['question_id'], 'video_id': item['video_id'],
                'question_type': question_type, 'question': item['question'], 'choices': item['choices'],
                'answer': item['answer'], 'correct_letter': correct_letter,
                'response': response, 'prediction': pred_letter, 'is_correct': is_correct,
            }
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"[Rank {local_rank}] 处理 {item.get('question_id', 'N/A')} 时发生严重错误: {e}")
            traceback.print_exc()

    # --- 6. 结果汇总与清理 ---
    dist.barrier()
    logger.info(f"[Rank {local_rank}] 推理完成，等待所有进程并开始汇总...")
    
    # 收集所有可能的问题类型
    all_question_types = sorted(list(set(q['question_type'] for q in all_questions)))
    
    # 将本进程的统计数据放入Tensor中
    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    for q_type in all_question_types:
        stats_list.extend([cnt_total[q_type], cnt_correct[q_type]])
    stats_tensor = torch.tensor(stats_list, dtype=torch.long, device=f'cuda:{local_rank}')
    
    # 使用 all_reduce 聚合所有进程的数据
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    if is_main_process:
        logger.info("--- 所有进程已完成，开始最终结果汇总 ---")
        base_output_path = osp.join(args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)

        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()
        
        if total_processed == 0:
            logger.info("所有进程均未成功处理任何问题。")
        else:
            accuracy = 100 * total_correct / total_processed
            logger.info(f"【总结果】: 总数={total_processed}, 正确={total_correct}, 准确率={accuracy:.2f}%")

        logger.info("--- 按问题类型分类准确率 ---")
        for i, q_type in enumerate(all_question_types):
            total_cat = stats_tensor[2 + i*2].item()
            correct_cat = stats_tensor[3 + i*2].item()
            if total_cat > 0:
                acc_cat = 100 * correct_cat / total_cat
                logger.info(f"  - {q_type:<25}: {correct_cat}/{total_cat} = {acc_cat:.2f}%")
        
        cost_time = int(time.time() - start_time)
        logger.info(f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()