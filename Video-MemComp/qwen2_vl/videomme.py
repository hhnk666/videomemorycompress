import os
import sys
import json
import argparse
import logging
import time
import traceback  # <<< 修复点 1：全局导入 traceback
from datetime import datetime
from collections import defaultdict
import re

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from decord import VideoReader
from transformers import AutoProcessor
# 确保您的模型代码在Python路径中
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# 为了通用性，我们直接从Hugging Face加载，如果使用本地模型，请取消上面的注释并修改from_pretrained路径
# from transformers import Qwen2VLForConditionalGeneration

# --- 提示模板 ---
PROMPT_TEMPLATE = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
Options: {options}
The best answer is:"""

# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

# import debugpy
# try:
#     debugpy.listen(("localhost", 9571))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
# --- 辅助函数 ---

def setup_logging(args, local_rank):
    """配置日志记录器，区分主进程和子进程"""
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.run_name}_{curr_time}_rank{local_rank}.log"
    log_path = os.path.join(args.result_dir, 'log', log_filename)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if local_rank == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    return curr_time


def extract_answer(s: str) -> str:
    """从模型响应中提取答案选项字母 (A, B, C, or D)"""
    s = s.strip()
    # 查找括号中的答案，例如 "The answer is (A)."
    match = re.search(r"\(([A-D])\)", s)
    if match:
        return match.group(1)
    
    # 查找紧跟在特定短语后的答案
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            break # 找到前缀后就跳出循环处理

    # 优先匹配开头的字母
    matches = re.search(r"^[A-D]", s)
    if matches:
        return matches.group(0)

    # 其次匹配单词边界的独立字母
    matches = re.search(r"\b[A-D]\b", s)
    if matches:
        return matches.group(0)

    return ""


def merge_jsonl_files(base_path, world_size):
    """合并由不同 rank 进程生成的多个 jsonl 文件为一个文件"""
    merged_data = []
    for rank in range(world_size):
        file_path = f"{base_path}_rank{rank}.jsonl"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        merged_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析文件 {file_path} 中的行: {line.strip()}")
                        continue
            os.remove(file_path) # 合并后删除原始文件

    if not merged_data:
        logger.warning(f"没有找到任何 {base_path}_rank*.jsonl 文件进行合并。")
        return

    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"成功合并 {len(merged_data)} 条记录到 {merged_file_path}")


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen2-VL 在 Video-MME 数据集上进行分布式评测")
    # --- 路径和名称参数 ---
    parser.add_argument("--model_path", type=str, required=True, help="Qwen2-VL 模型权重路径 (本地路径或Hugging Face ID)")
    parser.add_argument("--video_dir", type=str, required=True, help="Video-MME 视频文件所在目录")
    parser.add_argument("--task_parquet", type=str, required=True, help="Video-MME 任务描述的 parquet 文件路径")
    parser.add_argument("--result_dir", type=str, default="eval_results/videomme", help="存放结果、日志和输出的目录")
    parser.add_argument("--run_name", type=str, default="qwen2vl_videomme_eval", help="本次运行的名称，用于生成文件名")

    # --- 模型和推理参数 ---
    parser.add_argument("--n_frames", type=int, default=768, help="为每个视频采样的目标帧数 (建议减小默认值)") # <<< 建议修改点
    parser.add_argument("--min_frames", type=int, default=8, help="视频处理的最小帧数") # <<< 建议修改点
    parser.add_argument("--max_new_tokens", type=int, default=128, help="模型生成新token的最大数量")

    args = parser.parse_args()

    # --- 1. 分布式环境初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    # --- 2. 路径和日志设置 ---
    if is_main_process:
        for subdir in ['output', 'log']:
            os.makedirs(os.path.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()

    curr_time = setup_logging(args, local_rank)
    output_jsonl_path = os.path.join(args.result_dir, 'output', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")

    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # --- 3. 模型和预处理器加载 ---
    if not is_main_process: dist.barrier() # 等待主进程下载模型
    
    logger.info(f"[Rank {local_rank}] 开始加载模型: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=f"cuda:{local_rank}",
        attn_implementation="sdpa", # 推荐开启以加速
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    if is_main_process: dist.barrier() # 主进程加载完毕，通知其他进程



    # --- 4. 数据加载与分布式切分 ---
    try:
        task_df_full = pd.read_parquet(args.task_parquet)
    except Exception as e:
        logger.error(f"无法读取Parquet文件 {args.task_parquet}。错误: {e}")
        dist.destroy_process_group()
        sys.exit(1)

    num_samples = len(task_df_full)
    indices = list(range(num_samples))
    rank_indices = indices[local_rank::world_size]
    task_df = task_df_full.iloc[rank_indices].copy()
    task_df = task_df[task_df['duration'] != 'short'].copy()
    logger.info(f"[Rank {local_rank}] 数据切分完毕，将处理 {len(task_df)} / {num_samples} 个样本。")

    # --- 进度恢复 ---
    processed_ids = set()
    # output_jsonl_path="/root/bayes-gpfs-c9f955b46c074f048c960ec71693fa58/nyh/qwen2vl/videomme/output/0.50.756000_20251105_221045_rank0.jsonl"
    output_jsonl_path="/root/bayes-gpfs-c9f955b46c074f048c960ec71693fa58/nyh/qwen2vl/videomme/output/0.20.556000_20251130_001641_rank0.jsonl"
    if os.path.exists(output_jsonl_path):
        logger.info(f"[Rank {local_rank}] 从已有输出文件恢复进度: {output_jsonl_path}")
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['question_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
    if processed_ids:
        original_count = len(task_df)
        task_df = task_df[~task_df['question_id'].isin(processed_ids)]
        logger.info(f"[Rank {local_rank}] 跳过 {len(processed_ids)} 个已完成任务，剩余 {len(task_df)}/{original_count} 个。")

    # --- 5. 推理循环 ---
    start_time = time.time()
    results = []

    progress_bar = tqdm(task_df.itertuples(), total=len(task_df), disable=not is_main_process, desc=f"Rank {local_rank} Inference")
    for row in progress_bar:
        try:
            video_path = os.path.join(args.video_dir, row.videoID + '.mp4')
            if not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}, 跳过样本 {row.question_id}")
                continue

            # --- 动态帧数钳位逻辑 ---
            try:
                vr = VideoReader(video_path) 
                video_frame_count = len(vr)
            except Exception as e:
                logger.error(
                    f"无法读取视频 {video_path}, 跳过。\n"
                    f"  错误类型: {type(e).__name__}\n"
                    f"  错误信息: {e}\n"
                    f"  追溯信息: {traceback.format_exc()}"
                )
                continue

            final_nframes = max(args.min_frames, min(args.n_frames, video_frame_count))
            if final_nframes != args.n_frames:
                logger.info(f"ID {row.question_id}: 期望帧数 {args.n_frames} 超出视频有效范围 [1, {video_frame_count}]。已调整为 {final_nframes}。")
            
# ... 代码的其他部分 ...

            # --- 构建输入 (基于 qwen-vl-utils 源码的最终正确方案) ---
            options_str = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(row.options)])
            question_text = PROMPT_TEMPLATE.format(question=row.question, options=options_str)

            # 1. 准备 messages 列表，使用 "file://" URI 格式，并在字典中传递 nframes
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{os.path.abspath(video_path)}",
                        "nframes": final_nframes,
                    },
                    {"type": "text", "text": question_text},
                ]
            }]

            # 2. 准备纯文本部分
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. 从模型配置中获取正确的 patch_size
            patch_size = model.config.vision_config.patch_size

            # 4. 使用官方辅助函数和正确的参数提取视觉输入
            image_inputs, video_inputs = process_vision_info(
                messages,
                image_patch_size=patch_size
            )

            # 5. 将所有部分组合成最终输入
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to(model.device)

            # --- 推理 ---
            # ... 后续代码不变 ...

            # --- 推理 ---
            # ... 后续的 model.generate 和解码代码保持不变 ...
            # --- 推理 ---
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # --- 保存结果 ---
            output_dict = {
                'question_id': row.question_id,
                'video_id': row.videoID,
                'question': row.question,
                'options': row.options.tolist(),
                'answer': row.answer,
                'response': response,
                'extracted_answer': extract_answer(response)
            }
            
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
            results.append(output_dict)

        except Exception as e:
            logger.error(f"[Rank {local_rank}] 处理 {row.question_id} 时发生严重错误: {e}")
            traceback.print_exc() # <<< 现在这里可以正常工作了

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"[Rank {local_rank}] 推理完成，耗时: {cost_time} 秒。")

    # --- 6. 结果汇总与清理 ---
    dist.barrier()
    if is_main_process:
        logger.info("--- 所有进程已完成，开始汇总结果 ---")
        base_output_path = os.path.join(args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)

        # --- 计算最终准确率 ---
        merged_file_path = f"{base_output_path}_merged.jsonl"
        if os.path.exists(merged_file_path):
            final_df = pd.read_json(merged_file_path, lines=True)
            total = len(final_df)
            correct = (final_df['extracted_answer'] == final_df['answer']).sum()
            if total > 0:
                accuracy = 100 * correct / total
                logger.info(f"【总结果】: 总数={total}, 正确={correct}, 准确率={accuracy:.2f}%")
            else:
                logger.info("没有有效的评测结果。")
        else:
            logger.warning("合并后的结果文件不存在，无法计算最终准确率。")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()