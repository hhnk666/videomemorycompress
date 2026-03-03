import debugpy
import traceback
# <-- 1. 增加了 AutoConfig 导入
from transformers import LlavaOnevisionProcessor, AutoConfig
from llava_onevision import LlavaOnevisionForConditionalGeneration
import torch
import torch.distributed as dist
import json
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
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

# --- 默认参数定义 ---
RUN_NAME = "llava_onevision_videomme_streaming_test_yarn8"
CKPT_PATH = "/data1/nyh/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf"
TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval/result_videomme_llava_onevision"
SAMPLE_FPS = 0.2
YARN_FACTOR = 2.0
# --- 提示模板 ---
prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: {}
Options: {}
"""

# try:
#     debugpy.listen(("localhost", 9551))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

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
        "The best option is", "The correct option is", "Best answer:", "Best option:", "The correct option is:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
            if s.startswith(":") or s.startswith("."):
                s = s[1:].strip()
    matches = re.search(r"^\s*([A-D])", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    matches = re.search(r"[(\[]\s*([A-D])\s*[)\]]", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()
    matches = re.search(r"\b([A-D])\b", s, re.IGNORECASE)
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
                        continue
    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"成功合并 {len(merged_data)} 条记录到 {merged_file_path}")
    return merged_file_path


def main():
    parser = argparse.ArgumentParser(
        description="在 Video-MME 数据集上使用 LLaVA OneVision 进行流式分布式评测。")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sample_fps", type=float, default=SAMPLE_FPS)
    # --- 2. 增加了新的命令行参数 ---
    parser.add_argument("--extend_context", action='store_true',
                        default=True, help="启用上下文扩展 (YaRN)")
    parser.add_argument("--target_max_len", type=int,
                        default=128000, help="YaRN 扩展的目标上下文长度")
    parser.add_argument("--yarn_factor", type=float,
                        default=6.0, help="YaRN 的缩放因子")
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

    # --- 3. 模型加载 (核心修改部分) ---
    torch.manual_seed(1919810)
    logger.info("设置全局随机种子为 1234")

    if not is_main_process:
        dist.barrier()

    logger.info(f"正在从 {args.ckpt_path} 加载处理器...")
    processor = LlavaOnevisionProcessor.from_pretrained(args.ckpt_path)

    logger.info(f"正在从 {args.ckpt_path} 加载模型配置...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)

    if 1:
        # 定位到 text_config
        text_config = getattr(config, 'text_config', config)

        # 获取原始最大长度 (32768)
        original_max_len = text_config.max_position_embeddings

        # 根据缩放因子计算目标长度
        TARGET_MAX_LEN = int(original_max_len * YARN_FACTOR)

        print(f"正在通过 YaRN 扩展上下文：")
        print(f"  - 原始最大长度: {original_max_len}")
        print(f"  - YaRN 缩放因子 (λ): {YARN_FACTOR}")
        print(f"  - 计算出的目标长度: {TARGET_MAX_LEN}")

        # 🚨 关键修复 A: 将【原始长度】保存到新字段中
        text_config.original_max_position_embeddings = original_max_len

        # 设置 rope_scaling 参数
        text_config.rope_scaling = {
            "type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": original_max_len
        }

        # 最后，将 max_position_embeddings 更新为【目标长度】
        text_config.max_position_embeddings = TARGET_MAX_LEN

        print("上下文扩展已成功配置。")

    logger.info(f"正在从 {args.ckpt_path} 加载模型...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,  # <-- 传递修改后的配置
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # <-- 增加 attention 优化
        trust_remote_code=True,
    )

    if is_main_process:
        dist.barrier()

    model.to(device)
    logger.info(f"[Rank {local_rank}] 模型已成功加载到设备 {device}。")

    # --- 4. 数据加载与分布式切分 ---
    task_df_full = pd.read_parquet(args.task_parquet)
    num_samples = len(task_df_full)
    indices = list(range(num_samples))
    rank_indices = indices[local_rank::world_size]
    task_df = task_df_full.iloc[rank_indices].copy()
    # task_df = task_df[task_df['duration'] != 'short'].copy()
    logger.info(
        f"[Rank {local_rank}] 数据切分完毕，将处理 {len(task_df)} / {num_samples} 个样本。")

    processed_ids = set()
    if osp.exists(output_jsonl_path):
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
        logger.info(
            f"[Rank {local_rank}] 跳过 {len(processed_ids)} 个已完成任务，剩余 {len(task_df)}/{original_count} 个。")

    # --- 5. 推理循环 (采用流式处理) ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    progress_bar = tqdm(task_df.itertuples(), total=len(
        task_df), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for row in progress_bar:
        try:
            video_path = osp.join(args.video_dir, row.videoID + '.mp4')
            if not osp.exists(video_path):
                logger.warning(f"视频文件不存在，跳过: {video_path}")
                continue

            # --- 步骤 A: 准备视频数据 (已增加动态采样率和帧数上限) ---
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            duration_in_seconds = total_frames / video_fps if video_fps > 0 else 0

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


            # 计算帧间隔
            frame_interval = math.ceil(
                video_fps / current_sample_fps) if current_sample_fps > 0 else total_frames
            if frame_interval <= 0:
                frame_interval = 1

            # 初始采样
            frame_indices = [i for i in range(0, total_frames, frame_interval)]



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
                    chunk_prompt = "<video>"
                    inputs = processor(text=chunk_prompt, videos=[
                                       frame], return_tensors="pt").to(device)
                    chunk_len = inputs.input_ids.shape[1]
                    position_ids = torch.arange(
                        logical_seq_len, logical_seq_len + chunk_len, device=device).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(
                            input_ids=inputs.input_ids,
                            pixel_values_videos=inputs.pixel_values_videos,
                            use_cache=True,
                            past_key_values=past_key_values,
                            position_ids=position_ids
                        )
                        past_key_values = outputs.past_key_values
                    logical_seq_len += chunk_len

            # --- 步骤 C: 问题预填充与答案生成 ---
            question_text = prompt_template.format(row.question, '\n'.join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(row.options)]))
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
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                else:
                    generated_ids = model.generate(
                        input_ids=question_input_ids,
                        attention_mask=question_attention_mask,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
            input_token_len = inputs_question.input_ids.shape[1]
            generated_ids_only_answer = generated_ids[:, input_token_len:]

            # --- 步骤 D: 解码与评估 ---
            output_text = processor.batch_decode(
                generated_ids_only_answer, skip_special_tokens=True)
            response = output_text[0].strip()

            is_correct = extract_characters_regex(response) == row.answer
            if is_correct:
                cnt_correct['overall'] += 1
            cnt_total['overall'] += 1

            output_dict = {
                'question_id': row.question_id,
                'videoID': row.videoID,
                'question': row.question,
                'options': row.options.tolist(),
                'answer': row.answer,
                'response': response,
                'is_correct': is_correct,
            }
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理 {row.question_id} 时发生错误: {e}")
            traceback.print_exc()

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"[Rank {local_rank}] 推理完成，耗时: {cost_time} 秒。")

    # --- 6. 结果汇总与清理 ---
    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    stats_tensor = torch.tensor(stats_list, dtype=torch.long, device=device)

    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

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
        logger.info(
            f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (此为单个进程耗时，非总和)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
