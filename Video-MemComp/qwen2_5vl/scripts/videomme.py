import debugpy
import traceback
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available
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

# 确保自定义模型的定义在 Python 路径中
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from qwen_vl_utils import process_vision_info
    from qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLSdpaAttention,
    )
except ImportError:
    print("导入失败。请确保 'qwen_vl_utils' 和 'qwen2_5_vl' 模块可访问。")
    sys.exit(1)


# --- 默认参数定义 ---
RUN_NAME = "feature_0d5_dynamic_attn"
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"
TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval/result_videomme"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 448 * 448
MAX_FRAMES = 1145
MIN_FRAMES = 4
FPS = 0.25
# --- 提示模板 ---
prompt = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{}
Options: {}
The best answer is:"""
# try:
#     # 5678 is the default attach port in the VS Code debug configurations.
#     debugpy.listen(("localhost", 9511))
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


def time_to_seconds(time_str):
    """将时间字符串 H:M:S 转换为总秒数。"""
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


def extract_characters_regex(s):
    """从模型响应中提取答案选项字母。"""
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    if len(s.split()) > 10 and not re.search(r"\b[A-D]\b", s):
        return ""

    matches = re.search(r"^[A-D]", s)  # 优先匹配开头的字母
    if matches:
        return matches.group(0)

    matches = re.search(r"\b[A-D]\b", s)  # 其次匹配独立的字母
    if matches:
        return matches.group(0)

    return ""


def merge_jsonl_files(base_path, world_size):
    """合并由不同 rank 进程生成的多个 jsonl 文件为一个文件。"""
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
    parser = argparse.ArgumentParser(description="在 Video-MME 数据集上进行分布式评测。")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fps", type=float, default=FPS)
    args = parser.parse_args()

    # --- 1. 分布式环境初始化 ---
    # --- 1. 分布式环境初始化 (已修正) ---
    # 首先获取 rank，这是决定使用哪个 GPU 的关键
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 关键改动：首先设置当前进程要使用的 CUDA 设备
    # 这确保了后续所有 CUDA 操作（包括 NCCL 初始化）都在正确的 GPU 上进行
    torch.cuda.set_device(local_rank)

    # 然后再初始化进程组，它现在会自动绑定到上面设置的设备上
    dist.init_process_group(backend='nccl')

    # 最后获取 world_size
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    # --- 2. 路径和日志设置 ---
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 确保只有主进程创建目录
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'drop', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()  # 所有进程等待目录创建完毕

    # 每个进程使用独立的日志和输出文件
    log_path = osp.join(args.result_dir, 'log',
                        f"{args.run_name}_{curr_time}_rank{local_rank}.log")
    output_jsonl_path = osp.join(
        args.result_dir, 'output', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")
    dr_save_path = osp.join(
        args.result_dir, 'drop', f"{args.run_name}_{curr_time}_rank{local_rank}.jsonl")

    # 配置日志处理器
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
    torch.manual_seed(1234)
    logger.info("设置全局随机种子为 1234")

    if not is_flash_attn_2_available():
        logger.error("Flash Attention 2 不可用，程序将退出。")
        sys.exit(1)

    # 主进程负责下载和加载模型到 CPU，其他进程等待
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
        dist.barrier()  # 主进程加载完毕，通知其他进程可以开始了

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

    if is_main_process:
        logger.info("--- 注意力机制实现验证 ---")
        logger.info(f"ViT 注意力: {type(model.visual.blocks[0].attn)}")
        logger.info(f"语言模型注意力: {type(model.model.layers[0].self_attn)}")
        logger.info("-----------------------------")

    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info(f"[Rank {local_rank}] 预处理器加载完毕。")

    # --- 4. 数据加载与分布式切分 ---
    task_df_full = pd.read_parquet(args.task_parquet)
    # ==================== 代码修改处 ====================
    # 筛选出 duration 为 'long' 的行
    task_df_full = task_df_full[task_df_full['duration'] != 'long'].copy()
    logger.info("数据筛选已完成，仅保留 duration 为 'long' 的样本。")
    # ====================================================
    num_samples = len(task_df_full)
    indices = list(range(num_samples))
    rank_indices = indices[local_rank::world_size]
    task_df = task_df_full.iloc[rank_indices].copy()
    logger.info(
        f"[Rank {local_rank}] 数据切分完毕，将处理 {len(task_df)} / {num_samples} 个样本。")

    # 恢复进度的逻辑
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

    # --- 5. 推理循环 ---
    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    progress_bar = tqdm(task_df.itertuples(), total=len(
        task_df), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for row in progress_bar:
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'reset_streaming_state'):
                model.model.reset_streaming_state()
            elif hasattr(model, 'reset_streaming_state'):
                model.reset_streaming_state()

            if row.duration == "short":  # 小于3分钟 (3 * 60 = 180秒)
                current_fps = 4.0
            else:
                current_fps = 0.5
            video_path = osp.join(args.video_dir, row.videoID + '.mp4')
            messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                                                      "max_frames": args.max_frames, "min_frames": args.min_frames, "fps": current_fps}, {"type": "text", "text": prompt.format(row.question, '\n'.join(row.options.tolist()))}]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128, drop_method=args.drop_method,
                                           drop_threshold=args.drop_threshold, drop_absolute=(not args.drop_relative), dr_save_path=dr_save_path)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(
                inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = output_text[0]

            is_correct = extract_characters_regex(response) == row.answer
            if is_correct:
                cnt_correct['overall'] += 1
                cnt_correct[row.duration] += 1
            cnt_total['overall'] += 1
            cnt_total[row.duration] += 1

            output_dict = {
                'video_id': row.video_id, 'duration': row.duration, 'domain': row.domain, 'sub_category': row.sub_category,
                'url': row.url, 'videoID': row.videoID, 'question_id': row.question_id, 'task_type': row.task_type,
                'question': row.question, 'options': row.options.tolist(), 'answer': row.answer, 'response': response
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
    # 将本进程的统计数据放入 tensor 中，以便于 all_reduce
    stats_list = []
    duration_keys = sorted(task_df_full.duration.unique())
    stats_list.extend([cnt_total['overall'], cnt_correct['overall']])
    for duration in duration_keys:
        stats_list.extend([cnt_total[duration], cnt_correct[duration]])

    stats_tensor = torch.tensor(
        stats_list, dtype=torch.long, device=f'cuda:{local_rank}')

    # 所有进程在此同步，并对统计数据求和
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

    if is_main_process:
        logger.info("--- 所有进程已完成，开始汇总结果 ---")

        # 合并输出文件
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)

        # 合并丢弃率文件
        if args.drop_method not in [None, 'none']:
            base_drop_path = osp.join(
                args.result_dir, 'drop', f"{args.run_name}_{curr_time}")
            merge_jsonl_files(base_drop_path, world_size)

        # 从汇总后的 tensor 中解析结果
        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()

        if total_processed == 0:
            logger.info("所有进程均未处理任何问题。")
        else:
            accuracy = 100 * total_correct / total_processed
            logger.info(
                f"【总结果】: 总数={total_processed}, 正确={total_correct}, 准确率={accuracy:.1f}%")

        # 打印按时长的分类准确率
        for i, duration in enumerate(duration_keys):
            total_dur = stats_tensor[2 + i*2].item()
            correct_dur = stats_tensor[3 + i*2].item()
            if total_dur > 0:
                acc_dur = 100 * correct_dur / total_dur
                logger.info(
                    f"  - 时长 {duration}: {correct_dur}/{total_dur} = {acc_dur:.1f}%")

        logger.info(
            f"总推理耗时: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (此为单个进程耗时，非总和)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
