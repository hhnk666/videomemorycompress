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
import argparse
import sys

# 确保自定义模型定义在Python路径中
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from qwen_vl_utils import process_vision_info
    from qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLSdpaAttention,
    )
except ImportError as e:
    print(f"Import Error: 'qwen_vl_utils' 和 'qwen2_5_vl' 模块无法访问。")
    print(f"详情: {e}")
    sys.exit(1)

# --- 日志设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

# --- EgoSchema 提示模板 ---
prompt = """Based on the video, please answer the following multiple-choice question. Respond with only the letter (A, B, C, D, or E) of the best option.
Question: {}
Options: {}
The best answer is:"""

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

    matches = re.search(r"^\s*\(?([A-E])\)?", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    matches = re.search(r"\b([A-E])\b", s)
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


def create_submission_file(merged_jsonl_path, save_dir):
    if not osp.exists(merged_jsonl_path):
        logger.error(f"合并后的结果文件未找到: {merged_jsonl_path}")
        return

    results = []
    with open(merged_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    submission_data = []
    for r in results:
        pred_choice = r.get('pred_choice')
        if not pred_choice or pred_choice not in list('ABCDE'):
            pred_choice = 'A'  # 默认A

        answer_index = ord(pred_choice) - ord('A')
        submission_data.append(
            {'q_uid': r['video_id'], 'answer': answer_index})

    submission_df = pd.DataFrame(submission_data)
    submission_path = osp.join(save_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"EgoSchema 提交文件已创建: {submission_path}")


def evaluate_with_subset(merged_jsonl_path, subset_answers_path):
    if not all([osp.exists(merged_jsonl_path), osp.exists(subset_answers_path)]):
        logger.error(
            f"评估所需文件不存在。检查 {merged_jsonl_path} 和 {subset_answers_path}")
        return

    with open(subset_answers_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)

    predictions = {}
    with open(merged_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            video_id = data.get('video_id')
            pred_choice_letter = data.get('pred_choice')
            if video_id and pred_choice_letter and pred_choice_letter in list('ABCDE'):
                predictions[video_id] = ord(pred_choice_letter) - ord('A')

    correct_count = 0
    total_evaluated = 0
    for video_id, true_answer in ground_truths.items():
        if video_id in predictions:
            if predictions[video_id] == true_answer:
                correct_count += 1
            total_evaluated += 1

    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        logger.info("--- EgoSchema 子集评估结果 ---")
        logger.info(f"正确预测数: {correct_count}")
        logger.info(f"已评估问题数: {total_evaluated} (来自子集总数 {len(ground_truths)})")
        logger.info(f"准确率: {accuracy:.2f}%")
        logger.info("------------------------------------")
    else:
        logger.error("评估失败：模型预测结果中没有与子集答案匹配的项。")

# --- 主脚本 ---


def main():
    parser = argparse.ArgumentParser(description="EgoSchema 数据集上的分布式评估脚本")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--anno_path", type=str,
                        required=True, help="指向EgoSchema的full.json文件")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str,
                        default="eval/result_egoschema")
    parser.add_argument("--drop_method", type=str,
                        default='none', choices=['feature', 'pixel', 'none'])
    parser.add_argument("--drop_threshold", type=float, default=0.5)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--min_pixels", type=int, default=448*448)
    parser.add_argument("--max_pixels", type=int, default=448*448)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--nframes", type=int, default=128)
    parser.add_argument("--fps", type=float, default=0.25)
    parser.add_argument("--eval_subset_only", action="store_true",
                        help="如果设置，仅在subset_answers.json定义的子集上运行推理")
    parser.add_argument("--subset_answers_path", type=str,
                        default="/home/nyh/EgoSchema/subset_answers.json", help="用于过滤和/或本地评估的subset_answers.json路径")
    ### --- 新增/修改部分开始 --- ###
    parser.add_argument("--resume_path", type=str, default="",
                        help="用于恢复中断运行的基准输出文件路径（例如：/path/to/output/my_run_20251026_133900），不含'_rankN.jsonl'后缀")
    ### --- 新增/修改部分结束 --- ###
    args = parser.parse_args()

    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0

    # 路径和日志设置
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'drop_info', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()

    ### --- 新增/修改部分开始 --- ###
    # 根据是否为恢复模式，决定日志文件名和输出文件基准路径
    if args.resume_path:
        # 从提供的路径中提取基础运行名称
        base_name_for_log = osp.basename(args.resume_path)
        log_path = osp.join(args.result_dir, 'log',
                            f"{base_name_for_log}_rank{local_rank}.log")
    else:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = osp.join(args.result_dir, 'log',
                            f"{args.run_name}_{curr_time}_rank{local_rank}.log")

    # 日志文件以追加模式打开，以便续写日志
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    if is_main_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    ### --- 新增/修改部分结束 --- ###

    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程已启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # 模型和处理器加载
    torch.manual_seed(1234)
    if not is_flash_attn_2_available():
        logger.warning("Flash Attention 2 不可用。如果使用drop_method，可能会出现问题。")

    if not is_main_process:
        dist.barrier()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cpu",
    )
    model.model.size = 3
    if is_main_process:
        dist.barrier()

    llm_config = model.model.config
    for i, layer in enumerate(model.model.layers):
        original_flash_attn_module = layer.self_attn
        new_sdpa_attn_module = Qwen2_5_VLSdpaAttention(
            config=llm_config, layer_idx=i)
        new_sdpa_attn_module.load_state_dict(
            original_flash_attn_module.state_dict())
        layer.self_attn = new_sdpa_attn_module.to(dtype=torch.bfloat16)

    model.to(f'cuda:{local_rank}')
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    logger.info(f"[Rank {local_rank}] 模型和处理器加载成功。")

    # --- 数据加载与分发 ---
    if is_main_process:
        logger.info(f"正在从 {args.anno_path} 加载标注文件...")
        with open(args.anno_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        flattened_data = []
        for video_info in full_data:
            video_id = video_info.get('video_id')
            if not video_id:
                continue
            if 'conversations' not in video_info or not isinstance(video_info['conversations'], list):
                continue
            for conversation in video_info['conversations']:
                question = conversation.get('question')
                choices = conversation.get('choices')
                question_idx = conversation.get('question_idx')
                if question and isinstance(choices, list) and len(choices) == 5:
                    video_path = osp.join(args.video_dir, f"{video_id}.mp4")
                    if not osp.exists(video_path):
                        logger.warning(f"视频文件未找到，跳过: {video_path}")
                        continue
                    flattened_data.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'q_uid': question_idx,
                        'question': question,
                        'choices': choices
                    })

        logger.info(f"成功解析 {len(flattened_data)} 条有效的问答数据。")
        if args.eval_subset_only:
            logger.info("--- 仅子集模式已激活 ---")
            with open(args.subset_answers_path, 'r') as f:
                subset_video_ids = set(json.load(f).keys())
            original_count = len(flattened_data)
            flattened_data = [
                d for d in flattened_data if d['video_id'] in subset_video_ids]
            logger.info(
                f"数据集已从 {original_count} 条过滤至 {len(flattened_data)} 条。")
        dist.broadcast_object_list([flattened_data], src=0)
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=0)
        flattened_data = received_objects[0]

    if not flattened_data:
        logger.error(f"[Rank {local_rank}] 未收到任何可处理的数据。退出。")
        dist.destroy_process_group()
        return

    indices = list(range(len(flattened_data)))
    rank_data = [flattened_data[i] for i in indices[local_rank::world_size]]

    ### --- 新增/修改部分开始 --- ###
    # --- 路径设置与断点续跑逻辑 ---
### --- 新增/修改部分开始 --- ###
    # --- 路径设置与断点续跑逻辑 ---
    if args.resume_path:
        # 自动清理 resume_path，确保它是一个基准路径，无论用户输入是否带后缀
        base_output_path = re.sub(r'_rank\d+\.jsonl$', '', args.resume_path)
        if base_output_path != args.resume_path:
            logger.info(f"提供的 resume_path 已被自动修正为基准路径。")
        logger.info(
            f"--- [Rank {local_rank}] 恢复运行，使用基准路径: {base_output_path} ---")
    else:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        logger.info(
            f"--- [Rank {local_rank}] 开始新运行，使用基准路径: {base_output_path} ---")

    output_jsonl_path = f"{base_output_path}_rank{local_rank}.jsonl"
    dr_save_path = osp.join(args.result_dir, 'drop_info',
                            f"{osp.basename(base_output_path)}_rank{local_rank}.jsonl")

    # 检查已完成的任务并过滤
    completed_q_uids = set()
    if args.resume_path and osp.exists(output_jsonl_path):
        logger.info(
            f"[Rank {local_rank}] 正在读取已有的输出文件以确定进度: {output_jsonl_path}")
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_q_uids.add(json.loads(line)['q_uid'])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        f"[Rank {local_rank}] 无法解析或在文件中找不到 q_uid: {line.strip()}")
        logger.info(f"[Rank {local_rank}] 发现 {len(completed_q_uids)} 个已完成的任务。")

    # 根据已完成的 q_uid 过滤掉 rank_data 中的任务
    original_task_count = len(rank_data)
    rank_data_to_process = [
        item for item in rank_data if item['q_uid'] not in completed_q_uids]
    logger.info(
        f"[Rank {local_rank}] 任务过滤后: 剩余 {len(rank_data_to_process)} / 总计 {original_task_count} 个任务需要处理。")
    ### --- 新增/修改部分结束 --- ###

    # --- 推理循环 ---
    progress_bar = tqdm(rank_data_to_process, total=len(
        rank_data_to_process), disable=not is_main_process, desc=f"Rank {local_rank} Inference")
    for item in progress_bar:
        try:
            formatted_options = "\n".join(
                [f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(item['choices'])])
            messages = [{"role": "user", "content": [
                {"type": "video",
                    "video": item['video_path'], "nframes": args.nframes},
                {"type": "text", "text": prompt.format(
                    item['question'], formatted_options)}
            ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                           drop_method=args.drop_method, drop_threshold=args.drop_threshold,
                                           drop_absolute=(
                                               not args.drop_relative),
                                           dr_save_path=dr_save_path if args.drop_method != 'none' else None)

            generated_ids_trimmed = generated_ids[:,
                                                  inputs.input_ids.shape[1]:]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True)[0]
            pred_choice = extract_characters_regex(response)

            output_dict = {
                'q_uid': item['q_uid'],
                'video_id': item['video_id'],
                'question': item['question'],
                'choices': item['choices'],
                'response': response,
                'pred_choice': pred_choice
            }
            # 使用 'a' 模式，确保向现有文件追加内容
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理 video_id {item['video_id']} 时出错: {e}")
            traceback.print_exc()

    logger.info(f"[Rank {local_rank}] 推理完成。")

    # --- 结果聚合与提交文件生成 ---
    dist.barrier()
    if is_main_process:
        logger.info("--- 所有进程已完成。正在聚合结果... ---")
        ### --- 新增/修改部分开始 --- ###
        # 此处的 base_output_path 已根据是否恢复运行被正确设置
        merged_file_path = merge_jsonl_files(base_output_path, world_size)
        ### --- 新增/修改部分结束 --- ###

        evaluate_with_subset(merged_file_path, args.subset_answers_path)

        if not args.eval_subset_only:
            logger.info("正在创建Kaggle提交文件...")
            create_submission_file(merged_file_path, args.result_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
