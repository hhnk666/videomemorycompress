import traceback
# --- LLaVA OneVision 和视频处理相关的库 ---
from transformers import LlavaOnevisionProcessor, AutoConfig
# 确保 llava_onevision 模型定义在 Python 路径中
# 如果报错，请将 LLaVA-OneVision 的代码目录添加到 PYTHONPATH
from llava_onevision import LlavaOnevisionForConditionalGeneration
from decord import VideoReader, cpu
from PIL import Image
import math
# --- 基础库 ---
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

# --- 日志设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

# --- EgoSchema 提示模板 ---
prompt_template = """Based on the video, please answer the following multiple-choice question. Respond with only the letter (A, B, C, D, or E) of the best option.
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
            {'q_uid': r['q_uid'], 'answer': answer_index})  # 使用 q_uid

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
        print(f"EgoSchema 子集评估准确率: {accuracy:.2f}%")
    else:
        logger.error("评估失败：模型预测结果中没有与子集答案匹配的项。")
        print("EgoSchema 子集评估失败：模型预测结果中没有与子集答案匹配的项。")


# --- 主脚本 ---
def main():
    parser = argparse.ArgumentParser(
        description="在 EgoSchema 数据集上使用 LLaVA-OneVision 进行分布式评测")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True,
                        help="指向 EgoSchema 的 full.json 文件")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str,
                        default="eval/result_egoschema_llava")
    parser.add_argument("--sample_fps", type=float, default=0.5, help="视频采样帧率")
    parser.add_argument("--eval_subset_only", action="store_true",
                        help="如果设置，仅在 subset_answers.json 定义的子集上运行推理")
    parser.add_argument("--subset_answers_path", type=str,
                        default="/home/nyh/EgoSchema/subset_answers.json", help="用于过滤和/或本地评估的 subset_answers.json 路径")
    parser.add_argument("--resume_path", type=str, default="",
                        help="用于恢复中断运行的基准输出文件路径（不含'_rankN.jsonl'后缀）")
    parser.add_argument("--yarn_factor", type=float,
                        default=1.0, help="YaRN 的缩放因子")
    args = parser.parse_args()

    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    device = f"cuda:{local_rank}"
    # --- 路径和日志设置 ---
    # 只有主进程负责创建顶层目录结构，然后同步所有进程
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()  # 确保所有进程都等待目录创建完毕

# 路径和日志设置 (逻辑不变)
    # (为简洁此处省略)

    logger.info(f"--- [Rank {local_rank}/{world_size}] 进程已启动 ---")
    logger.info(f"运行配置: {vars(args)}")

    # --- 模型和处理器加载 (集成 YaRN 逻辑) ---
    torch.manual_seed(1234)
    if not is_main_process:
        dist.barrier()

    logger.info(f"正在从 {args.ckpt_path} 加载处理器...")
    processor = LlavaOnevisionProcessor.from_pretrained(args.ckpt_path)

    logger.info(f"正在从 {args.ckpt_path} 加载模型配置...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)

    # --- NEW: YaRN 上下文扩展逻辑 ---
    if 1:
        if not hasattr(config, 'text_config'):
            logger.error("在模型配置中找不到 'text_config'，无法应用 YaRN。")
        else:
            text_config = config.text_config
            original_max_len = text_config.max_position_embeddings
            target_max_len = int(original_max_len * args.yarn_factor)

            logger.info("--- 正在通过 YaRN 扩展上下文 ---")
            logger.info(f"  - 原始最大长度: {original_max_len}")
            logger.info(f"  - YaRN 缩放因子 (λ): {args.yarn_factor}")
            logger.info(f"  - 计算出的目标长度: {target_max_len}")

            # 设置 rope_scaling 参数
            text_config.rope_scaling = {
                "type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": original_max_len
            }
            # 更新模型配置以使用新的目标长度
            text_config.max_position_embeddings = target_max_len
            logger.info("--- 上下文扩展已成功配置 ---")

    logger.info(f"正在从 {args.ckpt_path} 加载模型...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,  # 传递修改后（或未修改）的配置
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    if is_main_process:
        dist.barrier()

    model.to(device)
    model.eval()
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

    # --- 断点续跑逻辑 ---
    if args.resume_path:
        base_output_path = re.sub(r'_rank\d+\.jsonl$', '', args.resume_path)
        logger.info(
            f"--- [Rank {local_rank}] 恢复运行，使用基准路径: {base_output_path} ---")
    else:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        logger.info(
            f"--- [Rank {local_rank}] 开始新运行，使用基准路径: {base_output_path} ---")

    output_jsonl_path = f"{base_output_path}_rank{local_rank}.jsonl"
    # output_jsonl_path = "/data1/nyh/llava_onevisioncopy/eval_results/egoschema_llava_runs/egoschema_llava_onevision_0.80.95.5/output/egoschema_llava_onevision_0.80.95.5_20251104_133702_rank0.jsonl"
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

    original_task_count = len(rank_data)
    rank_data_to_process = [
        item for item in rank_data if item['q_uid'] not in completed_q_uids]
    logger.info(
        f"[Rank {local_rank}] 任务过滤后: 剩余 {len(rank_data_to_process)} / 总计 {original_task_count} 个任务需要处理。")

    # --- 推理循环 (采用 LLaVA 流式处理) ---
    progress_bar = tqdm(rank_data_to_process, total=len(
        rank_data_to_process), disable=not is_main_process, desc=f"Rank {local_rank} Inference")
    for item in progress_bar:
        try:
            video_path = item['video_path']
            if not osp.exists(video_path):
                logger.warning(f"视频文件未找到，跳过: {video_path}")
                continue

            # --- 使用 Decord 和指定 FPS 进行视频帧采样 ---
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            native_fps = vr.get_avg_fps()

            frame_interval = math.ceil(native_fps / args.sample_fps)
            if frame_interval <= 0:
                frame_interval = 1

            frame_indices = list(range(0, total_frames, int(frame_interval)))
            video_frames = [Image.fromarray(
                frame) for frame in vr.get_batch(frame_indices).asnumpy()]

            if not video_frames:
                logger.warning(f"从视频 {video_path} 未采样到任何帧，跳过。")
                continue

            # --- LLaVA-OneVision 流式推理 ---
            past_key_values = None
            logical_seq_len = 0

            # 步骤 A: 预填充视频帧 (KV Cache warming)
            # 处理第一帧以初始化 KV Cache
            first_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n<video>"
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

            # 处理剩余帧，不断扩展 KV Cache
            if len(video_frames) > 1:
                for frame in tqdm(video_frames[1:], desc="Processing Frames", leave=False):
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

            # 步骤 B: 预填充问题并生成答案
            formatted_options = "\n".join(
                [f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(item['choices'])])
            question_text = prompt_template.format(
                item['question'], formatted_options)
            final_prompt = f"{question_text}<|im_end|>\n<|im_start|>assistant\n"

            inputs_question = processor(
                text=final_prompt, return_tensors="pt").to(device)
            question_input_ids = inputs_question.input_ids
            question_attention_mask = inputs_question.attention_mask
            question_len = question_input_ids.shape[1]

            with torch.no_grad():
                physical_past_len = past_key_values[0][0].shape[-2]
                attention_mask = torch.cat([
                    torch.ones((1, physical_past_len),
                               device=device, dtype=torch.long),
                    question_attention_mask
                ], dim=1)
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

            # 步骤 C: 解码和后处理
            input_token_len = inputs_question.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]
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
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(
                f"[Rank {local_rank}] 处理 q_uid {item['q_uid']} (video: {item['video_id']}) 时出错: {e}")
            traceback.print_exc()

    logger.info(f"[Rank {local_rank}] 推理完成。")

    # --- 结果聚合与提交文件生成 ---
    dist.barrier()
    if is_main_process:
        logger.info("--- 所有进程已完成。正在聚合结果... ---")
        merged_file_path = merge_jsonl_files(base_output_path, world_size)
        evaluate_with_subset(merged_file_path, args.subset_answers_path)

        if not args.eval_subset_only:
            logger.info("正在创建Kaggle提交文件...")
            create_submission_file(merged_file_path, args.result_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    # evaluate_with_subset("/data1/nyh/InternVL3_5/eval_results/egoschema_runs_internvlstreammem/egoschema_internvl3.5_100/output/egoschema_internvl3.5_100_20251120_171831.jsonl",
    #  "/home/nyh/EgoSchema/subset_answers.json")
