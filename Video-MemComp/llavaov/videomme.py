import debugpy
import traceback
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

RUN_NAME = "llava_onevision_videomme_streaming_test_yarn8"
CKPT_PATH = "/data1/nyh/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf"
TASK_PARQUET = "/pfs/Datasets/Video-MME/origin_data/videomme/test-00000-of-00001.parquet"
VIDEO_DIR = "/pfs/Datasets/Video-MME/origin_data/videos/data/"
RESULT_DIR = "eval/result_videomme_llava_onevision"
SAMPLE_FPS = 0.2
YARN_FACTOR = 2.0
prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: {}
Options: {}
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

def extract_characters_regex():
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
                            f"Failed to parse line in file {file_path}: {line.strip()}")
                        continue
    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Successfully merged {len(merged_data)} records to {merged_file_path}")
    return merged_file_path


def main():
    parser = argparse.ArgumentParser(
        description="Streaming distributed evaluation on Video-MME dataset using LLaVA OneVision.")
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--task_parquet", type=str, default=TASK_PARQUET)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--sample_fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--extend_context", action='store_true',
                        default=True, help="Enable context extension (YaRN)")
    parser.add_argument("--target_max_len", type=int,
                        default=128000, help="Target context length for YaRN extension")
    parser.add_argument("--yarn_factor", type=float,
                        default=6.0, help="Scaling factor for YaRN")
    args = parser.parse_args()

    # Distributed Environment Initialization
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    device = f"cuda:{local_rank}"

    # Path and Logging Configuration
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

    logger.info(f"--- [Rank {local_rank}/{world_size}] Process Started ---")
    logger.info(f"Run Configuration: {vars(args)}")

    torch.manual_seed(1919810)
    logger.info("Set global random seed to 1234")

    if not is_main_process:
        dist.barrier()

    logger.info(f"Loading processor from {args.ckpt_path}...")
    processor = LlavaOnevisionProcessor.from_pretrained(args.ckpt_path)

    logger.info(f"Loading model config from {args.ckpt_path}...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)

    if 1:
        text_config = getattr(config, 'text_config', config)
        original_max_len = text_config.max_position_embeddings
        TARGET_MAX_LEN = int(original_max_len * YARN_FACTOR)
        # Save [original length] to a new field
        text_config.original_max_position_embeddings = original_max_len

        # Set rope_scaling parameters
        text_config.rope_scaling = {
            "type": "yarn",
            "factor": YARN_FACTOR,
            "original_max_position_embeddings": original_max_len
        }

        # Finally, update max_position_embeddings to [target length]
        text_config.max_position_embeddings = TARGET_MAX_LEN

    logger.info(f"Loading model from {args.ckpt_path}...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  
        trust_remote_code=True,
    )

    if is_main_process:
        dist.barrier()

    model.to(device)
    logger.info(f"[Rank {local_rank}] Model successfully loaded to device {device}.")

    task_df_full = pd.read_parquet(args.task_parquet)
    num_samples = len(task_df_full)
    indices = list(range(num_samples))
    rank_indices = indices[local_rank::world_size]
    task_df = task_df_full.iloc[rank_indices].copy()
    logger.info(
        f"[Rank {local_rank}] Data sharding completed. This process will handle {len(task_df)} / {num_samples} samples.")

    processed_ids = set()
    if osp.exists(output_jsonl_path):
        logger.info(f"[Rank {local_rank}] Resuming progress from existing output file: {output_jsonl_path}")
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
            f"[Rank {local_rank}] Skipped {len(processed_ids)} completed tasks, {len(task_df)}/{original_count} remaining.")

    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    progress_bar = tqdm(task_df.itertuples(), total=len(
        task_df), disable=not is_main_process, desc=f"Rank {local_rank} Inference")

    for row in progress_bar:
        try:
            video_path = osp.join(args.video_dir, row.videoID + '.mp4')
            if not osp.exists(video_path):
                logger.warning(f"Video file not found, skipping: {video_path}")
                continue

            # Prepare Video Data
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            duration_in_seconds = total_frames / video_fps if video_fps > 0 else 0

            if duration_in_seconds < 1800.0:
                current_sample_fps = 0.5
            else:
                current_sample_fps = 0.2
            if duration_in_seconds <= 250:
                model.model.inter_frame_threshold = 0.95  
                model.model.intra_frame_threshold = 0.97  
            elif duration_in_seconds <= 1800:
                model.model.inter_frame_threshold = 0.8  
                model.model.intra_frame_threshold = 0.85  
            else:
                model.model.inter_frame_threshold = 0.72  
                model.model.intra_frame_threshold = 0.82  


            frame_interval = math.ceil(
                video_fps / current_sample_fps) if current_sample_fps > 0 else total_frames
            if frame_interval <= 0:
                frame_interval = 1

            frame_indices = [i for i in range(0, total_frames, frame_interval)]



            video_frames = [Image.fromarray(
                frame) for frame in vr.get_batch(frame_indices).asnumpy()]

            if not video_frames:
                logger.warning(f"No frames sampled from video {video_path}, skipping.")
                continue

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
                f"[Rank {local_rank}] Error occurred while processing {row.question_id}: {e}")
            traceback.print_exc()

    end_time = time.time()
    cost_time = int(end_time - start_time)
    logger.info(f"[Rank {local_rank}] Inference completed, elapsed time: {cost_time} seconds.")

    stats_list = [cnt_total['overall'], cnt_correct['overall']]
    stats_tensor = torch.tensor(stats_list, dtype=torch.long, device=device)

    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()

    if is_main_process:
        logger.info("--- All processes completed, starting result aggregation ---")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        merge_jsonl_files(base_output_path, world_size)
        total_processed = stats_tensor[0].item()
        total_correct = stats_tensor[1].item()
        if total_processed == 0:
            logger.info("No questions were processed by any process.")
        else:
            accuracy = 100 * total_correct / total_processed
            logger.info(
                f"[Overall Results]: Total={total_processed}, Correct={total_correct}, Accuracy={accuracy:.2f}%")
        logger.info(
            f"Total inference time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s (this is the time for a single process, not cumulative)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
