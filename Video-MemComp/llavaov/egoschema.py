import traceback
from transformers import LlavaOnevisionProcessor, AutoConfig
from llava_onevision import LlavaOnevisionForConditionalGeneration
from decord import VideoReader, cpu
from PIL import Image
import math
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

prompt_template = """Based on the video, please answer the following multiple-choice question. Respond with only the letter (A, B, C, D, or E) of the best option.
Question: {}
Options: {}
The best answer is:"""

def extract_characters_regex():
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
                            f"Failed to parse line in file {file_path}: {line.strip()}")
                        continue

    merged_file_path = f"{base_path}_merged.jsonl"
    with open(merged_file_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Successfully merged {len(merged_data)} records to {merged_file_path}")
    return merged_file_path


def create_submission_file(merged_jsonl_path, save_dir):
    if not osp.exists(merged_jsonl_path):
        logger.error(f"Merged result file not found: {merged_jsonl_path}")
        return

    results = []
    with open(merged_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    submission_data = []
    for r in results:
        pred_choice = r.get('pred_choice')
        if not pred_choice or pred_choice not in list('ABCDE'):
            pred_choice = 'A'  

        answer_index = ord(pred_choice) - ord('A')
        submission_data.append(
            {'q_uid': r['q_uid'], 'answer': answer_index})  

    submission_df = pd.DataFrame(submission_data)
    submission_path = osp.join(save_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"EgoSchema submission file created: {submission_path}")


def evaluate_with_subset(merged_jsonl_path, subset_answers_path):
    if not all([osp.exists(merged_jsonl_path), osp.exists(subset_answers_path)]):
        logger.error(
            f"Files required for evaluation do not exist. Check {merged_jsonl_path} and {subset_answers_path}")
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
        logger.info("--- EgoSchema Subset Evaluation Results ---")
        logger.info(f"Correct predictions: {correct_count}")
        logger.info(f"Questions evaluated: {total_evaluated} (from subset total {len(ground_truths)})")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info("------------------------------------")
        print(f"EgoSchema subset evaluation accuracy: {accuracy:.2f}%")
    else:
        logger.error("Evaluation failed: No model predictions match the subset answers.")
        print("EgoSchema subset evaluation failed: No model predictions match the subset answers.")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation on EgoSchema dataset using LLaVA-OneVision")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to EgoSchema's full.json file")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str,
                        default="eval/result_egoschema_llava")
    parser.add_argument("--sample_fps", type=float, default=0.5, help="Video sampling frame rate")
    parser.add_argument("--eval_subset_only", action="store_true",
                        help="If set, only run inference on the subset defined in subset_answers.json")
    parser.add_argument("--subset_answers_path", type=str,
                        default="/home/nyh/EgoSchema/subset_answers.json", help="Path to subset_answers.json for filtering and/or local evaluation")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to baseline output file for resuming interrupted run (without '_rankN.jsonl' suffix)")
    parser.add_argument("--yarn_factor", type=float,
                        default=1.0, help="Scaling factor for YaRN")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    is_main_process = local_rank == 0
    device = f"cuda:{local_rank}"
    if is_main_process:
        os.makedirs(args.result_dir, exist_ok=True)
        for subdir in ['output', 'log']:
            os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)
    dist.barrier()  
    logger.info(f"--- [Rank {local_rank}/{world_size}] Process started ---")
    logger.info(f"Run configuration: {vars(args)}")

    torch.manual_seed(1234)
    if not is_main_process:
        dist.barrier()

    logger.info(f"Loading processor from {args.ckpt_path}...")
    processor = LlavaOnevisionProcessor.from_pretrained(args.ckpt_path)

    logger.info(f"Loading model config from {args.ckpt_path}...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)

    if 1:
        if not hasattr(config, 'text_config'):
            logger.error("'text_config' not found in model config, cannot apply YaRN.")
        else:
            text_config = config.text_config
            original_max_len = text_config.max_position_embeddings
            target_max_len = int(original_max_len * args.yarn_factor)

            logger.info("--- Extending context via YaRN ---")
            logger.info(f"  - Original max length: {original_max_len}")
            logger.info(f"  - YaRN scaling factor (λ): {args.yarn_factor}")
            logger.info(f"  - Computed target length: {target_max_len}")

            text_config.rope_scaling = {
                "type": "yarn",
                "factor": args.yarn_factor,
                "original_max_position_embeddings": original_max_len
            }

            text_config.max_position_embeddings = target_max_len
            logger.info("--- Context extension successfully configured ---")

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
    model.eval()
    logger.info(f"[Rank {local_rank}] Model and processor loaded successfully.")
    if is_main_process:
        logger.info(f"Loading annotation file from {args.anno_path}...")
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
                        logger.warning(f"Video file not found, skipping: {video_path}")
                        continue
                    flattened_data.append({
                        'video_id': video_id,
                        'video_path': video_path,
                        'q_uid': question_idx,
                        'question': question,
                        'choices': choices
                    })

        logger.info(f"Successfully parsed {len(flattened_data)} valid QA entries.")
        if args.eval_subset_only:
            logger.info("--- Subset-only mode activated ---")
            with open(args.subset_answers_path, 'r') as f:
                subset_video_ids = set(json.load(f).keys())
            original_count = len(flattened_data)
            flattened_data = [
                d for d in flattened_data if d['video_id'] in subset_video_ids]
            logger.info(
                f"Dataset filtered from {original_count} items to {len(flattened_data)} items.")
        dist.broadcast_object_list([flattened_data], src=0)
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=0)
        flattened_data = received_objects[0]

    if not flattened_data:
        logger.error(f"[Rank {local_rank}] No processable data received. Exiting.")
        dist.destroy_process_group()
        return

    indices = list(range(len(flattened_data)))
    rank_data = [flattened_data[i] for i in indices[local_rank::world_size]]

    if args.resume_path:
        base_output_path = re.sub(r'_rank\d+\.jsonl$', '', args.resume_path)
        logger.info(
            f"--- [Rank {local_rank}] Resuming run, using base path: {base_output_path} ---")
    else:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_path = osp.join(
            args.result_dir, 'output', f"{args.run_name}_{curr_time}")
        logger.info(
            f"--- [Rank {local_rank}] Starting new run, using base path: {base_output_path} ---")

    output_jsonl_path = f"{base_output_path}_rank{local_rank}.jsonl"
    completed_q_uids = set()
    if args.resume_path and osp.exists(output_jsonl_path):
        logger.info(
            f"[Rank {local_rank}] Reading existing output file to determine progress: {output_jsonl_path}")
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_q_uids.add(json.loads(line)['q_uid'])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        f"[Rank {local_rank}] Failed to parse or find q_uid in file: {line.strip()}")
        logger.info(f"[Rank {local_rank}] Found {len(completed_q_uids)} completed tasks.")

    original_task_count = len(rank_data)
    rank_data_to_process = [
        item for item in rank_data if item['q_uid'] not in completed_q_uids]
    logger.info(
        f"[Rank {local_rank}] After filtering: {len(rank_data_to_process)} / {original_task_count} tasks remaining to process.")

    progress_bar = tqdm(rank_data_to_process, total=len(
        rank_data_to_process), disable=not is_main_process, desc=f"Rank {local_rank} Inference")
    for item in progress_bar:
        try:
            video_path = item['video_path']
            if not osp.exists(video_path):
                logger.warning(f"Video file not found, skipping: {video_path}")
                continue

            # --- Video frame sampling using Decord with specified FPS ---
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
                logger.warning(f"No frames sampled from video {video_path}, skipping.")
                continue
            past_key_values = None
            logical_seq_len = 0

            # Step A: Prefill video frames (KV Cache warming)
            # Process first frame to initialize KV Cache
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

            # prefill the question and generate answer
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
                f"[Rank {local_rank}] process q_uid {item['q_uid']} (video: {item['video_id']}) has error: {e}")
            traceback.print_exc()



    dist.barrier()
    if is_main_process:
        logger.info("--- all process are finished... ---")
        merged_file_path = merge_jsonl_files(base_output_path, world_size)
        evaluate_with_subset(merged_file_path, args.subset_answers_path)

        if not args.eval_subset_only:
            logger.info("creating file to submit...")
            create_submission_file(merged_file_path, args.result_dir)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

