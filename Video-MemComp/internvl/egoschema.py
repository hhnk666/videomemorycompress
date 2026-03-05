import traceback
import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# --- Ensure custom model definitions are in Python path ---
# Replace 'InternVL3_5' with the actual module path containing your model definitions
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from InternVL3_5 import InternVLChatModel
except ImportError as e:
    from transformers import AutoModelForCausalLM as InternVLChatModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
fmt = logging.Formatter(fmt_str)

prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D, or E) of the correct option.
{question}
Options: {options}
The best answer is:"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float('inf'), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                           for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * \
        target_aspect_ratio[0], image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    start, end = (bound[0], bound[1]) if bound else (0, max_frame / fps)
    start_idx, end_idx = max(first_idx, round(
        start * fps)), min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return np.clip(frame_indices, 0, max_frame).astype(int)


def load_video(video_path, num_segments=64, input_size=448, max_num=1):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(
            None, fps, max_frame, num_segments=num_segments)

        pixel_values_list, num_patches_list = [], []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img_tiles = dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = torch.stack([transform(tile) for tile in img_tiles])
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        return torch.cat(pixel_values_list), num_patches_list
    except Exception as e:
        return None, None


def extract_characters_regex():
    s = s.strip()
    answer_prefixes = ["The best answer is", "The correct answer is", "The answer is", "The answer",
                       "The best option is", "The correct option is", "Best answer:", "Best option:"]
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


def create_submission_file(result_jsonl_path, save_dir):
    if not osp.exists(result_jsonl_path):
        return logger.error(f"Result file not found: {result_jsonl_path}")
    results = [json.loads(line) for line in open(
        result_jsonl_path, 'r', encoding='utf-8')]
    # EgoSchema's q_uid is the question_idx
    submission_data = [{'q_uid': r['q_uid'], 'answer': ord(
        r.get('pred_choice', 'A')) - ord('A')} for r in results]
    submission_df = pd.DataFrame(submission_data)
    submission_path = osp.join(save_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)


def evaluate_with_subset(result_jsonl_path, subset_answers_path):
    if not all([osp.exists(result_jsonl_path), osp.exists(subset_answers_path)]):
        return logger.error("Files required for evaluation do not exist.")
    with open(subset_answers_path, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    predictions = {data['video_id']: ord(data['pred_choice']) - ord('A') for line in open(
        result_jsonl_path, 'r', encoding='utf-8') if (data := json.loads(line)).get('pred_choice') in 'ABCDE'}
    correct_count, total_evaluated = 0, 0
    for video_id, true_answer in ground_truths.items():
        if video_id in predictions:
            if predictions[video_id] == true_answer:
                correct_count += 1
            total_evaluated += 1
    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        logger.info(
            f"--- EgoSchema Subset Evaluation Results ---\nCorrect predictions: {correct_count}\nQuestions evaluated: {total_evaluated}\nAccuracy: {accuracy:.2f}%\n------------------------------------")
    else:
        logger.error("Evaluation failed: No model predictions match the subset answers.")


def main():
    parser = argparse.ArgumentParser(
        description="Single-GPU evaluation script for EgoSchema dataset (InternVL)")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str,
                        required=True, help="Path to InternVL-3.5 model weights")
    parser.add_argument("--anno_path", type=str, required=True,
                        help="Path to EgoSchema's full.json file")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str,
                        default="eval/result_egoschema_internvl")
    parser.add_argument("--nframes", type=int, default=64, help="Fixed number of sampled frames")
    parser.add_argument("--eval_subset_only", action="store_true",
                        help="If set, only run on the subset defined in subset_answers.json")
    parser.add_argument("--subset_answers_path", type=str,
                        default="/home/nyh/EgoSchema/subset_answers.json")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to output file for resuming interrupted run (e.g., /path/to/output.jsonl)")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    for subdir in ['output', 'log']:
        os.makedirs(osp.join(args.result_dir, subdir), exist_ok=True)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.run_name}_{curr_time}"
    log_path = osp.join(args.result_dir, 'log', f"{base_name}.log")

    # If resuming a run, use the basename of the resume file
    if args.resume_path:
        base_name = osp.splitext(osp.basename(args.resume_path))[0]
        log_path = osp.join(args.result_dir, 'log', f"{base_name}.log")

    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    torch.manual_seed(1234)
    model = InternVLChatModel.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,  
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.ckpt_path, trust_remote_code=True, use_fast=False)

    with open(args.anno_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    flattened_data = []
    for video_info in full_data:
        if not (video_id := video_info.get('video_id')) or 'conversations' not in video_info:
            continue
        for conv in video_info['conversations']:
            video_path = osp.join(args.video_dir, f"{video_id}.mp4")
            if not osp.exists(video_path):
                continue
            flattened_data.append({'video_id': video_id, 'video_path': video_path,
                                  'q_uid': conv['question_idx'], 'question': conv['question'], 'choices': conv['choices']})

    if args.eval_subset_only:
        with open(args.subset_answers_path, 'r') as f:
            subset_ids = set(json.load(f).keys())
        original_count = len(flattened_data)
        flattened_data = [
            d for d in flattened_data if d['video_id'] in subset_ids]
        logger.info(f"Dataset filtered from {original_count} items to {len(flattened_data)} items.")

    # --- Checkpoint/resume logic ---
    output_jsonl_path = args.resume_path if args.resume_path else osp.join(
        args.result_dir, 'output', f"{base_name}.jsonl")

    completed_q_uids = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_q_uids.add(json.loads(line)['q_uid'])
                except (json.JSONDecodeError, KeyError):
                    continue

    data_to_process = [
        item for item in flattened_data if item['q_uid'] not in completed_q_uids]
    logger.info(
        f"After filtering: {len(data_to_process)} / {len(flattened_data)} tasks remaining.")

    # --- Inference loop ---
    generation_config = dict(max_new_tokens=32, do_sample=False)
    for item in tqdm(data_to_process, desc="Inference Progress"):
        try:
            pixel_values, num_patches_list = load_video(
                item['video_path'], num_segments=args.nframes)
            if pixel_values is None:
                continue

            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join(
                [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            formatted_options = "\n".join(
                [f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(item['choices'])])
            question_text = prompt_template.format(
                question=item['question'], options=formatted_options)
            full_question = video_prefix + question_text

            response = model.chat(tokenizer, pixel_values, full_question, generation_config,
                                  num_patches_list=num_patches_list, history=None, return_history=False)
            if 'assistant\n' in response:
                response = response.split('assistant\n')[-1].strip()

            pred_choice = extract_characters_regex(response)

            output_dict = {'q_uid': item['q_uid'], 'video_id': item['video_id'],
                           'response': response, 'pred_choice': pred_choice}
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error processing q_uid {item['q_uid']}: {e}")
            traceback.print_exc()

    logger.info(f"Inference complete. All results saved to: {output_jsonl_path}")


    evaluate_with_subset(output_jsonl_path, args.subset_answers_path)
    if not args.eval_subset_only:
        create_submission_file(output_jsonl_path, args.result_dir)


if __name__ == "__main__":
    main()
