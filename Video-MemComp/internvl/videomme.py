import sys
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import json
import re
from transformers import AutoTokenizer
from decord import VideoReader, cpu
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Ensure custom model definitions are in Python path
# sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from InternVL3_5 import InternVLChatModel
except ImportError:
    from transformers import AutoModelForCausalLM as InternVLChatModel
import debugpy

# --- InternVL 3.5 video processing functions ---
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
        frame_indices = get_index(bound, fps, max_frame,
                                  first_idx=0, num_segments=num_segments)

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
        print(f"Error loading video {video_path}: {e}")
        return None, None

def extract_characters_regex():
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()

    matches = re.search(r"^\(?([A-D])\)?", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    matches = re.search(r"\b([A-D])\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    return ""


torch.manual_seed(1919810)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():

    CKPT_PATH = os.path.join(PROJECT_ROOT, 'model', 'InternVL3_5')
    TASK_PARQUET = os.path.join(PROJECT_ROOT, 'data', 'videomme', 'videomme', 'test-00000-of-00001.parquet')
    VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'videomme', 'videos')
    RESULT_FILE = os.path.join(PROJECT_ROOT, 'internvl','eval_results', 'videomme', 'result_videomme_internvl_100f.jsonl')
    NUM_FRAMES = 100

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    NUM_FRAMES = 100  

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    model = InternVLChatModel.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        device_map="auto"
    ).eval()
    model.inter_frame_threshold = 0.5
    tokenizer = AutoTokenizer.from_pretrained(
        CKPT_PATH, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=128, do_sample=False)
    task_df_full = pd.read_parquet(TASK_PARQUET)

    processed_ids = set()
    if os.path.exists(RESULT_FILE):
        print(f"Found existing result file: {RESULT_FILE}. Reading completed tasks...")
        with open(RESULT_FILE, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try:
                    data = json.loads(line)
                    if 'question_id' in data:
                        processed_ids.add(data['question_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line in result file: {line.strip()}")

        original_count = len(task_df_full)
        task_df = task_df_full[~task_df_full['question_id'].isin(
            processed_ids)]
    else:
        task_df = task_df_full

    prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
Options: {options}
The best answer is:"""

    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        for _, row in tqdm(task_df.iterrows(), total=len(task_df), desc="评测进度"):
            video_path = os.path.join(VIDEO_DIR, row.videoID + '.mp4')
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                continue

            pixel_values, num_patches_list = load_video(
                video_path, num_segments=NUM_FRAMES, max_num=1)

            if pixel_values is None:
                continue
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join(
                [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question_text = prompt_template.format(
                question=row.question,
                options='\n'.join(row.options.tolist())
            )
            full_question = video_prefix + question_text

            try:
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
                if 'assistant\n' in response:
                    response = response.split('assistant\n')[-1].strip()
                output_dict = {
                    'question_id': row.question_id,
                    'video_id': row.videoID,
                    'question': row.question,
                    'options': row.options.tolist(),
                    'answer': row.answer,
                    'response': response,
                    'extracted_answer': extract_characters_regex(response)
                }
                f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
                f.flush()

            except Exception as e:
                print(
                    f"Error processing video_id {row.videoID} (question_id: {row.question_id}): {e}")

    print(f"Evaluation complete. All results saved to: {RESULT_FILE}")

    # --- 5. Calculate accuracy ---
    # This part does not need modification; it reads the complete file containing all (historical + current) results for calculation
    correct_count = 0
    total_count = 0
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_count += 1
                if data['extracted_answer'] == data['answer']:
                    correct_count += 1
            except (json.JSONDecodeError, KeyError):
                continue  

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n--- Evaluation Results ---")
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
