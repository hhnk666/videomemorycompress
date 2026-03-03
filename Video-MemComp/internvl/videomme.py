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

# 确保自定义模型的定义在 Python 路径中
# sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
try:
    from InternVL3_5 import InternVLChatModel
except ImportError:
    print("导入失败。请确保 'InternVL3_5' 模块可访问。")
    # 替换为你的本地 InternVL 模型路径
    from transformers import AutoModelForCausalLM as InternVLChatModel
import debugpy

# --- InternVL 3.5 视频处理函数 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
try:
    # 5678 is the default attach port in the VS Code debug configurations.
    debugpy.listen(("localhost", 9521))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


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
    # 确保索引不越界
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


# --- 答案提取函数 ---
def extract_characters_regex(s):
    s = s.strip()
    # 移除解释性前缀
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
    ]
    for prefix in answer_prefixes:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()

    # 优先匹配开头的括号或字母
    matches = re.search(r"^\(?([A-D])\)?", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    # 其次匹配句子中的独立字母
    matches = re.search(r"\b([A-D])\b", s, re.IGNORECASE)
    if matches:
        return matches.group(1).upper()

    return ""


torch.manual_seed(1919810)


# --- 动态获取项目根目录 ---
# 假设当前脚本位于 project_root/internvl/xxx.py
# 则 os.path.abspath(__file__) 是脚本绝对路径
# dirname 第一次返回 project_root/internvl
# dirname 第二次返回 project_root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 将项目根目录加入 python 搜索路径，确保 InternVL3_5 等模块能被导入
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():

    CKPT_PATH = os.path.join(PROJECT_ROOT, 'model', 'InternVL3_5')
    TASK_PARQUET = os.path.join(PROJECT_ROOT, 'data', 'videomme', 'videomme', 'test-00000-of-00001.parquet')
    VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'videomme', 'videos')
    RESULT_FILE = os.path.join(PROJECT_ROOT, 'internvl','eval_results', 'videomme', 'result_videomme_internvl_100f.jsonl')
    NUM_FRAMES = 100

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    NUM_FRAMES = 100  # <--- 帧率选择

    # 创建结果目录
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    # --- 2. 加载模型和 Tokenizer ---
    print("正在加载模型...")
    # 注意：根据之前的讨论，如果遇到Flash Attention的错误，可能需要将 use_flash_attn 设置为 True 或 False
    # 在你的代码中是 False，这里保持一致
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
    print("模型加载完成。")

    # --- 3. 加载数据集并准备断点续传 ---
    task_df_full = pd.read_parquet(TASK_PARQUET)

    # --- 新增：断点续传逻辑 ---
    processed_ids = set()
    if os.path.exists(RESULT_FILE):
        print(f"发现已存在的结果文件: {RESULT_FILE}。正在读取已完成的任务...")
        with open(RESULT_FILE, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try:
                    # 解析已有的json行，以防文件损坏
                    data = json.loads(line)
                    if 'question_id' in data:
                        processed_ids.add(data['question_id'])
                except json.JSONDecodeError:
                    print(f"警告: 无法解析结果文件中的行: {line.strip()}")

        original_count = len(task_df_full)
        # 从任务列表中筛选掉已经处理过的条目
        task_df = task_df_full[~task_df_full['question_id'].isin(
            processed_ids)]
        print(
            f"已完成 {len(processed_ids)} 个任务。将继续处理剩余的 {len(task_df)} / {original_count} 个任务。")
    else:
        # 如果结果文件不存在，则处理全部任务
        task_df = task_df_full
        print(f"未发现结果文件，将开始处理全部 {len(task_df)} 个任务。")

    # --- 4. 推理循环 ---
    prompt_template = """Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
Options: {options}
The best answer is:"""

    # --- 修改：以追加模式 'a' 打开文件 ---
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        # --- 修改：迭代经过筛选的 task_df ---
        for _, row in tqdm(task_df.iterrows(), total=len(task_df), desc="评测进度"):
            video_path = os.path.join(VIDEO_DIR, row.videoID + '.mp4')
            if not os.path.exists(video_path):
                print(f"视频文件未找到: {video_path}")
                continue

            # 加载视频并设置帧数
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=NUM_FRAMES, max_num=1)

            if pixel_values is None:
                continue

            # 注意：确保你的GPU有足够显存处理64帧的pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # 构建问题
            video_prefix = ''.join(
                [f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question_text = prompt_template.format(
                question=row.question,
                options='\n'.join(row.options.tolist())
            )
            full_question = video_prefix + question_text

            try:
                # 模型推理
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )
                # --- 新增：在这里对 response 字符串进行清理 ---
                # 我们只取 'assistant\n' 之后的内容，这才是模型真正的回答
                if 'assistant\n' in response:
                    response = response.split('assistant\n')[-1].strip()

                # 记录结果
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

                # --- 新增：强制将缓冲区内容写入磁盘 ---
                f.flush()

            except Exception as e:
                print(
                    f"处理 video_id {row.videoID} (question_id: {row.question_id}) 时出错: {e}")

    print(f"评测完成，所有结果已保存至: {RESULT_FILE}")

    # --- 5. 计算准确率 ---
    # 这个部分不需要修改，它会读取包含所有（历史+本次）结果的完整文件进行计算
    correct_count = 0
    total_count = 0
    print("正在计算最终准确率...")
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_count += 1
                if data['extracted_answer'] == data['answer']:
                    correct_count += 1
            except (json.JSONDecodeError, KeyError):
                continue  # 跳过损坏的行

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n--- 评测结果 ---")
    print(f"总样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
