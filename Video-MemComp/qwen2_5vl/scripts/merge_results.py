#!/usr/bin/env python3
import os
import json
import argparse
import glob


def merge_results(result_dir, run_name):
    """合并多个GPU的结果文件"""

    # 查找所有相关的输出文件
    output_pattern = os.path.join(
        result_dir, "output", f"{run_name}_gpu*_*.jsonl")
    output_files = glob.glob(output_pattern)

    if not output_files:
        print(f"没有找到匹配的输出文件: {output_pattern}")
        return

    print(f"找到 {len(output_files)} 个输出文件:")
    for f in output_files:
        print(f"  {f}")

    # 合并输出文件
    merged_output = []
    for file_path in sorted(output_files):
        with open(file_path, 'r') as f:
            for line in f:
                merged_output.append(json.loads(line.strip()))

    # 保存合并后的结果
    merged_output_path = os.path.join(
        result_dir, f"{run_name}_merged_output.jsonl")
    with open(merged_output_path, 'w') as f:
        for item in merged_output:
            f.write(json.dumps(item) + '\n')

    print(f"合并输出保存到: {merged_output_path}")
    print(f"总共处理了 {len(merged_output)} 个样本")

    # 合并drop信息文件（如果存在）
    drop_pattern = os.path.join(result_dir, "drop", f"{run_name}_gpu*_*.jsonl")
    drop_files = glob.glob(drop_pattern)

    if drop_files:
        print(f"找到 {len(drop_files)} 个drop文件")
        merged_drop = []
        for file_path in sorted(drop_files):
            with open(file_path, 'r') as f:
                for line in f:
                    merged_drop.append(json.loads(line.strip()))

        merged_drop_path = os.path.join(
            result_dir, f"{run_name}_merged_drop.jsonl")
        with open(merged_drop_path, 'w') as f:
            for item in merged_drop:
                f.write(json.dumps(item) + '\n')

        print(f"合并drop信息保存到: {merged_drop_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    merge_results(args.result_dir, args.run_name)


if __name__ == "__main__":
    main()
