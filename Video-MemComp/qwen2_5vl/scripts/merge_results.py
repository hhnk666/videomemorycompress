#!/usr/bin/env python3
import os
import json
import argparse
import glob


def merge_results(result_dir, run_name):
    """Merge result files from multiple GPUs"""
    output_pattern = os.path.join(
        result_dir, "output", f"{run_name}_gpu*_*.jsonl")
    output_files = glob.glob(output_pattern)

    if not output_files:
        print(f"No matching output files found: {output_pattern}")
        return

    print(f"Found {len(output_files)} output files:")
    for f in output_files:
        print(f"  {f}")

    merged_output = []
    for file_path in sorted(output_files):
        with open(file_path, 'r') as f:
            for line in f:
                merged_output.append(json.loads(line.strip()))

    merged_output_path = os.path.join(
        result_dir, f"{run_name}_merged_output.jsonl")
    with open(merged_output_path, 'w') as f:
        for item in merged_output:
            f.write(json.dumps(item) + '\n')

    print(f"Merged output saved to: {merged_output_path}")
    print(f"Total samples processed: {len(merged_output)}")

    # Merge drop info files (if exist)
    drop_pattern = os.path.join(result_dir, "drop", f"{run_name}_gpu*_*.jsonl")
    drop_files = glob.glob(drop_pattern)

    if drop_files:
        print(f"Found {len(drop_files)} drop files")
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

        print(f"Merged drop info saved to: {merged_drop_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    merge_results(args.result_dir, args.run_name)


if __name__ == "__main__":
    main()
