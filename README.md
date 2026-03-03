# 🎬 Video-MemComp

---

# 📚 Table of Contents

* [📄 Paper](#-paper)
* [🔍 Overview](#-overview)
* [📊 Experimental Results](#-experimental-results)
* [✨ Key Features](#-key-features)
* [🧠 Supported Backbones](#-supported-backbones)
* [📁 Repository Structure](#-repository-structure)
* [📥 Model Download Instructions](#-model-download-instructions)

  * [Qwen2-VL-7B](#1️⃣-qwen2-vl-7b)
  * [Qwen2.5-VL](#2️⃣-qwen25-vl-3b--7b)
  * [LLaVA-OneVision-7B](#3️⃣-llava-onevision-7b)
  * [InternVL-3.5-2B](#4️⃣-internvl-35-2b)
* [📊 Supported Benchmarks](#-supported-benchmarks)
* [📥 Dataset Setup](#-dataset-setup)

  * [EgoSchema](#egoschema)
  * [MLVU](#mlvu)
  * [VideoMME](#videomme)
  * [OVO-Bench](#ovo-bench)
* [⚙️ Environment Setup](#️-environment-setup)
* [🚀 Running Experiments](#-running-experiments)
* [📊 Reproducibility](#-reproducibility)
* [📌 Notes](#-notes)
* [📎 Citation](#-citation)

---
## 📄 Paper

**Video-MemComp: Extreme O(1)-Memory Compression for Streaming Video Understanding via Taylor Expansion**

<img width="1660" height="1117" alt="image" src="https://github.com/user-attachments/assets/7b6fcf51-4101-434a-85eb-5f8187df02d1" />

> *Figure 1. While standard Video-LLM models suffer from linear O(T)-memory growth, Video-MemComp maintains a strictly O(1)-memory via a Taylor expansion-guided hybrid strategy, enabling efficient streaming video understanding.*

This repository contains the **official implementation** of Video-MemComp.

---

# 🔍 Overview

Video-MemComp is a structure-aware KV memory compression framework for long-form and streaming video understanding with Multimodal Large Language Models (MLLMs).

Unlike heuristic KV pruning or retrieval-based approaches, Video-MemComp is derived from a first-order Taylor expansion of error propagation and decomposes compression error into:

* **Local Approximation Error**

* **Layer-wise Structural Sensitivity**
  
<img width="1408" height="717" alt="image" src="https://github.com/user-attachments/assets/e784ccff-4d16-459b-8016-8f4190509c1a" />

> *Figure 2. Overview of our Video-MemComp Framework. Guided by Taylor expansion (top), we split global memory compression into two objectives: 1) Minimizing Local Approximation Error (middle) via Pre-RoPE Aggregation and Intra-layer KV Pruning. 2) Managing Structural Sensitivity (bottom) via a Tri-Hybrid Budget Allocation strategy.*

The framework enables:

* Strict **O(1)** KV memory w.r.t. video length
* No KV offloading
* No retrieval latency
* Strong performance across offline and streaming benchmarks

---

# 📊 Experimental Results

<img width="1610" height="805" alt="image" src="https://github.com/user-attachments/assets/3be76d0e-bc57-4655-80ec-68b812817701" />

> *Table 1. Performance Comparison on Offline Video Understanding Benchmarks. We evaluate Video-MemComp against state-of-the-art compression methods and full-cache baselines. Our method meets or exceeds full-cache performance under strict memory constraints.*

<br>

<img width="781" height="449" alt="image" src="https://github.com/user-attachments/assets/a79ac8cb-e435-4f94-9891-6d4559dcdf79" />

> *Figure 3. Performance on MLVU with Qwen2-VL-7B. Our method outperforms strong baselines across all memory budgets, and even surpasses the Full KV baseline at 6K, 12K, and 24K KV budgets, showing that Video-MemComp removes redundancy while preserving key semantics.*

---

# ✨ Key Features

* O(1) KV cache growth w.r.t. video length
* Works for both offline & streaming video benchmarks
* Supports multiple MLLM backbones
* Backbone-specific hyperparameters provided
* Reproducible evaluation scripts

---

# 🧠 Supported Backbones

| Model              | Status |
| ------------------ | ------ |
| Qwen2-VL-7B        | ✅      |
| Qwen2.5-VL-3B      | ✅      |
| Qwen2.5-VL-7B      | ✅      |
| LLaVA-OneVision-7B | ✅      |
| InternVL-3.5 (2B)  | ✅      |

---

# 📁 Repository Structure

After downloading models and datasets, your directory should look like:

```bash
Video-MemComp/
├── data/
│   ├── egoschema/
│   ├── mlvu/
│   ├── videomme/
│   └── ovobench/
│
├── model/
│   ├── InternVL3_5/
│   ├── Qwen2___5-VL-3B-Instruct/
│   ├── Qwen2___5-VL-7B-Instruct/
│   ├── qwen2vl/
│   └── llava-onevision-qwen2-7b-ov-hf/
│
├── internvl/
├── llavaov/
├── qwen2_5vl/
└── qwen2_vl/
```

> ⚠️ Model weights and datasets are NOT included in this repository and must be downloaded manually.

---

# 📥 Model Download Instructions

Please download the following models and place them under `model/` as shown above.

---

## 1️⃣ Qwen2-VL-7B

Model: Qwen2-VL

Download from:[https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

Place into:

```
model/qwen2vl/
```

---

## 2️⃣ Qwen2.5-VL (3B / 7B)

Model: Qwen2.5-VL

Download from:
[https://huggingface.co/Qwen](https://huggingface.co/Qwen)

Place into:

```
model/Qwen2___5-VL-3B-Instruct/
model/Qwen2___5-VL-7B-Instruct/
```

---

## 3️⃣ LLaVA-OneVision-7B

Model: LLaVA-OneVision

Download from:[https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)

Place into:

```
model/llava-onevision-qwen2-7b-ov-hf/
```

---

## 4️⃣ InternVL-3.5-2B

Model: InternVL

Download from:[https://huggingface.co/OpenGVLab/InternVL3_5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B)

Place into:

```
model/InternVL3_5/
```

---

# 📊 Supported Benchmarks

* VideoMME
* MLVU
* EgoSchema
* OVO-Bench

---

# 📥 Dataset Setup

---

## EgoSchema

Dataset: EgoSchema

Official repo:[https://github.com/egoschema/EgoSchema](https://github.com/egoschema/EgoSchema)

Expected structure:

```
data/egoschema/
├── full.json
├── subset_answers.json
└── videos/
```

---

## MLVU

Dataset: MLVU

Official repo:[https://github.com/MLVU-benchmark/MLVU](https://github.com/MLVU-benchmark/MLVU)

Place under:

```
data/mlvu/
```

---

## VideoMME

Dataset: VideoMME

Official repo:
[https://github.com/VideoMME/VideoMME](https://github.com/VideoMME/VideoMME)

Place under:

```
data/videomme/
```

Ensure videos are extracted into:

```
data/videomme/videos/
```

---

## OVO-Bench

Dataset: OVO-Bench

Place under:

```
data/ovobench/
├── chunked_videos/
├── src_videos/
└── ovo_bench_new.json
```

---

# ⚙️ Environment Setup

We recommend one Conda environment per backbone.

Example (Qwen2.5-VL):

```bash
conda env create -f qwen2_5vl/environment.yml
conda activate qwen2_5vl
pip install -r qwen2_5vl/requirements.txt
```

CUDA / PyTorch versions are backbone-specific.
Please follow the provided `environment.yml`.

---

# 🚀 Running Experiments

All experiments are launched via provided shell scripts.

Example:

### EgoSchema (Qwen2.5-VL)

```bash
cd qwen2_5vl/scripts
bash eval_egoschema.sh
```

### MLVU

```bash
bash eval_mlvu.sh
```

### VideoMME

```bash
bash eval_videomme.sh
```

### Streaming (OVO-Bench)

```bash
bash eval_ovobench.sh
```

---

# 📊 Reproducibility

* All hyperparameters follow the paper.
* Results are computed on official dev splits.
* No KV offloading.
* No retrieval-based augmentation.

---

# 📌 Notes

* Please verify model and dataset paths before running.
* It is recommended to first run subset/debug mode before full evaluation.
* GPU with large memory (e.g., A6000) is recommended for long videos.

---

# 📎 Citation
