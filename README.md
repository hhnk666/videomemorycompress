# video-memcomp
## 🔍 Overview

**Video-MemComp** is a structure-aware memory compression framework for **long-form and streaming video understanding** with Multimodal Large Language Models (MLLMs).

Unlike heuristic KV pruning or retrieval-based methods, Video-MemComp is **theoretically grounded** in a first-order Taylor expansion of error propagation, decomposing compression error into:

* **Local Approximation Error** (mitigated via *Pre-RoPE Aggregation*)
* **Layer-wise Structural Sensitivity** (managed via *dynamic budget allocation*)

This enables **strict O(1) KV memory** w.r.t. video length while preserving or even improving accuracy.

---

## ✨ Key Features

* ✅ **Strict O(1) KV cache** (constant w.r.t. video length)
* ✅ Works for **offline & streaming video benchmarks**
* ✅ Supports multiple MLLM backbones:

  * Qwen2-VL / Qwen2.5-VL
  * LLaVA-OneVision
  * InternVL 3.5
* ✅ No CPU offloading / retrieval latency
* ✅ Ready-to-run evaluation scripts

---

## 📁 Repository Structure

```bash
Video-MemComp/
├── internvl/
│   ├── environment.yml
│   ├── requirements.txt
│   ├── egoschema.py
│   ├── egoschema.sh
│   ├── evaluate_mlvu_internvl.py
│   ├── evaluate_mlvu_internvl.sh
│   ├── videomme.py
│   ├── videomme.sh
│   └── InternVL3_5/
│       ├── configuration_intern_vit.py
│       ├── configuration_internvl_chat.py
│       ├── modeling_intern_vit.py
│       ├── modeling_internvl_chat.py
│       └── conversation.py
│
├── llavaov/
│   ├── environment.yml
│   ├── requirements.txt
│   ├── chat.py
│   ├── egoschema.py
│   ├── egoschema.sh
│   ├── mlvu.py
│   ├── mlvu.sh
│   ├── videomme.py
│   ├── eval_videomme.sh
│   └── llava_onevision/
│       ├── configuration_llava_onevision.py
│       ├── modeling_llava_onevision.py
│       └── llavaov.py
│
├── qwen2_5vl/
│   ├── environment.yml
│   ├── requirements.txt
│   ├── qwen2_5_vl/
│   │   ├── configuration_qwen2_5_vl.py
│   │   ├── modeling_qwen2_5_vl_DTD.py
│   │   ├── modular_qwen2_5_vl.py
│   │   └── processing_qwen2_5_vl.py
│   └── scripts/
│       ├── eval_egoschema.py
│       ├── eval_egoschema.sh
│       ├── eval_mlvu.sh
│       ├── eval_ovobench.sh
│       ├── eval_videomme.sh
│       ├── videomme.py
│       └── test_mlvu_minimal.sh
│
└── qwen2_vl/
    ├── environment.yml
    ├── requirements.txt
    ├── modeling_qwen2_vl.py
    ├── mlvu.py
    ├── mlvu.sh
    ├── videomme.py
    └── videomme.sh

```

---

## 🧪 Supported Benchmarks

### Offline Video Understanding

* **VideoMME**
* **MLVU**
* **EgoSchema**

### Streaming Video Understanding

* **RVS-Ego / RVS-Movie**
* **OVO-Bench**
* **StreamingBench**

All results follow the **official dev-set protocol** used in prior work (ReKV, StreamMem, InfiniPot-V).

---

## 🧠 Supported Models

| Backbone           | state |
| ------------------ | ------ |
| Qwen2-VL-7B        | ✅      |
| Qwen2.5-VL-3B      | ✅      |
| LLaVA-OneVision-7B | ✅      |
| InternVL-3.5-2B    | ✅      |

Each backbone has **model-specific hyperparameters** (budget weights, thresholds), already encoded in the provided scripts.

---

## ⚙️ Environment Setup

We recommend **one Conda environment per backbone**.

### Example: Qwen2.5-VL

```bash
conda env create -f qwen2_5vl/environment.yml
conda activate qwen2_5vl
pip install -r qwen2_5vl/requirements.txt
```

### Example: InternVL 3.5

```bash
conda env create -f internvl/environment.yml
conda activate internvl
pip install -r internvl/requirements.txt
```

> ⚠️ CUDA, PyTorch, and FlashAttention versions are **model-dependent**.
> Please follow the `environment.yml` for each backbone.

---

## 🚀 Running Experiments

All experiments are launched via **shell scripts**.
Each script supports a **quick debug mode** and a **full evaluation mode**.

---

### 1️⃣ EgoSchema (Offline)

#### Qwen2.5-VL

```bash
cd qwen2_5vl/scripts
bash eval_egoschema.sh
```

Key options inside the script:

```bash
RUN_SUBSET_ONLY="true"   # quick validation (≈500 samples)
NFRAMES=768              # fixed frame budget
DROP_METHOD="feature"   # Video-MemComp compression
```

---

### 2️⃣ MLVU (Offline)

#### LLaVA-OneVision

```bash
cd llavaov
bash mlvu.sh
```

#### Qwen2.5-VL

```bash
cd qwen2_5vl/scripts
bash eval_mlvu.sh
```

---

### 3️⃣ VideoMME

#### InternVL

```bash
cd internvl
bash videomme.sh
```

#### Qwen / LLaVA

```bash
bash eval_videomme.sh
```

---

### 4️⃣ Streaming Benchmarks

#### OVO-Bench

```bash
bash eval_ovobench.sh
```

#### StreamingBench (Real-time)

```bash
bash eval_streamingbench.sh
```

---

## 📊 Efficiency & Memory

* **No KV offloading**
* **No retrieval latency**
* **OOM-free on single A6000**
* TTFT reduced by **~2×** compared to ReKV

See paper Table for details.

---

## 🔬 Ablation Studies

The repo supports ablations for:

* Inter-frame only vs Inter + Intra-frame compression
* Budget allocation strategies
* Spatial aggregation thresholds

These are controlled via:

```bash
DROP_METHOD
DROP_THRESHOLD
NFRAMES
```

---

## 📌 Notes & Tips

* All paths in scripts are **absolute** by default — please modify them to your local setup.
* `RUN_SUBSET_ONLY=true` is **highly recommended** before full runs.
* For distributed runs, scripts auto-select a free `MASTER_PORT`.

