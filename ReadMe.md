# 🧠 Human Behavior Atlas

Official repository for **Human Behavior Atlas (HBA)**, a large-scale benchmark for unified psychological and social behavior understanding.

Human Behavior Atlas provides a comprehensive evaluation framework spanning diverse behavioral domains, including emotion recognition, sentiment understanding, humor and sarcasm detection, intent recognition, non-verbal communication, and mental health indicators. The benchmark is designed to facilitate the development and evaluation of **socially intelligent AI systems** capable of grounded behavioral reasoning across multimodal inputs.

---

## 📰 News

- 📄 **[Feb 2026] Preprint on a follow-up paper, OmniSapiens 2.0 available on arXiv: https://arxiv.org/pdf/2602.10635**
- 🎉 **[Jan 2026] Human Behavior Atlas accepted to ICLR 2026 Main Conference**
- 📄 **[October 2025] Human Behavior Atlas Paper available on arXiv**
- 🤖 **[October 2025] OmniSapiens-7B RL model released on Hugging Face**

(This section will be continuously updated.)

---

## 📄 Paper

**Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding**  
Accepted to **ICLR 2026 Main Conference**

📎 Human Behavior Atlas Paper:  
https://openreview.net/forum?id=ZKE23BBvlQ

---

## 🤗 Dataset

The Human Behavior Atlas benchmark is publicly available on Hugging Face:

**Dataset:**  
https://huggingface.co/datasets/keentomato/human_behavior_atlas

This dataset aggregates and standardizes multiple behavioral datasets into a unified evaluation framework for multimodal behavioral understanding.

### Benchmark Structure

The downloaded dataset contains the following:

- **JSONL splits** — centralized metadata files that define all samples:
  - `final_v8_train_cleaned_2.jsonl` — training set
  - `final_v8_val_cleaned.jsonl` — validation set
  - `final_v8_test_cleaned.jsonl` — test set
- **Raw media files** — video, audio, and text files referenced by each sample
- **Behavioral features** — pre-extracted feature files (`.pt`) for pose, OpenSMILE audio features, etc.

Each line in the JSONL files is a self-contained sample with all the information needed for loading. For example:

```json
{
  "problem": "<audio>\nDon't forget a jacket.\nThe above is a speech recording along with the transcript from a clinical context. What emotion is the speaker expressing? Answer with one word from the following: anger, disgust, fear, happy, neutral, sad",
  "answer": "sad",
  "images": [],
  "videos": [],
  "audios": ["cremad_dataset_audio/1077_DFA_SAD_XX.wav"],
  "dataset": "cremad",
  "texts": [],
  "modality_signature": "text_audio",
  "ext_video_feats": ["pose/cremad_dataset_audio/1077_DFA_SAD_XX.pt"],
  "ext_audio_feats": ["opensmile/cremad_dataset_audio/1077_DFA_SAD_XX.pt"],
  "task": "emotion_cls",
  "class_label": "sad"
}
```

Key fields:
- `problem` / `answer` — the prompt and ground-truth label
- `images`, `videos`, `audios`, `texts` — relative paths to raw media files
- `ext_video_feats`, `ext_audio_feats` — relative paths to pre-extracted behavioral feature files (`.pt`)
- `modality_signature` — indicates which modalities are present (e.g., `text_audio`, `video`, `text_video`)
- `dataset` — source dataset name
- `task` / `class_label` — behavioral task type and label

The dataloader uses this JSONL as the centralized index to locate and load all raw media and feature files for each sample into the model.

> **Note:** The JSONL files sit at the root level of the downloaded dataset, while the raw media files and feature files are nested within subdirectories. All paths in the JSONL are relative to the root level (i.e., the same directory as the JSONL files themselves).

---

## 🤖 Models

The OmniSapiens-7B RL model, trained using GRPO on the Human Behavior Atlas benchmark, is available on [Huggingface](https://huggingface.co/ddvd233/OmniSapiens-7B-RL)

OmniSapiens-7B RL is the first iteration of a unified multimodal behavioral model for social reasoning and behavioral understanding across diverse behavioral domains.

OmniSapiens-7B SFT and BAM will be released shortly!

---

## ⚙️ Training

### Installation

We use [VERL](https://github.com/volcengine/verl) v0.5.0.dev (included as a submodule) for reinforcement learning training. Follow the steps below to set up the environment.

**1. Clone this repository with the VERL submodule:**

```bash
git clone --recurse-submodules https://github.com/MIT-MI/human_behavior_atlas.git
cd human_behavior_atlas
```

If you have already cloned the repository without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

**2. Create a conda environment:**

```bash
conda create -n verl python==3.12.2
conda activate verl
```

**3. Install PyTorch with CUDA support:**

Check https://pytorch.org/get-started/locally/ for the latest compatible version. For example:

```bash
# CUDA 12.6
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# OR CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**4. Install VERL dependencies:**

```bash
cd verl
pip install -r requirements.txt
```

**5. Install vLLM with audio support:**

```bash
pip install "vllm[audio]==0.10.2"
pip install -U "setuptools>=78,<80"
```

**6. Install FlashAttention from source:**

> **Note:** Do not install FlashAttention via `pip install flash-attn` — compiling it from source often leads to the most stable installation. Please refer to https://github.com/Dao-AILab/flash-attention to compile from source.

**7. Install remaining dependencies:**

```bash
# torchcodec (install AFTER flash-attention to avoid version conflicts)
pip install torchcodec==0.4.0 --index-url https://download.pytorch.org/whl/cu126

# ffmpeg
conda install -c conda-forge "ffmpeg=6.*" "libgcc-ng>=13" "libstdcxx-ng>=13"

# Additional packages
pip install ujson scikit-learn qwen-vl-utils

# MathRuler (for reward computation)
git clone https://github.com/hiyouga/MathRuler.git
cd MathRuler && pip install . && cd ..
```

**8. Set up PYTHONPATH:**

Add the following to your `~/.bashrc` (or equivalent), replacing the path with the absolute path to your cloned `verl` submodule:

```bash
export PYTHONPATH="/path/to/human_behavior_atlas/verl:$PYTHONPATH"
```

### Reinforcement Learning Training

After downloading the [HBA dataset](https://huggingface.co/datasets/keentomato/human_behavior_atlas), you can launch GRPO training using the provided script:

```bash
cd verl/examples/grpo_trainer
bash _train_grpo_hba.sh
```

Edit `_train_grpo_hba.sh` to configure data paths, GPU allocation, and training hyperparameters for your setup.

> **Note:** While the paper uses the GRPO method, the integration with VERL is intended to make it convenient to use other RL variants (e.g., REINFORCE, DAPO). See the [VERL documentation](https://verl.readthedocs.io/) for further instructions on alternative algorithms.

---

## 👤 Maintainer

This repository is maintained by:

**Keane Ong** [Google Scholar](https://scholar.google.com/citations?user=fMPgRDMAAAAJ&hl=en) | [LinkedIn](www.linkedin.com/in/kowy) | [Email](keaneong@mit.edu)

**Sabri Boughorbel** [Google Scholar](https://scholar.google.nl/citations?user=EhFf3j0AAAAJ&hl=en) | [LinkedIn](https://www.linkedin.com/in/sabriboughorbel/) | [Email](sabri.boughorbel@gmail.com)

---

## 📚 Citation

If you use Human Behavior Atlas or OmniSapiens in your research, please cite the following:

```bibtex
@inproceedings{
ong2026human,
title={Human Behavior Atlas: Benchmarking Unified Psychological And Social Behavior Understanding},
author={Keane Ong and Wei Dai and Carol Li and Dewei Feng and Hengzhi Li and Jingyao Wu and Jiaee Cheong and Rui Mao and Gianmarco Mengaldo and Erik Cambria and Paul Pu Liang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=ZKE23BBvlQ}
}
```
