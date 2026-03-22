# 🧠 Human Behavior Atlas

Official repository for **Human Behavior Atlas (HBA)**, a large-scale benchmark for human behavior analysis. 

Human Behavior Atlas provides a comprehensive training and evaluation framework spanning diverse behavioral domains, including emotion recognition, sentiment understanding, humor and sarcasm detection, intent recognition, non-verbal communication, and mental health indicators. The benchmark is designed to facilitate the development and evaluation of **socially intelligent foundation models** capable of grounded behavioral reasoning across multimodal inputs.

---

## 📰 News

- 🚀 **[March 2026] All code, models, and benchmark data for the ICLR paper Human Behavior Atlas has been uploaded!**
- 📄 **[Feb 2026] Preprint on a follow-up paper, OmniSapiens 2.0 available on arXiv: https://arxiv.org/pdf/2602.10635**
- 🎉 **[Jan 2026] Human Behavior Atlas accepted to ICLR 2026 Main Conference: https://openreview.net/forum?id=ZKE23BBvlQ**
- 📄 **[October 2025] Preprint of Human Behavior Atlas Paper released**
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

The following OmniSapiens models are publicly available on Hugging Face:

| Model | Description | Link |
|---|---|---|
| OmniSapiens-7B RL | Trained with GRPO on HBA | [ddvd233/OmniSapiens-7B-RL](https://huggingface.co/ddvd233/OmniSapiens-7B-RL) |
| OmniSapiens-7B SFT | SFT model with classification + QA heads | [keentomato/omnisapiens_sft](https://huggingface.co/keentomato/omnisapiens_sft) |
| OmniSapiens BAM — Humour Detection | BAM adapter for humour detection | [keentomato/omnisapiens_bam_humour_detection](https://huggingface.co/keentomato/omnisapiens_bam_humour_detection) |
| OmniSapiens BAM — Sentiment Polarity (MOSEI) | BAM adapter for sentiment polarity | [keentomato/omnisapiens_bam_sentiment_polarity_mosei](https://huggingface.co/keentomato/omnisapiens_bam_sentiment_polarity_mosei) |
| OmniSapiens BAM — Sarcasm Detection | BAM adapter for sarcasm detection | [keentomato/omnisapiens_bam_sarcasm_detection](https://huggingface.co/keentomato/omnisapiens_bam_sarcasm_detection) |

OmniSapiens-7B RL is the first iteration of a unified multimodal behavioral model for social reasoning and behavioral understanding across diverse behavioral domains. The BAM models pertaining to higher-performing specialized adapters released to support the community's downstream tasks. OmniSapiens-7B 2.0 will be released after the review process

---

## ⚙️ Training

### Installation

We use [VERL](https://github.com/volcengine/verl) v0.5.0.dev (included as a submodule) for reinforcement learning training. Follow the steps below to set up the environment. The same environment is utilized for SFT and BAM.

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

**8. Install LLM judge evaluation dependencies:**

```bash
pip install pandas tqdm tenacity anthropic openai wandb
```

These are required for `misc_eval_utils/llm_grader/llm_judge_eval.py`, which grades free-form QA outputs using an LLM judge.

**9. Set up PYTHONPATH:**

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

### Supervised Fine-Tuning (SFT)

SFT training follows a three-stage pipeline. Each stage builds on the previous one, so the stages must be run in order.

**Stage 1 — Classification Training**

```bash
cd sft
bash run_classification.sh
```

This trains the OmniSapiens model — a modified Qwen2.5-Omni with per-task classification heads — using LoRA fine-tuning across all behavioral datasets. The model is optimized for classification performance (cross-entropy loss). After training, note the path to the best-performing checkpoint; it will be used as input to Stage 2.

**Stage 2 — QA Training**

```bash
bash run_qa.sh
```

Set `--load_checkpoint_path` in `run_qa.sh` to the best checkpoint directory from Stage 1.

In this stage, the backbone (encoder + classification heads) is **frozen**, and only the **decoder (`lm_head`)** is trained with a language modeling loss. The intuition is that OmniSapiens is fundamentally an edited version of Qwen with appended classification heads, and classification performance is the primary optimization target. Unfreezing the backbone during QA training would risk disrupting the classification representations established in Stage 1. By training only the language model head, the model learns to produce free-form textual responses for open-ended QA datasets while leaving the classification behavior untouched.

**Stage 3 — BAM (Behavioral Adapter Module)**

```bash
bash run_bam.sh
```

Set the checkpoint path in `run_bam.sh` to the trained model from Stage 1 or Stage 2. BAM trains lightweight per-dataset residual adapters for video and audio features, projecting raw behavioral features into corrections applied to the model's penultimate hidden layer. The rest of the model remains frozen. The script processes one dataset at a time and automatically infers modality and task type from the data.

### Validation

Validation behavior differs by dataset type and applies uniformly across **all training modes** (RL, SFT classification, SFT QA, and BAM):

**Classification datasets** — Validation metrics are logged to W&B automatically during training. No additional steps are required; results will appear in your W&B dashboard as training progresses.

**QA datasets (IntentQA, MIMEQA, SIQ2)** — Because these tasks require free-form text generation, accuracy cannot be computed with exact string matching. After training, run the LLM judge evaluation script:

```bash
cd misc_eval_utils/llm_grader
bash run_llm_judge_eval.sh
```

Edit `run_llm_judge_eval.sh` to set the path to your model's prediction output file. The script uses an LLM judge (OpenAI or Anthropic) to grade each prediction against the ground truth and optionally logs results to W&B. Set the `MIT_OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable depending on which provider you use.

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
