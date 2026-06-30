# Unified Parquet SFT + GRPO on upstream verl

> **⚠️ This describes the upstream-verl _vision_ path, which lives on the
> [`parquet_dataloader`](https://github.com/MIT-MI/human_behavior_atlas/tree/parquet_dataloader)
> branch — not `main`.** `main`'s `verl/` submodule is the omni fork (`DDVD233/verl`), so the
> GRPO recipe below will not run from `main`. To use it: `git checkout parquet_dataloader`.
> For the omni SFT + GRPO/TARPO path that runs on `main`, see [`OMNI_TRAINING.md`](OMNI_TRAINING.md).

This branch (`parquet_dataloader`) adds a **model-agnostic, parquet-native** supervised
fine-tuning (SFT) pipeline and a **GRPO example that runs on upstream
[volcengine/verl](https://github.com/volcengine/verl) `@ main`** — no dependency on the
`mirl` / `DDVD233-verl` omni fork.

The `verl/` submodule on this branch points at **upstream `volcengine/verl @ main`**
(not the fork). The SFT path needs *no* verl at all; only GRPO uses it.

## Layout

`training/sft/` and `training/rl/` are sibling stages under `training/` (SFT → merge → GRPO):

```
human_behavior_atlas/
├── training/
│   ├── sft/  # supervised fine-tuning (accelerate; no verl needed)
│   └── rl/   # GRPO RL on upstream verl
└── verl/     # submodule → volcengine/verl @ main
```

## What's new

| File | Purpose |
|------|---------|
| `training/sft/dataset/sft_parquet_dataset.py` | Direct-from-parquet dataloader (pyarrow, no HF-datasets cache). Adds **modality filtering** (via `modality_signature`), **seek-based video frame sampling** (HBA clips are 40–235 MB), and inactive-tag stripping. |
| `training/sft/trainer/sft_trainer.py` | accelerate + FSDP teacher-forced LM trainer (response-only loss). |
| `training/sft/train_sft.py` | Model-agnostic entry point. Model class chosen by `model.arch`. |
| `training/sft/merge_lora.py` | Merge the LoRA adapter into the base model (simple VLM path + omni-thinker path). |
| `training/sft/configs/config_sft.yaml` | Vision SFT config (Qwen3-VL-4B). |
| `training/sft/configs/config_sft_gemma4_audio.yaml` | Audio SFT config (Gemma 4 E4B) — uses HBA's audio rows, upstream-only. |
| `training/rl/prepare_grpo_data.py` | Convert HBA parquet → **verl-native** RLHF parquet (prompt as chat messages, video → frame-path lists, `reward_model.ground_truth`). |
| `training/rl/reward_function.py` | verl `reward_manager=batch` reward (exact-match + format + cosine-sim). |
| `training/rl/run_grpo.sh` | `verl.trainer.main_ppo` GRPO launcher (fork-only keys removed). |

## Model / modality support on latest verl (June 2026)

HBA is modality-mixed and **audio-heavy** (≈95% audio, e.g. CREMA-D; sparse video).

| Path | Model | Status on **upstream** verl |
|------|-------|------------------------------|
| **SFT (vision)** | Qwen3-VL-4B-Instruct | ✅ works; trains on the image/video subset |
| **SFT (audio)** | Gemma 4 E4B (audio+image, no fork) | ✅ works (needs transformers 5.x); uses HBA's audio bulk |
| **GRPO (vision)** | Qwen3-VL-4B-Instruct | ✅ works; verl-native parquet + vLLM ≥ 0.11 |
| **GRPO (audio/omni)** | Qwen2.5/3-Omni, Gemma-omni | ⚠️ no clean upstream path yet — stays on the mirl fork or is experimental. `verl-omni` is for diffusion/generation RL, not understanding; Qwen3.5 has a [verl/transformers/vLLM version conflict](https://github.com/verl-project/verl/issues/5937). |

The dataloader/trainer are model-agnostic — the model is just a config value, so new
models can be added gradually by adding a config.

## Environment

The repo (with the upstream verl submodule):
```bash
git clone --recurse-submodules -b parquet_dataloader https://github.com/MIT-MI/human_behavior_atlas.git
# or, if already cloned:  git submodule update --init --recursive
```

**SFT env (light, no verl):**
```bash
conda create -n hba_sft python=3.11 && conda activate hba_sft
pip install -r training/sft/requirements_accelerate.txt
```

**GRPO env (upstream verl + vLLM ≥ 0.11):**
```bash
conda create -n hba_verl python=3.11 && conda activate hba_verl
pip install "vllm>=0.11,<0.12"
pip install -e ./verl                       # upstream volcengine/verl @ main submodule
pip install -r training/rl/requirements_grpo.txt
```

## Data

Point training at the parquet directory of
[`human_behavior_atlas_v2`](https://huggingface.co/datasets) (or any HF-native parquet
with columns `problem`, `answer`, `dataset`, `task`, `modality_signature`,
`images`/`videos`/`audios` list-of-binary).

`filter_by_modality: true` keeps only rows whose `modality_signature` matches the active
`modalities` — e.g. a vision model trains only on image/video rows.

## SFT

```bash
cd training/sft
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_sft.sh 1                          # Qwen3-VL-4B, 1 GPU
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_sft.sh 4 configs/config_sft_gemma4_audio.yaml
```

Key config knobs (`configs/config_sft.yaml`):
- `model.arch`: `auto` (AutoModelForImageTextToText: Qwen-VL, Gemma 3/4) · `omni_thinker` (Qwen2.5-Omni) · `causal_lm` (text-only).
- `dataset_config.modalities`: e.g. `"images,videos"` or `"audio"`.
- `dataset_config.filter_by_modality`: keep only matching rows.
- `dataset_config.video_nframes`, `max_prompt_length`, `max_length`.

Merge the LoRA adapter for inference / GRPO init:
```bash
python merge_lora.py --arch auto \
    --base_model Qwen/Qwen3-VL-4B-Instruct \
    --adapter_path ./checkpoints/sft_qwen3vl_hba/step_XX \
    --output_path  ./checkpoints/sft_qwen3vl_hba_merged
```

## GRPO (vision, upstream verl)

```bash
conda activate hba_verl
cd training/rl

# 1) Build verl-native parquet from the HBA video subset (one-time)
python prepare_grpo_data.py --data_dir $HBA_DATA_DIR --out_dir ./grpo_data \
    --split train      --modality video --max_samples 256
python prepare_grpo_data.py --data_dir $HBA_DATA_DIR --out_dir ./grpo_data \
    --split validation --modality video --max_samples 32

# 2) Train (defaults to base Qwen3-VL-4B; set MODEL_PATH to the merged SFT ckpt)
MODEL_PATH=../sft/checkpoints/sft_qwen3vl_hba_merged ./run_grpo.sh 1
```

`run_grpo.sh` removes the fork-only keys (`data.modalities`,
`data.train_modality_batching`, `data.format_prompt`): the format instruction is baked
into the prompt by `prepare_grpo_data.py`, and the parquet is verl-native.

## Verification status (this branch)

Validated on a single NVIDIA H20 (no fork):
- ✅ **SFT end-to-end**: Qwen3-VL-4B LoRA on the HBA video subset ran real optimizer
  steps with decreasing loss (`3.30 → 2.25 → 1.97 → 0.68` over 4 steps) and saved a
  checkpoint — full model load + multimodal forward/backward via the parquet dataloader.
- ✅ **GRPO data path**: HBA parquet → `prepare_grpo_data.py` → **loads through upstream
  `verl.utils.dataset.RLHFDataset`** with the Qwen3-VL-4B processor (chat template +
  `qwen_vl_utils` frame loading); `reward_model.ground_truth` and `data_source` resolve.
- ✅ Reward function (`compute_score_batch`) unit-checks; `hba_verl` env builds
  (verl @ main + vLLM 0.11.2 + transformers 4.57).

Full multi-GPU vLLM RL training and large-scale SFT convergence should be run on the
target cluster; the examples above are configured to run there directly.
