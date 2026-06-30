# Omni SFT + GRPO/TARPO on the verl fork (`parquet_dataloader_omni_mirl`)

This branch is the **omni / fork counterpart** to `parquet_dataloader`. It targets
**Qwen2.5-Omni-7B** (audio + video + image) on the **`DDVD233/verl` fork** (the omni-capable
verl, kept as the `verl/` submodule), reusing the same model-agnostic, parquet-native SFT
code but configured for the omni model, and adding the fork-only **GRPO** and **TARPO**
recipes that read the Human Behavior Atlas parquet directly.

| | `parquet_dataloader` | `parquet_dataloader_omni_mirl` (this branch) |
|---|---|---|
| verl | upstream `volcengine/verl @ main` | fork `DDVD233/verl` (submodule) |
| Model | Qwen3-VL-4B / Gemma 4 (vision/audio, no fork) | Qwen2.5-Omni-7B (omni) |
| Data into RL | converted verl-native parquet | **HBA parquet directly** (fork `data.modalities`) |
| RL algos | GRPO | **GRPO + TARPO** (TARPO is fork-only) |

## Layout

```
human_behavior_atlas/
├── training/
│   ├── sft/  # model-agnostic parquet SFT (config_sft.yaml -> Qwen2.5-Omni-7B, arch=omni_thinker)
│   └── rl/   # fork GRPO + TARPO (run_grpo.sh, run_tarpo.sh, reward_function/, format_prompt/)
└── verl/     # submodule → DDVD233/verl (omni fork)
```

The classification / QA / BAM omni trainers from `main` are retained under `training/sft/`.

## Environment (fork verl)

This path needs the **fork**, not upstream verl. Following the main ReadMe Training section:

```bash
git clone --recurse-submodules -b parquet_dataloader_omni_mirl https://github.com/MIT-MI/human_behavior_atlas.git
cd human_behavior_atlas
conda create -n verl python==3.12.2 && conda activate verl
# PyTorch (CUDA 12.6/12.8) per https://pytorch.org
pip install -r verl/requirements.txt
pip install "vllm[audio]==0.10.2"
pip install -r training/rl/requirements_grpo.txt
# FlashAttention from source (see main ReadMe), then:
export PYTHONPATH="$(pwd)/verl:$PYTHONPATH"
```

SFT alone (no verl) can also use the lighter env from `training/sft/requirements_accelerate.txt`,
but the Omni model class requires a transformers build that ships `Qwen2_5Omni*`.

## SFT (Qwen2.5-Omni-7B)

```bash
cd training/sft
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_sft.sh 4    # config_sft.yaml = omni
```
`config_sft.yaml` sets `model.arch: omni_thinker`, `modalities: "audio,videos"`,
`filter_by_modality: false` (train on the full modality mix). The parquet dataloader,
trainer and `merge_lora.py` are shared with `parquet_dataloader` and select the omni path
via `arch`.

Merge the LoRA'd thinker back into the full Omni model for RL:
```bash
python merge_lora.py --arch omni_thinker --base_model Qwen/Qwen2.5-Omni-7B \
    --adapter_path ./checkpoints/sft_qwen_omni_hba/step_XX \
    --output_path  ./checkpoints/sft_qwen_omni_hba_merged
```

## GRPO / TARPO (fork)

```bash
conda activate verl
cd training/rl
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_grpo.sh 4     # GRPO
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_tarpo.sh 4    # TARPO (fork-only)
```
Both auto-merge the latest SFT checkpoint if `MODEL_PATH` is missing, then run
`verl.trainer.main_ppo` with the fork's multimodal keys (`data.modalities`,
`train/val_modality_batching`) so the **HBA parquet is consumed directly** — no
conversion step. Rewards: `reward_function/human_behaviour.py` (GRPO) and
`human_behaviour_tarpo.py` (TARPO, task-aware); prompt template
`format_prompt/default.jinja`.

> **Why the fork?** TARPO (`adv_estimator=tarpo`) and the multimodal modality-batching
> keys are not in upstream verl (see the omni-RL discussion in `PARQUET_TRAINING.md`).
> For an upstream-verl, vision-only GRPO, use the `parquet_dataloader` branch instead.

## Status

The SFT path is the same code validated on `parquet_dataloader` (Qwen3-VL-4B SFT ran with
decreasing loss on an H20) with the omni model class selected by config. The GRPO/TARPO
recipes are the established fork recipes (they run on the `DDVD233/verl` env used for the
OmniSapiens results); run them on the fork env / cluster.
