# Setup Guide (omni / fork path — `main`)

A linear quickstart for training **Qwen2.5-Omni-7B** on the Human Behavior Atlas
**parquet** data (SFT + GRPO/HARPO on the `DDVD233/verl` fork). For the upstream-verl
vision path, see [`PARQUET_TRAINING.md`](PARQUET_TRAINING.md); for a code map see
[`CODE_STRUCTURE.md`](CODE_STRUCTURE.md).

---

## 1. Clone + initialize the verl submodule

```bash
git clone --recurse-submodules https://github.com/MIT-MI/human_behavior_atlas.git
cd human_behavior_atlas
# (if you cloned without --recurse-submodules: git submodule update --init --recursive)
```

## 2. Patch verl  ⟵ the important step

The `verl/` submodule is pinned to `DDVD233/verl @ hba_public_release`, which does **not**
include two HBA additions. They ship in this repo as patches and are applied on top of the
pinned submodule — **no write access to the verl fork is needed**:

| Patch | Adds |
|-------|------|
| `patches/0001-hba-binary-parquet-loader.patch` | Lazy loader for **embedded-binary parquet** (decode audio/video/image bytes), the modality-batching sampler fast-path, and Qwen2.5-Omni model detection. |
| `patches/0002-hba-harpo-advantage.patch` | The **HARPO** advantage estimator (`adv_estimator=harpo`). |

Apply both with one idempotent command:

```bash
bash scripts/setup_verl.sh
```

This inits the submodule (if needed) and applies every `patches/*.patch` in order. It is
**safe to re-run** — already-applied patches are detected and skipped:

```
[setup_verl] 0001-hba-binary-parquet-loader.patch: applied.
[setup_verl] 0002-hba-harpo-advantage.patch: applied.
[setup_verl] Done. verl is ready for HBA binary-parquet training (GRPO + HARPO).
```

> **Without this step**, training on the parquet fails at load with
> `pyarrow.lib.ArrowInvalid: Invalid UTF8 payload`, and `adv_estimator=harpo` is a no-op.

<details>
<summary>Manual apply / troubleshooting</summary>

```bash
# apply by hand:
git -C verl apply patches/0001-hba-binary-parquet-loader.patch
git -C verl apply patches/0002-hba-harpo-advantage.patch

# check whether already applied (exit 0 = applied):
git -C verl apply --reverse --check patches/0001-hba-binary-parquet-loader.patch

# revert:
git -C verl apply --reverse patches/000*.patch
```

If a patch fails to apply, the pinned submodule commit has likely moved — regenerate the
patch from your working tree with `git -C verl diff > patches/...`.
</details>

## 3. Environment

```bash
conda create -n verl python==3.12.2 && conda activate verl

# PyTorch (pick your CUDA) — see https://pytorch.org
pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

pip install -r verl/requirements.txt
pip install -r training/rl/requirements_grpo.txt
pip install "vllm[audio]==0.10.2" && pip install -U "setuptools>=78,<80"

# FlashAttention: build from source (https://github.com/Dao-AILab/flash-attention) — do NOT pip install flash-attn
# Then the media/runtime extras:
pip install torchcodec==0.4.0 --index-url https://download.pytorch.org/whl/cu126
conda install -c conda-forge "ffmpeg=6.*" "libgcc-ng>=13" "libstdcxx-ng>=13"
pip install ujson scikit-learn qwen-vl-utils av scipy

# Make verl importable (the run scripts also set this themselves):
export PYTHONPATH="$(pwd)/verl:$PYTHONPATH"
```

> SFT alone (no verl) can use the lighter env in `training/sft/requirements_accelerate.txt`.
> For the LLM-judge QA eval add: `pip install pandas tqdm tenacity anthropic openai wandb`.

## 4. Data

Download the parquet distribution and point `HBA_DATA_DIR` at it (each sample embeds its
media as bytes; columns `audios`/`videos`/`images` are `list<binary>`):

```bash
export HBA_DATA_DIR=/path/to/human_behavior_atlas_v2   # train-*.parquet, validation-*.parquet, test-*.parquet
```

## 5. Train

All launchers train on the **full** split and auto-resolve paths from their own location.
The first positional arg is the GPU count.

```bash
# --- SFT (Qwen2.5-Omni-7B, LoRA) ---
cd training/sft
SAVE_DIR=./checkpoints/sft_qwen_omni_hba HBA_DATA_DIR=$HBA_DATA_DIR ./run_sft.sh 4

# merge the LoRA'd thinker into the full Omni model for RL init:
python merge_lora.py --arch omni_thinker --base_model Qwen/Qwen2.5-Omni-7B \
    --adapter_path ./checkpoints/sft_qwen_omni_hba/step_XX \
    --output_path  ./checkpoints/sft_qwen_omni_hba_merged

# --- RL (one launcher per algorithm) ---
cd ../rl
MODEL_PATH=../sft/checkpoints/sft_qwen_omni_hba_merged HBA_DATA_DIR=$HBA_DATA_DIR ./run_grpo.sh 4    # GRPO
MODEL_PATH=../sft/checkpoints/sft_qwen_omni_hba_merged HBA_DATA_DIR=$HBA_DATA_DIR ./run_harpo.sh 4   # HARPO
```

Useful env overrides: `MODEL_PATH` (RL init checkpoint), `SAVE_DIR`, `HBA_DATA_DIR`.
If `MODEL_PATH` is unset, the RL scripts auto-merge the latest checkpoint found under
`training/sft/checkpoints/sft_qwen_omni_hba`. RL scripts also accept extra hydra overrides
appended on the CLI (e.g. `./run_grpo.sh 4 actor_rollout_ref.rollout.n=8`).

The 3-stage classifier-head SFT (`run_classification.sh` → `run_qa.sh` → `run_bam.sh`) and
the LLM-judge eval (`evaluation/llm_grader/run_llm_judge_eval.sh`) are documented in the
main [`../ReadMe.md`](../ReadMe.md).

---

## Quick sanity check

```bash
# 1) patch applied + harpo registered:
PYTHONPATH=verl python -c "from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY as R; print('harpo' in R)"
# 2) binary-parquet loader importable:
PYTHONPATH=verl python -c "from verl.utils.dataset.rl_dataset import RLHFDataset; print('ok')"
```
