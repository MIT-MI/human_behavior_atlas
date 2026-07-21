# Code Structure — Human Behavior Atlas (`parquet_dataloader` branch)

A map of the repository. For installation and the upstream-verl training path see
[`PARQUET_TRAINING.md`](PARQUET_TRAINING.md); for the high-level story see
[`../ReadMe.md`](../ReadMe.md).

> This is the **`parquet_dataloader`** branch — model-agnostic, **upstream**
> [`volcengine/verl @ main`](https://github.com/volcengine/verl) (the `verl/` submodule).
> For the omni / fork path (Qwen2.5-Omni + GRPO/TARPO on `DDVD233/verl`), see the
> [`main` branch](https://github.com/MIT-MI/human_behavior_atlas/blob/main/docs/OMNI_TRAINING.md).

---

## 1. Top-level layout

```
human_behavior_atlas/
├── verl/                  # RL/GRPO engine — submodule → volcengine/verl @ main (upstream)
├── training/
│   ├── sft/               # Supervised fine-tuning (Accelerate; no verl needed)
│   └── rl/                # GRPO data prep + reward + launch (upstream verl)
├── evaluation/
│   └── llm_grader/        # LLM-judge eval for free-form QA tasks
├── docs/
│   ├── CODE_STRUCTURE.md  # (this file)
│   └── PARQUET_TRAINING.md# upstream-verl SFT + GRPO walkthrough
└── ReadMe.md
```

---

## 2. The dataset

HBA parquet stores media as embedded **bytes** (`audios`/`videos`/`images` are
`list<binary>`); `modality_signature` marks which modalities a row has. This branch's SFT
loader decodes the bytes directly; its GRPO path first converts the parquet to a
**verl-native** RLHF parquet.

---

## 3. SFT — `training/sft/`

```bash
cd training/sft
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_sft.sh 1     # Qwen3-VL-4B
```
| Path | Role |
|------|------|
| `training/sft/dataset/sft_parquet_dataset.py` | Direct-from-parquet loader (pyarrow) with embedded-binary decode + modality filtering + seek-based video frame sampling. |
| `training/sft/trainer/sft_trainer.py` | Accelerate + FSDP teacher-forced LM trainer (response-only loss). |
| `training/sft/train_sft.py` | Model-agnostic entry point (`model.arch` picks the model class). |
| `training/sft/merge_lora.py` | Merge LoRA into the base model (for inference / GRPO init). |
| `training/sft/configs/config_sft*.yaml` | SFT configs (vision / audio). |

The classification / QA / BAM trainers (`train_classification.py`, `train_qa.py`,
`train_bam.py` + `models/`, `trainer/{cls,qa,bam}_trainer.py`) are also retained here.

---

## 4. GRPO — `training/rl/` (upstream verl)

```bash
cd training/rl
python prepare_grpo_data.py --data_dir $HBA_DATA_DIR --out_dir ./grpo_data --split train --modality video
MODEL_PATH=../sft/checkpoints/<merged> ./run_grpo.sh 1
```
| Path | Role |
|------|------|
| `training/rl/prepare_grpo_data.py` | Convert HBA parquet → **verl-native** RLHF parquet (prompt as chat messages, video → frame-path lists, `reward_model.ground_truth`). |
| `training/rl/reward_function.py` | verl `reward_manager=batch` reward (exact-match + format + cosine-sim). |
| `training/rl/run_grpo.sh` | `verl.trainer.main_ppo` GRPO launcher (consumes the verl-native parquet; fork-only keys removed). |

---

## 5. Evaluation — `evaluation/llm_grader/`

```bash
cd evaluation/llm_grader
bash run_llm_judge_eval.sh    # set prediction file + MIT_OPENAI_API_KEY / ANTHROPIC_API_KEY
```
