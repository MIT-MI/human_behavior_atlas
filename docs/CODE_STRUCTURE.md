# Code Structure — Human Behavior Atlas / OmniSapiens

A map of the repository so you can find things quickly. For installation and the
high-level story see [`../ReadMe.md`](../ReadMe.md); for the two training paths see
[`OMNI_TRAINING.md`](OMNI_TRAINING.md) and [`PARQUET_TRAINING.md`](PARQUET_TRAINING.md).

---

## 1. Top-level layout

```
human_behavior_atlas/
├── verl/                  # RL/GRPO engine (git submodule). Version depends on branch:
│                          #   main  -> DDVD233/verl  (omni fork)
│                          #   parquet_dataloader -> volcengine/verl @ main (upstream)
├── training/
│   ├── sft/               # Supervised fine-tuning (Accelerate; no verl needed)
│   └── rl/                # GRPO / TARPO launch scripts + reward functions
├── evaluation/
│   └── llm_grader/        # LLM-judge eval for free-form QA tasks
├── docs/
│   ├── CODE_STRUCTURE.md  # (this file)
│   ├── OMNI_TRAINING.md   # omni / fork-verl training path (main branch)
│   └── PARQUET_TRAINING.md# upstream-verl training path (parquet_dataloader branch)
└── ReadMe.md
```

### Branches
| Branch | `verl/` submodule | Focus |
|--------|-------------------|-------|
| `main` | `DDVD233/verl` (omni fork) | Qwen2.5-Omni-7B SFT + **GRPO + TARPO**, reads HBA parquet directly |
| `parquet_dataloader` | `volcengine/verl @ main` (upstream) | model-agnostic vision SFT + GRPO (verl-native parquet) |

Both branches share the `training/{sft,rl}` + `evaluation/` + `docs/` layout; only the
`verl/` submodule pointer and a few configs differ.

---

## 2. The dataset and its two media formats

HBA ships in two forms (see `ReadMe.md` → *Benchmark*):

| Form | Media storage | Loader |
|------|---------------|--------|
| **Parquet** (recommended) | media embedded as raw **bytes** — `audios`/`videos`/`images`/`ext_*_feats` are `list<binary>`. Self-contained. | binary / lazy loaders (§3, §4) |
| **Tar / JSONL** | JSONL rows hold **relative paths**; media in sibling folders. | string-path loaders |

`modality_signature` (`text_audio`, `text_video_audio`, …) drives **modality batching**:
each batch is homogeneous in modality so the multimodal processor stays consistent.

---

## 3. RL / GRPO — `training/rl/` + `verl/`

```bash
cd training/rl
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_grpo.sh 4     # GRPO
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_tarpo.sh 4    # TARPO (fork-only)
```

| Path | Role |
|------|------|
| `training/rl/run_grpo.sh` / `run_tarpo.sh` | Launchers. Resolve repo root, set `PYTHONPATH` to the `verl/` submodule, build the parquet shard list, auto-merge an SFT checkpoint if needed, run `verl.trainer.main_ppo`. |
| `training/rl/reward_function/human_behaviour.py` | GRPO reward (`human_behaviour_compute_score_batch`). |
| `training/rl/reward_function/human_behaviour_tarpo.py` | TARPO (task-aware) reward. |
| `training/rl/format_prompt/default.jinja` | Prompt wrapper. |
| `verl/verl/utils/dataset/rl_dataset.py` | **The RL data loader** (`RLHFDataset`) — reads parquet/jsonl, builds messages, tokenizes, emits multimodal tensors. |
| `verl/verl/trainer/main_ppo.py` | Trainer entrypoint; builds dataset + modality sampler (`create_rl_sampler`). |
| `verl/verl/utils/dataset/{audio,vision}_utils.py` | Media decode (path **or bytes**). |
| `verl/verl/workers/fsdp_workers.py` | Builds the FSDP actor/ref; selects the Qwen2.5-Omni *thinker*. |

### Custom binary-parquet support added to the fork verl

Stock verl only loads parquet whose media columns are **string paths**. The fork adds
first-class support for the **self-contained binary parquet**:

- `RLHFDataset._detect_binary_media()` → switches to lazy mode when media columns are `binary`.
- `RLHFDataset._read_files_lazy()` / `_lazy_get_row()` → reads only light text columns up
  front; decodes heavy media bytes on demand (LRU-cached per file).
- `audio_utils.process_audio` / `vision_utils.process_video` → raw-`bytes` decode branches
  (scipy/`wave`; PyAV).
- `main_ppo.create_rl_sampler()` → lazy fast-path that builds the modality index by reading
  only `modality_signature` via pyarrow (no per-row media decode).
- `fsdp_workers._is_qwen_omni()` → detects omni by `config.json` model_type (not just path),
  so SFT-merged checkpoints load the thinker.

> Without these, the binary parquet fails with `pyarrow ArrowInvalid: Invalid UTF8 payload`.
>
> **Where they live:** the `verl/` submodule is pinned to `DDVD233/verl @ hba_public_release`,
> which does **not** yet include these changes. They are carried in this repo as
> `patches/0001-hba-binary-parquet-loader.patch` and applied on top of the pinned submodule
> by `scripts/setup_verl.sh` (run once after `git submodule update --init`). No write access
> to the verl fork is required.

### Data flow (one GRPO step)
```
parquet shard(s)
  └─ RLHFDataset (lazy binary decode) → {input_ids, multi_modal_inputs, modality_signature}
       └─ ModalitySignatureBatchSampler → modality-homogeneous batch
            └─ vLLM rollout (n samples) → responses
                 └─ human_behaviour_compute_score_batch → rewards
                      └─ GRPO advantage → FSDP actor update
```

---

## 4. SFT — `training/sft/`

Two complementary SFT entry points live here:

**(a) Parquet-native generative SFT** (model-agnostic; used by both branches):
```bash
cd training/sft
HBA_DATA_DIR=/path/to/human_behavior_atlas_v2 ./run_sft.sh 4
```
| Path | Role |
|------|------|
| `training/sft/dataset/sft_parquet_dataset.py` | Direct-from-parquet loader with embedded-binary decode (`_decode_audio`/`_decode_video`/`_decode_image`) + modality filtering. |
| `training/sft/trainer/sft_trainer.py` | Accelerate + FSDP teacher-forced LM trainer. |
| `training/sft/train_sft.py` | Entry point; model class chosen by `model.arch`. |
| `training/sft/merge_lora.py` | Merge LoRA into the base/omni model (for GRPO init). |
| `training/sft/configs/config_sft*.yaml` | SFT configs (vision / omni / audio). |

**(b) Three-stage classifier-head SFT** (the OmniSapiens classifier pipeline):
```bash
cd training/sft
bash run_classification.sh   # Stage 1: LoRA + per-task classification heads
bash run_qa.sh               # Stage 2: train lm_head only (backbone frozen)
bash run_bam.sh              # Stage 3: per-dataset Behavioral Adapter Modules
```
| Path | Role |
|------|------|
| `training/sft/dataset/{base_dataset,sft_dataset}.py` | JSONL/parquet dataset + `OmniClassifierDataset`. |
| `training/sft/models/qwen2_5_omni_classifier_heads_decoder.py` | Qwen2.5-Omni + per-task heads. |
| `training/sft/models/bam_*.py` | BAM adapter + wrapped model. |
| `training/sft/trainer/{base,cls,qa,bam}_trainer.py` | Per-stage training loops. |
| `training/sft/label_maps/unified_label_map.json` | Canonical label sets. |

---

## 5. Evaluation — `evaluation/llm_grader/`

Classification metrics log to W&B during training. Free-form QA (IntentQA, MIMEQA, SIQ2)
is graded after training by an LLM judge:
```bash
cd evaluation/llm_grader
bash run_llm_judge_eval.sh    # set prediction file + MIT_OPENAI_API_KEY / ANTHROPIC_API_KEY
```

---

## 6. Environment

- One conda env serves both pipelines (`python==3.12`, verl deps, `vllm[audio]`,
  flash-attention from source, `torchcodec`, `av`/PyAV, `scipy`). See `ReadMe.md`.
- `verl` must be importable: `export PYTHONPATH="/abs/path/to/human_behavior_atlas/verl:$PYTHONPATH"`.
  `training/rl/run_grpo.sh` sets this automatically from its own location.
- GRPO inits from an SFT-merged Qwen2.5-Omni checkpoint (recommended) or base
  `Qwen/Qwen2.5-Omni-7B`.
