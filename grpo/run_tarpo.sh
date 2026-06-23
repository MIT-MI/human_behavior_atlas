#!/bin/bash
# TARPO for Qwen2.5-Omni-7B on Human Behavior Atlas — FORK path (DDVD233/verl).
#
# TARPO is a fork-only advantage estimator that adds task-level adaptation over GRPO:
#   task-level normalization + mixture adapters, CVaR tail boosting for imbalanced
#   tasks, class-based weighting, and per-task EMA buffers. It is NOT available in
#   upstream verl — this recipe requires the fork (the verl/ submodule on this branch).
#
# Usage:   ./run_tarpo.sh [NUM_GPUS]     (default 4)
# Env:     HBA_DATA_DIR, MODEL_PATH, SAVE_DIR
set -x

NUM_GPUS=${1:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS-1)))}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_USE_V1=0
export RAY_memory_usage_threshold=0.98
export RAY_NUM_CPUS_PER_TASK=4

GRPO_DIR="$(cd "$(dirname "$0")" && pwd)"
HBA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${HBA_ROOT}/verl:${PYTHONPATH}"
cd "$HBA_ROOT"

MODEL_PATH=${MODEL_PATH:-"${HBA_ROOT}/sft/checkpoints/sft_qwen_omni_hba_merged"}

if [ ! -f "$MODEL_PATH/config.json" ]; then
    SFT_CKPT_DIR="${HBA_ROOT}/sft/checkpoints/sft_qwen_omni_hba"
    LATEST_CKPT=$(ls -d ${SFT_CKPT_DIR}/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1 | xargs -I{} echo "${SFT_CKPT_DIR}/step_{}")
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: No SFT checkpoint in ${SFT_CKPT_DIR}. Run sft/run_sft.sh first."
        exit 1
    fi
    echo "[INFO] Auto-merging SFT checkpoint: $LATEST_CKPT -> $MODEL_PATH"
    python3 "${HBA_ROOT}/sft/merge_lora.py" --arch omni_thinker \
        --base_model "Qwen/Qwen2.5-Omni-7B" \
        --adapter_path "$LATEST_CKPT" --output_path "$MODEL_PATH" || { echo "merge failed"; exit 1; }
fi

DATA_DIR=${HBA_DATA_DIR:-"/path/to/human_behavior_atlas_v2"}
TRAIN_FILE_LIST="["
for f in ${DATA_DIR}/train-{00032..00292}-of-00293.parquet; do
    [ -f "$f" ] && TRAIN_FILE_LIST="${TRAIN_FILE_LIST}${f},"
done
TRAIN_FILE_LIST="${TRAIN_FILE_LIST%,}]"
VAL_FILE="${DATA_DIR}/validation-00004-of-00013.parquet"

SAVE_DIR=${SAVE_DIR:-"${HBA_ROOT}/grpo/checkpoints/tarpo_qwen_omni_hba"}
REWARD_FN="${GRPO_DIR}/reward_function/human_behaviour_tarpo.py"
FORMAT_PROMPT="${GRPO_DIR}/format_prompt/default.jinja"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=tarpo \
    data.train_files="$TRAIN_FILE_LIST" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.image_key=images \
    data.video_key=videos \
    data.prompt_key=problem \
    data.dataloader_num_workers=0 \
    data.modalities=\'audio,videos\' \
    data.train_modality_batching.enabled=True \
    data.train_modality_batching.drop_last=True \
    data.val_modality_batching.enabled=True \
    data.val_modality_batching.drop_last=False \
    data.format_prompt="$FORMAT_PROMPT" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$NUM_GPUS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=human_behaviour_compute_score_batch \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tarpo-qwen-omni-hba' \
    trainer.experiment_name='tarpo_sft_init' \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.val_before_train=False \
    trainer.test_freq=99999 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.advantage_save_dir="$SAVE_DIR/advantages" \
    trainer.advantage_plot_freq=15 \
    trainer.resume_mode=auto

echo "TARPO training completed!"
