#!/bin/bash
# GRPO on Human Behavior Atlas (video subset) with UPSTREAM verl (volcengine/verl @ main).
#
# Differences from the original mirl recipe:
#   * NO fork-only keys: data.modalities / data.train_modality_batching /
#     data.val_modality_batching / data.format_prompt are removed. The format
#     instruction is baked into the prompt by prepare_grpo_data.py, and the
#     parquet is verl-native (prompt = chat messages, videos = frame-path lists).
#   * Runs `python -m verl.trainer.main_ppo` from an env with upstream verl + vLLM>=0.11.
#
# Prerequisites:
#   1. conda activate hba_verl   (upstream verl@main + vLLM>=0.11 + transformers>=4.57)
#   2. Build the data once:
#        python prepare_grpo_data.py --data_dir $HBA_DATA_DIR --out_dir ./grpo_data \
#            --split train      --modality video
#        python prepare_grpo_data.py --data_dir $HBA_DATA_DIR --out_dir ./grpo_data \
#            --split validation --modality video
#
# Usage:
#   ./run_grpo.sh [NUM_GPUS]
# Env:
#   MODEL_PATH : actor init (default: Qwen/Qwen3-VL-4B-Instruct; normally the merged SFT ckpt)
#   DATA_DIR   : dir holding train.parquet / validation.parquet (default: ./grpo_data)
set -x

NUM_GPUS=${1:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS-1)))}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=${VLLM_USE_V1:-1}

cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
DATA_DIR=${DATA_DIR:-"./grpo_data"}
REWARD_FN="$(pwd)/reward_function.py"
SAVE_DIR=${SAVE_DIR:-"./checkpoints/grpo_qwen3vl_hba"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/validation.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.prompt_key=prompt \
    data.image_key=images \
    data.video_key=videos \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$NUM_GPUS" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score_batch \
    reward_model.reward_manager=batch \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='grpo-qwen3vl-hba' \
    trainer.experiment_name='grpo_hba_video' \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$SAVE_DIR"

echo "GRPO training finished."
