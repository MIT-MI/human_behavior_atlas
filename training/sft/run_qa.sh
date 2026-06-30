#!/bin/bash
# QA stage training: loads a frozen multi-head checkpoint, trains only the lm_head.
# Edit the paths below before running.

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TRAIN_FILE="/path/to/your/data/train.jsonl"
VAL_FILE="/path/to/your/data/val.jsonl"
TEST_FILE="/path/to/your/data/test.jsonl"
LABEL_MAP="/path/to/your/project/sft/label_maps/unified_label_map.json"
LOAD_CHECKPOINT="/path/to/your/checkpoint/dir/step_20"
SAVE_DIR="/path/to/your/checkpoint/dir"
VAL_DIR="$SAVE_DIR/test_results"
MODE="train"
PROJECT_NAME="human_behavior_atlas_sft_experiments"

ACTION=$([ "$MODE" = "test" ] && echo "testing" || echo "training")
echo "Launching QA lm_head ${ACTION}..."
accelerate launch --config_file configs/accelerate_config_qwen.yaml train_qa.py \
    --mode "$MODE" \
    --training_strategy lora \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --test_batch_size 1 \
    --lr 1e-4 \
    --epochs 5 \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --test_file "$TEST_FILE" \
    --label_map_path "$LABEL_MAP" \
    --load_checkpoint_path "$LOAD_CHECKPOINT" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --save_every_n_epochs 1 \
    --save_every_n_steps 10 \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 999999 \
    --gradient_accumulation_steps 8 \
    --use_scheduler \
    --scheduler_type cosine \
    --warmup_steps 50 \
    --format_prompt "" \
    --max_prompt_length 4096 \
    --qa_datasets intentqa mimeqa siq2 \
    --qa_loss_weight 1.0 \
    --project "${PROJECT_NAME}"

echo "QA ${ACTION} completed!"
