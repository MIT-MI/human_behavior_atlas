#!/bin/bash
# Unified SFT on Human Behavior Atlas parquet data (model-agnostic).
#
# Usage:
#   ./run_sft.sh [NUM_GPUS] [CONFIG] [extra train_sft.py args...]
#     NUM_GPUS : 1 (default), 2, or 4
#     CONFIG   : path to a config yaml (default: configs/config_sft.yaml)
#
# Env:
#   HBA_DATA_DIR : path to the human_behavior_atlas_v2 parquet directory
#   SAVE_DIR     : checkpoint output directory
#
# Examples:
#   HBA_DATA_DIR=/data/human_behavior_atlas_v2 ./run_sft.sh 1
#   HBA_DATA_DIR=/data/human_behavior_atlas_v2 ./run_sft.sh 4 configs/config_sft_gemma4_audio.yaml
set -x
cd "$(dirname "$0")"

NUM_GPUS=${1:-1}
CONFIG=${2:-configs/config_sft.yaml}

case "$NUM_GPUS" in
  1) ACCEL=configs/accelerate_config_sft_1gpu.yaml; export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ;;
  2) ACCEL=configs/accelerate_config_sft_2gpu.yaml ;;
  4) ACCEL=configs/accelerate_config_sft_4gpu.yaml ;;
  *) echo "Supported GPU counts: 1, 2, or 4"; exit 1 ;;
esac

DATA_DIR=${HBA_DATA_DIR:-/path/to/human_behavior_atlas_v2}
SAVE_DIR=${SAVE_DIR:-./checkpoints/sft_hba}

PYTHONUNBUFFERED=1 accelerate launch \
    --config_file "$ACCEL" \
    train_sft.py \
    --config "$CONFIG" \
    --train_files "$DATA_DIR" \
    --val_files "$DATA_DIR" \
    --save_checkpoint_dir "$SAVE_DIR" \
    "${@:3}"

echo "SFT training finished."
