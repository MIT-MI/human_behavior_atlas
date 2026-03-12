#!/bin/bash
# BAM adapter training: loads a frozen multi-head checkpoint and trains
# per-dataset video/audio residual hidden adapters.
#
# One model per dataset, one invocation per dataset.
# Task type (cls/qa) is inferred automatically from the 'task' field in each dataset's JSONL.
# Edit the paths below before running.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

TRAIN_FILE="/path/to/your/data/train.jsonl"
VAL_FILE="/path/to/your/data/val.jsonl"
TEST_FILE="/path/to/your/data/test.jsonl"
LABEL_MAP="/path/to/your/project/sft/label_maps/unified_label_map.json"
LOAD_CHECKPOINT="/path/to/your/checkpoint/dir/step_10"
BASE_SAVE_DIR="/path/to/your/checkpoint/dir"
PROJECT_NAME="bam_adapter_experiments"
MODE="test"
TMP_DIR="/path/to/your/data"

# Datasets to train adapters for (one run per dataset)
# BAM will be saved in separate subdirs under BASE_SAVE_DIR, e.g. $BASE_SAVE_DIR/bam_ptsd_in_the_wild, $BASE_SAVE_DIR/bam_another_dataset, etc.
INCLUDE_DATASETS=("ptsd_in_the_wild")

# ---- helpers ----
in_list() {
  local needle="$1"; shift || true
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

filter_jsonl() {
  local in_jsonl="$1" dataset="$2" out_jsonl="$3"
  python3 - "$in_jsonl" "$dataset" "$out_jsonl" <<'PY'
import sys, json
inp, ds, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(inp, 'r') as f, open(outp, 'w') as g:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("dataset") == ds:
            g.write(json.dumps(obj) + "\n")
PY
}

list_datasets() {
  python3 - "$1" <<'PY'
import sys, json
seen = set()
with open(sys.argv[1], 'r') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception: continue
        ds = obj.get("dataset")
        if isinstance(ds, str) and ds not in seen:
            seen.add(ds); print(ds)
PY
}

# Infer task type ('cls' or 'qa') from the first row's 'task' field suffix in a JSONL.
# e.g. task="emotion_cls" → "cls", task="intent_qa" → "qa"
get_task_type() {
  python3 - "$1" <<'PY'
import sys, json
with open(sys.argv[1], 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception: continue
        task = obj.get("task", "")
        print(task.rsplit("_", 1)[-1] if "_" in task else "cls")
        break
PY
}

# Returns 1 if any row in JSONL has a non-empty list for KEY, else 0.
has_descriptor() {
  python3 - "$1" "$2" <<'PY'
import sys, json
path, key = sys.argv[1], sys.argv[2]
with open(path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception: continue
        val = obj.get(key)
        if isinstance(val, list) and len(val) > 0:
            print(1); sys.exit(0)
print(0)
PY
}

echo "Collecting dataset names from JSONL files..."
mapfile -t ALL_DS_ARR < <(
  { list_datasets "$TRAIN_FILE"; list_datasets "$VAL_FILE"; } | sort -u
)

if [[ ${#INCLUDE_DATASETS[@]} -eq 1 && "${INCLUDE_DATASETS[0]}" == "all" ]]; then
  PROCESS_DS=("${ALL_DS_ARR[@]}")
else
  PROCESS_DS=()
  for DS in "${INCLUDE_DATASETS[@]}"; do
    if in_list "$DS" "${ALL_DS_ARR[@]}"; then
      PROCESS_DS+=("$DS")
    else
      echo "Warning: '$DS' not found in JSONLs; skipping."
    fi
  done
fi

if ((${#PROCESS_DS[@]} == 0)); then
  echo "No datasets to process. Exiting."
  exit 0
fi

ACTION=$([ "$MODE" = "test" ] && echo "testing" || echo "training")

for DS in "${PROCESS_DS[@]}"; do
  echo "-------------------------------------------"
  echo "Running BAM adapter ${ACTION} for: $DS"

  TRAIN_OUT="$TMP_DIR/bam_train_${DS}.jsonl"
  VAL_OUT="$TMP_DIR/bam_val_${DS}.jsonl"
  TEST_OUT="$TMP_DIR/bam_test_${DS}.jsonl"
  filter_jsonl "$TRAIN_FILE" "$DS" "$TRAIN_OUT"
  filter_jsonl "$VAL_FILE"   "$DS" "$VAL_OUT"
  filter_jsonl "$TEST_FILE"  "$DS" "$TEST_OUT"

  TRAIN_LINES=$(wc -l < "$TRAIN_OUT" || echo 0)
  VAL_LINES=$(wc -l < "$VAL_OUT" || echo 0)
  if [[ "$TRAIN_LINES" -eq 0 || "$VAL_LINES" -eq 0 ]]; then
    echo "Skipping $DS (train=$TRAIN_LINES, val=$VAL_LINES)."
    continue
  fi

  SAVE_DIR="${BASE_SAVE_DIR}/bam_${DS}"
  VAL_DIR="${SAVE_DIR}/validation_results"
  mkdir -p "$SAVE_DIR" "$VAL_DIR"

  # Infer task type from the filtered JSONL's task field
  TASK_TYPE=$(get_task_type "$TRAIN_OUT")
  HAS_VIDEO=$(has_descriptor "$TRAIN_OUT" "ext_video_feats")
  HAS_AUDIO=$(has_descriptor "$TRAIN_OUT" "ext_audio_feats")
  BAM_VIDEO_FLAG=$( [[ "$HAS_VIDEO" == "1" ]] && echo "--use_bam_video" || echo "" )
  BAM_AUDIO_FLAG=$( [[ "$HAS_AUDIO" == "1" ]] && echo "--use_bam_audio" || echo "" )
  echo "  train_file: $TRAIN_OUT  ($TRAIN_LINES lines)"
  echo "  val_file:   $VAL_OUT    ($VAL_LINES lines)"
  echo "  test_file:  $TEST_OUT"
  echo "  save_dir:   $SAVE_DIR"
  echo "  task_type:  $TASK_TYPE"
  echo "  has_video_feats: $HAS_VIDEO  has_audio_feats: $HAS_AUDIO"

  accelerate launch --config_file configs/accelerate_config_qwen.yaml train_bam.py \
    --mode "$MODE" \
    --task_type "$TASK_TYPE" \
    --dataset_name "$DS" \
    --bam_fresh_start \
    --training_strategy lora \
    --train_batch_size 2 \
    --val_batch_size 2 \
    --test_batch_size 2 \
    --lr 1e-4 \
    --base_lr 1e-4 \
    --bam_lr 5e-4 \
    --epochs 3 \
    --train_file "$TRAIN_OUT" \
    --val_file "$VAL_OUT" \
    --test_file "$TEST_OUT" \
    --label_map_path "$LABEL_MAP" \
    --load_checkpoint_path "$LOAD_CHECKPOINT" \
    --save_checkpoint_dir "$SAVE_DIR" \
    --validation_result_dir "$VAL_DIR" \
    --save_every_n_epochs None \
    --save_every_n_steps 10 \
    --validate_every_n_epochs 1 \
    --validate_every_n_steps None \
    --early_stopping_patience 99999 \
    --gradient_accumulation_steps 4 \
    --bam_stage bam_and_classifier_heads_only \
    ${BAM_VIDEO_FLAG:+$BAM_VIDEO_FLAG} \
    ${BAM_AUDIO_FLAG:+$BAM_AUDIO_FLAG} \
    --d_video_feat 3318 \
    --d_audio_feat 6373 \
    --bam_hidden_video 256 \
    --bam_hidden_audio 256 \
    --bam_p_moddrop_video 0.20 \
    --bam_p_moddrop_audio 0.20 \
    --bam_video_temporal meanstd \
    --bam_video_norm none \
    --bam_audio_norm l2 \
    --bam_audio_temporal none \
    --bam_video_alpha_init 4.0 \
    --bam_audio_alpha_init 4.0 \
    --bam_video_use_ln \
    --bam_audio_use_ln \
    --format_prompt "" \
    --max_prompt_length 4096 \
    --project "${PROJECT_NAME}"

  echo "Finished ${ACTION}: $DS"
done

echo "All BAM adapter ${ACTION} runs completed."
