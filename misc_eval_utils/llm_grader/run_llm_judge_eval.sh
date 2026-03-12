#!/bin/bash

# LLM Judge Evaluation Script
# This script runs the LLM judge evaluation with configurable parameters.
# Works for both verl (RL training) and sft training pipelines.


# Default values
# PLEASE PUT THE PREDICTIONS_TO_EVAL_PATH as the path within the save dir that points to the generated outputs for all questions, which should look like "full_test_or_val_generation_outputs/150.json"
PREDICTIONS_TO_EVAL_PATH="/path/to/your/results/test_preds.json"
GRADING_RESULTS_PATH="/path/to/your/results/test_preds_llm_grading_results.json"
GROUND_TRUTH_ANNOTATIONS_ROOT_DIR="/path/to/your/data/annotations"
PROVIDER="openai"
WANDB_PROJECT="llm_judge_eval"
WANDB_RUN_NAME="test_try_experiment_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --predictions_to_eval_path)
            PREDICTIONS_TO_EVAL_PATH="$2"
            shift 2
            ;;
        --grading_results_path)
            GRADING_RESULTS_PATH="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --ground_truth_annotations_root_dir)
            GROUND_TRUTH_ANNOTATIONS_ROOT_DIR="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_PROJECT=""
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --predictions_to_eval_path PATH       Path to the results JSON file"
            echo "  --grading_results_path PATH          Path to save graded results as JSON"
            echo "  --provider PROVIDER       LLM provider (openai or anthropic)"
            echo "  --wandb_project PROJECT   W&B project name"
            echo "  --wandb_run_name NAME     W&B run/experiment name"
            echo "  --ground_truth_annotations_root_dir PATH  Path to the annotation data root directory
  --no-wandb                Disable W&B logging"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --predictions_to_eval_path input.json --wandb_run_name my_experiment"
            echo "  $0 --provider anthropic --no-wandb"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required environment variables are set
if [ "$PROVIDER" = "openai" ] && [ -z "$MIT_OPENAI_API_KEY" ]; then
    echo "Error: MIT_OPENAI_API_KEY environment variable is not set"
    echo "Please export your OpenAI API key:"
    echo "  export MIT_OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

if [ "$PROVIDER" = "anthropic" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable is not set"
    echo "Please export your Anthropic API key:"
    echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if results file exists
if [ ! -f "$PREDICTIONS_TO_EVAL_PATH" ]; then
    echo "Error: Results file not found: $PREDICTIONS_TO_EVAL_PATH"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "LLM Judge Evaluation Configuration"
echo "=========================================="
echo "Results path:    $PREDICTIONS_TO_EVAL_PATH"
echo "Save path:       $GRADING_RESULTS_PATH"
echo "Annotations dir: $GROUND_TRUTH_ANNOTATIONS_ROOT_DIR"
echo "Provider:        $PROVIDER"
if [ -n "$WANDB_PROJECT" ]; then
    echo "W&B project:     $WANDB_PROJECT"
    echo "W&B run name:    $WANDB_RUN_NAME"
else
    echo "W&B:             Disabled"
fi
echo "=========================================="
echo ""

# Build the command
CMD="python llm_judge_eval.py --predictions_to_eval_path \"$PREDICTIONS_TO_EVAL_PATH\" --grading_results_path \"$GRADING_RESULTS_PATH\" --provider $PROVIDER --ground_truth_annotations_root_dir \"$GROUND_TRUTH_ANNOTATIONS_ROOT_DIR\""

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\""
fi

# Run the evaluation
echo "Starting evaluation..."
echo ""
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code $?"
    echo "=========================================="
    exit 1
fi
