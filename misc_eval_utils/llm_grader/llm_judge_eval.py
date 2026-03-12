import os
import sys
import json
import argparse
import asyncio
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from dataclasses import dataclass
from typing import Dict, Any, List
import anthropic
from openai import AsyncOpenAI
import wandb


# Grading instructions
MIMEQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answers regarding understanding of a mime performance.
You will be shown a "gold-standard" answer from a human annotator, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to determine whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate does not contain misleading information and does not hallucinate story plots not present in the reference answer.
3. Since the videos are mime performances, invisible actions, objects, or the mime actor portraying objects should be considered correct if and only if they are relevant to the question.
4. The candidate answer can be a good answer in place of the reference answer as long as they are in the same ballpark. However, the candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

SIQ_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answer regarding understanding of a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate captures the core meaning of the reference answer using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is semantically equivalent as the reference and does not add a claim that conflicts with it. 
3. The candidate should not assert a conflicting explanation or introduce factually incompatible details. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

INTENTQA_GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the question and answers about the intent behind actions in a video.
You will be shown a "gold-standard" answer from human annotators, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to judge whether the candidate gives a plausible interpretation of the intent that does not contradict the reference, using the following criteria:

1. The candidate must state at least one coherent, primary answer.
2. The candidate's explanation is in the same ballpark as the reference and does not add a claim that conflicts with it. The wording need not be the same; minor additions are allowed if they are consistent with the reference and the question.
3. The candidate should not assert a conflicting explanation, introduce factually incompatible details, or miss the core intent. The candidate must not refer to a different subject or object not supported by the question/reference. If the candidate's answer centers on a different primary subject/object than the reference, it is incorrect.

Evaluate only the first clause that directly answers the question; ignore preambles and later asides.
Output: Respond with exactly one JSON object: {"correct": true/false, "explanation": "…"}
"""

GRADE_PROMPT = """
Question:
"{question}"
Candidate Answer:
"{candidate_answer}"
Reference Answer:
"{ref_answer}"

Please evaluate the candidate answer based on the dataset-specific instructions.

Respond with exactly this format - a JSON object with two fields:
- "correct": true or false (boolean)
- "explanation": a very short, few phrases explanation of your decision (string)
Only respond with the JSON object, no other text or comments.
"""

@dataclass
class Evaluation:
    correct: bool
    explanation: str
    
    def to_dict(self):
        return {"correct": self.correct, "explanation": self.explanation}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(correct=data["correct"], explanation=data["explanation"])


class JSONParseError(Exception):
    def __init__(self, content: str):
        self.content = content
        super().__init__("JSON parse error")


class LLM:
    def __init__(self, llm_str: str, default_instructions: str | None = None, provider: str = "anthropic"):
        self.llm_str = llm_str
        self.instructions = default_instructions
        self.provider = provider
        
        if provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "openai":
            self.client = AsyncOpenAI(
                api_key=os.getenv("MIT_OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'anthropic', 'openai'")
    
    @retry(reraise=True, wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def create_completion(self, messages: List[Dict[str, str]]) -> Evaluation:
        if self.provider == "anthropic":
            # Convert messages to Claude format
            claude_messages = []
            system_message = self.instructions
            
            for msg in messages:
                if msg["role"] == "user":
                    claude_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=self.llm_str,
                messages=claude_messages,
                system=system_message,
                max_tokens=2048
            )
            
            original_content = response.content[0].text.strip()
            
        elif self.provider == "openai":
            # Convert messages to OpenAI format
            openai_messages = []
            if self.instructions:
                openai_messages.append({
                    "role": "user",
                    "content": self.instructions
                })
            
            # Add user messages
            for msg in messages:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Ask OpenAI for structured output matching the Evaluation schema
            response = await self.client.chat.completions.create(
                model=self.llm_str,
                messages=openai_messages,
                max_completion_tokens=2048,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Evaluation",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["correct", "explanation"],
                            "properties": {
                                "correct": {"type": "boolean"},
                                "explanation": {"type": "string"}
                            }
                        },
                        "strict": True
                    }
                }
            )
            
            # Prefer parsed structured output if SDK provides it
            try:
                parsed = response.choices[0].message.parsed  # type: ignore[attr-defined]
                if parsed is not None:
                    return Evaluation.from_dict(parsed)
            except Exception:
                pass
            
            original_content = response.choices[0].message.content.strip()
        
        try:
            # Handle JSON code blocks
            json_content = original_content
            if json_content.startswith("```json"):
                json_content = json_content[7:]
            if json_content.endswith("```"):
                json_content = json_content[:-3]
            
            json_data = json.loads(json_content.strip())
            return Evaluation.from_dict(json_data)
            
        except json.JSONDecodeError:
            # Trigger retry by raising; caller will fallback after retries
            raise JSONParseError(original_content)


def create_message(role: str, content: str) -> Dict[str, str]:
    return {"role": role, "content": content}


async def check_answer(question: str, candidate_answer: str, reference_answer: str, grade_instruction: str, provider: str, model: str) -> Evaluation:
    grader = LLM(llm_str=model, default_instructions=grade_instruction, provider=provider)
    prompt = GRADE_PROMPT.format(question=question, candidate_answer=candidate_answer, ref_answer=reference_answer)
    message = create_message("user", prompt)
    try:
        return await grader.create_completion([message])
    except JSONParseError as e:
        correct = "true" in e.content.lower()
        return Evaluation(correct=correct, explanation="Error parsing JSON, content: " + e.content)
    except RetryError as e:  # Safety net if reraise behavior changes
        exc = None
        try:
            exc = e.last_attempt.exception()  # type: ignore[attr-defined]
        except Exception:
            pass
        if isinstance(exc, JSONParseError):
            correct = "true" in exc.content.lower()
            return Evaluation(correct=correct, explanation="Error parsing JSON, content: " + exc.content)
        raise


async def evaluate_worker(semaphore, question: str, candidate_answer: str, reference_answer: str, grade_instruction: str, provider: str, model: str):
    async with semaphore:
        evaluation = await check_answer(question, candidate_answer, reference_answer, grade_instruction, provider, model)
        await asyncio.sleep(0.5)  # Rate limiting
        return evaluation


def load_annotations(ground_truth_annotations_root_dir):
    annotation_path = Path(ground_truth_annotations_root_dir)

    # Load annotation files
    mimeqa_annotation_path = annotation_path / "mimeqa" / "metadata.csv"
    siq2_annotation_train_path = annotation_path / "siq2" / "qa" / "qa_train.json"
    siq2_annotation_val_path = annotation_path / "siq2" / "qa" / "qa_val.json"
    intentqa_annotation_train_path = annotation_path / "intentqa" / "annotations" / "train.csv"
    intentqa_annotation_val_path = annotation_path / "intentqa" / "annotations" / "val.csv"
    intentqa_annotation_test_path = annotation_path / "intentqa" / "annotations" / "test.csv"

    mimeqa_annotation = pd.read_csv(mimeqa_annotation_path)
    siq2_annotation_train = pd.read_json(siq2_annotation_train_path, lines=True)
    siq2_annotation_val = pd.read_json(siq2_annotation_val_path, lines=True)
    siq2_annotation = pd.concat([siq2_annotation_train, siq2_annotation_val])
    intentqa_annotation_train = pd.read_csv(intentqa_annotation_train_path)
    intentqa_annotation_val = pd.read_csv(intentqa_annotation_val_path)
    intentqa_annotation_test = pd.read_csv(intentqa_annotation_test_path)
    intentqa_annotation = pd.concat([intentqa_annotation_train, intentqa_annotation_val, intentqa_annotation_test])

    # Create mapping dictionaries
    mimeqa_answer_question_map = dict(zip(mimeqa_annotation['reference_answer'].str.lower(), mimeqa_annotation['question']))
    siq2_answer_question_map = dict(zip(siq2_annotation['ans_corr'].str.lower(), siq2_annotation['q']))
    intentqa_answer_question_map = dict(zip(intentqa_annotation.apply(lambda row: row[f"a{row['answer']}"].lower(), axis=1), intentqa_annotation["question"]))

    return {
        'mimeqa': mimeqa_answer_question_map,
        'siq2': siq2_answer_question_map,
        'intentqa': intentqa_answer_question_map
    }


def convert_to_results_format(data):
    """Convert parallel arrays format to list of result objects."""
    predictions = data['predictions']
    ground_truths = data['ground_truths']
    datasets = data['datasets']

    # Extract prediction from \boxed{...} format
    import re
    def extract_boxed_answer(text):
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        return match.group(1) if match else text.strip()

    results = []
    for pred, gold, dataset in zip(predictions, ground_truths, datasets):
        results.append({
            'pred': extract_boxed_answer(pred),
            'gold': gold.lower(),
            'dataset': dataset
        })

    return results


def augment_results_with_questions(results_json, answer_question_maps):
    """
    Filter results to only include those with valid question mappings.
    Returns filtered list and prints dataset statistics.
    """
    filtered_results = []
    datasets_with_annotations = set()
    datasets_without_annotations = set()
    skipped_no_dataset = 0
    skipped_no_question = 0

    for row in results_json:
        dataset = row['dataset']

        # Skip datasets we don't have annotations for
        if dataset not in answer_question_maps:
            datasets_without_annotations.add(dataset)
            skipped_no_dataset += 1
            continue

        gold = row['gold']

        # Skip if we can't find the question for this gold answer
        if gold not in answer_question_maps[dataset]:
            skipped_no_question += 1
            continue

        # Add question and include in filtered results
        row['question'] = answer_question_maps[dataset][gold]
        datasets_with_annotations.add(dataset)
        filtered_results.append(row)

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Filtering Summary:")
    print("=" * 60)
    print(f"Total results loaded: {len(results_json)}")
    print(f"Results with valid annotations: {len(filtered_results)}")
    print(f"Results filtered out: {len(results_json) - len(filtered_results)}")
    print(f"  - No dataset mapping: {skipped_no_dataset}")
    print(f"  - No question mapping: {skipped_no_question}")
    print()
    print(f"Datasets WITH annotations ({len(datasets_with_annotations)}):")
    for dataset in sorted(datasets_with_annotations):
        count = sum(1 for r in filtered_results if r['dataset'] == dataset)
        print(f"  - {dataset}: {count} results")

    if datasets_without_annotations:
        print()
        print(f"Datasets WITHOUT annotations (skipped {len(datasets_without_annotations)}):")
        for dataset in sorted(datasets_without_annotations):
            print(f"  - {dataset}")
    print("=" * 60 + "\n")

    return filtered_results


async def evaluate_results(results_json, provider: str, model: str):
    grade_instructions = {
        "mimeqa": MIMEQA_GRADE_INSTRUCTION,
        "siq2": SIQ_GRADE_INSTRUCTION, 
        "intentqa": INTENTQA_GRADE_INSTRUCTION
    }

    max_workers = 50
    semaphore = asyncio.Semaphore(max_workers)

    tasks = []
    for row in results_json:
        # Skip datasets without grade instructions (should be filtered already, but double-check)
        if row['dataset'] not in grade_instructions:
            print(f"Warning: Skipping result with unknown dataset '{row['dataset']}'")
            continue

        tasks.append(evaluate_worker(
            semaphore,
            row['question'],
            row['pred'],
            row['gold'],
            grade_instructions[row['dataset']],
            provider,
            model
        ))

    print(f"Evaluating {len(tasks)} results with {provider} ({model})...")
    grade_completions = await tqdm.gather(*tasks, desc="Evaluating results")

    graded_results = []
    for completion, row in zip(grade_completions, results_json):
        graded_results.append({
            "dataset": row['dataset'],
            "pred": row['pred'],
            "gold": row['gold'],
            "question": row.get('question', 'Unknown'),
            "graded_result": completion.correct,
            "graded_result_explanation": completion.explanation,
        })

    return graded_results


def calculate_accuracy(graded_results):
    """Calculate accuracy per dataset."""
    dataset_accuracy = {}
    for row in graded_results:
        if row['dataset'] not in dataset_accuracy:
            dataset_accuracy[row['dataset']] = []
        dataset_accuracy[row['dataset']].append(row['graded_result'])
    
    accuracy_results = {}
    for dataset, results in dataset_accuracy.items():
        accuracy = sum(results) / len(results) if results else 0.0
        accuracy_results[dataset] = accuracy
    
    return accuracy_results


async def main():
    parser = argparse.ArgumentParser(description="Process results and calculate accuracy")
    parser.add_argument("--predictions_to_eval_path", help="Path to the results JSON file", default=None)
    parser.add_argument("--grading_results_path", help="Optional path to save graded results as JSON", default=None)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--wandb_project", help="Weights & Biases project name", default="temp_llm_judge_test")
    parser.add_argument("--wandb_run_name", help="Weights & Biases run/experiment name", default="temp_llm_judge_test")
    parser.add_argument("--ground_truth_annotations_root_dir", required=True,
                        help="Path to the human_behaviour_data annotation directory")

    args = parser.parse_args()

    # If running from debugger without args, set defaults here
    if args.predictions_to_eval_path is None:
        pass

    if args.grading_results_path is None:
        # Optionally set a default save path for debugging
        # args.grading_results_path = "/path/to/data/graded_results.json"
        pass

    if args.provider == "anthropic":
        model = "claude-3-5-haiku-20241022"
    elif args.provider == "openai":
        model = "gpt-5-nano-2025-08-07"

    # Initialize wandb if project name is provided
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "provider": args.provider,
                "model": model,
                "predictions_to_eval_path": args.predictions_to_eval_path,
            }
        )
        print(f"Initialized W&B project: {args.wandb_project}, run: {args.wandb_run_name}")

    # Check if results file exists
    if not os.path.exists(args.predictions_to_eval_path):
        print(f"Error: Results file '{args.predictions_to_eval_path}' not found")
        sys.exit(1)

    # Check for required environment variables based on provider
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is required when using Anthropic")
        sys.exit(1)
    elif args.provider == "openai" and not os.getenv("MIT_OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required when using OpenAI")
        sys.exit(1)

    # Load results
    print(f"Loading results from {args.predictions_to_eval_path}")
    with open(args.predictions_to_eval_path, 'r') as f:
        data = json.load(f)

    # Convert to results format if needed
    if isinstance(data, dict) and 'predictions' in data and 'ground_truths' in data and 'datasets' in data:
        print("Converting parallel arrays format to results format...")
        results_json = convert_to_results_format(data)
    else:
        # Normalize gold to lowercase to match annotation map keys (same as convert_to_results_format)
        for row in data:
            row['gold'] = row['gold'].lower()
        results_json = data

    print(f"Loaded {len(results_json)} results")

    # Load annotations and create mappings
    print("Loading annotation data...")
    answer_question_maps = load_annotations(args.ground_truth_annotations_root_dir)

    # Filter results and get only those with valid annotations
    filtered_question_results_sets = augment_results_with_questions(results_json, answer_question_maps)

    if not filtered_question_results_sets:
        print("Error: No results with valid annotations found. Exiting.")
        sys.exit(1)

    graded_results = await evaluate_results(filtered_question_results_sets, args.provider, model)

    accuracy_results = calculate_accuracy(graded_results)

    print("\nAccuracy Results:")
    print("-" * 40)
    overall_correct = sum(row['graded_result'] for row in graded_results)
    overall_total = len(graded_results)

    for dataset, accuracy in accuracy_results.items():
        dataset_total = len([r for r in graded_results if r['dataset'] == dataset])
        print(f"{dataset}: {accuracy:.4f} ({sum([r['graded_result'] for r in graded_results if r['dataset'] == dataset])}/{dataset_total})")

    print(f"\nOverall: {overall_correct/overall_total:.4f} ({overall_correct}/{overall_total})")

    # Log to wandb if enabled
    if use_wandb:
        # Log accuracy metrics
        wandb_metrics = {
            "overall_accuracy": overall_correct / overall_total,
            "total_evaluated": overall_total,
            "total_correct": overall_correct,
        }

        # Add per-dataset accuracies
        for dataset, accuracy in accuracy_results.items():
            dataset_total = len([r for r in graded_results if r['dataset'] == dataset])
            dataset_correct = sum([r['graded_result'] for r in graded_results if r['dataset'] == dataset])
            wandb_metrics[f"{dataset}_accuracy"] = accuracy
            wandb_metrics[f"{dataset}_total"] = dataset_total
            wandb_metrics[f"{dataset}_correct"] = dataset_correct

        wandb.log(wandb_metrics)

        # Create a wandb Table for detailed graded results
        table = wandb.Table(
            columns=["dataset", "question", "prediction", "gold", "correct", "explanation"],
            data=[
                [
                    row["dataset"],
                    row["question"],
                    row["pred"],
                    row["gold"],
                    row["graded_result"],
                    row["graded_result_explanation"]
                ]
                for row in graded_results
            ]
        )
        wandb.log({"graded_results": table})

        print(f"\nLogged results to W&B project: {args.wandb_project}")

    # Save graded results if path provided
    if args.grading_results_path:
        print(f"\nSaving graded results to {args.grading_results_path}")
        with open(args.grading_results_path, 'w') as f:
            json.dump(graded_results, f, indent=2)
        print("Results saved successfully!")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
