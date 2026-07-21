from typing import List, Dict, Optional
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Lazy initialization for SentenceTransformer
_sentence_transformer_loaded = False
_STModel: Optional["SentenceTransformer"] = None


def _ensure_st_model():
    """Load SentenceTransformer only once (lazy load)."""
    global _sentence_transformer_loaded, _STModel
    if not _sentence_transformer_loaded:
        _STModel = SentenceTransformer('all-MiniLM-L6-v2')
        _sentence_transformer_loaded = True
    return _STModel


def extract_boxed_content(text: str) -> str:
    """Extract content within \\boxed{}, [ ], or <answer>...</answer>."""
    for pattern in [
        r"\\boxed{([^}]*)}",      # \boxed{...}
        r"\[(.*?)\]",             # [ ... ]
        r"<answer>(.*?)</answer>" # <answer>...</answer>
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text


def format_reward(response: str) -> float:
    """Reward 1.0 if format matches expected <think>...</think> ... \\boxed{...}."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Exact string match (case-insensitive handled externally)."""
    return 1.0 if response == ground_truth else 0.0


def cosine_similarity_reward(pred_label: str, ground_truth: str, model) -> float:
    """
    Compute cosine similarity between two strings using embeddings.
    Returns scaled score in [0, 1].
    """
    embeddings = model.encode([pred_label, ground_truth], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Scale from [-1, 1] → [0, 1]
    return (cos_sim + 1.0) / 2.0


def _parse_type_from_task_id(task_id: str) -> str:
    """
    Robustly extract the trailing task type from strings like:
      "chsimsv2_sentiment_intensity_cls", "intent_qa", "mime_qna", "foo_q&a"
    Rules:
      - Only the FINAL token matters.
      - Accept exactly: "cls", "qa", "qna", "q&a" (case-insensitive).
      - Do NOT treat "classification" (or similar) as "cls".
      - Default to "cls" if no recognized suffix is found.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        return "cls"

    s = task_id.strip().strip("_").lower()
    tokens = re.split(r"[_\W]+", s)
    last = tokens[-1] if tokens else ""

    if last == "cls":
        return "cls"
    if last in {"qa", "qna", "q&a"}:
        return "qa"
    return "cls"


# ---------------------------
# Soft overlong punishment
# ---------------------------
def soft_overlong_punishment(
    response_length: int,
    max_response_length: int,
    overlong_buffer_length: int,
) -> float:
    """
    Piecewise penalty in [-1, 0]:
      - If len <= (max - buffer):        0.0   (no penalty)
      - If (max - buffer) < len <= max:  linear from 0.0 down to -1.0
      - If len > max:                    -1.0  (full penalty)
    """
    expected_len = max_response_length - overlong_buffer_length
    if response_length <= expected_len:
        return 0.0
    if response_length <= max_response_length:
        # Linearly decreases from 0 → -1 across the buffer window
        return (expected_len - response_length) / float(overlong_buffer_length)
    return -1.0


def human_behaviour_compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[str],
    task_ids: Optional[List[str]] = None,
    *,
    # ---- Overlength controls (optional; pass to activate) ----
    max_response_length: Optional[int] = 812,   # e.g., model max tokens for the *response*
    overlong_buffer_length: int = 128,           # grace window before max
    overlong_penalty_factor: float = 0.75,        # scales the (<=0) penalty added to score
    # Prefer token lengths from your tokenizer; fallback is len(text)
    response_lengths: Optional[List[int]] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute scores for each response:
      - type=cls → exact string match only (no cosine)
      - type=qa  → cosine similarity only
      - always includes format score
      - (optional) soft overlong penalty applied to the final score
    """
    if task_ids is None:
        # default all to CLS semantics if not provided
        task_ids = ["cls"] * len(solution_strs)

    assert len(solution_strs) == len(ground_truths) == len(task_ids), "Input length mismatch."

    format_weight = 0.2

    need_cosine = any(_parse_type_from_task_id(tid) == "qa" for tid in task_ids)
    st_model = _ensure_st_model() if need_cosine else None

    # If response_lengths not provided, approximate with raw string length
    if response_lengths is None:
        response_lengths = [len(s or "") for s in solution_strs]

    batch_scores = []
    for i, (predict_str, ground_truth, task_id) in enumerate(zip(solution_strs, ground_truths, task_ids)):
        task_type = _parse_type_from_task_id(task_id)

        # Normalize tag spacing, then extract boxed content
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str or "")
        pred_label = extract_boxed_content(full_response).strip().lower()
        gt_norm = (ground_truth or "").strip().lower()

        format_score = format_reward(full_response)

        if task_type == "cls":
            label_score = accuracy_reward(pred_label, gt_norm)
            label_weight = 1.0 - format_weight
            overall_score = label_weight * label_score + format_weight * format_score
            similarity_score = 0.0
        else:  # QA task
            similarity_score = cosine_similarity_reward(pred_label, gt_norm, st_model)
            similarity_weight = 1.0 - format_weight
            overall_score = similarity_weight * similarity_score + format_weight * format_score
            label_score = 0.0

        # Optional soft overlength penalty
        overlong_score = 0.0
        if max_response_length is not None:
            resp_len = int(response_lengths[i])
            overlong_score = soft_overlong_punishment(
                response_length=resp_len,
                max_response_length=int(max_response_length),
                overlong_buffer_length=int(overlong_buffer_length),
            )
            # overlong_score <= 0; factor lets you tune severity
            overall_score += overlong_penalty_factor * overlong_score

        batch_scores.append({
            "score": overall_score,
            "standard_score": label_score,
            "format_score": format_score,
            "similarity_score": similarity_score,
            "overlong_score": overlong_score,   # 0..-1 (0 if disabled)
            "task_type": task_type,
        })

    return batch_scores


if __name__ == "__main__":
    cls_response = "<think>Reasoning.....</think>\\boxed{anger}"
    qa_response = "<think>Reasoning....</think>\\boxed{The Eiffel Tower is in Paris.}"
    qa_response_two = "<think>Reasoning....</think>\\boxed{good.}"

    scores = human_behaviour_compute_score_batch(
        data_sources=["", "", ""],
        solution_strs=[cls_response, qa_response, qa_response_two],
        ground_truths=["anger", "The Eiffel Tower is located in Paris.", "bad."],
        extra_infos=["", "", ""],
        task_ids=["sen_intensity_data_cls", "intent_qa", "mime_qa"],
        # ---- overlength control (example) ----
        max_response_length=512,
        overlong_buffer_length=128,
        overlong_penalty_factor=1.0,
        response_lengths=[len(cls_response), len(qa_response), 600],  # pretend the 3rd is too long
    )
    print(scores)