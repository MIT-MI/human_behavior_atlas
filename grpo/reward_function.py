"""
Reward function for Human Behavior Atlas GRPO on UPSTREAM verl.

Matches verl's batch reward-manager interface (reward_manager=batch):
    compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
        -> List[Dict[str, float]]

Reward = exact-match accuracy + 0.2 * format + 0.5 * cosine-similarity(label, gt),
the same shaping used in the original mirl GRPO recipe, but with the
SentenceTransformer cached at module scope (the original reloaded it per batch).
"""
from typing import Dict, List
import re

import numpy as np

_EMBEDDER = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, ~23MB


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBEDDER


def extract_boxed_content(text: str) -> str:
    """Extract \\boxed{...}, [..], or <answer>..</answer> content; else the text."""
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1)
    markdown_match = re.search(r"\[(.*?)\]", text)
    if markdown_match:
        return markdown_match.group(1)
    answer_match = re.search(r"<answer>(.*?)</answer>", text)
    if answer_match:
        return answer_match.group(1)
    return text


def format_reward(response: str) -> float:
    """1.0 if response is <think>...</think> ... \\boxed{...} ..., else 0.0."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(pred_label: str, ground_truth: str) -> float:
    return 1.0 if pred_label == ground_truth else 0.0


def cosine_similarity_reward(pred_label: str, ground_truth: str) -> float:
    model = _get_embedder()
    emb = model.encode([pred_label, ground_truth], convert_to_numpy=True)
    p = emb[0] / (np.linalg.norm(emb[0]) + 1e-8)
    g = emb[1] / (np.linalg.norm(emb[1]) + 1e-8)
    return max(0.0, min(1.0, float(np.dot(p, g))))


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List = None,
    **kwargs,
) -> List[Dict[str, float]]:
    """verl batch reward entry point."""
    extra_infos = extra_infos or [None] * len(solution_strs)
    format_weight = 0.2
    sim_weight = 0.5

    out = []
    for predict_str, ground_truth in zip(solution_strs, ground_truths):
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str or "")
        pred_label = extract_boxed_content(full_response).lower()
        gt = (ground_truth or "").lower()

        fmt = format_reward(full_response)
        acc = accuracy_reward(pred_label, gt)
        sim = cosine_similarity_reward(pred_label, gt)
        out.append({
            "score": acc + format_weight * fmt + sim_weight * sim,
            "standard_score": acc,
            "format_score": fmt,
            "similarity_score": sim,
        })
    return out


# Backwards-compatible alias with the original mirl recipe name.
human_behaviour_compute_score_batch = compute_score_batch


if __name__ == "__main__":
    resp = ("<think>The speaker sounds angry.</think>\\boxed{anger} hope that helps.")
    print(compute_score_batch(["x"], [resp], ["anger"], [None]))
