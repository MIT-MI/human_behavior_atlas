from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_boxed_content(text: str) -> str:
    """
    Extract content within \boxed{} or similar boxing notations.

    Args:
        text (str): Text containing potentially boxed content.

    Returns:
        str: Extracted boxed content or the original text if no box found.
    """

    # Look for LaTeX \boxed{} notation
    boxed_match = re.search(r"\\boxed{([^}]*)}", text)
    if boxed_match:
        return boxed_match.group(1)

    # Look for markdown boxed notation (e.g., [boxed content])
    markdown_match = re.search(r"\[(.*?)\]", text)
    if markdown_match:
        return markdown_match.group(1)

    # Some response use <answer>...</answer> instead of \boxed{...}.
    answer_match = re.search(r"<answer>(.*?)</answer>", text)
    if answer_match:
        return answer_match.group(1)

    # Return the text as is if no boxed content is found
    return text

def format_reward(response: str) -> float:
    """
    Check whether the response matches the expected format.
    Here we require something like <think>...</think> ... \boxed{...}
    """
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Simple accuracy: exact match to ground truth string.
    """
    return 1.0 if response == ground_truth else 0.0

def cosine_similarity_reward(pred_label: str, ground_truth: str, model: SentenceTransformer) -> float:
    """
    Compute cosine similarity between predicted label and ground truth using embeddings.

    Args:
        pred_label: Predicted label string
        ground_truth: Ground truth string
        model: SentenceTransformer model for embeddings

    Returns:
        Cosine similarity score between 0 and 1
    """
    # Get embeddings for both strings
    embeddings = model.encode([pred_label, ground_truth], convert_to_numpy=True)

    # Compute cosine similarity
    pred_emb = embeddings[0]
    gt_emb = embeddings[1]

    # Normalize vectors
    pred_norm = pred_emb / np.linalg.norm(pred_emb)
    gt_norm = gt_emb / np.linalg.norm(gt_emb)

    # Compute cosine similarity
    cos_sim = np.dot(pred_norm, gt_norm)

    # Ensure the value is between 0 and 1
    return max(0.0, min(1.0, float(cos_sim)))

def human_behaviour_compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[str],
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute human behaviour scoring for batch inputs.

    Args:
        data_sources: List of data sources (unused here, but kept for interface compatibility)
        solution_strs: List of model prediction strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra information (unused here, kept for compatibility)

    Returns:
        List of score dictionaries
    """
    batch_scores = []
    format_weight = 0.2 # weight for format correctness

    # Initialize the embedding model (using a lightweight model)
    # 'all-MiniLM-L6-v2' is fast and only 22.7MB
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    for data_source, predict_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        # Normalize response formatting (e.g., qwen2.5vl quirks)
        full_response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)
        pred_label = extract_boxed_content(full_response).lower()  # handle qwen2.5vl-32b format
        ground_truth = ground_truth.lower()

        # print(pred_label)
        # Compute individual components
        format_score = format_reward(full_response)
        standard_score = accuracy_reward(pred_label, ground_truth)
        similarity_score = cosine_similarity_reward(pred_label, ground_truth, embedding_model)

        # Sum of all three rewards
        overall_score = standard_score + format_weight * format_score + 0.5 * similarity_score

        scores = {
            "score": overall_score,
            "standard_score": standard_score,
            "format_score": format_score,
            "similarity_score": similarity_score,
        }
        batch_scores.append(scores)

    return batch_scores


if __name__ == "__main__":
    response_str = (
        "<think>Well, I've listened to the speech recording. It sounds like the speaker is expressing anger. "
        "You know, the tone and the way the words are said seem to indicate frustration or annoyance. "
        "So, I'd say the emotion is anger.</think>\\boxed{anger}If you have any other questions or need more help, feel free to let me know."
    )

    data_sources = ["sample_audio.wav"]
    solution_strs = [response_str]
    ground_truths = ["anger"]
    extra_infos = [""]

    scores = human_behaviour_compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)
    print(scores)