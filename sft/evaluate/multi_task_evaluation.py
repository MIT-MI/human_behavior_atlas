import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
import torch


def _safe_div(num: float, den: float) -> float:
    """Safe division that returns 0.0 if denominator is 0."""
    return num / den if den > 0 else 0.0

def to_jsonable(o):
    # tensors
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist() if o.dim() or o.numel() > 1 else o.detach().cpu().item()
    # numpy scalars/arrays
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    # containers
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    # fallback: leave Python-native types as-is
    return o


def compute_class_counts_and_metrics(
    predictions: List[int],
    ground_truths: List[int],
    # num_classes: int,
) -> Dict[str, object]:
    """
    Single-label multiclass evaluation:
      1) Build per-class one-vs-rest TP/FP/FN and support (count).
      2) Compute per-class metrics.
         - Per-class 'accuracy' is defined as TP/support (i.e., recall).

    Args:
        predictions: List of predicted class indices
        ground_truths: List of ground truth class indices
        num_classes: Total number of classes

    Returns dict with:
      - class_metrics: {label: {precision, recall, f1, accuracy, count, confusion_matrix}}
      - pooled_counts: {'tp','fp','fn'}
      - active_classes: int
      - total_support: int
    """
    assert len(predictions) == len(ground_truths)
    
    # Convert to numpy arrays for easier manipulation
    y_pred = np.array(predictions)
    y_true = np.array(ground_truths)
    
    # Get all unique labels (which is the set of all ground truth labels)
    labels = set(y_true)
    
    N = len(y_true)
    # Per-class counts
    matrices = {c: {"tp": 0, "fp": 0, "fn": 0, "count": 0} for c in labels}

    for p, t in zip(y_pred, y_true):
        for c in labels:
            if t == c:
                matrices[c]["count"] += 1
            if p == c and t == c:
                matrices[c]["tp"] += 1
            elif p == c and t != c:
                matrices[c]["fp"] += 1
            elif p != c and t == c:
                matrices[c]["fn"] += 1

    # pooled counts for micro
    pooled_tp = sum(m["tp"] for m in matrices.values())
    pooled_fp = sum(m["fp"] for m in matrices.values())
    pooled_fn = sum(m["fn"] for m in matrices.values())

    # per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = {}
    active_classes = 0
    total_support = 0

    for c, m in matrices.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        support = m["count"]

        precision = _safe_div(tp, (tp + fp))
        recall    = _safe_div(tp, (tp + fn))
        f1        = _safe_div(2 * precision * recall, (precision + recall))
        accuracy  = recall  # per-class accuracy = TP / support

        class_metrics[str(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "count": support,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn},
        }

        if support > 0:
            active_classes += 1
            total_support += support

    return {
        "class_metrics": class_metrics,
        "pooled_counts": {"tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn},
        "active_classes": active_classes,
        "total_support": total_support,
    }

def compute_dataset_metrics(
    predictions: List[int], 
    ground_truths: List[int]
) -> Dict[str, Dict]:
    """
    Compute per-class metrics and dataset-level macro/micro/weighted metrics.
    Accuracy is corrected for single-label multiclass:
      - micro_accuracy = #correct / N
      - per-class 'accuracy' = TP/support (== recall)
    """
    summary = compute_class_counts_and_metrics(predictions, ground_truths)
    class_metrics = summary["class_metrics"]
    pooled = summary["pooled_counts"]
    active_classes = summary["active_classes"]
    total_support = summary["total_support"]  # == N

    # Accumulate macro & weighted
    keys = ["precision", "recall", "f1", "accuracy"]

    macro_sum = {k: 0.0 for k in keys}
    weighted_sum = {k: 0.0 for k in keys}

    for c, cm in class_metrics.items():
        support = cm["count"]
        if support > 0:
            for k in keys:
                macro_sum[k] += cm[k]
                weighted_sum[k] += cm[k] * support

    # Macro
    macro = {
        f"macro_{k}": _safe_div(macro_sum[k], active_classes) if active_classes > 0 else 0.0
        for k in keys
    }

    # Weighted
    weighted = {
        f"weighted_{k}": _safe_div(weighted_sum[k], total_support) if total_support > 0 else 0.0
        for k in keys
    }

    # Micro (pooled). In single-label multiclass:
    # micro_precision = micro_recall = micro_f1 = accuracy = pooled_tp / N
    PTP, PFP, PFN = pooled["tp"], pooled["fp"], pooled["fn"]
    micro_precision = _safe_div(PTP, (PTP + PFP))
    micro_recall    = _safe_div(PTP, (PTP + PFN))
    micro_f1        = _safe_div(2 * micro_precision * micro_recall, (micro_precision + micro_recall))
    micro_accuracy  = _safe_div(PTP, total_support)

    micro = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_accuracy,
    }

    dataset_metrics = {}
    dataset_metrics.update(macro)
    dataset_metrics.update(weighted)
    dataset_metrics.update(micro)

    return {
        "class_metrics": class_metrics,
        "dataset_metrics": dataset_metrics,
        "active_classes": active_classes,
    }

def compute_metrics_by_dataset(
    predictions: List[int],
    ground_truths: List[int],
    datasets: List[str],
    save_path: Optional[str] = None,
    global_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute metrics at the dataset level and a global mean across datasets.
    Aggregates the prefixed dataset-level metrics produced by compute_dataset_metrics.
    """
    
    # Save predictions and ground truths if save_path is provided
    if save_path and global_steps is not None:
        os.makedirs(save_path, exist_ok=True)
        payload = {
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "datasets": datasets,
                }
        payload = to_jsonable(payload)
        with open(os.path.join(save_path, f"val_generations_{global_steps}.json"), "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # Group by dataset
    grouped = defaultdict(lambda: {"preds": [], "gts": []})
    for p, t, d in zip(predictions, ground_truths, datasets):
        grouped[d]["preds"].append(p)
        grouped[d]["gts"].append(t)

    result: Dict[str, float] = {}
    discovered_metric_keys: List[str] = []

    for dataset_name, data in grouped.items():
        gts = data["gts"]
        preds = data["preds"]

        # Infer local num_classes from unique ground truths for this dataset
        active_labels = set(gts)
        if not active_labels:
            # nothing to evaluate for this dataset
            continue

        # Your compute_dataset_metrics already handles any OOS preds—no special handling here
        ds_res = compute_dataset_metrics(preds, gts)
        ds_metrics = ds_res["dataset_metrics"]

        if not discovered_metric_keys:
            discovered_metric_keys = sorted(ds_metrics.keys())

        for k in discovered_metric_keys:
            result[f"{dataset_name}/{k}"] = ds_metrics.get(k, 0.0)

    return result


# All logging is intentionally moved out to wandb_utils. This module only computes metrics.

def evaluate_predictions(
    predictions: List[int],
    ground_truths: List[int],
    datasets: Optional[List[str]] = None,
    global_num_classes: int = None,
    split_name: str = "validation",
    save_path: Optional[str] = None,
    global_steps: Optional[int] = None,
) -> Dict[str, object]:
    """
    Comprehensive evaluation function that computes all metrics.
    
    Args:
        predictions: List of predicted class indices
        ground_truths: List of ground truth class indices
        datasets: Optional list of dataset names for per-dataset evaluation
        num_classes: Total number of classes
        split_name: Name of the split (validation/test)
        save_path: Optional path to save results
        global_steps: Optional global step for saving
        log_to_wandb: Whether to log to wandb
        class_names: Optional list of class names
    
    Returns:
        Dictionary containing all evaluation results
    """
    # if global_num_classes is None:
    #     # Assuming that the class start from 0, this is the maximum number of classes involved
    #     global_num_classes = max(max(predictions), max(ground_truths)) + 1
    
    # Compute basic dataset metrics
    # This is basically the overall metrics for the entire validation set
    dataset_results = compute_dataset_metrics(predictions, ground_truths)
    

    # Prepare results
    results = {
        "aggregate_predictions": predictions,
        "aggregate_ground_truths": ground_truths,
        "aggregate_num_classes": global_num_classes,
        "split_name": split_name,
        "aggregate_metrics": dataset_results["dataset_metrics"],
        "aggregate_class_metrics": dataset_results["class_metrics"],
        "aggregate_active_classes": dataset_results["active_classes"],
    }
    
    # Add per-dataset metrics if datasets are provided
    

    per_dataset_metrics = compute_metrics_by_dataset(
        predictions=predictions, 
        ground_truths=ground_truths, 
        datasets=datasets, 
        save_path=save_path, 
        global_steps=global_steps
    )
    results["per_dataset_metrics"] = per_dataset_metrics
    
    return results

if __name__ == "__main__":
    # Test with synthetic data
    predictions = [0, 1, 1, 0, 1, 2, 2, 1, 1, 2]
    ground_truths = [0, 1, 1, 0, 0, 2, 2, 1, 2, 2]
    datasets = ["DatasetA"] * 10
    
    print("=== Testing multi_task_evaluation ===")
    
    # Test basic evaluation
    results = evaluate_predictions(
        predictions=predictions,
        ground_truths=ground_truths,
        datasets=datasets,
        split_name="test",
    )
    
    print("Dataset Metrics:")
    for key, value in results["aggregate_metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nPer-Dataset Metrics:")
    for key, value in results["per_dataset_metrics"].items():
        print(f"  {key}: {value:.4f}")
    
