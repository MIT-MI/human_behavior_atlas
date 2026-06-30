# evaluation.py
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch
from .helper_emo import compute_emotion_weighted_accuracies
from .helper_senti import compute_sentiment_collapsed_metrics
# from helper_emo import compute_emotion_weighted_accuracies
# from helper_senti import compute_sentiment_collapsed_metrics

# =============== utilities ===============

def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

def _accuracy(y_pred: List[int], y_true: List[int]) -> float:
    y_pred = np.asarray(y_pred); y_true = np.asarray(y_true)
    return float((y_pred == y_true).sum()) / float(max(1, y_true.size))


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


def _build_index_to_label(meta: Optional[Dict]) -> Dict[int, str]:
    """
    Build global index -> human-readable label from meta.global_classes across domains.
    """
    idx2lab: Dict[int, str] = {}
    if not meta or "global_classes" not in meta:
        return idx2lab
    for domain, entries in meta["global_classes"].items():
        for e in entries:
            idx2lab[e["index"]] = e["label"]
    return idx2lab

# =============== core metrics (as before, but adds label strings) ===============

def compute_class_counts_and_metrics(
    predictions: List[int],
    ground_truths: List[int],
    index_to_label: Optional[Dict[int, str]] = None,   # <<< NEW
) -> Dict[str, object]:
    """
    Single-label multiclass evaluation producing per-class TP/FP/FN/support
    and per-class metrics. Adds human-readable 'label_str' for interpretability.
    """
    assert len(predictions) == len(ground_truths)
    y_pred = np.array(predictions)
    y_true = np.array(ground_truths)

    labels = set(y_true)
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

    pooled_tp = sum(m["tp"] for m in matrices.values())
    pooled_fp = sum(m["fp"] for m in matrices.values())
    pooled_fn = sum(m["fn"] for m in matrices.values())

    class_metrics: Dict[str, Dict[str, float]] = {}
    active_classes = 0
    total_support = 0

    for c, m in matrices.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        support = m["count"]

        precision = _safe_div(tp, (tp + fp))
        recall    = _safe_div(tp, (tp + fn))
        f1        = _safe_div(2 * precision * recall, (precision + recall))
        accuracy  = recall  # TP/support

        class_metrics[str(c)] = {
            "label_index": int(c),                           # <<< NEW
            "class_name": index_to_label.get(int(c), "") if index_to_label else "",  # <<< NEW
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

def compute_set_metrics(
    predictions: List[int], 
    ground_truths: List[int],
    index_to_label: Optional[Dict[int, str]] = None,  # <<< threaded through
) -> Dict[str, Dict]:
    summary = compute_class_counts_and_metrics(predictions, ground_truths, index_to_label)
    class_metrics = summary["class_metrics"]
    pooled = summary["pooled_counts"]
    active_classes = summary["active_classes"]
    total_support = summary["total_support"]

    keys = ["precision", "recall", "f1", "accuracy"]
    macro_sum = {k: 0.0 for k in keys}
    weighted_sum = {k: 0.0 for k in keys}
    # ONLY SUM FOR THE ACTIVE CLASSES
    for _, cm in class_metrics.items():
        support = cm["count"]
        if support > 0:
            for k in keys:
                macro_sum[k] += cm[k]
                weighted_sum[k] += cm[k] * support

    macro = {f"macro_{k}": _safe_div(macro_sum[k], active_classes) if active_classes > 0 else 0.0 for k in keys}
    weighted = {f"weighted_{k}": _safe_div(weighted_sum[k], total_support) if total_support > 0 else 0.0 for k in keys}

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

# =============== dataset-level aggregation with domain add-ons ===============

def compute_metrics_by_dataset(
    predictions: List[int],
    ground_truths: List[int],
    datasets: List[str],
    save_path: Optional[str] = None,
    global_steps: Optional[int] = None,
    meta_config: Optional[Dict] = None,   # pass your JSON["meta"]
) -> Dict[str, float]:

   # Save predictions and ground truths if save_path is provided
    if save_path and global_steps is not None:
        os.makedirs(save_path, exist_ok=True)
        payload = {
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "datasets": datasets,
                }
        payload = to_jsonable(payload)
        with open(os.path.join(save_path, f"cls_val_generations_{global_steps}.json"), "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # DOMAIN SHOULD ALREADY BE DEFINED IN THE META DATA; we need to be careful about the meta file though; as it has strict format
    dataset_to_domain = meta_config.get("dataset_domain", {}) if meta_config else {}
    index_to_label = _build_index_to_label(meta_config) if meta_config else {}

    grouped = defaultdict(lambda: {"preds": [], "gts": []})
    for p, t, d in zip(predictions, ground_truths, datasets):
        grouped[d]["preds"].append(p)
        grouped[d]["gts"].append(t)

    result: Dict[str, float] = {}
    discovered_metric_keys: List[str] = []

    for dataset_name, data in grouped.items():
        gts = data["gts"]; preds = data["preds"]
        if not gts:
            continue
        
        # base metrics with label strings; # append the usual metrics
        ds_res = compute_set_metrics(preds, gts, index_to_label=index_to_label)
        ds_metrics = ds_res["dataset_metrics"]
        if not discovered_metric_keys:
            discovered_metric_keys = sorted(ds_metrics.keys())
        for k in discovered_metric_keys:
            result[f"{dataset_name}/{k}"] = ds_metrics.get(k, 0.0)

        # also save the per-class metrics;
        for _, cm in ds_res["class_metrics"].items():
            class_name = cm["class_name"]
            result[f"{dataset_name}/{class_name}/class_f1"] = cm["f1"]

        # domain add-ons
        domain = dataset_to_domain.get(dataset_name)

        if domain is None:
            # TODO print out the dataset that yields None as well
            print("Eval: Dataset that yielded none domain", dataset_name)
            continue

        if domain == "sentiment_intensity":
            # TODO: For baselines, probably have to change this so that it
            # TODO: works with <dataset>_<sentiment> mapping ;
            # TODO; i.e. remove the <dataset>_ prefix
            senti = compute_sentiment_collapsed_metrics(
                preds, gts, meta_config, compute_set_metrics  # inject standard fn
            )
            for k, v in senti.items():
                result[f"{dataset_name}/{k}"] = v
        # elif domain == "emotion":
        elif domain == "emotion":
            emo = compute_emotion_weighted_accuracies(
                preds, gts, index_to_label=index_to_label
            )
            # mean
            result[f"{dataset_name}/emotion_weighted_accuracy_mean"] = emo["mean_weighted_accuracy"]
            # per-class with readable, safe keys
            for wa_cm in emo["weighted_accuracy_per_class"]:
                class_name = wa_cm["class_name"]  # e.g., "anger", "pleasant_surprise"
                result[f"{dataset_name}/{class_name}/emotion_weighted_accuracy"] = wa_cm["weighted_accuracy"]

    return result

# =============== public entry point ===============

def evaluate_predictions(
    predictions: List[int],
    ground_truths: List[int],
    datasets: Optional[List[str]] = None,
    global_num_classes: int = None,
    split_name: str = "validation",
    save_path: Optional[str] = None,
    global_steps: Optional[int] = None,
    label_map_path: Optional[Dict] = None,
) -> Dict[str, object]:

    with open(label_map_path, "r") as f:
        cfg = json.load(f)
    meta_config = cfg["meta"]

    index_to_label = _build_index_to_label(meta_config) if meta_config else {}

    aggregate_dataset_results = compute_set_metrics(predictions, ground_truths, index_to_label=index_to_label)

    results = {
        "aggregate_predictions": predictions,
        "aggregate_ground_truths": ground_truths,
        "aggregate_num_classes": global_num_classes,
        "split_name": split_name,
        "aggregate_metrics": aggregate_dataset_results["dataset_metrics"],
        "aggregate_class_metrics": aggregate_dataset_results["class_metrics"],  # now with label_str
        "aggregate_active_classes": aggregate_dataset_results["active_classes"],
    }

    if datasets is not None:
        per_dataset_metrics = compute_metrics_by_dataset(
            predictions=predictions,
            ground_truths=ground_truths,
            datasets=datasets,
            save_path=save_path,
            global_steps=global_steps,
            meta_config=meta_config,
        )
        results["per_dataset_metrics"] = per_dataset_metrics
    else:
        results["per_dataset_metrics"] = {}


    return results

if __name__ == "__main__":
    # Example synthetic test set covering all three domains
    label_map_path = "/path/to/your/data/label_map.json"

    # Convenience aliases (global indices from your JSON)
    HNEG, NEG, WNEG, NEU, WPOS, POS, HPOS = 0, 1, 2, 3, 4, 5, 6
    ANGER, DISGUST, FEAR, HAPPY, ENEUT, CALM, SAD, SURPRISE = 7, 8, 9, 10, 11, 12, 13, 14
    NO_PTSD, PTSD, NO_DEP, DEP, NO_ANX, ANX = 15, 16, 17, 18, 19, 20

    predictions = [
        NEG, FEAR, POS, WNEG, HPOS,  # chsimsv2
        NEG, NEG, HPOS, NEU, WPOS,  # mosei_senti
        ANGER, ENEUT, ENEUT, SAD, HAPPY,  # meld
        CALM, SURPRISE, HAPPY, FEAR,      # ravdess
        NO_PTSD, NO_PTSD,                 # ptsd_in_the_wild
        ANX, ANX, NO_DEP, NO_DEP          # mmpsy
    ]

    ground_truths = [
        HNEG, NEU, WPOS, NEG, POS,   # chsimsv2
        NEG, NEU, POS, WNEG, WPOS,   # mosei_senti
        ANGER, HAPPY, ENEUT, SAD, SURPRISE,  # meld
        CALM, SURPRISE, ENEUT, FEAR,        # ravdess
        NO_PTSD, PTSD,                      # ptsd_in_the_wild
        ANX, NO_ANX, DEP, NO_DEP            # mmpsy
    ]

    datasets = [
        "chsimsv2", "chsimsv2", "chsimsv2", "chsimsv2", "chsimsv2",
        "mosei_senti", "mosei_senti", "mosei_senti", "mosei_senti", "mosei_senti",
        "meld", "meld", "meld", "meld", "meld",
        "ravdess", "ravdess", "ravdess", "ravdess",
        "ptsd_in_the_wild", "ptsd_in_the_wild",
        "mmpsy", "mmpsy", "mmpsy", "mmpsy"
    ]

    results = evaluate_predictions(
        predictions=predictions,
        ground_truths=ground_truths,
        datasets=datasets,
        split_name="synthetic_test",
        label_map_path=label_map_path,   # load from your JSON
    )

    print("\n=== Aggregate Metrics ===")
    for k, v in results["aggregate_metrics"].items():
        print(f"{k}: {v:.4f}")

    print("\n=== Per-Dataset Keys ===")
    for k in sorted(results["per_dataset_metrics"].keys()):
        print(f"{k}: {results['per_dataset_metrics'][k]:.4f}")
