# emotion_metrics.py
from typing import Dict, List, Optional, Tuple
import numpy as np

def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

def compute_emotion_weighted_accuracies(
    y_pred: List[int],
    y_true: List[int],
    index_to_label: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """
    For each class c:
      WA_c = 0.5 * (TP/P + TN/N)
    Returns:
      {
        "mean": float,
        "per_class": [
           {"class_index": int, "label": str, "label_key": str, "weighted_accuracy": float}
           ...
        ]
      }
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    classes = sorted(set(y_true.tolist()))

    if len(y_true) == 0 or len(classes) == 0:
        return {"mean_weighted_accuracy": 0.0, "weighted_accuracy_per_class": []}

    wa_per_class = []
    accs = []
    for c in classes:
        P = int((y_true == c).sum())
        N = int((y_true != c).sum())
        TP = int(((y_true == c) & (y_pred == c)).sum())
        TN = int(((y_true != c) & (y_pred != c)).sum())
        wa_c = 0.5 * (_safe_div(TP, P) + _safe_div(TN, N))
        class_name = index_to_label.get(int(c), str(c)) if index_to_label else str(c)
        
        # to standardise keep this the same as the main function
        wa_per_class.append({
            "label_index": int(c),
            "class_name": class_name,
            "weighted_accuracy": float(wa_c),
        })
        accs.append(wa_c)

    mean_wa = float(np.mean(accs)) if accs else 0.0
    return {"mean_weighted_accuracy": mean_wa, "weighted_accuracy_per_class": wa_per_class}
