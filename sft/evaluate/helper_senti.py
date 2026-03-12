# sentiment_metrics.py
from typing import Dict, List, Optional, Tuple
import numpy as np

def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

def _order_sentiment_labels(meta: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, int]]:
    """
    Returns:
      label_order: canonical 7-list in your taxonomy (by label text)
      label2pos: map label -> 0..6 "position" (canonical order)
      global2pos: map global_index -> 0..6 "position"; Essentially making sure that our global indices are always mapped to the 0-6 conical sentiment scale
    """
    # Expect these exact 7 canonical labels in meta.global_classes['sentiment_intensity']
    # (Your JSON shows them.)
    canonical = [
        "highly negative",
        "negative",
        "weakly negative",
        "neutral",
        "weakly positive",
        "positive",
        "highly positive",
    ]
    gc = meta["global_classes"]["sentiment_intensity"]  # list of sentiment labels {index, label}
    label_to_global = {entry["label"]: entry["index"] for entry in gc}
    # Ensure all canonicals exist in the meta file
    missing = [lab for lab in canonical if lab not in label_to_global]
    if missing:
        raise ValueError(f"Sentiment labels missing in meta: {missing}")

    label2pos = {lab: i for i, lab in enumerate(canonical)}
    global2pos = {label_to_global[lab]: label2pos[lab] for lab in canonical}
    return canonical, label2pos, global2pos

def _sentiment_sets(meta: Dict) -> Dict[str, set]:
    """Derive NEG/NEU/POS sets from meta, in terms of *positions* 0..6."""
    _, _, g2p = _order_sentiment_labels(meta)
    # negative group
    neg_labels = {"highly negative", "negative", "weakly negative"}
    pos_labels = {"weakly positive", "positive", "highly positive"}
    neu_label = "neutral"

    # Map via label->global->position
    gc = meta["global_classes"]["sentiment_intensity"]
    label2global = {e["label"]: e["index"] for e in gc}

    NEG = { g2p[label2global[l]] for l in neg_labels }
    POS = { g2p[label2global[l]] for l in pos_labels }
    NEU = { g2p[label2global[neu_label]] }
    ALL = set(range(7))
    return {"NEG": NEG, "POS": POS, "NEU": NEU, "ALL": ALL}

def _translate_global_to_pos(seq: List[int], meta: Dict) -> List[int]:
    """Map global indices to 0..6 sentiment positions; basically a look up for the meta labels, so that they are in the correct order"""
    _, _, g2p = _order_sentiment_labels(meta)
    out = []
    for v in seq:
        if v in g2p:
            out.append(g2p[v])
        else:
            out.append(-1)
        
    return out

def _collapse_positions(seq_pos: List[int], mode: int, meta: Dict) -> List[int]:
    """
    Collapse 0..6 positions to target classes.
      mode=7: identity -> 0..6
      mode=5: {HNEG, NEG-group, NEU, POS-group, HPOS} -> 0..4
      mode=3: {NEG, NEU, POS} -> 0..2
      mode=2: {NEG, POS} -> 0..1 (NEU goes to NEG by convention)
    """
    sets = _sentiment_sets(meta)
    NEG, POS, NEU = sets["NEG"], sets["POS"], sets["NEU"]
    out = []
    for v in seq_pos:
        if mode == 7:
            out.append(v)
        elif mode == 5:
            # 0:HNEG, 1:NEG-group, 2:NEU, 3:POS-group, 4:HPOS
            if v == min(NEG):        # HNEG at canonical position 0
                out.append(0)
            elif v in (NEG - {min(NEG)}): # in the negative group but not the lowest negative
                out.append(1)
            elif v in NEU:
                out.append(2)
            elif v in (POS - {max(POS)}):
                out.append(3)
            elif v == max(POS):      # HPOS at canonical position 6
                out.append(4)
            else:
                # if its not inside then it should be wrong; i.e. append -1
                out.append(-1)
        elif mode == 3:
            if v in NEG:
                out.append(0)
            elif v in NEU:
                out.append(1)
            elif v in POS:
                out.append(2)
            else:
                out.append(-1)
                
        elif mode == 2:
            # Binary collapse: POS → 1, NEG → 0
            # NOTE: Neutral is *excluded* here. Handle separately if needed.
            if v in POS:
                out.append(1)
            elif v in NEG:
                out.append(0)
            else:
                # Neutral (or anything unexpected)
                # Append the wrong position index
                out.append(-1)
        else:
            raise ValueError(f"Unsupported sentiment collapse mode: {mode}")
    return out

def _weighted_f1_and_acc_standard(y_pred: List[int], y_true: List[int], compute_dataset_metrics) -> Tuple[float, float]:
    """Use your standard compute_dataset_metrics to fetch weighted F1 and accuracy."""
    ds = compute_dataset_metrics(y_pred, y_true)["dataset_metrics"]
    return ds.get("weighted_f1", 0.0), ds.get("micro_accuracy", 0.0)

def compute_sentiment_collapsed_metrics(
    y_pred_global: List[int],
    y_true_global: List[int],
    meta: Dict,
    compute_dataset_metrics,  # inject your standard function to avoid circular imports
) -> Dict[str, float]:
    """
    Filter to sentiment domain (by membership in meta.global_classes['sentiment_intensity']),
    then compute ACC2/3/5/7 and F1w2/3/5/7.
    """
    # Build set of sentiment global indices from meta
    senti_globals = {e["index"] for e in meta["global_classes"]["sentiment_intensity"]}

    idxs = [i for i, t in enumerate(y_true_global) if t in senti_globals]
    if not idxs:
        return {}

    y_true_sent = [y_true_global[i] for i in idxs]
    y_pred_sent = [y_pred_global[i] for i in idxs]

    # Map to the labels and predictions according to the meta file to the sentiment 0..6 canonical positions
    y_true_pos = _translate_global_to_pos(y_true_sent, meta)
    y_pred_pos = _translate_global_to_pos(y_pred_sent, meta)

    metrics: Dict[str, float] = {}

    # mode 7
    y_t7 = _collapse_positions(y_true_pos, 7, meta)
    y_p7 = _collapse_positions(y_pred_pos, 7, meta)
    f1w7, acc7 = _weighted_f1_and_acc_standard(y_p7, y_t7, compute_dataset_metrics)
    metrics.update({"ACC7": acc7, "F1w7": f1w7})

    # mode 5
    y_t5 = _collapse_positions(y_true_pos, 5, meta)
    y_p5 = _collapse_positions(y_pred_pos, 5, meta)
    f1w5, acc5 = _weighted_f1_and_acc_standard(y_p5, y_t5, compute_dataset_metrics)
    metrics.update({"ACC5": acc5, "F1w5": f1w5})

    # mode 3
    y_t3 = _collapse_positions(y_true_pos, 3, meta)
    y_p3 = _collapse_positions(y_pred_pos, 3, meta)
    f1w3, acc3 = _weighted_f1_and_acc_standard(y_p3, y_t3, compute_dataset_metrics)
    metrics.update({"ACC3": acc3, "F1w3": f1w3})

    # mode 2 (exclude neutral GT)
    sets = _sentiment_sets(meta)
    NEU = sets["NEU"]
    keep = [i for i, t in enumerate(y_t7) if t not in NEU]  # y_t7 is 0..6 positions
    if keep:
        y_t2_src = [y_true_pos[i] for i in keep]
        y_p2_src = [y_pred_pos[i] for i in keep]
        y_t2 = _collapse_positions(y_t2_src, 2, meta)
        y_p2 = _collapse_positions(y_p2_src, 2, meta)
        f1w2, acc2 = _weighted_f1_and_acc_standard(y_p2, y_t2, compute_dataset_metrics)
        metrics.update({"ACC2": acc2, "F1w2": f1w2})
    else:
        metrics.update({"ACC2": 0.0, "F1w2": 0.0})

    return metrics
