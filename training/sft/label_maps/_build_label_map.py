#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unified label map in one pass:
  1. Read JSONL(s), collect unique (dataset, answer) pairs.
  2. Discover QA datasets (task suffix '_qa') and exclude them.
  3. Map each (dataset, answer) directly to a global class index
     using domain canonicals and synonym tables.
  4. Write a single output JSON — no intermediate file needed.

Global class space (25 classes):
  Sentiment intensity  → indices  0–6
  Emotion              → indices  7–14
  Mental health PTSD   → indices 15–16
  Mental health Depr.  → indices 17–18
  Mental health Anx.   → indices 19–20
  Sarcasm              → indices 21–22
  Humour               → indices 23–24
"""

import json
import os
import re
import sys
import gzip
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# ==========
# Paths
# ==========
INPUT_JSONLS = [
    "/path/to/your/data/train.jsonl"
]
OUTPUT_JSON = "/path/to/your/project/sft/label_maps/unified_label_map_v8.json"

# ==========
# Canonicals & Synonyms
# ==========
SENTIMENT_CANONICAL = [
    "highly negative",
    "negative",
    "weakly negative",
    "neutral",
    "weakly positive",
    "positive",
    "highly positive",
]

EMOTION_CANONICAL = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "calm",
    "sad",
    "surprise",
]

MH_PTSD_CANONICAL = ["no ptsd", "ptsd"]
MH_DEPR_CANONICAL = ["no depression", "depression"]
MH_ANX_CANONICAL  = ["no anxiety", "anxiety"]
SARCASM_CANONICAL = ["not sarcasm", "sarcasm"]
HUMOUR_CANONICAL  = ["not humour", "humour"]

EMOTION_SYNONYMS = {
    "angry": "anger",
    "happiness": "happy",
    "joy": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    "pleasant surprise": "surprise",
}

SENTIMENT_SYNONYMS = {
    "strongly negative": "highly negative",
    "negative": "negative",
    "weakly negative": "weakly negative",
    "neutral": "neutral",
    "weakly positive": "weakly positive",
    "positive": "positive",
    "strongly positive": "highly positive",
}

BOOL_SYNONYMS_TRUE  = {"true", "yes", "1", "y", "t"}
BOOL_SYNONYMS_FALSE = {"false", "no", "0", "n", "f"}

# ==========
# Dataset → domain
# ==========
DATASET_DOMAIN = {
    "chsimsv2":          "sentiment_intensity",
    "mosei_senti":       "sentiment_intensity",
    "meld_senti":        "sentiment_intensity",
    "cremad":            "emotion",
    "meld_emotion":      "emotion",
    "mosei_emotion":     "emotion",
    "tess":              "emotion",
    "ptsd_in_the_wild":  "mental_health_ptsd",
    "mmpsy_depression":  "mental_health_depression",
    "mmpsy_anxiety":     "mental_health_anxiety",
    "daicwoz":           "mental_health_depression",
    "mmsd":              "sarcasm",
    "urfunny":           "humour",
}

# ==========
# Global class list (stable indices)
# ==========
GLOBAL_CLASSES = (
    [("sentiment_intensity",      lab) for lab in SENTIMENT_CANONICAL] +   #  0–6
    [("emotion",                  lab) for lab in EMOTION_CANONICAL]    +   #  7–14
    [("mental_health_ptsd",       lab) for lab in MH_PTSD_CANONICAL]    +   # 15–16
    [("mental_health_depression", lab) for lab in MH_DEPR_CANONICAL]    +   # 17–18
    [("mental_health_anxiety",    lab) for lab in MH_ANX_CANONICAL]     +   # 19–20
    [("sarcasm",                  lab) for lab in SARCASM_CANONICAL]    +   # 21–22
    [("humour",                   lab) for lab in HUMOUR_CANONICAL]         # 23–24
)
GLOBAL_CLASS_TO_INDEX = {dl: i for i, dl in enumerate(GLOBAL_CLASSES)}
NUM_CLASSES = len(GLOBAL_CLASSES)  # 25

# ==========
# Helpers
# ==========

def open_maybe_gzip(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def discover_qa_datasets(paths) -> set:
    """Return dataset names where any row has task suffix '_qa'."""
    qa = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open_maybe_gzip(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                task    = obj.get("task", "")
                dataset = obj.get("dataset", "")
                if isinstance(task, str) and "_" in task and isinstance(dataset, str) and dataset:
                    if task.rsplit("_", 1)[-1] == "qa":
                        qa.add(dataset)
    return qa


def read_pairs(paths, exclude_datasets: set):
    """Yield (dataset, answer) for all CLS rows, skipping QA datasets."""
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] File not found: {p}", file=sys.stderr)
            continue
        with open_maybe_gzip(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ds  = obj.get("dataset")
                ans = obj.get("answer")
                if ds is None or ans is None:
                    continue
                if ds in exclude_datasets:
                    continue
                yield ds, ans


def longest_prefix_parse(key: str, known_datasets):
    for ds in sorted(known_datasets, key=len, reverse=True):
        if key.startswith(ds + "_"):
            return ds, key[len(ds) + 1:]
        if key == ds:
            return ds, ""
    return None, key


def normalize_booleanish(raw: str):
    low = raw.strip().lower()
    if low in BOOL_SYNONYMS_TRUE:
        return True
    if low in BOOL_SYNONYMS_FALSE:
        return False
    return None


def resolve_global_index(dataset: str, answer: str, report: dict):
    """
    Map a (dataset, answer) pair to a global class index.
    Returns (global_index, unified_label, domain, notes) or raises on failure.
    """
    key = f"{dataset}_{answer}"
    domain = DATASET_DOMAIN.get(dataset)
    if domain is None:
        report["unknown_dataset"].append(key)
        raise ValueError(f"Dataset '{dataset}' not in DATASET_DOMAIN. Add it or exclude it.")

    raw = answer.strip()
    low = raw.lower()
    notes = ""

    if domain == "sentiment_intensity":
        uni_label = SENTIMENT_SYNONYMS.get(low, low)
        if uni_label not in SENTIMENT_CANONICAL:
            notes = f"Normalized '{raw}' -> '{uni_label}', NOT in canonical set."
            report["noncanonical_sentiment"].append(key)

    elif domain == "emotion":
        uni_label = EMOTION_SYNONYMS.get(low, low)
        if uni_label not in EMOTION_CANONICAL:
            notes = f"Normalized '{raw}' -> '{uni_label}', NOT in canonical set."
            report["noncanonical_emotion"].append(key)

    elif domain == "mental_health_ptsd":
        low2 = " ".join(low.split())
        uni_label = "ptsd" if low2 == "ptsd" else ("no ptsd" if low2 == "no ptsd" else low2)
        if uni_label not in MH_PTSD_CANONICAL:
            notes = "Unrecognized PTSD label"

    elif domain == "mental_health_depression":
        low2 = " ".join(low.split())
        b = normalize_booleanish(low2)
        if b is True:
            uni_label = "depression"
        elif b is False:
            uni_label = "no depression"
        elif low2 in {"depression", "no depression"}:
            uni_label = low2
        else:
            uni_label = low2
            notes = "Unrecognized depression label"

    elif domain == "mental_health_anxiety":
        low2 = " ".join(low.split())
        uni_label = "anxiety" if low2 == "anxiety" else ("no anxiety" if low2 == "no anxiety" else low2)
        if uni_label not in MH_ANX_CANONICAL:
            notes = "Unrecognized anxiety label"

    elif domain == "sarcasm":
        b = normalize_booleanish(raw)
        if b is None:
            uni_label = low
            notes = "Unrecognized boolean label for sarcasm."
            report["noncanonical_binary"].append(key)
        else:
            uni_label = "sarcasm" if b else "not sarcasm"

    elif domain == "humour":
        b = normalize_booleanish(raw)
        if b is None:
            uni_label = low
            notes = "Unrecognized boolean label for humour."
            report["noncanonical_binary"].append(key)
        else:
            uni_label = "humour" if b else "not humour"

    else:
        uni_label = low
        notes = "Unknown domain; pass-through."

    global_idx = GLOBAL_CLASS_TO_INDEX.get((domain, uni_label))
    if global_idx is None:
        report["unmappable"].append(f"{key} -> ({domain}, {uni_label})")
        raise ValueError(
            f"({domain}, '{uni_label}') is NOT in GLOBAL_CLASSES. "
            f"Extend canonicals/synonyms or fix the label."
        )

    return global_idx, uni_label, domain, notes


# ==========
# Main
# ==========

def main():
    # 1) Discover QA datasets
    qa_datasets = discover_qa_datasets(INPUT_JSONLS)
    print(f"QA datasets (excluded): {sorted(qa_datasets) or 'none'}")

    # 2) Collect unique (dataset, answer) pairs, skipping QA datasets
    seen = set()
    ordered = []
    per_ds: dict[str, list] = defaultdict(list)
    for ds, ans in read_pairs(INPUT_JSONLS, qa_datasets):
        if (ds, ans) not in seen:
            seen.add((ds, ans))
            per_ds[ds].append(ans)
            ordered.append((ds, ans))

    if not ordered:
        print("[ERROR] No valid CLS pairs found.", file=sys.stderr)
        sys.exit(1)

    for ds in per_ds:
        per_ds[ds] = sorted(per_ds[ds])

    # 3) Build label_mapping (ds_answer → global index) + unified metadata
    report = {
        "unknown_dataset":       [],
        "noncanonical_sentiment": [],
        "noncanonical_emotion":  [],
        "noncanonical_binary":   [],
        "unmappable":            [],
        "excluded_qa_datasets":  sorted(qa_datasets),
    }

    label_mapping  = {}   # "cremad_anger" → global_idx
    unified_detail = {}   # "cremad_anger" → {domain, unified_label, global_index, notes}

    for ds, ans in sorted(ordered, key=lambda x: (x[0], x[1])):
        key = f"{ds}_{ans}"
        if key in label_mapping:
            continue
        global_idx, uni_label, domain, notes = resolve_global_index(ds, ans, report)
        label_mapping[key]  = global_idx
        unified_detail[key] = {
            "domain":        domain,
            "unified_label": uni_label,
            "global_index":  global_idx,
            "notes":         notes,
        }

    # 4) Build global_classes nested by domain
    global_classes_nested = {}
    for i, (d, l) in enumerate(GLOBAL_CLASSES):
        global_classes_nested.setdefault(d, []).append({"index": i, "label": l})

    output = {
        "label_mapping":  label_mapping,
        "num_classes":    NUM_CLASSES,
        "datasets":       sorted(per_ds.keys()),
        "dataset_labels": {ds: sorted(lbls) for ds, lbls in sorted(per_ds.items())},
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "excluded_qa_datasets": sorted(qa_datasets),
            "domains": {
                "sentiment_intensity":      SENTIMENT_CANONICAL,
                "emotion":                  EMOTION_CANONICAL,
                "mental_health_ptsd":       MH_PTSD_CANONICAL,
                "mental_health_depression": MH_DEPR_CANONICAL,
                "mental_health_anxiety":    MH_ANX_CANONICAL,
                "sarcasm":                  SARCASM_CANONICAL,
                "humour":                   HUMOUR_CANONICAL,
            },
            "dataset_domain":    DATASET_DOMAIN,
            "global_classes":    global_classes_nested,
            "unified_detail":    unified_detail,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(label_mapping)} classes ({NUM_CLASSES} global) to {OUTPUT_JSON}")

    # Report
    print("\n=== Build Report ===")
    for k, vals in report.items():
        print(f"- {k} ({len(vals)}):")
        for item in vals:
            print(f"    • {item}")


if __name__ == "__main__":
    main()
