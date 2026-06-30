"""
Convert Human Behavior Atlas parquet into UPSTREAM-verl-native RLHF parquet.

HBA stores raw media bytes + a `problem`/`answer`/`modality_signature` schema and
relied on the mirl fork's runtime `data.modalities` / `format_prompt` machinery.
Upstream verl expects a self-describing parquet:
    data_source : str
    prompt      : [{"role": "user", "content": "<video>... + instruction"}]
    videos      : [[frame_path_0, ...]]      (list of videos; each a frame list)
    images      : [{"bytes": ...}]           (when --modality image)
    reward_model: {"style": "rule", "ground_truth": <answer>}
    ability     : <task>
    extra_info  : {split, index, answer, question, dataset, task}

For video we seek-sample N frames and write small JPGs (HBA clips are 40-235MB),
so the output is light and deterministic, and Qwen-VL's `qwen_vl_utils` consumes
the frame-path lists directly.

Usage:
    python prepare_grpo_data.py \
        --data_dir /data/human_behavior_atlas_v2 \
        --out_dir  ./grpo_data --split train --modality video \
        --nframes 8
"""
import argparse
import glob
import io
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

INSTRUCTION = (
    r"You FIRST think about the reasoning process as an internal monologue and then "
    r"provide the final answer. The reasoning process MUST BE enclosed within "
    r"<think> </think> tags. The final answer MUST BE put in \boxed{}."
)


def find_files(data_dir, split):
    files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files for split '{split}' in {data_dir}")
    return files


def sample_video_frames(video_bytes, nframes):
    """Seek-sample `nframes` RGB frames from raw video bytes (PyAV)."""
    import av

    frames = []
    container = av.open(io.BytesIO(video_bytes))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        n_total = stream.frames or 0
        duration = stream.duration
        if duration and n_total > nframes:
            start = stream.start_time or 0
            for i in range(nframes):
                ts = int(start + duration * i / nframes)
                try:
                    container.seek(ts, stream=stream, any_frame=False, backward=True)
                    for frame in container.decode(video=0):
                        frames.append(frame.to_ndarray(format="rgb24"))
                        break
                except Exception:
                    continue
        if not frames:
            count = 0
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                count += 1
                if count >= 512:
                    break
            if len(frames) > nframes:
                idx = np.linspace(0, len(frames) - 1, nframes, dtype=int)
                frames = [frames[i] for i in idx]
    finally:
        container.close()
    return frames


def build_prompt(problem):
    # Vision-only: drop <audio> tags, keep <video>/<image>.
    text = (problem or "").replace("<audio>", "").strip()
    content = f"{text} {INSTRUCTION}"
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--modality", choices=["video", "image"], default="video")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap on samples per split (default: None = use all)")
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--out_name", default=None, help="output parquet filename (default: {split}.parquet)")
    args = ap.parse_args()

    keyword = args.modality  # 'video' or 'image' substring in modality_signature
    media_dir = os.path.join(args.out_dir, "media", args.split)
    os.makedirs(media_dir, exist_ok=True)

    files = find_files(args.data_dir, args.split)
    print(f"[prepare] {len(files)} parquet files for split '{args.split}'")

    rows = []
    n = 0
    for f in files:
        if args.max_samples is not None and n >= args.max_samples:
            break
        df = pd.read_parquet(f)
        if "modality_signature" in df.columns:
            df = df[df["modality_signature"].str.contains(keyword, case=False, na=False)]
        for _, row in df.iterrows():
            if args.max_samples is not None and n >= args.max_samples:
                break
            answer = str(row.get("answer", ""))
            problem = str(row.get("problem", ""))
            data_source = str(row.get("dataset", "hba"))
            task = str(row.get("task", "unknown"))

            record = {
                "data_source": data_source,
                "prompt": build_prompt(problem),
                "ability": task,
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": args.split,
                    "index": n,
                    "answer": answer,
                    "question": problem,
                    "dataset": data_source,
                    "task": task,
                },
            }

            if args.modality == "video":
                vids = row.get("videos")
                if vids is None or len(vids) == 0 or vids[0] is None:
                    continue
                frames = sample_video_frames(vids[0], args.nframes)
                if not frames:
                    continue
                sample_dir = os.path.join(media_dir, str(n))
                os.makedirs(sample_dir, exist_ok=True)
                frame_paths = []
                for k, fr in enumerate(frames):
                    p = os.path.abspath(os.path.join(sample_dir, f"frame_{k:03d}.jpg"))
                    Image.fromarray(fr).convert("RGB").save(p, quality=90)
                    frame_paths.append(p)
                record["videos"] = [frame_paths]  # one video = list of frame paths
            else:  # image
                imgs = row.get("images")
                if imgs is None or len(imgs) == 0 or imgs[0] is None:
                    continue
                record["images"] = [{"bytes": bytes(b)} for b in imgs if b is not None and len(b) > 0]

            rows.append(record)
            n += 1
        print(f"[prepare] {os.path.basename(f)} -> {n} samples so far")

    if not rows:
        raise SystemExit(f"No '{args.modality}' rows found in split '{args.split}'.")

    out_name = args.out_name or f"{args.split}.parquet"
    out_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[prepare] wrote {len(rows)} samples -> {out_path}")
    print(f"[prepare] media frames under {media_dir}")


if __name__ == "__main__":
    main()
