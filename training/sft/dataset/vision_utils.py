"""
Vision utilities for loading and preprocessing image/video inputs.
Extracted from verl/verl/utils/dataset/vision_utils.py.
"""
from io import BytesIO
from typing import Optional, Dict
import os
import time
import traceback

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def process_image(image) -> Image.Image:
    if isinstance(image, str):
        image = {"type": "image", "image": image, "min_pixels": 65536, "max_pixels": 524288}
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))
    try:
        return fetch_image(image)
    except Exception as e:
        print(e)
        dummy_image = Image.new("RGB", (224, 224))
        return process_image(dummy_image)


def process_video(video, nframes=None, fps=None, fps_min_frames=None, fps_max_frames=None, *, name_hint=None, debug=False) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] uint8 tensor."""
    start_t = time.perf_counter()

    if video == "dummy" or (isinstance(video, dict) and video.get("type") == "dummy"):
        return torch.zeros((4, 3, 224, 224), dtype=torch.uint8)

    if isinstance(video, str):
        video = {
            "type": "video", "video": video,
            "min_pixels": 147456, "max_pixels": 147456, "nframes": 4
        }

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError("Video format must be dict with key 'video'.")

    video = dict(video)
    assert nframes is None or fps is None, "Can't use both `nframes` and `fps`."
    if "nframes" not in video and "fps" not in video:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    try:
        return fetch_video(video)
    except Exception as e:
        if debug:
            print(f"[process_video][error] {e}\n{traceback.format_exc()}")
        return torch.zeros((4, 3, 224, 224), dtype=torch.uint8)
