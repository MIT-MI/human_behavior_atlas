"""
SFT dataset for HuggingFace-native parquet files with embedded binary media.

Reads parquet files directly with pyarrow — no HF datasets cache (avoids
duplicating 300GB+ of data on disk). Model-agnostic: works with any
transformers VLM/Omni processor (Qwen2.5-VL, Qwen3-VL, Gemma 3/4, Qwen2.5-Omni).

Enhancements over the original omni-only loader:
  * Optional modality filtering via the `modality_signature` column, so a
    vision-only model can train on just the image/video rows of a mixed
    (audio-heavy) corpus like Human Behavior Atlas.
  * Seek-based video frame sampling (HBA clips can be 200MB+), with a hard
    decode cap and a sequential fallback.
  * Inactive-modality tags (e.g. `<audio>` when audio is disabled) are stripped
    from the prompt text instead of leaking through as literal text.
"""
import glob
import io
import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from jinja2 import Template
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from dataset.dataset_utils import compute_position_id_with_mask


# Maps a configured modality -> the substring that identifies it in the
# dataset's `modality_signature` column (e.g. "text_video_audio").
_MODALITY_KEYWORDS = {"images": "image", "videos": "video", "audio": "audio"}
# Tags that may appear inline in the prompt text for each modality.
_MODALITY_TAGS = {"images": "<image>", "videos": "<video>", "audio": "<audio>"}


def _parse_split(split: str):
    """Parse HF-style split notation like 'train[:10%]' or 'validation'.

    Returns (split_name, start_frac, end_frac).
    """
    match = re.match(r'^(\w+)\[(\d*%?):(\d*%?)\]$', split)
    if match:
        name = match.group(1)
        start_str = match.group(2)
        end_str = match.group(3)
        start = float(start_str.rstrip('%')) / 100.0 if start_str else 0.0
        end = float(end_str.rstrip('%')) / 100.0 if end_str else 1.0
        return name, start, end
    return split, 0.0, 1.0


class SFTParquetDataset(Dataset):
    """
    Dataset for SFT training on parquet files with embedded binary media.
    Reads parquet files directly with pyarrow — zero cache overhead.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config,
        processor: Optional[ProcessorMixin] = None,
        split: str = "train",
    ):
        # Flatten nested lists and ensure plain strings
        if isinstance(data_files, (str, bytes)):
            data_files = [data_files]
        flat = []
        for f in data_files:
            if isinstance(f, (list, ListConfig)):
                flat.extend(str(x) for x in f)
            else:
                flat.append(str(f))
        self.data_files = flat
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.split = split

        self.prompt_key = config.get("prompt_key", "problem")
        self.response_key = config.get("response_key", "answer")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.audio_key = config.get("audio_key", "audios")
        self.modalities = set(config.get("modalities", "images,videos").split(","))

        self.max_prompt_length = config.get("max_prompt_length", 4096)
        self.max_response_length = config.get("max_response_length", 512)
        self.max_length = config.get("max_length", self.max_prompt_length + self.max_response_length)
        self.truncation = config.get("truncation", "right")
        self.audio_max_seconds = config.get("audio_max_seconds", 10)
        self.video_nframes = config.get("video_nframes", 8)

        # Optional: keep only rows whose `modality_signature` matches the active
        # modalities (e.g. drop pure-audio rows when training a vision model).
        self.filter_by_modality = bool(config.get("filter_by_modality", False))
        self.modality_signature_key = config.get("modality_signature_key", "modality_signature")
        self._active_keywords = [
            _MODALITY_KEYWORDS[m] for m in self.modalities if m in _MODALITY_KEYWORDS
        ]
        # Tags to strip from the prompt because their modality is disabled.
        self._strip_tags = [
            tag for mod, tag in _MODALITY_TAGS.items() if mod not in self.modalities
        ]

        self.format_prompt_path = config.get("format_prompt", None)
        self.format_prompt = self._load_format_prompt()

        self._read_files()

    def _load_format_prompt(self) -> Optional[Template]:
        if self.format_prompt_path and os.path.exists(self.format_prompt_path):
            with open(self.format_prompt_path, "r", encoding="utf-8") as f:
                return Template(f.read())
        return None

    def _find_parquet_files(self, path: str, split_name: str) -> list:
        """Find parquet files matching the split name in a directory."""
        if os.path.isfile(path) and path.endswith(".parquet"):
            return [path]
        if not os.path.isdir(path):
            raise ValueError(f"Path not found: {path}")

        # Match HF naming convention: {split}-XXXXX-of-XXXXX.parquet
        pattern = os.path.join(path, f"{split_name}-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            # Fallback: try all parquet files
            files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise ValueError(f"No parquet files found in {path} for split '{split_name}'")
        return files

    def _sig_matches(self, sig) -> bool:
        """Whether a modality_signature row matches the active modalities."""
        if not self._active_keywords:
            return True
        sig = (sig or "").lower()
        return any(kw in sig for kw in self._active_keywords)

    def _read_files(self):
        """Load parquet file metadata. Applies modality filtering + split slicing."""
        split_name, start_frac, end_frac = _parse_split(self.split)

        all_files = []
        for path in self.data_files:
            all_files.extend(self._find_parquet_files(path, split_name))

        print(f"[SFTParquetDataset] Found {len(all_files)} parquet files for split '{split_name}'")

        self._file_paths = all_files
        self._file_row_counts = []
        for f in all_files:
            self._file_row_counts.append(pq.ParquetFile(f).metadata.num_rows)
        total_rows = sum(self._file_row_counts)

        # Cumulative offsets for global-row -> (file, local_row) mapping.
        self._cumulative_rows = []
        cumsum = 0
        for count in self._file_row_counts:
            self._cumulative_rows.append(cumsum)
            cumsum += count

        self._index = None  # explicit (file_idx, local_row) list when filtering
        if self.filter_by_modality and self._active_keywords:
            full_index = self._build_filtered_index(all_files)
            n = len(full_index)
            start = int(n * start_frac)
            end = int(n * end_frac)
            self._index = full_index[start:end]
            self._total_rows = len(self._index)
            print(f"[SFTParquetDataset] Modality filter {sorted(self.modalities)}: "
                  f"{n}/{total_rows} rows matched; using [{start_frac*100:.0f}%:{end_frac*100:.0f}%] "
                  f"= {self._total_rows} samples")
        else:
            start_row = int(total_rows * start_frac)
            end_row = int(total_rows * end_frac)
            self._total_rows = end_row - start_row
            self._global_offset = start_row
            print(f"[SFTParquetDataset] Total rows: {total_rows}, "
                  f"using [{start_frac*100:.0f}%:{end_frac*100:.0f}%] = {self._total_rows} samples")

        # Cache for loaded file DataFrames (small LRU).
        self._file_cache = {}
        self._cache_max = 2

    def _build_filtered_index(self, all_files):
        """Read the lightweight modality_signature column and keep matching rows."""
        full_index = []
        for fi, f in enumerate(all_files):
            try:
                col = pq.read_table(f, columns=[self.modality_signature_key])
                sigs = col.column(0).to_pylist()
            except (KeyError, Exception) as e:  # noqa: B014 - column may be absent
                warnings.warn(
                    f"[SFTParquetDataset] modality filtering disabled for {os.path.basename(f)}: "
                    f"could not read '{self.modality_signature_key}' ({e}); keeping all rows."
                )
                sigs = [None] * self._file_row_counts[fi]
            for li, sig in enumerate(sigs):
                if self._sig_matches(sig):
                    full_index.append((fi, li))
        return full_index

    def _global_to_file_row(self, idx: int):
        """Map a dataset index to (file_index, row_within_file)."""
        if self._index is not None:
            return self._index[idx]
        global_row = self._global_offset + idx
        # Binary search for the file
        lo, hi = 0, len(self._cumulative_rows) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cumulative_rows[mid] <= global_row:
                lo = mid
            else:
                hi = mid - 1
        file_idx = lo
        local_row = global_row - self._cumulative_rows[file_idx]
        return file_idx, local_row

    def _load_file(self, file_idx: int) -> pd.DataFrame:
        """Load a parquet file, with simple caching."""
        if file_idx in self._file_cache:
            return self._file_cache[file_idx]
        if len(self._file_cache) >= self._cache_max:
            oldest = next(iter(self._file_cache))
            del self._file_cache[oldest]
        df = pd.read_parquet(self._file_paths[file_idx])
        self._file_cache[file_idx] = df
        return df

    def _get_row(self, idx: int) -> dict:
        """Get a single row by dataset index."""
        file_idx, local_row = self._global_to_file_row(idx)
        df = self._load_file(file_idx)
        row = df.iloc[local_row]
        return row.to_dict()

    def __len__(self):
        return self._total_rows

    def _decode_audio(self, audio_bytes):
        """Decode raw audio bytes to tensor using scipy + torchaudio resample."""
        import scipy.io.wavfile as wav
        import torchaudio

        buf = io.BytesIO(audio_bytes)
        sr, audio_np = wav.read(buf)

        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        elif audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        audio_data = torch.from_numpy(audio_np)

        target_sr = 16000
        if self.processor and hasattr(self.processor, "feature_extractor"):
            if hasattr(self.processor.feature_extractor, "sampling_rate"):
                target_sr = self.processor.feature_extractor.sampling_rate

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio_data = resampler(audio_data)

        if self.audio_max_seconds:
            max_samples = int(self.audio_max_seconds * target_sr)
            if audio_data.shape[0] > max_samples:
                audio_data = audio_data[:max_samples]

        return audio_data, target_sr

    def _decode_video(self, video_bytes, nframes=None):
        """Decode raw video bytes to [nframes, 3, H, W] uint8 tensor using PyAV.

        Uses timestamp seeking to sample frames without decoding the whole clip
        (HBA videos can be 200MB+). Falls back to a capped sequential decode.
        """
        import av

        nframes = nframes or self.video_nframes
        frames = []
        container = av.open(io.BytesIO(video_bytes))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            n_total = stream.frames or 0
            duration = stream.duration  # in stream.time_base units

            # Preferred path: seek to evenly spaced timestamps, grab 1 frame each.
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

            # Fallback: sequential decode with a hard cap to avoid OOM.
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

        if not frames:
            return torch.zeros((nframes, 3, 224, 224), dtype=torch.uint8)

        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # [N,3,H,W]
        return video_tensor

    def _decode_image(self, image_bytes):
        """Decode raw image bytes to PIL Image."""
        from PIL import Image

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def _nonempty(seq) -> bool:
        """Length check that is safe for None and numpy arrays (pandas returns arrays)."""
        return seq is not None and len(seq) > 0

    def _build_messages(self, example: dict):
        """Build chat messages from the prompt text, inserting multimodal tags."""
        prompt_text = example.get(self.prompt_key, "")

        # Strip tags for modalities that are disabled (e.g. <audio> for a VL model)
        # so they don't leak through as literal text.
        for tag in self._strip_tags:
            prompt_text = prompt_text.replace(tag, "")

        has_audio = "audio" in self.modalities and self._nonempty(example.get(self.audio_key))
        has_video = "videos" in self.modalities and self._nonempty(example.get(self.video_key))
        has_image = "images" in self.modalities and self._nonempty(example.get(self.image_key))

        if self.format_prompt:
            prompt_text = self.format_prompt.render(content=prompt_text)

        if has_audio or has_video or has_image:
            tag_patterns = []
            if "images" in self.modalities:
                tag_patterns.append("<image>")
            if "videos" in self.modalities:
                tag_patterns.append("<video>")
            if "audio" in self.modalities:
                tag_patterns.append("<audio>")

            content_list = []
            if tag_patterns:
                pattern = "(" + "|".join(re.escape(t) for t in tag_patterns) + ")"
                segments = [s for s in re.split(pattern, prompt_text) if s]
                for seg in segments:
                    if seg == "<image>" and "images" in self.modalities:
                        content_list.append({"type": "image"})
                    elif seg == "<video>" and "videos" in self.modalities:
                        content_list.append({"type": "video"})
                    elif seg == "<audio>" and "audio" in self.modalities:
                        content_list.append({"type": "audio"})
                    else:
                        content_list.append({"type": "text", "text": seg})
            else:
                content_list.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content_list}]
        else:
            messages = [{"role": "user", "content": prompt_text}]

        if has_audio:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                )}]
            })

        return messages

    def __getitem__(self, item):
        row = self._get_row(item)

        messages = self._build_messages(row)
        answer = row.get(self.response_key, "")

        # Build processor inputs
        if self.processor is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="System prompt modified")
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )

            processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

            if "audio" in self.modalities and row.get(self.audio_key) is not None and len(row[self.audio_key]) > 0:
                audios_np = []
                for audio_bytes in row[self.audio_key]:
                    if audio_bytes is not None and len(audio_bytes) > 0:
                        arr, sr = self._decode_audio(audio_bytes)
                        audios_np.append(arr.numpy().astype("float32"))
                if audios_np:
                    processor_kwargs["audio"] = audios_np

            if "videos" in self.modalities and row.get(self.video_key) is not None and len(row[self.video_key]) > 0:
                videos = []
                for video_bytes in row[self.video_key]:
                    if video_bytes is not None and len(video_bytes) > 0:
                        videos.append(self._decode_video(video_bytes))
                if videos:
                    processor_kwargs["videos"] = videos

            if "images" in self.modalities and row.get(self.image_key) is not None and len(row[self.image_key]) > 0:
                images = []
                for img_bytes in row[self.image_key]:
                    if img_bytes is not None and len(img_bytes) > 0:
                        images.append(self._decode_image(img_bytes))
                if images:
                    processor_kwargs["images"] = images

            model_inputs = self.processor(**processor_kwargs)

            prompt_input_ids = model_inputs["input_ids"][0]
            prompt_attention_mask = model_inputs["attention_mask"][0]
            multi_modal_inputs = {
                k: v for k, v in model_inputs.items()
                if k not in ("input_ids", "attention_mask", "second_per_grid_ts")
            }
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            prompt_input_ids = enc["input_ids"][0]
            prompt_attention_mask = enc["attention_mask"][0]
            multi_modal_inputs = {}

        # Tokenize response
        response_str = answer + self.tokenizer.eos_token
        resp_enc = self.tokenizer(response_str, return_tensors="pt", add_special_tokens=False)
        response_ids = resp_enc["input_ids"][0]
        response_attention_mask = resp_enc["attention_mask"][0]

        prompt_len = prompt_input_ids.shape[0]
        response_len = response_ids.shape[0]

        # Concatenate prompt + response
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=0)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=0)

        # Truncate or pad to max_length
        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
        elif seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

        # Build loss mask: mask prompt tokens, keep response tokens
        loss_mask = attention_mask.clone()
        effective_prompt_len = min(prompt_len, self.max_length)
        if effective_prompt_len > 1:
            loss_mask[:effective_prompt_len - 1] = 0
        effective_total = min(prompt_len + response_len, self.max_length)
        if effective_total > 0:
            loss_mask[effective_total - 1] = 0

        position_ids = compute_position_id_with_mask(attention_mask)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs

        result["dataset_name"] = row.get("dataset", "unknown")
        result["task"] = row.get("task", "unknown")
        result["ground_truth"] = answer
        result["prompt_len"] = prompt_len

        return result
