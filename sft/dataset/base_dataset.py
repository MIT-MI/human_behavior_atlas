"""
Standalone base dataset for multi-task classification.
Self-contained replacement for verl's RLHFDataset — no verl dependency.
"""
import copy
import logging
import os
import re
import shutil
import time
import warnings
from typing import Optional

import datasets
import torch
from jinja2 import Template
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from dataset.dataset_utils import postprocess_data, compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _copy_to_local(src: str, cache_dir: Optional[str] = None) -> str:
    """
    For local filesystem paths: return the path as-is.
    For remote paths (hdfs://, s3://): copy to cache_dir and return local path.
    """
    if src.startswith("hdfs://") or src.startswith("s3://"):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/sft_dataset")
        os.makedirs(cache_dir, exist_ok=True)
        local_name = os.path.basename(src)
        local_path = os.path.join(cache_dir, local_name)
        if not os.path.exists(local_path):
            shutil.copy2(src, local_path)
        return local_path
    return src


class BaseDataset(Dataset):
    """
    Standalone dataset that loads JSONL/parquet files, tokenizes with a processor,
    and handles multimodal inputs (images, videos, audio).

    Drop-in replacement for verl's RLHFDataset.
    """

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/sft_dataset"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.audio_key = config.get("audio_key", "audios")
        self.modalities = set(config.get("modalities", "images,videos,audio").split(","))

        self.max_prompt_length = config.get("max_prompt_length", 4096)
        print("WARNING: max_prompt_length is set to", self.max_prompt_length)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.task_filter = config.get("task_filter", None)  # "cls" or "qa" or None (no filter)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        if isinstance(data_files, str):
            self.base_dir = os.path.dirname(os.path.abspath(data_files))
        else:
            self.base_dir = os.path.dirname(os.path.abspath(self.data_files[0]))

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.format_prompt_path = config.get("format_prompt", None)
        self.format_prompt = self._load_format_prompt()

        self._download()
        self._read_files_and_tokenize()

    def _load_format_prompt(self) -> Optional[Template]:
        if self.format_prompt_path:
            print("Loading format prompt from:", self.format_prompt_path)
            with open(self.format_prompt_path, 'r', encoding='utf-8') as f:
                return Template(f.read())
        return None

    def _download(self):
        for i, path in enumerate(self.data_files):
            self.data_files[i] = _copy_to_local(path, self.cache_dir)

    def _read_files_and_tokenize(self):
        features = datasets.Features({
            "problem": datasets.Value("string"),
            "answer": datasets.Value("string"),
            "images": datasets.Sequence(datasets.Value("string")),
            "videos": datasets.Sequence(datasets.Value("string")),
            "audios": datasets.Sequence(datasets.Value("string")),
            "dataset": datasets.Value("string"),
            "task": datasets.Value("string"),
            "class_label": datasets.Value("string"),
            "texts": datasets.Sequence(datasets.Value("string")),
            "modality_signature": datasets.Value("string"),
            "ext_video_feats": datasets.Sequence(datasets.Value("string")),
            "ext_audio_feats": datasets.Sequence(datasets.Value("string")),
        })

        dataframes = []
        for path in self.data_files:
            if path.endswith(".parquet"):
                df = datasets.load_dataset("parquet", data_files=path, features=features)["train"]
            elif path.endswith(".json") or path.endswith(".jsonl"):
                df = datasets.load_dataset("json", data_files=path, features=features)["train"]
            else:
                raise ValueError(f"Unsupported file format: {path}. Only .parquet, .json, .jsonl supported.")
            dataframes.append(df)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

        if self.task_filter is not None:
            suffix = self.task_filter
            self.dataframe = self.dataframe.filter(
                lambda doc: isinstance(doc["task"], str) and
                            doc["task"].rsplit("_", 1)[-1] == suffix,
                desc=f"Filtering to task_filter='{suffix}'",
            )
            print(f"task_filter='{suffix}': {len(self.dataframe)} rows remaining")

        self.dataframe = self._maybe_filter_long_prompts(self.dataframe)

    def _maybe_filter_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe

        from dataset.vision_utils import process_image, process_video
        from dataset.audio_utils import process_audio

        tokenizer = self.tokenizer
        processor = self.processor

        if processor is not None:
            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                processor_kwargs = {"text": [raw_prompt]}
                if "images" in self.modalities and self.image_key in doc and len(doc[self.image_key]) > 0:
                    processor_kwargs["images"] = [process_image(img) for img in doc[self.image_key]]
                if "videos" in self.modalities and self.video_key in doc and len(doc[self.video_key]) > 0:
                    processor_kwargs["videos"] = [process_video(v) for v in doc[self.video_key]]
                if "audio" in self.modalities and doc.get(self.audio_key):
                    audios = []
                    for audio in doc[self.audio_key]:
                        audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                        arr, _ = process_audio(audio_path, processor)
                        audios.append(arr.detach().cpu().numpy().astype("float32"))
                    processor_kwargs["audio"] = audios
                return len(processor(**processor_kwargs)["input_ids"][0])
        else:
            def doc2len(doc) -> int:
                return len(tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True))

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download()
            self._read_files_and_tokenize()
        else:
            print("old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages = example.get(self.prompt_key)
        if isinstance(messages, str):
            messages = [messages]

        has_multimodal = (
            ("images" in self.modalities and self.image_key in example) or
            ("videos" in self.modalities and self.video_key in example) or
            ("audio" in self.modalities and self.audio_key in example)
        )

        if has_multimodal:
            new_messages = []
            for message in messages:
                new_message = copy.deepcopy(message)
                if isinstance(new_message, str):
                    new_message = {"role": "user", "content": new_message}
                content = new_message["content"]

                if self.format_prompt:
                    content = self.format_prompt.render(content=content)

                image_count = len(example.get(self.image_key, []))
                video_count = len(example.get(self.video_key, []))
                audio_count = len(example.get(self.audio_key, []))

                if content.count("<image>") < image_count:
                    content = "<image>" * (image_count - content.count("<image>")) + content
                if content.count("<video>") < video_count:
                    content = "<video>" * (video_count - content.count("<video>")) + content
                if content.count("<audio>") < audio_count:
                    content = "<audio>" * (audio_count - content.count("<audio>")) + content

                tag_patterns = []
                if "images" in self.modalities:
                    tag_patterns.append("<image>")
                if "videos" in self.modalities:
                    tag_patterns.append("<video>")
                if "audio" in self.modalities:
                    tag_patterns.append("<audio>")

                content_list = []
                if tag_patterns:
                    pattern = "(" + "|".join(tag_patterns) + ")"
                    segments = [s for s in re.split(pattern, content) if s != ""]
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
                    content_list.append({"type": "text", "text": content})

                new_message["content"] = content_list
                new_messages.append(new_message)
        else:
            new_messages = copy.deepcopy(messages)
            if isinstance(new_messages, str):
                new_messages = [{"role": "user", "content": new_messages}]
            elif isinstance(new_messages, list) and isinstance(new_messages[0], str):
                new_messages = [{"role": "user", "content": new_messages}]
            if self.format_prompt:
                for i, msg in enumerate(new_messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            new_messages[i]["content"] = self.format_prompt.render(content=content)

        return new_messages

    def __getitem__(self, item):
        from dataset.vision_utils import process_image, process_video
        from dataset.audio_utils import process_audio

        row_dict: dict = self.dataframe[item]

        sig = row_dict.get("modality_signature", None)
        if not isinstance(sig, str) or len(sig.strip()) == 0:
            raise ValueError(
                f"[Dataset] Missing modality_signature for idx={item}. "
                "Preprocess your JSONL with signatures first."
            )

        row_dict["debug_prompts"] = row_dict[self.prompt_key]
        row_dict["data_source"] = "unknown"

        if 'reward_model' not in row_dict:
            answer = row_dict.get('answer') or row_dict.get('ground_truth')
            if answer is None:
                raise ValueError("No answer or ground_truth found in the row_dict.")
            row_dict['reward_model'] = {'ground_truth': answer}

        for key, val in row_dict.items():
            if val is None:
                row_dict[key] = []

        video_rel_path = row_dict.get('ext_video_feats', None)
        if video_rel_path:
            row_dict["ext_video_feats_path"] = os.path.join(self.base_dir, video_rel_path[0])
        else:
            row_dict["ext_video_feats_path"] = None

        audio_rel_path = row_dict.get('ext_audio_feats', None)
        if audio_rel_path:
            row_dict["ext_audio_feats_path"] = os.path.join(self.base_dir, audio_rel_path[0])
        else:
            row_dict["ext_audio_feats_path"] = None

        messages = self._build_messages(row_dict)

        if "audio" in self.modalities:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                )}]
            })

        model_inputs = {}

        if self.processor is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="System prompt modified")
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            multi_modal_data = {}
            processor_kwargs = {"text": [raw_prompt], "return_tensors": "pt"}

            if "images" in self.modalities and self.image_key in row_dict and row_dict.get(self.image_key) and len(row_dict[self.image_key]) > 0:
                images = []
                for img in row_dict[self.image_key]:
                    img = os.path.join(self.base_dir, img) if isinstance(img, str) else img
                    images.append(process_image(img))
                multi_modal_data["image"] = images
                processor_kwargs["images"] = images

            if "videos" in self.modalities and self.video_key in row_dict and row_dict.get(self.video_key) and len(row_dict[self.video_key]) > 0:
                videos = []
                for vid in row_dict[self.video_key]:
                    vid = os.path.join(self.base_dir, vid) if isinstance(vid, str) else vid
                    videos.append(process_video(vid))
                multi_modal_data["video"] = [v.numpy() for v in videos]
                processor_kwargs["videos"] = videos

            if "audio" in self.modalities and self.audio_key in row_dict and row_dict.get(self.audio_key) and len(row_dict[self.audio_key]) > 0:
                audios_np = []
                audios_np_sr = []
                for audio in row_dict[self.audio_key]:
                    audio_path = os.path.join(self.base_dir, audio) if isinstance(audio, str) else audio
                    arr_tensor, sr = process_audio(audio_path, self.processor)
                    arr = arr_tensor.detach().cpu().numpy().astype("float32")
                    audios_np.append(arr)
                    audios_np_sr.append((arr, int(sr)))
                multi_modal_data["audio"] = audios_np_sr
                processor_kwargs["audio"] = audios_np

            model_inputs = self.processor(**processor_kwargs)

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            model_inputs.pop("second_per_grid_ts", None)

            row_dict["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # RoPE position IDs for Qwen2VL; fall back to cumsum-based for other models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            try:
                from verl.models.transformers.qwen2_vl import get_rope_index
                position_ids = [get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )]
            except ImportError:
                position_ids = compute_position_id_with_mask(attention_mask)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-( self.max_prompt_length - left_half):]

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        row_dict["index"] = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict.get("extra_info", {}).get("interaction_kwargs", {})

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()
