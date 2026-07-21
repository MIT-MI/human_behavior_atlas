import os
import torch
from torch.utils.data import BatchSampler

from dataset.base_dataset import BaseDataset


def _resolve_single_path(maybe_path):
    """Return the first existing path from a string or list of strings; else None."""
    if maybe_path is None:
        return None
    if isinstance(maybe_path, (list, tuple)):
        for p in maybe_path:
            if isinstance(p, str) and os.path.exists(p):
                return p
        return None
    if isinstance(maybe_path, str) and os.path.exists(maybe_path):
        return maybe_path
    return None


def load_feat_or_none(path, kind: str):
    """Try to load a torch .pt feature file. Returns None on failure."""
    resolved = _resolve_single_path(path)
    if resolved is None:
        return None
    try:
        return torch.load(resolved, map_location="cpu")
    except Exception:
        return None


class OmniClassifierDataset(BaseDataset):
    """
    Extends BaseDataset with classification label mapping and
    external video/audio feature loading (.pt files).

    Optionally supports mixed QA + classification via `qa_datasets`: rows
    whose dataset name appears in that set are treated as language-modeling
    tasks (answer stored as `lm_labels`, `labels` set to 0). All other rows
    go through the normal label-map classification path.
    """

    def __init__(self, *args, label_key='answer', label_map=None, dataset_key='dataset', qa_datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key
        self.label_map = label_map
        self.dataset_key = dataset_key
        self.qa_datasets = set([d.lower() for d in (qa_datasets or [])])

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)

        video_feats_path = row_dict.get('ext_video_feats_path', row_dict.get('ext_video_feats', None))
        audio_feats_path = row_dict.get('ext_audio_feats_path', row_dict.get('ext_audio_feats', None))
        row_dict['video_feats'] = load_feat_or_none(video_feats_path, kind="video")
        row_dict['audio_feats'] = load_feat_or_none(audio_feats_path, kind="audio")

        original_answer = row_dict.get(self.label_key, "").lower()
        dataset_name = row_dict.get(self.dataset_key, "").lower()
        task = str(row_dict.get("task", "")).lower()

        # QA rows: determined by task suffix (_qa) or explicit qa_datasets set
        if task.endswith("_qa") or dataset_name in self.qa_datasets:
            row_dict["lm_labels"] = original_answer
            row_dict["labels"] = torch.tensor(0, dtype=torch.long)
            return row_dict

        # Classification rows: map answer string to class index
        full_label_key = f"{dataset_name}_{original_answer}" if dataset_name else original_answer

        if self.label_map is None:
            raise ValueError(f"label_map must be provided for mapping raw labels '{full_label_key}' to class indices")

        label_map = {k.lower(): v for k, v in self.label_map.items()}
        label = label_map.get(full_label_key, 0)
        if label == 0 and full_label_key not in label_map:
            raise ValueError(f"Label key '{full_label_key}' not found in label map.")

        row_dict["labels"] = torch.tensor(label, dtype=torch.long)
        row_dict.setdefault("lm_labels", "")
        return row_dict


class SkipBatchSampler(torch.utils.data.Sampler):
    """
    Wraps an existing BatchSampler and skips the first `skip_batches` batches.
    Advances sampler indices without calling __getitem__ — safe for checkpoint resumption.
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int):
        self.batch_sampler = batch_sampler
        self.skip_batches = int(max(0, skip_batches))
        self.batch_size = getattr(batch_sampler, 'batch_size', None)
        self.drop_last = getattr(batch_sampler, 'drop_last', False)

    def __iter__(self):
        it = iter(self.batch_sampler)
        for _ in range(self.skip_batches):
            try:
                next(it)
            except StopIteration:
                return
        yield from it

    def __len__(self):
        try:
            return max(0, len(self.batch_sampler) - self.skip_batches)
        except TypeError:
            return 0
