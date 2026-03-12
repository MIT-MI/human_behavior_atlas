# models/bam_utils.py
"""
Utilities for BAM training:
  - Feature extraction from OpenPose (video) and OpenSmile (audio)
  - Temporal pooling and normalization helpers
  - Building BehavioralAdapterModule instances
"""
from typing import Optional, Iterable, Dict, Literal
import torch
import torch.nn.functional as F
import os

from .bam_adapter import BehavioralAdapterModule


# ---------------------------------------------------------------------------
# Temporal pooling and normalization
# ---------------------------------------------------------------------------

PoolMode = Literal["none", "mean", "meanstd", "meanstdp25p75"]

def _pad_trunc_1d(x: torch.Tensor | None, target_dim: int) -> torch.Tensor:
    if x is None:
        print(f"[WARN] _pad_trunc_1d: received None; zero-filling (target_dim={target_dim})")
        return torch.zeros(target_dim)
    D = x.numel()
    if D == target_dim:
        return x
    if D > target_dim:
        return x[:target_dim]
    out = x.new_zeros(target_dim)
    out[:D] = x
    return out

def _maybe_normalize(v: torch.Tensor, norm: str | None) -> torch.Tensor:
    if norm is None or norm == "none":
        return v
    if norm == "l2":
        return v / v.norm(p=2).clamp_min(1e-6)
    if norm == "zscore":
        m, s = v.mean(), v.std(unbiased=False).clamp_min(1e-6)
        return (v - m) / s
    raise ValueError(f"Unknown norm: {norm}")

def pool_temporal(x: torch.Tensor, mode: PoolMode = "meanstd") -> torch.Tensor | None:
    """x: [T, D]. Returns pooled tensor or None if invalid/empty."""
    if x is None:
        print("[WARN] pool_temporal: received None tensor")
        return None
    x = x.float()
    if x.ndim != 2:
        print(f"[WARN] pool_temporal: expected [T, D], got {tuple(x.shape)}")
        return None
    T, D = x.shape
    if T == 0 or D == 0:
        print(f"[WARN] pool_temporal: empty input with shape [T={T}, D={D}]")
        return None
    if mode == "none":
        if T != 1:
            print(f"[WARN] pool_temporal: mode='none' expects T==1, got T={T}")
            return None
        return x.squeeze(0)
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "meanstd":
        return torch.cat([x.mean(dim=0), x.std(dim=0, unbiased=False)], dim=0)
    if mode == "meanstdp25p75":
        m = x.mean(dim=0)
        s = x.std(dim=0, unbiased=False)
        p25 = x.kthvalue(max(1, int(0.25 * T)), dim=0).values
        p75 = x.kthvalue(max(1, int(0.75 * T)), dim=0).values
        return torch.cat([m, s, p25, p75], dim=0)
    print(f"[WARN] pool_temporal: unknown pooling mode: {mode}")
    return None


# ---------------------------------------------------------------------------
# OpenPose video features
# ---------------------------------------------------------------------------

def openpose_dict_to_framewise(data: Dict[str, torch.Tensor] | None, use_conf: bool = True) -> torch.Tensor | None:
    if data is None or not isinstance(data, dict):
        print("[WARN] openpose_to_framewise: input is None or not dict")
        return None
    chunks = []
    for k in ("pose", "face", "left_hand", "right_hand"):
        if k not in data:
            continue
        t = data[k]
        if t is None:
            print(f"[WARN] openpose_to_framewise: part '{k}' is None")
            continue
        if not torch.is_tensor(t):
            try:
                t = torch.as_tensor(t)
            except Exception as e:
                print(f"[WARN] openpose_to_framewise: cannot to_tensor '{k}': {e.__class__.__name__}")
                continue
        t = t.float()
        if t.ndim != 3 or t.shape[-1] not in (2, 3):
            print(f"[WARN] openpose_to_framewise: bad shape for '{k}': {tuple(t.shape)}")
            continue
        if t.shape[0] == 0:
            print(f"[WARN] openpose_to_framewise: empty frames for '{k}'")
            continue
        if not use_conf:
            t = t[..., :2]
        chunks.append(t.reshape(t.shape[0], -1))
    if not chunks:
        print(f"[WARN] openpose_to_framewise: no valid parts | keys={list(data.keys())}")
        return None
    try:
        return torch.cat(chunks, dim=-1)
    except Exception as e:
        print(f"[WARN] openpose_to_framewise: concat failed: {e.__class__.__name__}")
        return None

def build_video_feat_single(openpose: Dict[str, torch.Tensor] | None,
                            temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
                            use_conf: bool = True,
                            norm: Optional[str] = None) -> torch.Tensor | None:
    seq = openpose_dict_to_framewise(openpose, use_conf=use_conf)
    if seq is None:
        return None
    v = pool_temporal(seq, mode=temporal_mode)
    if v is None:
        return None
    return _maybe_normalize(v, norm)

def build_video_feats_batch(openpose_list: Iterable[Dict[str, torch.Tensor] | None],
                            device: Optional[torch.device] = None,
                            temporal_mode: Literal["mean", "meanstd", "meanstdp25p75"] = "meanstd",
                            use_conf: bool = True,
                            norm: Optional[str] = None,
                            target_dim: int = None) -> torch.Tensor | None:
    if target_dim is None:
        print("[WARN] build_video_feats_batch: target_dim not set")
        return None
    vecs: list[torch.Tensor] = []
    for idx, op in enumerate(openpose_list):
        v = build_video_feat_single(op, temporal_mode=temporal_mode, use_conf=use_conf, norm=norm)
        if v is None or v.numel() == 0:
            keys = list(op.keys()) if isinstance(op, dict) else None
            tname = type(op).__name__ if op is not None else "None"
            print(f"[WARN] build_video_feats_batch: invalid sample at idx={idx} | type={tname} keys={keys}")
            return None
        vecs.append(_pad_trunc_1d(v, target_dim))
    try:
        out = torch.stack(vecs, dim=0)
    except Exception as e:
        print(f"[WARN] build_video_feats_batch: stack failed: {e.__class__.__name__} | B={len(vecs)}")
        return None
    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------------------
# OpenSmile audio features
# ---------------------------------------------------------------------------

def opensmile_to_framewise(d: Dict | None) -> torch.Tensor | None:
    if d is None or not isinstance(d, dict):
        print("[WARN] opensmile_to_framewise: input is None or not dict")
        return None
    if "features" not in d:
        print(f"[WARN] opensmile_to_framewise: missing 'features' | keys={list(d.keys())}")
        return None
    x = d["features"]
    if x is None:
        print("[WARN] opensmile_to_framewise: 'features' is None")
        return None
    try:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
    except Exception as e:
        print(f"[WARN] opensmile_to_framewise: tensor convert failed: {e.__class__.__name__}")
        return None
    if x.numel() == 0:
        print("[WARN] opensmile_to_framewise: 'features' empty")
        return None
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        print(f"[WARN] opensmile_to_framewise: bad shape {tuple(x.shape)} (want [T,D] or [D])")
        return None
    return x

def build_audio_feat_single(opensmile_dict: Dict | None,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None) -> torch.Tensor | None:
    seq = opensmile_to_framewise(opensmile_dict)
    if seq is None:
        return None
    v = pool_temporal(seq, mode=temporal_mode)
    if v is None:
        return None
    return _maybe_normalize(v, norm)

def build_audio_feats_batch(opensmile_list: Iterable[Dict | None],
                            device: torch.device | None = None,
                            temporal_mode: PoolMode = "none",
                            norm: str | None = None,
                            target_dim: int | None = None) -> torch.Tensor | None:
    if target_dim is None:
        print("[WARN] build_audio_feats_batch: target_dim not set")
        return None
    vecs: list[torch.Tensor] = []
    for idx, d in enumerate(opensmile_list):
        v = build_audio_feat_single(d, temporal_mode=temporal_mode, norm=norm)
        if v is None or v.numel() == 0:
            keys = list(d.keys()) if isinstance(d, dict) else None
            tname = type(d).__name__ if d is not None else "None"
            print(f"[WARN] build_audio_feats_batch: invalid sample at idx={idx} | type={tname} keys={keys}")
            return None
        vecs.append(_pad_trunc_1d(v, target_dim))
    try:
        out = torch.stack(vecs, dim=0)
    except Exception as e:
        print(f"[WARN] build_audio_feats_batch: stack failed: {e.__class__.__name__} | B={len(vecs)}")
        return None
    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------------------
# Building BehavioralAdapterModule instances
# ---------------------------------------------------------------------------

def maybe_build_hidden_adapters(
    *,
    use_bam_video: bool,
    use_bam_audio: bool,
    bam_hidden_video: int,
    bam_hidden_audio: int,
    p_moddrop_video: float,
    p_moddrop_audio: float,
    out_dim_hidden: int,         # backbone pooled hidden size H
    d_video_feat: Optional[int] = None,
    d_audio_feat: Optional[int] = None,
    video_use_ln: bool = False,
    video_alpha_init: float = 1.0,
    audio_use_ln: bool = False,
    audio_alpha_init: float = 1.0,
):
    """Build hidden-space BAM adapters per modality. Returns (video_adapter, audio_adapter)."""
    video_adapter = None
    audio_adapter = None

    if use_bam_video:
        if d_video_feat is None:
            raise ValueError("USE_BAM_VIDEO=True but d_video_feat was not provided")
        video_adapter = BehavioralAdapterModule(
            feat_dim=int(d_video_feat),
            hidden=int(bam_hidden_video),
            out_dim=int(out_dim_hidden),
            p_moddrop=float(p_moddrop_video),
            use_ln=bool(video_use_ln),
            alpha_init=float(video_alpha_init),
        )

    if use_bam_audio:
        if d_audio_feat is None:
            raise ValueError("USE_BAM_AUDIO=True but d_audio_feat was not provided")
        audio_adapter = BehavioralAdapterModule(
            feat_dim=int(d_audio_feat),
            hidden=int(bam_hidden_audio),
            out_dim=int(out_dim_hidden),
            p_moddrop=float(p_moddrop_audio),
            use_ln=bool(audio_use_ln),
            alpha_init=float(audio_alpha_init),
        )

    return video_adapter, audio_adapter
