"""
Audio utilities for loading and preprocessing audio inputs.
Extracted from verl/verl/utils/dataset/audio_utils.py.
"""
from typing import Tuple, Union
import torch
import torchaudio


def process_audio(audio: Union[str, dict], processor=None, max_seconds: float = 10) -> Tuple[torch.Tensor, int]:
    """
    Load audio, convert to mono, resample to the processor's sample rate, and clip to max_seconds.
    Returns (audio_tensor [num_samples], sample_rate).
    """
    if isinstance(audio, dict):
        audio_path = audio.get("audio", audio)
    else:
        audio_path = audio

    try:
        audio_data, original_sr = torchaudio.load(audio_path)

        if processor and hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            target_sr = processor.feature_extractor.sampling_rate
        else:
            target_sr = 16000

        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(original_sr, target_sr)
            audio_data = resampler(audio_data)
        else:
            target_sr = original_sr

        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=False)
        else:
            audio_data = audio_data.squeeze(0)

        if max_seconds:
            max_samples = int(max_seconds * target_sr)
            if audio_data.shape[0] > max_samples:
                print("Clipping audio to max_seconds")
                audio_data = audio_data[:max_samples]

        return audio_data, target_sr

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        dummy_audio = torch.zeros((int(16000 * 0.5),), dtype=torch.float32)
        return dummy_audio, 16000
