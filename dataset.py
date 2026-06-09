import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.audio import load_audio, waveform_to_spectrogram


def _resolve_audio_dir(path: str) -> str:
    entries = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    if len(entries) == 1:
        inner = os.path.join(path, entries[0])
        wav_files = [name for name in os.listdir(inner) if name.lower().endswith(".wav")]
        if wav_files:
            return inner
    return path


def _list_wav_map(path: str) -> Dict[str, str]:
    audio_dir = _resolve_audio_dir(path)
    wav_map: Dict[str, str] = {}
    for name in sorted(os.listdir(audio_dir)):
        if name.lower().endswith(".wav"):
            wav_map[name] = os.path.join(audio_dir, name)
    return wav_map


def build_paired_file_list(clean_dir: str, noisy_dir: str) -> List[Tuple[str, str, str]]:
    clean_map = _list_wav_map(clean_dir)
    noisy_map = _list_wav_map(noisy_dir)
    common_names = sorted(set(clean_map.keys()) & set(noisy_map.keys()))

    if not common_names:
        raise RuntimeError(
            f"No paired wav files found between '{clean_dir}' and '{noisy_dir}'."
        )

    return [(name, noisy_map[name], clean_map[name]) for name in common_names]


def split_pairs(
    pairs: Sequence[Tuple[str, str, str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_size = max(1, int(len(items) * val_ratio))
    val_pairs = items[:val_size]
    train_pairs = items[val_size:]
    return train_pairs, val_pairs


@dataclass
class SpectrogramStats:
    mean: float
    std: float

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "SpectrogramStats":
        return cls(mean=float(values["mean"]), std=float(values["std"]))


def compute_dataset_stats(
    pairs: Sequence[Tuple[str, str, str]],
    sample_rate: int,
    segment_seconds: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    max_items: Optional[int] = None,
) -> SpectrogramStats:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    subset = list(pairs[:max_items]) if max_items is not None else list(pairs)
    for _, noisy_path, _ in subset:
        waveform = load_audio(
            noisy_path,
            target_sr=sample_rate,
            target_seconds=segment_seconds,
        )
        spec = waveform_to_spectrogram(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        total_sum += float(spec.sum())
        total_sq_sum += float(np.square(spec).sum())
        total_count += int(spec.size)

    mean = total_sum / max(total_count, 1)
    variance = max(total_sq_sum / max(total_count, 1) - mean ** 2, 1e-8)
    return SpectrogramStats(mean=mean, std=float(np.sqrt(variance)))


class AudioDenoisingDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str, str]],
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        normalize_stats: Optional[SpectrogramStats] = None,
        augment_noise_prob: float = 0.0,
        augment_noise_scale: float = 0.01,
    ) -> None:
        self.pairs = list(pairs)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalize_stats = normalize_stats
        self.augment_noise_prob = augment_noise_prob
        self.augment_noise_scale = augment_noise_scale

        if not self.pairs:
            raise ValueError("AudioDenoisingDataset received no file pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _normalize(self, spec: np.ndarray) -> np.ndarray:
        if self.normalize_stats is None:
            mean = float(spec.mean())
            std = float(spec.std() + 1e-8)
            return (spec - mean) / std

        return (spec - self.normalize_stats.mean) / (self.normalize_stats.std + 1e-8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_name, noisy_path, clean_path = self.pairs[index]

        noisy_waveform = load_audio(
            noisy_path,
            target_sr=self.sample_rate,
            target_seconds=self.segment_seconds,
        )
        clean_waveform = load_audio(
            clean_path,
            target_sr=self.sample_rate,
            target_seconds=self.segment_seconds,
        )

        if self.augment_noise_prob > 0.0 and random.random() < self.augment_noise_prob:
            noisy_waveform = noisy_waveform + np.random.normal(
                loc=0.0,
                scale=self.augment_noise_scale,
                size=noisy_waveform.shape,
            ).astype(np.float32)
            noisy_waveform = np.clip(noisy_waveform, -1.0, 1.0)

        noisy_spec = waveform_to_spectrogram(
            noisy_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        clean_spec = waveform_to_spectrogram(
            clean_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        noisy_spec = self._normalize(noisy_spec)
        clean_spec = self._normalize(clean_spec)

        return {
            "file_name": file_name,
            "noisy_path": noisy_path,
            "clean_path": clean_path,
            "noisy": torch.from_numpy(noisy_spec).unsqueeze(0).float(),
            "clean": torch.from_numpy(clean_spec).unsqueeze(0).float(),
        }
