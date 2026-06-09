import os
import wave
from typing import Optional

import librosa
import numpy as np


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    target_seconds: Optional[float] = 2.0,
) -> np.ndarray:
    waveform, _ = librosa.load(file_path, sr=target_sr, mono=True)
    waveform = waveform.astype(np.float32)

    if target_seconds is not None:
        target_length = int(target_sr * target_seconds)
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[:target_length]

    return waveform


def waveform_to_spectrogram(
    waveform: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> np.ndarray:
    stft = librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    magnitude = np.abs(stft)
    return np.log1p(magnitude).astype(np.float32)


def waveform_to_stft(
    waveform: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> np.ndarray:
    return librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )


def spectrogram_to_waveform(
    log_magnitude: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    length: Optional[int] = None,
    griffin_lim_iters: int = 32,
) -> np.ndarray:
    magnitude = np.expm1(np.maximum(log_magnitude, 0.0))
    waveform = librosa.griffinlim(
        magnitude,
        n_iter=griffin_lim_iters,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        length=length,
    )
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


def spectrogram_to_waveform_with_phase(
    log_magnitude: np.ndarray,
    phase_reference_stft: np.ndarray,
    hop_length: int = 128,
    win_length: int = 512,
    length: Optional[int] = None,
) -> np.ndarray:
    magnitude = np.expm1(log_magnitude).astype(np.float32)
    phase = np.angle(phase_reference_stft)
    complex_stft = magnitude * np.exp(1j * phase)
    waveform = librosa.istft(
        complex_stft,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
    )
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


def save_waveform(file_path: str, waveform: np.ndarray, sample_rate: int = 16000) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    pcm = np.int16(np.clip(waveform, -1.0, 1.0) * 32767.0)
    with wave.open(file_path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
