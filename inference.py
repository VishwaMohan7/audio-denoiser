import argparse
import json
import os

import numpy as np
import torch

from dataset import SpectrogramStats
from model import UNetDenoiser
from utils.audio import (
    load_audio,
    save_waveform,
    spectrogram_to_waveform_with_phase,
    waveform_to_stft,
    waveform_to_spectrogram,
)
from utils.visualization import plot_spectrogram_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise a single audio file.")
    parser.add_argument("--input", type=str, required=True, help="Path to noisy input wav file.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference/denoised.wav",
        help="Where to save the denoised waveform.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="outputs/inference/spectrogram_comparison.png",
        help="Where to save the spectrogram visualization.",
    )
    parser.add_argument("--model-path", type=str, default="models/best_denoiser.pth")
    parser.add_argument("--stats-path", type=str, default="models/normalization_stats.json")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="Chunk size used for inference. The full audio is processed chunk by chunk.",
    )
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_stats(path: str) -> SpectrogramStats:
    with open(path, "r", encoding="utf-8") as handle:
        return SpectrogramStats.from_dict(json.load(handle))


def normalize(spec: np.ndarray, stats: SpectrogramStats) -> np.ndarray:
    return (spec - stats.mean) / (stats.std + 1e-8)


def denormalize(spec: np.ndarray, stats: SpectrogramStats) -> np.ndarray:
    return spec * (stats.std + 1e-8) + stats.mean


def split_waveform_into_chunks(waveform: np.ndarray, chunk_length: int) -> tuple[list[np.ndarray], int]:
    original_length = len(waveform)
    chunks: list[np.ndarray] = []

    for start in range(0, original_length, chunk_length):
        chunk = waveform[start:start + chunk_length]
        if len(chunk) < chunk_length:
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
        chunks.append(chunk.astype(np.float32))

    if not chunks:
        chunks.append(np.zeros(chunk_length, dtype=np.float32))

    return chunks, original_length


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    stats = load_stats(args.stats_path)

    waveform = load_audio(
        args.input,
        target_sr=args.sample_rate,
        target_seconds=None,
    )
    chunk_length = int(args.sample_rate * args.segment_seconds)
    chunks, original_length = split_waveform_into_chunks(waveform, chunk_length)

    model = UNetDenoiser().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    denoised_chunks: list[np.ndarray] = []
    first_noisy_spec = None
    first_predicted_spec = None

    with torch.no_grad():
        for chunk_index, chunk in enumerate(chunks, start=1):
            noisy_stft = waveform_to_stft(
                chunk,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
            )
            noisy_spec = waveform_to_spectrogram(
                chunk,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
            )
            noisy_spec_normalized = normalize(noisy_spec, stats)
            noisy_tensor = (
                torch.from_numpy(noisy_spec_normalized)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            predicted_tensor = model(noisy_tensor)
            predicted_spec = predicted_tensor[0, 0].cpu().numpy()
            predicted_spec = denormalize(predicted_spec, stats)

            denoised_chunk = spectrogram_to_waveform_with_phase(
                predicted_spec,
                phase_reference_stft=noisy_stft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=chunk_length,
            )
            denoised_chunks.append(denoised_chunk)

            if chunk_index == 1:
                first_noisy_spec = noisy_spec
                first_predicted_spec = predicted_spec

            print(f"Processed chunk {chunk_index}/{len(chunks)}")

    denoised_waveform = np.concatenate(denoised_chunks)[:original_length]

    save_waveform(args.output, denoised_waveform, sample_rate=args.sample_rate)
    if first_noisy_spec is not None and first_predicted_spec is not None:
        plot_spectrogram_triplet(
            noisy_spec=first_noisy_spec,
            clean_spec=first_predicted_spec,
            predicted_spec=first_predicted_spec,
            save_path=args.plot_path,
            title_prefix="Inference (first chunk) - ",
        )

    print(f"Denoised audio saved to: {args.output}")
    print(f"Spectrogram plot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()
