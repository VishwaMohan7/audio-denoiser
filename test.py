import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AudioDenoisingDataset, SpectrogramStats, build_paired_file_list
from model import UNetDenoiser
from utils.audio import (
    load_audio,
    save_waveform,
    spectrogram_to_waveform_with_phase,
    waveform_to_stft,
)
from utils.visualization import plot_spectrogram_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the audio denoising model.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model-path", type=str, default="models/best_denoiser.pth")
    parser.add_argument("--stats-path", type=str, default="models/normalization_stats.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(spec: np.ndarray, stats: SpectrogramStats) -> np.ndarray:
    return spec * (stats.std + 1e-8) + stats.mean


def load_stats(path: str) -> SpectrogramStats:
    with open(path, "r", encoding="utf-8") as handle:
        return SpectrogramStats.from_dict(json.load(handle))


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    clean_test_dir = os.path.join(args.data_root, "clean_testset_wav")
    noisy_test_dir = os.path.join(args.data_root, "noisy_testset_wav")
    test_pairs = build_paired_file_list(clean_test_dir, noisy_test_dir)
    print(f"Test pairs: {len(test_pairs)}")

    stats = load_stats(args.stats_path)
    test_dataset = AudioDenoisingDataset(
        test_pairs,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        normalize_stats=stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNetDenoiser().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    output_audio_dir = os.path.join(args.output_dir, "audio")
    output_plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    total_loss = 0.0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, start=1):
            file_name = batch["file_name"][0]
            noisy_path = batch["noisy_path"][0]
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            prediction = model(noisy)
            loss = mse_loss(prediction, clean) + l1_loss(prediction, clean)
            total_loss += loss.item()

            noisy_spec = batch["noisy"][0, 0].cpu().numpy()
            clean_spec = batch["clean"][0, 0].cpu().numpy()
            pred_spec = prediction[0, 0].cpu().numpy()

            noisy_spec = denormalize(noisy_spec, stats)
            clean_spec = denormalize(clean_spec, stats)
            pred_spec = denormalize(pred_spec, stats)

            noisy_waveform = load_audio(
                noisy_path,
                target_sr=args.sample_rate,
                target_seconds=args.segment_seconds,
            )
            noisy_stft = waveform_to_stft(
                noisy_waveform,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
            )

            waveform = spectrogram_to_waveform_with_phase(
                pred_spec,
                phase_reference_stft=noisy_stft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                length=int(args.sample_rate * args.segment_seconds),
            )
            save_waveform(
                os.path.join(output_audio_dir, file_name),
                waveform,
                sample_rate=args.sample_rate,
            )
            plot_spectrogram_triplet(
                noisy_spec=noisy_spec,
                clean_spec=clean_spec,
                predicted_spec=pred_spec,
                save_path=os.path.join(
                    output_plot_dir, f"{os.path.splitext(file_name)[0]}_comparison.png"
                ),
                title_prefix=f"{file_name} - ",
            )

            print(
                f"Processed {batch_index}/{len(test_loader)} | file={file_name} "
                f"| loss={loss.item():.6f}"
            )

    avg_loss = total_loss / max(len(test_loader), 1)
    print(f"\nAverage test loss: {avg_loss:.6f}")
    print(f"Denoised audio saved to: {output_audio_dir}")
    print(f"Spectrogram plots saved to: {output_plot_dir}")


if __name__ == "__main__":
    main()
