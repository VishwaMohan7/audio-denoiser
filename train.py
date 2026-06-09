import argparse
import json
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import (
    AudioDenoisingDataset,
    SpectrogramStats,
    build_paired_file_list,
    compute_dataset_stats,
    split_pairs,
)
from model import UNetDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net audio denoising model.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-noise-prob", type=float, default=0.2)
    parser.add_argument("--augment-noise-scale", type=float, default=0.005)
    parser.add_argument("--stats-max-items", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--best-model-path", type=str, default="models/best_denoiser.pth")
    parser.add_argument("--stats-path", type=str, default="models/normalization_stats.json")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint created by this script to continue training from.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mse_loss: nn.Module,
    l1_loss: nn.Module,
) -> torch.Tensor:
    return mse_loss(prediction, target) + l1_loss(prediction, target)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    mse_loss: nn.Module,
    l1_loss: nn.Module,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0

    for batch_index, batch in enumerate(loader, start=1):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        with torch.set_grad_enabled(train):
            prediction = model(noisy)
            loss = combined_loss(prediction, clean, mse_loss, l1_loss)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        print(
            f"    [{'train' if train else 'val'}] batch {batch_index}/{len(loader)} "
            f"loss={loss.item():.6f}"
        )

    return total_loss / max(len(loader), 1)


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_or_compute_stats(args: argparse.Namespace, train_pairs) -> SpectrogramStats:
    if os.path.exists(args.stats_path):
        with open(args.stats_path, "r", encoding="utf-8") as handle:
            return SpectrogramStats.from_dict(json.load(handle))

    stats = compute_dataset_stats(
        pairs=train_pairs,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_items=args.stats_max_items,
    )
    save_json(args.stats_path, stats.to_dict())
    return stats


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    clean_train_dir = os.path.join(args.data_root, "clean_trainset_28spk_wav")
    noisy_train_dir = os.path.join(args.data_root, "noisy_trainset_28spk_wav")

    all_pairs = build_paired_file_list(clean_train_dir, noisy_train_dir)
    train_pairs, val_pairs = split_pairs(
        all_pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Paired training files: {len(all_pairs)}")
    print(f"Train split: {len(train_pairs)}")
    print(f"Validation split: {len(val_pairs)}")

    stats = load_or_compute_stats(args, train_pairs)
    print(f"Normalization mean={stats.mean:.6f}, std={stats.std:.6f}")

    common_dataset_kwargs = dict(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        normalize_stats=stats,
    )

    train_dataset = AudioDenoisingDataset(
        train_pairs,
        augment_noise_prob=args.augment_noise_prob,
        augment_noise_scale=args.augment_noise_scale,
        **common_dataset_kwargs,
    )
    val_dataset = AudioDenoisingDataset(val_pairs, **common_dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )


def maybe_resume_training(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float]:
    resume_path = args.resume
    if not resume_path:
        auto_resume_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(auto_resume_path):
            resume_path = auto_resume_path
            print(f"Found latest checkpoint automatically: {resume_path}")

    if not resume_path:
        return 1, float("inf")

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    completed_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("val_loss", float("inf")))
    start_epoch = completed_epoch + 1

    print(
        f"Resuming from checkpoint: {resume_path} | "
        f"completed_epoch={completed_epoch} | next_epoch={start_epoch} | "
        f"best_seen_val_loss={best_val_loss:.6f}"
    )
    return start_epoch, best_val_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")
    train_loader, val_loader = build_dataloaders(args)

    model = UNetDenoiser().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    start_epoch, best_val_loss = maybe_resume_training(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    if start_epoch > args.epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is beyond requested --epochs {args.epochs}. Nothing to do."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | lr={current_lr:.6f}")

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            train=True,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            train=False,
        )

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch} complete | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f}"
        )

        latest_checkpoint = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
        save_checkpoint(latest_checkpoint, model, optimizer, scheduler, epoch, val_loss)

        epoch_checkpoint = os.path.join(
            args.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth"
        )
        save_checkpoint(epoch_checkpoint, model, optimizer, scheduler, epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                args.best_model_path,
            )
            print(f"Saved new best model to {args.best_model_path}")

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
