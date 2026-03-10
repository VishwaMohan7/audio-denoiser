import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AudioDenoisingDataset
from model import DenoiseUNet
import math

def calculate_snr(clean, denoised):
    # Calculate SNR (Signal-to-Noise Ratio) in dB
    noise = clean - denoised
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power.item() == 0:
        return 100.0 # Return arbitrarily high SNR if perfect match
        
    snr = 10 * math.log10(signal_power.item() / noise_power.item())
    return snr

def evaluate_model(clean_dir, noisy_dir, model_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = AudioDenoisingDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, target_sr=16000, duration=2.0)
    
    if len(dataset) == 0:
        print("Warning: No files found in the dataset directory.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load Model
    model = DenoiseUNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {model_path}.")
        return

    model.eval()
    criterion = nn.L1Loss()
    
    total_loss = 0.0
    total_snr = 0.0
    num_samples = 0

    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            output = model(noisy)
            
            # Match lengths (in case of slight dimension shifting in U-Net)
            min_length = min(output.shape[2], clean.shape[2])
            output = output[:, :, :min_length]
            clean = clean[:, :, :min_length]
            
            loss = criterion(output, clean)
            total_loss += loss.item() * noisy.size(0)
            
            # Calculate average SNR for the batch
            batch_snr = calculate_snr(clean, output)
            total_snr += batch_snr * noisy.size(0)
            num_samples += noisy.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Evaluating Step [{batch_idx}/{len(dataloader)}]...")
            
    avg_loss = total_loss / num_samples
    avg_snr = total_snr / num_samples
    
    print("\n==== Evaluation Results ====")
    print(f"Average L1 Loss: {avg_loss:.4f}")
    print(f"Average SNR: {avg_snr:.2f} dB")
    print("============================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Audio Denoising Model Accuracy (Loss/SNR)")
    
    script_dir = os.path.dirname(os.path.abspath(__name__))
    default_model = os.path.join(script_dir, "models", "denoise_model.pth")
    
    parser.add_argument('--clean_dir', type=str, required=True, help="Directory containing test clean audio files")
    parser.add_argument('--noisy_dir', type=str, required=True, help="Directory containing test noisy audio files")
    parser.add_argument('--model_path', type=str, default=default_model, help="Path to the trained PyTorch model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    evaluate_model(args.clean_dir, args.noisy_dir, args.model_path, args.batch_size)
