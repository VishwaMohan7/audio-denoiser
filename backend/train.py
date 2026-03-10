import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AudioDenoisingDataset
from model import DenoiseUNet

def train_model(clean_dir, noisy_dir, num_epochs=10, batch_size=16, learning_rate=1e-3, save_path="../models/denoise_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Dataset & DataLoader
    dataset = AudioDenoisingDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, target_sr=16000, duration=2.0)
    
    if len(dataset) == 0:
        print("Warning: No files found in the dataset directory.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, Loss, Optimizer
    model = DenoiseUNet().to(device)
    criterion = nn.L1Loss() # L1 Loss often works better for audio waveform
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            
            # Ensure output shape matches clean shape (can differ off by 1 due to conv layers)
            min_length = min(output.shape[2], clean.shape[2])
            output = output[:, :, :min_length]
            clean = clean[:, :, :min_length]
            
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"====> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Audio Denoising Model")
    parser.add_argument('--clean_dir', type=str, required=True, help="Directory containing clean audio files")
    parser.add_argument('--noisy_dir', type=str, required=True, help="Directory containing noisy audio files")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__name__))
    save_target = os.path.join(script_dir, "models", "denoise_model.pth")
    
    train_model(args.clean_dir, args.noisy_dir, num_epochs=args.epochs, batch_size=args.batch_size, save_path=save_target)
