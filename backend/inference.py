import torch
import torchaudio
import os
try:
    from .model import DenoiseUNet
except ImportError:
    from model import DenoiseUNet

class AudioDenoiseInference:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenoiseUNet().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: Model not found at {model_path}. Using initialized weights (untrained).")
            
        self.model.eval()
        self.target_sr = 16000
        
    def denoise_file(self, input_path, output_path):
        waveform, sr = torchaudio.load(input_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(self.device)
        
        # Pad to multiple of 16 for U-Net
        original_length = waveform.shape[2]
        pad_size = (16 - (original_length % 16)) % 16
        if pad_size > 0:
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            
        with torch.no_grad():
            output_waveform = self.model(waveform)
            
        # Remove padding
        if pad_size > 0:
            output_waveform = output_waveform[:, :, :-pad_size]
            
        # Remove batch dimension
        output_waveform = output_waveform.squeeze(0).cpu()
        
        # Save output
        torchaudio.save(output_path, output_waveform, self.target_sr)
        return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Denoise an audio file")
    
    script_dir = os.path.dirname(os.path.abspath(__name__))
    default_model = os.path.join(script_dir, "models", "denoise_model.pth")
    
    parser.add_argument('--model_path', type=str, default=default_model)
    parser.add_argument('--input', type=str, required=True, help="Input noisy audio file")
    parser.add_argument('--output', type=str, default="output.wav", help="Output cleaned audio file")
    args = parser.parse_args()
    
    infer = AudioDenoiseInference(args.model_path)
    infer.denoise_file(args.input, args.output)
    print(f"Denoised audio saved to {args.output}")
