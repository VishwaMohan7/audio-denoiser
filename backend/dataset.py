import os
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, target_sr=16000, duration=2.0):
        """
        Args:
            clean_dir (str): Path to clean audio files.
            noisy_dir (str): Path to noisy audio files.
            target_sr (int): Target sampling rate.
            duration (float): Target duration in seconds to pad/truncate.
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration)
        
        # We assume clean and noisy files have the same filenames for pairing
        if os.path.exists(clean_dir):
            self.filenames = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
        else:
            self.filenames = []
        
    def __len__(self):
        return len(self.filenames)
        
    def _process_audio(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            
        # Pad or truncate
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        return waveform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)
        
        clean_audio = self._process_audio(clean_path)
        noisy_audio = self._process_audio(noisy_path)
        
        return noisy_audio, clean_audio
