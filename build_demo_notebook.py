import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Text Intro
intro = """\
# Deep Learning Audio Denoising Demonstration

This notebook demonstrates the results of the 1D U-Net audio denoising model. 
It loads a **clean**, **noisy**, and the model-generated **denoised** audio sample to visualize the differences using waveforms and spectrograms. It also provides audio playback!
"""

# Import Cell
imports = """\
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import torch
import torchaudio

# Load local inference module
from backend.inference import AudioDenoiseInference

# Initialize Inference Model
# (Ensure you run this terminal command first to train if you haven't: python backend/train.py ...)
model_infer = AudioDenoiseInference('models/denoise_model.pth')
"""

# Load Data Cell
load_data = """\
# Paths to your test files (Change these if needed)
noisy_audio_path = 'dataset/noisy_testset_wav/noisy_testset_wav/p232_001.wav'
clean_audio_path = 'dataset/clean_testset_wav/clean_testset_wav/p232_001.wav'
denoised_audio_path = 'demo_denoised_output.wav'

# Run Inference to generate the denoised file
print("Denoising the audio... Please wait.")
model_infer.denoise_file(noisy_audio_path, denoised_audio_path)
print("Denoising complete!")

# Load Audio using Librosa for visualization 
# (sampling rate = 16000 as used by the model)
sr = 16000
clean_wav, _ = librosa.load(clean_audio_path, sr=sr)
noisy_wav, _ = librosa.load(noisy_audio_path, sr=sr)
denoised_wav, _ = librosa.load(denoised_audio_path, sr=sr)
"""

# Waveform comparison
waveform_cell = """\
# 1. Waveform Comparison
plt.figure(figsize=(15, 8))

# Clean
plt.subplot(3, 1, 1)
librosa.display.waveshow(clean_wav, sr=sr, color="blue")
plt.title('Clean Audio Waveform')
plt.xlabel('')
plt.ylabel('Amplitude')

# Noisy
plt.subplot(3, 1, 2)
librosa.display.waveshow(noisy_wav, sr=sr, color="red")
plt.title('Noisy Audio Waveform')
plt.xlabel('')
plt.ylabel('Amplitude')

# Denoised
plt.subplot(3, 1, 3)
librosa.display.waveshow(denoised_wav, sr=sr, color="green")
plt.title('Denoised Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
"""

# Spectrogram Comparison
spectrogram_cell = """\
# 2. Spectrogram Comparison
def plot_spectrogram(y, sr, title, ax):
    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # Convert amplitude to decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title)
    return img

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot all three
plot_spectrogram(clean_wav, sr, 'Clean Audio Spectrogram', axes[0])
plot_spectrogram(noisy_wav, sr, 'Noisy Audio Spectrogram', axes[1])
img = plot_spectrogram(denoised_wav, sr, 'Denoised Audio Spectrogram', axes[2])

# Add a colorbar
fig.colorbar(img, ax=axes, format="%+2.f dB")
plt.show()
"""

# Audio Playback
playback_cell = """\
# 3. Audio Playback
print("Noisy Audio:")
display(ipd.Audio(noisy_audio_path))

print("\\nDenoised Audio:")
display(ipd.Audio(denoised_audio_path))

print("\\nClean Audio (Ground Truth):")
display(ipd.Audio(clean_audio_path))
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(intro),
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_code_cell(load_data),
    nbf.v4.new_code_cell(waveform_cell),
    nbf.v4.new_code_cell(spectrogram_cell),
    nbf.v4.new_code_cell(playback_cell)
]

with open('audio_demo.ipynb', 'w') as f:
    nbf.write(nb, f)

print("audio_demo.ipynb created successfully!")
