import matplotlib
matplotlib.use('Agg') # Headless backend for Flask
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

def plot_spectrogram(y, sr, title, ax):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    return img

def generate_visualizations(noisy_path, denoised_path, output_dir, req_id):
    """
    Generates and saves waveform and spectrogram comparison images.
    Returns the filenames of the generated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    sr = 16000
    noisy_wav, _ = librosa.load(noisy_path, sr=sr)
    denoised_wav, _ = librosa.load(denoised_path, sr=sr)
    
    # Generate Waveform Comparison
    plt.style.use('dark_background')
    fig_wave, axes_wave = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Noisy Waveform
    librosa.display.waveshow(noisy_wav, sr=sr, color="#ef4444", ax=axes_wave[0])
    axes_wave[0].set_title('Noisy Audio', color='white')
    axes_wave[0].set_ylabel('Amplitude')
    axes_wave[0].tick_params(colors='white')
    
    # Denoised Waveform
    librosa.display.waveshow(denoised_wav, sr=sr, color="#10b981", ax=axes_wave[1])
    axes_wave[1].set_title('Denoised Audio', color='white')
    axes_wave[1].set_xlabel('Time (s)')
    axes_wave[1].set_ylabel('Amplitude')
    axes_wave[1].tick_params(colors='white')
    
    plt.tight_layout()
    waveform_filename = f"{req_id}_waveform.png"
    waveform_path = os.path.join(output_dir, waveform_filename)
    fig_wave.savefig(waveform_path, transparent=True, dpi=150)
    plt.close(fig_wave)
    
    # Generate Spectrogram Comparison
    fig_spec, axes_spec = plt.subplots(1, 2, figsize=(12, 5))
    
    # Noisy Spectrogram
    plot_spectrogram(noisy_wav, sr, 'Noisy Spectrogram', axes_spec[0])
    
    # Denoised Spectrogram
    img = plot_spectrogram(denoised_wav, sr, 'Denoised Spectrogram', axes_spec[1])
    
    plt.tight_layout()
    spectrogram_filename = f"{req_id}_spectrogram.png"
    spectrogram_path = os.path.join(output_dir, spectrogram_filename)
    fig_spec.savefig(spectrogram_path, transparent=True, dpi=150)
    plt.close(fig_spec)
    
    return waveform_filename, spectrogram_filename
