# Audio Denoising with PyTorch

This project trains a U-Net based speech denoising model on paired noisy and clean audio. It converts waveforms to log-magnitude STFT spectrograms, learns to predict clean spectrograms from noisy inputs, and reconstructs waveforms with Griffin-Lim during evaluation.

## Project Structure

```text
data/
models/
  checkpoints/
outputs/
utils/
  audio.py
  visualization.py
dataset.py
inference.py
model.py
train.py
test.py
README.md
```

## Features

- Paired noisy/clean dataset loading with filename matching
- Resampling to 16 kHz with fixed-length trimming or padding
- STFT magnitude extraction with `log1p` scaling
- Spectrogram normalization using training-set statistics
- U-Net based CNN with encoder, decoder, and skip connections
- Combined `MSE + L1` loss
- Adam optimizer with learning rate scheduler
- Checkpoint saving and best-model tracking
- Griffin-Lim waveform reconstruction for predicted spectrograms
- Output audio export and spectrogram comparison plots
- Optional extra noise augmentation during training

## Expected Dataset Layout

Place the dataset inside [`data/`](D:/temps/dl/data) with these folders:

```text
data/
  clean_trainset_28spk_wav/
  noisy_trainset_28spk_wav/
  clean_testset_wav/
  noisy_testset_wav/
  trainset_28spk_txt/
  testset_txt/
```

The current code pairs files by matching `.wav` filenames between the clean and noisy folders.

## Install Dependencies

The project uses Python 3.10+ and these libraries:

```bash
pip install torch librosa numpy matplotlib
```

## Step-by-Step Run Instructions

1. Open a terminal in [`D:\temps\dl`](D:/temps/dl).
2. Install dependencies:

   ```bash
   pip install torch librosa numpy matplotlib
   ```

3. Start training:

   ```bash
   python train.py --data-root data --epochs 20 --batch-size 8 --learning-rate 0.001
   ```

4. After training completes, the best model will be saved to:

   ```text
   models/best_denoiser.pth
   ```

5. Run evaluation and generate denoised outputs:

   ```bash
   python test.py --data-root data --model-path models/best_denoiser.pth
   ```

6. Check the generated files:

   ```text
   outputs/audio/
   outputs/plots/
   ```

7. Denoise a single custom noisy file:

   ```bash
   python inference.py --input path\\to\\noisy.wav --output outputs\\inference\\denoised.wav
   ```

## Training Notes

- Default sample rate: `16000`
- Default segment length: `2.0` seconds
- Default STFT config: `n_fft=512`, `hop_length=128`, `win_length=512`
- Validation split is created from the training set with `val_ratio=0.1`
- Best normalization statistics are saved in `models/normalization_stats.json`
- Checkpoints are saved in `models/checkpoints/`

## Example Training Command

```bash
python train.py ^
  --data-root data ^
  --epochs 20 ^
  --batch-size 8 ^
  --learning-rate 0.001 ^
  --augment-noise-prob 0.2 ^
  --augment-noise-scale 0.005
```

## Example Test Command

```bash
python test.py ^
  --data-root data ^
  --model-path models/best_denoiser.pth ^
  --griffin-lim-iters 32
```

## Outputs

- Trained model: `models/best_denoiser.pth`
- Latest and per-epoch checkpoints: `models/checkpoints/`
- Denoised audio: `outputs/audio/`
- Spectrogram comparison plots: `outputs/plots/`
- Single-file inference output: `outputs/inference/`

## Important Notes

- `librosa` and `matplotlib` must be installed before training or testing.
- Griffin-Lim reconstructs audio from magnitude-only spectrograms, so output quality can be improved later by moving to phase-aware methods.
- The provided dataset also contains 56-speaker folders, but this project is configured around the requested 28-speaker training set.
