# Audio Denoising AI Web Application

A full-stack deep learning application to clean up noisy audio files. Built with PyTorch, Flask, and Vanilla Web Technologies (HTML/CSS/JS).

## Project Features
- **PyTorch backend**: Uses a 1D U-Net CNN for audio waveform denoising.
- **Efficient Dataset Handling**: Custom lazy-loading Dataset to train on massive >20GB datasets without RAM issues.
- **Flask REST API**: A lightweight robust backend for handling audio processing requests.
- **Premium UI**: Glassmorphism UI built with vanilla CSS.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- (Optional but recommended) CUDA-compatible GPU

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training the Model
To train the model on your own dataset (noisy and clean paired WAV files):
```bash
python backend/train.py --clean_dir /path/to/clean_audio --noisy_dir /path/to/noisy_audio --epochs 10 --batch_size 16
```
The trained model will be saved to `models/denoise_model.pth`.

### 4. Running the Web App locally
To start the Flask server:
```bash
python app.py
```
After starting the server, go to [http://localhost:5000](http://localhost:5000) in your web browser.

## File Structure Description
- `backend/dataset.py`: PyTorch data loaders optimizing for huge dataset processing. Resamples and standardizes chunks.
- `backend/model.py`: PyTorch Deep Learning model implementation (1D U-Net).
- `backend/train.py`: Training loop script.
- `backend/inference.py`: Model inference class, handles padding/predicting on single files.
- `app.py`: Flask Web Server and routing logic.
- `frontend/`: Web app UI assets.