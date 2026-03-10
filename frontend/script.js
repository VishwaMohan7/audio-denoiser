document.addEventListener('DOMContentLoaded', () => {
    const audioInput = document.getElementById('audio-input');
    const fileNameDisplay = document.getElementById('file-name');
    const dropZone = document.getElementById('drop-zone');
    const denoiseBtn = document.getElementById('denoise-btn');
    const resultSection = document.getElementById('result-section');
    const noisyPlayer = document.getElementById('noisy-player');
    const cleanPlayer = document.getElementById('clean-player');
    const waveformImg = document.getElementById('waveform-img');
    const spectrogramImg = document.getElementById('spectrogram-img');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');

    let selectedFile = null;

    // Handle File Selection
    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please select a valid audio file.');
            return;
        }
        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        denoiseBtn.disabled = false;

        // Hide result section if it was visible from a previous run
        resultSection.classList.add('hidden');
        waveformImg.classList.add('hidden');
        spectrogramImg.classList.add('hidden');

        // Load noisy audio into the original player for preview
        const url = URL.createObjectURL(file);
        noisyPlayer.src = url;
    }

    // Handle API submission
    denoiseBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State update
        denoiseBtn.disabled = true;
        btnText.textContent = 'Denoising & Plotting...';
        btnLoader.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('audio', selectedFile);

        try {
            const response = await fetch('/api/denoise', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Success
                cleanPlayer.src = data.clean_audio_url;

                // Show visualizations
                waveformImg.src = data.waveform_url;
                spectrogramImg.src = data.spectrogram_url;
                waveformImg.classList.remove('hidden');
                spectrogramImg.classList.remove('hidden');

                resultSection.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Failed to denoise audio');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Restore UI
            denoiseBtn.disabled = false;
            btnText.textContent = 'Denoise Audio';
            btnLoader.classList.add('hidden');
        }
    });
});
