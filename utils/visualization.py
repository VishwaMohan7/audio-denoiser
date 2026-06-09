import os

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram_triplet(
    noisy_spec: np.ndarray,
    clean_spec: np.ndarray,
    predicted_spec: np.ndarray,
    save_path: str,
    title_prefix: str = "",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [noisy_spec, clean_spec, predicted_spec]
    titles = ["Noisy", "Clean", "Predicted"]

    for axis, spec, title in zip(axes, specs, titles):
        image = axis.imshow(spec, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(f"{title_prefix}{title}")
        axis.set_xlabel("Frames")
        axis.set_ylabel("Frequency Bins")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
