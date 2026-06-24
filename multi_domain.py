import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img) / 255.0

def fft_log_magnitude(rgb):
    """Compute the log-magnitude FFT spectrum, averaged across channels."""
    gray = rgb.mean(axis=-1)
    f = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f)
    log_mag = np.log1p(np.abs(f_shifted))
    return log_mag

def differentiable_lbp(rgb):
    """
    Mirror your model's LBP computation. The standard approach:
    convert to grayscale, then for each pixel compare to its 3x3 local average.
    Replace this with whatever your model.py actually does.
    """
    gray = rgb.mean(axis=-1)
    # Local average via 3x3 mean filter
    from scipy.ndimage import uniform_filter
    local_avg = uniform_filter(gray, size=3)
    lbp = (gray > local_avg).astype(np.float32)
    return lbp

real_img = load_image("D:/sweet/binary_deepfake_detection/data/celeba2.jpg")
fake_img = load_image("D:/sweet/binary_deepfake_detection/data/stylegan_celeba2.png")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, (img, label) in enumerate([(real_img, "Real"), (fake_img, "Fake")]):
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"{label} — RGB", fontweight="bold")
    axes[row, 1].imshow(fft_log_magnitude(img), cmap="viridis")
    axes[row, 1].set_title(f"{label} — FFT log-magnitude", fontweight="bold")
    axes[row, 2].imshow(differentiable_lbp(img), cmap="gray")
    axes[row, 2].set_title(f"{label} — LBP map", fontweight="bold")
    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("fig_multi_domain_input.png", dpi=300)