"""
Figure 1: Phase spectrum analysis - Real vs Fake comparison
Run this on your local machine with your DFFD/CIFAKE images.

USAGE:
    1. Update REAL_IMAGE_PATH and FAKE_IMAGE_PATH below to point to your images
    2. Run: python figure_1_phase_spectrum.py
    3. Output: figure_1_phase_spectrum.png and .pdf in the same folder

NOTES:
    - Images are automatically resized to 224x224
    - Works with .png, .jpg, .jpeg
    - FFT computation matches paper Eqs. 1, 2, 3, 7
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# ============================================================
# UPDATE THESE TWO PATHS TO YOUR ACTUAL IMAGES:
# ============================================================
REAL_IMAGE_PATH = r'D:\sweet\binary_deepfake_detection\images\celeba4.jpg'     # path to one real DFFD image
FAKE_IMAGE_PATH = r'D:\sweet\binary_deepfake_detection\images\fake_celeba1.png'     # path to one fake DFFD image
# ============================================================

OUTPUT_PATH = 'figure_1_phase_spectrum.png'
IMG_SIZE = 224


def load_and_prepare(path):
    """Load image, convert to RGB, resize to 224x224, normalize to [0,1]."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  Image not found: {path}\n"
            f"  Please update REAL_IMAGE_PATH and FAKE_IMAGE_PATH at the top of this script\n"
            f"  to point to your actual DFFD/CIFAKE images."
        )
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img) / 255.0


def compute_fft_components(img_rgb):
    """
    Compute FFT log-magnitude and phase from RGB image.
    Matches paper equations:
      Eq. 1: grayscale conversion (ITU-R BT.601)
      Eq. 2: 2D FFT with zero-frequency centered
      Eq. 3: log-magnitude with stability constant
      Eq. 7: phase via four-quadrant arctangent
    """
    # Grayscale (Eq. 1)
    gray = (0.299 * img_rgb[:, :, 0]
            + 0.587 * img_rgb[:, :, 1]
            + 0.114 * img_rgb[:, :, 2])

    # Centered 2D FFT (Eq. 2)
    F_shifted = np.fft.fftshift(np.fft.fft2(gray))

    # Log magnitude (Eq. 3)
    log_magnitude = np.log(np.abs(F_shifted) + 1e-8)

    # Phase (Eq. 7)
    phase = np.angle(F_shifted)

    return log_magnitude, phase


# ===== Load images =====
print("Loading images...")
real_img = load_and_prepare(REAL_IMAGE_PATH)
fake_img = load_and_prepare(FAKE_IMAGE_PATH)
print(f"  Real: {REAL_IMAGE_PATH}")
print(f"  Fake: {FAKE_IMAGE_PATH}")

# ===== Compute FFT components =====
real_logmag, real_phase = compute_fft_components(real_img)
fake_logmag, fake_phase = compute_fft_components(fake_img)

# Use a shared color scale for log-magnitude so the two rows are comparable
mag_vmin = min(real_logmag.min(), fake_logmag.min())
mag_vmax = max(real_logmag.max(), fake_logmag.max())

# ===== Build figure =====
fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))

col_titles = ['RGB image', 'FFT log-magnitude', 'FFT phase']
row_titles = ['Real', 'Fake']

# Row 0: Real
axes[0, 0].imshow(real_img)
axes[0, 1].imshow(real_logmag, cmap='viridis', vmin=mag_vmin, vmax=mag_vmax)
axes[0, 2].imshow(real_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)

# Row 1: Fake
axes[1, 0].imshow(fake_img)
axes[1, 1].imshow(fake_logmag, cmap='viridis', vmin=mag_vmin, vmax=mag_vmax)
axes[1, 2].imshow(fake_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)

# Row labels
for r, title in enumerate(row_titles):
    axes[r, 0].set_ylabel(title, fontsize=12, fontweight='bold')

# Column titles
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=11, fontweight='bold')

# Turn off axis ticks
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Phase colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='twilight',
                            norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Phase (rad)', fontsize=10)
cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

plt.tight_layout(rect=[0, 0, 0.91, 1])

# Save
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(OUTPUT_PATH.replace('.png', '.pdf'),
            bbox_inches='tight', facecolor='white')

print(f"\nSaved:")
print(f"  {OUTPUT_PATH}")
print(f"  {OUTPUT_PATH.replace('.png', '.pdf')}")
print("\nDone.")