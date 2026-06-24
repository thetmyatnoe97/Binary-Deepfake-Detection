"""
Figure B: Phase discontinuity visualization (TEMPLATE)
2x3 grid: rows = real/fake, columns = RGB / FFT magnitude / FFT phase
 
This is a TEMPLATE that uses synthetic placeholder images.
Replace the load_image() calls with your actual paths to a real and a fake
image from the DFFD or CIFAKE dataset.
 
The visualization logic (FFT magnitude/phase computation) is correct and matches
your paper's Equation 2-3 and Equation 7. You only need to swap the input images.
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
 
 
def compute_fft_components(rgb_image):
    """Compute log-magnitude and phase from RGB image, matching paper Eqs. 2,3,7."""
    # Convert to grayscale using ITU-R BT.601 (matching Eq. 1)
    gray = (0.299 * rgb_image[:, :, 0]
            + 0.587 * rgb_image[:, :, 1]
            + 0.114 * rgb_image[:, :, 2])
 
    # 2D FFT with zero-frequency centered (Eq. 2)
    F = np.fft.fft2(gray)
    F_shifted = np.fft.fftshift(F)
 
    # Log magnitude (Eq. 3)
    magnitude = np.log(np.abs(F_shifted) + 1e-8)
 
    # Phase via four-quadrant arctan (Eq. 7)
    phase = np.angle(F_shifted)
 
    return magnitude, phase
 
 
def make_synthetic_natural_image(seed=0, size=224):
    """
    Synthesize a 'natural-looking' image (smooth, low-frequency dominant).
    Used as placeholder for the 'real' panel.
    """
    rng = np.random.default_rng(seed)
    # 1/f noise pattern - natural images have power law spectra
    freqs_y = np.fft.fftfreq(size).reshape(-1, 1)
    freqs_x = np.fft.fftfreq(size).reshape(1, -1)
    radial = np.sqrt(freqs_y**2 + freqs_x**2) + 1e-6
 
    img = np.zeros((size, size, 3))
    for c in range(3):
        random_phase = rng.uniform(0, 2*np.pi, (size, size))
        spectrum = (1.0 / radial) * np.exp(1j * random_phase)
        channel = np.real(np.fft.ifft2(spectrum))
        # Normalize to [0, 1]
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        img[:, :, c] = channel
    return img
 
 
def make_synthetic_gan_image(seed=1, size=224):
    """
    Synthesize an image that mimics GAN upsampling artifacts:
    natural-like content + periodic high-frequency grid pattern.
    Used as placeholder for the 'fake' panel.
    """
    base = make_synthetic_natural_image(seed=seed, size=size)
    # Add 4x upsampling grid artifact (period 4 pixels)
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    grid = 0.05 * (np.sin(2 * np.pi * y / 4) * np.sin(2 * np.pi * x / 4))
    # Apply to all channels
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c] + grid, 0, 1)
    return base
 
 
# ---------- LOAD IMAGES ----------
# REPLACE THESE LINES with paths to your actual real/fake images:
real_img = np.array(Image.open('/sweet/binary_deepfake_detection/images/celeba4.jpg').resize((224, 224))) / 255.0
fake_img = np.array(Image.open('/sweet/binary_deepfake_detection/images/fake_celeba1.png').resize((224, 224))) / 255.0
 
# Synthetic placeholders for now:
real_img = make_synthetic_natural_image(seed=42)
fake_img = make_synthetic_gan_image(seed=42)
 
real_mag, real_phase = compute_fft_components(real_img)
fake_mag, fake_phase = compute_fft_components(fake_img)
 
# ---------- BUILD FIGURE ----------
fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
 
col_titles = ['RGB image', 'FFT log-magnitude', 'FFT phase']
row_titles = ['Real', 'Fake (synthetic)']
 
# Row 0: real
axes[0, 0].imshow(real_img)
axes[0, 1].imshow(real_mag, cmap='viridis')
axes[0, 2].imshow(real_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
 
# Row 1: fake
axes[1, 0].imshow(fake_img)
axes[1, 1].imshow(fake_mag, cmap='viridis')
axes[1, 2].imshow(fake_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
 
# Add row labels on the left
for r, title in enumerate(row_titles):
    axes[r, 0].set_ylabel(title, fontsize=12, fontweight='bold')
 
# Add column titles
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=11, fontweight='bold')
 
# Turn off ticks
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
 
# Add a colorbar for the phase (rightmost column)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='twilight',
                            norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Phase (rad)', fontsize=10)
cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cb.set_ticklabels(['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
 
plt.tight_layout(rect=[0, 0, 0.91, 1])
plt.savefig('/sweet/binary_deepfake_detection/figures/phase_discontinuity_TEMPLATE.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/sweet/binary_deepfake_detection/figures/figure_b_phase_discontinuity_TEMPLATE.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure B template saved. Replace synthetic images with real DFFD/CIFAKE samples.")
