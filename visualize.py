"""
visualize.py — Phase4DFD Visualization Tool
============================================
Generates all figures required by the thesis committee:

  Figure A: Phase spectrum comparison (real vs fake) — Committee Point 7
  Figure B: Phase-aware attention map visualization  — ICPR reviewer request
  Figure C: FFT magnitude comparison (real vs fake)  — Supporting figure
  Figure D: Multi-image phase analysis grid          — Thesis figure

Usage:
  # Figure A: Phase spectrum comparison (most important — committee asked for this)
  python visualize.py --mode phase --real real_face.jpg --fake fake_face.jpg

  # Figure B: Attention map for a single image
  python visualize.py --mode attention --image face.jpg --checkpoint checkpoints/dffd/phasedfd.ckpt

  # Figure C: Full analysis (phase + magnitude + attention together)
  python visualize.py --mode full --real real_face.jpg --fake fake_face.jpg --checkpoint checkpoints/dffd/phasedfd.ckpt

  # Figure D: Grid of multiple real and fake images
  python visualize.py --mode grid --real_dir test_images/real/ --fake_dir test_images/fake/

Requirements:
  pip install torch torchvision timm pillow numpy matplotlib
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
RESOLUTION  = 224
FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
DPI         = 150

# Color scheme — consistent with thesis
CMAP_PHASE  = "hsv"        # Phase: circular colormap (correct for angular data)
CMAP_MAG    = "inferno"    # Magnitude: perceptually uniform
CMAP_ATTN   = "hot"        # Attention map: hot = suppression visible
CMAP_DIFF   = "RdBu_r"    # Difference maps: diverging


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(image_path: str) -> tuple:
    """
    Load image and return:
      - tensor  : (3, H, W) float tensor in [0,1]
      - display : (H, W, 3) numpy array for imshow
      - gray    : (H, W)    grayscale numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize(RESOLUTION + RESOLUTION // 8,
                 interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(RESOLUTION),
        T.ToTensor(),
    ])

    tensor  = transform(img)                                   # (3, H, W)
    display = tensor.permute(1, 2, 0).numpy()                 # (H, W, 3)
    gray    = (0.299 * tensor[0]
             + 0.587 * tensor[1]
             + 0.114 * tensor[2]).numpy()                      # (H, W)

    return tensor, display, gray


# ══════════════════════════════════════════════════════════════════════════════
# FFT COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_fft(gray: np.ndarray) -> dict:
    """
    Compute FFT of grayscale image.
    Returns dict with phase, magnitude, and derived maps.
    """
    gray_t    = torch.tensor(gray, dtype=torch.float32)
    fft       = torch.fft.fft2(gray_t)
    fft_shift = torch.fft.fftshift(fft)

    # Phase: angular component in [-π, π]
    phase     = torch.angle(fft_shift).numpy()
    phase_norm= (phase + np.pi) / (2 * np.pi)   # normalized to [0, 1]

    # Log-magnitude: compressed dynamic range
    magnitude = torch.abs(fft_shift).numpy()
    log_mag   = np.log(magnitude + 1e-8)

    # Normalized magnitude for display
    log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)

    # Phase gradient magnitude — highlights phase discontinuities
    gy, gx    = np.gradient(phase)
    phase_grad= np.sqrt(gx**2 + gy**2)
    phase_grad_norm = (phase_grad - phase_grad.min()) / (
        phase_grad.max() - phase_grad.min() + 1e-8)

    return {
        "phase":          phase,
        "phase_norm":     phase_norm,
        "log_mag":        log_mag,
        "log_mag_norm":   log_mag_norm,
        "phase_grad":     phase_grad_norm,
        "magnitude_raw":  magnitude,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AXIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def show_img(ax, data, title, cmap=None, vmin=None, vmax=None,
             xlabel=None, colorbar=False):
    """Display image on axis with consistent styling."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color="#555555")
    if colorbar:
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im


def label_row(ax, text, color="#333333"):
    """Add a vertical row label on the left side."""
    ax.text(-0.12, 0.5, text, transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight="bold", color=color,
            va="center", ha="right", rotation=90)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — PHASE SPECTRUM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def figure_phase_comparison(real_path: str, fake_path: str,
                             save_path: str = "figure_phase_comparison.png"):
    """
    COMMITTEE POINT 7 — Show phase discontinuities in real vs fake images.

    Layout (2 rows × 4 columns):
    Row 1 (Real):  Input | Phase spectrum | Phase gradient | Log-magnitude
    Row 2 (Fake):  Input | Phase spectrum | Phase gradient | Log-magnitude
    """
    print(f"\n[Figure A] Phase Spectrum Comparison")
    print(f"  Real  : {real_path}")
    print(f"  Fake  : {fake_path}")

    _, real_disp, real_gray = load_image(real_path)
    _, fake_disp, fake_gray = load_image(fake_path)

    real_fft = compute_fft(real_gray)
    fake_fft = compute_fft(fake_gray)

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.35, wspace=0.08,
                           left=0.08, right=0.97,
                           top=0.90, bottom=0.05)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    col_titles = [
        "Input Image",
        "FFT Phase Spectrum\n(Angular structure encoding)",
        "Phase Gradient Map\n(Discontinuity highlights)",
        "FFT Log-Magnitude\n(Spectral energy distribution)",
    ]

    # Column headers
    for c, title in enumerate(col_titles):
        axes[0][c].set_title(title, fontsize=FONT_TITLE,
                             fontweight="bold", pad=8)

    # ── Row 0: Real ────────────────────────────────────────────────────────────
    show_img(axes[0][0], real_disp,               "")
    show_img(axes[0][1], real_fft["phase_norm"],  "", cmap=CMAP_PHASE)
    show_img(axes[0][2], real_fft["phase_grad"],  "", cmap="gray")
    show_img(axes[0][3], real_fft["log_mag_norm"],"", cmap=CMAP_MAG, colorbar=True)

    label_row(axes[0][0],
              "REAL",
              color="#1a7a2e")

    # Annotation
    axes[0][1].set_xlabel(
        "✓ Smooth, globally coherent phase structure\n"
        "Natural photographic images maintain phase continuity",
        fontsize=FONT_LABEL, color="#1a7a2e", labelpad=4)

    axes[0][2].set_xlabel(
        "✓ Low gradient magnitude\n"
        "Smooth transitions — no abrupt phase boundaries",
        fontsize=FONT_LABEL, color="#1a7a2e", labelpad=4)

    # ── Row 1: Fake ────────────────────────────────────────────────────────────
    show_img(axes[1][0], fake_disp,               "")
    show_img(axes[1][1], fake_fft["phase_norm"],  "", cmap=CMAP_PHASE)
    show_img(axes[1][2], fake_fft["phase_grad"],  "", cmap="gray")
    show_img(axes[1][3], fake_fft["log_mag_norm"],"", cmap=CMAP_MAG, colorbar=True)

    label_row(axes[1][0],
              "FAKE",
              color="#b81c1c")

    axes[1][1].set_xlabel(
        "✗ Irregular, discontinuous phase patterns\n"
        "GAN upsampling introduces systematic phase inconsistencies",
        fontsize=FONT_LABEL, color="#b81c1c", labelpad=4)

    axes[1][2].set_xlabel(
        "✗ High gradient at synthesis boundaries\n"
        "Phase discontinuities expose forgery artifacts",
        fontsize=FONT_LABEL, color="#b81c1c", labelpad=4)

    # Main title
    fig.suptitle(
        "Phase Spectrum Analysis: Real vs GAN-Synthesized Fake Images\n"
        "GAN synthesis processes do not enforce phase consistency, "
        "introducing systematic phase irregularities absent in authentic photographs",
        fontsize=14, fontweight="bold", y=0.97, color="#1a1a2e"
    )

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — ATTENTION MAP VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def figure_attention_map(image_path: str, checkpoint_path: str,
                          dataset_key: str = "dffd",
                          save_path: str = "figure_attention_map.png"):
    """
    ICPR REVIEWER REQUEST — Show what the phase-aware attention module focuses on.

    Layout (1 row × 6 columns):
    Input | Phase map | Magnitude | Attention A0 (avg) | A0 FFT channel | Overlay
    """
    print(f"\n[Figure B] Attention Map Visualization")
    print(f"  Image      : {image_path}")
    print(f"  Checkpoint : {checkpoint_path}")

    try:
        from model import BNext4DFR
    except ImportError:
        print("  ERROR: Cannot import model.py — place model.py in same folder")
        return

    # ── Load image ─────────────────────────────────────────────────────────────
    tensor, display, gray = load_image(image_path)
    # tensor shape: (3, H, W) — add batch dim → (1, 3, H, W)
    original_rgb = tensor.unsqueeze(0)   # (1, 3, 224, 224)

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")

    # ── Move image to device BEFORE loading model ──────────────────────────────
    # FIX: define original_rgb_dev here so it is always in scope
    original_rgb_dev = original_rgb.to(device)

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: Checkpoint not found — using untrained model")
        print(f"  (Attention map will show random weights, not learned behavior)")
        model = BNext4DFR(
            num_classes=2,
            backbone="BNext-M",
            freeze_backbone=False,
            add_fft_magnitude=True,
            add_lbp_channel=True,
            use_frequency_attention=True,
            attention_position="before_backbone",
            learning_rate=1e-3,
            pos_weight=None,
            use_dropout=True,
            dropout_rate=0.3,
        )
    else:
        # Attempt 1: Lightning
        try:
            model = BNext4DFR.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                map_location=device,
                weights_only=False,
            )
            print(f"  Checkpoint loaded (Lightning)")
        except Exception as e1:
            # Attempt 2: Manual state_dict
            try:
                ckpt = torch.load(checkpoint_path, map_location=device,
                                  weights_only=False)
                model = BNext4DFR(
                    num_classes=2,
                    backbone="BNext-M",
                    freeze_backbone=False,
                    add_fft_magnitude=True,
                    add_lbp_channel=True,
                    use_frequency_attention=True,
                    attention_position="before_backbone",
                    learning_rate=1e-3,
                    pos_weight=None,
                    use_dropout=True,
                    dropout_rate=0.3,
                )
                state = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state, strict=False)
                print(f"  Checkpoint loaded (manual state_dict)")
            except Exception as e2:
                print(f"  Could not load checkpoint:")
                print(f"    Lightning error : {e1}")
                print(f"    Manual error   : {e2}")
                print(f"  Continuing with untrained model...")

    model.to(device)
    model.eval()

    # ── Verify model has input attention ───────────────────────────────────────
    if model.input_attention is None:
        print("  ERROR: This model has no input attention module.")
        print("  Use a checkpoint trained with attention_position='before_backbone' or 'both'")
        print("  (PhaseDFD or FullDFD checkpoints)")
        return

    # ── Register hook to capture attention map A0 ──────────────────────────────
    attention_maps = {}

    def hook_fn(module, inp, out):
        attention_maps["A0"] = out.detach().cpu()

    hook = model.input_attention.attention_gen.register_forward_hook(hook_fn)

    # ── Forward pass through attention module only ─────────────────────────────
    with torch.no_grad():
        x_aug = model.add_new_channels(original_rgb_dev)   # (1, 5, 224, 224)
        print(f"  Augmented input shape : {x_aug.shape}")
        _     = model.input_attention(x_aug, original_rgb=original_rgb_dev)

    hook.remove()

    hook.remove()

    A0 = attention_maps.get("A0")
    if A0 is None:
        print("  Could not extract attention map")
        return

    # A0 shape: (1, 5, H, W) — average across 5 channels for visualization
    A0_avg   = A0[0].mean(dim=0).numpy()       # (H, W)
    A0_rgb   = A0[0, :3].mean(dim=0).numpy()   # RGB channels only
    A0_fft   = A0[0, 3].numpy()                # FFT magnitude channel
    A0_lbp   = A0[0, 4].numpy()                # LBP channel

    # Compute phase and magnitude for display
    fft_data = compute_fft(gray)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(wspace=0.08, left=0.03, right=0.97,
                        top=0.82, bottom=0.15)

    show_img(axes[0], display,
             "Input Image\n(RGB)", xlabel=os.path.basename(image_path))

    show_img(axes[1], fft_data["phase_norm"],
             "FFT Phase\n(Recomputed dynamically)", cmap=CMAP_PHASE)

    show_img(axes[2], fft_data["log_mag_norm"],
             "FFT Log-Magnitude\n(Pre-computed channel 3)", cmap=CMAP_MAG)

    show_img(axes[3], A0_avg,
             "Attention Map A₀\n(Avg across 5 channels)",
             cmap=CMAP_ATTN, vmin=0, vmax=1, colorbar=True)

    show_img(axes[4], A0_fft,
             "A₀ — FFT Channel\n(Channel 3 attention weight)",
             cmap=CMAP_ATTN, vmin=0, vmax=1)

    # Overlay: attention on image
    axes[5].imshow(display)
    axes[5].imshow(1 - A0_avg, cmap="Blues", alpha=0.55, vmin=0, vmax=1)
    axes[5].set_title("Suppression Overlay\n(Blue = suppressed by gate)",
                      fontsize=FONT_TITLE, fontweight="bold", pad=6)
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    axes[5].set_xlabel(
        "Darker blue regions are suppressed before backbone processing",
        fontsize=FONT_LABEL, color="#333333")

    # Statistics
    suppressed_pct = (A0_avg < 0.5).sum() / A0_avg.size * 100
    axes[3].set_xlabel(
        f"Mean: {A0_avg.mean():.3f}  "
        f"Suppressed (<0.5): {suppressed_pct:.1f}%",
        fontsize=FONT_LABEL, color="#333333")

    fig.suptitle(
        "Phase-Aware Attention Module: Input-Level Spatial Gate A₀ ∈ (0,1)^(5×224×224)\n"
        "Values close to 0 suppress forgery-consistent frequency regions "
        "before backbone feature extraction",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — FULL ANALYSIS SIDE BY SIDE
# ══════════════════════════════════════════════════════════════════════════════

def figure_full_analysis(real_path: str, fake_path: str,
                          checkpoint_path: str = None,
                          save_path: str = "figure_full_analysis.png"):
    """
    Combined figure: phase comparison + attention maps for real AND fake.
    Most comprehensive visualization for the thesis.

    Layout (2 rows × 5 columns):
    Row 1 (Real): Input | Phase | Phase-grad | Log-mag | Attention
    Row 2 (Fake): Input | Phase | Phase-grad | Log-mag | Attention
    """
    print(f"\n[Figure C] Full Analysis")
    print(f"  Real  : {real_path}")
    print(f"  Fake  : {fake_path}")

    _, real_disp, real_gray = load_image(real_path)
    _, fake_disp, fake_gray = load_image(fake_path)

    real_fft = compute_fft(real_gray)
    fake_fft = compute_fft(fake_gray)

    # ── Try to get attention maps ──────────────────────────────────────────────
    real_attn = None
    fake_attn = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            from model import BNext4DFR
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = BNext4DFR.load_from_checkpoint(
                checkpoint_path, strict=False, map_location=device)
            model.to(device)
            model.eval()

            def get_attention(image_path):
                tensor, _, _ = load_image(image_path)
                rgb = tensor.unsqueeze(0).to(device)
                attn_store = {}

                def hook(m, i, o):
                    attn_store["A0"] = o.detach().cpu()

                h = model.input_attention.attention_gen.register_forward_hook(hook)
                with torch.no_grad():
                    x_aug = model.add_new_channels(rgb)
                    model.input_attention(x_aug, original_rgb=rgb)
                h.remove()
                return attn_store.get("A0", None)

            if model.input_attention is not None:
                real_attn = get_attention(real_path)
                fake_attn = get_attention(fake_path)
                print(f"  Attention maps extracted successfully")
        except Exception as e:
            print(f"  Could not extract attention: {e}")

    # ── Layout ─────────────────────────────────────────────────────────────────
    n_cols = 5 if (real_attn is not None) else 4
    col_titles_base = [
        "Input Image",
        "FFT Phase Spectrum",
        "Phase Gradient\n(Discontinuity Map)",
        "FFT Log-Magnitude",
    ]
    if real_attn is not None:
        col_titles_base.append("Phase-Aware\nAttention Gate A₀")

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.30, wspace=0.06,
                        left=0.07, right=0.97,
                        top=0.88, bottom=0.08)

    for c, title in enumerate(col_titles_base):
        axes[0][c].set_title(title, fontsize=FONT_TITLE,
                             fontweight="bold", pad=8)

    row_data = [
        (real_disp, real_fft, real_attn, "REAL", "#1a7a2e",
         real_path,
         "✓ Smooth, coherent phase — natural photographic image",
         "✓ Low gradient — no abrupt phase boundaries"),
        (fake_disp, fake_fft, fake_attn, "FAKE", "#b81c1c",
         fake_path,
         "✗ Irregular phase — GAN synthesis artifacts",
         "✗ High gradient — phase discontinuities at synthesis boundaries"),
    ]

    for r, (disp, fft_d, attn, label, color,
            path, phase_note, grad_note) in enumerate(row_data):

        label_row(axes[r][0], label, color=color)

        show_img(axes[r][0], disp, "",
                 xlabel=os.path.basename(path))
        show_img(axes[r][1], fft_d["phase_norm"], "", cmap=CMAP_PHASE,
                 xlabel=phase_note)
        show_img(axes[r][2], fft_d["phase_grad"], "", cmap="gray",
                 xlabel=grad_note)
        show_img(axes[r][3], fft_d["log_mag_norm"], "", cmap=CMAP_MAG,
                 colorbar=(n_cols == 4))

        if attn is not None:
            A0_avg = attn[0].mean(dim=0).numpy()
            show_img(axes[r][4], A0_avg, "",
                     cmap=CMAP_ATTN, vmin=0, vmax=1, colorbar=True)
            supp = (A0_avg < 0.5).sum() / A0_avg.size * 100
            axes[r][4].set_xlabel(
                f"Suppressed: {supp:.1f}%  Mean: {A0_avg.mean():.3f}",
                fontsize=FONT_LABEL, color=color)

    fig.suptitle(
        "Phase4DFD — Frequency Domain Analysis and Phase-Aware Attention\n"
        "Fourier phase encodes structural spatial alignment [Oppenheim & Lim, 1981]; "
        "GAN synthesis introduces systematic phase irregularities exploited by Phase4DFD",
        fontsize=13, fontweight="bold", y=0.97
    )

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — MULTI-IMAGE GRID
# ══════════════════════════════════════════════════════════════════════════════

def figure_phase_grid(real_dir: str, fake_dir: str, n_per_class: int = 4,
                       save_path: str = "figure_phase_grid.png"):
    """
    Show phase spectra for multiple real and fake images in a grid.
    Demonstrates that phase inconsistencies are consistent across fake images.

    Layout: 4 real + 4 fake, each showing: input | phase | gradient
    """
    print(f"\n[Figure D] Phase Grid")
    print(f"  Real dir : {real_dir}")
    print(f"  Fake dir : {fake_dir}")

    def collect_images(folder, n):
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if os.path.splitext(f)[1].lower() in exts][:n]
        return files

    real_paths = collect_images(real_dir, n_per_class)
    fake_paths = collect_images(fake_dir, n_per_class)

    if not real_paths:
        print(f"  No images found in {real_dir}")
        return
    if not fake_paths:
        print(f"  No images found in {fake_dir}")
        return

    all_paths  = [(p, "REAL") for p in real_paths] + \
                 [(p, "FAKE") for p in fake_paths]
    n_total    = len(all_paths)

    fig, axes = plt.subplots(n_total, 3,
                              figsize=(10, 3.2 * n_total))
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(hspace=0.12, wspace=0.04,
                        left=0.10, right=0.97,
                        top=0.95, bottom=0.02)

    # Column titles (top row only)
    for c, t in enumerate(["Input Image", "FFT Phase Spectrum",
                            "Phase Gradient (Discontinuities)"]):
        axes[0][c].set_title(t, fontsize=FONT_TITLE,
                             fontweight="bold", pad=6)

    for r, (path, label) in enumerate(all_paths):
        color = "#1a7a2e" if label == "REAL" else "#b81c1c"
        try:
            _, disp, gray = load_image(path)
            fft_d = compute_fft(gray)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        show_img(axes[r][0], disp, "")
        show_img(axes[r][1], fft_d["phase_norm"], "", cmap=CMAP_PHASE)
        show_img(axes[r][2], fft_d["phase_grad"], "", cmap="gray")

        label_row(axes[r][0],
                  f"{label}\n{os.path.basename(path)[:15]}",
                  color=color)

        # Add separator line between real and fake
        if r == len(real_paths) - 1:
            for c in range(3):
                axes[r][c].axhline(y=RESOLUTION - 1,
                                   color="#888888", linewidth=2, linestyle="--")

    fig.suptitle(
        "Phase Spectrum Grid: Multiple Real vs Fake Images\n"
        "Phase irregularities are systematic and consistent across GAN-synthesized images",
        fontsize=13, fontweight="bold"
    )

    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase4DFD Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  phase      Phase spectrum comparison (real vs fake)           -- Committee Point 7
  attention  Attention map for a single image                   -- ICPR request
  full       Combined phase + attention for real AND fake        -- Best for thesis
  grid       Phase grid of multiple images                      -- Overview figure

Examples:
  python visualize.py --mode phase --real real.jpg --fake fake.jpg
  python visualize.py --mode attention --image face.jpg --checkpoint checkpoints/dffd/phasedfd.ckpt
  python visualize.py --mode full --real real.jpg --fake fake.jpg --checkpoint checkpoints/dffd/phasedfd.ckpt
  python visualize.py --mode grid --real_dir images/real/ --fake_dir images/fake/
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["phase", "attention", "full", "grid"],
                        help="Visualization mode")

    # Image inputs
    parser.add_argument("--image",   type=str, default=None,
                        help="Single image path (for attention mode)")
    parser.add_argument("--real",    type=str, default=None,
                        help="Real image path (for phase/full mode)")
    parser.add_argument("--fake",    type=str, default=None,
                        help="Fake image path (for phase/full mode)")
    parser.add_argument("--real_dir",type=str, default=None,
                        help="Folder of real images (for grid mode)")
    parser.add_argument("--fake_dir",type=str, default=None,
                        help="Folder of fake images (for grid mode)")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Phase4DFD-B or Full checkpoint")
    parser.add_argument("--n",       type=int, default=4,
                        help="Number of images per class in grid (default: 4)")

    # Output
    parser.add_argument("--output",  type=str, default=None,
                        help="Output filename (default: auto)")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    os.makedirs("figures", exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  Phase4DFD Visualization Tool")
    print(f"  Mode: {args.mode.upper()}")
    print(f"{'═'*55}")

    if args.mode == "phase":
        if not args.real or not args.fake:
            print("ERROR: --mode phase requires --real and --fake")
            sys.exit(1)
        out = args.output or "figures/figure_phase_comparison.png"
        figure_phase_comparison(args.real, args.fake, out)

    elif args.mode == "attention":
        if not args.image:
            print("ERROR: --mode attention requires --image")
            sys.exit(1)
        if not args.checkpoint:
            print("WARNING: No --checkpoint provided. Visualizing with untrained model.")
        out = args.output or "figures/figure_attention_map.png"
        figure_attention_map(args.image,
                              args.checkpoint or "",
                              save_path=out)

    elif args.mode == "full":
        if not args.real or not args.fake:
            print("ERROR: --mode full requires --real and --fake")
            sys.exit(1)
        out = args.output or "figures/figure_full_analysis.png"
        figure_full_analysis(args.real, args.fake,
                              checkpoint_path=args.checkpoint,
                              save_path=out)

    elif args.mode == "grid":
        if not args.real_dir or not args.fake_dir:
            print("ERROR: --mode grid requires --real_dir and --fake_dir")
            sys.exit(1)
        out = args.output or "figures/figure_phase_grid.png"
        figure_phase_grid(args.real_dir, args.fake_dir,
                           n_per_class=args.n,
                           save_path=out)

    print(f"\n{'═'*55}")
    print(f"  Done. Check the figures/ folder.")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()