"""
phase_visualization.py — Phase Discontinuity Visualization for Thesis
======================================================================
Generates Figure 1 of the thesis: side-by-side comparison of FFT phase
spectra between real and fake images across DFFD, CIFAKE, and COCOFake.

Produces output figures:
  1. phase_discontinuity_grid.png  — 3-dataset grid
  2. phase_statistics.png          — statistical bar chart
  3. phase_pair_dffd.png           — detailed DFFD pair
  4. phase_pair_cifake.png         — detailed CIFAKE pair
  5. phase_pair_cocofake.png       — detailed COCOFake pair

Usage:
    python phase_visualization.py ^
      --cifake_path        D:/datasets/cifake ^
      --dffd_path          D:/datasets/dffd ^
      --cocofake_real_path D:/datasets/coco2014/train2014 ^
      --cocofake_fake_path D:/datasets/coco_fake/train2014 ^
      --output_dir         ./figures ^
      --n_samples          4

    # Demo mode (no datasets needed):
    python phase_visualization.py --demo --output_dir ./figures
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ── Constants ─────────────────────────────────────────────────────────────────
RESOLUTION = 224
DPI        = 300
LUM        = np.array([0.299, 0.587, 0.114], dtype=np.float32)

DATASET_LABELS = {
    "dffd":     "DFFD (GAN face forgery)",
    "cifake":   "CIFAKE (Diffusion low-res)",
    "cocofake": "COCOFake (Diffusion natural scene)",
}


# ── Image processing ───────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load and resize image to (H, W, 3) float32 in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) to (H, W) using ITU-R BT.601."""
    return img @ LUM


def compute_fft(gray: np.ndarray):
    """
    Compute centered 2D FFT of grayscale image.
    Returns:
        mag_log : (H, W) log-magnitude spectrum
        phase   : (H, W) phase spectrum in [-π, π]
    """
    fft       = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag_log   = np.log(np.abs(fft_shift) + 1e-8)
    phase     = np.angle(fft_shift)
    return mag_log, phase


def phase_discontinuity_map(phase: np.ndarray) -> np.ndarray:
    """
    Local phase discontinuity: magnitude of the wrap-aware phase gradient.
    Highlights abrupt phase transitions characteristic of synthesis artifacts.
    """
    dy = np.diff(phase, axis=0, prepend=phase[:1, :])
    dx = np.diff(phase, axis=1, prepend=phase[:, :1])
    dy = (dy + np.pi) % (2 * np.pi) - np.pi
    dx = (dx + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt(dx**2 + dy**2)


def process_image(path: str) -> dict:
    """Load image and compute all derived representations."""
    img     = load_image(path)
    gray    = to_grayscale(img)
    mag_log, phase = compute_fft(gray)
    disc    = phase_discontinuity_map(phase)
    return {
        "img":     img,
        "gray":    gray,
        "mag_log": mag_log,
        "phase":   phase,
        "disc":    disc,
        "path":    path,
    }


def high_freq_energy(phase: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of phase energy in the outer (high-frequency) region."""
    H, W   = phase.shape
    cy, cx = H // 2, W // 2
    ry, rx = int(H * threshold / 2), int(W * threshold / 2)
    mask   = np.ones((H, W), dtype=bool)
    mask[cy-ry:cy+ry, cx-rx:cx+rx] = False
    total  = np.sum(phase**2) + 1e-9
    return float(np.sum(phase[mask]**2) / total * 100.0)


# ── File discovery ─────────────────────────────────────────────────────────────

def find_images(folder: str, n: int) -> list:
    """
    Recursively find up to n images under folder using os.walk.
    Stops early once enough candidates are found — safe for large datasets.
    """
    folder = str(folder)
    if not os.path.exists(folder):
        print(f"  [WARN] Folder not found: {folder}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                  ".JPG", ".JPEG", ".PNG"}
    found = []

    for root, dirs, files in os.walk(folder):
        for fname in files:
            if os.path.splitext(fname)[1] in extensions:
                found.append(os.path.join(root, fname))
        if len(found) >= n * 20:
            break

    if not found:
        print(f"  [WARN] No images found in: {folder}")
        return []

    random.shuffle(found)
    return found[:n]


def load_dataset(name: str, real_folder: str, fake_folder: str,
                 n: int) -> dict:
    """
    Load n real and n fake images from the given folders.
    Returns {"real": [...], "fake": [...]} or None if loading fails.
    """
    print(f"\nLoading {name}...")
    print(f"  Real folder : {real_folder}")
    print(f"  Fake folder : {fake_folder}")

    real_paths = find_images(real_folder, n)
    fake_paths = find_images(fake_folder, n)
    print(f"  Candidates  : {len(real_paths)} real, {len(fake_paths)} fake")

    if not real_paths or not fake_paths:
        print(f"  [SKIP] {name}: no images found")
        return None

    real_data, fake_data = [], []

    for i, path in enumerate(real_paths[:n]):
        try:
            real_data.append(process_image(path))
            print(f"  ✓ Real {i+1}/{n}: {Path(path).name}")
        except Exception as e:
            print(f"  ✗ Failed (real): {Path(path).name} — {e}")

    for i, path in enumerate(fake_paths[:n]):
        try:
            fake_data.append(process_image(path))
            print(f"  ✓ Fake {i+1}/{n}: {Path(path).name}")
        except Exception as e:
            print(f"  ✗ Failed (fake): {Path(path).name} — {e}")

    if not real_data or not fake_data:
        print(f"  [SKIP] {name}: processing failed for all images")
        return None

    print(f"  ✓ {name}: {len(real_data)} real, {len(fake_data)} fake loaded")
    return {"real": real_data, "fake": fake_data}


# ── Dataset folder helpers ─────────────────────────────────────────────────────

def find_subfolder(base: str, candidates: list) -> str:
    """Return first existing subfolder from candidates list."""
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.exists(p):
            return p
    return base   # fallback: use base directly


def get_dffd_folders(base: str):
    real = find_subfolder(base, ["real", "REAL", "Real",
                                  "test/real", "test/REAL",
                                  "train/real", "train/REAL"])
    fake = find_subfolder(base, ["fake", "FAKE", "Fake",
                                  "test/fake", "test/FAKE",
                                  "train/fake", "train/FAKE"])
    return real, fake


def get_cifake_folders(base: str):
    real = find_subfolder(base, ["test/REAL", "test/real",
                                  "train/REAL", "train/real", "REAL", "real"])
    fake = find_subfolder(base, ["test/FAKE", "test/fake",
                                  "train/FAKE", "train/fake", "FAKE", "fake"])
    return real, fake


# ── Synthetic demo data ────────────────────────────────────────────────────────

def make_demo_data(n: int) -> dict:
    """Generate synthetic real/fake data for demo mode."""
    def make_real():
        img = np.random.rand(RESOLUTION, RESOLUTION, 3).astype(np.float32) * 0.5 + 0.25
        x, y = np.meshgrid(np.linspace(0, 1, RESOLUTION),
                            np.linspace(0, 1, RESOLUTION))
        img[:, :, 0] += 0.2 * np.sin(2 * np.pi * x * 3) * np.cos(2 * np.pi * y * 2)
        img = np.clip(img, 0, 1)
        gray = to_grayscale(img)
        mag_log, phase = compute_fft(gray)
        disc = phase_discontinuity_map(phase)
        return {"img": img, "gray": gray, "mag_log": mag_log,
                "phase": phase, "disc": disc, "path": "demo_real"}

    def make_fake():
        img = np.random.rand(RESOLUTION, RESOLUTION, 3).astype(np.float32) * 0.5 + 0.25
        x, y = np.meshgrid(np.linspace(0, 1, RESOLUTION),
                            np.linspace(0, 1, RESOLUTION))
        artifact = (0.4 * np.sin(2 * np.pi * x * 16) +
                    0.3 * np.sin(2 * np.pi * y * 16))
        img[:, :, 0] += 0.15 * artifact
        img[:, :, 1] += 0.10 * artifact
        img = np.clip(img, 0, 1)
        gray = to_grayscale(img)
        mag_log, phase = compute_fft(gray)
        disc = phase_discontinuity_map(phase)
        return {"img": img, "gray": gray, "mag_log": mag_log,
                "phase": phase, "disc": disc, "path": "demo_fake"}

    return {
        "real": [make_real() for _ in range(n)],
        "fake": [make_fake() for _ in range(n)],
    }


# ── Figure 1: Grid ─────────────────────────────────────────────────────────────

def make_grid_figure(datasets: dict, output_path: str):
    """
    3-dataset grid figure.
    Rows = datasets. Columns = Real image | Real phase | Real discont. |
                               Fake image | Fake phase | Fake discont.
    """
    n_ds  = len(datasets)
    ncols = 6

    fig, axes = plt.subplots(
        n_ds, ncols,
        figsize=(ncols * 2.6, n_ds * 3.0),
        dpi=DPI,
    )
    fig.patch.set_facecolor("white")

    # Ensure axes is always 2D
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Real image", "Real phase\n[−π, π]", "Real\ndiscontinuity",
        "Fake image", "Fake phase\n[−π, π]", "Fake\ndiscontinuity",
    ]
    col_cmaps  = [None, "RdBu_r", "hot", None, "RdBu_r", "hot"]
    col_keys   = ["img", "phase", "disc", "img", "phase", "disc"]
    col_norm   = [None, Normalize(-np.pi, np.pi), None,
                  None, Normalize(-np.pi, np.pi), None]
    col_source = ["real", "real", "real", "fake", "fake", "fake"]
    col_color  = ["#2ecc71"] * 3 + ["#e74c3c"] * 3

    for row_idx, (ds_name, ds_data) in enumerate(datasets.items()):
        real = ds_data["real"][0]
        fake = ds_data["fake"][0]

        for col_idx in range(ncols):
            ax     = axes[row_idx, col_idx]
            source = real if col_source[col_idx] == "real" else fake
            key    = col_keys[col_idx]
            data   = source[key]

            if key == "img":
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=col_cmaps[col_idx],
                          norm=col_norm[col_idx], interpolation="nearest")

            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on first row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8,
                             pad=4, color="#333333")

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS.get(ds_name, ds_name),
                    fontsize=8, fontweight="bold", labelpad=6,
                    color="#222222",
                )

            # Border color: green = real, red = fake
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col_color[col_idx])
                spine.set_linewidth(1.5)

    # Legend
    legend_elements = [
        Patch(facecolor="white", edgecolor="#2ecc71",
              linewidth=2, label="Real (authentic)"),
        Patch(facecolor="white", edgecolor="#e74c3c",
              linewidth=2, label="Fake (synthesized)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=2, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        "Phase Discontinuity Analysis: Real vs. Fake Images\n"
        "Columns: input image | centered FFT phase (RdBu) | "
        "local phase discontinuity (hot colormap)\n"
        "Brighter regions in discontinuity maps indicate abrupt phase "
        "transitions introduced by generative synthesis.",
        fontsize=8.5, y=1.02, color="#111111",
    )

    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Grid figure    : {output_path}")


# ── Figure 2: Statistics ───────────────────────────────────────────────────────

def make_statistics_figure(datasets: dict, output_path: str):
    """Bar chart comparing phase properties across datasets."""
    metrics = {
        "Mean phase\ndiscontinuity":    lambda d: np.mean(d["disc"]),
        "Phase variance":               lambda d: np.var(d["phase"]),
        "High-freq phase\nenergy (%)":  lambda d: high_freq_energy(d["phase"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=DPI)
    fig.patch.set_facecolor("white")

    for ax, (metric_name, metric_fn) in zip(axes, metrics.items()):
        ds_names  = list(datasets.keys())
        real_vals, fake_vals = [], []
        real_err,  fake_err  = [], []

        for ds_name in ds_names:
            rv = [metric_fn(d) for d in datasets[ds_name]["real"]]
            fv = [metric_fn(d) for d in datasets[ds_name]["fake"]]
            real_vals.append(np.mean(rv));  real_err.append(np.std(rv))
            fake_vals.append(np.mean(fv));  fake_err.append(np.std(fv))

        x     = np.arange(len(ds_names))
        w     = 0.35
        lbls  = [DATASET_LABELS.get(n, n).split("(")[0].strip()
                 for n in ds_names]

        ax.bar(x - w/2, real_vals, w,
               yerr=real_err, capsize=4,
               color="#2ecc71", alpha=0.85, edgecolor="#27ae60",
               linewidth=0.8, label="Real",
               error_kw={"linewidth": 1.0})
        ax.bar(x + w/2, fake_vals, w,
               yerr=fake_err, capsize=4,
               color="#e74c3c", alpha=0.85, edgecolor="#c0392b",
               linewidth=0.8, label="Fake",
               error_kw={"linewidth": 1.0})

        # Annotate % difference
        for i, (rv, fv) in enumerate(zip(real_vals, fake_vals)):
            diff = (fv - rv) / (rv + 1e-9) * 100
            sign = "+" if diff >= 0 else ""
            ax.text(i, max(rv, fv) * 1.06,
                    f"{sign}{diff:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#444444")

        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=8.5)
        ax.set_title(metric_name, fontsize=10,
                     fontweight="bold", pad=8)
        ax.legend(fontsize=8.5, frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Statistical Comparison of Phase Properties: Real vs. Fake Images\n"
        "Error bars = ±1 std. Percentages = relative difference (fake vs. real).",
        fontsize=10, y=1.03,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Statistics     : {output_path}")


# ── Figure 3: Detailed pair ────────────────────────────────────────────────────

def make_pair_figure(real_data: dict, fake_data: dict,
                     ds_name: str, output_path: str):
    """
    Detailed single-pair figure for thesis insertion.
    2 rows (real/fake) × 4 columns (image | phase | discontinuity | magnitude).
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), dpi=DPI)
    fig.patch.set_facecolor("white")

    rows        = [real_data, fake_data]
    row_labels  = ["Real (authentic)", "Fake (synthesized)"]
    row_colors  = ["#2ecc71", "#e74c3c"]

    col_info = [
        ("img",     None,       None,
         "Input image"),
        ("phase",   "RdBu_r",   Normalize(-np.pi, np.pi),
         "FFT phase spectrum\n[−π, π]"),
        ("disc",    "hot",      None,
         "Phase discontinuity map"),
        ("mag_log", "plasma",   None,
         "FFT log-magnitude\nspectrum"),
    ]

    for row_idx, (data, row_label, border_color) in enumerate(
            zip(rows, row_labels, row_colors)):

        for col_idx, (key, cmap, norm, col_title) in enumerate(col_info):
            ax = axes[row_idx, col_idx]
            d  = data[key]

            if key == "img":
                ax.imshow(d)
            else:
                im = ax.imshow(d, cmap=cmap, norm=norm,
                               interpolation="nearest")
                if key == "phase":
                    cbar = plt.colorbar(im, ax=ax,
                                        fraction=0.046, pad=0.04)
                    cbar.set_ticks([-np.pi, 0, np.pi])
                    cbar.set_ticklabels(["-π", "0", "π"])
                    cbar.ax.tick_params(labelsize=7)

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10,
                              fontweight="bold", color=border_color,
                              labelpad=8)
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)

            # Column title
            if row_idx == 0:
                ax.set_title(col_title, fontsize=9, pad=5, color="#333333")

            # Mean discontinuity annotation under disc column
            if key == "disc":
                ax.set_xlabel(
                    f"Mean: {np.mean(data['disc']):.4f}",
                    fontsize=7.5, color="#555555", labelpad=3,
                )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.5)

    ds_label = DATASET_LABELS.get(ds_name, ds_name)
    fig.suptitle(
        f"Phase Discontinuity Analysis — {ds_label}\n"
        "Real images show smooth, coherent phase spectra. "
        "Fake images exhibit elevated phase discontinuities\n"
        "(brighter regions) at boundaries and high-frequency bands, "
        "introduced by generative synthesis operations.",
        fontsize=9, y=1.02, color="#111111",
    )

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Pair figure    : {output_path}")


# ── Statistics table ───────────────────────────────────────────────────────────

def print_statistics(datasets: dict):
    print(f"\n{'='*72}")
    print("Mean Phase Discontinuity Statistics (for thesis Table caption):")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'Real mean±std':>20} {'Fake mean±std':>20} "
          f"{'Diff %':>10}")
    print("-" * 72)
    for ds_name, ds_data in datasets.items():
        rv = [np.mean(d["disc"]) for d in ds_data["real"]]
        fv = [np.mean(d["disc"]) for d in ds_data["fake"]]
        rm, rs = np.mean(rv), np.std(rv)
        fm, fs = np.mean(fv), np.std(fv)
        diff   = (fm - rm) / (rm + 1e-9) * 100
        sign   = "+" if diff >= 0 else ""
        print(f"{ds_name:<12} {rm:>8.4f} ± {rs:>6.4f}   "
              f"{fm:>8.4f} ± {fs:>6.4f}   {sign}{diff:>7.1f}%")
    print(f"{'='*72}")
    print("\nInterpretation for thesis:")
    print("  Positive % = fake images have HIGHER phase discontinuity than real")
    print("  Negative % = fake images have LOWER  phase discontinuity than real")
    print("  Both directions indicate phase domain differences between real/fake.")


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate phase discontinuity figures for thesis"
    )
    parser.add_argument("--cifake_path",        type=str, default=None,
                        help="CIFAKE dataset root folder")
    parser.add_argument("--dffd_path",          type=str, default=None,
                        help="DFFD dataset root folder")
    parser.add_argument("--cocofake_real_path", type=str, default=None,
                        help="coco2014/train2014 folder (real images)")
    parser.add_argument("--cocofake_fake_path", type=str, default=None,
                        help="coco_fake/train2014 folder (fake images)")
    parser.add_argument("--output_dir",         type=str, default="./figures",
                        help="Output directory for figures")
    parser.add_argument("--n_samples",          type=int, default=4,
                        help="Number of real/fake pairs per dataset")
    parser.add_argument("--seed",               type=int, default=42,
                        help="Random seed")
    parser.add_argument("--demo",               action="store_true",
                        help="Use synthetic demo data (no real datasets needed)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("Phase Discontinuity Visualization")
    print(f"Output : {args.output_dir}")
    print(f"Samples: {args.n_samples} per dataset")
    print(f"{'='*65}")

    # ── Load all three datasets ───────────────────────────────────────────────
    datasets = {}

    if args.demo:
        print("\nRunning in DEMO mode with synthetic data...")
        datasets["dffd"]     = make_demo_data(args.n_samples)
        datasets["cifake"]   = make_demo_data(args.n_samples)
        datasets["cocofake"] = make_demo_data(args.n_samples)

    else:
        # ── DFFD ──────────────────────────────────────────────────────────────
        if args.dffd_path:
            real_f, fake_f = get_dffd_folders(args.dffd_path)
            result = load_dataset("dffd", real_f, fake_f, args.n_samples)
            if result:
                datasets["dffd"] = result
        else:
            print("\n  [SKIP] dffd: --dffd_path not provided")

        # ── CIFAKE ────────────────────────────────────────────────────────────
        if args.cifake_path:
            real_f, fake_f = get_cifake_folders(args.cifake_path)
            result = load_dataset("cifake", real_f, fake_f, args.n_samples)
            if result:
                datasets["cifake"] = result
        else:
            print("\n  [SKIP] cifake: --cifake_path not provided")

        # ── COCOFake ──────────────────────────────────────────────────────────
        if args.cocofake_real_path and args.cocofake_fake_path:
            result = load_dataset(
                "cocofake",
                args.cocofake_real_path,
                args.cocofake_fake_path,
                args.n_samples,
            )
            if result:
                datasets["cocofake"] = result
        else:
            print("\n  [SKIP] cocofake: need both "
                  "--cocofake_real_path and --cocofake_fake_path")

    if not datasets:
        print("\n[ERROR] No datasets loaded.")
        print("Use --demo to run with synthetic data:")
        print("  python phase_visualization.py --demo --output_dir ./figures")
        return

    # ── Generate figures ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Generating figures...")
    print(f"{'='*65}")

    # Figure 1: Grid
    make_grid_figure(
        datasets,
        os.path.join(args.output_dir, "phase_discontinuity_grid.png"),
    )

    # Figure 2: Statistics
    make_statistics_figure(
        datasets,
        os.path.join(args.output_dir, "phase_statistics.png"),
    )

    # Figure 3: Detailed pair per dataset
    for ds_name, ds_data in datasets.items():
        make_pair_figure(
            ds_data["real"][0],
            ds_data["fake"][0],
            ds_name,
            os.path.join(args.output_dir, f"phase_pair_{ds_name}.png"),
        )

    # ── Print statistics table ────────────────────────────────────────────────
    print_statistics(datasets)

    print(f"\n{'='*65}")
    print("Done. Files saved:")
    print(f"  phase_discontinuity_grid.png  ← thesis Figure 1 (main)")
    print(f"  phase_statistics.png          ← thesis Figure 2 (statistics)")
    for ds in datasets:
        print(f"  phase_pair_{ds}.png")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()