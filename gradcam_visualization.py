"""
gradcam_visualization.py — GradCAM Class Activation Map Visualization
======================================================================
Generates Figure 2 of the thesis: GradCAM heatmaps showing WHERE the
model looks when classifying real vs fake images across DFFD, CIFAKE,
and COCOFake datasets.
 
This directly addresses:
  - Committee: "show class activation maps"
  - Reviewer 3: "class activation maps are not provided, making it
    impossible to visually demonstrate whether phase-aware attention
    truly guides the model to focus on the forged regions"
 
How GradCAM works:
  1. Forward pass through the model to get prediction
  2. Backpropagate gradients to the TARGET layer (last conv layer of backbone)
  3. Global average pool the gradients → channel weights
  4. Weighted sum of feature maps → raw heatmap
  5. ReLU + normalize + resize to input resolution
  6. Overlay on original image
 
Outputs:
  gradcam_grid.png          — 3-dataset comparison grid (thesis Figure 2)
  gradcam_pair_dffd.png     — detailed DFFD pair
  gradcam_pair_cifake.png   — detailed CIFAKE pair
  gradcam_pair_cocofake.png — detailed COCOFake pair
 
Usage:
    python gradcam_visualization.py ^
      --cifake_ckpt   checkpoints/cifake/phasedfd_m.ckpt ^
      --dffd_ckpt     checkpoints/dffd/phasedfd_m.ckpt ^
      --cocofake_ckpt checkpoints/cocofake/phasedfd_m.ckpt ^
      --cifake_path        D:/datasets/cifake ^
      --dffd_path          D:/datasets/dffd ^
      --cocofake_real_path D:/datasets/coco2014/train2014 ^
      --cocofake_fake_path D:/datasets/coco_fake/train2014 ^
      --output_dir    ./figures ^
      --n_samples     3
 
Requirements:
    pip install torch torchvision timm lightning matplotlib pillow numpy
    pip install grad-cam   (pytorch-grad-cam library)
"""
 
import os
import sys
import argparse
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from PIL import Image
 
import torch
import torch.nn.functional as F
import torchvision.transforms as T
 
 
# ── Constants ─────────────────────────────────────────────────────────────────
RESOLUTION = 224
DPI        = 300
ALPHA      = 0.5    # heatmap overlay transparency
 
DATASET_LABELS = {
    "dffd":     "DFFD\n(GAN face forgery)",
    "cifake":   "CIFAKE\n(Diffusion low-res)",
    "cocofake": "COCOFake\n(Diffusion natural scene)",
}
 
# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
 
 
# ── Image utilities ───────────────────────────────────────────────────────────
 
def load_image_tensor(path: str) -> tuple:
    """
    Load image and return:
        tensor : (1, 3, H, W) normalized tensor for model input
        display: (H, W, 3) uint8 numpy array for visualization
    """
    img    = Image.open(path).convert("RGB")
    img    = img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    display = np.array(img)
 
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    tensor = tfm(img).unsqueeze(0)   # (1, 3, H, W)
    return tensor, display
 
 
def find_images(folder: str, n: int) -> list:
    """Find up to n images recursively using os.walk."""
    if not os.path.exists(folder):
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp",
                  ".JPG", ".JPEG", ".PNG"}
    found = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] in extensions:
                found.append(os.path.join(root, f))
        if len(found) >= n * 20:
            break
    random.shuffle(found)
    return found[:n]
 
 
# ── GradCAM implementation ────────────────────────────────────────────────────
 
class GradCAM:
    """
    GradCAM for BNext4DFR model.
 
    Hooks into the last convolutional layer of the BNext backbone
    to capture activations and gradients during forward/backward pass.
 
    Target layer: base_model.layer4 (last residual block of BNext)
    This produces a 7×7 spatial feature map for 224×224 input.
    """
 
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model   = model
        self.device  = device
        self.grads   = None
        self.acts    = None
        self.handles = []
 
        # Find the target layer — last conv block of BNext backbone
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise RuntimeError(
                "Could not find target layer in BNext backbone.\n"
                "Expected: base_model.layer4 or similar last conv block."
            )
        print(f"  GradCAM target layer: {target_layer.__class__.__name__}")
 
        # Register hooks
        self.handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
 
    def _find_target_layer(self):
        """Find the last convolutional layer in the BNext backbone."""
        base = self.model.base_model
 
        # Try common BNext layer names in order of preference
        for attr in ["layer4", "layer3", "features", "blocks"]:
            if hasattr(base, attr):
                layer = getattr(base, attr)
                print(f"  Found backbone attribute: {attr}")
                return layer
 
        # Fallback: return the last module that has parameters
        last_layer = None
        for name, module in base.named_modules():
            if len(list(module.parameters())) > 0:
                last_layer = module
        return last_layer
 
    def _save_activation(self, module, input, output):
        self.acts = output.detach()
 
    def _save_gradient(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()
 
    def __call__(self, tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """
        Compute GradCAM heatmap for the given input tensor.
 
        Args:
            tensor      : (1, 3, H, W) normalized image tensor
            target_class: 0=fake, 1=real. None = use predicted class.
 
        Returns:
            heatmap : (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        tensor = tensor.to(self.device).requires_grad_(True)
 
        # Forward pass
        self.model.zero_grad()
        output = self.model(tensor)
        logit  = output["logits"][0, 0]   # scalar logit
        prob   = torch.sigmoid(logit).item()
        pred   = 1 if prob > 0.5 else 0
 
        # Use target class or predicted class
        if target_class is None:
            target_class = pred
 
        # Score to backpropagate:
        # For class=1 (real): backprop through the positive logit
        # For class=0 (fake): backprop through the negative logit
        if target_class == 1:
            score = logit
        else:
            score = -logit
 
        # Backward pass
        self.model.zero_grad()
        score.backward()
 
        # Check gradients and activations were captured
        if self.grads is None or self.acts is None:
            raise RuntimeError(
                "GradCAM hooks did not capture gradients/activations.\n"
                "The target layer may not be in the forward path."
            )
 
        # Pool gradients over spatial dimensions → channel weights
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
 
        # Weighted combination of activation maps
        cam = (weights * self.acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)                                      # only positive influence
 
        # Normalize to [0, 1]
        cam = cam.squeeze()   # (H, W)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)
 
        # Resize to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(RESOLUTION, RESOLUTION),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
 
        return cam, prob, pred
 
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
 
 
# ── Overlay function ──────────────────────────────────────────────────────────
 
def apply_heatmap(image: np.ndarray, cam: np.ndarray,
                  alpha: float = ALPHA) -> np.ndarray:
    """
    Overlay GradCAM heatmap on the original image.
    image : (H, W, 3) uint8
    cam   : (H, W) float in [0, 1]
    Returns (H, W, 3) uint8 overlay
    """
    # Apply colormap (jet: blue=low attention, red=high attention)
    cmap    = plt.cm.jet
    heatmap = cmap(cam)[:, :, :3]           # (H, W, 3) float in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
 
    # Blend
    img_float  = image.astype(np.float32)
    heat_float = heatmap.astype(np.float32)
    overlay    = (1 - alpha) * img_float + alpha * heat_float
    return np.clip(overlay, 0, 255).astype(np.uint8)
 
 
# ── Model loading ─────────────────────────────────────────────────────────────
 
def load_model(ckpt_path: str, device: torch.device):
    """Load BNext4DFR from checkpoint."""
    try:
        sys.path.insert(0, os.getcwd())
        from model import BNext4DFR
    except ImportError:
        raise ImportError(
            "Cannot import model.py. Run this script from your project folder:\n"
            "  cd D:\\sweet\\binary_deepfake_detection\n"
            "  python gradcam_visualization.py ..."
        )
 
    print(f"  Loading: {ckpt_path}")
 
    # Try Lightning checkpoint first
    for strict in [True, False]:
        try:
            model = BNext4DFR.load_from_checkpoint(
                ckpt_path, strict=strict, map_location=device
            )
            model.to(device).eval()
            print(f"  ✓ Loaded (Lightning, strict={strict})")
            return model
        except Exception:
            continue
 
    # Manual fallback
    try:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})
        model = BNext4DFR(**hparams)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(device).eval()
        print(f"  ✓ Loaded (manual state_dict)")
        return model
    except Exception as e:
        raise RuntimeError(f"All loading methods failed: {e}")
 
 
# ── Dataset folder helpers ────────────────────────────────────────────────────
 
def find_subfolder(base: str, candidates: list) -> str:
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.exists(p):
            return p
    return base
 
 
def get_cifake_folders(base: str):
    real = find_subfolder(base, ["test/REAL", "test/real",
                                  "train/REAL", "train/real", "REAL", "real"])
    fake = find_subfolder(base, ["test/FAKE", "test/fake",
                                  "train/FAKE", "train/fake", "FAKE", "fake"])
    return real, fake
 
 
def get_dffd_folders(base: str):
    real = find_subfolder(base, ["real", "REAL", "Real",
                                  "test/real", "test/REAL",
                                  "train/real", "train/REAL"])
    fake = find_subfolder(base, ["fake", "FAKE", "Fake",
                                  "test/fake", "test/FAKE",
                                  "train/fake", "train/FAKE"])
    return real, fake
 
 
# ── Per-dataset processing ────────────────────────────────────────────────────
 
def process_dataset(name: str, model, gradcam: GradCAM,
                    real_folder: str, fake_folder: str,
                    n: int, device: torch.device) -> dict:
    """
    Run GradCAM on n real and n fake images from the given folders.
    Returns dict with results ready for plotting.
    """
    print(f"\n  Processing {name}...")
 
    real_paths = find_images(real_folder, n)
    fake_paths = find_images(fake_folder, n)
 
    if not real_paths or not fake_paths:
        print(f"  [SKIP] {name}: no images found")
        return None
 
    results = {"real": [], "fake": []}
 
    for label, paths in [("real", real_paths[:n]),
                          ("fake", fake_paths[:n])]:
        for path in paths:
            try:
                tensor, display = load_image_tensor(path)
 
                # GradCAM — target class matches the true label
                target = 1 if label == "real" else 0
                cam, prob, pred = gradcam(tensor, target_class=target)
 
                overlay = apply_heatmap(display, cam)
 
                correct = (pred == target)
                results[label].append({
                    "display":  display,
                    "cam":      cam,
                    "overlay":  overlay,
                    "prob":     prob,
                    "pred":     pred,
                    "correct":  correct,
                    "path":     path,
                    "label":    label,
                })
                status = "✓" if correct else "✗"
                pred_str = "REAL" if pred == 1 else "FAKE"
                print(f"    {status} {label.upper():4s} → pred:{pred_str} "
                      f"P(real)={prob:.3f}  {Path(path).name}")
 
            except Exception as e:
                import traceback
                print(f"    ✗ Failed: {Path(path).name} — {e}")
                traceback.print_exc()
 
    if not results["real"] or not results["fake"]:
        return None
 
    return results
 
 
# ── Figure: Grid ──────────────────────────────────────────────────────────────
 
def make_gradcam_grid(all_results: dict, output_path: str):
    """
    Main thesis figure: GradCAM grid.
    Rows = datasets. For each row shows one real pair and one fake pair.
    Each pair: Original | GradCAM overlay
    """
    n_ds  = len(all_results)
    ncols = 4   # Real orig | Real CAM | Fake orig | Fake CAM
 
    fig, axes = plt.subplots(
        n_ds, ncols,
        figsize=(ncols * 3.2, n_ds * 3.4),
        dpi=DPI,
    )
    fig.patch.set_facecolor("white")
 
    if n_ds == 1:
        axes = axes[np.newaxis, :]
 
    col_titles = [
        "Real image\n(authentic)",
        "GradCAM — Real\n(where model looks)",
        "Fake image\n(synthesized)",
        "GradCAM — Fake\n(where model looks)",
    ]
    col_colors = ["#2ecc71", "#2ecc71", "#e74c3c", "#e74c3c"]
 
    for row_idx, (ds_name, ds_results) in enumerate(all_results.items()):
        real_r = ds_results["real"][0]
        fake_r = ds_results["fake"][0]
 
        data_seq = [
            (real_r["display"], None),
            (real_r["display"], real_r["cam"]),
            (fake_r["display"], None),
            (fake_r["display"], fake_r["cam"]),
        ]
 
        for col_idx, (img, cam) in enumerate(data_seq):
            ax = axes[row_idx, col_idx]
 
            if cam is None:
                ax.imshow(img)
            else:
                overlay = apply_heatmap(img, cam)
                ax.imshow(overlay)
                # Overlay colorbar on last two cols
                if col_idx in [1, 3]:
                    sm = plt.cm.ScalarMappable(
                        cmap="jet",
                        norm=plt.Normalize(0, 1)
                    )
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax,
                                        fraction=0.046, pad=0.04)
                    cbar.set_ticks([0, 0.5, 1])
                    cbar.set_ticklabels(["Low", "Mid", "High"])
                    cbar.ax.tick_params(labelsize=6)
 
            ax.set_xticks([])
            ax.set_yticks([])
 
            # Column titles on first row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8.5,
                             pad=5, color="#333333")
 
            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS.get(ds_name, ds_name),
                    fontsize=8.5, fontweight="bold",
                    color="#222222", labelpad=6,
                )
 
            # Prediction annotation on GradCAM columns
            if col_idx in [1, 3]:
                r   = real_r if col_idx == 1 else fake_r
                lbl = "REAL" if r["pred"] == 1 else "FAKE"
                clr = "#2ecc71" if r["correct"] else "#e74c3c"
                ax.text(
                    0.5, 0.02,
                    f"Pred: {lbl}  P(real)={r['prob']:.3f}",
                    transform=ax.transAxes,
                    fontsize=7, color="white",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=clr, alpha=0.8),
                )
 
            # Border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col_colors[col_idx])
                spine.set_linewidth(1.5)
 
    # Legend
    legend_elements = [
        Patch(facecolor="white", edgecolor="#2ecc71",
              linewidth=2, label="Real (authentic)"),
        Patch(facecolor="white", edgecolor="#e74c3c",
              linewidth=2, label="Fake (synthesized)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=2,
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )
 
    fig.suptitle(
        "GradCAM Class Activation Maps: Where the Model Looks\n"
        "Heatmap colors: blue = low attention, red = high attention.\n"
        "Each pair shows the original image and the corresponding GradCAM overlay.",
        fontsize=9, y=1.02, color="#111111",
    )
 
    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  ✓ Grid saved: {output_path}")
 
 
# ── Figure: Detailed pair ─────────────────────────────────────────────────────
 
def make_gradcam_pair(ds_results: dict, ds_name: str,
                      output_path: str, n_pairs: int = 2):
    """
    Detailed figure for one dataset showing n_pairs real/fake pairs.
    Rows = pairs. Columns = Real orig | Real CAM | Fake orig | Fake CAM
    """
    n_pairs = min(n_pairs,
                  len(ds_results["real"]),
                  len(ds_results["fake"]))
 
    fig, axes = plt.subplots(
        n_pairs, 4,
        figsize=(14, n_pairs * 3.8),
        dpi=DPI,
    )
    fig.patch.set_facecolor("white")
 
    if n_pairs == 1:
        axes = axes[np.newaxis, :]
 
    col_titles = [
        "Real image",
        "GradCAM (Real)\nAttention heatmap",
        "Fake image",
        "GradCAM (Fake)\nAttention heatmap",
    ]
    col_colors = ["#2ecc71", "#2ecc71", "#e74c3c", "#e74c3c"]
 
    for pair_idx in range(n_pairs):
        real_r = ds_results["real"][pair_idx]
        fake_r = ds_results["fake"][pair_idx]
 
        data_seq = [
            (real_r["display"], None,          real_r),
            (real_r["display"], real_r["cam"], real_r),
            (fake_r["display"], None,          fake_r),
            (fake_r["display"], fake_r["cam"], fake_r),
        ]
 
        for col_idx, (img, cam, r) in enumerate(data_seq):
            ax = axes[pair_idx, col_idx]
 
            if cam is None:
                ax.imshow(img)
            else:
                ax.imshow(apply_heatmap(img, cam))
                sm = plt.cm.ScalarMappable(
                    cmap="jet", norm=plt.Normalize(0, 1)
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax,
                                    fraction=0.046, pad=0.04)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["Low", "High"])
                cbar.ax.tick_params(labelsize=7)
 
                # Prediction label
                lbl = "REAL" if r["pred"] == 1 else "FAKE"
                clr = "#27ae60" if r["correct"] else "#c0392b"
                mark = "✓" if r["correct"] else "✗"
                ax.set_xlabel(
                    f"{mark} Pred: {lbl}  |  P(real) = {r['prob']:.4f}",
                    fontsize=8, color=clr, labelpad=4,
                )
 
            ax.set_xticks([])
            ax.set_yticks([])
 
            if pair_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9,
                             pad=5, color="#333333")
 
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col_colors[col_idx])
                spine.set_linewidth(1.5)
 
    ds_label = DATASET_LABELS.get(ds_name, ds_name).replace("\n", " ")
    fig.suptitle(
        f"GradCAM Analysis — {ds_label}\n"
        "Red regions = high model attention (most influential for classification decision).\n"
        "Blue regions = low model attention (minimal influence on prediction).",
        fontsize=9.5, y=1.02, color="#111111",
    )
 
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Pair saved : {output_path}")
 
 
# ── Argument parser ───────────────────────────────────────────────────────────
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GradCAM class activation maps for thesis"
    )
    # Checkpoints
    parser.add_argument("--cifake_ckpt",   type=str, default=None,
                        help="CIFAKE checkpoint path (.ckpt)")
    parser.add_argument("--dffd_ckpt",     type=str, default=None,
                        help="DFFD checkpoint path (.ckpt)")
    parser.add_argument("--cocofake_ckpt", type=str, default=None,
                        help="COCOFake checkpoint path (.ckpt)")
 
    # Dataset paths
    parser.add_argument("--cifake_path",        type=str, default=None)
    parser.add_argument("--dffd_path",          type=str, default=None)
    parser.add_argument("--cocofake_real_path", type=str, default=None,
                        help="coco2014/train2014 (real images)")
    parser.add_argument("--cocofake_fake_path", type=str, default=None,
                        help="coco_fake/train2014 (fake images)")
 
    # Settings
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument("--n_samples",  type=int, default=3,
                        help="Real/fake pairs per dataset (default: 3)")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print(f"\n{'='*65}")
    print("GradCAM Class Activation Map Visualization")
    print(f"Device : {device}")
    print(f"Output : {args.output_dir}")
    print(f"{'='*65}")
 
    # ── Dataset configs ───────────────────────────────────────────────────────
    dataset_configs = {
        "dffd": {
            "ckpt":       args.dffd_ckpt,
            "real_folder": get_dffd_folders(args.dffd_path)[0]
                           if args.dffd_path else None,
            "fake_folder": get_dffd_folders(args.dffd_path)[1]
                           if args.dffd_path else None,
        },
        "cifake": {
            "ckpt":        args.cifake_ckpt,
            "real_folder": get_cifake_folders(args.cifake_path)[0]
                           if args.cifake_path else None,
            "fake_folder": get_cifake_folders(args.cifake_path)[1]
                           if args.cifake_path else None,
        },
        "cocofake": {
            "ckpt":        args.cocofake_ckpt,
            "real_folder": args.cocofake_real_path,
            "fake_folder": args.cocofake_fake_path,
        },
    }
 
    all_results = {}
 
    for ds_name, config in dataset_configs.items():
        ckpt = config["ckpt"]
        rf   = config["real_folder"]
        ff   = config["fake_folder"]
 
        if not ckpt:
            print(f"\n[SKIP] {ds_name}: no checkpoint provided "
                  f"(--{ds_name}_ckpt)")
            continue
        if not rf or not ff:
            print(f"\n[SKIP] {ds_name}: dataset path not provided")
            continue
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {ds_name}: checkpoint not found: {ckpt}")
            continue
 
        print(f"\n{'─'*65}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'─'*65}")
 
        try:
            # Load model
            model   = load_model(ckpt, device)
            gradcam = GradCAM(model, device)
 
            # Run GradCAM
            ds_results = process_dataset(
                ds_name, model, gradcam,
                rf, ff, args.n_samples, device,
            )
 
            gradcam.remove_hooks()
 
            if ds_results:
                all_results[ds_name] = ds_results
 
                # Individual detailed pair figure
                make_gradcam_pair(
                    ds_results, ds_name,
                    os.path.join(args.output_dir,
                                 f"gradcam_pair_{ds_name}.png"),
                    n_pairs=min(args.n_samples, 2),
                )
 
        except Exception as e:
            import traceback
            print(f"  [ERROR] {ds_name}: {e}")
            traceback.print_exc()
 
    # ── Grid figure ───────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*65}")
        print("Generating grid figure...")
        make_gradcam_grid(
            all_results,
            os.path.join(args.output_dir, "gradcam_grid.png"),
        )
 
        # ── Accuracy summary ──────────────────────────────────────────────────
        print(f"\n{'='*65}")
        print("Classification Accuracy on Visualization Samples:")
        print(f"{'='*65}")
        print(f"{'Dataset':<12} {'Real correct':>14} {'Fake correct':>14}")
        print("-" * 44)
        for ds_name, ds_r in all_results.items():
            r_correct = sum(r["correct"] for r in ds_r["real"])
            f_correct = sum(r["correct"] for r in ds_r["fake"])
            r_total   = len(ds_r["real"])
            f_total   = len(ds_r["fake"])
            print(f"{ds_name:<12} {r_correct}/{r_total} ({r_correct/r_total*100:.0f}%)"
                  f"        {f_correct}/{f_total} ({f_correct/f_total*100:.0f}%)")
        print(f"{'='*65}")
 
        print(f"\nDone. Files saved to: {args.output_dir}")
        print(f"  gradcam_grid.png          ← thesis Figure 2 (main)")
        for ds in all_results:
            print(f"  gradcam_pair_{ds}.png")
 
    else:
        print("\n[ERROR] No datasets processed successfully.")
        print("Check that checkpoint paths exist and dataset folders are correct.")
 
 
if __name__ == "__main__":
    main()
 