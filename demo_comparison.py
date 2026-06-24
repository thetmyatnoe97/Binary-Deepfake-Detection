"""
demo_comparison_minimal.py - Phase4DFD vs Baseline Deepfake Detection
================================================================================
Yuan Ze University - Department of CS&E - Thesis 2026

FEATURES:
  - ✅ 9 Thesis Models (with Attention) vs 6 Baseline Models (without Attention)
  - ✅ Loads from separate model.py (thesis) and model_baseline.py (baseline)
  - ✅ RGB→BGR conversion for model input
  - ✅ Automatic saving of uncertain predictions (< 100% confidence)
  - ✅ Side-by-side comparison visualization

Run:
    python demo_comparison_minimal.py

REQUIREMENTS:
  - model.py         : Thesis BNext4DFR (with attention modules)
  - model_baseline.py: Baseline BNext4DFR (without attention modules)
  - checkpoints/     : trained model files
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import inspect

# Import both model versions
try:
    from model import BNext4DFR as BNext4DFR_Thesis
    print("✓ Loaded thesis BNext4DFR (with attention)")
except ImportError as e:
    print(f"✗ Failed to load thesis model: {e}")
    BNext4DFR_Thesis = None

try:
    from model_baseline import BNext4DFR as BNext4DFR_Baseline
    print("✓ Loaded baseline BNext4DFR (without attention)")
except ImportError as e:
    print(f"✗ Failed to load baseline model: {e}")
    BNext4DFR_Baseline = None


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = "checkpoints"

DATASET_CHOICES = ["DFFD", "CIFAKE", "COCOFake"]
DATASET_DIRS    = {"DFFD": "dffd", "CIFAKE": "cifake", "COCOFake": "cocofake"}
DATASET_PREFIXES = {"DFFD": "dffd", "CIFAKE": "cifake", "COCOFake": "coco_fake"}

# =============================================================================
# THESIS MODELS (9 configurations with Attention)
# =============================================================================

THESIS_MODELS = {
    "PhaseDFD-T": {
        "ckpt_prefix": "phasedfd",
        "backbone": "BNext-T",
        "fft": True, "lbp": True, "attn": True, "pos": "before_backbone",
        "desc": "Phase-Aware Input Attention - BNext-Tiny",
    },
    "PhaseDFD-S": {
        "ckpt_prefix": "phasedfd",
        "backbone": "BNext-S",
        "fft": True, "lbp": True, "attn": True, "pos": "before_backbone",
        "desc": "Phase-Aware Input Attention - BNext-Small",
    },
    "PhaseDFD-M": {
        "ckpt_prefix": "phasedfd",
        "backbone": "BNext-M",
        "fft": True, "lbp": True, "attn": True, "pos": "before_backbone",
        "desc": "Phase-Aware Input Attention - BNext-Middle",
    },
    "DualDFD-T": {
        "ckpt_prefix": "dualdfd",
        "backbone": "BNext-T",
        "fft": True, "lbp": True, "attn": True, "pos": "after_backbone",
        "desc": "Channel-Spatial Feature Attention - BNext-Tiny",
    },
    "DualDFD-S": {
        "ckpt_prefix": "dualdfd",
        "backbone": "BNext-S",
        "fft": True, "lbp": True, "attn": True, "pos": "after_backbone",
        "desc": "Channel-Spatial Feature Attention - BNext-Small",
    },
    "DualDFD-M": {
        "ckpt_prefix": "dualdfd",
        "backbone": "BNext-M",
        "fft": True, "lbp": True, "attn": True, "pos": "after_backbone",
        "desc": "Channel-Spatial Feature Attention - BNext-Middle",
    },
    "FullDFD-T": {
        "ckpt_prefix": "fulldfd",
        "backbone": "BNext-T",
        "fft": True, "lbp": True, "attn": True, "pos": "both",
        "desc": "Both Attention Modules - BNext-Tiny",
    },
    "FullDFD-S": {
        "ckpt_prefix": "fulldfd",
        "backbone": "BNext-S",
        "fft": True, "lbp": True, "attn": True, "pos": "both",
        "desc": "Both Attention Modules - BNext-Small",
    },
    "FullDFD-M": {
        "ckpt_prefix": "fulldfd",
        "backbone": "BNext-M",
        "fft": True, "lbp": True, "attn": True, "pos": "both",
        "desc": "Both Attention Modules - BNext-Middle",
    },
}

# =============================================================================
# BASELINE MODELS (6 configurations - BNext without Attention)
# =============================================================================

BASELINE_MODELS = {
    "BNext-T": {
        "ckpt_prefix": None,
        "backbone": "BNext-T",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": True,
        "desc": "BNext-Tiny (FFT+LBP, no attention)",
    },
    "BNext-T-Unfrozen": {
        "ckpt_prefix": None,
        "backbone": "BNext-T",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": False,
        "desc": "BNext-Tiny Unfrozen (FFT+LBP, no attention)",
    },
    "BNext-S": {
        "ckpt_prefix": None,
        "backbone": "BNext-S",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": True,
        "desc": "BNext-Small (FFT+LBP, no attention)",
    },
    "BNext-S-Unfrozen": {
        "ckpt_prefix": None,
        "backbone": "BNext-S",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": False,
        "desc": "BNext-Small Unfrozen (FFT+LBP, no attention)",
    },
    "BNext-M": {
        "ckpt_prefix": None,
        "backbone": "BNext-M",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": True,
        "desc": "BNext-Middle (FFT+LBP, no attention)",
    },
    "BNext-M-Unfrozen": {
        "ckpt_prefix": None,
        "backbone": "BNext-M",
        "fft": True, "lbp": True, "attn": False, "pos": "none",
        "backbone_frozen": False,
        "desc": "BNext-Middle Unfrozen (FFT+LBP, no attention)",
    },
}

MODEL_META = {**THESIS_MODELS, **BASELINE_MODELS}

RESOLUTION = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRAIN_TRANSFORM = T.Compose([
    T.Resize(RESOLUTION + RESOLUTION // 8, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(RESOLUTION),
    T.ToTensor(),
])

_model_cache = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_filename(model_name: str, dataset_name: str) -> str:
    """Generate checkpoint filename with .ckpt extension."""
    if model_name not in MODEL_META:
        return None
    meta = MODEL_META[model_name]
    size = model_name.split("-")[1].lower()
    
    if meta["ckpt_prefix"] is not None:
        return f"{meta['ckpt_prefix']}_{size}.ckpt"
    
    dataset_prefix = DATASET_PREFIXES.get(dataset_name, dataset_name.lower())
    if "Unfrozen" in model_name:
        return f"{dataset_prefix}_{size}_unfrozen.ckpt"
    else:
        return f"{dataset_prefix}_{size}.ckpt"


def prep_image(pil_img: Image.Image):
    """Preprocess image: RGB→BGR conversion to match training data."""
    img = pil_img.convert("RGB")
    tensor = _TRAIN_TRANSFORM(img).unsqueeze(0)  # (1,3,224,224) RGB
    
    # CRITICAL: Convert RGB → BGR to match training (OpenCV/BGR format)
    tensor = tensor[:, [2, 1, 0], :, :]  # Swap channels: RGB → BGR
    
    # For display, convert back to RGB so it looks correct
    display_bgr = tensor[0].permute(1, 2, 0).numpy() * 255
    display = display_bgr[:, :, [2, 1, 0]]  # Back to RGB for display
    display = display.astype(np.uint8)
    
    return tensor, display


@torch.no_grad()
def predict(model, tensor):
    """Run inference. Returns (prob_real, pred, logit_value)."""
    model.eval()
    out = model(tensor.to(DEVICE))
    logit = out["logits"][0, 0]
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob > 0.5 else 0
    return prob, pred, logit.item()


def save_uncertain_prediction(pil_image, fig_image, model_name, dataset_name, prob, pred, confidence):
    """Save both original input image and result comparison figure."""
    from datetime import datetime
    
    output_dir = "uncertain_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    verdict = "REAL" if pred == 1 else "FAKE"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original input image
    input_filename = f"{timestamp}_{model_name}_{verdict}_{confidence:.1f}pct_INPUT.png"
    input_filepath = os.path.join(output_dir, input_filename)
    pil_image.save(input_filepath)
    print(f"  [✓] Saved input image: {input_filename}")
    
    # Save result comparison figure
    result_filename = f"{timestamp}_{model_name}_{verdict}_{confidence:.1f}pct_RESULT.png"
    result_filepath = os.path.join(output_dir, result_filename)
    fig_image.save(result_filepath)
    print(f"  [✓] Saved result figure: {result_filename}")
    
    # Log to text file
    log_path = os.path.join(output_dir, "predictions_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} | Model: {model_name} ({dataset_name}) | "
                f"Verdict: {verdict} | Confidence: {confidence:.2f}% | P(Real): {prob*100:.2f}%\n")


def load_model(model_name: str, dataset_name: str):
    """Load checkpoint with correct model class (thesis or baseline)."""
    cache_key = f"{model_name}_{dataset_name}"
    if cache_key in _model_cache:
        print(f"  [✓] Loaded from cache: {model_name} ({dataset_name})")
        return _model_cache[cache_key]
    
    # Determine which model class to use
    if model_name in THESIS_MODELS:
        model_class = BNext4DFR_Thesis
        model_type = "THESIS (with Attention)"
    else:
        model_class = BNext4DFR_Baseline
        model_type = "BASELINE (no Attention)"
    
    if model_class is None:
        print(f"  [✗] {model_type} model class not loaded")
        return None
    
    meta      = MODEL_META[model_name]
    ds_dir    = DATASET_DIRS[dataset_name]
    ckpt_name = get_checkpoint_filename(model_name, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, ds_dir, ckpt_name)
    
    print(f"\n  Loading {model_name} [{dataset_name}] ({model_type})...")
    print(f"    Path: {ckpt_path}")
    
    # Check if file exists
    if not os.path.exists(ckpt_path):
        print(f"    [✗] File not found")
        return None
    
    print(f"    [✓] File found ({os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB)")
    
    # DEBUG: Check hyperparameters
    try:
        temp_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = temp_ckpt.get("hyper_parameters", {})
        print(f"    Hyperparameters: {hparams}")
        if 'pos_weight' in hparams:
            print(f"    pos_weight={hparams['pos_weight']:.4f}")
    except Exception:
        pass
    
    # Try Lightning checkpoint first
    for strict in [True, False]:
        try:
            print(f"    Trying Lightning load (strict={strict})...")
            model = model_class.load_from_checkpoint(
                ckpt_path, strict=strict, map_location=DEVICE)
            model.to(DEVICE).eval()
            _model_cache[cache_key] = model
            print(f"    [✓] Loaded successfully!")
            return model
        except Exception as e:
            print(f"    Lightning load failed: {str(e)[:100]}")
            continue
    
    # Fallback: manual state dict loading
    try:
        print(f"    Trying manual state_dict load...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})
        
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_hparams = {k: v for k, v in hparams.items() if k in valid_params}
        
        model = model_class(**filtered_hparams)
        state_dict = ckpt["state_dict"]
        model_state = model.state_dict()
        
        # Remove mismatched keys
        keys_to_remove = []
        for key in state_dict.keys():
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del state_dict[key]
        
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()
        _model_cache[cache_key] = model
        print(f"    [✓] Loaded successfully!")
        return model
    except Exception as e:
        print(f"    [✗] Manual load failed: {e}")
        return None


# =============================================================================
# COMPARISON FIGURE
# =============================================================================

def make_comparison_figure(display, 
                          prob_left, pred_left, model_left, logit_left,
                          prob_right, pred_right, model_right, logit_right) -> Image.Image:
    """Create comparison: input image + large prediction results."""
    
    label_left  = "REAL" if pred_left == 1 else "FAKE"
    label_right = "REAL" if pred_right == 1 else "FAKE"
    color_left  = "#16a34a" if pred_left == 1 else "#dc2626"
    color_right = "#16a34a" if pred_right == 1 else "#dc2626"
    
    conf_left = prob_left * 100 if pred_left == 1 else (1 - prob_left) * 100
    conf_right = prob_right * 100 if pred_right == 1 else (1 - prob_right) * 100
    
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    
    # Title bar
    ax_title = fig.add_axes([0.05, 0.92, 0.9, 0.07])
    ax_title.set_facecolor("#1e293b")
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "DEEPFAKE DETECTION RESULT",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    
    # Input image
    ax_img = fig.add_axes([0.05, 0.08, 0.3, 0.8])
    ax_img.imshow(display)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Input Image", fontsize=14, fontweight="bold", pad=10)
    for sp in ax_img.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#1e293b")
        sp.set_linewidth(2)
    
    # Thesis model result
    ax_thesis = fig.add_axes([0.38, 0.08, 0.28, 0.8])
    ax_thesis.set_facecolor("#dcfce7" if pred_left == 1 else "#fee2e2")
    ax_thesis.set_xlim(0, 1)
    ax_thesis.set_ylim(0, 1)
    ax_thesis.axis("off")
    
    for sp in ax_thesis.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color_left)
        sp.set_linewidth(4)
    
    ax_thesis.text(0.5, 0.85, "THESIS MODEL", ha="center", va="center",
                   fontsize=12, fontweight="bold", color="#0369a1", 
                   transform=ax_thesis.transAxes,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#0369a1", linewidth=2))
    
    ax_thesis.text(0.5, 0.65, model_left, ha="center", va="center",
                   fontsize=13, fontweight="bold", color="#1e293b",
                   transform=ax_thesis.transAxes)
    
    ax_thesis.text(0.5, 0.52, label_left, ha="center", va="center",
                   fontsize=72, fontweight="900", color=color_left,
                   transform=ax_thesis.transAxes)
    
    ax_thesis.text(0.5, 0.35, f"{conf_left:.1f}%", ha="center", va="center",
                   fontsize=36, fontweight="bold", color=color_left,
                   transform=ax_thesis.transAxes)
    
    real_pct_left = prob_left * 100
    fake_pct_left = (1 - prob_left) * 100
    ax_thesis.text(0.5, 0.18, f"Real: {real_pct_left:.1f}% | Fake: {fake_pct_left:.1f}%", 
                   ha="center", va="center", fontsize=13, color="#1e293b", fontweight="600",
                   transform=ax_thesis.transAxes)
    
    # Baseline model result
    ax_baseline = fig.add_axes([0.68, 0.08, 0.28, 0.8])
    ax_baseline.set_facecolor("#dcfce7" if pred_right == 1 else "#fee2e2")
    ax_baseline.set_xlim(0, 1)
    ax_baseline.set_ylim(0, 1)
    ax_baseline.axis("off")
    
    for sp in ax_baseline.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color_right)
        sp.set_linewidth(4)
    
    ax_baseline.text(0.5, 0.85, "BASELINE MODEL", ha="center", va="center",
                     fontsize=12, fontweight="bold", color="#be185d",
                     transform=ax_baseline.transAxes,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce7f3", edgecolor="#be185d", linewidth=2))
    
    ax_baseline.text(0.5, 0.65, model_right, ha="center", va="center",
                     fontsize=13, fontweight="bold", color="#1e293b",
                     transform=ax_baseline.transAxes)
    
    ax_baseline.text(0.5, 0.52, label_right, ha="center", va="center",
                     fontsize=72, fontweight="900", color=color_right,
                     transform=ax_baseline.transAxes)
    
    ax_baseline.text(0.5, 0.35, f"{conf_right:.1f}%", ha="center", va="center",
                     fontsize=36, fontweight="bold", color=color_right,
                     transform=ax_baseline.transAxes)
    
    real_pct_right = prob_right * 100
    fake_pct_right = (1 - prob_right) * 100
    ax_baseline.text(0.5, 0.18, f"Real: {real_pct_right:.1f}% | Fake: {fake_pct_right:.1f}%", 
                     ha="center", va="center", fontsize=13, color="#1e293b", fontweight="600",
                     transform=ax_baseline.transAxes)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =============================================================================
# COMPARISON INFERENCE
# =============================================================================

def compare(pil_image, thesis_model, thesis_dataset, baseline_model, baseline_dataset):
    """Run inference on both models and return comparison."""
    
    if pil_image is None:
        return None
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: {thesis_model} vs {baseline_model}")
    print(f"{'='*70}")
    
    tensor, display = prep_image(pil_image)
    print(f"Image tensor: shape={tuple(tensor.shape)}, min={tensor.min():.3f}, max={tensor.max():.3f} [BGR]")
    
    # Load both models with verbose output
    print(f"\n[THESIS MODEL]")
    model_thesis = load_model(thesis_model, thesis_dataset)
    
    print(f"\n[BASELINE MODEL]")
    model_baseline = load_model(baseline_model, baseline_dataset)
    
    # Check if both models loaded
    if model_thesis is None:
        error_msg = f"""
        <div style='color:#ef4444;padding:20px;font-family:monospace;
                    background:#fee2e2;border-radius:8px;
                    border:2px solid #fca5a5;text-align:center;'>
          <b>❌ THESIS MODEL FAILED TO LOAD</b><br><br>
          {thesis_model} [{thesis_dataset}]<br>
          Check console output for details.
        </div>
        """
        return None
    
    if model_baseline is None:
        error_msg = f"""
        <div style='color:#ef4444;padding:20px;font-family:monospace;
                    background:#fee2e2;border-radius:8px;
                    border:2px solid #fca5a5;text-align:center;'>
          <b>❌ BASELINE MODEL FAILED TO LOAD</b><br><br>
          {baseline_model} [{baseline_dataset}]<br>
          Check console output for details.
        </div>
        """
        return None
    
    # Run both models with debug output
    print(f"\n[INFERENCE]")
    prob_thesis, pred_thesis, logit_thesis = predict(model_thesis, tensor)
    thesis_conf = prob_thesis * 100 if pred_thesis == 1 else (1 - prob_thesis) * 100
    print(f"  Thesis: pred={'REAL' if pred_thesis==1 else 'FAKE'}, confidence={thesis_conf:.2f}%")
    
    prob_baseline, pred_baseline, logit_baseline = predict(model_baseline, tensor)
    baseline_conf = prob_baseline * 100 if pred_baseline == 1 else (1 - prob_baseline) * 100
    print(f"  Baseline: pred={'REAL' if pred_baseline==1 else 'FAKE'}, confidence={baseline_conf:.2f}%")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create comparison figure FIRST (before saving)
    fig_img = make_comparison_figure(display,
                                     prob_thesis, pred_thesis, thesis_model, logit_thesis,
                                     prob_baseline, pred_baseline, baseline_model, logit_baseline)
    
    # Save uncertain predictions (confidence < 100%)
    print(f"\n[SAVING UNCERTAIN PREDICTIONS]")
    if thesis_conf < 100.0:
        save_uncertain_prediction(pil_image, fig_img, thesis_model, thesis_dataset, 
                                prob_thesis, pred_thesis, thesis_conf)
    else:
        print(f"  Thesis: 100% confidence (not saved)")
    
    if baseline_conf < 100.0:
        save_uncertain_prediction(pil_image, fig_img, baseline_model, baseline_dataset, 
                                prob_baseline, pred_baseline, baseline_conf)
    else:
        print(f"  Baseline: 100% confidence (not saved)")
    
    return fig_img


# =============================================================================
# CSS
# =============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;600;700&display=swap');
body, .gradio-container {
    background: #f1f5f9 !important; color: #1e293b !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gr-panel, .gr-box, .gr-form, .gr-block {
    background: #ffffff !important; border-color: #e2e8f0 !important;
}
label, .gr-label, .gr-input label, p {
    color: #1e293b !important; font-size: 14px !important;
    font-weight: 600 !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #4f46e5, #0ea5e9) !important;
    border: none !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; letter-spacing: 1px !important;
    font-size: 16px !important; padding: 16px !important;
    border-radius: 8px !important; color: white !important;
}
.gr-button-primary:hover {
    opacity: 0.90 !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(79,70,229,0.35) !important;
}
select, input, textarea {
    background: #ffffff !important; border-color: #cbd5e1 !important;
    color: #1e293b !important; font-size: 13px !important;
}
footer { display: none !important; }
"""


# =============================================================================
# UI
# =============================================================================

def build_ui():
    with gr.Blocks(css=CSS, title="Phase4DFD - Deepfake Detection") as demo:
        gr.HTML("""
        <div style="text-align:center;padding:20px 0;
                    background:linear-gradient(135deg,#1e293b,#0f172a);
                    border-radius:12px;margin-bottom:16px;">
          <div style="font-family:'Space Mono',monospace;font-size:12px;
                      color:#94a3b8;letter-spacing:4px;margin-bottom:8px;">
            YUAN ZE UNIVERSITY - MS THESIS 2026
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:32px;
                      font-weight:700;color:#ffffff;letter-spacing:2px;">
            DEEPFAKE DETECTION
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:13px;
                      color:#cbd5e1;margin-top:8px;letter-spacing:1px;">
            Thesis Model vs Baseline Comparison
          </div>
        </div>
        """)

        # Image Upload
        with gr.Column():
            gr.Markdown("### Upload Test Image", elem_classes="section-header")
            image_input = gr.Image(type="pil", label="Image", height=300)

        # Model Selection (Side by Side)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Thesis Model", elem_classes="section-header")
                thesis_model = gr.Dropdown(
                    choices=list(THESIS_MODELS.keys()),
                    value="PhaseDFD-S",
                    label="Select Model",
                    info="PhaseDFD / DualDFD / FullDFD with Attention")
                thesis_dataset = gr.Dropdown(
                    choices=DATASET_CHOICES,
                    value="DFFD",
                    label="Dataset Checkpoint",
                    info="Training dataset")

            with gr.Column():
                gr.Markdown("### Baseline Model", elem_classes="section-header")
                baseline_model = gr.Dropdown(
                    choices=list(BASELINE_MODELS.keys()),
                    value="BNext-S",
                    label="Select Model",
                    info="BNext without Attention")
                baseline_dataset = gr.Dropdown(
                    choices=DATASET_CHOICES,
                    value="DFFD",
                    label="Dataset Checkpoint",
                    info="Training dataset")

        # Run Button
        gr.Markdown("", elem_classes="spacer")
        compare_btn = gr.Button("🔍 DETECT & COMPARE", variant="primary", size="lg")

        # Results
        gr.Markdown("### Detection Result", elem_classes="section-header")
        result_image = gr.Image(type="pil", label="Result", height=600)

        # Connect button
        compare_btn.click(
            fn=compare,
            inputs=[image_input, thesis_model, thesis_dataset, baseline_model, baseline_dataset],
            outputs=[result_image])

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("Phase4DFD - Deepfake Detection Comparison Demo")
    print(f"Device: {DEVICE}")
    print(f"Preprocessing: RGB→BGR conversion enabled")
    print(f"{'='*70}\n")

    # Check both model files
    missing_files = []
    if not os.path.exists("model.py"):
        print("❌ ERROR: model.py (thesis) not found")
        missing_files.append("model.py")
    else:
        print("✅ Found model.py (thesis with attention)")
    
    if not os.path.exists("model_baseline.py"):
        print("❌ ERROR: model_baseline.py (baseline) not found")
        missing_files.append("model_baseline.py")
    else:
        print("✅ Found model_baseline.py (baseline without attention)")
    
    if missing_files:
        print(f"\n❌ Missing: {', '.join(missing_files)}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"📊 Models available:")
    print(f"  • Thesis models: {len(THESIS_MODELS)} configurations")
    print(f"    - With Phase-Aware & Channel-Spatial Attention")
    print(f"  • Baseline models: {len(BASELINE_MODELS)} configurations")
    print(f"    - BNext backbone only (no attention)")
    print(f"{'='*70}\n")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=False, show_error=True, inbrowser=True)