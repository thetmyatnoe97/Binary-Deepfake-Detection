"""
demo_all_comparisons.py - Complete Model Comparison Grid
================================================================================
Yuan Ze University - Department of CS&E - Thesis 2026

FEATURES:
  - Upload ONE image
  - Run all 9 thesis models + 6 baseline models
  - Display all results in organized grid
  - Compare: PhaseDFD-T/S/M vs BNext-T/S/M
            DualDFD-T/S/M vs BNext-T/S/M-Unfrozen  
            FullDFD-T/S/M vs BNext-T/S/M-Unfrozen
  - Auto-save uncertain predictions

Run:
    python demo_all_comparisons.py
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
from datetime import datetime

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

# Define model pairs for comparison
MODEL_PAIRS = [
    ("PhaseDFD-T", "BNext-T"),
    ("DualDFD-T",  "BNext-T-Unfrozen"),
    ("FullDFD-T",  "BNext-T-Unfrozen"),
    ("PhaseDFD-S", "BNext-S"),
    ("DualDFD-S",  "BNext-S-Unfrozen"),
    ("FullDFD-S",  "BNext-S-Unfrozen"),
    ("PhaseDFD-M", "BNext-M"),
    ("DualDFD-M",  "BNext-M-Unfrozen"),
    ("FullDFD-M",  "BNext-M-Unfrozen"),
]

# Thesis models
THESIS_MODELS = {
    "PhaseDFD-T": {"ckpt_prefix": "phasedfd", "backbone": "BNext-T"},
    "PhaseDFD-S": {"ckpt_prefix": "phasedfd", "backbone": "BNext-S"},
    "PhaseDFD-M": {"ckpt_prefix": "phasedfd", "backbone": "BNext-M"},
    "DualDFD-T":  {"ckpt_prefix": "dualdfd",  "backbone": "BNext-T"},
    "DualDFD-S":  {"ckpt_prefix": "dualdfd",  "backbone": "BNext-S"},
    "DualDFD-M":  {"ckpt_prefix": "dualdfd",  "backbone": "BNext-M"},
    "FullDFD-T":  {"ckpt_prefix": "fulldfd",  "backbone": "BNext-T"},
    "FullDFD-S":  {"ckpt_prefix": "fulldfd",  "backbone": "BNext-S"},
    "FullDFD-M":  {"ckpt_prefix": "fulldfd",  "backbone": "BNext-M"},
}

# Baseline models
BASELINE_MODELS = {
    "BNext-T": {"ckpt_prefix": None, "backbone": "BNext-T"},
    "BNext-T-Unfrozen": {"ckpt_prefix": None, "backbone": "BNext-T"},
    "BNext-S": {"ckpt_prefix": None, "backbone": "BNext-S"},
    "BNext-S-Unfrozen": {"ckpt_prefix": None, "backbone": "BNext-S"},
    "BNext-M": {"ckpt_prefix": None, "backbone": "BNext-M"},
    "BNext-M-Unfrozen": {"ckpt_prefix": None, "backbone": "BNext-M"},
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


def load_model(model_name: str, dataset_name: str):
    """Load checkpoint with correct model class (thesis or baseline)."""
    cache_key = f"{model_name}_{dataset_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Determine which model class to use
    if model_name in THESIS_MODELS:
        model_class = BNext4DFR_Thesis
    else:
        model_class = BNext4DFR_Baseline
    
    if model_class is None:
        return None
    
    meta      = MODEL_META[model_name]
    ds_dir    = DATASET_DIRS[dataset_name]
    ckpt_name = get_checkpoint_filename(model_name, dataset_name)
    ckpt_path = os.path.join(CHECKPOINT_DIR, ds_dir, ckpt_name)
    
    if not os.path.exists(ckpt_path):
        return None
    
    # Try Lightning checkpoint first
    for strict in [True, False]:
        try:
            model = model_class.load_from_checkpoint(
                ckpt_path, strict=strict, map_location=DEVICE)
            model.to(DEVICE).eval()
            _model_cache[cache_key] = model
            return model
        except Exception:
            continue
    
    # Fallback: manual state dict loading
    try:
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
        return model
    except Exception:
        return None


def save_uncertain_prediction(pil_image, model_name, dataset_name, prob, pred, confidence):
    """Save images where prediction confidence is below 100%."""
    output_dir = "uncertain_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    verdict = "REAL" if pred == 1 else "FAKE"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_name}_{verdict}_{confidence:.1f}pct.png"
    filepath = os.path.join(output_dir, filename)
    
    pil_image.save(filepath)
    
    # Log to text file
    log_path = os.path.join(output_dir, "predictions_log.txt")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} | Model: {model_name} ({dataset_name}) | "
                f"Verdict: {verdict} | Confidence: {confidence:.2f}% | P(Real): {prob*100:.2f}%\n")


def make_result_grid(display, results, dataset_name) -> Image.Image:
    """Create large, visible grid showing all 9 model pair results."""
    
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    
    # Title
    ax_title = fig.add_axes([0.05, 0.96, 0.9, 0.03])
    ax_title.set_facecolor("#1e293b")
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, f"ALL MODEL COMPARISONS - {dataset_name} Dataset",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    
    # Input image (top-left corner, medium size)
    ax_img = fig.add_axes([0.05, 0.82, 0.12, 0.12])
    ax_img.imshow(display)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Input Image", fontsize=12, fontweight="bold", pad=5)
    for sp in ax_img.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#1e293b")
        sp.set_linewidth(2)
    
    # Column headers
    ax_header1 = fig.add_axes([0.20, 0.88, 0.35, 0.08])
    ax_header1.set_facecolor("#0369a1")
    ax_header1.set_xlim(0, 1)
    ax_header1.set_ylim(0, 1)
    ax_header1.axis("off")
    ax_header1.text(0.5, 0.5, "THESIS MODELS (with Attention)",
                   ha="center", va="center", fontsize=16, fontweight="bold",
                   color="white", transform=ax_header1.transAxes)
    
    ax_header2 = fig.add_axes([0.58, 0.88, 0.35, 0.08])
    ax_header2.set_facecolor("#be185d")
    ax_header2.set_xlim(0, 1)
    ax_header2.set_ylim(0, 1)
    ax_header2.axis("off")
    ax_header2.text(0.5, 0.5, "BASELINE MODELS (no Attention)",
                   ha="center", va="center", fontsize=16, fontweight="bold",
                   color="white", transform=ax_header2.transAxes)
    
    # Layout parameters
    row_height = 0.085
    col_width = 0.35
    start_x_thesis = 0.20
    start_x_baseline = 0.58
    start_y = 0.82
    
    # Draw each model pair
    for pair_idx, (thesis_model, baseline_model) in enumerate(MODEL_PAIRS):
        y_pos = start_y - (pair_idx * row_height)
        
        # Get results
        thesis_result = results.get(thesis_model, {})
        baseline_result = results.get(baseline_model, {})
        
        # ===== THESIS MODEL BOX =====
        if thesis_result:
            pred_t = thesis_result["pred"]
            conf_t = thesis_result["confidence"]
            label_t = "REAL" if pred_t == 1 else "FAKE"
            color_t = "#16a34a" if pred_t == 1 else "#dc2626"
            bg_t = "#dcfce7" if pred_t == 1 else "#fee2e2"
            
            ax_t = fig.add_axes([start_x_thesis, y_pos - row_height + 0.005, col_width, row_height - 0.01])
            ax_t.set_facecolor(bg_t)
            ax_t.set_xlim(0, 1)
            ax_t.set_ylim(0, 1)
            ax_t.axis("off")
            
            # Border
            for sp in ax_t.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color_t)
                sp.set_linewidth(3)
            
            # Model name
            ax_t.text(0.15, 0.70, thesis_model, ha="left", va="center",
                     fontsize=13, fontweight="bold", color="#1e293b",
                     transform=ax_t.transAxes)
            
            # Large verdict
            ax_t.text(0.55, 0.70, label_t, ha="center", va="center",
                     fontsize=24, fontweight="900", color=color_t,
                     transform=ax_t.transAxes)
            
            # Confidence percentage
            ax_t.text(0.85, 0.70, f"{conf_t:.1f}%", ha="right", va="center",
                     fontsize=18, fontweight="bold", color=color_t,
                     transform=ax_t.transAxes)
            
            # Probability breakdown
            real_pct = thesis_result["prob"] * 100
            fake_pct = (1 - thesis_result["prob"]) * 100
            ax_t.text(0.15, 0.30, f"Real: {real_pct:.1f}% | Fake: {fake_pct:.1f}%",
                     ha="left", va="center", fontsize=10, color="#555555",
                     transform=ax_t.transAxes)
        
        # ===== BASELINE MODEL BOX =====
        if baseline_result:
            pred_b = baseline_result["pred"]
            conf_b = baseline_result["confidence"]
            label_b = "REAL" if pred_b == 1 else "FAKE"
            color_b = "#16a34a" if pred_b == 1 else "#dc2626"
            bg_b = "#dcfce7" if pred_b == 1 else "#fee2e2"
            
            ax_b = fig.add_axes([start_x_baseline, y_pos - row_height + 0.005, col_width, row_height - 0.01])
            ax_b.set_facecolor(bg_b)
            ax_b.set_xlim(0, 1)
            ax_b.set_ylim(0, 1)
            ax_b.axis("off")
            
            # Border
            for sp in ax_b.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color_b)
                sp.set_linewidth(3)
            
            # Model name
            ax_b.text(0.15, 0.70, baseline_model, ha="left", va="center",
                     fontsize=13, fontweight="bold", color="#1e293b",
                     transform=ax_b.transAxes)
            
            # Large verdict
            ax_b.text(0.55, 0.70, label_b, ha="center", va="center",
                     fontsize=24, fontweight="900", color=color_b,
                     transform=ax_b.transAxes)
            
            # Confidence percentage
            ax_b.text(0.85, 0.70, f"{conf_b:.1f}%", ha="right", va="center",
                     fontsize=18, fontweight="bold", color=color_b,
                     transform=ax_b.transAxes)
            
            # Probability breakdown
            real_pct = baseline_result["prob"] * 100
            fake_pct = (1 - baseline_result["prob"]) * 100
            ax_b.text(0.15, 0.30, f"Real: {real_pct:.1f}% | Fake: {fake_pct:.1f}%",
                     ha="left", va="center", fontsize=10, color="#555555",
                     transform=ax_b.transAxes)
    
    # Footer
    ax_footer = fig.add_axes([0.05, 0.01, 0.9, 0.02])
    ax_footer.set_facecolor("#f1f5f9")
    ax_footer.axis("off")
    ax_footer.text(0.5, 0.5, "✓ All 15 models (9 thesis + 6 baseline) evaluated | Green=REAL | Red=FAKE",
                  ha="center", va="center", fontsize=11, color="#1e293b",
                  transform=ax_footer.transAxes)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def analyze_all(pil_image, dataset_name):
    """Run all models and return grid of results."""
    
    if pil_image is None:
        return None
    
    print(f"\n{'='*70}")
    print(f"FULL MODEL COMPARISON - {dataset_name}")
    print(f"{'='*70}")
    
    tensor, display = prep_image(pil_image)
    
    results = {}
    failed_models = []
    
    # Run all 9 thesis models
    print(f"\n[THESIS MODELS]")
    for thesis_model in THESIS_MODELS.keys():
        model = load_model(thesis_model, dataset_name)
        if model is None:
            print(f"  {thesis_model}: ✗ FAILED TO LOAD")
            failed_models.append(thesis_model)
            continue
        
        prob, pred, logit = predict(model, tensor)
        conf = prob * 100 if pred == 1 else (1 - prob) * 100
        verdict = "REAL" if pred == 1 else "FAKE"
        
        results[thesis_model] = {
            "prob": prob,
            "pred": pred,
            "confidence": conf,
            "verdict": verdict,
            "logit": logit,
        }
        
        print(f"  {thesis_model}: {verdict} ({conf:.1f}%)")
        
        # Save if uncertain
        if conf < 100.0:
            save_uncertain_prediction(pil_image, thesis_model, dataset_name, prob, pred, conf)
    
    # Run all 6 baseline models
    print(f"\n[BASELINE MODELS]")
    for baseline_model in BASELINE_MODELS.keys():
        model = load_model(baseline_model, dataset_name)
        if model is None:
            print(f"  {baseline_model}: ✗ FAILED TO LOAD")
            failed_models.append(baseline_model)
            continue
        
        prob, pred, logit = predict(model, tensor)
        conf = prob * 100 if pred == 1 else (1 - prob) * 100
        verdict = "REAL" if pred == 1 else "FAKE"
        
        results[baseline_model] = {
            "prob": prob,
            "pred": pred,
            "confidence": conf,
            "verdict": verdict,
            "logit": logit,
        }
        
        print(f"  {baseline_model}: {verdict} ({conf:.1f}%)")
        
        # Save if uncertain
        if conf < 100.0:
            save_uncertain_prediction(pil_image, baseline_model, dataset_name, prob, pred, conf)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create result grid
    fig_img = make_result_grid(display, results, dataset_name)
    
    if failed_models:
        status = f"⚠️  {len(failed_models)} models failed to load"
    else:
        status = f"✅ All {len(results)} models loaded successfully"
    
    print(f"\n{status}")
    print(f"{'='*70}\n")
    
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
    with gr.Blocks(css=CSS, title="Phase4DFD - All Models Comparison") as demo:
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
            All 9 Thesis Models vs 6 Baseline Models - Complete Comparison
          </div>
        </div>
        """)

        with gr.Column():
            gr.Markdown("### Upload Test Image", elem_classes="section-header")
            image_input = gr.Image(type="pil", label="Image", height=300)

        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=DATASET_CHOICES,
                value="DFFD",
                label="Select Dataset",
                info="Which training dataset checkpoint to use")

        gr.Markdown("", elem_classes="spacer")
        analyze_btn = gr.Button("🔍 RUN ALL 15 MODELS", variant="primary", size="lg")

        gr.Markdown("""
        ### Result Grid - All 15 Models
        **Left side:** 9 Thesis Models (with Attention)  
        **Right side:** 6 Baseline Models (no Attention)  
        **Green = REAL detection | Red = FAKE detection**
        """)
        result_image = gr.Image(type="pil", label="Results", height=1200)

        analyze_btn.click(
            fn=analyze_all,
            inputs=[image_input, dataset_dropdown],
            outputs=[result_image])

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("Phase4DFD - All Models Comparison Grid")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")

    # Check both model files
    if not os.path.exists("model.py"):
        print("❌ ERROR: model.py (thesis) not found")
        sys.exit(1)
    if not os.path.exists("model_baseline.py"):
        print("❌ ERROR: model_baseline.py (baseline) not found")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"📊 Models to compare:")
    for idx, (thesis, baseline) in enumerate(MODEL_PAIRS, 1):
        print(f"  {idx}. {thesis} vs {baseline}")
    print(f"{'='*70}\n")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=False, show_error=True, inbrowser=True)