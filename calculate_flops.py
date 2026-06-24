# calculate_flops.py
import torch
from model import BNext4DFR

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    FVCORE = True
except ImportError:
    print("Install: pip install fvcore")
    FVCORE = False

models = {
    "BNext-T baseline":    dict(backbone="BNext-T", add_fft_magnitude=False, add_lbp_channel=False, use_frequency_attention=False, attention_position="before_backbone"),
    "BNext-S baseline":    dict(backbone="BNext-S", add_fft_magnitude=False, add_lbp_channel=False, use_frequency_attention=False, attention_position="before_backbone"),
    "BNext-M baseline":    dict(backbone="BNext-M", add_fft_magnitude=False, add_lbp_channel=False, use_frequency_attention=False, attention_position="before_backbone"),
    "PhaseDFDＴ (ours)":  dict(backbone="BNext-T", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="before_backbone"),
    "DualDFDＴ (ours)":  dict(backbone="BNext-T", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="after_backbone"),
    "FullDFDＴ (ours)":dict(backbone="BNext-T", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="both"),
    "PhaseDFDＳ (ours)":  dict(backbone="BNext-S", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="before_backbone"),
    "DualDFDＳ (ours)":  dict(backbone="BNext-S", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="after_backbone"),
    "FullDFDＳ (ours)":dict(backbone="BNext-S", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="both"),
    "PhaseDFDＭ (ours)":  dict(backbone="BNext-M", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="before_backbone"),
    "DualDFDＭ (ours)":  dict(backbone="BNext-M", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="after_backbone"),
    "FullDFDＭ (ours)":dict(backbone="BNext-M", add_fft_magnitude=True,  add_lbp_channel=True,  use_frequency_attention=True,  attention_position="both"),
}

dummy = torch.randn(1, 3, 224, 224)
print(f"\n{'Model':<25} {'Params (M)':>12} {'FLOPs (G)':>12} {'Memory (MB)':>12}")
print("-" * 65)

for name, cfg in models.items():
    model = BNext4DFR(num_classes=2, **cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    # Binary memory = params * 1 bit / 8 bits per byte / 1MB
    binary_memory = sum(p.numel() for p in model.parameters()) / 8 / 1e6

    if FVCORE:
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flop_g = flops.total() / 1e9
    else:
        flop_g = 0.0

    print(f"{name:<25} {total_params:>11.1f}M {flop_g:>11.3f}G {binary_memory:>10.1f}MB")