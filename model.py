import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import timm
import lightning as L

from BNext.src.bnext import BNext

from torchmetrics.functional import accuracy, auroc

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except Exception:
    FlopCountAnalysis = None
    parameter_count = None


# =============================================================================
# ATTENTION MODULE 1: Phase-Aware Attention (INPUT-LEVEL, before backbone)
# =============================================================================

class PhaseAwareAttention(nn.Module):
    """
    Input-level phase-aware attention module.

    Operates on the augmented input tensor (B, C, H, W) where C = 3 + N
    (RGB + optional FFT magnitude + optional LBP).

    Two parallel pathways:
        - Magnitude path : reads pre-computed FFT magnitude from channel index 3
        - Phase path     : recomputes FFT phase internally from original_rgb

    Both pathways output (B, hidden_dim, H, W) feature maps which are
    concatenated and passed through attention_gen to produce a soft
    per-channel spatial attention map A₀ ∈ (0,1)^(B,C,H,W).

    The attended output is: x̃ = x ⊙ A₀  (element-wise multiply)

    Args:
        channels  : number of input channels C (e.g. 4 for RGB+FFT, 5 for RGB+FFT+LBP)
        reduction : channel reduction ratio for hidden_dim (default 8)

    NOTE: Requires add_fft_magnitude=True — magnitude must be at channel index 3.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # hidden_dim: clamped to minimum 16 to avoid degenerate tiny networks
        # e.g. channels=4: max(4//8, 16) = max(0, 16) = 16
        hidden_dim = max(channels // reduction, 16)
        self.channels = channels

        # --- Magnitude pathway ---
        # Input: pre-computed log FFT magnitude (B, 1, H, W)
        # Output: spatial frequency energy features (B, hidden_dim, H, W)
        self.magnitude_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Phase pathway ---
        # Input: normalized FFT phase recomputed from original_rgb (B, 1, H, W)
        # Output: structural phase coherence features (B, hidden_dim, H, W)
        self.phase_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Attention generator ---
        # Input: concatenated [mag_feat, phase_feat] (B, 2*hidden_dim, H, W)
        # Output: per-channel spatial attention map (B, channels, H, W) ∈ (0,1)
        self.attention_gen = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, original_rgb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x            : augmented input (B, C, H, W), C = channels
            original_rgb : original RGB image (B, 3, H, W), used to recompute phase

        Returns:
            x̃ : attended input (B, C, H, W), same shape as x
        """
        # Safe fallback: if original_rgb not provided, skip attention
        if original_rgb is None:
            return x

        # ------------------------------------------------------------------
        # Magnitude path: read from pre-computed channel index 3
        # This was computed in add_new_channels() as log|FFTShift(F(gray))|
        # ------------------------------------------------------------------
        magnitude = x[:, 3:4, :, :]  # (B, 1, H, W)

        # ------------------------------------------------------------------
        # Phase path: recompute fresh from original_rgb
        # Phase is never stored as a channel — it must remain in angular form
        # [-π, π] to preserve its structural meaning.
        # ------------------------------------------------------------------
        # Step 1: RGB → grayscale (ITU-R BT.601 coefficients)
        gray = (
            0.299 * original_rgb[:, 0]
            + 0.587 * original_rgb[:, 1]
            + 0.114 * original_rgb[:, 2]
        ).unsqueeze(1)  # (B, 1, H, W)

        # Step 2: 2D FFT → shift zero-frequency to center → extract phase angle
        fft = torch.fft.fft2(gray, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft)
        phase = torch.angle(fft_shifted)  # (B, 1, H, W), range [-π, π]

        # Step 3: Normalize phase to [0, 1] for numerical stability in conv layers
        phase_norm = (phase + np.pi) / (2 * np.pi)  # (B, 1, H, W)

        # ------------------------------------------------------------------
        # Process through separate learned pathways
        # ------------------------------------------------------------------
        mag_feat = self.magnitude_conv(magnitude)    # (B, hidden_dim, H, W)
        phase_feat = self.phase_conv(phase_norm)     # (B, hidden_dim, H, W)

        # ------------------------------------------------------------------
        # Fuse and generate attention map
        # ------------------------------------------------------------------
        combined = torch.cat([mag_feat, phase_feat], dim=1)  # (B, 2*hidden_dim, H, W)
        attention = self.attention_gen(combined)              # (B, C, H, W) ∈ (0,1)

        # ------------------------------------------------------------------
        # Apply attention: element-wise soft gating
        # ------------------------------------------------------------------
        return x * attention  # (B, C, H, W)


# =============================================================================
# ATTENTION MODULE 2: CBAM-style Feature Attention (FEATURE-LEVEL, after backbone)
# =============================================================================

class FeatureCBAMAttention(nn.Module):
    """
    Feature-level CBAM-style attention module.

    Operates on backbone output feature maps (B, 2048, H, W).
    Applies two sequential attention mechanisms:
        1. Channel attention (SE-Net style): recalibrates which feature
           channels are most informative → (B, 2048, 1, 1)
        2. Spatial attention: identifies which spatial locations in the
           7×7 feature grid are most discriminative → (B, 1, 7, 7)

    NOTE: Phase computation is NOT performed here — FFT phase has no
    meaningful interpretation on abstract 2048-dim backbone feature maps.

    Args:
        channels  : number of feature channels (e.g. 2048 for BNext-M)
        reduction : channel reduction ratio for hidden_dim (default 8)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden_dim = max(channels // reduction, 16)

        # --- Channel attention (SE-Net style) ---
        # Global average pool → compress → expand → sigmoid
        # Output: (B, channels, 1, 1) — one weight per channel
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # (B, C, 1, 1)
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # --- Spatial attention ---
        # Pointwise conv to compress channels → spatial importance map
        # Output: (B, 1, H, W) — one weight per spatial location
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, original_rgb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x            : backbone feature map (B, C, H, W)
            original_rgb : ignored — not needed at feature level (accepted for
                           API compatibility with PhaseAwareAttention)

        Returns:
            attended features (B, C, H, W), same shape as x
        """
        # Apply channel attention first (CBAM sequential order)
        x = x * self.channel_attn(x)  # (B, C, 1, 1) broadcasts over H, W

        # Apply spatial attention on channel-attended features
        x = x * self.spatial_attn(x)  # (B, 1, H, W) broadcasts over C

        return x


# =============================================================================
# MAIN MODEL: BNext4DFR
# =============================================================================

class BNext4DFR(L.LightningModule):
    """
    BNext for Deepfake Recognition (BNext4DFR).

    Extends BNext binary neural network backbone with:
        - Optional FFT magnitude channel augmentation
        - Optional LBP texture channel augmentation
        - Optional Phase-Aware Attention (input-level, before backbone)
        - Optional CBAM Feature Attention (feature-level, after backbone)
        - Channel adapter (input_channels → 3) for backbone compatibility
        - Deep MLP classification head with dropout + batch normalization

    Args:
        num_classes           : number of output classes (currently only 2 supported)
        backbone              : BNext variant — "BNext-T" | "BNext-S" | "BNext-M" | "BNext-L"
        freeze_backbone       : freeze backbone weights initially (unfrozen by callback at epoch 5)
        add_fft_magnitude     : add log FFT magnitude as extra input channel
        add_lbp_channel       : add differentiable LBP approximation as extra input channel
        use_frequency_attention: enable Phase-Aware and/or CBAM attention modules
        attention_position    : where to apply attention —
                                "before_backbone" (Phase-Aware only)
                                "after_backbone"  (CBAM only)
                                "both"            (Phase-Aware + CBAM)
        learning_rate         : base learning rate for optimizer
        pos_weight            : positive class weight for BCE loss (handles class imbalance)
        use_dropout           : enable dropout in classification head
        dropout_rate          : base dropout rate (head uses rate, rate/2, rate/3)

    CONSTRAINT: use_frequency_attention=True requires add_fft_magnitude=True,
    because PhaseAwareAttention reads the FFT magnitude from channel index 3.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "BNext-M",
        freeze_backbone: bool = True,

        # Feature augmentation
        add_fft_magnitude: bool = False,
        add_lbp_channel: bool = False,

        # Attention configuration
        use_frequency_attention: bool = False,
        attention_position: str = "both",  # "before_backbone" | "after_backbone" | "both"

        # Training hyperparameters
        learning_rate: float = 1e-3,
        pos_weight: float = None,

        # Regularization
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize epoch outputs buffer
        self.epoch_outs = []

        # ── Store config ──────────────────────────────────────────────────
        self.num_classes = int(num_classes)
        self.learning_rate = float(learning_rate)
        self.pos_weight = pos_weight
        self.use_dropout = bool(use_dropout)
        self.dropout_rate = float(dropout_rate)
        self.use_frequency_attention = bool(use_frequency_attention)
        self.attention_position = attention_position
        self.add_fft_magnitude = bool(add_fft_magnitude)
        self.add_lbp_channel = bool(add_lbp_channel)

        # ── Validate attention config ─────────────────────────────────────
        if self.use_frequency_attention and attention_position in ["before_backbone", "both"]:
            if not self.add_fft_magnitude:
                raise ValueError(
                    "PhaseAwareAttention (before_backbone) requires add_fft_magnitude=True.\n"
                    "The module reads FFT magnitude from channel index 3.\n"
                    "Either set add_fft_magnitude=True or use attention_position='after_backbone'."
                )

        # ── Backbone loading ──────────────────────────────────────────────
        size_map = {
            "BNext-T": "tiny",
            "BNext-S": "small",
            "BNext-M": "middle",
            "BNext-L": "large",
        }
        if backbone not in size_map:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. "
                f"Choose from {list(size_map.keys())}"
            )

        size = size_map[backbone]
        self.base_model = nn.ModuleDict({"module": BNext(num_classes=1000, size=size)})

        pretrained_state_dict = torch.load(
            f"pretrained/{size}_checkpoint.pth.tar",
            map_location="cpu",
            weights_only=False,
        )
        self.base_model.load_state_dict(pretrained_state_dict)
        self.base_model = self.base_model.module
        print(f"✓ Loaded pretrained {backbone} backbone")

        # ── Input channel configuration ───────────────────────────────────
        # Count extra channels beyond RGB
        self.new_channels = sum([
            self.add_fft_magnitude,   # channel index 3 (if enabled)
            self.add_lbp_channel,     # channel index 4 (if both enabled), or 3 (if FFT disabled)
        ])
        self.input_channels = 3 + self.new_channels

        print(f"✓ Input channels: {self.input_channels}")
        print(f"  - RGB            : 3 channels (indices 0,1,2)")
        if self.add_fft_magnitude:
            print(f"  - FFT magnitude  : 1 channel  (index 3)")
        if self.add_lbp_channel:
            idx = 3 + int(self.add_fft_magnitude)
            print(f"  - LBP texture    : 1 channel  (index {idx})")
        if self.use_frequency_attention:
            print(f"  - Phase          : recomputed internally in PhaseAwareAttention")

        # ── Channel adapter ───────────────────────────────────────────────
        # Projects augmented input (3+N channels) down to 3 channels
        # for compatibility with ImageNet-pretrained backbone.
        if self.new_channels > 0:
            self.adapter = nn.Conv2d(
                self.input_channels, 3, kernel_size=1, bias=True
            )
            # Initialize: identity for RGB channels, small random for extras
            with torch.no_grad():
                self.adapter.weight.zero_()
                self.adapter.weight[:3, :3, 0, 0] = torch.eye(3)
                self.adapter.weight[:, 3:, 0, 0] = (
                    torch.randn(3, self.new_channels) * 0.01
                )
                if self.adapter.bias is not None:
                    self.adapter.bias.zero_()
            print(f"✓ Adapter: {self.input_channels} → 3 channels")
        else:
            self.adapter = nn.Identity()
            print(f"✓ No adapter needed (RGB only)")

        # ── Backbone configuration ────────────────────────────────────────
        self.inplanes = self.base_model.fc.in_features
        self.base_model.deactive_last_layer = True
        self.base_model.fc = nn.Identity()
        print(f"✓ Backbone feature dimension: {self.inplanes}")

        # ── Backbone freezing ─────────────────────────────────────────────
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False
            print(f"✓ Backbone frozen (BackboneUnfreezingCallback will unfreeze at epoch 5)")
        else:
            print(f"✓ Backbone trainable from start")

        # ── Attention modules ─────────────────────────────────────────────
        if self.use_frequency_attention:
            print(f"\n✓ Attention Configuration:")
            print(f"  Position : {attention_position}")

            # Input-level: Phase-Aware Attention (before backbone)
            if attention_position in ["before_backbone", "both"]:
                self.input_attention = PhaseAwareAttention(
                    channels=self.input_channels
                )
                print(
                    f"  Before backbone : PhaseAwareAttention "
                    f"(channels={self.input_channels}, hidden_dim="
                    f"{max(self.input_channels // 8, 16)})"
                )
            else:
                self.input_attention = None
                print(f"  Before backbone : None")

            # Feature-level: CBAM Attention (after backbone)
            if attention_position in ["after_backbone", "both"]:
                self.feature_attention = FeatureCBAMAttention(
                    channels=self.inplanes
                )
                print(
                    f"  After backbone  : FeatureCBAMAttention "
                    f"(channels={self.inplanes}, hidden_dim="
                    f"{max(self.inplanes // 8, 16)})"
                )
            else:
                self.feature_attention = None
                print(f"  After backbone  : None")
        else:
            self.input_attention = None
            self.feature_attention = None
            print(f"\n✗ No attention modules")

        # ── Classification head ───────────────────────────────────────────
        head_out = 1 if self.num_classes == 2 else self.num_classes

        if self.use_dropout:
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),            # 0.30
                nn.Linear(self.inplanes, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate / 2),        # 0.15
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate / 3),        # 0.10
                nn.Linear(256, head_out),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(self.inplanes, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, head_out),
            )

        print(f"✓ Classification head: {self.inplanes} → 512 → 256 → {head_out}")

        # ── Parameter summary ─────────────────────────────────────────────
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n✓ Model ready:")
        print(f"  Total parameters     : {total_params:,}")
        print(f"  Trainable parameters : {trainable_params:,}")
        print(f"  Frozen parameters    : {total_params - trainable_params:,}\n")

    # =========================================================================
    # CHANNEL AUGMENTATION
    # =========================================================================

    def add_new_channels(self, images: torch.Tensor) -> torch.Tensor:
        """
        Augment RGB input with optional extra channels.

        Channels are appended in this fixed order:
            index 0,1,2 : R, G, B  (always present)
            index 3     : FFT magnitude (if add_fft_magnitude=True)
            index 3 or 4: LBP texture  (if add_lbp_channel=True)

        NOTE: Phase is NOT added as a channel. It is recomputed inside
        PhaseAwareAttention from original_rgb to preserve its angular structure.

        Args:
            images : (B, 3, H, W) RGB images, values in [0, 1]

        Returns:
            (B, 3+N, H, W) where N = self.new_channels
        """
        # Grayscale for frequency/texture computation
        gray = (
            0.299 * images[:, 0]
            + 0.587 * images[:, 1]
            + 0.114 * images[:, 2]
        )  # (B, H, W)

        new_channels = []

        # ── FFT Magnitude Spectrum ──────────────────────────────────────
        if self.add_fft_magnitude:
            fft = torch.fft.fft2(gray, dim=(-2, -1))
            fft_shifted = torch.fft.fftshift(fft)
            magnitude = torch.abs(fft_shifted)
            # Log scale to compress dynamic range
            magnitude_log = torch.log(magnitude + 1e-8)  # (B, H, W)
            new_channels.append(magnitude_log)

        # ── Differentiable LBP Approximation ───────────────────────────
        if self.add_lbp_channel:
            gray_4d = gray.unsqueeze(1)  # (B, 1, H, W)
            # Compare each pixel to its local neighborhood average (3×3)
            neighbors = F.avg_pool2d(gray_4d, kernel_size=3, stride=1, padding=1)
            lbp = (neighbors > gray_4d).float().squeeze(1)  # (B, H, W)
            new_channels.append(lbp)

        if new_channels:
            extra = torch.stack(new_channels, dim=1)        # (B, N, H, W)
            return torch.cat([images, extra], dim=1)        # (B, 3+N, H, W)

        return images

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            x : (B, 3, H, W) RGB input images, values in [0, 1]

        Returns:
            dict with key "logits" : (B, 1) raw logits (before sigmoid)
        """
        # ── Save original RGB before any augmentation ────────────────────
        # Needed by PhaseAwareAttention to recompute FFT phase internally
        original_rgb = x.clone()  # (B, 3, H, W)

        # ── Step 1: Augment channels ─────────────────────────────────────
        if self.new_channels > 0:
            x = self.add_new_channels(x)  # (B, 3+N, H, W)

        # ── Step 2: Input-level Phase-Aware Attention (before adapter) ───
        if self.input_attention is not None:
            x = self.input_attention(x, original_rgb=original_rgb)  # (B, 3+N, H, W)

        # ── Step 3: Channel adapter (3+N → 3) ───────────────────────────
        x = self.adapter(x)  # (B, 3, H, W)

        # ── Step 4: ImageNet normalization ───────────────────────────────
        mean = torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_MEAN,
            device=x.device,
        ).view(1, 3, 1, 1)
        std = torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_STD,
            device=x.device,
        ).view(1, 3, 1, 1)
        x = (x - mean) / std  # (B, 3, H, W)

        # ── Step 5: Backbone ─────────────────────────────────────────────
        features = self.base_model(x)  # (B, 2048, 7, 7) or (B, 2048)

        # ── Step 6: Feature-level CBAM Attention (after backbone) ────────
        # Only applied if features are still spatial (4D)
        if self.feature_attention is not None:
            if features.dim() == 4:
                features = self.feature_attention(features)  # (B, 2048, 7, 7)
            # If features are already pooled (2D), skip — nothing to attend over

        # ── Step 7: Global Average Pooling ───────────────────────────────
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 2048)
        elif features.dim() == 3:
            features = features.mean(dim=1)                           # (B, 2048)
        # features.dim() == 2: already pooled, no action needed

        # ── Step 8: Classification head ───────────────────────────────────
        logits = self.head(features)  # (B, 1)

        return {"logits": logits}

    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================

    def configure_optimizers(self):
        """
        AdamW optimizer with component-wise learning rates:
            - adapter      : learning_rate       (full LR, small module)
            - attention    : learning_rate       (full LR, new modules)
            - head         : learning_rate       (full LR, new module)
            - backbone     : learning_rate × 0.1 (lower LR, pretrained)

        Backbone parameters are only added when freeze_backbone=False.
        BackboneUnfreezingCallback adds backbone params at epoch 5.

        Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-5)
        """
        param_groups = []

        # Adapter
        if not isinstance(self.adapter, nn.Identity):
            param_groups.append({
                "params": list(self.adapter.parameters()),
                "lr": self.learning_rate,
                "name": "adapter",
            })

        # Attention modules (both input and feature level)
        attn_params = []
        if self.input_attention is not None:
            attn_params += list(self.input_attention.parameters())
        if self.feature_attention is not None:
            attn_params += list(self.feature_attention.parameters())
        if attn_params:
            param_groups.append({
                "params": attn_params,
                "lr": self.learning_rate,
                "name": "attention",
            })

        # Classification head
        param_groups.append({
            "params": list(self.head.parameters()),
            "lr": self.learning_rate,
            "name": "head",
        })

        # Backbone — only if not frozen
        if not self.freeze_backbone:
            param_groups.append({
                "params": list(self.base_model.parameters()),
                "lr": self.learning_rate * 0.1,
                "name": "backbone",
            })

        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # =========================================================================
    # LIGHTNING HOOKS
    # =========================================================================

    def on_train_start(self):
        """Log FLOPs and parameter count at start of training."""
        if FlopCountAnalysis is not None and parameter_count is not None:
            with torch.no_grad():
                try:
                    dummy = torch.randn(1, 3, 224, 224, device=self.device)
                    flops = FlopCountAnalysis(self, dummy)
                    params = parameter_count(self)[""]
                    self.log_dict(
                        {"flops": flops.total(), "parameters": params},
                        prog_bar=True,
                        logger=True,
                    )
                except Exception:
                    pass

    def on_train_end(self):
        if FlopCountAnalysis is None or parameter_count is None:
            print("\n[FLOPs] fvcore not available — skipping FLOPs report.")
            return

        print("\n" + "=" * 65)
        print("Model Complexity Report (post-training)")
        print("=" * 65)
        try:
            self.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224, device=self.device)
                flops = FlopCountAnalysis(self, dummy)
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                params = parameter_count(self)

                total_flops = flops.total()
                total_params = params[""]

                print(f"  Input resolution   : 224 × 224 × 3")
                print(f"  Total FLOPs        : {total_flops:,}  ({total_flops / 1e9:.3f} GFLOPs)")
                print(f"  Total parameters   : {total_params:,}  ({total_params / 1e6:.2f} M)")

                # Per-module breakdown
                by_module = flops.by_module()
                top_modules = [
                    ("adapter",           "adapter"),
                    ("input_attention",   "input_attention"),
                    ("base_model",        "base_model"),
                    ("feature_attention", "feature_attention"),
                    ("head",              "head"),
                ]
                print("\n  Per-module FLOPs:")
                for label, key in top_modules:
                    val = by_module.get(key, 0)
                    if val > 0:
                        pct = 100.0 * val / (total_flops + 1e-9)
                        print(f"    {label:<20s}: {val:>15,}  ({pct:.1f}%)")

        except Exception as e:
            print(f"  [FLOPs] Calculation failed: {e}")
        print("=" * 65 + "\n")

    def _reset_epoch_state(self):
        self._clear_memory()
        self.epoch_outs = []

    def on_train_epoch_start(self):
        self._reset_epoch_state()

    def on_validation_epoch_start(self):
        self._reset_epoch_state()

    def on_test_epoch_start(self):
        self._reset_epoch_state()

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="test")

    def _step(self, batch, batch_idx, phase: str):
        """
        Shared step for train / val / test.

        Loss:
            Train : 0.7 × BCE + 0.3 × Focal  (focal helps with hard examples)
            Val   : BCE only
            Test  : BCE only
        """
        images = batch["image"].to(self.device)
        labels = batch["is_real"]

        # Ensure labels are 1D float
        if labels.dim() > 1:
            labels = labels[:, 0]
        labels = labels.float().to(self.device)

        # Forward pass
        outs = {"phase": phase, "labels": labels}
        outs.update(self(images))

        # ── Loss computation ──────────────────────────────────────────────
        if self.num_classes == 2:
            logits = outs["logits"][:, 0]  # (B,)

            bce_loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=labels,
                pos_weight=(
                    torch.as_tensor(self.pos_weight, device=self.device)
                    if self.pos_weight is not None
                    else None
                ),
            )

            if phase == "train":
                # Focal loss: down-weights easy examples, focuses on hard ones
                probs = torch.sigmoid(logits)
                pt = labels * probs + (1 - labels) * (1 - probs)
                focal_weight = (1 - pt) ** 2
                focal_loss = (
                    focal_weight
                    * F.binary_cross_entropy_with_logits(
                        logits, labels, reduction="none"
                    )
                ).mean()
                loss = 0.7 * bce_loss + 0.3 * focal_loss
            else:
                loss = bce_loss
        else:
            raise NotImplementedError("Only binary classification (num_classes=2) is supported.")

        # ── Detach tensors for logging ────────────────────────────────────
        for k in outs:
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()

        # ── Per-step logging ──────────────────────────────────────────────
        log_dict = {f"{phase}_loss": loss.detach().cpu()}
        if phase in {"train", "val"}:
            current_lr = (
                self.trainer.optimizers[0].param_groups[0]["lr"]
                if self.trainer is not None
                else self.learning_rate
            )
            log_dict[f"{phase}_learning_rate"] = current_lr

        self.log_dict(log_dict, prog_bar=True, logger=True)
        self.epoch_outs.append(outs)

        return loss

    def _on_epoch_end(self):
        """
        Compute and log epoch-level metrics:
            accuracy, AUC, precision, recall, F1
        """
        self._clear_memory()

        with torch.no_grad():
            if not self.epoch_outs:
                return

            labels = torch.cat(
                [b["labels"] for b in self.epoch_outs], dim=0
            )
            if labels.dim() > 1:
                labels = labels.squeeze()

            logits = torch.cat(
                [b["logits"] for b in self.epoch_outs], dim=0
            )[:, 0]

            phases = [
                b["phase"]
                for b in self.epoch_outs
                for _ in range(len(b["labels"]))
            ]

            for phase in ["train", "val", "test"]:
                idxs = [i for i, p in enumerate(phases) if p == phase]
                if not idxs:
                    continue

                ph_logits = logits[idxs]
                ph_labels = labels[idxs].long()

                # Accuracy & AUC
                metrics = {
                    "acc": accuracy(
                        preds=ph_logits, target=ph_labels, task="binary"
                    ),
                    "auc": auroc(
                        preds=ph_logits, target=ph_labels, task="binary"
                    ),
                }

                # Precision, Recall, F1
                preds_bin = (torch.sigmoid(ph_logits) > 0.5).long()
                tp = ((preds_bin == 1) & (ph_labels == 1)).sum().item()
                fp = ((preds_bin == 1) & (ph_labels == 0)).sum().item()
                fn = ((preds_bin == 0) & (ph_labels == 1)).sum().item()

                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)

                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

                self.log_dict(
                    {
                        f"{phase}_{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (torch.Tensor, int, float))
                    },
                    prog_bar=True,
                    logger=True,
                )

    def on_train_epoch_end(self):
        self._on_epoch_end()

    def on_validation_epoch_end(self):
        self._on_epoch_end()

    def on_test_epoch_end(self):
        self._on_epoch_end()

    def _clear_memory(self):
        """Release GPU cache between epochs."""
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("BNext4DFR — Smoke Test")
    print("=" * 65)

    configs = [
        # (description,               fft,   lbp,   attn,  position)
        ("Baseline: RGB only",         False, False, False, "before_backbone"),
        ("RGB + FFT + Phase-Aware",    True,  False, True,  "before_backbone"),
        ("RGB + FFT + LBP + CBAM",     True,  True,  True,  "after_backbone"),
        ("RGB + FFT + LBP + Both",     True,  True,  True,  "both"),
        ("RGB + FFT + Both Attention", True,  False, True,  "both"),
    ]

    all_passed = True

    for desc, fft, lbp, attn, pos in configs:
        print(f"\n{'─'*65}")
        print(f"Config : {desc}")
        print(f"{'─'*65}")
        try:
            model = BNext4DFR(
                num_classes=2,
                backbone="BNext-M",
                add_fft_magnitude=fft,
                add_lbp_channel=lbp,
                use_frequency_attention=attn,
                attention_position=pos,
            )
            model.eval()

            with torch.no_grad():
                x = torch.randn(2, 3, 224, 224)
                out = model(x)

            logits = out["logits"]
            assert logits.shape == (2, 1), f"Unexpected output shape: {logits.shape}"

            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"  ✓ Output shape      : {logits.shape}")
            print(f"  ✓ Total params      : {total:,}")
            print(f"  ✓ Trainable params  : {trainable:,}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            all_passed = False

    print(f"\n{'=' * 65}")
    if all_passed:
        print("All smoke tests passed ✓")
    else:
        print("Some tests FAILED ✗ — check errors above")
    print("=" * 65)