from pprint import pprint
import argparse
import os
os.environ["WANDB_MODE"] = "offline"
from datetime import datetime
import random
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import wandb

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback

from cifake_dataset import CIFAKEDataset
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
from face_dataset import FaceDataset
from face140k_dataset import Face140kDataset
from forensics_dataset import ForensicsDataset

from model import BNext4DFR
from lib.util import load_config



# ============================================================================
# CUSTOM CALLBACK: BACKBONE UNFREEZING
# ============================================================================

class BackboneUnfreezingCallback(Callback):
    """
    Automatically unfreeze backbone after specified epochs.
    Also adjusts learning rate for backbone parameters.
    """
    def __init__(self, unfreeze_at_epoch=5):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfrozen = False
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        current_epoch = trainer.current_epoch
        
        if current_epoch == self.unfreeze_at_epoch and not self.unfrozen:
            print("\n" + "="*70)
            print(f"🔓 UNFREEZING BACKBONE at Epoch {current_epoch}")
            print("="*70)
            
            # Unfreeze all backbone parameters
            for param in pl_module.base_model.parameters():
                param.requires_grad = True
            
            # Update module flag
            pl_module.freeze_backbone = False
            
            # Add backbone parameters to optimizer with lower learning rate
            optimizer = trainer.optimizers[0]
            backbone_params = [p for p in pl_module.base_model.parameters() if p.requires_grad]
            
            optimizer.add_param_group({
                'params': backbone_params,
                'lr': pl_module.learning_rate * 0.1,  # 10x lower LR for backbone
                'name': 'backbone'
            })
            
            self.unfrozen = True
            
            # Print statistics
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Backbone LR: {pl_module.learning_rate * 0.1:.6f}")
            print("="*70 + "\n")


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def args_func():
    parser = argparse.ArgumentParser(
        description="Train BNext deepfake detector with phase-aware attention"
    )
    
    parser.add_argument(
        "--cfg",
        type=str,
        default="./configs/cifake_baseline.cfg",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=[
            "baseline",                # RGB only
            "fft_lbp",                 # RGB + FFT + LBP (paper)
            "attention_before",        # + Attention before backbone
            "attention_after",         # + Attention after backbone
            "attention_both",          # + Attention both positions
        ],
        help="Experiment name"
    )
    
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=5,
        help="Number of epochs to keep backbone frozen"
    )
    
    parser.add_argument(
        "--use_two_stage",
        action="store_true",
        default=True,
        help="Use two-stage training (frozen then unfrozen)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    return args


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # Load config
    cfg = load_config(args.cfg)

    print("\n" + "="*70)
    print(f"EXPERIMENT: {args.experiment}")
    print("="*70 + "\n")

    pprint(cfg)

    # Preliminary setup
    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # ========================================
    # DATASET LOADING
    # ========================================
    
    print(f"\n{'='*70}")
    print(f"Loading dataset: {cfg['dataset']['name']}")
    print(f"{'='*70}\n")
    
    if cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        train_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "coco_fake":
        print(
            f"Loading COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
        )
        train_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="train",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Loading DFFD dataset from {cfg['dataset']['dffd_path']}")
        train_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "artifact":
        print(f"Loading Artifact dataset from {cfg['dataset']['artifact_path']}")
        train_dataset = ArtifactDataset(
            dataset_path=cfg["dataset"]["artifact_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = ArtifactDataset(
            dataset_path=cfg["dataset"]["artifact_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "face":
        print(f"Loading Face dataset from {cfg['dataset']['face_path']}")
        train_dataset = FaceDataset(
            dataset_path=cfg["dataset"]["face_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = FaceDataset(
            dataset_path=cfg["dataset"]["face_path"],
            split="test",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "face140k":
        print(f"Loading Face_140k dataset from {cfg['dataset']['face140k_path']}")
        train_dataset = Face140kDataset(
            dataset_path=cfg["dataset"]["face140k_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = Face140kDataset(
            dataset_path=cfg["dataset"]["face140k_path"],
            split="test",
            resolution=cfg["train"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "forensics":
        print(f"Loading Forensics dataset from {cfg['dataset']['forensics_path']}")
        train_dataset = ForensicsDataset(
            dataset_path=cfg["dataset"]["forensics_path"],
            split="train",
            resolution=cfg["train"]["resolution"],
        )
        val_dataset = ForensicsDataset(
            dataset_path=cfg["dataset"]["forensics_path"],
            split="val",
            resolution=cfg["train"]["resolution"],
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']['name']}")


    # ========================================
    # DATA LOADERS
    # ========================================

    num_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print("\n" + "="*70)
    print("LABEL VERIFICATION TEST")
    print("="*70)

    # Check first 10 samples
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        label = sample["is_real"]
        image_path = sample.get("image_path", "unknown")
        print(f"  Sample {i}: is_real={label}, path={image_path}")

    # Check statistics
    num_real = sum(1 for x in train_dataset.items if x["is_real"] is True)
    num_fake = sum(1 for x in train_dataset.items if x["is_real"] is False)
    pos_weight = num_fake / (num_real + 1e-9)

    print(f"\nClass distribution:")
    print(f"  Real (is_real=True):  {num_real}")
    print(f"  Fake (is_real=False): {num_fake}")
    print(f"  pos_weight: {pos_weight:.4f}")
    print(f"\nIf pos_weight > 1: Model biased toward predicting REAL ⚠️")
    print(f"If pos_weight < 1: Model biased toward predicting FAKE ⚠️")
    print("="*70 + "\n")

    # ========================================
    # CLASS WEIGHTS CALCULATION
    # ========================================

    num_real = sum(1 for x in train_dataset.items if x["is_real"] is True)
    num_fake = sum(1 for x in train_dataset.items if x["is_real"] is False)

    pos_weight = num_fake / (num_real + 1e-9)  # weight for positive class (REAL=1)

    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Real (label=1): {num_real}")  
    print(f"  Fake (label=0): {num_fake}")
    print(f"  pos_weight (for REAL): {pos_weight:.4f}")
    print(f"{'='*60}\n")

    print("Label sanity check:")
    print("  is_real=1 means REAL")
    print("  is_real=0 means FAKE")
    print("  Example labels:", [float(train_dataset[i]['is_real']) for i in range(10)])


    # ========================================
    # MODEL INITIALIZATION
    # ========================================
     # Disable two-stage if resuming
    if args.resume_from_checkpoint is not None:
        print("🔁 Resuming from checkpoint — disabling backbone freezing")
        cfg["model"]["freeze_backbone"] = False
        args.use_two_stage = False

    net = BNext4DFR(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["dataset"]["labels"],
        freeze_backbone=cfg["model"]["freeze_backbone"],

        # Feature augmentation
        add_fft_magnitude=cfg["model"].get("add_fft_magnitude", False),
        add_lbp_channel=cfg["model"].get("add_lbp_channel", False),

        # Attention configuration
        use_frequency_attention=cfg["model"].get("use_frequency_attention", False),
        attention_position=cfg["model"].get("attention_position", "both"),
        
        # Training parameters
        learning_rate=cfg["train"].get("learning_rate", 1e-3),
        pos_weight=pos_weight,
        
        # Regularization
        use_dropout=cfg["model"].get("use_dropout", True),
        dropout_rate=cfg["model"].get("dropout_rate", 0.3)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # ========================================
    # CALLBACKS
    # ========================================
    
    callbacks = [
       ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,          # ← keeps only the single best
            mode="max",
            filename=cfg["dataset"]["name"] + "_" + cfg["model"]["backbone"] + f"_{args.experiment}" + "_{epoch:02d}-{val_acc:.4f}",
            save_last=False,       # ← no separate last.ckpt
        ), 
        EarlyStopping(
            monitor="val_acc",
            patience=15,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Add backbone unfreezing callback if two-stage training
    if args.use_two_stage and args.freeze_backbone_epochs > 0:
        unfreeze_callback = BackboneUnfreezingCallback(
            unfreeze_at_epoch=args.freeze_backbone_epochs
        )
        callbacks.append(unfreeze_callback)
        print(f"✓ Two-stage training enabled (unfreeze at epoch {args.freeze_backbone_epochs})\n")

    # ========================================
    # WANDB LOGGING
    # ========================================
    
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "DFAD_BNext_PhaseAware"
    
    training_mode = "two_stage" if args.use_two_stage else "single_stage"
    run = f"{cfg['dataset']['name']}_{cfg['model']['backbone']}_{args.experiment}_{training_mode}_{date}"
    
    # Remove illegal characters
    run = re.sub(r'[:;,#?/\'"\\]', '_', run)

    # Check WandB connectivity
    try:
        wandb.ensure_configured()
        if wandb.run is None:
            import requests
            try:
                requests.get("https://api.wandb.ai", timeout=3)
                offline = False
            except Exception:
                offline = True
        else:
            offline = False
    except Exception:
        offline = True

    if offline:
        os.environ["WANDB_MODE"] = "offline"
        print("WandB set to OFFLINE mode.")

    logger = WandbLogger(project=project, name=run, id=run, log_model=False) if not offline else None

    # ========================================
    # TRAINER SETUP
    # ========================================
    
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg["train"]["mixed_precision"] else 32,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg["train"]["accumulation_batches"],
        limit_train_batches=cfg["train"]["limit_train_batches"],
        limit_val_batches=cfg["train"]["limit_val_batches"],
        max_epochs=cfg["train"]["epoch_num"],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    # ========================================
    # TRAINING
    # ========================================
    
    print(f"\n{'='*60}")
    print(f"Starting Training:")
    print(f"  Project: {project}")
    print(f"  Run name: {run}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Training mode: {training_mode}")
    if args.use_two_stage:
        print(f"  Stage 1: Epochs 0-{args.freeze_backbone_epochs-1} (frozen)")
        print(f"  Stage 2: Epochs {args.freeze_backbone_epochs}-{cfg['train']['epoch_num']} (unfrozen)")
    print(f"  Max epochs: {cfg['train']['epoch_num']}")
    print(f"  Batch size: {cfg['train']['batch_size']}")
    print(f"  Accumulation: {cfg['train']['accumulation_batches']}")
    print(f"  Effective batch: {cfg['train']['batch_size'] * cfg['train']['accumulation_batches']}")
    print(f"  Mixed precision: {cfg['train']['mixed_precision']}")
    print(f"{'='*60}\n")

    if args.resume_from_checkpoint:
        print(f"🔁 Resuming training from: {args.resume_from_checkpoint}\n")

    # Train!
    trainer.fit(
        model=net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    
    if trainer.checkpoint_callback.best_model_path:
        print(f"\n{'='*60}")
        print(f"Training Complete! 🎉")
        print(f"  Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print(f"  Best val_acc: {trainer.checkpoint_callback.best_model_score:.4f}")
        print(f"{'='*60}\n")