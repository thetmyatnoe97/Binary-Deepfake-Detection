from pprint import pprint
import argparse
import gc
from os.path import join
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import random

from cifake_dataset import CIFAKEDataset
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
from face_dataset import FaceDataset
from face140k_dataset import Face140kDataset
from forensics_dataset import ForensicsDataset

# ✅ Fixed import
from model import BNext4DFR
from lib.util import load_config


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/cifake_baseline.cfg",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file",
        required=True,  # ✅ Make it required
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # Load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # Preliminary setup
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # Get data
    if cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        test_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "coco_fake":
        print(
            f"Loading COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
        )
        test_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Loading DFFD dataset from {cfg['dataset']['dffd_path']}")
        test_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "face":
        print(f"Loading Face dataset from {cfg['dataset']['face_path']}")
        test_dataset = FaceDataset(
            dataset_path=cfg["dataset"]["face_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "face140k":
        print(f"Loading Face_140k dataset from {cfg['dataset']['face140k_path']}")
        test_dataset = Face140kDataset(
            dataset_path=cfg["dataset"]["face140k_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "forensics":
        print(f"Loading Forensics dataset from {cfg['dataset']['forensics_path']}")
        test_dataset = ForensicsDataset(
            dataset_path=cfg["dataset"]["forensics_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "artifact":
        print(f"Loading Artifact dataset from {cfg['dataset']['artifact_path']}")
        test_dataset = ArtifactDataset(
            dataset_path=cfg["dataset"]["artifact_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )

    # Load dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Load checkpoint
    checkpoint_path = args.checkpoint
    
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # ✅ Load model from checkpoint properly
    try:
        net = BNext4DFR.load_from_checkpoint(
            checkpoint_path,
            strict=True  # Try strict loading first
        )
        print(f"✅ Successfully loaded checkpoint!")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}")
        print(f"Attempting to load with strict=False...")
        try:
            net = BNext4DFR.load_from_checkpoint(
                checkpoint_path,
                strict=False
            )
            print(f"✅ Loaded with strict=False")
        except Exception as e2:
            print(f"❌ Failed to load checkpoint: {e2}")
            print("\nTrying manual loading...")
            
            # Manual loading as fallback
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Get hyperparameters from checkpoint
            hparams = checkpoint.get('hyper_parameters', {})
            
            # Create model with saved hyperparameters
            net = BNext4DFR(**hparams)
            
            # Load state dict
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"✅ Manually loaded checkpoint")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()  # ✅ Set to evaluation mode
    
    # Print model info
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model Information:")
    print(f"  Backbone: {net.hparams.get('backbone', 'Unknown')}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # ✅ Check correct attributes
    if hasattr(net, 'add_fft_magnitude'):
        print(f"  FFT Magnitude: {net.add_fft_magnitude}")
    if hasattr(net, 'add_lbp_channel'):
        print(f"  LBP Channel: {net.add_lbp_channel}")
    if hasattr(net, 'use_frequency_attention'):
        print(f"  Frequency Attention: {net.use_frequency_attention}")
        if net.use_frequency_attention:
            print(f"  Attention Position: {net.attention_position}")
    print(f"{'='*60}\n")

    # Setup test logger
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "DFAD_BNext_Test"
    run_label = os.path.basename(args.cfg).split(".")[0]
    run = cfg["dataset"]["name"] + f"_test_{date}_{run_label}"
    
    # WandB setup
    try:
        import wandb
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
    
    # Trainer setup
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
        limit_test_batches=cfg["test"]["limit_test_batches"],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Testing:")
    print(f"  Dataset: {cfg['dataset']['name']}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {cfg['test']['batch_size']}")
    print(f"{'='*60}\n")
    
    # Test the model
    results = trainer.test(model=net, dataloaders=test_loader)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}")
    for key, value in results[0].items():
        if 'test_' in key:
            metric_name = key.replace('test_', '').replace('_', ' ').title()
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    print(f"{'='*60}\n")

    # Save predictions if requested
    if args.save_predictions:
        print("Saving predictions...")
        predictions_dir = "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Get all predictions
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        net = net.to(device)  # Make sure model is on GPU
        net.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["is_real"]
                
                # Handle label shape
                if labels.dim() > 1:
                    labels = labels[:, 0]
                labels = labels.cpu().numpy()
                
                outputs = net(images)
                logits = outputs["logits"][:, 0].cpu()
                probs = torch.sigmoid(logits).numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)
        
        # Save to file
        checkpoint_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
        results_file = join(
            predictions_dir, 
            f"{cfg['dataset']['name']}_{checkpoint_name}_predictions_{date}.npz"
        )
        np.savez(
            results_file,
            predictions=np.array(all_preds),
            labels=np.array(all_labels),
            probabilities=np.array(all_probs)
        )
        print(f"Predictions saved to: {results_file}")
        
        # Calculate and display confusion matrix
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(all_labels, all_preds)
            print(f"\n{'='*60}")
            print(f"Confusion Matrix:")
            print(f"{'='*60}")
            print(f"                Predicted")
            print(f"              Fake    Real")
            print(f"Actual Fake   {cm[0, 0]:<6}  {cm[0, 1]:<6}")
            print(f"       Real   {cm[1, 0]:<6}  {cm[1, 1]:<6}")
            print(f"{'='*60}\n")
            
            print(f"Classification Report:")
            print(f"{'='*60}")
            print(classification_report(
                all_labels, 
                all_preds, 
                target_names=['Fake', 'Real'],
                digits=4
            ))
            print(f"{'='*60}\n")
        except ImportError:
            print("sklearn not installed. Skipping detailed metrics.")