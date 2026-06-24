"""
cross_dataset_eval_missing.py
=============================
Run ONLY the cells missing from the original sweep:
  - (source=DFFD,   target=COCOFake) for all 9 model variants
  - (source=CIFAKE, target=COCOFake) for all 9 model variants
That is 18 evaluations total.

Same infrastructure as cross_dataset_eval.py, but the cell list is hard-coded
to the missing pairs so the script does not waste time re-running cells
that already have results in the previous CSV.

Likely cause of the original gap: the cocofake config files
(cocofakeT_before.cfg, etc.) were not found, or the COCOFake test
dataloader failed silently. Before running this, please:

  1) Verify that the following config files EXIST in your configs folder:
        cocofakeT_before.cfg   cocofakeS_before.cfg   cocofakeM_before.cfg
        cocofakeT_after.cfg    cocofakeS_after.cfg    cocofakeM_after.cfg
        cocofakeT_both.cfg     cocofakeS_both.cfg     cocofakeM_both.cfg

  2) If any are missing, copy them from the (training) cocofake configs
     you used to train those checkpoints in the first place. The config
     should specify dataset.name = "coco_fake" and the correct paths.

  3) Run a dry test first by trying a single cell:
        python cross_dataset_eval_missing.py --dry_run

Output
------
Appends rows to cross_dataset_results_missing.csv.
You can then concatenate with your original CSV to get the complete matrix.
"""

import argparse
import gc
import os
import random
import sys
import warnings
from datetime import datetime
from os.path import join, exists
from pprint import pprint

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

from cifake_dataset import CIFAKEDataset
from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
from model import BNext4DFR
from lib.util import load_config

# ============================================================
# PATHS — edit to match your environment
# ============================================================
PROJECT_ROOT = r"D:/sweet/binary_deepfake_detection"
CKPT_ROOT    = join(PROJECT_ROOT, "checkpoints")
CONFIG_ROOT  = join(PROJECT_ROOT, "configs")
RESULTS_DIR  = join(PROJECT_ROOT, "cross_dataset_results")

# ============================================================
# THE 18 MISSING CELLS
# ============================================================
ALL_MODELS = [
    "phasedfd_t", "phasedfd_s", "phasedfd_m",
    "dualdfd_t",  "dualdfd_s",  "dualdfd_m",
    "fulldfd_t",  "fulldfd_s",  "fulldfd_m",
]
MISSING_TARGETS_BY_SOURCE = {
    "dffd":   ["cocofake"],
    "cifake": ["cocofake"],
}

# Helpers (same as the main script)
_ATTENTION_SUFFIX = {
    "phasedfd": "before",
    "dualdfd":  "after",
    "fulldfd":  "both",
}
_SIZE_SUFFIX = {"t": "T", "s": "S", "m": "M"}


def checkpoint_path(model_variant, source_dataset):
    return join(CKPT_ROOT, source_dataset, f"{model_variant}.ckpt")


def config_path(model_variant, target_dataset):
    family, size = model_variant.split("_")
    position = _ATTENTION_SUFFIX[family]
    size_suf = _SIZE_SUFFIX[size]
    # Map dataset name to the config-filename prefix.
    # User confirmed COCOFake configs are named with "coco" prefix
    # (e.g. cocoM_before.cfg) rather than "cocofake".
    DATASET_PREFIX = {
        "dffd":     "dffd",
        "cifake":   "cifake",
        "cocofake": "coco",
    }
    prefix = DATASET_PREFIX.get(target_dataset, target_dataset)
    candidates = [
        join(CONFIG_ROOT, f"{prefix}{size_suf}_{position}.cfg"),
        join(CONFIG_ROOT, f"{prefix}{size_suf}_{position}.yaml"),
        join(CONFIG_ROOT, f"{prefix}{size_suf}_{position}.yml"),
    ]
    for c in candidates:
        if exists(c):
            return c
    return None


def build_test_dataset(cfg):
    name = cfg["dataset"]["name"]
    res  = cfg["test"]["resolution"]
    if name == "cifake":
        return CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test", resolution=res,
        )
    if name == "coco_fake":
        return COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val", mode="single", resolution=res,
        )
    if name == "dffd":
        return DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="test", resolution=res,
        )
    raise ValueError(f"Unsupported dataset name in config: {name}")


def load_model(ckpt_path, device):
    try:
        net = BNext4DFR.load_from_checkpoint(ckpt_path, strict=True)
    except Exception:
        try:
            net = BNext4DFR.load_from_checkpoint(ckpt_path, strict=False)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hparams = ckpt.get("hyper_parameters", {})
            net = BNext4DFR(**hparams)
            net.load_state_dict(ckpt["state_dict"], strict=False)
    return net.to(device).eval()


def extract_metrics(results_dict):
    acc, auc = None, None
    for k, v in results_dict.items():
        kl = k.lower()
        if "acc" in kl and acc is None:
            acc = float(v)
        if "auc" in kl and auc is None:
            auc = float(v)
    return acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned cells and check file existence.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Subset of models to evaluate (default: all 9).")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Override the default output CSV path.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = args.output_csv or join(
        RESULTS_DIR, f"cross_dataset_missing_{timestamp}.csv"
    )

    # Build the list of missing cells
    plan = []
    for model in args.models:
        if model not in ALL_MODELS:
            print(f"WARNING: unknown model '{model}', skipping.")
            continue
        for source, targets in MISSING_TARGETS_BY_SOURCE.items():
            for target in targets:
                plan.append((model, source, target))

    print("=" * 64)
    print(" Cross-dataset evaluation — MISSING CELLS ONLY")
    print("=" * 64)
    print(f" Models:       {args.models}")
    print(f" Missing cells: {len(plan)}  (out of 18 possible)")
    print(f" Output CSV:   {output_csv}")
    print("=" * 64)

    if args.dry_run:
        print("\n-- DRY RUN --")
        missing_files = 0
        for i, (m, s, t) in enumerate(plan, 1):
            ckpt = checkpoint_path(m, s)
            cfg  = config_path(m, t)
            ckpt_ok = exists(ckpt)
            cfg_ok = cfg is not None and exists(cfg)
            status = "OK" if (ckpt_ok and cfg_ok) else "MISSING FILES"
            print(f"[{i:>2}] {m}  {s} -> {t}   [{status}]")
            print(f"     ckpt:   {ckpt}     exists={ckpt_ok}")
            print(f"     config: {cfg}     exists={cfg_ok}")
            if not ckpt_ok or not cfg_ok:
                missing_files += 1
        print(f"\nDry run done. Cells with missing files: {missing_files}/{len(plan)}")
        if missing_files:
            print("Fix the missing files above before running for real.")
        return

    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("model,source_dataset,target_dataset,accuracy,auc,n_test,timestamp\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")

    for idx, (model, src, tgt) in enumerate(plan, 1):
        print("\n" + "=" * 64)
        print(f"[{idx}/{len(plan)}]  {model}   trained on {src}   ->   tested on {tgt}")
        print("=" * 64)

        ckpt = checkpoint_path(model, src)
        cfg_path_ = config_path(model, tgt)

        if not exists(ckpt):
            print(f"  SKIP: checkpoint missing: {ckpt}")
            continue
        if cfg_path_ is None or not exists(cfg_path_):
            print(f"  SKIP: config missing for target {tgt}, model {model}")
            print(f"        looked for: {CONFIG_ROOT}/{tgt}*_*.cfg")
            continue

        cfg = load_config(cfg_path_)
        print(f"  Checkpoint: {ckpt}")
        print(f"  Config:     {cfg_path_}")
        print(f"  Test set:   {cfg['dataset']['name']}  @ {cfg['test']['resolution']}px")

        seed = cfg["test"]["seed"]
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

        try:
            test_dataset = build_test_dataset(cfg)
        except Exception as e:
            print(f"  ERROR building dataset: {e}")
            continue

        test_loader = DataLoader(
            test_dataset, batch_size=cfg["test"]["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True,
            persistent_workers=False,
        )
        n_test = len(test_dataset)
        print(f"  Samples:    {n_test}")

        net = load_model(ckpt, device)

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
            limit_test_batches=cfg["test"]["limit_test_batches"],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        try:
            out = trainer.test(model=net, dataloaders=test_loader, verbose=False)
            acc, auc = extract_metrics(out[0]) if out else (None, None)
        except Exception as e:
            print(f"  ERROR during evaluation: {e}")
            acc, auc = None, None

        if acc is None or auc is None:
            print(f"  WARNING: could not extract metrics")
            acc = acc if acc is not None else float("nan")
            auc = auc if auc is not None else float("nan")

        if acc is not None and acc <= 1.0:
            acc *= 100.0
        if auc is not None and auc <= 1.0:
            auc *= 100.0

        print(f"  Result: Accuracy = {acc:.2f}%   AUC = {auc:.2f}%")

        with open(output_csv, "a", encoding="utf-8") as f:
            ts = datetime.utcnow().isoformat() + "Z"
            f.write(f"{model},{src},{tgt},{acc:.4f},{auc:.4f},{n_test},{ts}\n")

        del net, trainer, test_loader, test_dataset
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print(f" Done. Results written to: {output_csv}")
    print("=" * 64)


if __name__ == "__main__":
    main()