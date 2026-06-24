"""
Standalone COCOFake counter. Edit the paths below to match your actual layout.
"""
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# === EDIT THESE TO MATCH YOUR ACTUAL LAYOUT ===
COCO2014_PATH = "D:/sweet/binary_deepfake_detection/datasets/coco2014"
COCOFAKE_PATH = "D:/sweet/binary_deepfake_detection/datasets/coco_fake"  # change if different

# Path to the ACTIVE COCOFake dataset class file (NOT the -backup version)
DATASET_FILE = "D:/sweet/binary_deepfake_detection/coco_fake_dataset.py"
# === END EDIT ===

# Verify paths exist before doing anything
print(f"Checking paths...")
print(f"  COCO 2014:  {COCO2014_PATH} -> {'EXISTS' if os.path.isdir(COCO2014_PATH) else 'MISSING'}")
print(f"  COCOFake:   {COCOFAKE_PATH} -> {'EXISTS' if os.path.isdir(COCOFAKE_PATH) else 'MISSING'}")
print(f"  Class file: {DATASET_FILE} -> {'EXISTS' if os.path.isfile(DATASET_FILE) else 'MISSING'}")

if not os.path.isdir(COCO2014_PATH):
    print(f"\n[FATAL] COCO 2014 directory does not exist. Update COCO2014_PATH.")
    sys.exit(1)
if not os.path.isdir(COCOFAKE_PATH):
    print(f"\n[FATAL] COCOFake directory does not exist. Update COCOFAKE_PATH.")
    print(f"\nLooking for COCO-related folders under datasets/:")
    parent = os.path.dirname(COCOFAKE_PATH)
    if os.path.isdir(parent):
        for entry in sorted(os.listdir(parent)):
            full = os.path.join(parent, entry)
            if os.path.isdir(full):
                print(f"  {full}")
    sys.exit(1)
if not os.path.isfile(DATASET_FILE):
    print(f"\n[FATAL] Dataset class file does not exist. Update DATASET_FILE.")
    sys.exit(1)

# Load the dataset class directly from file (bypasses any package issues)
spec = importlib.util.spec_from_file_location("coco_fake_module", DATASET_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COCOFakeDataset = module.COCOFakeDataset
print(f"[OK] Loaded COCOFakeDataset from {DATASET_FILE}\n")

# Run counts
totals = {"real": 0, "fake": 0}
for split in ["train", "val"]:
    print(f"Counting {split}...", flush=True)
    try:
        ds = COCOFakeDataset(
            coco2014_path=COCO2014_PATH,
            coco_fake_path=COCOFAKE_PATH,
            split=split, mode="single", resolution=224,
        )
        real = sum(1 for x in ds.items if x["is_real"])
        fake = len(ds.items) - real
        ratio = f"{real/fake:.2f}:1" if fake > 0 else "all real"
        print(f"  {split:>5}: total={len(ds):>8,}  real={real:>7,}  fake={fake:>7,}  (real:fake = {ratio})")
        totals["real"] += real
        totals["fake"] += fake
    except Exception as e:
        print(f"  {split:>5}: ERROR -> {type(e).__name__}: {e}")

print(f"\n  TOTAL: real={totals['real']:>7,}  fake={totals['fake']:>7,}")