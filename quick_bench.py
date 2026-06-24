# Save as quick_bench.py in your project folder and run it
import torch
import time
from cifake_dataset import CIFAKEDataset
from torch.utils.data import DataLoader
from model import BNext4DFR

print("[1] Loading dataset...")
dataset = CIFAKEDataset(
    dataset_path="D:/sweet/binary_deepfake_detection/datasets/cifake",
    split="train", resolution=224,
)
loader = DataLoader(dataset, batch_size=32, num_workers=8,
                    pin_memory=True, prefetch_factor=4, persistent_workers=True)

print("[2] Loading model...")
model = BNext4DFR(
    num_classes=2, backbone="BNext-T",
    add_fft_magnitude=True, add_lbp_channel=True,
    use_frequency_attention=True, attention_position="before_backbone",
)
device = torch.device("cuda")
model = model.to(device).eval()

print("[3] Benchmarking 20 batches...")
loader_iter = iter(loader)
times = []
for i in range(20):
    batch = next(loader_iter)
    images = batch["image"].to(device)
    start = time.time()
    with torch.no_grad():
        out = model(images)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Batch {i+1:02d}: {elapsed:.3f}s")

avg = sum(times)/len(times)
total_batches = len(loader)
print(f"\nAvg per batch : {avg:.3f}s")
print(f"Total batches : {total_batches}")
print(f"Estimated epoch: {avg * total_batches / 60:.1f} minutes")