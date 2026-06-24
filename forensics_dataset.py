from os import listdir
from os.path import exists, isdir, join
import json
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.v2 as Tv2
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIPULATION_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
COMPRESSIONS = {"raw", "c23", "c40"}

# Official split JSON lives here inside the dataset root
SPLIT_DIR = "splits"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def _load_split_ids(dataset_path: str, split: str) -> Optional[List[List[str]]]:
    """
    Load the official FF++ split JSON.

    Returns a list of [source_id, target_id] pairs, or None if the file
    does not exist (triggering the fallback random split).
    """
    json_path = join(dataset_path, SPLIT_DIR, f"{split}.json")
    if not exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)  # e.g. [["000", "001"], ["002", "003"], ...]


def _video_ids_from_split(pairs: List[List[str]]) -> List[str]:
    """Extract unique source video ids from the split pairs."""
    return sorted({pair[0] for pair in pairs})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForensicsDataset(Dataset):
    """
    FaceForensics++ dataset loader compatible with BNext4DFR.

    Args:
        dataset_path        : Root of the FF++ dataset directory.
        split               : "train" | "val" | "test"
        compression         : "raw" | "c23" | "c40"  (default "c23")
        manipulation_methods: List of manipulation folders to use as FAKE.
                              None → all four: Deepfakes, Face2Face,
                              FaceSwap, NeuralTextures.
        resolution          : Square crop size for the model (default 224).
        balance             : If True, downsample the majority class so that
                              real and fake counts are equal (default True).
        max_frames_per_video: Maximum frames to sample from each video
                              folder. None → use all frames (default None).
        seed                : Random seed for fallback split and balancing.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        compression: str = "c23",
        manipulation_methods: Optional[List[str]] = None,
        resolution: int = 224,
        balance: bool = True,
        max_frames_per_video: Optional[int] = None,
        seed: int = 42,
    ):
        assert isdir(dataset_path), f"Dataset root not found: {dataset_path}"
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        assert compression in COMPRESSIONS, (
            f"Invalid compression '{compression}'. Choose from {COMPRESSIONS}"
        )

        self.dataset_path = dataset_path
        self.split = split
        self.compression = compression
        self.resolution = resolution
        self.balance = balance
        self.max_frames_per_video = max_frames_per_video
        self.seed = seed

        # Resolve manipulation methods
        if manipulation_methods is None:
            self.manipulation_methods = MANIPULATION_METHODS
        else:
            for m in manipulation_methods:
                assert m in MANIPULATION_METHODS, (
                    f"Unknown manipulation method '{m}'. "
                    f"Choose from {MANIPULATION_METHODS}"
                )
            self.manipulation_methods = manipulation_methods

        # Build item list
        self.items = self._parse_dataset()

        print(
            f"[ForensicsDataset] split={split} | compression={compression} | "
            f"methods={self.manipulation_methods}\n"
            f"  Real frames : {sum(1 for x in self.items if x['is_real'])}\n"
            f"  Fake frames : {sum(1 for x in self.items if not x['is_real'])}\n"
            f"  Total       : {len(self.items)}"
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _parse_dataset(self) -> List[dict]:
        """
        Walk the FF++ directory tree and collect (image_path, is_real, method)
        triples respecting the split.
        """
        # ── Determine which video IDs belong to this split ───────────
        split_pairs = _load_split_ids(self.dataset_path, self.split)

        if split_pairs is not None:
            # Official split: video IDs are the source ids in the pairs
            split_video_ids = set(_video_ids_from_split(split_pairs))
            # Build lookup set of (source, target) pairs for fake matching
            split_pairs_set = {(p[0], p[1]) for p in split_pairs}
            use_official_split = True
        else:
            # Fallback: random 80/10/10 split over all video folders found
            print(
                f"[ForensicsDataset] Warning: split JSON not found under "
                f"'{join(self.dataset_path, SPLIT_DIR)}'. "
                f"Using random 80/10/10 fallback split."
            )
            split_video_ids, split_pairs_set = self._fallback_split()
            use_official_split = False

        real_items = []
        fake_items = []

        # ── REAL frames ───────────────────────────────────────────────
        real_root = join(
            self.dataset_path,
            "original_sequences",
            "youtube",
            self.compression,
            "faces",
        )
        if isdir(real_root):
            for video_id in sorted(listdir(real_root)):
                if video_id not in split_video_ids:
                    continue
                video_dir = join(real_root, video_id)
                if not isdir(video_dir):
                    continue
                frames = self._collect_frames(video_dir)
                for frame_path in frames:
                    real_items.append({
                        "image_path": frame_path,
                        "is_real": True,
                        "method": "original",
                        "video_id": video_id,
                    })
        else:
            print(f"[ForensicsDataset] Warning: real sequence root not found: {real_root}")

        # ── FAKE frames (one subfolder per method) ────────────────────
        for method in self.manipulation_methods:
            fake_root = join(
                self.dataset_path,
                "manipulated_sequences",
                method,
                self.compression,
                "faces",
            )
            if not isdir(fake_root):
                print(
                    f"[ForensicsDataset] Warning: fake sequence root not found "
                    f"for method '{method}': {fake_root}"
                )
                continue

            for folder_name in sorted(listdir(fake_root)):
                folder_path = join(fake_root, folder_name)
                if not isdir(folder_path):
                    continue

                # Folder names are either "SRC_TGT" (e.g. "000_001")
                # or just "SRC" for some methods.
                parts = folder_name.split("_")
                if len(parts) >= 2:
                    src_id, tgt_id = parts[0], parts[1]
                    # Filter by official pairs if split is available
                    if use_official_split:
                        if (src_id, tgt_id) not in split_pairs_set:
                            continue
                    else:
                        if src_id not in split_video_ids:
                            continue
                else:
                    # Single-id folder (rare edge case)
                    src_id = parts[0]
                    if src_id not in split_video_ids:
                        continue

                frames = self._collect_frames(folder_path)
                for frame_path in frames:
                    fake_items.append({
                        "image_path": frame_path,
                        "is_real": False,
                        "method": method,
                        "video_id": folder_name,
                    })

        # ── Optional balancing ────────────────────────────────────────
        if self.balance and real_items and fake_items:
            rng = random.Random(self.seed)
            n = min(len(real_items), len(fake_items))
            real_items = rng.sample(real_items, n)
            fake_items = rng.sample(fake_items, n)

        items = real_items + fake_items

        # Shuffle so real/fake are interleaved
        rng = random.Random(self.seed)
        rng.shuffle(items)

        return items

    def _collect_frames(self, video_dir: str) -> List[str]:
        """Return sorted image paths in a video folder, optionally capped."""
        frames = sorted(
            join(video_dir, f)
            for f in listdir(video_dir)
            if _is_image(f)
        )
        if self.max_frames_per_video is not None:
            frames = frames[: self.max_frames_per_video]
        return frames

    def _fallback_split(self):
        """
        Build a deterministic random 80/10/10 split from all video IDs found
        in the original_sequences folder.

        Returns (split_video_ids: set, split_pairs_set: set).
        The pairs set is empty (unused in fallback mode).
        """
        real_root = join(
            self.dataset_path,
            "original_sequences",
            "youtube",
            self.compression,
            "faces",
        )
        if not isdir(real_root):
            return set(), set()

        all_ids = sorted(
            d for d in listdir(real_root) if isdir(join(real_root, d))
        )

        rng = random.Random(self.seed)
        rng.shuffle(all_ids)

        n = len(all_ids)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        if self.split == "train":
            chosen = set(all_ids[:n_train])
        elif self.split == "val":
            chosen = set(all_ids[n_train: n_train + n_val])
        else:  # test
            chosen = set(all_ids[n_train + n_val:])

        return chosen, set()

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.items)

    def read_image(self, path: str) -> torch.Tensor:
        """
        Load an image and apply split-appropriate augmentations.

        Train: random flip + color jitter + random crop
        Val / Test: center crop only

        Returns float32 tensor (3, H, W) in [0, 1].
        NOTE: No ImageNet normalisation — BNext4DFR does it in forward().
        """
        image = Image.open(path).convert("RGB")
        resize_to = self.resolution + self.resolution // 8  # e.g. 224 → 252

        if self.split == "train":
            transforms = Tv2.Compose([
                Tv2.Resize(resize_to, interpolation=Tv2.InterpolationMode.BILINEAR),
                Tv2.RandomHorizontalFlip(p=0.5),
                Tv2.RandomVerticalFlip(p=0.1),
                Tv2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                Tv2.RandomCrop(self.resolution),
                Tv2.ToImage(),
                Tv2.ToDtype(torch.float32, scale=True),
            ])
        else:
            transforms = Tv2.Compose([
                Tv2.Resize(resize_to, interpolation=Tv2.InterpolationMode.BILINEAR),
                Tv2.CenterCrop(self.resolution),
                Tv2.ToImage(),
                Tv2.ToDtype(torch.float32, scale=True),
            ])

        return transforms(image)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        return {
            "image_path": item["image_path"],
            "image": self.read_image(item["image_path"]),
            "is_real": torch.tensor(
                1.0 if item["is_real"] else 0.0, dtype=torch.float32
            ),
            "method": item["method"],
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_image(image: torch.Tensor):
        import matplotlib.pyplot as plt
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.show()
        plt.close()

    def _plot_labels_distribution(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt

        # Count per method
        from collections import Counter
        method_counts = Counter(item["method"] for item in self.items)
        real_count = sum(1 for item in self.items if item["is_real"])
        fake_count = len(self.items) - real_count

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: real vs fake
        axes[0].bar(["Real", "Fake"], [real_count, fake_count], color=["steelblue", "tomato"])
        axes[0].set_title(f"[FF++/{self.compression}] Real vs Fake — {self.split}")
        axes[0].set_ylabel("Frame count")
        for i, v in enumerate([real_count, fake_count]):
            axes[0].text(i, v + max(real_count, fake_count) * 0.01, str(v), ha="center")

        # Right: per-method breakdown
        methods = list(method_counts.keys())
        counts = [method_counts[m] for m in methods]
        colors = ["steelblue"] + ["tomato"] * (len(methods) - 1)
        axes[1].bar(methods, counts, color=colors)
        axes[1].set_title(f"Frames per method — {self.split}")
        axes[1].set_ylabel("Frame count")
        axes[1].tick_params(axis="x", rotation=20)
        for i, v in enumerate(counts):
            axes[1].text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved label distribution plot to: {save_path}")
        else:
            plt.show()
        plt.close()

    def get_method_stats(self) -> dict:
        """Return per-method frame counts as a dict."""
        from collections import Counter
        return dict(Counter(item["method"] for item in self.items))


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/faceforensics++"
    compression  = sys.argv[2] if len(sys.argv) > 2 else "c23"

    print(f"\nDataset root : {dataset_path}")
    print(f"Compression  : {compression}\n")

    for split in ["train", "val", "test"]:
        ds = ForensicsDataset(
            dataset_path=dataset_path,
            split=split,
            compression=compression,
            max_frames_per_video=10,  # cap for quick test
        )

        sample = ds[0]
        print(f"\n[{split}] Sample keys:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor{tuple(v.shape)}, dtype={v.dtype}, "
                      f"range=[{v.min():.3f}, {v.max():.3f}]")
            else:
                print(f"  {k}: {v}")

        print(f"  Method stats: {ds.get_method_stats()}")
        ds._plot_labels_distribution(save_path=f"_{split}_{compression}_labels_ff++.png")

    print("\nSanity check passed ✓")