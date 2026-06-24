"""
crop_faces_forensics.py — Crop faces from FF++ frames using MTCNN

Reads from:  .../c23/images/<video_id>/*.png
Writes to:   .../c23/faces/<video_id>/*.png

Usage:
    python crop_faces_forensics.py \
        --dataset_path D:\sweet\binary_deepfake_detection\datasets\FF++ \
        --compression c23
"""

import os
import argparse
from os.path import join, isdir, exists
from os import makedirs, listdir
from PIL import Image
import torch

MANIPULATION_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ---------------------------------------------------------------------------
# Load MTCNN once globally
# ---------------------------------------------------------------------------

def load_mtcnn(image_size: int = 224, device: str = "cpu"):
    from facenet_pytorch import MTCNN
    return MTCNN(
        image_size=image_size,
        margin=30,          # extra pixels around face box
        min_face_size=40,   # ignore tiny detections
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,     # only largest / most confident face
        device=device,
        post_process=False, # return [0,255] not normalized tensor
    )


# ---------------------------------------------------------------------------
# Crop a single image
# ---------------------------------------------------------------------------

def crop_and_save(mtcnn, src_path: str, dst_path: str, image_size: int):
    """
    Detect and crop the face from src_path, save to dst_path.
    Falls back to a plain center-crop resize if no face is detected.
    """
    img = Image.open(src_path).convert("RGB")

    # MTCNN returns a uint8 tensor (C,H,W) in [0,255] when post_process=False
    face_tensor = mtcnn(img)

    if face_tensor is not None:
        # Convert tensor (C,H,W) uint8 → PIL Image
        face_np = face_tensor.permute(1, 2, 0).byte().numpy()
        face_img = Image.fromarray(face_np)
    else:
        # Fallback: no face detected — use center crop
        w, h = img.size
        side = min(w, h)
        left   = (w - side) // 2
        top    = (h - side) // 2
        face_img = img.crop((left, top, left + side, top + side))
        face_img = face_img.resize((image_size, image_size), Image.BILINEAR)

    face_img.save(dst_path)


# ---------------------------------------------------------------------------
# Process one video folder
# ---------------------------------------------------------------------------

def process_video_folder(
    mtcnn,
    src_dir: str,
    dst_dir: str,
    image_size: int,
    label: str,
):
    makedirs(dst_dir, exist_ok=True)
    frames = sorted(f for f in listdir(src_dir) if f.lower().endswith(".png"))

    if not frames:
        return 0

    saved = 0
    for fname in frames:
        dst_path = join(dst_dir, fname)
        if exists(dst_path):          # skip already-cropped frames
            saved += 1
            continue
        src_path = join(src_dir, fname)
        try:
            crop_and_save(mtcnn, src_path, dst_path, image_size)
            saved += 1
        except Exception as e:
            print(f"    [WARN] {src_path}: {e}")

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crop faces from FF++ frames using MTCNN"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Root directory of the FF++ dataset"
    )
    parser.add_argument(
        "--compression", type=str, default="c23",
        choices=["raw", "c23", "c40"],
        help="Compression level (default: c23)"
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Output face crop size in pixels (default: 224)"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=MANIPULATION_METHODS,
        choices=MANIPULATION_METHODS,
        help="Which manipulation methods to crop (default: all)"
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"FF++ Face Cropping")
    print(f"  Dataset    : {args.dataset_path}")
    print(f"  Compression: {args.compression}")
    print(f"  Image size : {args.image_size}x{args.image_size}")
    print(f"  Device     : {device}")
    print(f"  Methods    : {args.methods}")
    print(f"{'='*60}\n")

    # ── Load MTCNN ────────────────────────────────────────────────────
    print("Loading MTCNN face detector...")
    mtcnn = load_mtcnn(image_size=args.image_size, device=device)
    print("MTCNN loaded.\n")

    # ── REAL sequences ────────────────────────────────────────────────
    print("[1/5] Cropping REAL (original_sequences/youtube)")
    real_images = join(
        args.dataset_path,
        "original_sequences", "youtube", args.compression, "images"
    )
    real_faces = join(
        args.dataset_path,
        "original_sequences", "youtube", args.compression, "faces"
    )

    if not isdir(real_images):
        print(f"  [SKIP] Not found: {real_images}")
    else:
        video_ids = sorted(listdir(real_images))
        total = len(video_ids)
        for i, vid_id in enumerate(video_ids, 1):
            src = join(real_images, vid_id)
            dst = join(real_faces,  vid_id)
            if not isdir(src):
                continue
            n = process_video_folder(mtcnn, src, dst, args.image_size, "REAL")
            print(f"  [{i:4d}/{total}] REAL/{vid_id} → {n} faces")

    # ── FAKE sequences ────────────────────────────────────────────────
    for mi, method in enumerate(args.methods, 2):
        print(f"\n[{mi}/{len(args.methods)+1}] Cropping FAKE ({method})")
        fake_images = join(
            args.dataset_path,
            "manipulated_sequences", method, args.compression, "images"
        )
        fake_faces = join(
            args.dataset_path,
            "manipulated_sequences", method, args.compression, "faces"
        )

        if not isdir(fake_images):
            print(f"  [SKIP] Not found: {fake_images}")
            continue

        video_ids = sorted(listdir(fake_images))
        total = len(video_ids)
        for i, vid_id in enumerate(video_ids, 1):
            src = join(fake_images, vid_id)
            dst = join(fake_faces,  vid_id)
            if not isdir(src):
                continue
            n = process_video_folder(mtcnn, src, dst, args.image_size, method)
            print(f"  [{i:4d}/{total}] {method}/{vid_id} → {n} faces")

    # ── Done ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Face cropping complete!")
    print(f"\nFaces saved to:")
    print(f"  original_sequences/youtube/{args.compression}/faces/")
    for method in args.methods:
        print(f"  manipulated_sequences/{method}/{args.compression}/faces/")
    print(f"\nNext: update forensics_dataset.py to use 'faces' instead of 'images'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()