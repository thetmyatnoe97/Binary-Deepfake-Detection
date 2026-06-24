"""
extract_frames_forensics.py — Extract FF++ video frames using OpenCV (no ffmpeg needed)

Usage:
    python extract_frames_forensics.py \
        --dataset_path D:\sweet\binary_deepfake_detection\datasets\FF++ \
        --compression c23 \
        --fps 1 \
        --max_frames 30
"""

import os
import argparse
from os.path import join, isdir
from os import makedirs, listdir

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV not found. Install it with: conda install -c conda-forge opencv")


MANIPULATION_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: int,
    max_frames: int,
):
    """
    Extract frames from a single video using OpenCV.

    Args:
        video_path : path to .mp4 file
        output_dir : folder where .png frames will be saved
        fps        : how many frames to extract per second of video
        max_frames : maximum total frames to extract
    """
    makedirs(output_dir, exist_ok=True)

    # Skip if already fully extracted
    existing = [f for f in listdir(output_dir) if f.endswith(".png")]
    if len(existing) >= max_frames:
        print(f"  [SKIP] Already done: {output_dir} ({len(existing)} frames)")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0  # safe fallback

    # How many original frames to skip between each saved frame
    frame_interval = max(1, round(video_fps / fps))

    frame_idx = 0    # current frame position in the video
    saved = 0        # how many frames we have saved so far

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = join(output_dir, f"{saved:04d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  [OK] {os.path.basename(video_path)} → {saved} frames saved")


# ---------------------------------------------------------------------------
# Folder-level processing
# ---------------------------------------------------------------------------

def process_sequence_folder(
    videos_dir: str,
    images_dir: str,
    fps: int,
    max_frames: int,
):
    """Process all .mp4 files in a videos/ folder into a parallel images/ folder."""
    if not isdir(videos_dir):
        print(f"  [SKIP] Folder not found: {videos_dir}")
        return

    video_files = sorted(f for f in listdir(videos_dir) if f.endswith(".mp4"))
    total = len(video_files)

    if total == 0:
        print(f"  [SKIP] No .mp4 files found in: {videos_dir}")
        return

    print(f"  Found {total} videos → saving frames to: {images_dir}")

    for i, video_file in enumerate(video_files, start=1):
        video_path = join(videos_dir, video_file)
        video_id   = os.path.splitext(video_file)[0]   # e.g. "000" or "000_001"
        output_dir = join(images_dir, video_id)

        print(f"  [{i:4d}/{total}]", end=" ")
        extract_frames_from_video(video_path, output_dir, fps, max_frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract FF++ frames to images/ using OpenCV (no ffmpeg needed)"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Root directory of the FF++ dataset"
    )
    parser.add_argument(
        "--compression", type=str, default="c23",
        choices=["raw", "c23", "c40"],
        help="Compression level to extract (default: c23)"
    )
    parser.add_argument(
        "--fps", type=int, default=1,
        help="Frames per second to extract (default: 1)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=30,
        help="Maximum frames to extract per video (default: 30)"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=MANIPULATION_METHODS,
        choices=MANIPULATION_METHODS,
        help="Which manipulation methods to extract (default: all four)"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"FF++ Frame Extraction (OpenCV)")
    print(f"  Dataset path : {args.dataset_path}")
    print(f"  Compression  : {args.compression}")
    print(f"  FPS          : {args.fps}")
    print(f"  Max frames   : {args.max_frames}")
    print(f"  Methods      : {args.methods}")
    print(f"{'='*60}\n")

    # ── REAL sequences ────────────────────────────────────────────────
    print("[1/5] Extracting REAL (original_sequences/youtube)")
    real_videos = join(
        args.dataset_path,
        "original_sequences", "youtube", args.compression, "videos"
    )
    real_images = join(
        args.dataset_path,
        "original_sequences", "youtube", args.compression, "images"
    )
    process_sequence_folder(real_videos, real_images, args.fps, args.max_frames)

    # ── FAKE sequences (one per method) ──────────────────────────────
    for i, method in enumerate(args.methods, start=2):
        print(f"\n[{i}/{len(args.methods)+1}] Extracting FAKE ({method})")
        fake_videos = join(
            args.dataset_path,
            "manipulated_sequences", method, args.compression, "videos"
        )
        fake_images = join(
            args.dataset_path,
            "manipulated_sequences", method, args.compression, "images"
        )
        process_sequence_folder(fake_videos, fake_images, args.fps, args.max_frames)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"\nResulting structure:")
    print(f"  original_sequences/youtube/{args.compression}/images/<id>/*.png  <- REAL")
    for method in args.methods:
        print(f"  manipulated_sequences/{method}/{args.compression}/images/<src>_<tgt>/*.png  <- FAKE")
    print(f"\nNext step: python forensics_dataset.py {args.dataset_path} {args.compression}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()