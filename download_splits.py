import urllib.request
import os

SPLITS_DIR = r"D:\sweet\binary_deepfake_detection\datasets\FF++\splits"
os.makedirs(SPLITS_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"

for split in ["train.json", "val.json", "test.json"]:
    url = f"{BASE_URL}/{split}"
    dst = os.path.join(SPLITS_DIR, split)
    print(f"Downloading {split}...")
    urllib.request.urlretrieve(url, dst)
    print(f"  Saved to {dst}")

print("\nDone! splits/ folder is ready.")
