import os
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")

data_dir = PROJECT_ROOT / "data" / "retinal"

for split in ["train", "validation", "test"]:
    s_dir = data_dir / split
    print(f"\n=== SPLIT: {split} ===")
    for c in range(5):
        c_dir = s_dir / str(c)
        if not c_dir.exists():
            continue
        files = list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpg"))
        if files:
            sample = Image.open(files[0])
            s_arr = np.array(sample)
            print(f"  Class {c}: {len(files)} files | Sample: {files[0].name} | Size: {sample.size} | Mode: {sample.mode} | Mean RGB: {s_arr.mean(axis=(0,1)).round(1)}")
