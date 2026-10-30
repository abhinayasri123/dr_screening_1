import sys
from pathlib import Path
import os
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")

p2 = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"
im = Image.open(p2)
arr = np.array(im.convert("RGB"))

print("User Image 2 details:")
print(f"Shape: {arr.shape}")
print("Corner (0:20, 0:20):", arr[0:5, 0:5, :])
print("Max pixel in image:", arr.max(), "Min pixel:", arr.min())

# Check how many pure black pixels (0,0,0) or near black (<10)
near_black = (arr < 15).all(axis=2)
print("Near-black fraction:", near_black.mean())

# Check how many pure white pixels (>240)
near_white = (arr > 240).all(axis=2)
print("Near-white fraction:", near_white.mean())

# Center 50x50 patch
cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
center_patch = arr[cy-25:cy+25, cx-25:cx+25, :]
print(f"Center patch mean RGB: {center_patch.mean(axis=(0,1)).round(1)}, std: {center_patch.std(axis=(0,1)).round(1)}")
