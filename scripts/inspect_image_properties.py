import numpy as np
import cv2
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")

img1_path = PROJECT_ROOT / "uploads" / "bc4aed6c-8b96-439b-9789-58118d9295a2.jpg"
img2_path = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"
dataset_sample = PROJECT_ROOT / "data" / "retinal" / "test" / "0" / "demo_cls0_test_000.png"

for label, p in [("User Image 1 (OIP.jpg)", img1_path), ("User Image 2 (d4d3...)", img2_path), ("Dataset Image (demo_cls0...)", dataset_sample)]:
    im = Image.open(p)
    arr = np.array(im.convert("RGB"))
    print(f"=== {label} ===")
    print(f"  Dimensions: {im.size} (W x H), Aspect Ratio: {im.size[0]/im.size[1]:.2f}")
    print(f"  Channel Means: R={arr[:,:,0].mean():.1f}, G={arr[:,:,1].mean():.1f}, B={arr[:,:,2].mean():.1f}")
    print(f"  Channel Stds:  R={arr[:,:,0].std():.1f}, G={arr[:,:,1].std():.1f}, B={arr[:,:,2].std():.1f}")
    
    # Check borders (outer 5%)
    h, w, _ = arr.shape
    border_pixels = np.concatenate([
        arr[:int(h*0.05), :, :].reshape(-1, 3),
        arr[-int(h*0.05):, :, :].reshape(-1, 3),
        arr[:, :int(w*0.05), :].reshape(-1, 3),
        arr[:, -int(w*0.05):, :].reshape(-1, 3),
    ])
    print(f"  Border Mean RGB: {border_pixels.mean(axis=0).round(1)} (Black circular mask border expected in fundus photography)")
    
    # Check center circle vs background
    mask = (arr[:,:,0] > 15) | (arr[:,:,1] > 15) | (arr[:,:,2] > 15)
    foreground_pct = mask.mean() * 100
    print(f"  Non-black foreground percentage: {foreground_pct:.1f}%")
