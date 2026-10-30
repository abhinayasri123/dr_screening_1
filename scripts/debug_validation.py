import sys
from pathlib import Path
PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.quality_checker import assess_image_quality, validate_retinal_fundus_image
import numpy as np
from PIL import Image

data_dir = PROJECT_ROOT / "data" / "retinal"
sample_imgs = list(data_dir.rglob("*.png"))[:20]
print(f"Testing {len(sample_imgs)} sample retinal images from dataset...\n", flush=True)

rejected_count = 0
for p in sample_imgs:
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB"))
    is_ret, msg, m = validate_retinal_fundus_image(arr)
    reason = m.get("reason", "OK")
    prob = m.get("neural_retinal_prob", None)
    if not is_ret:
        rejected_count += 1
        print(f"REJECTED: {p.name} | reason: {reason} | metrics: {m}")
    else:
        print(f"ACCEPTED: {p.name} | prob: {prob}")

print(f"\nSummary for data/retinal samples: {len(sample_imgs) - rejected_count} accepted, {rejected_count} rejected.")

upload_dir = PROJECT_ROOT / "uploads"
upload_imgs = [p for p in upload_dir.glob("*.png") if not p.name.startswith("gradcam_")][:20]
print(f"\nTesting {len(upload_imgs)} sample uploads...\n", flush=True)

u_rejected_count = 0
for p in upload_imgs:
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB"))
    is_ret, msg, m = validate_retinal_fundus_image(arr)
    reason = m.get("reason", "OK")
    prob = m.get("neural_retinal_prob", None)
    if not is_ret:
        u_rejected_count += 1
        print(f"REJECTED UPLOAD: {p.name} | reason: {reason} | metrics: {m}")
    else:
        print(f"ACCEPTED UPLOAD: {p.name} | prob: {prob}")

print(f"\nSummary for uploads: {len(upload_imgs) - u_rejected_count} accepted, {u_rejected_count} rejected.")
