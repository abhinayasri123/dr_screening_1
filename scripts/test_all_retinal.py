import sys
from pathlib import Path
PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.quality_checker import validate_retinal_fundus_image, assess_image_quality
import numpy as np
from PIL import Image

data_dir = PROJECT_ROOT / "data" / "retinal"
all_retinal_files = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))
print(f"Total retinal files found in data/: {len(all_retinal_files)}", flush=True)

rejected = []
accepted = []

for i, p in enumerate(all_retinal_files):
    try:
        with Image.open(p) as img:
            arr = np.array(img.convert("RGB"))
        is_ret, msg, m = validate_retinal_fundus_image(arr)
        if is_ret:
            accepted.append(p.name)
        else:
            rejected.append((p.name, m))
    except Exception as e:
        rejected.append((p.name, {"error": str(e)}))
    if (i + 1) % 10 == 0 or (i + 1) == len(all_retinal_files):
        print(f"Processed {i+1}/{len(all_retinal_files)}: {len(accepted)} accepted, {len(rejected)} rejected.", flush=True)

print(f"Results for dataset files: {len(accepted)} accepted, {len(rejected)} rejected.", flush=True)
if rejected:
    print("First 10 rejections in dataset:", flush=True)
    for name, r in rejected[:10]:
        print(f"  {name}: {r}", flush=True)
