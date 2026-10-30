import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np
from ml.datasets.quality_checker import validate_retinal_fundus_image, assess_image_quality

print("Imported quality_checker successfully.", flush=True)

# Test 5 genuine retinal images
retinal_files = list((PROJECT_ROOT / "data" / "retinal").rglob("*.png"))[:5]
print(f"Testing {len(retinal_files)} genuine retinal images:", flush=True)

for rf in retinal_files:
    t0 = time.time()
    with Image.open(rf) as im:
        arr = np.array(im.convert("RGB"))
    is_ret, msg, metrics = validate_retinal_fundus_image(arr)
    dt = time.time() - t0
    status = "ACCEPT" if is_ret else "REJECT"
    print(f"  [{status}] {rf.name} ({dt:.2f}s) -> Retinal={is_ret}, Metrics: {metrics}", flush=True)

# Test a portrait or non-retinal image
bench_dir = PROJECT_ROOT / "test_benchmark_dataset"
non_retinal = list(bench_dir.glob("*.png"))[:3]
print(f"\nTesting {len(non_retinal)} benchmark non-retinal images:", flush=True)
for nf in non_retinal:
    t0 = time.time()
    with Image.open(nf) as im:
        arr = np.array(im.convert("RGB"))
    is_ret, msg, metrics = validate_retinal_fundus_image(arr)
    dt = time.time() - t0
    status = "ACCEPT" if is_ret else "REJECT"
    print(f"  [{status}] {nf.name} ({dt:.2f}s) -> Retinal={is_ret}, Metrics: {metrics}", flush=True)
