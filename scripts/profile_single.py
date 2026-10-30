import sys, time
from pathlib import Path
PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np

# Take a file from uploads
upload_files = list((PROJECT_ROOT / "uploads").glob("*.png"))
fundus_files = [f for f in upload_files if not f.name.startswith("gradcam_")]
print("Upload files found:", len(fundus_files))
if fundus_files:
    test_img = fundus_files[0]
else:
    test_img = list((PROJECT_ROOT / "data" / "retinal" / "test").rglob("*.png"))[0]

print(f"Testing image: {test_img} (Size: {test_img.stat().st_size} bytes)")

t0 = time.time()
with Image.open(test_img) as img:
    arr = np.array(img.convert("RGB"))
print(f"Loaded image array shape: {arr.shape} in {time.time()-t0:.3f}s")

# Test quality checker
from ml.datasets.quality_checker import validate_retinal_fundus_image, assess_image_quality
t1 = time.time()
is_ret, msg, metrics = validate_retinal_fundus_image(arr)
print(f"validate_retinal_fundus_image returned in {time.time()-t1:.3f}s:")
print(f"  is_retinal={is_ret}")
print(f"  message={msg}")
print(f"  metrics={metrics}")
