import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np
import cv2
from ml.datasets.quality_checker import validate_retinal_fundus_image

# Load a genuine retinal fundus image
rf_path = list((PROJECT_ROOT / "data" / "retinal").rglob("*.png"))[0]
with Image.open(rf_path) as im:
    orig = np.array(im.convert("RGB"))

print(f"Base image: {rf_path.name}, shape={orig.shape}", flush=True)
is_ret, msg, m = validate_retinal_fundus_image(orig)
print(f"1. Raw image: Retinal={is_ret}, reason={m.get('reason')}", flush=True)

# Variation A: Square crop of the center (no black circular margin)
h, w = orig.shape[:2]
center_crop = orig[h//4:3*h//4, w//4:3*w//4]
is_ret_crop, msg_crop, m_crop = validate_retinal_fundus_image(center_crop)
print(f"2. Center square crop (no circle): Retinal={is_ret_crop}, reason={m_crop.get('reason')}", flush=True)

# Variation B: Image with slight cool timestamp / watermark (e.g. 3.5% cyan/blue pixels)
ts_img = orig.copy()
# Add a timestamp box in cyan (top-left)
cv2.rectangle(ts_img, (5, 5), (60, 25), (0, 200, 255), -1)
is_ret_ts, msg_ts, m_ts = validate_retinal_fundus_image(ts_img)
print(f"3. Timestamp/overlay (cyan): Retinal={is_ret_ts}, reason={m_ts.get('reason')}", flush=True)

# Variation C: Image with hard exudates / cotton wool spots / flash glare
ex_img = orig.copy()
for cx, cy in [(100, 100), (120, 110), (110, 130), (90, 120), (130, 95)]:
    cv2.circle(ex_img, (cx, cy), 15, (240, 240, 240), -1)
is_ret_ex, msg_ex, m_ex = validate_retinal_fundus_image(ex_img)
print(f"4. Severe DR exudates / cotton wool spots: Retinal={is_ret_ex}, reason={m_ex.get('reason')}", flush=True)

# Variation D: Image with slightly different color temperature / camera sensor
pale_img = cv2.convertScaleAbs(orig, alpha=1.2, beta=30)
is_ret_pale, msg_pale, m_pale = validate_retinal_fundus_image(pale_img)
print(f"5. Pale / high flash illumination: Retinal={is_ret_pale}, reason={m_pale.get('reason')}", flush=True)
