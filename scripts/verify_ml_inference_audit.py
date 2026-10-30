import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backend.services.inference_service import get_inference_service
from ml.datasets.quality_checker import assess_image_quality

def run_audit():
    print("=" * 70)
    print("VERIFYING ML INFERENCE AUDIT & STABILITY ENHANCEMENTS")
    print("=" * 70)

    svc = get_inference_service()

    img1 = PROJECT_ROOT / "uploads" / "bc4aed6c-8b96-439b-9789-58118d9295a2.jpg"
    img2 = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"

    print("\n--- TEST 1: User Image 1 (Low-res thumbnail 220x220) ---")
    q1 = assess_image_quality(str(img1))
    print(f"Quality Check: is_gradable={q1['is_gradable']}, status={q1['quality_status']}")
    print(f"Reasons: {q1['reasons']}")
    
    pred1 = svc.predict(str(img1), mode="image_only")
    print(f"Prediction: {pred1.predicted_label} (Class {pred1.predicted_class})")
    print(f"Confidence: {pred1.confidence*100:.2f}% | Uncertainty: {pred1.uncertainty_level}")
    print(f"Is Gradable: {pred1.is_gradable} | Quality Status: {pred1.quality_status}")
    print(f"Quality Warnings: {pred1.quality_warnings}")
    print(f"Disclaimer: {pred1.disclaimer[:90]}...")
    assert pred1.is_gradable is False, "Expected Image 1 to be marked ungradable!"
    assert pred1.uncertainty_level == "High", "Expected Image 1 to have High uncertainty!"

    print("\n--- TEST 2: User Image 2 (High-res 1024x735 Google image) ---")
    q2 = assess_image_quality(str(img2))
    print(f"Quality Check: is_gradable={q2['is_gradable']}, status={q2['quality_status']}")
    print(f"Resolution: {q2['resolution']}")
    
    pred2 = svc.predict(str(img2), mode="image_only")
    print(f"Prediction: {pred2.predicted_label} (Class {pred2.predicted_class})")
    print(f"Confidence: {pred2.confidence*100:.2f}% | Uncertainty: {pred2.uncertainty_level}")
    print(f"OOD Detected: {pred2.ood_detected}")
    print(f"Probabilities: { {k: round(v, 4) for k, v in pred2.probabilities.items()} }")
    assert pred2.ood_detected is True, "Expected Image 2 to trigger OOD detection!"
    assert pred2.confidence < 1.0, "Expected temperature-scaled confidence < 100.0%!"

    print("\n--- TEST 3: Grad-CAM Explanations ---")
    gc1 = svc.generate_gradcam(str(img1), target_class=pred1.predicted_class)
    gc2 = svc.generate_gradcam(str(img2), target_class=pred2.predicted_class)
    print(f"Grad-CAM 1: {os.path.exists(gc1)} ({gc1})")
    print(f"Grad-CAM 2: {os.path.exists(gc2)} ({gc2})")
    assert os.path.exists(gc1) and os.path.getsize(gc1) > 0
    assert os.path.exists(gc2) and os.path.getsize(gc2) > 0

    print("\n" + "=" * 70)
    print("ALL AUDIT VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_audit()
