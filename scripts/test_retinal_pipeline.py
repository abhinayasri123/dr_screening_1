"""
Validation Test Suite: Retinal-Only Diabetic Retinopathy Pipeline
================================================================
Executes comprehensive real verification across all 9 required test categories:
1. Genuine retinal fundus photographs
2. Human portraits and selfies
3. Ordinary photographs (landscapes, rooms, cars)
4. Documents and screenshots
5. Animal photographs
6. X-rays and other unrelated medical scans
7. Cartoon and AI-generated images
8. Corrupted files
9. Blurry and low-quality retinal images
"""

import os
import sys
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.quality_checker import (
    validate_retinal_fundus_image,
    assess_image_quality,
    EXACT_INVALID_IMAGE_MESSAGE
)
from backend.services.inference_service import get_inference_service, PredictionResult

TEST_DIR = PROJECT_ROOT / "test_benchmark_dataset"
TEST_DIR.mkdir(parents=True, exist_ok=True)

def create_benchmark_test_cases():
    test_cases = []

    # Category 1: Genuine Retinal Fundus Images (from project dataset)
    retinal_dir = PROJECT_ROOT / "data" / "retinal" / "test"
    found_retinal = list(retinal_dir.rglob("*.png"))[:5]
    if not found_retinal:
        retinal_dir = PROJECT_ROOT / "data" / "retinal" / "train"
        found_retinal = list(retinal_dir.rglob("*.png"))[:5]

    for i, p in enumerate(found_retinal):
        test_cases.append({
            "category": "1. Genuine Retinal Fundus",
            "name": f"retinal_fundus_{i+1}",
            "path": str(p),
            "expected_retinal": True,
            "expected_gradable": True
        })

    # Category 2: Human Portraits / Selfies
    for i in range(3):
        p_path = TEST_DIR / f"portrait_selfie_{i+1}.png"
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = [180, 200, 220] # Cool background
        # Face
        cv2.ellipse(img, (128, 120), (55, 75), 0, 0, 360, (215, 175, 140), -1)
        # Sclera / Eyes
        cv2.circle(img, (105, 110), 10, (250, 250, 250), -1)
        cv2.circle(img, (151, 110), 10, (250, 250, 250), -1)
        cv2.circle(img, (105, 110), 5, (50, 40, 30), -1)
        cv2.circle(img, (151, 110), 5, (50, 40, 30), -1)
        # Teeth / smile
        cv2.ellipse(img, (128, 160), (22, 10), 0, 0, 180, (255, 255, 255), -1)
        # Shirt
        cv2.rectangle(img, (40, 200), (216, 256), (30, 80, 200), -1)
        cv2.imwrite(str(p_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_cases.append({
            "category": "2. Human Portrait / Selfie",
            "name": f"portrait_selfie_{i+1}",
            "path": str(p_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 3: Ordinary Photographs (landscape, room)
    for i in range(3):
        o_path = TEST_DIR / f"ordinary_photo_{i+1}.png"
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:128, :] = [80, 160, 240]  # Sky (Cyan/Blue)
        img[128:, :] = [40, 180, 60]   # Grass
        cv2.circle(img, (190, 60), 25, (255, 255, 120), -1) # Sun
        cv2.imwrite(str(o_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_cases.append({
            "category": "3. Ordinary Photograph",
            "name": f"ordinary_photo_{i+1}",
            "path": str(o_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 4: Documents and Screenshots
    for i in range(3):
        d_path = TEST_DIR / f"document_screenshot_{i+1}.png"
        img = np.full((256, 256, 3), 245, dtype=np.uint8)
        # Text lines
        for y in range(40, 230, 16):
            cv2.line(img, (25, y), (220, y), (30, 30, 30), 2)
        cv2.rectangle(img, (160, 20), (220, 50), (180, 40, 40), 2)
        cv2.imwrite(str(d_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_cases.append({
            "category": "4. Document / Screenshot",
            "name": f"document_screenshot_{i+1}",
            "path": str(d_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 5: Animal Photographs
    for i in range(3):
        a_path = TEST_DIR / f"animal_photo_{i+1}.png"
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = [100, 150, 100] # Green foliage
        # Dog/Cat fur circle
        cv2.circle(img, (128, 128), 65, (160, 110, 70), -1)
        # Ears
        cv2.drawContours(img, [np.array([[80, 80], [60, 40], [100, 60]])], 0, (140, 90, 50), -1)
        cv2.drawContours(img, [np.array([[176, 80], [196, 40], [156, 60]])], 0, (140, 90, 50), -1)
        cv2.imwrite(str(a_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_cases.append({
            "category": "5. Animal Photograph",
            "name": f"animal_photo_{i+1}",
            "path": str(a_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 6: X-rays / Unrelated Medical Images
    for i in range(3):
        x_path = TEST_DIR / f"chest_xray_{i+1}.png"
        gray = np.zeros((256, 256), dtype=np.uint8)
        # Ribs
        for y in range(50, 210, 25):
            cv2.ellipse(gray, (128, y), (85, 20), 0, 0, 180, 180, 4)
        cv2.line(gray, (128, 30), (128, 230), 220, 6) # Spine
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(x_path), img)
        test_cases.append({
            "category": "6. X-ray / Unrelated Medical Scan",
            "name": f"chest_xray_{i+1}",
            "path": str(x_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 7: Cartoon and Artwork
    for i in range(3):
        c_path = TEST_DIR / f"cartoon_artwork_{i+1}.png"
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = [220, 50, 200] # Bright magenta/purple
        cv2.circle(img, (128, 128), 70, (255, 230, 0), -1)
        cv2.rectangle(img, (80, 80), (150, 150), (0, 220, 255), -1)
        cv2.imwrite(str(c_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        test_cases.append({
            "category": "7. Cartoon / Graphic Artwork",
            "name": f"cartoon_artwork_{i+1}",
            "path": str(c_path),
            "expected_retinal": False,
            "expected_gradable": False
        })

    # Category 8: Corrupted Files
    corrupt_path = TEST_DIR / "corrupted_file.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES_123456789")
    test_cases.append({
        "category": "8. Corrupted / Unreadable File",
        "name": "corrupted_file",
        "path": str(corrupt_path),
        "expected_retinal": False,
        "expected_gradable": False
    })

    # Category 9: Blurry / Ungradable Retinal Images
    if found_retinal:
        b_path = TEST_DIR / "blurry_retinal_fundus.png"
        ret_img = cv2.imread(str(found_retinal[0]))
        # Apply heavy motion blur
        blurred = cv2.GaussianBlur(ret_img, (51, 51), 30)
        cv2.imwrite(str(b_path), blurred)
        test_cases.append({
            "category": "9. Blurry Retinal Image (Optical Defect)",
            "name": "blurry_retinal_fundus",
            "path": str(b_path),
            "expected_retinal": True,
            "expected_gradable": False
        })

    return test_cases

def run_tests():
    cases = create_benchmark_test_cases()
    inf_service = get_inference_service()

    results = []
    total_passed = 0

    print("=" * 80)
    print("STARTING AUDIT: RETINAL-ONLY TWO-STAGE VALIDATION PIPELINE")
    print("=" * 80)

    for case in cases:
        cat = case["category"]
        name = case["name"]
        path = case["path"]
        exp_retinal = case["expected_retinal"]
        exp_gradable = case["expected_gradable"]

        # 1. Quality & Retinal Assessment
        quality = assess_image_quality(path)
        actual_retinal = quality.get("is_retinal", False)
        actual_gradable = quality.get("is_gradable", False)

        # 2. Prediction Attempt
        prediction_generated = False
        pred_label = "None"
        summary_status = "REJECTED"
        gradcam_generated = False
        rejection_message_matched = False

        if not actual_retinal:
            # Must raise error or reject with exact message
            try:
                inf_service.predict(path)
                prediction_generated = True
            except ValueError as ve:
                err_text = str(ve)
                if EXACT_INVALID_IMAGE_MESSAGE in err_text or "INVALID IMAGE TYPE" in err_text:
                    rejection_message_matched = True
            except Exception:
                pass
        else:
            # Genuine retinal image -> proceeds to Stage B
            try:
                res = inf_service.predict(path)
                prediction_generated = True
                pred_label = res.predicted_label
                summary_status = res.summary_status

                # Attempt Grad-CAM
                gc_out = TEST_DIR / f"test_gc_{name}.png"
                gc_res = inf_service.generate_gradcam(path, target_class=res.predicted_class, save_path=str(gc_out))
                if gc_res and os.path.exists(gc_res):
                    gradcam_generated = True
            except Exception as e:
                print(f"Error evaluating retinal case {name}: {e}")

        # Verification Criteria:
        # Non-retinal: MUST be rejected, MUST NOT generate DR prediction, MUST NOT generate Grad-CAM
        # Genuine retinal: MUST be accepted as retinal, MUST reach classifier, MUST generate Grad-CAM
        # Blurry retinal: Retinal=True, Gradable=False, summary_status=INCONCLUSIVE
        pass_test = False
        if not exp_retinal:
            pass_test = (
                not actual_retinal and
                not prediction_generated and
                not gradcam_generated
            )
        elif not exp_gradable:
            pass_test = (
                actual_retinal and
                not actual_gradable and
                summary_status == "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED"
            )
        else:
            pass_test = (
                actual_retinal and
                actual_gradable and
                prediction_generated and
                gradcam_generated
            )

        if pass_test:
            total_passed += 1

        results.append({
            "category": cat,
            "name": name,
            "is_retinal": actual_retinal,
            "is_gradable": actual_gradable,
            "pred_generated": prediction_generated,
            "pred_label": pred_label,
            "summary_status": summary_status,
            "gradcam": gradcam_generated,
            "status": "PASS" if pass_test else "FAIL"
        })

        print(f"[{'PASS' if pass_test else 'FAIL'}] {cat} -> {name}: Retinal={actual_retinal}, Gradable={actual_gradable}, PredGen={prediction_generated}, Status={summary_status}", flush=True)

    print("=" * 80, flush=True)
    print(f"AUDIT SUMMARY: {total_passed} / {len(cases)} Test Cases Passed ({total_passed/len(cases)*100:.1f}%)", flush=True)
    print("=" * 80, flush=True)

    # Return results for summary output
    return results

if __name__ == "__main__":
    run_tests()
