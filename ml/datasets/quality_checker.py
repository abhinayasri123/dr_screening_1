"""
Image Quality and Retinal Authenticity Screening
================================================
Implements the strict Two-Stage validation pipeline:
Stage A: Rigorous Retinal Fundus Image Validation
- Rejects portraits, selfies, ordinary photos, animals, documents, screenshots,
  cartoons, X-rays/MRIs, and corrupted files.
- Uses both:
  1. A dedicated neural classifier (MobileNetV3-Small binary retinal detector).
  2. Multi-spectral optical, anatomical, and aperture hemodynamic analysis.

Stage B: Optical Quality Assessment
- Evaluates blur (Laplacian variance), illumination extremes, and gradability.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

BLUR_THRESHOLD_DEFAULT = 3.0        # Minimum Laplacian sharpness variance (scaled for standard inputs)
DARK_THRESHOLD_DEFAULT = 20.0       # Mean luminance below this indicates underexposure
BRIGHT_THRESHOLD_DEFAULT = 230.0    # Mean luminance above this indicates overexposure
MIN_DIMENSION_DEFAULT = 200         # Minimum pixel width/height

# User-mandated exact invalid image message
EXACT_INVALID_IMAGE_MESSAGE = (
    "INVALID IMAGE TYPE\n\n"
    "This image does not appear to be a suitable retinal fundus photograph.\n\n"
    "Please upload a clear retinal fundus image.\n\n"
    "No diabetic retinopathy prediction has been generated."
)

_cached_detector = None
_detector_failed = False

def _get_retinal_detector():
    """Lazy loader for the Stage A neural binary retinal detector."""
    global _cached_detector, _detector_failed
    if _cached_detector is not None:
        return _cached_detector
    if _detector_failed:
        return None

    try:
        # Optimize CPU thread count for fast inference
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except Exception:
            pass

        from ml.models.retinal_detector import load_retinal_detector
        project_root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = project_root / "checkpoints" / "retinal_detector.pth"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt_path.exists():
            model = load_retinal_detector(str(ckpt_path), device)
            _cached_detector = (model, device)
            return _cached_detector
        else:
            print(f"[quality_checker] Checkpoint not found at {ckpt_path}")
            return None
    except Exception as e:
        print(f"[quality_checker] Could not load retinal detector neural model: {e}")
        _detector_failed = True
        return None

def validate_retinal_fundus_image(img_rgb: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Rigorously verifies whether an image represents genuine retinal fundus photography
    rather than an unrelated natural scene, portrait, document, screenshot, animal photo,
    or unrelated medical scan (X-ray/MRI).
    
    Accepts ONLY suitable retinal fundus images across diverse clinical cameras,
    resolutions, and pathological variations (e.g. severe exudates, cotton-wool spots).
    
    Rejects:
    - Human portraits and selfies
    - Ordinary photos (landscapes, rooms, vehicles)
    - Documents and screenshots
    - Animal photographs
    - Grayscale radiographs (X-rays, CTs, MRIs)
    - Cartoons and graphic artwork
    - Corrupted or unreadable files
    
    Returns (is_retinal, message, metrics_dict).
    """
    if img_rgb is None or not isinstance(img_rgb, np.ndarray) or img_rgb.size == 0:
        return False, EXACT_INVALID_IMAGE_MESSAGE, {"error": "Image array is null or empty"}

    h, w = img_rgb.shape[:2]
    if h < 80 or w < 80:
        return False, EXACT_INVALID_IMAGE_MESSAGE, {"reason": f"Resolution below clinical minimum ({w}x{h})"}

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > 3.5:
        return False, EXACT_INVALID_IMAGE_MESSAGE, {"reason": f"Non-ophthalmic aspect ratio ({aspect:.2f})"}

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    # 1. Monochromatic / Radiographic rejection (Chest X-rays, MRIs, CTs)
    # Radiographs have almost identical R, G, B channels across all pixels (grayscale)
    diff_rg = float(np.mean(np.abs(r.astype(float) - g.astype(float))))
    diff_gb = float(np.mean(np.abs(g.astype(float) - b.astype(float))))
    if diff_rg < 3.0 and diff_gb < 3.0:
        return False, EXACT_INVALID_IMAGE_MESSAGE, {
            "reason": "Monochromatic / grayscale radiograph detected (X-ray or MRI scan without retinal chrominance)",
            "diff_rg": diff_rg,
            "diff_gb": diff_gb
        }

    # 2. Segment non-black foreground (retinal aperture or tissue)
    fg = gray > 15
    fg_count = int(np.count_nonzero(fg))
    total_pixels = h * w
    if fg_count < (0.05 * total_pixels):
        return False, EXACT_INVALID_IMAGE_MESSAGE, {"reason": "Image is predominantly dark/empty"}

    fg_r = r[fg].astype(float)
    fg_g = g[fg].astype(float)
    fg_b = b[fg].astype(float)

    mean_r = float(np.mean(fg_r))
    mean_g = float(np.mean(fg_g))
    mean_b = float(np.mean(fg_b))

    # 3. Cool / non-retinal hue analysis (cyan, pure blue, purple: Hue 85 to 145 in HSV)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_vals = hsv[:, :, 0][fg]
    s_vals = hsv[:, :, 1][fg]
    blue_cyan_pixels = (h_vals >= 85) & (h_vals <= 145) & (s_vals > 25)
    blue_cyan_pct = float(np.count_nonzero(blue_cyan_pixels) / fg_count) * 100.0

    # 4. Hemoglobin absorption / Blue channel suppression
    warm_ratio = float((mean_r + mean_g) / (2.0 * mean_b + 1e-4))

    # 5. Retinal hue concentration (red, amber, orange, yellow-green: Hue <= 55 or Hue >= 160)
    retinal_hues = ((h_vals <= 55) | (h_vals >= 160)) & (s_vals > 15)
    retinal_hue_pct = float(np.count_nonzero(retinal_hues) / fg_count) * 100.0

    # 6. Neutral white/gray pixel calculation (sclera, document paper, bright glare)
    neutral_white = (fg_r > 160) & (np.abs(fg_r - fg_b) < 18) & (np.abs(fg_r - fg_g) < 18)
    neutral_pct = float(np.count_nonzero(neutral_white) / fg_count) * 100.0

    # 7. Optical Aperture circularity check
    contours, _ = cv2.findContours((gray > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_circle = False
    circularity = 0.0
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_c)
        perimeter = cv2.arcLength(largest_c, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * area / (perimeter * perimeter))
            if circularity > 0.50 and (area / total_pixels) > 0.15:
                has_circle = True

    # 8. Neural Classifier Inference (Stage A Trained RetinalDetector)
    detector_tuple = _get_retinal_detector()
    neural_prob = None
    if detector_tuple is not None:
        model, device = detector_tuple
        try:
            pil_img = Image.fromarray(img_rgb).resize((224, 224))
            tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(pil_img).unsqueeze(0).to(device)
            
            with torch.inference_mode():
                neural_prob = model.predict_retinal_probability(tensor)
        except Exception as e:
            print(f"[quality_checker] Neural check error: {e}")
            neural_prob = None

    metrics = {
        "rb_ratio": round(mean_r / (mean_b + 1e-4), 2),
        "warm_ratio": round(warm_ratio, 2),
        "retinal_hue_pct": round(retinal_hue_pct, 1),
        "blue_cyan_pct": round(blue_cyan_pct, 1),
        "neutral_pct": round(neutral_pct, 1),
        "has_circular_aperture": has_circle,
        "neural_retinal_prob": round(neural_prob, 3) if neural_prob is not None else None
    }

    # DECISION LOGIC:
    # Retinal fundus photography has distinct physiological optical properties:
    # Hemoglobin absorption creates strong red/amber concentration and suppresses blue.
    # Non-retinal images (portraits, landscapes, documents, cartoons, X-rays) either:
    # - Are monochromatic grayscale (X-rays: diff_rg < 3.0 and diff_gb < 3.0)
    # - Have substantial cool blue/cyan (skies, blue shirts, UI buttons: blue_cyan_pct > 20%)
    # - Have high neutral white across the entire frame (documents: neutral_pct > 40%)
    # - Lack retinal warm chrominance ratio (warm_ratio < 1.10)
    
    # Check optical spectral profile:
    has_retinal_spectrum = (
        warm_ratio >= 1.10 and
        retinal_hue_pct >= 45.0 and
        blue_cyan_pct <= 20.0 and
        neutral_pct <= 45.0 and
        diff_rg >= 4.0
    )

    # A. If Neural Classifier is available:
    if neural_prob is not None:
        # Case 1: Neural model confirms retinal (>= 0.35)
        if neural_prob >= 0.35:
            # Guard against edge-case non-retinal images with unusual features
            if blue_cyan_pct > 30.0:
                return False, EXACT_INVALID_IMAGE_MESSAGE, {
                    **metrics,
                    "reason": f"High cool blue/cyan proportion ({blue_cyan_pct:.1f}% > 30%)"
                }
            if neutral_pct > 55.0:
                return False, EXACT_INVALID_IMAGE_MESSAGE, {
                    **metrics,
                    "reason": f"High neutral white proportion ({neutral_pct:.1f}% > 55%)"
                }
            return True, "Valid retinal fundus photograph verified.", metrics

        # Case 2: Neural model outputs < 0.35.
        # Check if the image is a genuine square-cropped / zoomed retinal photograph
        # from a clinical camera without circular aperture mask.
        if has_retinal_spectrum and (mean_r > mean_b * 1.15):
            # Verified as genuine retinal fundus image by retinal chrominance and hemoglobin absorption
            return True, "Valid retinal fundus photograph verified (ophthalmic chrominance analysis).", metrics

        # Otherwise rejected as non-retinal
        return False, EXACT_INVALID_IMAGE_MESSAGE, {
            **metrics,
            "reason": f"Non-retinal image verified (neural retinal prob {neural_prob:.2f} < 0.35, lacks retinal chrominance profile)"
        }

    # B. Fallback Heuristics (if neural model cannot be loaded):
    if not has_retinal_spectrum:
        reason = "Non-retinal image detected: "
        if blue_cyan_pct > 20.0:
            reason += f"cool blue/cyan pixels {blue_cyan_pct:.1f}% > 20%"
        elif warm_ratio < 1.10:
            reason += f"warm ratio {warm_ratio:.2f} < 1.10"
        elif retinal_hue_pct < 45.0:
            reason += f"retinal hue concentration {retinal_hue_pct:.1f}% < 45%"
        elif neutral_pct > 45.0:
            reason += f"neutral white pixels {neutral_pct:.1f}% > 45%"
        else:
            reason += "chrominance does not match retinal fundus"
            
        return False, EXACT_INVALID_IMAGE_MESSAGE, {
            **metrics,
            "reason": reason
        }

    return True, "Valid retinal fundus photograph verified.", metrics

def assess_image_quality(
    image_input: Any,
    blur_threshold: float = BLUR_THRESHOLD_DEFAULT,
    dark_threshold: float = DARK_THRESHOLD_DEFAULT,
    bright_threshold: float = BRIGHT_THRESHOLD_DEFAULT,
    min_dimension: int = MIN_DIMENSION_DEFAULT,
) -> Dict[str, Any]:
    """
    Evaluates retinal image authenticity and optical quality from a file path, PIL Image, or NumPy array.
    Step 1: Rigorously verifies whether the image is genuinely a retinal fundus photograph.
    Step 2: If verified as retinal, assesses clinical gradability (blur, underexposure, overexposure).
    """
    reasons = []
    
    # 1. Load image
    img_rgb = None
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return {
                "is_gradable": False,
                "is_retinal": False,
                "is_corrupt": True,
                "reasons": [EXACT_INVALID_IMAGE_MESSAGE],
                "exact_message": EXACT_INVALID_IMAGE_MESSAGE,
                "blur_score": 0.0,
                "mean_luminance": 0.0,
                "resolution": (0, 0),
                "quality_status": "CORRUPT_OR_MISSING",
                "recommendation": "File missing. Please re-upload retinal photograph."
            }
        try:
            with Image.open(image_input) as pil_img:
                pil_img.verify()
            with Image.open(image_input) as pil_img:
                img_rgb = np.array(pil_img.convert("RGB"))
        except Exception as e:
            return {
                "is_gradable": False,
                "is_retinal": False,
                "is_corrupt": True,
                "reasons": [EXACT_INVALID_IMAGE_MESSAGE],
                "exact_message": EXACT_INVALID_IMAGE_MESSAGE,
                "blur_score": 0.0,
                "mean_luminance": 0.0,
                "resolution": (0, 0),
                "quality_status": "CORRUPT",
                "recommendation": "File corrupt. Please upload an intact image."
            }
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            img_rgb = image_input
        else:
            return {
                "is_gradable": False,
                "is_retinal": False,
                "is_corrupt": True,
                "reasons": [EXACT_INVALID_IMAGE_MESSAGE],
                "exact_message": EXACT_INVALID_IMAGE_MESSAGE,
                "blur_score": 0.0,
                "mean_luminance": 0.0,
                "resolution": (0, 0),
                "quality_status": "INVALID_SHAPE",
                "recommendation": "Unsupported format."
            }
    else:
        return {
            "is_gradable": False,
            "is_retinal": False,
            "is_corrupt": True,
            "reasons": [EXACT_INVALID_IMAGE_MESSAGE],
            "exact_message": EXACT_INVALID_IMAGE_MESSAGE,
            "blur_score": 0.0,
            "mean_luminance": 0.0,
            "resolution": (0, 0),
            "quality_status": "INVALID_INPUT",
            "recommendation": "Unsupported format."
        }

    h, w = img_rgb.shape[:2]

    # Optical quality assessment (computed across all images)
    optical_reasons = []

    # Check optical resolution against clinical screening threshold
    if h < min_dimension or w < min_dimension:
        optical_reasons.append(f"Image resolution ({w}x{h}) below clinical screening minimum ({min_dimension}x{min_dimension}).")

    # Grayscale conversion for sharpness and luminance
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Blur assessment (Laplacian variance on illuminated foreground)
    fg_mask = (gray > 15).astype(np.uint8)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if np.count_nonzero(fg_mask) > 100:
        blur_score = float(laplacian[fg_mask > 0].var())
    else:
        blur_score = float(laplacian.var())

    if blur_score < blur_threshold:
        optical_reasons.append(f"Significant motion or optical blur detected (sharpness variance {blur_score:.1f} < threshold {blur_threshold:.1f}).")

    # Illumination assessment
    mean_luminance = float(np.mean(gray[gray > 15])) if np.count_nonzero(gray > 15) > 100 else float(np.mean(gray))
    if mean_luminance < dark_threshold:
        optical_reasons.append(f"Severe underexposure / dark illumination (mean brightness {mean_luminance:.1f} < threshold {dark_threshold:.1f}).")
    elif mean_luminance > bright_threshold:
        optical_reasons.append(f"Severe overexposure / sensor saturation (mean brightness {mean_luminance:.1f} > threshold {bright_threshold:.1f}).")

    # Stage A: Rigorous Retinal Fundus Authenticity Verification
    is_retinal, rejection_reason, retinal_metrics = validate_retinal_fundus_image(img_rgb)
    if not is_retinal:
        all_reasons = [EXACT_INVALID_IMAGE_MESSAGE] + optical_reasons
        return {
            "is_gradable": False,
            "is_retinal": False,
            "is_corrupt": False,
            "reasons": all_reasons,
            "exact_message": EXACT_INVALID_IMAGE_MESSAGE,
            "blur_score": round(blur_score, 2),
            "mean_luminance": round(mean_luminance, 2),
            "resolution": (w, h),
            "quality_status": "INVALID_IMAGE_TYPE",
            "recommendation": "No diabetic retinopathy prediction has been generated.",
            "retinal_metrics": retinal_metrics
        }

    # Stage B: Final verdict for verified retinal image
    is_gradable = len(optical_reasons) == 0
    quality_status = "GRADABLE" if is_gradable else "UNGRADABLE_DEFECT"

    return {
        "is_gradable": is_gradable,
        "is_retinal": True,
        "is_corrupt": False,
        "reasons": optical_reasons,
        "blur_score": round(blur_score, 2),
        "mean_luminance": round(mean_luminance, 2),
        "resolution": (w, h),
        "quality_status": quality_status,
        "recommendation": "Accept for CNN inference" if is_gradable else "Flag for repeat acquisition or manual ophthalmologist review",
        "retinal_metrics": retinal_metrics
    }
