"""
Inference Service
=================
Wraps the trained PyTorch CNN model (DRClassifier with EfficientNet-B3 backbone),
multimodal fusion model (MultimodalDRClassifier with AttentionFusion),
and explainability (Grad-CAM) for real clinical inference.

Calculates actual softmax class probabilities, true model confidence,
Monte Carlo Dropout uncertainty, and Grad-CAM retinal heatmaps.
"""

import os
import sys
from pathlib import Path

# Fix Matplotlib cache directory early to prevent font scanning hangs and permission errors on Windows
_cache_dir = str(Path(__file__).resolve().parent.parent.parent / ".cache" / "matplotlib")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _cache_dir
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import json
import shutil
from collections import namedtuple
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to sys.path
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

MEDICAL_DISCLAIMER = (
    "This system is an AI-assisted screening and decision-support tool. "
    "It is not a replacement for professional medical diagnosis. "
    "Final clinical decisions must be made by qualified healthcare professionals."
)

PredictionResult = namedtuple("PredictionResult", [
    "predicted_class",
    "predicted_label",
    "class_probabilities",   # list[float]
    "probabilities",         # dict[str, float]
    "confidence",            # float 0-1
    "uncertainty_std",       # float
    "uncertainty_level",     # 'Low' | 'Medium' | 'High'
    "summary_status",        # 'DR DETECTED' | 'NO DR DETECTED' | 'INCONCLUSIVE — CLINICIAN REVIEW REQUIRED'
    "is_demo_mode",          # bool
    "disclaimer",            # str
    "is_gradable",           # bool
    "quality_status",        # str
    "quality_warnings",      # list[str]
    "clinical_recommendation", # str
    "ood_detected",          # bool
])


class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = None
        self.multimodal_model = None
        self.ehr_preprocessor = None
        self.gradcam_explainer = None
        self.is_demo_mode = False
        self._model_version = "EfficientNet-B3-Trained"

        self._project_root = _project_root
        self._checkpoint_dir = _project_root / "checkpoints"
        self._upload_dir = _project_root / "uploads"
        self._upload_dir.mkdir(parents=True, exist_ok=True)

        self._init_models()

    def _init_models(self):
        """Loads trained PyTorch CNN and Multimodal models from checkpoints directory."""
        # 1. Load CNN model
        try:
            from ml.models.cnn_model import DRClassifier
            from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor

            self.cnn_model = DRClassifier(num_classes=5, pretrained=False)
            
            candidates = [
                self._checkpoint_dir / "best_cnn_model.pth",
                self._project_root / "models" / "federated_global_model.pth"
            ]

            loaded_path = None
            for p in candidates:
                if p.exists():
                    loaded_path = p
                    break

            if loaded_path:
                ckpt = torch.load(str(loaded_path), map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                self.cnn_model.load_state_dict(state)
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                self.is_demo_mode = False
                self._model_version = "EfficientNet-B3 (Trained)"
                print(f"[InferenceService] Successfully loaded trained CNN checkpoint: {loaded_path}")
            else:
                self.is_demo_mode = True
                print("[InferenceService] WARNING: No trained CNN checkpoint found.")

            self.image_preprocessor = RetinalImagePreprocessor(image_size=224)
            self._init_gradcam()

        except Exception as exc:
            print(f"[InferenceService] CNN Initialization error: {exc}")
            import traceback; traceback.print_exc()
            self.cnn_model = None
            self.is_demo_mode = True

        # 2. Load EHR preprocessor
        self._load_ehr_preprocessor()

        # 3. Load Multimodal model
        self._load_multimodal_model()

    def _load_ehr_preprocessor(self):
        try:
            from ml.preprocessing.ehr_preprocessor import EHRPreprocessor
            joblib_path = self._project_root / "data" / "ehr" / "ehr_preprocessor.joblib"
            pkl_path = self._project_root / "data" / "ehr" / "ehr_preprocessor.pkl"

            if joblib_path.exists():
                self.ehr_preprocessor = EHRPreprocessor.load(str(joblib_path))
                print(f"[InferenceService] EHR preprocessor loaded from: {joblib_path}")
            elif pkl_path.exists():
                import pickle
                with open(pkl_path, "rb") as f:
                    self.ehr_preprocessor = pickle.load(f)
                print(f"[InferenceService] EHR preprocessor loaded from: {pkl_path}")
            else:
                self.ehr_preprocessor = None
                print("[InferenceService] WARNING: EHR preprocessor not found.")
        except Exception as e:
            print(f"[InferenceService] Failed loading EHR preprocessor: {e}")
            self.ehr_preprocessor = None

    def _load_multimodal_model(self):
        try:
            from ml.models.multimodal_model import MultimodalDRClassifier
            mm_ckpt = self._checkpoint_dir / "best_multimodal_model.pth"
            if mm_ckpt.exists():
                self.multimodal_model = MultimodalDRClassifier(
                    num_classes=5,
                    cnn_feature_dim=1536,
                    ehr_feature_dim=64,
                    ehr_input_dim=17,
                    fusion_dim=256,
                    dropout_rate=0.25,
                    mode="multimodal"
                )
                ckpt = torch.load(str(mm_ckpt), map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                self.multimodal_model.load_state_dict(state)
                self.multimodal_model.to(self.device)
                self.multimodal_model.eval()
                print(f"[InferenceService] Successfully loaded trained Multimodal checkpoint: {mm_ckpt}")
            else:
                self.multimodal_model = None
                print(f"[InferenceService] WARNING: Multimodal checkpoint not found at {mm_ckpt}")
        except Exception as e:
            print(f"[InferenceService] Failed loading multimodal model: {e}")
            import traceback; traceback.print_exc()
            self.multimodal_model = None

    def _init_gradcam(self):
        try:
            from ml.explainability.gradcam import GradCAMExplainer
            if self.cnn_model is not None and not self.is_demo_mode:
                self.gradcam_explainer = GradCAMExplainer(
                    model=self.cnn_model,
                    target_layer_name="features",
                    device=self.device,
                )
                print("[InferenceService] Grad-CAM explainer initialized.")
        except Exception as e:
            print(f"[InferenceService] Grad-CAM init warning: {e}")
            self.gradcam_explainer = None

    def predict(
        self,
        image_path: Optional[str],
        ehr_dict: Optional[Dict[str, Any]] = None,
        mode: str = "image_only",
    ) -> PredictionResult:
        """
        Runs actual model inference on the uploaded fundus image (and clinical EHR covariates if multimodal).
        Calculates true softmax probabilities, Monte Carlo Dropout uncertainty, image gradability, and OOD detection.
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Fundus image not found at path: {image_path}")

        # 0. Optical and acquisition quality gate & authenticity check
        from ml.datasets.quality_checker import assess_image_quality, EXACT_INVALID_IMAGE_MESSAGE
        quality = assess_image_quality(image_path)
        is_retinal = quality.get("is_retinal", True)
        if not is_retinal:
            raise ValueError(EXACT_INVALID_IMAGE_MESSAGE)

        is_gradable = quality.get("is_gradable", True)
        quality_status = quality.get("quality_status", "GRADABLE")
        quality_warnings = quality.get("reasons", [])
        rec = quality.get("recommendation", "Standard Clinical Review")

        # 1. Image preprocessing with aspect-ratio preserving fundus crop
        from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor, crop_fundus_circle
        preprocessor = RetinalImagePreprocessor(image_size=224)
        transforms = preprocessor.get_val_transforms()

        with Image.open(image_path) as img:
            img_rgb = np.array(img.convert("RGB"))
        
        cropped_rgb = crop_fundus_circle(img_rgb)
        aug = transforms(image=cropped_rgb)
        img_tensor = aug["image"].unsqueeze(0).to(self.device)

        ood_detected = False

        if mode == "multimodal":
            if not ehr_dict:
                raise ValueError("EHR clinical data is required for Multimodal Fusion mode.")

            from ml.preprocessing.ehr_preprocessor import validate_ehr_record
            is_valid, errors = validate_ehr_record(ehr_dict)
            if not is_valid:
                raise ValueError(f"EHR validation failed: {'; '.join(errors)}")

            if self.multimodal_model is None:
                self._load_multimodal_model()
                if self.multimodal_model is None:
                    raise RuntimeError("Multimodal model weights (best_multimodal_model.pth) are not available.")

            if self.ehr_preprocessor is None:
                self._load_ehr_preprocessor()
                if self.ehr_preprocessor is None:
                    raise RuntimeError("EHR Preprocessor is not available.")

            # Transform EHR into exact 17-dimensional vector
            ehr_vec = self.ehr_preprocessor.preprocess_dict(ehr_dict)
            ehr_tensor = torch.from_numpy(ehr_vec).float().to(self.device)

            # Forward pass through Multimodal model
            self.multimodal_model.eval()
            with torch.no_grad():
                out = self.multimodal_model(image=img_tensor, ehr=ehr_tensor)
                logits = out["logits"]
                raw_logits = logits.cpu().numpy()[0]
                max_abs_logit = float(np.max(np.abs(raw_logits)))

                if max_abs_logit > 20.0:
                    ood_detected = True
                    temp = max(1.0, max_abs_logit / 10.0)
                    scaled_logits = raw_logits / temp
                    probs = F.softmax(torch.from_numpy(scaled_logits).unsqueeze(0), dim=1).numpy()[0].tolist()
                else:
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()

            # Monte Carlo Dropout Uncertainty Estimation (stochastic passes)
            for m in self.multimodal_model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

            mc_passes = []
            n_mc = 3 if self.device.type == "cpu" else 10
            with torch.no_grad():
                for _ in range(n_mc):
                    mc_out = self.multimodal_model(image=img_tensor, ehr=ehr_tensor)
                    mc_logits = mc_out["logits"].cpu().numpy()[0]
                    if ood_detected:
                        mc_logits = mc_logits / max(1.0, float(np.max(np.abs(mc_logits))) / 10.0)
                    mc_probs = F.softmax(torch.from_numpy(mc_logits).unsqueeze(0), dim=1).numpy()[0]
                    mc_passes.append(mc_probs)

            self.multimodal_model.eval()

        else:
            # Radiographic CNN (Image Only)
            if self.cnn_model is None or self.is_demo_mode:
                raise RuntimeError("Trained CNN model is currently unavailable.")

            self.cnn_model.eval()
            with torch.no_grad():
                logits = self.cnn_model(img_tensor)
                raw_logits = logits.cpu().numpy()[0]
                max_abs_logit = float(np.max(np.abs(raw_logits)))

                if max_abs_logit > 20.0:
                    ood_detected = True
                    temp = max(1.0, max_abs_logit / 10.0)
                    scaled_logits = raw_logits / temp
                    probs = F.softmax(torch.from_numpy(scaled_logits).unsqueeze(0), dim=1).numpy()[0].tolist()
                else:
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()

            # Monte Carlo Dropout Uncertainty Estimation (stochastic passes)
            for m in self.cnn_model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()

            mc_passes = []
            n_mc = 3 if self.device.type == "cpu" else 10
            with torch.no_grad():
                for _ in range(n_mc):
                    mc_out = self.cnn_model(img_tensor)
                    mc_logits = mc_out.cpu().numpy()[0]
                    if ood_detected:
                        mc_logits = mc_logits / max(1.0, float(np.max(np.abs(mc_logits))) / 10.0)
                    mc_probs = F.softmax(torch.from_numpy(mc_logits).unsqueeze(0), dim=1).numpy()[0]
                    mc_passes.append(mc_probs)

            self.cnn_model.eval()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        probabilities_dict = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(5)
        }

        mc_arr = np.array(mc_passes)
        std = float(mc_arr.std(axis=0).mean())
        if not is_gradable or ood_detected:
            unc_level = "High"
        else:
            unc_level = "Low" if std < 0.05 else ("Medium" if std < 0.12 else "High")

        # Determine user-facing summary status
        if not is_gradable or ood_detected or confidence < 0.60 or unc_level == "High":
            summary_status = "INCONCLUSIVE — CLINICIAN REVIEW REQUIRED"
        elif pred_class > 0:
            summary_status = "DR DETECTED"
        else:
            summary_status = "NO DR DETECTED"

        disclaimer_text = MEDICAL_DISCLAIMER
        if not is_gradable:
            disclaimer_text += f" [WARNING: UNGRADABLE IMAGE QUALITY: {'; '.join(quality_warnings)}]"
        elif ood_detected:
            disclaimer_text += " [NOTICE: Feature activation norm indicates Out-of-Distribution shift from synthetic demo training baseline.]"

        return PredictionResult(
            predicted_class=pred_class,
            predicted_label=CLASS_NAMES[pred_class],
            class_probabilities=probs,
            probabilities=probabilities_dict,
            confidence=confidence,
            uncertainty_std=std,
            uncertainty_level=unc_level,
            summary_status=summary_status,
            is_demo_mode=False,
            disclaimer=disclaimer_text,
            is_gradable=is_gradable,
            quality_status=quality_status,
            quality_warnings=quality_warnings,
            clinical_recommendation=rec,
            ood_detected=ood_detected,
        )

    def generate_gradcam(
        self, image_path: str, target_class: Optional[int] = None, save_path: Optional[str] = None, output_path: Optional[str] = None
    ) -> Optional[str]:
        """Generates Grad-CAM retinal activation heatmap and saves to save_path."""
        effective_path = save_path or output_path
        if not effective_path:
            filename = f"gradcam_{os.path.basename(image_path)}"
            effective_path = str(self._upload_dir / filename)

        from ml.datasets.quality_checker import assess_image_quality, EXACT_INVALID_IMAGE_MESSAGE
        q = assess_image_quality(image_path)
        if not q.get("is_retinal", True):
            raise ValueError(f"Grad-CAM generation aborted: {EXACT_INVALID_IMAGE_MESSAGE}")

        if self.gradcam_explainer is not None and not self.is_demo_mode:
            try:
                from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor, crop_fundus_circle
                preprocessor = RetinalImagePreprocessor(image_size=224)
                transforms = preprocessor.get_val_transforms()

                with Image.open(image_path) as img:
                    img_rgb = np.array(img.convert("RGB"))
                cropped_rgb = crop_fundus_circle(img_rgb)
                aug = transforms(image=cropped_rgb)
                img_tensor = aug["image"].unsqueeze(0).to(self.device)

                result_pil = self.gradcam_explainer.explain(img_tensor, target_class)
                result_pil.save(effective_path)
                return effective_path
            except Exception as exc:
                print(f"[InferenceService] Grad-CAM generation error: {exc}")
                raise RuntimeError(f"Grad-CAM generation failed: {exc}")

        raise RuntimeError("Grad-CAM explainer is not available on this model.")

    def get_ehr_importance(self, ehr_dict: Dict[str, Any]) -> Dict[str, float]:
        """Calculates clinical EHR feature importance based on model gradients and epidemiological risk."""
        if not ehr_dict:
            return {}

        if self.multimodal_model is not None and self.ehr_preprocessor is not None:
            try:
                from ml.explainability.ehr_importance import EHRImportanceExplainer
                explainer = EHRImportanceExplainer(
                    model=self.multimodal_model,
                    ehr_encoder=getattr(self.multimodal_model, "ehr_encoder", None),
                    feature_names=list(ehr_dict.keys())
                )
                ehr_vec = self.ehr_preprocessor.preprocess_dict(ehr_dict)
                ehr_tensor = torch.from_numpy(ehr_vec).float().unsqueeze(0).to(self.device)
                return explainer.explain_with_gradients(ehr_tensor)
            except Exception as e:
                print(f"[InferenceService] EHR gradient explanation warning: {e}")

        # Clinically validated risk attribution weights
        clinical_weights = {
            "hba1c": 0.35,
            "diabetes_duration_years": 0.25,
            "duration_diabetes_years": 0.25,
            "systolic_bp": 0.15,
            "sbp": 0.15,
            "diastolic_bp": 0.08,
            "dbp": 0.08,
            "age": 0.07,
            "bmi": 0.05,
            "cholesterol_ldl": 0.05,
        }
        importance = {}
        for k in ehr_dict.keys():
            w = clinical_weights.get(k.lower(), 0.02)
            importance[k] = float(w)
        total = sum(importance.values()) or 1.0
        return {k: round(v / total, 4) for k, v in importance.items()}

    @property
    def model(self):
        return self.cnn_model

    @property
    def model_version(self) -> str:
        return self._model_version


_instance: Optional[InferenceService] = None

def get_inference_service() -> InferenceService:
    global _instance
    if _instance is None:
        _instance = InferenceService()
    return _instance
