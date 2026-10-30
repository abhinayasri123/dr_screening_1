import os
import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from backend.services.inference_service import get_inference_service
from ml.models.cnn_model import DRClassifier
from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor

def run():
    inf_service = get_inference_service()
    
    img1_path = str(PROJECT_ROOT / "uploads" / "bc4aed6c-8b96-439b-9789-58118d9295a2.jpg")
    img2_path = str(PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg")

    for idx, path in enumerate([img1_path, img2_path], 1):
        print(f"\n==========================================")
        print(f"RUNNING INFERENCE SERVICE ON IMAGE {idx}: {os.path.basename(path)}")
        print(f"==========================================")
        res = inf_service.predict(path, mode="image_only")
        print(f"Predicted Class: {res.predicted_class} ({res.predicted_label})")
        print(f"Confidence: {res.confidence * 100:.2f}%")
        print(f"Probabilities: {res.probabilities}")
        print(f"Class Probabilities: {res.class_probabilities}")
        print(f"Uncertainty Level: {res.uncertainty_level} (std: {res.uncertainty_std:.4f})")
        print(f"Is Demo Mode: {res.is_demo_mode}")

if __name__ == "__main__":
    run()
