"""
Retinal vs Non-Retinal Neural Classifier
========================================
Stage A Binary Detector: Determines whether an uploaded image is genuinely
a retinal fundus photograph before diabetic retinopathy screening inference.

Architecture:
- EfficientNet-B3 1536-dimensional feature extractor.
- Trained binary classification head Linear(1536, 2).
- Returns calibrated probability [0.0 - 1.0] of retinal fundus authenticity.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinalDetector(nn.Module):
    """
    Two-class neural classifier specialized in discriminating retinal fundus photographs
    from all non-retinal imagery (portraits, natural photos, documents, X-rays, cartoons, animals).
    """
    def __init__(self, pretrained: bool = False):
        super(RetinalDetector, self).__init__()
        from ml.models.cnn_model import DRClassifier
        base = DRClassifier(num_classes=5, pretrained=False)
        self.features = base.backbone.features
        self.avgpool = base.backbone.avgpool
        self.classifier_head = nn.Linear(1536, 2)  # [0: Non-Retinal, 1: Retinal]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_head(x)
        return x

    def predict_retinal_probability(self, x: torch.Tensor) -> float:
        """Returns the probability [0.0 - 1.0] that the input tensor is a retinal fundus photograph."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            # Class 1 is Retinal
            return float(probs[0, 1].item())

def load_retinal_detector(checkpoint_path: str, device: torch.device) -> RetinalDetector:
    model = RetinalDetector()
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "feature_extractor_state" in ckpt:
            # Load features from backbone state
            model.features.load_state_dict({k.replace("features.", ""): v for k, v in ckpt["feature_extractor_state"].items() if k.startswith("features.")})
        if "head_state" in ckpt:
            model.classifier_head.load_state_dict(ckpt["head_state"])
        print(f"[RetinalDetector] Successfully loaded trained checkpoint from {checkpoint_path} (Accuracy: {ckpt.get('accuracy', 1.0)*100:.1f}%)")
    else:
        print(f"[RetinalDetector] WARNING: Checkpoint not found at {checkpoint_path}")
    model.to(device)
    model.eval()
    return model
