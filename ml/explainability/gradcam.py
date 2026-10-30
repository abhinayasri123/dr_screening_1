import os
from pathlib import Path
_cache_dir = str(Path(__file__).resolve().parent.parent.parent / ".cache" / "matplotlib")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _cache_dir

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

def get_target_layer(model, layer_name):
    """Navigates model to find layer."""
    if hasattr(model, 'backbone'):
        # For efficientnet
        if layer_name == 'features':
            return [model.backbone.features[-1]]
    return [list(model.children())[-1]]

class GradCAMExplainer:
    def __init__(self, model, target_layer_name='features', device='cuda'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        target_layers = get_target_layer(self.model, target_layer_name)
        self.cam = GradCAM(model=self.model, target_layers=target_layers)
        
    def get_overlay(self, image_np, cam_mask, alpha=0.5):
        """Returns colored overlay."""
        # Assume image_np is RGB in [0, 1]
        visualization = show_cam_on_image(image_np, cam_mask, use_rgb=True, image_weight=alpha)
        return visualization

    def explain(self, image_tensor, target_class=None):
        """Returns PIL Image of heatmap overlay."""
        if len(image_tensor.shape) == 3:
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
        else:
            input_tensor = image_tensor.to(self.device)
            
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Denormalize image for visualization
        img_np = input_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        vis = self.get_overlay(img_np, grayscale_cam)
        return Image.fromarray(vis)
        
    def explain_batch(self, image_tensors, target_classes=None):
        """Returns list of PIL Images."""
        results = []
        for i in range(image_tensors.shape[0]):
            tc = target_classes[i] if target_classes is not None else None
            results.append(self.explain(image_tensors[i], tc))
        return results
        
    def save_explanation(self, image_pil, save_path):
        image_pil.save(save_path)
