import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from ml.models.cnn_model import DRClassifier
from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor

def run():
    model = DRClassifier(num_classes=5, pretrained=False)
    ckpt = torch.load("checkpoints/best_cnn_model.pth", map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    prep = RetinalImagePreprocessor(image_size=224)
    tf = prep.get_val_transforms()

    img2_path = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"
    test_cls0 = PROJECT_ROOT / "data" / "retinal" / "test" / "0" / "demo_cls0_test_000.png"
    test_cls2 = PROJECT_ROOT / "data" / "retinal" / "test" / "2" / "demo_cls2_test_000.png"

    for name, p in [("User Image 2", img2_path), ("Test Class 0", test_cls0), ("Test Class 2", test_cls2)]:
        im = Image.open(p)
        rgb = np.array(im.convert("RGB"))
        t = tf(image=rgb)["image"].unsqueeze(0)
        
        with torch.no_grad():
            feat = model.backbone.features(t)
            pooled = model.backbone.avgpool(feat)
            flat = torch.flatten(pooled, 1)
            
            # Classifier head
            c1 = model.backbone.classifier[1](flat)
            c2 = F.relu(c1)
            logits = model.backbone.classifier[4](c2)
            probs = F.softmax(logits, dim=1).numpy()[0]

        print(f"\n=== {name} ===")
        print(f"  Feature map shape: {feat.shape}, min: {feat.min():.2f}, max: {feat.max():.2f}, mean: {feat.mean():.2f}")
        print(f"  Flattened (1536-d) min: {flat.min():.2f}, max: {flat.max():.2f}, mean: {flat.mean():.2f}")
        print(f"  Linear 1 (512-d)   min: {c1.min():.2f}, max: {c1.max():.2f}, mean: {c1.mean():.2f}")
        print(f"  Final Logits (5-d): {logits.numpy()[0].round(2)}")
        print(f"  Softmax Probs:      {probs.round(4)}")

if __name__ == "__main__":
    run()
