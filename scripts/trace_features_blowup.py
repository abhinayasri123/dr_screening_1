import sys
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image

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

    p2 = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"
    with Image.open(p2) as im:
        rgb = np.array(im.convert("RGB"))
    t = tf(image=rgb)["image"].unsqueeze(0)

    print(f"Input tensor stats: min={t.min():.2f}, max={t.max():.2f}, mean={t.mean():.2f}, std={t.std():.2f}")

    x = t
    for idx, layer in enumerate(model.backbone.features):
        x = layer(x)
        print(f"Layer {idx} ({layer.__class__.__name__}): shape={tuple(x.shape)}, min={x.min():.2f}, max={x.max():.2f}, mean={x.mean():.2f}")

if __name__ == "__main__":
    run()
