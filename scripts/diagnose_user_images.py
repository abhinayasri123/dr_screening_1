import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from ml.models.cnn_model import DRClassifier
from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def run():
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS OF MODEL CHECKPOINT AND USER UPLOADED IMAGES")
    print("=" * 70)

    device = torch.device("cpu")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_cnn_model.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'direct_tensor'}")
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Trained epoch: {ckpt.get('epoch')}")
        print(f"Stored metrics: {ckpt.get('metrics')}")
    
    model = DRClassifier(num_classes=5, pretrained=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    img1_path = PROJECT_ROOT / "uploads" / "bc4aed6c-8b96-439b-9789-58118d9295a2.jpg"
    img2_path = PROJECT_ROOT / "uploads" / "7222f5f9-0087-4793-bbba-11da66065c7b.jpg"

    # Also test labeled test images from each class
    test_samples = [
        ("User Image 1 (OIP.jpg)", img1_path, "Unknown"),
        ("User Image 2 (d4d31ced...)", img2_path, "Unknown"),
    ]
    for c in range(5):
        c_folder = PROJECT_ROOT / "data" / "retinal" / "test" / str(c)
        c_imgs = list(c_folder.glob("*.png"))
        if c_imgs:
            test_samples.append((f"Test Class {c} Sample", c_imgs[0], CLASS_NAMES[c]))

    for size in [224, 300]:
        print(f"\n" + "-" * 50)
        print(f"EVALUATING WITH PREPROCESSOR IMAGE_SIZE = {size}")
        print("-" * 50)
        prep = RetinalImagePreprocessor(image_size=size)
        tf = prep.get_val_transforms()

        for name, path, true_label in test_samples:
            if not os.path.exists(path):
                print(f"Missing {path}")
                continue
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
            
            t = tf(image=rgb)["image"].unsqueeze(0)
            with torch.no_grad():
                logits = model(t)
                probs = F.softmax(logits, dim=1).numpy()[0]
            
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            print(f"\n{name} [Ground Truth: {true_label}]:")
            print(f"  Shape: {rgb.shape} -> Preprocessed Tensor: {tuple(t.shape)}")
            print(f"  Raw Logits: {np.round(logits.numpy()[0], 4)}")
            print(f"  Probabilities: {np.round(probs, 4)}")
            print(f"  Prediction: Class {pred_idx} ({CLASS_NAMES[pred_idx]}), Confidence = {conf*100:.2f}%")

if __name__ == "__main__":
    run()
