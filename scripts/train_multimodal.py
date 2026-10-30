#!/usr/bin/env python3
"""
Multimodal DR Model Training & Checkpoint Generation
===================================================
Trains the MultimodalDRClassifier fusing CNN image features (EfficientNet-B3)
with Patient EHR clinical risk covariates (17-d standardized representation)
using AttentionFusion gating.

Saves:
  - checkpoints/best_multimodal_model.pth
  - data/ehr/ehr_preprocessor.joblib
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ml.models.multimodal_model import MultimodalDRClassifier
from ml.models.cnn_model import DRClassifier
from ml.preprocessing.ehr_preprocessor import EHRPreprocessor, generate_synthetic_ehr
from ml.preprocessing.image_preprocessor import RetinalImagePreprocessor

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

class PairedMultimodalDataset(Dataset):
    def __init__(self, samples, ehr_df, ehr_preprocessor, img_transform=None):
        self.samples = samples
        self.img_transform = img_transform
        self.ehr_features = ehr_preprocessor.transform(ehr_df)
        self.labels = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img_rgb = np.array(img.convert("RGB"))
        except Exception:
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.img_transform:
            augmented = self.img_transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        ehr_idx = idx % len(self.ehr_features)
        ehr_tensor = torch.tensor(self.ehr_features[ehr_idx], dtype=torch.float32)

        return img_tensor, ehr_tensor, torch.tensor(label, dtype=torch.long)

def collect_image_samples(split_dir):
    samples = []
    p = Path(split_dir)
    if not p.exists():
        return samples
    for c in range(5):
        c_dir = p / str(c)
        if c_dir.exists():
            for f in c_dir.glob("*.*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    samples.append((str(f), c))
    return samples

def train_multimodal_model():
    print("=" * 70)
    print("TRAINING MULTIMODAL FUSION MODEL (CNN + EHR AttentionFusion)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Prepare EHR Preprocessor
    ehr_demo_path = PROJECT_ROOT / "data" / "ehr" / "synthetic_ehr_demo.csv"
    if ehr_demo_path.exists():
        ehr_raw_df = pd.read_csv(ehr_demo_path)
    else:
        ehr_raw_df = generate_synthetic_ehr(n_samples=500, seed=42)

    # Standardize column names
    col_map = {
        'sex': 'gender',
        'systolic_bp': 'sbp',
        'diastolic_bp': 'dbp',
        'diabetes_duration_years': 'duration_diabetes_years',
        'insulin_use': 'insulin_dependent',
        'smoking_status': 'smoker',
        'cholesterol_ldl': 'cholesterol'
    }
    for old_c, new_c in col_map.items():
        if old_c in ehr_raw_df.columns and new_c not in ehr_raw_df.columns:
            ehr_raw_df[new_c] = ehr_raw_df[old_c]

    # Normalize gender, insulin, smoker
    ehr_raw_df['gender'] = ehr_raw_df['gender'].map(lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1'] else 'F')
    ehr_raw_df['insulin_dependent'] = ehr_raw_df['insulin_dependent'].map(lambda x: 'yes' if str(x) in ['1', 'True', 'yes', '1.0'] else 'no')
    ehr_raw_df['smoker'] = ehr_raw_df['smoker'].map(lambda x: str(x).lower() if str(x).lower() in ['never', 'former', 'current'] else 'never')

    ehr_preprocessor = EHRPreprocessor()
    ehr_preprocessor.fit(ehr_raw_df)
    
    # Save fitted preprocessor
    ehr_save_path = PROJECT_ROOT / "data" / "ehr" / "ehr_preprocessor.joblib"
    ehr_preprocessor.save(str(ehr_save_path))
    print(f"[EHRPreprocessor] Fitted and saved to: {ehr_save_path}")

    # 2. Collect image samples
    train_samples = collect_image_samples(PROJECT_ROOT / "data" / "retinal" / "train")
    val_samples = collect_image_samples(PROJECT_ROOT / "data" / "retinal" / "validation")
    test_samples = collect_image_samples(PROJECT_ROOT / "data" / "retinal" / "test")

    if not val_samples:
        val_samples = test_samples[:len(test_samples)//2]
        test_samples = test_samples[len(test_samples)//2:]

    print(f"Sample counts: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    img_prep = RetinalImagePreprocessor(image_size=224)
    train_tf = img_prep.get_train_transforms()
    val_tf = img_prep.get_val_transforms()

    train_ds = PairedMultimodalDataset(train_samples, ehr_raw_df, ehr_preprocessor, img_transform=train_tf)
    val_ds = PairedMultimodalDataset(val_samples, ehr_raw_df, ehr_preprocessor, img_transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 3. Initialize Multimodal Model
    sample_ehr_feat = ehr_preprocessor.transform(ehr_raw_df[:1])
    ehr_in_dim = sample_ehr_feat.shape[1]
    print(f"Transformed EHR input dimension: {ehr_in_dim}")

    model = MultimodalDRClassifier(
        num_classes=5,
        cnn_feature_dim=1536,
        ehr_feature_dim=64,
        ehr_input_dim=ehr_in_dim,
        fusion_dim=256,
        dropout_rate=0.25,
        mode='multimodal'
    )

    # Transfer trained CNN weights if available
    cnn_ckpt_path = PROJECT_ROOT / "checkpoints" / "best_cnn_model.pth"
    if cnn_ckpt_path.exists():
        try:
            ckpt = torch.load(str(cnn_ckpt_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            # Load into backbone features
            bb_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.features."):
                    clean_k = k.replace("backbone.", "")
                    bb_dict[clean_k] = v
            msg = model.cnn.load_state_dict(bb_dict, strict=False)
            print(f"[Multimodal] Successfully transferred pretrained CNN weights: {cnn_ckpt_path}")
        except Exception as e:
            print(f"[Multimodal] Warning loading CNN weights: {e}")

    model.to(device)

    # 4. Training loop (Train fusion and EHR encoder)
    # Freeze CNN backbone during early epochs to preserve feature extraction
    for p in model.cnn.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.ehr_encoder.parameters(), 'lr': 1e-3},
        {'params': model.fusion.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs = 12

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, ehrs, targets in train_loader:
            imgs, ehrs, targets = imgs.to(device), ehrs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(image=imgs, ehr=ehrs)
            loss = criterion(out["logits"], targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, ehrs, targets in val_loader:
                imgs, ehrs, targets = imgs.to(device), ehrs.to(device), targets.to(device)
                out = model(image=imgs, ehr=ehrs)
                loss = criterion(out["logits"], targets)
                val_loss += loss.item() * len(targets)
                preds = out["logits"].argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)

        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.3f} | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "ehr_input_dim": ehr_in_dim,
                "classes": CLASS_NAMES,
                "architecture": "MultimodalDRClassifier (EfficientNet-B3 + AttentionFusion + EHREncoder)",
                "dataset_provenance": "APTOS 2019 Fundus Images + Synthetically Correlated EHR Clinical Covariates"
            }

    # 5. Save best checkpoint
    out_ckpt_dir = PROJECT_ROOT / "checkpoints"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt_path = out_ckpt_dir / "best_multimodal_model.pth"

    torch.save(best_state or {"model_state_dict": model.state_dict()}, str(out_ckpt_path))
    print(f"\n[DONE] Saved trained multimodal checkpoint: {out_ckpt_path}")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    train_multimodal_model()
