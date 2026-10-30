"""
Training Script for Stage A Retinal vs Non-Retinal Detector
===========================================================
Trains a binary classifier (RetinalDetector) to distinguish genuine retinal
fundus photographs from all arbitrary images (portraits, documents, X-rays, etc.).
Saves checkpoint to checkpoints/retinal_detector.pth with training metrics.
"""

import os
import sys
from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.retinal_detector import RetinalDetector

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def generate_non_retinal_samples(output_dir: Path, num_samples: int = 150):
    """Generates diverse non-retinal synthetic benchmark samples across 6 distinct categories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = ["portrait", "document", "xray", "cartoon", "nature", "screenshot"]
    
    for i in range(num_samples):
        cat = categories[i % len(categories)]
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        if cat == "portrait":
            # Human face/portrait simulation: skin tone oval, hair, eyes, clothing
            skin_color = [random.randint(180, 240), random.randint(140, 190), random.randint(120, 160)] # RGB
            bg_color = [random.randint(80, 180), random.randint(100, 200), random.randint(150, 240)] # Cool backdrop
            img[:] = bg_color
            # Face oval
            cv2.ellipse(img, (128, 120), (55, 75), 0, 0, 360, skin_color, -1)
            # Eyes
            cv2.circle(img, (108, 110), 8, (255, 255, 255), -1)
            cv2.circle(img, (148, 110), 8, (255, 255, 255), -1)
            cv2.circle(img, (108, 110), 4, (40, 40, 40), -1)
            cv2.circle(img, (148, 110), 4, (40, 40, 40), -1)
            # Clothes / shirt
            shirt_color = [random.randint(30, 220), random.randint(30, 220), random.randint(150, 255)]
            cv2.rectangle(img, (40, 200), (216, 256), shirt_color, -1)
            
        elif cat == "document":
            # White/cream paper document with text lines and header
            img[:] = random.randint(235, 255)
            # Text lines
            for y in range(40, 240, 14):
                line_w = random.randint(120, 210)
                cv2.line(img, (25, y), (25 + line_w, y), (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)), 2)
            # Stamp or logo
            cv2.rectangle(img, (180, 20), (230, 50), (200, 50, 50), 2)
            
        elif cat == "xray":
            # Monochromatic radiograph (Chest X-ray / bone scan)
            gray = np.zeros((256, 256), dtype=np.uint8)
            # Ribs / spine curves
            for y in range(50, 220, 25):
                cv2.ellipse(gray, (128, y), (90, 20), 0, 0, 180, random.randint(160, 220), 5)
            cv2.line(gray, (128, 30), (128, 240), random.randint(190, 240), 8)
            # Lung fields (darker)
            cv2.ellipse(gray, (80, 130), (35, 55), 0, 0, 360, random.randint(30, 70), -1)
            cv2.ellipse(gray, (176, 130), (35, 55), 0, 0, 360, random.randint(30, 70), -1)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
        elif cat == "cartoon":
            # Bright saturated geometric illustration / cartoon
            img[:] = [random.randint(100, 255), random.randint(50, 200), random.randint(200, 255)]
            cv2.circle(img, (128, 128), 60, (255, 220, 0), -1)
            cv2.rectangle(img, (70, 70), (120, 120), (0, 255, 100), -1)
            cv2.line(img, (0, 0), (256, 256), (0, 0, 0), 4)
            
        elif cat == "nature":
            # Landscape: blue sky top, green grass bottom
            img[:128, :] = [random.randint(60, 120), random.randint(140, 200), random.randint(220, 255)] # Sky
            img[128:, :] = [random.randint(30, 70), random.randint(140, 220), random.randint(30, 80)]   # Grass
            cv2.circle(img, (200, 60), 30, (255, 255, 180), -1) # Sun
            
        elif cat == "screenshot":
            # Computer UI screenshot with window header, buttons, and scrollbar
            img[:] = [240, 242, 245]
            cv2.rectangle(img, (0, 0), (256, 32), (50, 60, 80), -1)
            cv2.circle(img, (15, 16), 5, (255, 80, 80), -1)
            cv2.circle(img, (30, 16), 5, (255, 200, 80), -1)
            cv2.circle(img, (45, 16), 5, (80, 220, 120), -1)
            cv2.rectangle(img, (20, 50), (140, 90), (0, 120, 240), -1) # Button
            cv2.rectangle(img, (240, 32), (256, 256), (210, 215, 220), -1) # Scrollbar
            
        # Add slight natural sensor noise
        noise = np.random.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        filepath = output_dir / f"non_retinal_{cat}_{i:03d}.png"
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class RetinalVsNonRetinalDataset(Dataset):
    def __init__(self, retinal_paths, non_retinal_paths, transform=None):
        self.samples = []
        for p in retinal_paths:
            self.samples.append((p, 1)) # Retinal = 1
        for p in non_retinal_paths:
            self.samples.append((p, 0)) # Non-Retinal = 0
            
        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
        except Exception:
            img_rgb = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = transforms.ToTensor()(img_rgb)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

def train_retinal_detector():
    print("[1/5] Gathering Retinal and Generating Non-Retinal Benchmark Data...")
    retinal_dir = PROJECT_ROOT / "data" / "retinal"
    retinal_files = list(retinal_dir.rglob("*.png")) + list(retinal_dir.rglob("*.jpg"))
    print(f"Found {len(retinal_files)} genuine retinal fundus images.")

    non_retinal_dir = PROJECT_ROOT / "data" / "non_retinal"
    generate_non_retinal_samples(non_retinal_dir, num_samples=180)
    non_retinal_files = list(non_retinal_dir.glob("*.png"))
    print(f"Prepared {len(non_retinal_files)} diverse non-retinal images.")

    # Split 80% train, 20% validation
    random.shuffle(retinal_files)
    random.shuffle(non_retinal_files)

    n_r_train = int(len(retinal_files) * 0.8)
    n_nr_train = int(len(non_retinal_files) * 0.8)

    train_retinal = retinal_files[:n_r_train]
    val_retinal = retinal_files[n_r_train:]

    train_non_retinal = non_retinal_files[:n_nr_train]
    val_non_retinal = non_retinal_files[n_nr_train:]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RetinalVsNonRetinalDataset(train_retinal, train_non_retinal, transform=train_transform)
    val_dataset = RetinalVsNonRetinalDataset(val_retinal, val_non_retinal, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"[2/5] Initializing RetinalDetector (MobileNetV3-Small Backbone)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinalDetector(pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 12
    best_acc = 0.0
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "retinal_detector.pth"

    print(f"[3/5] Commencing Training over {epochs} Epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "train_acc": train_acc,
                "architecture": "MobileNetV3-Small Binary Retinal Detector",
                "classes": ["Non-Retinal", "Retinal"],
                "target_size": 224
            }, str(checkpoint_path))

    print(f"[4/5] Checkpoint successfully saved to: {checkpoint_path}")
    print(f"[5/5] Final Model Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_retinal_detector()
