#!/usr/bin/env python3
"""
Demo Data Creator
==================
Creates a tiny demo dataset so the full pipeline can be tested
WITHOUT downloading the real APTOS 2019 dataset.

⚠️ DEMO DATA — NOT FOR RESEARCH PURPOSES.
   Synthetic images do not represent real retinal pathology.
   Use only for software testing and UI demonstration.

Usage:
    python scripts/create_demo_data.py
"""

import os
import sys
import json
import random
import struct
import zlib
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available — will create minimal PNG bytes manually.")


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
TRAIN_PER_CLASS = 8
VAL_PER_CLASS   = 3
TEST_PER_CLASS  = 3

# Colours that loosely differentiate classes (not medically accurate)
CLASS_COLORS = [
    (180, 80,  80),   # 0 No DR — dark red fundus
    (160, 90,  60),   # 1 Mild
    (140, 100, 50),   # 2 Moderate
    (120, 110, 40),   # 3 Severe
    (100, 120, 30),   # 4 Proliferative DR
]


def make_synthetic_fundus(cls: int, idx: int, size: int = 224) -> "Image.Image":
    """
    Create a plausible-looking synthetic fundus image.
    Uses a dark circular background with a central optic disc and
    radial vessels in class-specific colours. NOT medically realistic.
    """
    rng = np.random.default_rng(cls * 100 + idx)
    img = Image.new("RGB", (size, size), color=(10, 5, 5))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 2 - 5

    # Dark fundus circle
    base_col = CLASS_COLORS[cls]
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(
        max(0, base_col[0] - 60), max(0, base_col[1] - 40), max(0, base_col[2] - 40)
    ))

    # Radial blood vessels
    n_vessels = 6 + cls * 2
    for v in range(n_vessels):
        angle = 2 * np.pi * v / n_vessels + rng.uniform(-0.1, 0.1)
        length = r * rng.uniform(0.4, 0.85)
        x1 = int(cx + 10 * np.cos(angle))
        y1 = int(cy + 10 * np.sin(angle))
        x2 = int(cx + length * np.cos(angle))
        y2 = int(cy + length * np.sin(angle))
        vessel_col = (max(0, base_col[0] - 80), 20, 20)
        draw.line([(x1, y1), (x2, y2)], fill=vessel_col, width=max(1, 3 - cls // 2))

    # Optic disc
    od_r = size // 10
    draw.ellipse([cx - od_r, cy - od_r, cx + od_r, cy + od_r],
                 fill=(220, 180, 120))

    # Microaneurysms / lesions increase with class severity
    n_lesions = cls * 3
    for _ in range(n_lesions):
        lx = int(rng.integers(cx - r + 20, cx + r - 20))
        ly = int(rng.integers(cy - r + 20, cy + r - 20))
        lr = rng.integers(2, 5)
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr],
                     fill=(220, 80, 80))

    # Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def create_split(split_name: str, n_per_class: int, retinal_dir: Path):
    split_dir = retinal_dir / split_name
    print(f"  Creating {split_name} split ({n_per_class} images/class) …")
    for cls in range(5):
        cls_dir = split_dir / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            if PIL_AVAILABLE:
                img = make_synthetic_fundus(cls, i)
                img_path = cls_dir / f"demo_cls{cls}_{split_name}_{i:03d}.png"
                img.save(str(img_path))
            else:
                # Minimal valid 1x1 PNG
                img_path = cls_dir / f"demo_cls{cls}_{split_name}_{i:03d}.png"
                _write_minimal_png(img_path, cls)
    print(f"    [OK] {split_name}: {5 * n_per_class} images")


def _write_minimal_png(path: Path, cls: int):
    """Write a 10x10 PNG with class-specific colour without PIL."""
    col = CLASS_COLORS[cls]

    def _png_chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    w, h = 10, 10
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b""
    for _ in range(h):
        raw += b"\x00" + bytes(col) * w
    idat = zlib.compress(raw)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", idat)
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def create_demo_checkpoint(checkpoints_dir: Path):
    """Save a random-weight model checkpoint so demo mode shows 'loaded'."""
    try:
        import torch
        from ml.models.cnn_model import DRClassifier
        model = DRClassifier(num_classes=5, pretrained=False)
        ckpt = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "metrics": {"accuracy": 0.0, "qwk": 0.0, "note": "DEMO_RANDOM_WEIGHTS"},
        }
        path = checkpoints_dir / "best_cnn_model.pth"
        torch.save(ckpt, str(path))
        print(f"  [OK] Demo checkpoint saved: {path}")
    except Exception as e:
        print(f"  [WARN] Could not create demo checkpoint: {e}")


def main():
    print("=" * 60)
    print("DEMO DATA CREATOR")
    print("[WARNING] NOT FOR RESEARCH -- SYNTHETIC IMAGES ONLY")
    print("=" * 60)

    retinal_dir    = _project_root / "data" / "retinal"
    checkpoints_dir = _project_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    create_split("train",      TRAIN_PER_CLASS, retinal_dir)
    create_split("validation", VAL_PER_CLASS,   retinal_dir)
    create_split("test",       TEST_PER_CLASS,  retinal_dir)

    print("\nCreating demo checkpoint ...")
    create_demo_checkpoint(checkpoints_dir)

    print("\nGenerating synthetic EHR demo data ...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(_project_root / "scripts" / "generate_synthetic_ehr.py"),
             "--n-samples", "100"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  [OK] Synthetic EHR data created")
        else:
            print(f"  [WARN] EHR gen warning: {result.stderr[:200]}")
    except Exception as e:
        print(f"  [WARN] EHR gen skipped: {e}")

    print("\n" + "=" * 60)
    print("[OK] Demo data ready!")
    print("=" * 60)
    total = (TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS) * 5
    print(f"  Total synthetic images: {total}")
    print(f"  Classes: {CLASS_NAMES}")
    print("""
Next steps:
  1. pip install -r requirements.txt
  2. uvicorn backend.main:app --reload        (start backend)
  3. cd frontend && npm install && npm run dev (start frontend)
  4. Open http://localhost:5173
  5. Login: admin@dr-screening.com / admin123

[REMINDER] Run scripts/setup_dataset.py to download the REAL
APTOS 2019 dataset for actual training and research.
""")


if __name__ == "__main__":
    main()
