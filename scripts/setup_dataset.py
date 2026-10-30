#!/usr/bin/env python3
"""
Dataset Setup Script — APTOS 2019 Blindness Detection
======================================================
Downloads the APTOS 2019 dataset from Kaggle, organises it into the
required class-folder structure, and creates stratified train/val/test splits.

Dataset: https://www.kaggle.com/competitions/aptos2019-blindness-detection
Labels : 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR

Usage:
    python scripts/setup_dataset.py
    python scripts/setup_dataset.py --data-dir data --seed 42
"""

import os
import sys
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


def check_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return False


def print_credential_instructions():
    print("\n" + "=" * 60)
    print("KAGGLE CREDENTIALS REQUIRED")
    print("=" * 60)
    print("""
To download APTOS 2019 you need a free Kaggle account and API token.

Step 1 — Create a free Kaggle account (if you don't have one):
          https://www.kaggle.com

Step 2 — Join the competition and accept the rules:
          https://www.kaggle.com/competitions/aptos2019-blindness-detection
          Click "Join Competition" → accept the rules.

Step 3 — Generate an API token:
          https://www.kaggle.com/settings → API → "Create New Token"
          This downloads  kaggle.json

Step 4 — Place kaggle.json:
          Windows: C:\\Users\\<YOU>\\.kaggle\\kaggle.json
          Linux/Mac: ~/.kaggle/kaggle.json
          OR set environment variables:
            KAGGLE_USERNAME=your_username
            KAGGLE_KEY=your_api_key

Step 5 — Re-run this script:
          python scripts/setup_dataset.py

Alternative — Manual download:
          Download train.zip from the competition page,
          extract to  data/raw/aptos2019/
          then re-run with  --skip-download
""")
    print("=" * 60 + "\n")


def download_dataset(download_dir: Path) -> Path:
    """Download APTOS 2019 competition data via kaggle CLI or kagglehub."""
    download_dir.mkdir(parents=True, exist_ok=True)

    # Try kagglehub first (newer)
    try:
        import kagglehub
        print("Downloading via kagglehub …")
        path = kagglehub.competition_download("aptos2019-blindness-detection",
                                               path=str(download_dir))
        return Path(path)
    except Exception as e:
        print(f"  kagglehub failed ({e}), trying kaggle CLI …")

    # Fallback: kaggle CLI
    import subprocess
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", "aptos2019-blindness-detection",
        "-p", str(download_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("kaggle CLI error:\n", result.stderr)
        raise RuntimeError(
            "Could not download dataset. "
            "Install kaggle:  pip install kaggle\n"
            + result.stderr
        )

    # Unzip if needed
    zip_path = download_dir / "aptos2019-blindness-detection.zip"
    if zip_path.exists():
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(download_dir)

    return download_dir


def find_csv_and_image_dir(raw_dir: Path):
    """Locate train.csv and train_images/ inside raw_dir (handles nested dirs)."""
    csv_path = None
    img_dir  = None

    for p in raw_dir.rglob("train.csv"):
        csv_path = p
        break

    for p in raw_dir.rglob("train_images"):
        if p.is_dir():
            img_dir = p
            break

    return csv_path, img_dir


def create_class_folders(retinal_dir: Path):
    for split in ["train", "validation", "test"]:
        for cls in range(5):
            d = retinal_dir / split / str(cls)
            d.mkdir(parents=True, exist_ok=True)
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()


def copy_images(df: pd.DataFrame, img_dir: Path, dest_split_dir: Path,
                split_name: str) -> dict:
    """Copy images to class-folder structure. Returns per-class counts."""
    counts = {c: 0 for c in range(5)}
    missing = 0

    for _, row in df.iterrows():
        img_name = str(row["id_code"]).strip()
        label    = int(row["diagnosis"])

        # Try both with and without extension
        src = img_dir / f"{img_name}.png"
        if not src.exists():
            src = img_dir / f"{img_name}.jpg"
        if not src.exists():
            src = img_dir / img_name
        if not src.exists():
            missing += 1
            continue

        dst_dir = dest_split_dir / str(label)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        counts[label] += 1

    if missing:
        print(f"  ⚠  {missing} images not found in {img_dir} — skipped.")

    return counts


def check_data_leakage(train_df, val_df, test_df):
    train_ids = set(train_df["id_code"])
    val_ids   = set(val_df["id_code"])
    test_ids  = set(test_df["id_code"])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    if tv or tt or vt:
        print(f"  ❌ DATA LEAKAGE DETECTED: train∩val={len(tv)}, "
              f"train∩test={len(tt)}, val∩test={len(vt)}")
    else:
        print("  ✅ No data leakage — all splits are disjoint.")


def print_split_stats(name: str, df: pd.DataFrame):
    print(f"\n  {name} ({len(df)} images):")
    for c in range(5):
        n = (df["diagnosis"] == c).sum()
        pct = n / len(df) * 100 if len(df) else 0
        print(f"    Class {c} ({CLASS_NAMES[c]}): {n:5d}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and organise APTOS 2019 dataset for DR Screening project"
    )
    parser.add_argument("--data-dir",      type=str, default="data",
                        help="Project data directory (default: data/)")
    parser.add_argument("--download-dir",  type=str, default="data/raw",
                        help="Directory to download raw Kaggle data into")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Random seed for stratified split")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if raw data already present")
    args = parser.parse_args()

    # Resolve paths relative to project root (one level up from scripts/)
    script_dir  = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    data_dir    = project_dir / args.data_dir
    retinal_dir = data_dir / "retinal"
    download_dir = project_dir / args.download_dir

    print("=" * 60)
    print("APTOS 2019 Dataset Setup")
    print("=" * 60)
    print(f"  Project root : {project_dir}")
    print(f"  Data dir     : {retinal_dir}")
    print(f"  Download dir : {download_dir}")
    print(f"  Random seed  : {args.seed}")

    # ── 1. Credentials check ────────────────────────────────────────────────
    if not args.skip_download:
        if not check_kaggle_credentials():
            print_credential_instructions()
            sys.exit(1)
        print("\n✅ Kaggle credentials found.")

    # ── 2. Download ─────────────────────────────────────────────────────────
    if args.skip_download:
        raw_dir = download_dir
        print("\nSkipping download — using existing data in", raw_dir)
    else:
        print("\nDownloading APTOS 2019 …")
        raw_dir = download_dataset(download_dir)
        print(f"  → Raw data at: {raw_dir}")

    # ── 3. Locate CSV + image dir ────────────────────────────────────────────
    csv_path, img_dir = find_csv_and_image_dir(raw_dir)
    if csv_path is None:
        print(f"\n❌ Could not find train.csv under {raw_dir}")
        print("   Please extract the dataset manually to data/raw/aptos2019/")
        sys.exit(1)
    if img_dir is None:
        print(f"\n❌ Could not find train_images/ under {raw_dir}")
        sys.exit(1)

    print(f"\n  CSV          : {csv_path}")
    print(f"  Images dir   : {img_dir}")

    df = pd.read_csv(csv_path)
    print(f"\n  Total labelled images: {len(df)}")
    print_split_stats("Full dataset", df)

    # ── 4. Stratified split ──────────────────────────────────────────────────
    # First split off test, then split remainder into train+val
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, stratify=df["diagnosis"],
        random_state=args.seed
    )
    relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, stratify=train_val_df["diagnosis"],
        random_state=args.seed
    )

    print(f"\n  Split sizes — Train: {len(train_df)}, "
          f"Val: {len(val_df)}, Test: {len(test_df)}")
    check_data_leakage(train_df, val_df, test_df)

    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)

    # ── 5. Create folder structure ───────────────────────────────────────────
    print("\nCreating class folders …")
    create_class_folders(retinal_dir)

    # ── 6. Copy images ───────────────────────────────────────────────────────
    print("Copying train images …")
    copy_images(train_df, img_dir, retinal_dir / "train", "train")

    print("Copying validation images …")
    copy_images(val_df, img_dir, retinal_dir / "validation", "validation")

    print("Copying test images …")
    copy_images(test_df, img_dir, retinal_dir / "test", "test")

    # ── 7. Save split CSVs ───────────────────────────────────────────────────
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    val_df.to_csv(splits_dir / "val_split.csv",   index=False)
    test_df.to_csv(splits_dir / "test_split.csv",  index=False)
    print(f"\n  Split CSVs saved to {splits_dir}/")

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Dataset setup complete!")
    print("=" * 60)
    print(f"""
Next steps:
  1. Install Python dependencies:
       pip install -r requirements.txt

  2. Generate synthetic EHR demo data:
       python scripts/generate_synthetic_ehr.py

  3. Train the CNN:
       python ml/training/train.py --mode cnn --epochs 30

  4. Run Federated Learning simulation:
       python scripts/run_fl_simulation.py

  5. Start the backend:
       uvicorn backend.main:app --reload

  6. Start the frontend:
       cd frontend && npm install && npm run dev
""")


if __name__ == "__main__":
    main()
