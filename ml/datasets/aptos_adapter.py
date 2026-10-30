"""
APTOS 2019 Blindness Detection Dataset Adapter
=============================================
Primary dataset adapter for Kaggle APTOS 2019 Blindness Detection.
License: Competition Rules / EyePACS & Aravind Eye Hospital Research Use
Dataset Link: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Discovers dynamic directory layouts, validates matching images and labels,
records class counts, and provides leakage-free splitting.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from PIL import Image

from .base_adapter import BaseDatasetAdapter, ICDR_GRADES

class APTOSDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, data_root: str):
        super().__init__(
            data_root=data_root,
            name="APTOS 2019 Blindness Detection",
            license_info="Kaggle APTOS 2019 Dataset License / Aravind Eye Care System & EyePACS Research Use",
            citation="APTOS 2019 Blindness Detection, Asia Pacific Tele-Ophthalmology Society & EyePACS (2019)."
        )

    def discover_and_load(self) -> pd.DataFrame:
        """
        Dynamically locates CSV and image directories across possible extraction formats.
        """
        possible_csv_paths = [
            self.data_root / "train.csv",
            self.data_root / "aptos2019-blindness-detection" / "train.csv",
            self.data_root / "train" / "train.csv",
            self.data_root / "data" / "train.csv",
        ]

        csv_path = None
        for p in possible_csv_paths:
            if p.exists():
                csv_path = p
                break

        # Check for image directories
        possible_img_dirs = [
            self.data_root / "train_images",
            self.data_root / "aptos2019-blindness-detection" / "train_images",
            self.data_root / "train",
            self.data_root / "images",
        ]

        img_dir = None
        for d in possible_img_dirs:
            if d.exists() and d.is_dir():
                # Verify it contains at least some images
                files = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
                if files:
                    img_dir = d
                    break

        records = []

        # Case 1: CSV exists
        if csv_path and img_dir:
            raw_df = pd.read_csv(csv_path)
            for _, row in raw_df.iterrows():
                id_code = str(row["id_code"]).strip()
                diag = int(row["diagnosis"])
                
                # Try finding image with extension
                matched_img = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = img_dir / f"{id_code}{ext}"
                    if candidate.exists():
                        matched_img = candidate
                        break

                if matched_img:
                    records.append({
                        "id_code": id_code,
                        "image_path": str(matched_img),
                        "diagnosis": diag,
                        "diagnosis_name": ICDR_GRADES.get(diag, "Unknown"),
                        "dataset": "APTOS2019",
                        "eye": "unspecified"
                    })

        # Case 2: Subfolder layout by class (e.g. data/retinal/train/{0,1,2,3,4})
        if not records:
            subdirs = [self.data_root / "retinal" / "train", self.data_root / "train", self.data_root]
            for base in subdirs:
                if (base / "0").exists() and (base / "4").exists():
                    for grade in range(5):
                        g_dir = base / str(grade)
                        if g_dir.exists():
                            for f in list(g_dir.glob("*.png")) + list(g_dir.glob("*.jpg")):
                                records.append({
                                    "id_code": f.stem,
                                    "image_path": str(f),
                                    "diagnosis": grade,
                                    "diagnosis_name": ICDR_GRADES[grade],
                                    "dataset": "APTOS2019_LocalPartition",
                                    "eye": "unspecified"
                                })
                    if records:
                        break

        self.manifest_df = pd.DataFrame(records)
        return self.manifest_df
