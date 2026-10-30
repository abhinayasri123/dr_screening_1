"""
IDRiD Dataset Adapter (Indian Diabetic Retinopathy Image Dataset)
================================================================
Grand Challenge: https://idrid.grand-challenge.org/
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Citation: Porwal et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD)", IEEE Dataport (2018).

Acquisition notes:
- 516 digitized retinal fundus images acquired from rural eye clinics in Nanded, India.
- 50-degree field of view captured with Kowa VX-10 alpha digital fundus camera.
- Ground truth provides Retinopathy Grade (0-4) according to ICDR and Diabetic Macular Edema (DME) risk (0-2).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from .base_adapter import BaseDatasetAdapter, ICDR_GRADES

class IDRiDDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, data_root: str):
        super().__init__(
            data_root=data_root,
            name="IDRiD (Indian Diabetic Retinopathy Image Dataset)",
            license_info="CC BY 4.0 (Creative Commons Attribution 4.0 International)",
            citation="Porwal, P., et al. Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research. Data 2018, 3, 25."
        )

    def discover_and_load(self) -> pd.DataFrame:
        """
        Locates IDRiD grading ground truth CSV and images.
        Expected CSV columns: 'Image name', 'Retinopathy grade', 'Risk of macular edema '
        """
        possible_csv_paths = [
            self.data_root / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv",
            self.data_root / "IDRiD_Disease Grading_Training Labels.csv",
            self.data_root / "idrid_training_labels.csv",
            self.data_root / "train.csv"
        ]

        csv_path = None
        for p in possible_csv_paths:
            if p.exists():
                csv_path = p
                break

        possible_img_dirs = [
            self.data_root / "B. Disease Grading" / "1. Original Images" / "a. Training Set",
            self.data_root / "Original Images" / "a. Training Set",
            self.data_root / "images",
            self.data_root / "train_images"
        ]

        img_dir = None
        for d in possible_img_dirs:
            if d.exists() and d.is_dir():
                img_dir = d
                break

        records = []
        if csv_path and img_dir:
            raw_df = pd.read_csv(csv_path)
            # Find image name column
            name_col = next((c for c in raw_df.columns if "Image" in c or "name" in c.lower()), None)
            grade_col = next((c for c in raw_df.columns if "Retinopathy" in c or "grade" in c.lower()), None)
            dme_col = next((c for c in raw_df.columns if "macular" in c.lower() or "dme" in c.lower()), None)

            if name_col and grade_col:
                for _, row in raw_df.iterrows():
                    id_name = str(row[name_col]).strip()
                    try:
                        grade = int(row[grade_col])
                    except (ValueError, TypeError):
                        continue

                    # Find file
                    matched_img = None
                    for ext in [".jpg", ".tif", ".png", ".jpeg"]:
                        candidate = img_dir / f"{id_name}{ext}"
                        if candidate.exists():
                            matched_img = candidate
                            break

                    if matched_img and 0 <= grade <= 4:
                        dme_val = int(row[dme_col]) if dme_col and pd.notna(row[dme_col]) else None
                        records.append({
                            "id_code": id_name,
                            "image_path": str(matched_img),
                            "diagnosis": grade,
                            "diagnosis_name": ICDR_GRADES[grade],
                            "dme_risk": dme_val,
                            "dataset": "IDRiD",
                            "camera": "Kowa VX-10 alpha",
                            "provenance": "Nanded Clinic, India"
                        })

        self.manifest_df = pd.DataFrame(records)
        return self.manifest_df
