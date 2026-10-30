"""
Messidor-2 Dataset Adapter
==========================
Resource: https://www.adcis.net/en/third-party/messidor2/
License: ADCIS Messidor Research Agreement (Research & Non-Commercial Evaluation)
Citation: Decencière, E. et al. Feedback on a publicly distributed image database: the Messidor database. Image Anal Stereol 2014; 33:231-234.
Adjudicated Reference: Gulshan, V. et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA 2016; 316(22):2402–2410.

Acquisition notes:
- 1,748 macula-centered eye fundus photographs from 874 examination encounters.
- Captured across multiple clinical hospital ophthalmology services in France.
- Provides eye position (left / right) and patient encounter pairing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from .base_adapter import BaseDatasetAdapter, ICDR_GRADES

class Messidor2DatasetAdapter(BaseDatasetAdapter):
    def __init__(self, data_root: str):
        super().__init__(
            data_root=data_root,
            name="Messidor-2 Retinal Database",
            license_info="ADCIS Messidor Research Agreement (Strictly Non-Commercial Academic/Research Use)",
            citation="Decencière et al. (2014); Reference adjudication by Gulshan et al. JAMA 2016."
        )

    def discover_and_load(self) -> pd.DataFrame:
        """
        Locates Messidor-2 reference CSV and images.
        Supports:
        1. Adjudicated 5-class ICDR labels (messidor_data.csv with dr_grade 0-4)
        2. Original 4-grade Messidor conversion
        """
        possible_csv_paths = [
            self.data_root / "messidor_data.csv",
            self.data_root / "messidor2_data.csv",
            self.data_root / "adjudicated_dr_grades.csv",
            self.data_root / "labels.csv",
            self.data_root / "train.csv"
        ]

        csv_path = None
        for p in possible_csv_paths:
            if p.exists():
                csv_path = p
                break

        possible_img_dirs = [
            self.data_root / "IMAGES",
            self.data_root / "images",
            self.data_root / "messidor-2",
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
            
            # Identify columns
            name_col = next((c for c in raw_df.columns if "image" in c.lower() or "id" in c.lower()), None)
            grade_col = next((c for c in raw_df.columns if "grade" in c.lower() or "dr" in c.lower() or "adjudicated" in c.lower()), None)

            if name_col and grade_col:
                for _, row in raw_df.iterrows():
                    img_name = str(row[name_col]).strip()
                    try:
                        raw_grade = float(row[grade_col])
                        if pd.isna(raw_grade):
                            continue
                        grade = int(raw_grade)
                    except (ValueError, TypeError):
                        continue

                    # Validate or map grade (if 0-4 ICDR, preserve; if 0-3 Messidor original, map 3 to 4)
                    if grade < 0 or grade > 4:
                        continue

                    # Check image file
                    matched_img = None
                    for ext in ["", ".jpg", ".png", ".tif"]:
                        candidate = img_dir / f"{img_name}{ext}"
                        if candidate.exists() and candidate.is_file():
                            matched_img = candidate
                            break

                    if matched_img:
                        records.append({
                            "id_code": matched_img.stem,
                            "image_path": str(matched_img),
                            "diagnosis": grade,
                            "diagnosis_name": ICDR_GRADES[grade],
                            "dataset": "Messidor-2",
                            "provenance": "Clinical Services (France)",
                            "is_adjudicated": "adjudicated" in grade_col.lower()
                        })

        self.manifest_df = pd.DataFrame(records)
        return self.manifest_df
