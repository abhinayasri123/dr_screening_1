"""
Base Dataset Adapter Interface
==============================
Defines the standard contract for medical retinal fundus dataset ingestion.
Provides:
- Directory discovery & file validation
- Ground truth label mapping to standard 5-class ICDR scale:
    0: No DR
    1: Mild DR
    2: Moderate DR
    3: Severe DR
    4: Proliferative DR
- Reproducible, leakage-free stratified splitting (Train / Val / Test)
- Manifest generation and dataset provenance tracking
"""

from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .quality_checker import assess_image_quality

ICDR_GRADES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

class BaseDatasetAdapter(ABC):
    def __init__(self, data_root: str, name: str, license_info: str, citation: str):
        self.data_root = Path(data_root)
        self.name = name
        self.license_info = license_info
        self.citation = citation
        self.records: List[Dict[str, Any]] = []
        self.manifest_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def discover_and_load(self) -> pd.DataFrame:
        """
        Discovers local files, maps native labels to ICDR 0-4,
        validates image readable integrity, and returns standard DataFrame.
        """
        pass

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validates the loaded dataset records, checks file existence and quality,
        and outputs a quality & balance summary.
        """
        if self.manifest_df is None or len(self.manifest_df) == 0:
            return {"status": "EMPTY", "valid_count": 0, "corrupt_count": 0}

        valid_records = []
        corrupt_records = []

        for idx, row in self.manifest_df.iterrows():
            img_path = str(row["image_path"])
            quality = assess_image_quality(img_path)
            if quality["is_corrupt"]:
                corrupt_records.append({
                    "id": row.get("id_code", idx),
                    "path": img_path,
                    "reasons": quality["reasons"]
                })
            else:
                valid_records.append({
                    **row.to_dict(),
                    "blur_score": quality["blur_score"],
                    "mean_luminance": quality["mean_luminance"],
                    "is_gradable": quality["is_gradable"]
                })

        valid_df = pd.DataFrame(valid_records)
        class_dist = {}
        if not valid_df.empty and "diagnosis" in valid_df.columns:
            class_dist = valid_df["diagnosis"].value_counts().to_dict()

        return {
            "status": "VALIDATED",
            "dataset_name": self.name,
            "total_declared": len(self.manifest_df),
            "valid_count": len(valid_df),
            "corrupt_count": len(corrupt_records),
            "corrupt_records": corrupt_records,
            "class_distribution": class_dist,
            "license": self.license_info,
            "citation": self.citation
        }

    def create_splits(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates reproducible stratified train, validation, and held-out test splits.
        Guarantees no leakage between splits.
        """
        if self.manifest_df is None or len(self.manifest_df) == 0:
            raise ValueError(f"Cannot split empty dataset in {self.name}")

        df = self.manifest_df.copy()
        stratify_col = "diagnosis" if "diagnosis" in df.columns else None
        
        # Check if any class has too few instances for stratified splitting
        can_stratify = False
        if stratify_col:
            counts = df[stratify_col].value_counts()
            n_classes = len(counts)
            min_test = int(len(df) * test_size)
            if counts.min() >= 2 and min_test >= n_classes:
                can_stratify = True

        # Split train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col] if can_stratify else None,
            random_state=random_state
        )

        # Split train vs val
        adjusted_val_size = val_size / (1.0 - test_size)
        can_stratify_val = False
        if can_stratify:
            v_counts = train_val_df[stratify_col].value_counts()
            min_val = int(len(train_val_df) * adjusted_val_size)
            if v_counts.min() >= 2 and min_val >= len(v_counts):
                can_stratify_val = True

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df[stratify_col] if can_stratify_val else None,
            random_state=random_state
        )

        train_df = train_df.assign(split="train")
        val_df = val_df.assign(split="validation")
        test_df = test_df.assign(split="test")

        self.manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return train_df, val_df, test_df

    def export_manifest(self, output_path: str):
        """Exports versioned split manifest CSV."""
        if self.manifest_df is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.manifest_df.to_csv(output_path, index=False)
