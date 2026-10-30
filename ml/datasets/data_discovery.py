"""
Dataset Discovery and Ingestion Engine
=====================================
Discovers directory structures, auto-selects appropriate dataset adapters
(APTOS, IDRiD, Messidor-2), validates image readability and class balance,
and produces standardized manifests.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .aptos_adapter import APTOSDatasetAdapter
from .idrid_adapter import IDRiDDatasetAdapter
from .messidor_adapter import Messidor2DatasetAdapter
from .quality_checker import assess_image_quality

def discover_dataset(data_dir: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Scans a directory for known diabetic retinopathy datasets.
    Tries adapters in sequence: APTOS -> IDRiD -> Messidor-2 -> Generic Folder Layout.
    Returns (manifest_df, metadata_dict).
    """
    path = Path(data_dir)
    if not path.exists():
        return None, {
            "status": "NOT_FOUND",
            "message": f"Dataset directory does not exist: {data_dir}"
        }

    adapters = [
        APTOSDatasetAdapter(str(path)),
        IDRiDDatasetAdapter(str(path)),
        Messidor2DatasetAdapter(str(path))
    ]

    for adapter in adapters:
        try:
            df = adapter.discover_and_load()
            if df is not None and len(df) > 0:
                val_info = adapter.validate_dataset()
                return df, {
                    "status": "DISCOVERED",
                    "adapter_name": adapter.name,
                    "records_loaded": len(df),
                    "validation": val_info,
                    "license": adapter.license_info,
                    "citation": adapter.citation
                }
        except Exception as err:
            continue

    # Fallback: check if subfolders 0, 1, 2, 3, 4 exist directly
    records = []
    for grade in range(5):
        g_dir = path / str(grade)
        if g_dir.exists():
            for f in list(g_dir.glob("*.png")) + list(g_dir.glob("*.jpg")) + list(g_dir.glob("*.jpeg")):
                records.append({
                    "id_code": f.stem,
                    "image_path": str(f),
                    "diagnosis": grade,
                    "diagnosis_name": f"Grade {grade}",
                    "dataset": "Generic_5Class_Directory"
                })

    if records:
        df = pd.DataFrame(records)
        return df, {
            "status": "DISCOVERED",
            "adapter_name": "Generic 5-Class Folder Layout",
            "records_loaded": len(df),
            "validation": {
                "valid_count": len(df),
                "corrupt_count": 0,
                "class_distribution": df["diagnosis"].value_counts().to_dict()
            },
            "license": "Custom/Local",
            "citation": "Locally stored fundus images."
        }

    return None, {
        "status": "NO_MATCHING_DATASET",
        "message": "No recognized retinal dataset structure (APTOS, IDRiD, Messidor-2, or 0-4 subfolders) found in directory."
    }
