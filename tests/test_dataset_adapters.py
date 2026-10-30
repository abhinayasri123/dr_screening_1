import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from ml.datasets.quality_checker import assess_image_quality
from ml.datasets.aptos_adapter import APTOSDatasetAdapter
from ml.datasets.idrid_adapter import IDRiDDatasetAdapter
from ml.datasets.messidor_adapter import Messidor2DatasetAdapter

def test_quality_checker_gradable_synthetic_image():
    # Crisp retinal fundus synthetic disc
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw high contrast circular pattern
    for r in range(50, 120, 10):
        img[r:300-r, r:300-r] = (180, 50 + r, 40)
    
    res = assess_image_quality(img)
    assert res["is_gradable"] is True
    assert res["quality_status"] == "GRADABLE"
    assert res["blur_score"] > 80.0
    assert 25.0 <= res["mean_luminance"] <= 230.0

def test_quality_checker_detects_blur():
    # Uniform smooth image (near zero variance)
    img_blurred = np.full((300, 300, 3), 120, dtype=np.uint8)
    res = assess_image_quality(img_blurred)
    assert res["is_gradable"] is False
    assert any("blur" in r.lower() for r in res["reasons"])

def test_quality_checker_detects_dark_underexposure():
    # Very dark image
    img_dark = np.full((300, 300, 3), 10, dtype=np.uint8)
    res = assess_image_quality(img_dark)
    assert res["is_gradable"] is False
    assert any("underexposure" in r.lower() or "dark" in r.lower() for r in res["reasons"])

def test_aptos_adapter_discovery_on_manifest(tmp_path):
    # Create mock APTOS folder
    img_dir = tmp_path / "train_images"
    img_dir.mkdir()
    (tmp_path / "train.csv").write_text("id_code,diagnosis\n001_mock,0\n002_mock,2\n")
    
    # Create matching mock image files
    Image.new("RGB", (300, 300), color=(150, 60, 40)).save(img_dir / "001_mock.png")
    Image.new("RGB", (300, 300), color=(150, 60, 40)).save(img_dir / "002_mock.png")
    
    adapter = APTOSDatasetAdapter(str(tmp_path))
    df = adapter.discover_and_load()
    assert len(df) == 2
    assert "diagnosis" in df.columns
    assert df.iloc[0]["diagnosis"] == 0
    assert df.iloc[1]["diagnosis"] == 2

def test_split_leakage_prevention(tmp_path):
    # Create 20 mock samples
    records = [{"id_code": f"pat_{i}", "diagnosis": i % 5, "image_path": f"/mock/{i}.png"} for i in range(20)]
    df = pd.DataFrame(records)
    
    adapter = APTOSDatasetAdapter(str(tmp_path))
    adapter.manifest_df = df
    train_df, val_df, test_df = adapter.create_splits(test_size=0.2, val_size=0.2, random_state=42)
    
    # Assert zero overlap
    train_ids = set(train_df["id_code"])
    val_ids = set(val_df["id_code"])
    test_ids = set(test_df["id_code"])
    
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
