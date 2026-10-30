import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from PIL import Image

from ml.preprocessing.image_preprocessor import (
    validate_image,
    RetinalImagePreprocessor,
    compute_class_weights
)
from ml.preprocessing.ehr_preprocessor import EHRPreprocessor, EHR_FEATURES

def test_image_preprocessor_resize_and_normalize():
    preprocessor = RetinalImagePreprocessor(image_size=300)
    img_np = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)
    
    val_transforms = preprocessor.get_val_transforms()
    augmented = val_transforms(image=img_np)
    tensor = augmented["image"]
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 300, 300)
    # Check normalized range (roughly)
    assert tensor.dtype == torch.float32

def test_train_transforms_apply():
    preprocessor = RetinalImagePreprocessor(image_size=300)
    img_np = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    train_transforms = preprocessor.get_train_transforms()
    augmented = train_transforms(image=img_np)
    tensor = augmented["image"]
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 300, 300)

def test_ehr_preprocessor_fit_transform(sample_ehr_df):
    preprocessor = EHRPreprocessor()
    preprocessor.fit(sample_ehr_df)
    
    transformed = preprocessor.transform(sample_ehr_df)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_ehr_df)
    assert transformed.shape[1] > 10
    assert not np.isnan(transformed).any()

def test_ehr_preprocessor_missing_values(sample_ehr_df):
    preprocessor = EHRPreprocessor()
    preprocessor.fit(sample_ehr_df)
    
    # Introduce NaNs in test df
    test_df = sample_ehr_df.copy()
    test_df.loc[0, "hba1c"] = np.nan
    test_df.loc[1, "gender"] = np.nan
    
    transformed = preprocessor.transform(test_df)
    assert not np.isnan(transformed).any()

def test_ehr_preprocessor_save_load(sample_ehr_df):
    preprocessor = EHRPreprocessor()
    preprocessor.fit(sample_ehr_df)
    orig_transformed = preprocessor.transform(sample_ehr_df)
    
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        preprocessor.save(tmp_path)
        loaded = EHRPreprocessor.load(tmp_path)
        loaded_transformed = loaded.transform(sample_ehr_df)
        np.testing.assert_allclose(orig_transformed, loaded_transformed, rtol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_class_weights_computation():
    labels = [0, 0, 0, 0, 1, 1, 2, 3, 4]
    weights = compute_class_weights(labels, num_classes=5)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape[0] == 5
    # Rare classes should have higher weight than frequent class 0
    assert weights[4] > weights[0]
