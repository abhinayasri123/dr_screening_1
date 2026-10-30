import pytest
import torch
import tempfile
import os
import numpy as np
from PIL import Image

from ml.models.cnn_model import DRClassifier, load_model, save_model
from ml.models.ehr_encoder import EHREncoder
from ml.models.multimodal_model import MultimodalDRClassifier
from ml.uncertainty.mc_dropout import MCDropoutEstimator

def test_cnn_model_forward(sample_image_tensor):
    model = DRClassifier(num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(sample_image_tensor)
    assert out.shape == (2, 5)

def test_cnn_model_save_load(sample_image_tensor):
    model = DRClassifier(num_classes=5, pretrained=False)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        model.eval()
        save_model(model, tmp_path, epoch=1, metrics={"acc": 0.85})
        loaded = load_model(tmp_path, num_classes=5, device="cpu")
        loaded.eval()
        with torch.no_grad():
            orig_out = model(sample_image_tensor)
            loaded_out = loaded(sample_image_tensor)
        torch.testing.assert_close(orig_out, loaded_out)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_ehr_encoder_forward():
    encoder = EHREncoder(input_dim=17, hidden_dims=[128, 64], output_dim=64)
    x = torch.randn(4, 17)
    out = encoder(x)
    assert out.shape == (4, 64)

def test_multimodal_forward_all_modes(sample_image_tensor):
    batch_size = sample_image_tensor.shape[0]
    ehr_tensor = torch.randn(batch_size, 17)
    
    # 1. Multimodal mode
    model = MultimodalDRClassifier(num_classes=5, ehr_input_dim=17, mode="multimodal")
    model.eval()
    with torch.no_grad():
        res = model(image=sample_image_tensor, ehr=ehr_tensor)
    assert res["logits"].shape == (batch_size, 5)
    assert res["image_features"] is not None
    assert res["ehr_features"] is not None
    
    # 2. Image-only mode
    model.set_mode("image_only")
    with torch.no_grad():
        res_img = model(image=sample_image_tensor)
    assert res_img["logits"].shape == (batch_size, 5)
    
    # 3. EHR-only mode
    model.set_mode("ehr_only")
    with torch.no_grad():
        res_ehr = model(ehr=ehr_tensor)
    assert res_ehr["logits"].shape == (batch_size, 5)

def test_mc_dropout_uncertainty(sample_image_tensor):
    model = MultimodalDRClassifier(num_classes=5, ehr_input_dim=17, mode="image_only")
    estimator = MCDropoutEstimator(model, n_passes=5, device=torch.device("cpu"))
    
    res = estimator.predict_with_uncertainty(image_tensor=sample_image_tensor[0:1])
    assert "mean_proba" in res
    assert "std_proba" in res
    assert "predicted_class" in res
    assert len(res["mean_proba"]) == 5
    assert len(res["std_proba"]) == 5
    assert 0 <= res["predicted_class"] < 5
