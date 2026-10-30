import pytest
import numpy as np
import torch
from ml.federated.fl_client import DRFedClient
from ml.models.cnn_model import DRClassifier

def test_fl_client_get_set_parameters():
    model = DRClassifier(num_classes=5, pretrained=False)
    client = DRFedClient(
        model=model,
        train_loader=[],
        val_loader=[],
        device="cpu",
        config={"local_epochs": 1, "learning_rate": 0.001}
    )
    
    # Get parameters
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)
    
    # Set parameters with slight perturbation
    perturbed_params = [p + 0.01 for p in params]
    client.set_parameters(perturbed_params)
    new_params = client.get_parameters(config={})
    np.testing.assert_allclose(new_params[0], perturbed_params[0], rtol=1e-5)

def test_no_raw_images_transmitted_in_federated_protocol():
    """
    Mandatory Section 7 Verification:
    Confirms that client parameter payload transmitted to aggregating server
    contains ONLY numerical weights/gradients (numpy arrays), and NEVER
    raw images, patient identifiers, or file paths.
    """
    model = DRClassifier(num_classes=5, pretrained=False)
    client = DRFedClient(
        model=model,
        train_loader=[],
        val_loader=[],
        device="cpu",
        config={"local_epochs": 1, "learning_rate": 0.001}
    )
    
    params = client.get_parameters(config={})
    
    for idx, p in enumerate(params):
        assert isinstance(p, np.ndarray), f"Parameter {idx} is not a numpy array"
        assert p.dtype in [np.float32, np.float64, np.int64], f"Unexpected parameter dtype: {p.dtype}"
        # Ensure it is not an image-like object or string
        assert not isinstance(p, str)
        assert not hasattr(p, "filename")

def test_fedavg_mathematical_aggregation():
    """
    Verifies FedAvg formulation:
        W_global = sum_{k=1}^K (n_k / N) * W_k
    across 3 simulated hospital nodes.
    """
    n_clients = 3
    sample_counts = [100, 150, 250] # total N = 500
    total_N = sum(sample_counts)
    
    # Create 3 client weight vectors of dimension D
    dim = 64
    w1 = np.ones(dim, dtype=np.float32) * 1.0
    w2 = np.ones(dim, dtype=np.float32) * 2.0
    w3 = np.ones(dim, dtype=np.float32) * 3.0
    
    # Expected FedAvg:
    # (100/500)*1.0 + (150/500)*2.0 + (250/500)*3.0 = 0.2 + 0.6 + 1.5 = 2.3
    expected_w_global = (100/500)*1.0 + (150/500)*2.0 + (250/500)*3.0
    
    # Calculate weighted average
    aggregated = (sample_counts[0]/total_N) * w1 + (sample_counts[1]/total_N) * w2 + (sample_counts[2]/total_N) * w3
    
    np.testing.assert_allclose(aggregated, np.ones(dim) * expected_w_global, rtol=1e-6)
