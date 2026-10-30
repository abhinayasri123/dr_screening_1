import os
import sys
from pathlib import Path

# Prevent network timeouts and font-scan hangs in test runners
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["SKIP_CREATE_TABLES"] = "1"
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)



import pytest
import numpy as np
import pandas as pd
import torch
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def sample_image_tensor():
    return torch.randn(2, 3, 300, 300)

@pytest.fixture
def sample_ehr_dict():
    return {
        "age": 58,
        "gender": "M",
        "hba1c": 8.4,
        "sbp": 140,
        "dbp": 90,
        "cholesterol": 210,
        "duration_diabetes_years": 12.0,
        "insulin_dependent": "yes",
        "smoker": "former",
        "bmi": 28.5,
        "creatinine": 1.1,
        "visual_acuity_left": 0.8,
        "visual_acuity_right": 0.7
    }

@pytest.fixture
def sample_ehr_df():
    data = {
        "age": [45, 60, 52, 70, 38],
        "gender": ["M", "F", "F", "M", "M"],
        "hba1c": [6.5, 9.2, 7.8, 11.0, 5.9],
        "sbp": [120, 150, 135, 160, 115],
        "dbp": [80, 95, 85, 100, 75],
        "cholesterol": [180, 240, 210, 260, 170],
        "duration_diabetes_years": [4.0, 15.0, 8.0, 22.0, 2.0],
        "insulin_dependent": ["no", "yes", "yes", "yes", "no"],
        "smoker": ["never", "former", "never", "current", "never"],
        "bmi": [24.0, 31.2, 27.8, 33.5, 22.1],
        "creatinine": [0.9, 1.4, 1.0, 1.8, 0.8],
        "visual_acuity_left": [1.0, 0.5, 0.7, 0.3, 1.0],
        "visual_acuity_right": [1.0, 0.6, 0.8, 0.4, 0.9]
    }
    return pd.DataFrame(data)
