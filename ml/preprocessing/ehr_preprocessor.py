import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Medical Disclaimer: This software is for educational purposes only and is not intended
# for use in the diagnosis, treatment, mitigation, or prevention of disease.

EHR_FEATURES = [
    'age', 'gender', 'hba1c', 'sbp', 'dbp', 'cholesterol', 'duration_diabetes_years',
    'insulin_dependent', 'smoker', 'bmi', 'creatinine', 'visual_acuity_left', 'visual_acuity_right'
]

ALIAS_MAP = {
    'sex': 'gender',
    'systolic_bp': 'sbp',
    'diastolic_bp': 'dbp',
    'diabetes_duration_years': 'duration_diabetes_years',
    'diabetes_duration': 'duration_diabetes_years',
    'insulin_use': 'insulin_dependent',
    'smoking_status': 'smoker',
    'cholesterol_ldl': 'cholesterol'
}

def normalize_ehr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names and categorical values."""
    res = df.copy()
    for old_c, new_c in ALIAS_MAP.items():
        if old_c in res.columns and new_c not in res.columns:
            res[new_c] = res[old_c]

    # Convert blood_glucose to hba1c if hba1c is missing
    if 'hba1c' not in res.columns or res['hba1c'].isna().all():
        if 'blood_glucose' in res.columns:
            # Formula: eAG (mg/dL) = 28.7 * HbA1c - 46.7  => HbA1c = (eAG + 46.7) / 28.7
            res['hba1c'] = (pd.to_numeric(res['blood_glucose'], errors='coerce') + 46.7) / 28.7

    # Normalize categorical values
    if 'gender' in res.columns:
        res['gender'] = res['gender'].apply(
            lambda x: 'M' if str(x).upper() in ['M', 'MALE', '1'] else ('F' if str(x).upper() in ['F', 'FEMALE', '0'] else 'M')
        )
    if 'insulin_dependent' in res.columns:
        res['insulin_dependent'] = res['insulin_dependent'].apply(
            lambda x: 1 if str(x).lower() in ['1', 'true', 'yes', '1.0'] else 0
        )
    if 'smoker' in res.columns:
        res['smoker'] = res['smoker'].apply(
            lambda x: 'current' if str(x).lower() in ['current', '2'] else ('former' if str(x).lower() in ['former', '1'] else 'never')
        )
    return res

class EHRPreprocessor:
    def __init__(self):
        self.num_features = [
            'age', 'hba1c', 'sbp', 'dbp', 'cholesterol',
            'duration_diabetes_years', 'bmi', 'creatinine',
            'visual_acuity_left', 'visual_acuity_right'
        ]
        self.cat_features = ['gender', 'insulin_dependent', 'smoker']
        
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        # Define fixed categories to guarantee exactly 17 output dimensions: 10 + 2 + 2 + 3 = 17
        self.encoder = OneHotEncoder(
            categories=[['F', 'M'], [0, 1], ['never', 'former', 'current']],
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.is_fitted = False
        
    def fit(self, df):
        df_norm = normalize_ehr_df(df)
        for col in EHR_FEATURES:
            if col not in df_norm.columns:
                df_norm[col] = np.nan

        # Provide fallback defaults if column is entirely NaN
        fallbacks = {
            'age': 55.0, 'hba1c': 7.5, 'sbp': 125.0, 'dbp': 75.0,
            'cholesterol': 190.0, 'duration_diabetes_years': 8.0,
            'bmi': 27.5, 'creatinine': 1.0, 'visual_acuity_left': 0.8,
            'visual_acuity_right': 0.8, 'gender': 'M',
            'insulin_dependent': 0, 'smoker': 'never'
        }
        for col, val in fallbacks.items():
            if df_norm[col].isna().all():
                df_norm[col] = val

        self.num_imputer.fit(df_norm[self.num_features])
        self.cat_imputer.fit(df_norm[self.cat_features])
        
        num_imputed = self.num_imputer.transform(df_norm[self.num_features])
        cat_imputed = self.cat_imputer.transform(df_norm[self.cat_features])
        
        self.scaler.fit(num_imputed)
        self.encoder.fit(cat_imputed)
        
        self.is_fitted = True
        return self
        
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
            
        df_norm = normalize_ehr_df(df)
        for col in EHR_FEATURES:
            if col not in df_norm.columns:
                df_norm[col] = np.nan
                
        num_imputed = self.num_imputer.transform(df_norm[self.num_features])
        cat_imputed = self.cat_imputer.transform(df_norm[self.cat_features])
        
        num_scaled = self.scaler.transform(num_imputed)
        cat_encoded = self.encoder.transform(cat_imputed)
        
        return np.hstack((num_scaled, cat_encoded))
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def preprocess_dict(self, record_dict: dict) -> np.ndarray:
        """Preprocesses a single patient clinical record dictionary into a (1, 17) array."""
        df = pd.DataFrame([record_dict])
        return self.transform(df)
        
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path):
        return joblib.load(path)
        
    def get_feature_names(self):
        cat_names = self.encoder.get_feature_names_out(self.cat_features)
        return self.num_features + list(cat_names)

def generate_synthetic_ehr(n_samples=500, seed=42):
    """Generates realistic synthetic EHR records."""
    np.random.seed(seed)
    
    data = {
        'age': np.random.normal(58, 12, n_samples).clip(18, 90).astype(int),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'hba1c': np.random.normal(7.6, 1.4, n_samples).clip(4.5, 14.0).round(1),
        'sbp': np.random.normal(128, 16, n_samples).clip(90, 200).astype(int),
        'dbp': np.random.normal(78, 10, n_samples).clip(55, 120).astype(int),
        'cholesterol': np.random.normal(195, 38, n_samples).clip(100, 320).round(1),
        'duration_diabetes_years': np.random.exponential(9, n_samples).clip(0, 35).round(1),
        'insulin_dependent': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'smoker': np.random.choice(['never', 'former', 'current'], n_samples, p=[0.55, 0.30, 0.15]),
        'bmi': np.random.normal(28.2, 4.8, n_samples).clip(16, 48).round(1),
        'creatinine': np.random.normal(1.05, 0.28, n_samples).clip(0.4, 4.0).round(2),
        'visual_acuity_left': np.random.uniform(0.3, 1.0, n_samples).round(2),
        'visual_acuity_right': np.random.uniform(0.3, 1.0, n_samples).round(2),
        'SYNTHETIC_FLAG': [True] * n_samples
    }
    return pd.DataFrame(data)

def validate_ehr_record(record_dict):
    """Validates single clinical EHR record with clinical range checks."""
    if not isinstance(record_dict, dict):
        return False, "EHR data must be a valid dictionary."

    # Clinical range checks
    val_map = {
        'age': (18, 110, "Patient age must be between 18 and 110 years."),
        'duration_diabetes_years': (0, 70, "Diabetes duration must be between 0 and 70 years."),
        'diabetes_duration_years': (0, 70, "Diabetes duration must be between 0 and 70 years."),
        'diabetes_duration': (0, 70, "Diabetes duration must be between 0 and 70 years."),
        'hba1c': (3.5, 20.0, "HbA1c level must be between 3.5% and 20.0%."),
        'sbp': (60, 260, "Systolic blood pressure must be between 60 and 260 mmHg."),
        'systolic_bp': (60, 260, "Systolic blood pressure must be between 60 and 260 mmHg."),
        'dbp': (35, 160, "Diastolic blood pressure must be between 35 and 160 mmHg."),
        'diastolic_bp': (35, 160, "Diastolic blood pressure must be between 35 and 160 mmHg."),
        'bmi': (12.0, 75.0, "BMI must be between 12.0 and 75.0 kg/m²."),
        'blood_glucose': (25, 800, "Blood glucose must be between 25 and 800 mg/dL."),
    }

    for key, (lo, hi, msg) in val_map.items():
        if key in record_dict and record_dict[key] not in [None, '']:
            try:
                v = float(record_dict[key])
                if v < lo or v > hi:
                    return False, f"Range Validation Error: {msg} (Provided: {v})"
            except (ValueError, TypeError):
                return False, f"Invalid numeric value for {key}: '{record_dict[key]}'"

    return True, "Valid"
