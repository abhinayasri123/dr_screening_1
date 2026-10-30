#!/usr/bin/env python3
"""
Synthetic EHR Demo Data Generator
===================================
Generates realistic-looking but COMPLETELY SYNTHETIC patient EHR records.

⚠️  WARNING: This data is NOT derived from real patients.
     It exists ONLY for software integration testing and UI demonstration.
     NEVER use it for medical conclusions or present it as real clinical data.

Usage:
    python scripts/generate_synthetic_ehr.py
    python scripts/generate_synthetic_ehr.py --n-samples 1000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Banner ──────────────────────────────────────────────────────────────────

SYNTHETIC_BANNER = """
================================================================================
  SYNTHETIC / DEMO DATA -- NOT FOR CLINICAL USE
  ------------------------------------------------------------------------------
  The records in this file are entirely computer-generated.
  They do NOT represent real patients, real medical histories,
  or real clinical measurements.
  DO NOT use for any medical, research, or statistical purposes.
================================================================================
"""


# ── Feature generation helpers ───────────────────────────────────────────────

def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_synthetic_ehr(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate n_samples synthetic EHR records with medically plausible
    (but entirely fictional) distributions.

    Returns a DataFrame with all 13 feature columns plus metadata.
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ─────────────────────────────────────────────────────
    age = _clamp(rng.normal(loc=55, scale=12, size=n_samples), 18, 90).round(0).astype(int)
    sex = rng.choice(["M", "F"], size=n_samples, p=[0.52, 0.48])

    # Diabetes duration (years) — correlated with age: older → potentially longer
    base_duration = (age - 30) / 5
    diabetes_duration = _clamp(
        rng.exponential(scale=8, size=n_samples) + rng.normal(0, 2, n_samples),
        0, 40
    ).round(1)

    # HbA1c (%) — higher with longer duration / worse control
    hba1c_mean = 7.0 + 0.05 * diabetes_duration + rng.normal(0, 0.8, n_samples)
    hba1c = _clamp(hba1c_mean, 4.0, 14.0).round(1)

    # Blood pressure — higher with age and BMI
    bmi = _clamp(rng.normal(loc=28, scale=5, size=n_samples), 15.0, 50.0).round(1)
    systolic_bp = _clamp(
        110 + 0.4 * (age - 40) + 0.5 * (bmi - 25) + rng.normal(0, 8, n_samples),
        90, 200
    ).round(0).astype(int)
    diastolic_bp = _clamp(
        70 + 0.15 * (age - 40) + 0.3 * (bmi - 25) + rng.normal(0, 6, n_samples),
        60, 130
    ).round(0).astype(int)

    # Smoking
    smoking_status = rng.choice(
        ["never", "former", "current"],
        size=n_samples, p=[0.50, 0.30, 0.20]
    )

    # Insulin use — more likely with longer duration
    insulin_prob = np.clip(0.2 + 0.015 * diabetes_duration, 0, 0.85)
    insulin_use = (rng.random(n_samples) < insulin_prob).astype(int)

    # Previous laser treatment — correlated with higher HbA1c
    laser_prob = np.clip(0.05 + 0.04 * (hba1c - 7.0), 0, 0.6)
    prev_laser_treatment = (rng.random(n_samples) < laser_prob).astype(int)

    # Complications — correlated with duration
    nephropathy_prob = np.clip(0.05 + 0.008 * diabetes_duration, 0, 0.5)
    nephropathy = (rng.random(n_samples) < nephropathy_prob).astype(int)

    neuropathy_prob = np.clip(0.08 + 0.01 * diabetes_duration, 0, 0.6)
    neuropathy = (rng.random(n_samples) < neuropathy_prob).astype(int)

    # Cholesterol LDL (mg/dL)
    cholesterol_ldl = _clamp(
        rng.normal(loc=110, scale=30, size=n_samples),
        50, 250
    ).round(1)

    # ── Synthetic patient IDs ─────────────────────────────────────────────
    patient_ids = [f"SYNTH-{i+1:05d}" for i in range(n_samples)]

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "patient_id":              patient_ids,
        "age":                     age,
        "sex":                     sex,
        "diabetes_duration_years": diabetes_duration,
        "hba1c":                   hba1c,
        "systolic_bp":             systolic_bp,
        "diastolic_bp":            diastolic_bp,
        "bmi":                     bmi,
        "smoking_status":          smoking_status,
        "insulin_use":             insulin_use,
        "prev_laser_treatment":    prev_laser_treatment,
        "nephropathy":             nephropathy,
        "neuropathy":              neuropathy,
        "cholesterol_ldl":         cholesterol_ldl,
        # Metadata
        "is_synthetic":            True,
        "data_warning":            "SYNTHETIC_DATA_DO_NOT_USE_FOR_CLINICAL_PURPOSES",
    })

    return df


def print_stats(df: pd.DataFrame):
    numeric_cols = [
        "age", "diabetes_duration_years", "hba1c",
        "systolic_bp", "diastolic_bp", "bmi", "cholesterol_ldl"
    ]
    print("\nSummary statistics (numeric features):")
    print(df[numeric_cols].describe().round(2).to_string())

    print("\nCategorical features:")
    for col in ["sex", "smoking_status"]:
        print(f"\n  {col}:")
        print(df[col].value_counts(normalize=True).mul(100).round(1)
              .to_string())

    binary_cols = ["insulin_use", "prev_laser_treatment", "nephropathy", "neuropathy"]
    print("\nBinary feature prevalence (%):")
    for col in binary_cols:
        pct = df[col].mean() * 100
        print(f"  {col}: {pct:.1f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo EHR records for the DR Screening project"
    )
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic records to generate (default: 500)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output CSV path (default: data/ehr/synthetic_ehr_demo.csv)")
    args = parser.parse_args()

    # Default output path relative to project root
    script_dir  = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    output_path = (
        Path(args.output) if args.output
        else project_dir / "data" / "ehr" / "synthetic_ehr_demo.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(SYNTHETIC_BANNER)
    print(f"Generating {args.n_samples} synthetic EHR records (seed={args.seed}) ...")

    df = generate_synthetic_ehr(n_samples=args.n_samples, seed=args.seed)

    print_stats(df)

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(df)} records to: {output_path}")
    print(SYNTHETIC_BANNER)


if __name__ == "__main__":
    main()
