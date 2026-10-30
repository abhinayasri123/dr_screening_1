# EHR Feature Schema

## IMPORTANT NOTICE

> **The EHR data used in this project is SYNTHETIC/DEMO DATA only.**
> It is NOT derived from real patient records.
> It is used ONLY for software integration testing and UI demonstration.
> It must NEVER be presented as real clinical data or used for medical conclusions.

---

## Schema Definition

| Feature | Type | Range / Categories | Description |
|---------|------|--------------------|-------------|
| `age` | Numeric | 18–90 years | Patient age |
| `sex` | Categorical | M, F | Biological sex |
| `diabetes_duration_years` | Numeric | 0–40 | Years since diabetes diagnosis |
| `hba1c` | Numeric | 4.0–14.0 % | Glycated haemoglobin |
| `systolic_bp` | Numeric | 90–200 mmHg | Systolic blood pressure |
| `diastolic_bp` | Numeric | 60–130 mmHg | Diastolic blood pressure |
| `bmi` | Numeric | 15.0–50.0 | Body mass index |
| `smoking_status` | Categorical | never, former, current | Smoking history |
| `insulin_use` | Binary | 0, 1 | Insulin therapy |
| `prev_laser_treatment` | Binary | 0, 1 | Previous laser photocoagulation |
| `nephropathy` | Binary | 0, 1 | Diabetic kidney disease |
| `neuropathy` | Binary | 0, 1 | Diabetic peripheral neuropathy |
| `cholesterol_ldl` | Numeric | 50–250 mg/dL | LDL cholesterol |

---

## Encoding Notes

- Categorical → one-hot encoded during preprocessing
- Binary → 0/1 already
- Numeric → standardized (z-score) per feature statistics

## Missing Values

- Missing numeric values → imputed with training-set median
- Missing categorical → imputed with training-set mode
- Preprocessing pipeline persisted as `data/ehr/ehr_preprocessor.pkl`
