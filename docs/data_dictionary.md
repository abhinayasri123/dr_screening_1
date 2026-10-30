# Database Data Dictionary & Entity Relationship Specifications

This data dictionary provides the schema definitions, relational constraints, data types, and nullability requirements for the relational database powering the Diabetic Retinopathy Screening platform.

---

## 1. Entity Overview

The schema is divided into four functional domains:
1. **Access Control & Identity:** `users`
2. **Clinical Encounters & Images:** `patients`, `screening_encounters`, `retinal_images`, `clinical_metadata`
3. **Inference, Review & Explainability:** `predictions`, `clinical_reviews`
4. **Governance, Auditing & Federated Learning:** `audit_events`, `model_versions`, `training_runs`, `evaluation_runs`, `federated_experiments`, `federated_rounds`, `federated_clients`

---

## 2. Table Schemas

### `users`
Stores authenticated system users with role-based access control (RBAC).

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `email` | VARCHAR | No | Unique, Indexed | Login email address |
| `hashed_password` | VARCHAR | No | - | Salted SHA-256 / bcrypt hash |
| `full_name` | VARCHAR | No | - | Provider or personnel display name |
| `role` | VARCHAR | No | Default 'clinician' | RBAC: 'admin', 'clinician', 'technician', 'researcher' |
| `is_active` | BOOLEAN | No | Default True | User account active flag |
| `created_at` | TIMESTAMP | No | UTC default | Account creation timestamp |

### `patients`
Pseudonymized patient demographic registry.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `patient_code` | VARCHAR | No | Unique, Indexed | Hospital MRN or pseudonym (`PAT-XXXXXX`) |
| `full_name` | VARCHAR | No | - | Patient legal name |
| `date_of_birth` | DATE | Yes | - | Patient date of birth |
| `sex` | VARCHAR(10) | Yes | - | 'M', 'F', 'Other', 'Unknown' |
| `contact_email` | VARCHAR | Yes | - | Patient contact email address |
| `created_at` | TIMESTAMP | No | UTC default | Patient registration timestamp |

### `screening_encounters`
Represents an individual clinical diabetic retinopathy screening episode.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `case_number` | VARCHAR | No | Unique, Indexed | Encounter reference (`SCR-YYYYMMDD-XXXX`) |
| `patient_id` | VARCHAR(36) | No | FK(`patients.id`) | Relational link to patient |
| `encounter_date` | TIMESTAMP | No | UTC default | Date and time encounter was initiated |
| `status` | VARCHAR | No | Indexed | 'AWAITING_IMAGE', 'AWAITING_PREDICTION', 'AWAITING_REVIEW', 'COMPLETED', 'UNGRADABLE' |
| `notes` | TEXT | Yes | - | Clinical notes from triage technician |
| `created_at` | TIMESTAMP | No | UTC default | Record creation timestamp |

### `retinal_images`
Digital fundus images uploaded for an encounter and their automated quality assessment metrics.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `encounter_id` | VARCHAR(36) | No | FK(`screening_encounters.id`) | Encounter reference |
| `file_path` | VARCHAR | No | - | Storage path on disk or object store |
| `filename` | VARCHAR | No | - | Original uploaded filename |
| `width` | INTEGER | No | - | Image width in pixels |
| `height` | INTEGER | No | - | Image height in pixels |
| `blur_score` | FLOAT | No | - | OpenCV Laplacian variance score |
| `mean_luminance`| FLOAT | No | - | Mean grayscale intensity (0-255) |
| `is_gradable` | BOOLEAN | No | - | True if focus >= 80 and luminance 25-230 |
| `quality_status`| VARCHAR | No | - | 'GRADABLE_OPTIMAL', 'BLURRY_UNGRADABLE', etc. |
| `uploaded_at` | TIMESTAMP | No | UTC default | Upload timestamp |

### `clinical_metadata`
Optional clinical risk factor measurements associated with a screening encounter.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `encounter_id` | VARCHAR(36) | No | FK(`screening_encounters.id`) | Associated screening encounter |
| `patient_id` | VARCHAR(36) | No | FK(`patients.id`) | Associated patient |
| `age` | INTEGER | Yes | - | Patient age in years |
| `diabetes_duration` | FLOAT | Yes | - | Duration of diabetes diagnosis in years |
| `hba1c` | FLOAT | Yes | - | Glycated hemoglobin percentage (%) |
| `systolic_bp` | FLOAT | Yes | - | Systolic blood pressure (mmHg) |
| `diastolic_bp` | FLOAT | Yes | - | Diastolic blood pressure (mmHg) |
| `bmi` | FLOAT | Yes | - | Body mass index (kg/m^2) |
| `creatinine` | FLOAT | Yes | - | Serum creatinine (mg/dL) |
| `medical_history` | TEXT | Yes | - | Comorbidities and current medications |

### `predictions`
Automated AI inference results, confidence, uncertainty, and explainability paths.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `patient_id` | VARCHAR(36) | No | FK(`patients.id`) | Associated patient |
| `encounter_id` | VARCHAR(36) | Yes | FK(`screening_encounters.id`) | Associated encounter |
| `predicted_class` | INTEGER | No | - | Predicted ICDR severity grade (0 to 4) |
| `predicted_label` | VARCHAR | No | - | Human-readable ICDR diagnosis label |
| `confidence` | FLOAT | No | - | Softmax probability for predicted class (0.0 to 1.0) |
| `class_probabilities` | JSON | No | - | Softmax distribution across all 5 classes |
| `uncertainty_std` | FLOAT | Yes | - | Monte Carlo Dropout prediction standard deviation |
| `uncertainty_level` | VARCHAR | Yes | - | 'Low' (< 0.05), 'Medium' (0.05-0.12), 'High' (> 0.12) |
| `gradcam_path` | VARCHAR | Yes | - | Local file path to generated Grad-CAM heatmap |
| `model_version` | VARCHAR | No | - | Identifying string of the active model checkpoint |
| `created_at` | TIMESTAMP | No | UTC default | Inference execution timestamp |

### `clinical_reviews`
Clinician-in-the-loop audit record capturing specialist review, acceptance, modification, or rejection.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `prediction_id` | VARCHAR(36) | No | FK(`predictions.id`) | Referenced AI prediction |
| `reviewer_id` | VARCHAR(36) | No | FK(`users.id`) | Reviewing clinician ID |
| `decision` | VARCHAR | No | - | 'ACCEPTED', 'REJECTED', 'MODIFIED', 'UNGRADABLE' |
| `clinician_grade` | INTEGER | Yes | - | Final verified grade (0 to 4) |
| `final_diagnosis` | VARCHAR | Yes | - | Final diagnostic text |
| `clinician_notes` | TEXT | Yes | - | Detailed findings by retina specialist |
| `requires_followup` | BOOLEAN | No | Default False | Triage flag for specialist referral |
| `followup_urgency` | VARCHAR | Yes | - | 'Routine', 'Priority', 'Urgent' |
| `review_timestamp` | TIMESTAMP | No | UTC default | Review completion timestamp |

### `audit_events`
Append-only immutable system and compliance ledger.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `timestamp` | TIMESTAMP | No | UTC default, Indexed | Event timestamp |
| `user_id` | VARCHAR(36) | Yes | - | User responsible for event |
| `user_email` | VARCHAR | Yes | - | Cached user email address |
| `action` | VARCHAR | No | Indexed | e.g. 'IMAGE_UPLOADED', 'CLINICAL_REVIEW_SUBMITTED' |
| `resource_type` | VARCHAR | No | - | 'screening', 'patient', 'model', 'auth' |
| `resource_id` | VARCHAR(36) | Yes | - | ID of the target resource |
| `details` | JSON | Yes | - | Structured contextual metadata |

### `model_versions`
Registry of trained CNN models with validation metrics and deployment state.

| Column | Type | Nullable | Constraints | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | PK | Primary key UUID |
| `name` | VARCHAR | No | - | Model version name |
| `architecture` | VARCHAR | No | - | e.g. 'EfficientNet-B3' |
| `checkpoint_path` | VARCHAR | No | - | Path to .pth checkpoint file |
| `checksum` | VARCHAR | No | - | Cryptographic SHA-256 checksum |
| `dataset_manifest` | VARCHAR | Yes | - | Path to training manifest CSV |
| `accuracy` | FLOAT | Yes | - | Top-1 accuracy score |
| `quadratic_weighted_kappa` | FLOAT | Yes | - | QWK clinical grading metric |
| `macro_f1` | FLOAT | Yes | - | Unweighted mean F1 score |
| `sensitivity` | FLOAT | Yes | - | Sensitivity at referable DR threshold (>= Grade 2) |
| `specificity` | FLOAT | Yes | - | Specificity at referable DR threshold (<= Grade 1) |
| `brier_score` | FLOAT | Yes | - | Probabilistic calibration metric |
| `metrics_json` | JSON | Yes | - | Full evaluation matrix |
| `is_active` | BOOLEAN | No | Default False | Flag denoting active model for screening |
| `status` | VARCHAR | No | Default 'ACTIVE' | 'ACTIVE', 'ARCHIVED', 'VALIDATING' |
| `created_at` | TIMESTAMP | No | UTC default | Version creation timestamp |
