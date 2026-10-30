# Security, Privacy & Compliance Architecture

This document describes the security controls, data protection mechanisms, threat modeling, and regulatory alignment (HIPAA, GDPR) implemented throughout the platform.

---

## 1. Regulatory Alignment (HIPAA & GDPR)

### HIPAA Security Rule Compliance (§ 164.312)
- **Technical Safeguards:**
  - **Access Control (§ 164.312(a)(1)):** Unique user identification, mandatory JWT bearer token authentication, and role-based access restrictions.
  - **Emergency Access (§ 164.312(a)(2)(ii)):** Designated system administrator roles with audited override capabilities.
  - **Transmission Security (§ 164.312(e)(1)):** Enforced TLS 1.3 encryption across all network boundaries.
  - **Audit Controls (§ 164.312(b)):** Immutable, append-only `audit_events` table capturing all read/write mutations, image uploads, AI inferences, and clinician reviews.

### GDPR Technical Measures (Article 32)
- **Pseudonymization (Article 32(1)(a)):** Patient records are identified in screening encounters via synthetic case numbers (`SCR-YYYYMMDD-XXXX`). Retinal fundus photographs do not embed patient metadata in EXIF headers.
- **Data Minimization (Article 5(1)(c)):** Clinical metadata (HbA1c, BP, BMI) is optional and strictly decoupled from image feature vectors.

---

## 2. Authentication & Role-Based Access Control (RBAC)

The system enforces four distinct operational roles with explicit permission boundaries:

| Role | Permissions | Endpoints Authorized |
|---|---|---|
| **Administrator** (`admin`) | Full system governance, user provisioning, model activation/rollback, audit trail inspection. | `/api/v1/admin/*`, `/api/v1/models/*` |
| **Clinician** (`clinician`) | Review queue access, acceptance, grade modification, escalation, clinical PDF report download. | `/api/v1/screenings/{id}/review`, `/report/pdf` |
| **Technician** (`technician`) | Patient registration, fundus image upload, automated quality checks. | `/api/v1/patients`, `/api/v1/screenings/{id}/image` |
| **Researcher** (`researcher`) | Federated simulation execution, training run inspection, metrics analysis, dataset manifest exports. | `/api/v1/federated/*`, `/api/v1/metrics/*` |

---

## 3. Cryptographic Controls & Password Storage

- **Password Hashing:** Passwords are never stored in plaintext. They are salted with a 16-byte cryptographically secure pseudorandom token (`secrets.token_hex(16)`) and hashed using SHA-256 / bcrypt with constant-time comparison (`secrets.compare_digest`) to prevent timing side-channel attacks.
- **Session Tokens:** Stateless JSON Web Tokens (JWT) signed with HMAC-SHA256 (`HS256`) using a 256-bit secret key, with automatic 24-hour expiration.
- **Model Integrity Checksums:** Model checkpoint files (`best_cnn_model.pth`) are hashed via SHA-256 upon registry entry; corrupted or tampered files are rejected during service initialization.

---

## 4. Threat Modeling & Failure Mode Mitigations

| Threat | Attack Vector | Mitigation Implemented |
|---|---|---|
| **Adversarial / Corrupt Image Upload** | Malformed TIFF/JPEG or decompression bomb | Strict Pillow verification, size checks (< 50MB), minimum resolution (224x224), OpenCV decoding in sandboxed buffer. |
| **Data Leakage in Splits** | Overlap between train and test patients | Stratified deterministic splitting by image hash with manifest verification (`split_manifest.csv`). |
| **Model Inversion / Privacy Extraction** | Extracting training images from weights | Federated FedAvg transfers only aggregated weights; Differential Privacy (DP) clipping hooks integrated; zero image transmission verified in tests. |
| **Clinician Malpractice / Unchecked AI** | Clinician blindly accepting erroneous AI prediction | Side-by-side presentation of Grad-CAM, MC-Dropout uncertainty score, and mandatory review checklist with non-overwriting audit logging. |
