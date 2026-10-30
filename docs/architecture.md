# System Architecture & Technical Specifications

**Privacy-Preserving Diabetic Retinopathy Screening Using CNN-Based Retinal Image Classification, Federated Learning, Explainable AI, and a Clinician-in-the-Loop Web Application**

---

## 1. Executive Summary

This platform is a clinical decision-support research application designed to assist ophthalmologists and healthcare networks in early screening and triage of Diabetic Retinopathy (DR). The system integrates a deep convolutional neural network (EfficientNet-B3), explainable AI saliency mapping (Grad-CAM), Monte Carlo Dropout epistemic uncertainty estimation, simulated hospital federated training (Flower FedAvg), non-overwriting clinician review queues, and automated Platypus PDF clinical report generation.

```
+---------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER                                 |
|   React 18 + Vite SPA | Tailwind CSS | Lucide Icons | Responsive Clinician Dashboard   |
+---------------------------------------------------------------------------------------+
                                           |  HTTPS / REST / JSON
+---------------------------------------------------------------------------------------+
|                                APPLICATION API BACKEND                                |
|   FastAPI Async ASGI Server | Python 3.13 | Pydantic v2 Validation | JWT Security     |
|   - /api/v1/screenings (Workflow)          - /api/v1/auth (RBAC Authentication)       |
|   - /api/v1/patients   (PHI / Demographics)- /api/v1/models (Registry & Governance)   |
|   - /api/v1/federated  (FL Simulation)     - /health/live & /health/ready (Probes)    |
+---------------------------------------------------------------------------------------+
            |                                         |                       |
+-------------------------+      +-------------------------------+    +-----------------+
|   MACHINE LEARNING      |      |   PERSISTENCE & AUDIT         |    | REPORTING & XAI |
| EfficientNet-B3 CNN     |      | SQLite / PostgreSQL Async     |    | ReportLab PDF   |
| Grad-CAM Feature Hook   | <--> | SQLAlchemy 2.0 AsyncSession   | <->| OpenCV Quality  |
| MC-Dropout Uncertainty  |      | Immutable AuditEvent Ledger   |    | Saliency Maps   |
| FedAvg Multi-Node Coord |      | Model Version Manifests       |    | CSV Archival    |
+-------------------------+      +-------------------------------+    +-----------------+
```

---

## 2. Component Interactions & Data Flow

1. **Patient Intake & Encounter Creation:**
   - A triage technician registers a patient with a non-identifying pseudonymized case number (`SCR-YYYYMMDD-XXXX`).
   - Demographic information is securely decoupled from image arrays.
   - Optional clinical risk indicators (HbA1c, systolic/diastolic blood pressure, diabetes duration, BMI) are captured in a strictly modular `ClinicalMetadata` record.

2. **Automated Retinal Image Quality Screening:**
   - The user uploads a digital retinal fundus photograph (PNG/JPG).
   - The OpenCV quality assessment engine computes Laplacian focus variance (detecting blur < 80.0), photometric luminance distribution (underexposure < 25.0, overexposure > 230.0), and dimensional integrity (>= 224x224).
   - If an image fails quality gates, the encounter is automatically flagged as `UNGRADABLE`, halting automated inference and dispatching the case for urgent clinical re-take or specialist manual review.

3. **Inference & Epistemic Uncertainty Estimation:**
   - The gradable image is processed via `RetinalImagePreprocessor` using standard ImageNet normalization.
   - The forward pass through EfficientNet-B3 generates five-class logits corresponding to the International Clinical Diabetic Retinopathy (ICDR) scale (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
   - Softmax probabilities are computed alongside Monte Carlo Dropout (stochastic forward passes enabling dropout during evaluation) to quantify epistemic uncertainty ($\sigma$). Predictions with $\sigma > 0.12$ trigger a High Uncertainty warning.

4. **Class Activation Mapping (Grad-CAM):**
   - Gradients of the target predicted class score are hooked back to the penultimate convolutional feature map (`backbone.features.8`).
   - Spatial weights are rectified (ReLU) and bilinearly interpolated over the original image, generating a high-resolution heatmap overlay highlighting ocular features driving the classification.

5. **Clinician-in-the-Loop Review Queue:**
   - An authorized ophthalmologist or retinal specialist reviews the encounter in the dedicated queue.
   - The specialist can **Accept**, **Reject**, **Modify Grade (0-4)**, or mark as **Ungradable**.
   - Crucially, clinician decisions **never overwrite** the AI prediction; both records are preserved side-by-side with timestamps, specialist notes, and reason codes.

6. **Audit Logging & Report Generation:**
   - Every system mutation is recorded in the append-only `AuditEvent` table.
   - A Platypus PDF clinical report is synthesized on demand, displaying institutional header branding, patient/case demographics, quality assessment metrics, AI confidence and MC-Dropout variance, Grad-CAM overlays, clinician sign-off signatures, and prominent regulatory safety disclaimers.

---

## 3. Security Boundaries & Isolation Zones

- **Public Internet / Ingress:** Protected by reverse proxy (Nginx or Cloudflare) enforcing TLS 1.3, rate limiting, and CORS origins.
- **Application Sandbox:** Non-root Docker container running the FastAPI ASGI server with read-only application files and strictly segregated `uploads/` and `reports/` storage volumes.
- **Model Storage:** Checkpoint files (`.pth`) validated against cryptographic SHA-256 checksums before loading to prevent model tampering.
- **Federated Training Network:** Encrypted Flower RPC channels transmitting only aggregated weight tensors (`state_dict`). Hospital client private training images never leave their local virtual environment.
