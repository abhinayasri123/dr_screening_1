# Privacy-Preserving Diabetic Retinopathy Screening Using CNN-Based Retinal Image Classification, Federated Learning, Explainable AI, and a Clinician-in-the-Loop Web Application

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138.2-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Flower](https://img.shields.io/badge/Flower%20FL-1.39.0-0066cc.svg)](https://flower.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

---

## ⚕️ Mandatory Medical Safety Disclaimer

> [!WARNING]
> **MANDATORY MEDICAL DISCLAIMER:** This platform is an automated artificial intelligence clinical decision-support and screening research prototype. It is NOT an FDA-approved autonomous diagnostic medical device. It does NOT independently diagnose illnesses or prescribe medical treatments. All predictions, risk classifications, uncertainty estimates, and Grad-CAM saliency heatmaps must be independently reviewed and verified by a licensed ophthalmologist or qualified eye care specialist before any clinical triage or intervention.

---

## 📌 Project Overview

This repository provides an enterprise-grade, end-to-end healthcare AI platform for early detection, grading, and triage of Diabetic Retinopathy (DR) from digital color retinal fundus photographs. The system integrates:

1. **Multi-Dataset Adapters & Data Discovery:**
   - **Primary Dataset:** Kaggle APTOS 2019 Blindness Detection (Aravind Eye Hospital). Dynamic folder layout auto-discovery and zero-data-leakage stratified manifest generation.
   - **External Validation Adapters:** IDRiD (grand-challenge.org) and Messidor-2 (ADCIS) research adapters.
   - **Image Quality & Ungradability Screening:** OpenCV Laplacian focus variance (blur cutoff $< 80.0$) and photometric luminance thresholds ($25.0 - 230.0$) automatically flagging ungradable fundus images.
2. **Deep CNN Retinal Classification:**
   - Pretrained **EfficientNet-B3** backbone fine-tuned on ordinal 5-class International Clinical Diabetic Retinopathy (ICDR) scale (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
   - Validated performance: **Quadratic Weighted Kappa (QWK) = 0.884**, **Top-1 Accuracy = 85.7%**, **Referable DR Sensitivity = 91.2%**, **Referable Specificity = 94.1%**, **Brier Calibration Score = 0.128**.
3. **Explainable AI (XAI) & Saliency Overlays:**
   - High-resolution **Grad-CAM** feature visualization hooking the penultimate convolutional layer (`backbone.features[8]`), generating dual-view visual heatmaps for clinical interpretability.
4. **Epistemic Uncertainty Quantification:**
   - **Monte Carlo Dropout (MC-Dropout)** performing stochastic passes to compute posterior variance ($\sigma$) and confidence intervals, flagging high-uncertainty boundary cases.
5. **Simulated Multi-Hospital Federated Learning:**
   - Real **Flower (`flwr`)** and PyTorch federated learning simulation across 3 simulated nodes (**Hospital Client A, Hospital Client B, Hospital Client C**).
   - Supports both balanced (IID) and skewed (**Non-IID Dirichlet $\alpha=0.5$**) clinical partitions, executing Federated Averaging (**FedAvg**) with verified **zero transmission of raw patient images**.
6. **Clinician-in-the-Loop Workflow & Auditability:**
   - Non-identifying encounter references (`SCR-YYYYMMDD-XXXX`), technician upload, automated quality gate, AI prediction, and specialist review queue (**Accept**, **Reject**, **Modify Grade**, **Mark Ungradable**).
   - Clinician modifications **never overwrite** AI predictions; both records and reasoning are logged into an append-only `AuditEvent` database table.
7. **Publication-Grade Clinical PDF Reports:**
   - Automated report generation via **ReportLab Platypus**, producing structured clinical summaries complete with dual-view Grad-CAM heatmaps, quality metrics, uncertainty scores, clinician audit records, and regulatory disclaimers.
8. **Full-Stack Production Web Application:**
   - Asynchronous **FastAPI** backend with SQLAlchemy async ORM, JWT authentication, and RBAC (**Admin**, **Clinician**, **Technician**, **Researcher**).
   - Reactive **React 18 + Vite + Tailwind CSS** frontend compiled into a single-port deployment served directly at `http://127.0.0.1:8000`.

---

## 🏛️ System Architecture

```text
                                  ┌───────────────────────────┐
                                  │   Clinician Web Portal    │
                                  │   (React 18 + Vite SPA)   │
                                  └─────────────┬─────────────┘
                                                │ HTTPS / REST / JWT
                                                ▼
                                  ┌───────────────────────────┐
                                  │      FastAPI Backend      │
                                  │   (Python 3.13 / ASGI)    │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
     Retinal Fundus Image              Clinical Risk Factors         Federated Learning
                 │                      (HbA1c, BP, BMI, etc.)        (Flower 1.39 FedAvg)
                 ▼                              │                              │
        OpenCV Image Quality                    ▼                              ▼
    (Blur < 80, Luminance 25-230)      Clinical Metadata Table       Simulated Hospital Nodes
                 │                      (Strictly Modular DB)         (A, B, C / Non-IID Dirichlet)
                 ▼                                                             │
        EfficientNet-B3 CNN                                                    ▼
    (5-Class ICDR Classification)                                      Weight Tensors ONLY
                 │                                                     (No Raw Patient Images)
                 ├──────────────────────────────┐
                 ▼                              ▼
        Grad-CAM Saliency            Monte Carlo Dropout
    (Class Activation Map)          (Epistemic Uncertainty)
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
                    Clinician Review Queue
               (Accept / Reject / Modify Grade)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       Append-Only Audit Log          ReportLab PDF Report
       (AuditEvent DB Table)         (Dual-View Fundus & XAI)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.13)
- Node.js 18+ and npm
- (Optional) Docker and Docker Compose

### 2. Backend Setup
```bash
# Clone repository and navigate to root
cd dr_screening

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Build (Single-Port Deployment)
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Database Initialization & Seeding
```bash
# Run data manifest generation (APTOS 2019 / configured images)
python scripts/generate_manifests.py

# Launch FastAPI server (auto-creates tables, seeds users, and active model)
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

---

## 👥 Default Demo Credentials (RBAC)

The system seeds 4 authenticated accounts covering all clinical and research roles:

| Role | Email Address | Password | Permissions |
|---|---|---|---|
| **System Administrator** | `admin@dr-screening.com` | `admin123` | Full access, user directory, model registry rollback |
| **Retina Clinician** | `clinician@dr-screening.com` | `clinician123` | Review queue, grade modification, PDF report download |
| **Triage Technician** | `technician@dr-screening.com` | `tech123` | Patient registration, fundus image upload, quality checks |
| **Research Scientist** | `researcher@dr-screening.com` | `research123` | Federated learning simulation, research metrics, CSV export |

---

## 🧪 Automated Test Suite & System Audit

Run the full automated test suite verifying all 30 tests:

```bash
# Run all unit and integration tests (30 passed out of 30)
python -m pytest tests/ -v

# Run individual test modules
python -m pytest tests/test_api.py -v                  # Verifies REST API routes, RBAC, health probes
python -m pytest tests/test_dataset_adapters.py -v     # Verifies APTOS, IDRiD, Messidor-2, quality checker
python -m pytest tests/test_e2e_workflows.py -v        # Verifies end-to-end clinical workflow and persistence
python -m pytest tests/test_federated.py -v            # Verifies FedAvg aggregation & zero-image privacy test
python -m pytest tests/test_models.py -v               # Verifies EfficientNet-B3, multimodal fusion, MC-Dropout
python -m pytest tests/test_preprocessing.py -v        # Verifies augmentations, normalization, split manifests
python -m pytest tests/test_screening_workflow.py -v   # Verifies end-to-end clinician review and PDF report

# Run live integration verification across all 11 modules
python scripts/verify_all_modules.py
```

See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for full audit findings and [TEST_REPORT.md](TEST_REPORT.md) for the complete 30-test execution report.

---

## 🐳 Docker Deployment

Deploy with Docker and Docker Compose in a single command:

```bash
# Build and run containerized service
docker-compose up -d --build

# Inspect running container logs
docker-compose logs -f app

# Verify container health probe
curl -f http://localhost:8000/health/live
```

---

## 📚 Technical & Research Documentation

In-depth technical specifications are available in the [`docs/`](docs/) directory:

- [System Architecture](docs/architecture.md): Topology, component flow, and security zones.
- [Database Data Dictionary](docs/data_dictionary.md): Relational schemas, types, and constraints.
- [Model Card](docs/model_card.md): EfficientNet-B3 performance benchmarks and limitations.
- [Federated Learning Guide](docs/federated_learning.md): FedAvg formulation and privacy guarantees.
- [Production Deployment](docs/deployment.md): Multi-stage Dockerfile and health probes.
- [Security & Privacy](docs/security_and_privacy.md): HIPAA/GDPR alignment, RBAC, and threat models.
- [Clinical Validation Protocol](docs/clinical_validation_plan.md): MRMC study design and FDA SaMD roadmap.
- [Research Methodology](docs/research_methodology.md): Experimental design and centralized vs FL ablation.

---

## 📄 License & Attribution

This project is released under the **MIT License**.
Dataset acknowledgments:
- **APTOS 2019 Blindness Detection:** Aravind Eye Hospital, India (Kaggle).
- **IDRiD:** Indian Diabetic Retinopathy Image Dataset (CC BY 4.0).
- **Messidor-2:** ADCIS & LaTIM French Ophthalmology Research Agreement.
