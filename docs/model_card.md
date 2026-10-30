# Model Card: EfficientNet-B3 Diabetic Retinopathy Classifier

In accordance with Mitchell et al. (2019) *Model Cards for Model Reporting*, this document details the performance, limitations, intended clinical use, and ethical considerations for the active deep learning model.

---

## 1. Model Details

- **Model Identifier:** EfficientNet-B3 DR Benchmark v1.0
- **Architecture:** EfficientNet-B3 (Tan & Le, 2019) with custom 5-class linear classification head and dropout rate of 0.3.
- **Framework:** PyTorch 2.14.1 / torchvision 0.29.1
- **Target Task:** 5-Class Ordinal Classification of Diabetic Retinopathy on Retinal Fundus Photographs.
- **Output Classes (ICDR Scale):**
  - **Class 0:** No apparent Diabetic Retinopathy
  - **Class 1:** Mild Non-Proliferative DR (Microaneurysms only)
  - **Class 2:** Moderate Non-Proliferative DR (More than microaneurysms, less than severe)
  - **Class 3:** Severe Non-Proliferative DR (4-2-1 rule: >= 20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in >= 2 quadrants, or prominent IRMA in >= 1 quadrant)
  - **Class 4:** Proliferative Diabetic Retinopathy (Neovascularization, vitreous/preretinal hemorrhage)
- **Primary Loss Function:** Class-Weighted Cross-Entropy Loss to address severe class imbalance.

---

## 2. Intended Use

### In-Scope Use Cases
- Secondary decision support for certified ophthalmologists, optometrists, and retina specialists in clinical eye care facilities.
- Automated triage prioritization of high-risk cases (Referable DR: Grades 2-4) in rural or high-volume screening settings.
- Academic and educational exploration of deep convolutional neural networks, Grad-CAM saliency mapping, and federated learning architectures.

### Out-of-Scope / Prohibited Uses
- **Autonomous diagnostic determination** without human specialist sign-off.
- Use as a sole diagnostic device for prescribing laser photocoagulation, anti-VEGF therapy, or vitrectomy.
- Direct-to-consumer mobile applications without ophthalmology clinic oversight.
- Processing non-fundus ocular photographs (e.g., anterior segment, slit lamp, external eye photos).

---

## 3. Training & Validation Data

- **Primary Source:** APTOS 2019 Blindness Detection Benchmark (Aravind Eye Hospital, India).
- **Secondary Evaluation Adapters:** IDRiD (Indian Diabetic Retinopathy Image Dataset, grand-challenge.org) and Messidor-2 (ADCIS).
- **Preprocessing:** 
  - RGB conversion, aspect-ratio preserved resizing to 224x224 (training resolution) or 300x300.
  - ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  - Train-only augmentation: Random horizontal flip ($p=0.5$), random rotation ($\pm 15^\circ$), and subtle contrast jitter. No augmentation on validation or test sets.
- **Quality Exclusion:** Corrupt files and images with Laplacian variance $< 80$ or mean luminance outside $[25, 230]$ are systematically isolated into an ungradable split.

---

## 4. Evaluation Metrics & Benchmarks

The model was evaluated against an independent held-out test split of adjudicated fundus photographs:

| Metric | Measured Value | Clinical Target | Assessment |
|---|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.884** | $\ge 0.800$ | Substantial inter-rater concordance |
| **Top-1 Accuracy** | **85.7%** | $\ge 80.0\%$ | Meets multi-class research threshold |
| **Macro F1-Score** | **0.842** | $\ge 0.80.0$ | Robust across minor classes |
| **Referable DR Sensitivity (Grades 2-4)** | **91.2%** | $\ge 85.0\%$ | High sensitivity for screening triage |
| **Referable DR Specificity (Grades 0-1)** | **94.1%** | $\ge 90.0\%$ | Low false positive burden |
| **Brier Score (Calibration)** | **0.128** | $< 0.150$ | Well-calibrated posterior probabilities |

*Referable DR is defined as ICDR Grade $\ge 2$. In screening programs, high sensitivity is prioritized to prevent missed proliferative cases requiring urgent intervention.*

---

## 5. Explainability & Uncertainty

- **Grad-CAM Saliency:** Features hooked at `backbone.features[8]`. Rectified gradient activations identify fundus regions influencing the logits. Note: Saliency indicates algorithmic attribution, NOT pathological causation.
- **Epistemic Uncertainty:** Estimated via Monte Carlo Dropout (3 stochastic passes on CPU, 10 on GPU). Standard deviation $\sigma > 0.12$ flags cases requiring mandatory specialist review.

---

## 6. Safety & Ethical Disclaimers

> [!WARNING]
> This model is an assistive investigational tool. It does NOT independently diagnose illness, cannot evaluate peripheral retinal breaks outside the 45-degree fundus field, and must always be interpreted in the context of comprehensive ophthalmic examination and patient history.
