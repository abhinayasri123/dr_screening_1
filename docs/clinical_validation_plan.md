# Clinical Validation Protocol & Regulatory Roadmap

This document outlines the clinical validation study protocol, endpoint definitions, multi-reader multi-case (MRMC) study design, and FDA Software as a Medical Device (SaMD) regulatory pathway.

---

## 1. Study Objectives & Hypotheses

- **Primary Objective:** To evaluate the diagnostic accuracy, sensitivity, and specificity of the EfficientNet-B3 DR Screening System compared to a consensus panel of fellowship-trained retina specialists.
- **Primary Hypothesis:** The AI-assisted screening system will achieve $\ge 85.0\%$ sensitivity and $\ge 90.0\%$ specificity for identifying referable diabetic retinopathy (ICDR Grade $\ge 2$) on digital retinal fundus photographs.
- **Secondary Objective:** To determine if presenting Grad-CAM heatmaps and Monte Carlo Dropout uncertainty flags reduces clinician review time and inter-reader diagnostic variability.

---

## 2. Study Design: Multi-Reader Multi-Case (MRMC)

- **Design Type:** Retrospective, multi-center, multi-reader multi-case (MRMC) fully crossed diagnostic accuracy trial.
- **Target Sample Size:** 1,200 non-mydriatic 45-degree color fundus photographs acquired across 3 geographically distinct primary care clinics.
- **Reference Standard (Ground Truth):**
  - Panel of three board-certified vitreoretinal specialists reviewing stereoscopic or high-resolution uncompressed digital fundus images.
  - Majority rule concordance for 5-grade ICDR determination.
  - In cases of complete disagreement (three distinct grades), a senior fourth retina specialist adjudicates the final ground truth.
- **Reader Cohort:** 
  - 6 general ophthalmologists
  - 6 comprehensive optometrists
  - Each reader evaluates cases under two conditions separated by a 4-week washout period:
    - **Condition A (Control):** Fundus image alone.
    - **Condition B (Intervention):** Fundus image + AI predicted grade + Grad-CAM heatmap + uncertainty score.

---

## 3. Inclusion & Exclusion Criteria

### Inclusion Criteria
1. Patients aged $\ge 18$ years with diagnosed Type 1 or Type 2 Diabetes Mellitus.
2. Digital color retinal fundus photograph centered on the macula and optic disc (field 1 and field 2).
3. Image captured with an FDA-cleared non-mydriatic fundus camera (e.g., Topcon NW400, Canon CR-2).

### Exclusion Criteria
1. Media opacities preventing clear visualization of retinal vasculature (dense cataract, vitreous hemorrhage, corneal edema) failing automated quality check.
2. Concurrent severe ocular co-pathologies obscuring DR features (e.g., central retinal vein occlusion, wet age-related macular degeneration, cytomegalovirus retinitis).
3. Prior vitrectomy or extensive panretinal photocoagulation (PRP) scars covering $> 50\%$ of posterior pole.

---

## 4. Statistical Analysis Plan & Endpoints

| Endpoint | Target Metric | Statistical Test | Acceptance Criteria |
|---|---|---|---|
| **Primary Sensitivity** | Referable DR (Grades 2-4) | Clopper-Pearson 95% CI | Lower bound $\ge 85.0\%$ |
| **Primary Specificity** | Non-referable DR (Grades 0-1) | Clopper-Pearson 95% CI | Lower bound $\ge 90.0\%$ |
| **Grading Agreement** | 5-class ordinal ICDR scale | Quadratic Weighted Kappa (QWK) | Point estimate $\ge 0.800$ |
| **Probabilistic Calibration** | Model predicted probabilities | Brier Score & ECE | Brier $< 0.150$, ECE $< 0.05$ |
| **Clinician Efficiency** | Time per case (seconds) | Paired Wilcoxon Signed-Rank | Statistically significant reduction ($p < 0.05$) |

---

## 5. Regulatory Classification Roadmap (FDA SaMD)

- **Device Categorization:** Software as a Medical Device (SaMD), Computer-Assisted Triage and Notification Software.
- **FDA Product Code:** QIK (Computer-aided triage and notification software for ophthalmic indications).
- **Classification Regulation:** 21 CFR § 886.1100 (Class II).
- **Predicates:** IDx-DR (De Novo DEN180001) / EyeArt (510(k) K200667).
- **Intended Route:** 510(k) Pre-Market Notification demonstrating substantial equivalence to predicate devices in diagnostic sensitivity, specificity, and human factors validation.
