# Dataset Provenance & Data Management Documentation

## 1. Primary Benchmark Dataset: APTOS 2019 Blindness Detection
- **Original Source**: Kaggle / EyePACS / Aravind Eye Care System (Tamil Nadu, India)
- **Clinical Protocol**: 45-degree field-of-view color retinal fundus imaging acquired by screening technicians in rural diagnostic camps.
- **Grading Standard**: International Clinical Diabetic Retinopathy (ICDR) severity scale graded by multiple independent retina specialists.
- **Reference URL**: [https://www.kaggle.com/competitions/aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **License**: Strictly non-commercial academic research use under APTOS 2019 Competition terms.

## 2. Additional Supported Datasets
### IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Source**: Grand Challenge / IEEE Dataport, 2018.
- **Acquisition**: Kowa VX-10 alpha digital fundus camera, 50-degree FOV, rural Maharashtra clinics.
- **Classes**: 5-grade ICDR DR severity (0-4) + DME risk (0-2).
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### Messidor-2
- **Source**: ADCIS / Messidor Research Consortium, France.
- **Acquisition**: Color retinal fundus imaging across French academic ophthalmology centers.
- **Reference Adjudication**: 5-class consensus panel (Gulshan et al., JAMA 2016).
- **License**: ADCIS Messidor Research Agreement.

## 3. Data Integrity & Leakage Prevention
- **Splits**: Stratified Train (40 samples), Validation (15 samples), and Test (15 samples).
- **Zero Leakage**: No images or patient records overlap between training and evaluation splits.
- **Image Quality**: Evaluated via OpenCV Laplacian focus metric and photometric luminance bounds.
