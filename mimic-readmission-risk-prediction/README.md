# Interpretable 30-Day Hospital Readmission Risk Prediction

**Author:** Samuel Siakpebru  
**Goal:** Predict 30-day hospital readmission risk at discharge to support care-management triage.

## Project Summary
This project builds an end-to-end pipeline using **MIMIC-IV v3.1** (Beth Israel Deaconess Medical Center EHR) to predict whether a patient will be readmitted within 30 days after discharge.  
The model is evaluated with metrics appropriate for class imbalance (especially **PR-AUC**) and produces an interpretable demo output:  
**patient admission → risk probability → risk tier → top reasons (SHAP) → suggested follow-up action (rule-based).**

## Data
- **Dataset:** MIMIC-IV v3.1 (credentialed access via PhysioNet)
- **Unit of prediction:** hospital admission (**hadm_id**) at discharge
- **Label:** readmitted within 30 days of discharge (binary)
- **Split:** patient-level split by **subject_id** (no patient overlap between train/val/test)

> **Note:** The dataset is NOT included in this repo due to PhysioNet data use restrictions.

## Features
- **v1 (structured discharge-time features):** demographics, LOS, ED LOS + missingness, prior admissions + time since last discharge, discharge destination (SNF flag), admin categories (insurance, admission/discharge location, etc.)
- **v2 (+ labs):** core lab panel summaries (e.g., creatinine, sodium, potassium, WBC, hemoglobin, platelets, etc.) + missingness flags

## Models
All models use the same patient-level split and feature set for fair comparison:
- **Logistic Regression** (baseline, interpretable)
- **XGBoost** (main tabular model; trained with early stopping and class imbalance handling)
- **FCNN** (neural baseline on encoded tabular features)

## Key Results (Test Set, threshold=0.50)
- **Winner by PR-AUC:** XGB + Labs (retrain)  
- Calibration improved with Platt scaling (better probability reliability / Brier score)

(Replace with your final numbers if you want a static summary here.)

## Notebook
- `mimic_readmission_risk_prediction.ipynb` — Full pipeline:
  - data loading + cohort creation
  - label construction (30-day readmit)
  - feature engineering (v1 + labs)
  - training + evaluation (LogReg, XGB, FCNN)
  - calibration (raw vs Platt vs isotonic)
  - risk tiers + demo table
  - SHAP explanations + fairness slices (sex/age)

## How to Run
1. Obtain PhysioNet credentialed access to MIMIC-IV v3.1.
2. Update paths in the notebook to your local/Colab Drive extraction directory.
3. Run the notebook top-to-bottom.

## Disclaimer
This is a project for decision-support exploration only. Results are not clinical guidance and are not validated for deployment.
