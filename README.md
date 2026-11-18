# 🎓 Student Academic Risk & Early-Warning System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange.svg)
![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-ml-101.streamlit.app/)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end, leak-free machine learning classification pipeline and interactive Streamlit web dashboard designed to identify students at risk of academic failure (`G3 < 10`) immediately following Period 1 internal examinations (`G1`), providing institutions with a 2-to-3-month window for proactive academic intervention.[cite: 1]

---

## 🚀 Key Engineering Highlights

* **Early-Warning Formulation (Zero Target Leakage):** Late-stage Period 2 marks (`G2`) and final exam scores (`G3`) are strictly withheld from the feature matrix so the model learns genuine behavioral and early-academic risk signals rather than relying on near-final grade proxies.[cite: 1]
* **Leak-Free Scikit-Learn Architecture:** Encapsulates `StandardScaler` (14 numerical/ordinal attributes) and `OneHotEncoder` (17 categorical attributes) inside a `ColumnTransformer` and `Pipeline`, ensuring preprocessing statistics are fit strictly within training folds during 5-Fold Stratified Cross-Validation.[cite: 1]
* **Multi-Subject Cohort Generalization:** Trains, benchmarks, and serializes separate champion pipelines for both **Mathematics** (`student-mat.csv`, N = 395) and **Portuguese Language** (`student-por.csv`, N = 649).[cite: 1]
* **Local & Global Model Explainability:** Extracts global model coefficients (`src/explain.py`) and computes real-time per-student risk attribution (`x_scaled,i · w_i`) directly inside the web interface so academic advisors can see *why* a student was flagged.[cite: 1]
* **Dynamic Threshold Tuning & Automated Testing:** Supports adjustable probability decision boundaries to maximize recall on at-risk students, backed by a parameterized `pytest` verification suite.[cite: 1]

---

## 🛠️ System Architecture

```text
Raw Cohort Telemetry (Mathematics / Portuguese)
      │
      ▼
Stratified 80/20 Train-Test Split
      │
      ▼
ColumnTransformer Preprocessing Pipeline
  ├── StandardScaler (14 Numerical Features)
  └── OneHotEncoder (17 Categorical Features)
      │
      ▼
5-Fold Stratified Cross-Validation
[Logistic Regression | Random Forest | GBDT]
      │
      ▼
Serialized Champion Pipelines (.joblib)
      │
      ▼
Streamlit Dashboard + Per-Student Risk Attribution
```[cite: 2]

---

## 📁 Repository Structure

```text
student-performance-ml/
├── app.py                              # Interactive Streamlit dashboard (multi-subject + explainability)
├── README.md                           # Project documentation & experimental benchmarks
├── requirements.txt                    # Reproducible Python dependencies
├── LICENSE                             # MIT License
│
├── data/
│   └── raw/
│       ├── student-mat.csv             # UCI Mathematics cohort dataset (N = 395)
│       └── student-por.csv             # UCI Portuguese Language cohort dataset (N = 649)
│
├── models/
│   ├── math_pipeline.joblib            # Serialized champion pipeline for Mathematics
│   ├── por_pipeline.joblib             # Serialized champion pipeline for Portuguese
│   └── feature_importance.png          # Global model coefficient bar chart
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Exploratory data analysis & target correlation study
│
├── src/
│   ├── download_data.py                # Dataset ingestion script
│   ├── pipeline.py                     # Leak-free ColumnTransformer builder
│   ├── train.py                        # Multi-subject CV benchmarking & model serialization
│   ├── predict.py                      # CLI inference engine with threshold tuning
│   └── explain.py                      # Global feature coefficient extraction & visualization
│
└── tests/
    └── test_pipeline.py                # Automated pytest suite verifying artifact loading & probability bounds
```[cite: 3]

---

## 📊 Experimental Results (Mathematics Cohort)

### 1. 5-Fold Stratified Cross-Validation

| Candidate Classifier | Mean Macro F1-Score | Fold Std Dev | Mean ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (L2 Regularized)** | **0.8280** | **± 0.0305** | **0.9202** |
| Random Forest (Balanced Weights) | 0.8146 | ± 0.0317 | 0.9104 |
| Gradient Boosting Classifier | 0.8039 | ± 0.0737 | 0.9218 |[cite: 4]

> **Architectural Insight:** Regularized Logistic Regression achieved superior out-of-fold generalization with lower variance across folds compared to higher-capacity tree ensembles on tabular datasets of this sample size (N = 395).[cite: 4]

### 2. Held-Out Test Set Performance (N = 79)

* **ROC-AUC Score:** `0.8803`[cite: 4]
* **Overall Accuracy:** `78.0%`[cite: 4]
* **Macro F1-Score:** `0.76`[cite: 4]

---

## 🔍 Top Predictive Drivers

* **Positive Signals (Increase Odds of Passing):**
  * Period 1 Assessment Score `G1` (`β = +3.33`)[cite: 4]
  * Father's Occupation `Fjob_other` (`β = +0.97`)[cite: 4]
* **Negative Signals (Increase Risk of Failure):**
  * Non-parent Guardian `guardian_other` (`β = -0.86`)[cite: 4]
  * Romantic Relationship `romantic_yes` (`β = -0.77`)[cite: 4]
  * Student Age `age` (`β = -0.55`)[cite: 4]

---

## 💻 Local Setup & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Shravanigs09/student-performance-ml.git](https://github.com/Shravanigs09/student-performance-ml.git)
cd student-performance-ml
```[cite: 4, 5]

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```[cite: 5]

### 3. Train & Serialize Multi-Subject Models
```bash
python src/train.py
```[cite: 5]

### 4. Run Automated Unit Tests
```bash
pytest
```[cite: 5]

### 5. Launch the Web Application
```bash
streamlit run app.py
```[cite: 5]

---

## 📜 License & Dataset Attribution

* **License:** Distributed under the MIT License.[cite: 5]
* **Dataset:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. UCI Machine Learning Repository.[cite: 5]