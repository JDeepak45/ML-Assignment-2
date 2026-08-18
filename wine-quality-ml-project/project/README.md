# Wine Quality Classification — ML Assignment 2

## a. Problem Statement

Predicting wine quality from physicochemical lab measurements is a classic
multi-class classification problem in the food & beverage industry: given
chemistry readings from a wine sample (acidity, sugar, sulfur dioxide,
alcohol, etc.), predict whether the wine falls into a **low**, **medium**,
or **high** quality tier. This project implements and compares six
classical machine learning classifiers on this task and exposes them
through an interactive Streamlit web application.

## b. Dataset Description

**Dataset:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J., 2009,
*"Modeling wine preferences by data mining from physicochemical
properties"*, Decision Support Systems). Two files are provided by UCI —
`winequality-red.csv` (1599 samples) and `winequality-white.csv` (4898
samples) — both containing physicochemical tests on *Vinho Verde* wine
samples from northern Portugal, with a sensory `quality` score (0–10)
assigned by wine experts.

**Preprocessing (`model/build_dataset.py`):**
- Combined the red and white CSVs into a single dataset and added a
  `wine_type` feature (0 = red, 1 = white) so wine color is available as a
  predictor.
- Bucketed the original numeric `quality` score into a 3-class
  categorical target (`quality_class`) to match the assignment's
  classification requirement: `quality <= 5` → **low**, `quality == 6` →
  **medium**, `quality >= 7` → **high**.

| Property | Value |
|---|---|
| Instances | 6497 (1599 red + 4898 white) |
| Features | 12 (11 physicochemical + wine type) |
| Target | 3-class — `low` (2384) / `medium` (2836) / `high` (1277) |
| Missing values | None |

Features: `fixed_acidity`, `volatile_acidity`, `citric_acid`,
`residual_sugar`, `chlorides`, `free_sulfur_dioxide`,
`total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`,
`wine_type` — 12 features, meeting the assignment's minimum of 12, and
6497 instances, well above the minimum of 500.

The held-out test split (20% of the data, stratified) is saved as
`test_data.csv` in this repository and is the file used for the Streamlit
app's upload/demo feature.

## c. GitHub Repository Link

> `https://github.com/JDeepak45/ML-Assignment-2.git`

## d. Models Used

All six models were trained on the same 80/20 stratified train/test split
of the dataset above (`model/train_models.py`). Logistic Regression, kNN,
and SVM were trained on standardized features (`StandardScaler`, fit on the
training set only); Decision Tree, Naive Bayes, and Random Forest were
trained on the raw features. Precision/Recall/F1 use **weighted averaging**
across the three classes; AUC uses **weighted one-vs-rest**.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5823 | 0.7324 | 0.5873 | 0.5823 | 0.5763 | 0.3268 |
| Decision Tree | 0.6462 | 0.7219 | 0.6488 | 0.6462 | 0.6469 | 0.4484 |
| kNN | 0.5954 | 0.7535 | 0.5977 | 0.5954 | 0.5963 | 0.3631 |
| Naive Bayes | 0.4877 | 0.6570 | 0.5038 | 0.4877 | 0.4886 | 0.2237 |
| Random Forest (Ensemble) | 0.7292 | 0.8826 | 0.7364 | 0.7292 | 0.7290 | 0.5697 |
| SVM | 0.6292 | 0.7711 | 0.6374 | 0.6292 | 0.6203 | 0.4025 |

*(Six models are reported — the five listed in the assignment brief plus SVM
as the sixth, per the assignment text referring to "all 6 ML models.")*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest of the "standard" models — the physicochemical features have non-linear, overlapping relationships with quality (this is a known hard property of this dataset), so a single linear decision boundary struggles to separate the three classes, especially the adjacent low/medium boundary. |
| Decision Tree | Surprisingly competitive on its own (Accuracy 0.646, better than LR/kNN/SVM here) because it can carve out non-linear, axis-aligned splits that match how a few key features (alcohol, volatile acidity, density) interact — but a single tree is still prone to overfitting fine-grained noise in borderline "medium" wines. |
| kNN | Only modest gains over Logistic Regression despite scaling — with 11 correlated chemistry features, local Euclidean distance is noisy, and the heavy class overlap between low/medium wines limits how well nearest-neighbor voting can separate them. |
| Naive Bayes | Clearly the weakest model (Accuracy 0.488, MCC 0.224) — several features are strongly correlated in real wine chemistry (e.g., free vs. total sulfur dioxide, fixed acidity vs. pH), which directly violates the Gaussian independence assumption Naive Bayes relies on. |
| Random Forest (Ensemble) | The clear winner across every metric (Accuracy 0.729, AUC 0.883, MCC 0.570) — averaging many decision trees captures the same non-linear feature interactions as a single tree while sharply reducing overfitting, which matters a lot on this noisy, overlapping-class dataset. |
| SVM | Middle of the pack — the RBF kernel captures some non-linearity but doesn't match Random Forest's ability to handle the mix of correlated, differently-scaled chemistry features and the imbalanced "high" quality class. |
| **Overall Winner for this dataset** | **Random Forest (Ensemble)** — highest accuracy, AUC, precision, recall, F1, and MCC of all six models by a clear margin, making it the best fit for this noisy, real-world wine chemistry dataset. |

## Repository Structure

```
project-folder/
│-- app.py                          # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv                   # held-out test split used in experiments
│-- data_raw/
│   │-- winequality-red.csv         # original UCI file
│   │-- winequality-white.csv       # original UCI file
│-- model/
│   │-- build_dataset.py            # combines red+white, buckets quality -> class
│   │-- wine_quality_full.csv       # combined, preprocessed dataset
│   │-- train_models.py             # trains all 6 models + saves metrics/artifacts
│   │-- *.pkl                       # saved trained models, scaler, label encoder
│   │-- metrics_summary.csv         # comparison table (machine-generated)
│   │-- feature_names.json
│   │-- class_names.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/build_dataset.py      # (re)builds the dataset, only needed once
python model/train_models.py       # (re)trains all models, only needed once
streamlit run app.py
```

**Note on scikit-learn versions:** pickled scikit-learn models are only
guaranteed to load correctly with the *same* scikit-learn version they were
trained with. If you pull this repo onto a machine (or deploy it) with a
different scikit-learn version than what produced the committed `.pkl`
files, `app.py` detects this automatically at startup (it does a quick
sanity `predict_proba` call on each model) and **retrains all models from
scratch in the current environment** before serving the app — no manual
steps needed. This also means the very first run after a fresh
`pip install` may take a little longer while it retrains.

## Streamlit App Features

- CSV upload of test data (with a `quality_class` column for evaluation, or
  without it for prediction-only mode)
- Model selection dropdown across all 6 trained classifiers
- Live display of Accuracy, AUC, Precision, Recall, F1, and MCC on the
  uploaded data (weighted averaging for the 3-class problem)
- Confusion matrix heatmap and full classification report
- Reference comparison table of all 6 models on the original held-out test
  set

## Live App

> `https://by2p2puvuvhecnghrfnngz.streamlit.app/`

## Data Source & Citation

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine
Quality [Dataset]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/186/wine+quality
