"""
Streamlit app: Wine Quality Classification Demo (3-class)
Upload test CSV data, pick a trained model, see evaluation metrics,
confusion matrix and classification report.
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(page_title="Wine Quality Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "SVM": "svm.pkl",
}
SCALED_MODELS = {"Logistic Regression", "kNN", "SVM"}

REQUIRED_ARTIFACTS = [
    "scaler.pkl", "label_encoder.pkl", "feature_names.json",
    "class_names.json", "metrics_summary.csv",
] + list(MODEL_FILES.values())


def _artifacts_present() -> bool:
    return all((MODEL_DIR / f).exists() for f in REQUIRED_ARTIFACTS)


def _artifacts_loadable() -> bool:
    """Sanity-check that pickled sklearn models actually load and run in
    THIS environment's sklearn version (catches cross-version pickle
    incompatibilities like the classic LogisticRegression 'multi_class'
    AttributeError)."""
    try:
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "feature_names.json") as f:
            feat_names = json.load(f)
        dummy = pd.DataFrame(np.zeros((1, len(feat_names))), columns=feat_names)
        dummy_scaled = scaler.transform(dummy)
        for fname in MODEL_FILES.values():
            with open(MODEL_DIR / fname, "rb") as f:
                model = pickle.load(f)
            model.predict_proba(dummy_scaled)  # exercises the failure path
        return True
    except Exception:
        return False


def _ensure_trained_models():
    """Build the dataset and (re)train all models in THIS environment if
    artifacts are missing or were pickled with an incompatible sklearn
    version. Keeps the app self-healing across machines/deployments."""
    if _artifacts_present() and _artifacts_loadable():
        return
    with st.spinner(
        "Preparing models for this environment (first run, or a "
        "scikit-learn version mismatch was detected) — this takes a "
        "minute..."
    ):
        subprocess.run(
            [sys.executable, str(BASE_DIR / "model" / "build_dataset.py")],
            check=True, cwd=str(BASE_DIR),
        )
        subprocess.run(
            [sys.executable, str(BASE_DIR / "model" / "train_models.py")],
            check=True, cwd=str(BASE_DIR),
        )


_ensure_trained_models()


@st.cache_resource
def load_scaler():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_label_encoder():
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(name):
    with open(MODEL_DIR / MODEL_FILES[name], "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


@st.cache_data
def load_class_names():
    with open(MODEL_DIR / "class_names.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics_summary():
    return pd.read_csv(MODEL_DIR / "metrics_summary.csv")


st.title("🍷 Wine Quality Classification — Model Comparison App")
st.caption(
    "Machine Learning Assignment 2 · Dataset: UCI Wine Quality "
    "(red + white Vinho Verde samples) · 12 features · 6497 instances · "
    "3-class classification (low / medium / high quality)"
)

# --- Sidebar: data upload + model selection --------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV, must include a 'quality_class' column)", type=["csv"]
)

model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: Use the provided `test_data.csv` from the repository, which contains "
    "the held-out test split for this dataset."
)

feature_names = load_feature_names()
class_names = load_class_names()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded — showing results on the bundled `test_data.csv`.")
    df = pd.read_csv(BASE_DIR / "test_data.csv")

# --- Validate columns --------------------------------------------------------
missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

has_target = "quality_class" in df.columns

st.subheader("📄 Data preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

X = df[feature_names]
scaler = load_scaler()
le = load_label_encoder()
model = load_model(model_name)

if model_name in SCALED_MODELS:
    X_input = scaler.transform(X)
else:
    X_input = X

y_pred_enc = model.predict(X_input)
y_prob = model.predict_proba(X_input)
y_pred_labels = le.inverse_transform(y_pred_enc)

st.subheader(f"🧪 Predictions — {model_name}")
pred_df = df.copy()
pred_df["predicted_quality_class"] = y_pred_labels
for i, cname in enumerate(class_names):
    pred_df[f"prob_{cname}"] = np.round(y_prob[:, i], 4)
st.dataframe(pred_df.head(20), use_container_width=True)

# --- Metrics (only if ground-truth target is present) -----------------------
if has_target:
    y_true_enc = le.transform(df["quality_class"])

    st.subheader("📊 Evaluation metrics on uploaded data")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    auc = roc_auc_score(y_true_enc, y_prob, multi_class="ovr", average="weighted")
    prec = precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    rec = recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    f1 = f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true_enc, y_pred_enc)

    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC (weighted OvR)", f"{auc:.4f}")
    col3.metric("Precision (wtd)", f"{prec:.4f}")
    col4.metric("Recall (wtd)", f"{rec:.4f}")
    col5.metric("F1 (wtd)", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(class_names)))
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true_enc, y_pred_enc, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.warning(
        "Uploaded CSV has no 'quality_class' column — predictions are shown "
        "above, but evaluation metrics/confusion matrix require ground-truth labels."
    )

# --- Full model comparison table --------------------------------------------
st.subheader("🏆 All-model comparison (on original held-out test set)")
st.dataframe(load_metrics_summary(), use_container_width=True)
