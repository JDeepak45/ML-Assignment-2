"""
Train 6 classification models on the real UCI Wine Quality dataset
(Cortez et al., 2009 - combined red + white samples), evaluate them
(multi-class metrics), and persist artifacts for the Streamlit app.

Run:
    python model/build_dataset.py   # creates wine_quality_full.csv from data_raw/
    python model/train_models.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(MODEL_DIR / "wine_quality_full.csv")
feature_names = [c for c in df.columns if c != "quality_class"]
X = df[feature_names]

le = LabelEncoder()
y = le.fit_transform(df["quality_class"])
class_names = list(le.classes_)
print("Classes:", class_names, "-> encoded as", list(range(len(class_names))))

with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW test data (features + true label, human-readable class name)
# as the CSV used for the assignment's "test data" requirement / Streamlit
# upload demo.
test_df = X_test.copy()
test_df["quality_class"] = le.inverse_transform(y_test)
test_df.to_csv(BASE_DIR / "test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# ---------------------------------------------------------------------------
# 3. Scale features (fit on train only) -- needed for LR/kNN/SVM
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---------------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE
    ),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

SCALED_MODELS = {"Logistic Regression", "kNN", "SVM"}

results = []

for name, model in models.items():
    if name in SCALED_MODELS:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(MODEL_DIR / f"{fname}.pkl", "wb") as f:
        pickle.dump(model, f)

# ---------------------------------------------------------------------------
# 5. Save metrics table + feature/class names for the app
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)

with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f)

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

print("\n=== Final comparison table (weighted avg for multi-class Precision/Recall/F1) ===")
print(results_df.to_string(index=False))
print("\nAll models and artifacts saved in:", MODEL_DIR)
