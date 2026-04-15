"""
VentureVerse – Model Training Pipeline
========================================
Dataset  : Kaggle  → manishkc06/startup-success-prediction  (startup_data.csv)
Source   : Crunchbase (public, pre-2014 US startups)
Target   : status  → 1 = acquired (success), 0 = closed (failure)

Models   : Logistic Regression, Random Forest, XGBoost
Output   : ventureverse_model.joblib   (best model by ROC-AUC)
           model_results_summary.json  (full metrics for report / IPD)

Run      : python train_model.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
DATA_FILE = "startup_data.csv"
MODEL_OUT = "ventureverse_model.joblib"
RESULTS_OUT = "model_results_summary.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the Crunchbase startup CSV and perform initial cleaning."""
    df = pd.read_csv(path, encoding="ISO-8859-1")

    print(f"✅  Loaded: {path}")
    print(f"    Shape : {df.shape}")
    print(f"    Columns: {df.columns.tolist()}\n")

    # ── Target ──────────────────────────────────────────────
    # Keep only acquired (success) and closed (failure)
    if "status" not in df.columns:
        # Some versions use 'labels' (1 = acquired, 0 = closed)
        if "labels" in df.columns:
            df["status"] = df["labels"].map({1: "acquired", 0: "closed"})
        else:
            raise ValueError("❌ No 'status' or 'labels' column found.")

    df = df[df["status"].isin(["acquired", "closed"])].copy()
    df["success"] = (df["status"] == "acquired").astype(int)

    print(f"✅  Target distribution (after filter):")
    print(f"    {df['success'].value_counts().to_dict()}\n")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create / clean features that align with VentureVerse's
    pre-launch indicator narrative.
    """

    # ── Numeric conversions ─────────────────────────────────
    num_cols = [
        "age_first_funding_year",
        "age_last_funding_year",
        "age_first_milestone_year",
        "age_last_milestone_year",
        "relationships",
        "funding_rounds",
        "funding_total_usd",
        "milestones",
        "avg_participants",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Derived features (pre-launch compatible) ────────────
    # Time between first and last funding (proxy for funding velocity)
    if "age_first_funding_year" in df.columns and "age_last_funding_year" in df.columns:
        df["funding_duration"] = (
            df["age_last_funding_year"] - df["age_first_funding_year"]
        ).clip(lower=0)

    # Average funding per round
    if "funding_total_usd" in df.columns and "funding_rounds" in df.columns:
        safe_rounds = df["funding_rounds"].replace(0, 1)
        df["avg_funding_per_round"] = df["funding_total_usd"] / safe_rounds

    # Log-transform funding (reduces skew)
    if "funding_total_usd" in df.columns:
        df["log_funding"] = np.log1p(df["funding_total_usd"].fillna(0))

    # ── Category consolidation ──────────────────────────────
    if "category_code" in df.columns:
        df["category_code"] = df["category_code"].fillna("other").astype(str)
        # Keep top categories; group rare ones as "other"
        top_cats = df["category_code"].value_counts().nlargest(12).index
        df["category_code"] = df["category_code"].where(
            df["category_code"].isin(top_cats), "other"
        )

    # ── State consolidation ─────────────────────────────────
    if "state_code" in df.columns:
        df["state_code"] = df["state_code"].fillna("other").astype(str)
        top_states = df["state_code"].value_counts().nlargest(8).index
        df["state_code"] = df["state_code"].where(
            df["state_code"].isin(top_states), "other"
        )

    return df


def select_features(df: pd.DataFrame):
    """
    Return X, y and the feature-name lists used by the pipeline.
    Only features available at pre-launch / early stage are included.
    """

    numeric_features = [
        "age_first_funding_year",
        "age_last_funding_year",
        "age_first_milestone_year",
        "age_last_milestone_year",
        "relationships",
        "funding_rounds",
        "funding_total_usd",
        "milestones",
        "avg_participants",
        "funding_duration",
        "avg_funding_per_round",
        "log_funding",
    ]

    binary_features = [
        "has_VC",
        "has_angel",
        "has_roundA",
        "has_roundB",
        "has_roundC",
        "has_roundD",
        "is_top500",
    ]

    categorical_features = [
        "category_code",
        "state_code",
    ]

    # Keep only columns that actually exist in the dataframe
    numeric_features = [c for c in numeric_features if c in df.columns]
    binary_features = [c for c in binary_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    all_features = numeric_features + binary_features + categorical_features

    X = df[all_features].copy()
    y = df["success"].copy()

    # Ensure binary cols are int
    for b in binary_features:
        X[b] = pd.to_numeric(X[b], errors="coerce").fillna(0).astype(int)

    print(f"✅  Feature selection:")
    print(f"    Numeric  ({len(numeric_features)}): {numeric_features}")
    print(f"    Binary   ({len(binary_features)}): {binary_features}")
    print(f"    Category ({len(categorical_features)}): {categorical_features}")
    print(f"    Total features: {len(all_features)}\n")

    return X, y, numeric_features, binary_features, categorical_features


def build_preprocessor(numeric_features, binary_features, categorical_features):
    """Sklearn ColumnTransformer matching the selected features."""

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("bin", binary_pipe, binary_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )

    return preprocessor


def build_models(preprocessor):
    """Return dict of named pipelines."""

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=5000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ]),
    }

    return models


def cross_validate_models(models, X_train, y_train):
    """5-fold stratified CV; returns list of result dicts."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "f1", "accuracy", "balanced_accuracy", "precision", "recall"]

    results = []

    for name, pipe in models.items():
        cv_out = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        row = {"name": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"cv_{metric}_mean"] = float(np.mean(cv_out[key]))
            row[f"cv_{metric}_std"] = float(np.std(cv_out[key]))

        results.append(row)

        print(f"  {name:25s}  →  ROC-AUC: {row['cv_roc_auc_mean']:.3f} ± {row['cv_roc_auc_std']:.3f}"
              f"   Acc: {row['cv_accuracy_mean']:.3f}   F1: {row['cv_f1_mean']:.3f}")

    return results


def evaluate_holdout(model, X_test, y_test):
    """Full holdout evaluation; returns metrics dict."""

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "classification_report": classification_report(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    return metrics


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    # 1) Load & clean
    df = load_and_clean(DATA_FILE)

    # 2) Feature engineering
    df = engineer_features(df)

    # 3) Select features
    X, y, num_feats, bin_feats, cat_feats = select_features(df)

    # 4) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"✅  Split: train={len(X_train)}, test={len(X_test)}\n")

    # 5) Build preprocessor & models
    preprocessor = build_preprocessor(num_feats, bin_feats, cat_feats)
    models = build_models(preprocessor)

    # 6) Cross-validation
    print("── Cross-Validation (5-fold stratified) ──")
    cv_results = cross_validate_models(models, X_train, y_train)

    # 7) Select best by CV ROC-AUC
    best_row = max(cv_results, key=lambda r: r["cv_roc_auc_mean"])
    best_name = best_row["name"]
    best_pipe = models[best_name]

    print(f"\n✅  Best model (by CV ROC-AUC): {best_name}"
          f"  ({best_row['cv_roc_auc_mean']:.3f})\n")

    # 8) Fit best model on full training set & evaluate holdout
    best_pipe.fit(X_train, y_train)
    holdout = evaluate_holdout(best_pipe, X_test, y_test)

    print("── Holdout Results ──")
    print(holdout["classification_report"])
    print(f"  ROC-AUC : {holdout['roc_auc']:.3f}")
    print(f"  PR-AUC  : {holdout['pr_auc']:.3f}")
    print(f"  Confusion Matrix: {holdout['confusion_matrix']}\n")

    # 9) Save model
    joblib.dump(best_pipe, MODEL_OUT)
    print(f"✅  Model saved → {MODEL_OUT}")

    # 10) Save results JSON (for IPD report / charts)
    summary = {
        "winner": best_name,
        "holdout_metrics": holdout,
        "all_model_results": cv_results,
        "features": {
            "numeric": num_feats,
            "binary": bin_feats,
            "categorical": cat_feats,
        },
    }

    with open(RESULTS_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✅  Results saved → {RESULTS_OUT}")
    print(f"\n✅  Features used (numeric) : {num_feats}")
    print(f"✅  Features used (binary)  : {bin_feats}")
    print(f"✅  Features used (category): {cat_feats}")


if __name__ == "__main__":
    main()
