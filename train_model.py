"""
VentureVerse - Model Training Pipeline
Trains ML models to predict startup success based on Crunchbase data.

Kashish Jadhav (w2035589)
BSc Computer Science Final Project
University of Westminster, 2025-2026
"""

# Imports


import json
import warnings
import sys

# Fix for Windows terminals that can't display emoji characters
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np       # Numerical operations (e.g. log, mean)
import pandas as pd      # Data manipulation (DataFrames)
import joblib            # Save/load the trained model to disk

# scikit-learn: the main ML library
from sklearn.model_selection import (
    train_test_split,    # Split data into training and testing sets
    StratifiedKFold,     # K-fold cross-validation that preserves class ratios
    cross_validate,      # Run cross-validation with multiple metrics
)
from sklearn.preprocessing import (
    OneHotEncoder,       # Convert categories (e.g. "CA") into binary columns
    StandardScaler,      # Normalise numeric features to mean=0, std=1
)
from sklearn.compose import ColumnTransformer   # Apply different transforms to different columns
from sklearn.pipeline import Pipeline            # Chain preprocessing + model into one object
from sklearn.impute import SimpleImputer         # Fill in missing values
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,              # % of correct predictions
    balanced_accuracy_score,     # Accuracy adjusted for class imbalance
    precision_score,             # Of all predicted "success", how many were correct?
    recall_score,                # Of all actual "success", how many did we find?
    f1_score,                    # Harmonic mean of precision and recall
    roc_auc_score,               # Area under ROC curve (overall ranking quality)
    average_precision_score,     # Area under Precision-Recall curve
    classification_report,       # Full text report of all metrics
    confusion_matrix,            # 2x2 matrix: TP, FP, TN, FN
)

# XGBoost: gradient-boosted decision trees (state-of-the-art)
from xgboost import XGBClassifier

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")


# Configuration


DATA_FILE = "startup_data.csv"                 # Input dataset
MODEL_OUT = "ventureverse_model.joblib"        # Where to save the best model
RESULTS_OUT = "model_results_summary.json"     # Where to save all metrics
RANDOM_STATE = 42                              # Fixed seed for reproducibility
TEST_SIZE = 0.20                               # 20% of data held out for testing


# Step 1: Data Cleaning


def load_and_clean(path):
    """
    Loads the Crunchbase startup CSV file and does basic cleaning.

    What it does:
      1. Reads the CSV file into a pandas DataFrame
      2. Keeps only rows where status is 'acquired' or 'closed'
         (ignores 'operating' rows — we can't label them yet)
      3. Creates a binary 'success' column:
         - 1 = acquired (success)
         - 0 = closed (failure)

    Args:
        path: file path to the CSV (e.g. "startup_data.csv")

    Returns:
        Cleaned pandas DataFrame with a 'success' column
    """
    # Read the CSV file
    df = pd.read_csv(path, encoding="ISO-8859-1")

    print(f"✅  Loaded: {path}")
    print(f"    Shape : {df.shape}")
    print(f"    Columns: {df.columns.tolist()}\n")

    # - Create the target column -
    # The dataset has a "status" column with values like
    # "acquired", "closed", "operating". We only want the first two.
    if "status" not in df.columns:
        # Some versions of the dataset use a "labels" column instead
        if "labels" in df.columns:
            df["status"] = df["labels"].map({1: "acquired", 0: "closed"})
        else:
            raise ValueError("❌ No 'status' or 'labels' column found.")

    # Keep only acquired and closed startups
    df = df[df["status"].isin(["acquired", "closed"])].copy()

    # Convert to binary: 1 = success (acquired), 0 = failure (closed)
    df["success"] = (df["status"] == "acquired").astype(int)

    print(f"✅  Target distribution (after filter):")
    print(f"    {df['success'].value_counts().to_dict()}\n")

    return df


# Step 2: Feature Engineering


def engineer_features(df):
    """
    Creates and cleans features that the ML models will use.

    Feature engineering is about creating new, useful columns
    from existing data. For example:
      - funding_duration = how long between first and last funding
      - avg_funding_per_round = total funding ÷ number of rounds
      - log_funding = log-transformed funding (reduces skew)

    Also consolidates rare categories:
      - Industries with few examples → grouped as "other"
      - States with few examples → grouped as "other"

    Args:
        df: the cleaned DataFrame from load_and_clean()

    Returns:
        DataFrame with new engineered features added
    """

    # - Convert numeric columns to proper number types -
    # Some columns might have been read as strings; fix that.
    numeric_columns = [
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

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Funding Duration -years passed between first- last funding round.
    # A longer duration means interest.
    if "age_first_funding_year" in df.columns and "age_last_funding_year" in df.columns:
        df["funding_duration"] = (
            df["age_last_funding_year"] - df["age_first_funding_year"]
        ).clip(lower=0)  # clip(lower=0) means: no negative values

    # Average Funding Per Round - Total funding/number of rounds.
    # Higher values mean more investor interested.
    if "funding_total_usd" in df.columns and "funding_rounds" in df.columns:
        safe_rounds = df["funding_rounds"].replace(0, 1)  # Avoid dividing by zero
        df["avg_funding_per_round"] = df["funding_total_usd"] / safe_rounds

    # Log-Transformed Funding - Funding ranges from $0 to $1 billion+. That huge range
    # makes it hard for models to learn. Log-transform compresses
    # it: log(1 + 10000) ≈ 9.2, log(1 + 100000000) ≈ 18.4
    if "funding_total_usd" in df.columns:
        df["log_funding"] = np.log1p(df["funding_total_usd"].fillna(0))

    
    # Industry consolidate- Keep the 12 most common industries; group rest as "other".
    # to prevents the model from overfitting to rare categories with only a handful of examples.
    if "category_code" in df.columns:
        df["category_code"] = df["category_code"].fillna("other").astype(str)
        top_categories = df["category_code"].value_counts().nlargest(12).index
        df["category_code"] = df["category_code"].where(
            df["category_code"].isin(top_categories), "other"
        )


    # Keep top 8 states; group the rest as "other"
    if "state_code" in df.columns:
        df["state_code"] = df["state_code"].fillna("other").astype(str)
        top_states = df["state_code"].value_counts().nlargest(8).index
        df["state_code"] = df["state_code"].where(
            df["state_code"].isin(top_states), "other"
        )

    return df


# Step 3: Feature Selection


def select_features(df):
    """
    Picks which columns the ML model should use for prediction.

    We use three types of features:
      1. Numeric  — continuous numbers (funding, age, etc.)
      2. Binary   — yes/no flags (has_VC, is_top500, etc.)
      3. Category — text labels (industry, state)

    Only features available BEFORE launch are included.
    (We're predicting success before it happens, so we can't
    use post-launch data like revenue or user count.)

    Args:
        df: the engineered DataFrame

    Returns:
        X: feature DataFrame
        y: target Series (0 or 1)
        numeric_features: list of numeric column names
        binary_features: list of binary column names
        categorical_features: list of categorical column names
    """

    # - Define the three groups of features -
    numeric_features = [
        "age_first_funding_year",     # Years from founding to first funding
        "age_last_funding_year",      # Years from founding to latest funding
        "age_first_milestone_year",   # Years from founding to first milestone
        "age_last_milestone_year",    # Years from founding to latest milestone
        "relationships",              # Number of key connections (advisors, etc.)
        "funding_rounds",             # How many funding rounds completed
        "funding_total_usd",          # Total money raised in USD
        "milestones",                 # Number of milestones achieved
        "avg_participants",           # Average investors per round
        "funding_duration",           # Time span of funding (engineered)
        "avg_funding_per_round",      # Funding ÷ rounds (engineered)
        "log_funding",                # Log-transformed funding (engineered)
    ]

    binary_features = [
        "has_VC",       # Has venture capital backing? (1 = yes, 0 = no)
        "has_angel",    # Has angel investor? (1 = yes, 0 = no)
        "has_roundA",   # Completed Series A round?
        "has_roundB",   # Completed Series B round?
        "has_roundC",   # Completed Series C round?
        "has_roundD",   # Completed Series D round?
        "is_top500",    # Backed by a Top-500 VC firm?
    ]

    categorical_features = [
        "category_code",  # Industry (e.g. "software", "biotech")
        "state_code",     # US state (e.g. "CA", "NY")
    ]

    # Keep only columns that actually exist in the data
    # (in case a column is missing from the CSV)
    numeric_features = [c for c in numeric_features if c in df.columns]
    binary_features = [c for c in binary_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    all_features = numeric_features + binary_features + categorical_features

    # Split into X (features) and y (target)
    X = df[all_features].copy()
    y = df["success"].copy()

    # Make sure binary columns are integers (0 or 1)
    for col in binary_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    print(f"✅  Feature selection:")
    print(f"    Numeric  ({len(numeric_features)}): {numeric_features}")
    print(f"    Binary   ({len(binary_features)}): {binary_features}")
    print(f"    Category ({len(categorical_features)}): {categorical_features}")
    print(f"    Total features: {len(all_features)}\n")

    return X, y, numeric_features, binary_features, categorical_features


# Step 4: Preprocessing


def build_preprocessor(numeric_features, binary_features, categorical_features):
    """
    Creates an sklearn ColumnTransformer that preprocesses each
    type of feature differently:

      - Numeric features:
          1. Fill missing values with the median (middle value)
          2. Scale to mean=0, std=1 (StandardScaler)

      - Binary features:
          1. Fill missing values with the most common value

      - Categorical features:
          1. Fill missing values with "other"
          2. One-hot encode (e.g. "CA" → [1,0,0,...], "NY" → [0,1,0,...])

    This preprocessor is combined with the ML model into a
    Pipeline, so all transformations happen automatically.

    Args:
        numeric_features:     list of numeric column names
        binary_features:      list of binary column names
        categorical_features: list of categorical column names

    Returns:
        sklearn ColumnTransformer object
    """

    # Pipeline for numeric columns: fill missing → scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Pipeline for binary columns: just fill missing values
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Pipeline for categorical columns: fill missing → one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine all three pipelines into one ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("bin", binary_pipeline, binary_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# Step 5: Model Definitions


def build_models(preprocessor):
    """
    Creates three ML model pipelines. Each pipeline combines
    the preprocessor (step 4) with a classifier.

    Models:
      1. Logistic Regression — linear model, good baseline
      2. Random Forest       — ensemble of 500 decision trees
      3. XGBoost             — gradient-boosted trees (usually best)

    Each model is wrapped in a Pipeline so that preprocessing
    and prediction happen in one step.

    Args:
        preprocessor: the ColumnTransformer from build_preprocessor()

    Returns:
        dict of {model_name: Pipeline}
    """

    models = {
        # - Model 1: Logistic Regression -
        # Simple linear model. Good for understanding which
        # features matter most. Fast to train.
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=5000,            # Max iterations for convergence
                C=1.0,                    # Regularisation strength
                solver="lbfgs",           # Optimisation algorithm
                random_state=RANDOM_STATE,
            )),
        ]),

        # - Model 2: Random Forest -
        # Builds 500 decision trees and averages their predictions.
        # More robust than a single tree, handles non-linear patterns.
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=500,         # Number of trees
                max_depth=10,             # Max depth of each tree
                min_samples_leaf=5,       # Min samples in a leaf node
                random_state=RANDOM_STATE,
                n_jobs=-1,                # Use all CPU cores
            )),
        ]),

        # - Model 3: XGBoost -
        # State-of-the-art gradient boosting. Builds trees
        # sequentially, each one correcting the previous errors.
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=600,         # Number of boosting rounds
                learning_rate=0.05,       # Step size per round
                max_depth=6,              # Max depth of each tree
                subsample=0.85,           # Use 85% of data per tree
                colsample_bytree=0.85,    # Use 85% of features per tree
                reg_lambda=1.0,           # L2 regularisation
                random_state=RANDOM_STATE,
                eval_metric="logloss",    # Loss function for binary classification
                n_jobs=-1,                # Use all CPU cores
            )),
        ]),
    }

    return models


# Step 6: Cross-Validation


def cross_validate_models(models, X_train, y_train):
    """
    Runs 5-fold stratified cross-validation on each model.

    What is cross-validation?
      Instead of training once, we split the training data into
      5 parts ("folds"). We train on 4 folds and test on the 5th,
      then repeat 5 times so every fold gets tested. This gives
      us a more reliable estimate of how well the model works.

    "Stratified" means each fold has the same ratio of
    success/failure cases as the full dataset.

    Args:
        models:  dict of {name: Pipeline}
        X_train: training features
        y_train: training labels

    Returns:
        list of result dicts with mean and std for each metric
    """

    # Create 5 stratified folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Metrics to measure
    scoring = ["roc_auc", "f1", "accuracy", "balanced_accuracy", "precision", "recall"]

    results = []

    for name, pipeline in models.items():
        # Run cross-validation
        cv_output = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,  # Parallel processing
        )

        # Collect the results as a dict
        row = {"name": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"cv_{metric}_mean"] = float(np.mean(cv_output[key]))
            row[f"cv_{metric}_std"] = float(np.std(cv_output[key]))

        results.append(row)

        # Print a summary line for each model
        print(
            f"  {name:25s}  →  "
            f"ROC-AUC: {row['cv_roc_auc_mean']:.3f} ± {row['cv_roc_auc_std']:.3f}   "
            f"Acc: {row['cv_accuracy_mean']:.3f}   "
            f"F1: {row['cv_f1_mean']:.3f}"
        )

    return results


# Step 7: Final Evaluation


def evaluate_holdout(model, X_test, y_test):
    """
    After choosing the best model, we evaluate it on the
    holdout test set (data the model has never seen before).

    This gives us an unbiased estimate of real-world performance.

    Metrics computed:
      - accuracy          : % of correct predictions
      - balanced_accuracy  : accuracy adjusted for class imbalance
      - precision          : of predicted "successes", how many were right?
      - recall             : of actual "successes", how many did we catch?
      - f1                 : balance between precision and recall
      - roc_auc            : overall model ranking quality (0.5 = random, 1.0 = perfect)
      - pr_auc             : precision-recall curve area
      - confusion_matrix   : [[TN, FP], [FN, TP]]

    Args:
        model:  the best Pipeline (already fitted)
        X_test: test features
        y_test: true test labels

    Returns:
        dict of metric name → value
    """

    # Get predictions
    predictions = model.predict(X_test)                # Binary: 0 or 1
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of success

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "classification_report": classification_report(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    return metrics


# Main pipeline execution


def main():
    """
    Orchestrates the entire model training process:
      1. Load and clean the dataset
      2. Engineer new features
      3. Select the features for the model
      4. Split into training and testing sets
      5. Build the preprocessor and models
      6. Cross-validate all models
      7. Pick the best model (by ROC-AUC)
      8. Train the best model on full training data
      9. Evaluate on the holdout test set
     10. Save the model and results to disk
    """

    # Step 1: Load and clean
    print("- Loading Dataset -")
    df = load_and_clean(DATA_FILE)

    # Step 2: Feature engineering
    print("- Engineering Features -")
    df = engineer_features(df)

    # Step 3: Select features
    print("- Selecting Features -")
    X, y, num_feats, bin_feats, cat_feats = select_features(df)

    # Step 4: Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Keeps the same success/failure ratio in both sets
    )
    print(f"✅  Split: train={len(X_train)}, test={len(X_test)}\n")

    # Step 5: Build preprocessor and models
    preprocessor = build_preprocessor(num_feats, bin_feats, cat_feats)
    models = build_models(preprocessor)

    # Step 6: Cross-validation (5-fold)
    print("- Cross-Validation (5-fold stratified) -")
    cv_results = cross_validate_models(models, X_train, y_train)

    # Step 7: Select the best model (highest CV ROC-AUC)
    best_result = max(cv_results, key=lambda r: r["cv_roc_auc_mean"])
    best_name = best_result["name"]
    best_pipeline = models[best_name]

    print(f"\n✅  Best model (by CV ROC-AUC): {best_name}"
          f"  ({best_result['cv_roc_auc_mean']:.3f})\n")

    # Step 8: Fit the best model on ALL training data
    best_pipeline.fit(X_train, y_train)

    # Step 9: Evaluate on the holdout test set
    holdout_metrics = evaluate_holdout(best_pipeline, X_test, y_test)

    print("- Holdout Results -")
    print(holdout_metrics["classification_report"])
    print(f"  ROC-AUC : {holdout_metrics['roc_auc']:.3f}")
    print(f"  PR-AUC  : {holdout_metrics['pr_auc']:.3f}")
    print(f"  Confusion Matrix: {holdout_metrics['confusion_matrix']}\n")

    # Step 10a: Save the model
    joblib.dump(best_pipeline, MODEL_OUT)
    print(f"✅  Model saved → {MODEL_OUT}")

    # Step 10b: Save the full results as JSON
    summary = {
        "winner": best_name,
        "holdout_metrics": holdout_metrics,
        "all_model_results": cv_results,
        "features": {
            "numeric": num_feats,
            "binary": bin_feats,
            "categorical": cat_feats,
        },
    }

    with open(RESULTS_OUT, "w") as file:
        json.dump(summary, file, indent=2, default=str)

    print(f"✅  Results saved → {RESULTS_OUT}")
    print(f"\n✅  Features used (numeric) : {num_feats}")
    print(f"✅  Features used (binary)  : {bin_feats}")
    print(f"✅  Features used (category): {cat_feats}")


# - Entry point: run the main function -
if __name__ == "__main__":
    main()
