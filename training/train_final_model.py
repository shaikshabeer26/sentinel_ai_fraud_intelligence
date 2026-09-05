
import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SENTINEL - FINAL FRAUD DETECTION MODEL
# ============================================================

DATA_PATH = "data/processed/sentinel_transactions.csv"
MODEL_PATH = "models/sentinel_fraud_model.pkl"

print("=" * 60)
print("SENTINEL - FINAL MODEL TRAINING")
print("=" * 60)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\n[1/6] Loading dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {data.shape}")

# ------------------------------------------------------------
# 2. Separate target and features
# ------------------------------------------------------------

print("\n[2/6] Preparing features...")

# IMPORTANT:
# TransactionID is deliberately excluded.
# It is an identifier, not a transaction behavior feature.

X = data.drop(columns=["isFraud", "TransactionID"])
y = data["isFraud"]

print(f"Features used: {X.shape[1]}")
print(f"Fraud cases: {y.sum()}")
print(f"Legitimate cases: {(y == 0).sum()}")

# ------------------------------------------------------------
# 3. Define feature types
# ------------------------------------------------------------

numeric_features = [
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1"
]

categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

# ------------------------------------------------------------
# 4. Build preprocessing pipeline
# ------------------------------------------------------------

print("\n[3/6] Building preprocessing pipeline...")

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# ------------------------------------------------------------
# 5. Build Random Forest model
# ------------------------------------------------------------

print("\n[4/6] Building Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

sentinel_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# ------------------------------------------------------------
# 6. Train and save model
# ------------------------------------------------------------

print("\n[5/6] Training model...")
print("This may take some time...")

sentinel_model.fit(X, y)

print("\n[6/6] Saving model...")

os.makedirs("models", exist_ok=True)

model_data = {
    "model": sentinel_model,
    "threshold": 0.45,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(model_data, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print(f"\nModel saved to:")
print(MODEL_PATH)

print("\nModel configuration:")
print("Algorithm      : Random Forest")
print("Trees          : 100")
print("Class Weight   : balanced")
print("Fraud Threshold: 0.45")
print("Features       : 14")
print("TransactionID  : EXCLUDED")

print("\nSENTINEL model is ready.")