import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/sentinel_transactions.csv"
MODEL_PATH = "models/sentinel_v3_model.pkl"
EVAL_PATH = "data/processed/sentinel_v3_threshold_analysis.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("              SENTINEL V3 - RISK INTELLIGENCE")
print("=" * 70)

print("\n[1/10] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

TARGET = "isFraud"

numeric_base = [
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

categorical_base = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[2/10] Creating held-out test set...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TARGET]
)

train_df = train_df.copy()
test_df = test_df.copy()

print("Training samples:", len(train_df))
print("Testing samples :", len(test_df))


# ============================================================
# FRAUD DNA
# ============================================================

print("\n[3/10] Building FraudDNA behavioral features...")


def build_fraud_dna(data, reference_df):
    x = data.copy()

    # Amount behavior
    x["dna_log_amount"] = np.log1p(
        x["TransactionAmt"].clip(lower=0)
    )

    amount_median = reference_df["TransactionAmt"].median()

    x["dna_amount_vs_median"] = (
        x["TransactionAmt"] / (amount_median + 1e-6)
    )

    # Time behavior
    seconds_day = x["TransactionDT"] % 86400

    x["dna_hour"] = (
        seconds_day // 3600
    ).astype(int)

    x["dna_time_of_day"] = (
        x["dna_hour"] / 24.0
    )

    x["dna_is_night"] = (
        (x["dna_hour"] < 6) |
        (x["dna_hour"] >= 23)
    ).astype(int)

    x["dna_day_index"] = (
        x["TransactionDT"] // 86400
    )

    # Missingness fingerprint
    x["dna_missing_count"] = x[
        numeric_base + categorical_base
    ].isna().sum(axis=1)

    x["dna_missing_email"] = (
        x["P_emaildomain"].isna() &
        x["R_emaildomain"].isna()
    ).astype(int)

    x["dna_missing_address"] = (
        x["addr1"].isna() |
        x["addr2"].isna()
    ).astype(int)

    x["dna_missing_distance"] = (
        x["dist1"].isna()
    ).astype(int)

    # Email relationship
    p = x["P_emaildomain"].fillna("").astype(str).str.lower()
    r = x["R_emaildomain"].fillna("").astype(str).str.lower()

    x["dna_email_match"] = (
        (p != "") &
        (r != "") &
        (p == r)
    ).astype(int)

    x["dna_email_mismatch"] = (
        (p != "") &
        (r != "") &
        (p != r)
    ).astype(int)

    # Distance
    x["dna_distance_missing"] = (
        x["dist1"].isna()
    ).astype(int)

    x["dna_distance_log"] = np.log1p(
        x["dist1"].fillna(0).clip(lower=0)
    )

    return x


train_dna = build_fraud_dna(
    train_df,
    train_df
)

test_dna = build_fraud_dna(
    test_df,
    train_df
)


# ============================================================
# FRAUD CHAIN
# ============================================================

print("\n[4/10] Building FraudChain relationship features...")


chain_columns = [
    "card1",
    "card2",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain"
]


def safe_key(series):
    return series.fillna("__MISSING__").astype(str)


def build_chain_features(data, reference_df):

    x = data.copy()
    ref = reference_df.copy()

    # Individual entity frequency
    for col in chain_columns:

        ref_key = safe_key(ref[col])
        data_key = safe_key(x[col])

        counts = ref_key.value_counts()

        x["chain_" + col + "_freq"] = (
            data_key.map(counts).fillna(0)
        )

    # Relationship frequencies
    pairs = [
        ("card1", "addr1"),
        ("card1", "addr2"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("addr1", "P_emaildomain"),
        ("addr1", "R_emaildomain")
    ]

    for a, b in pairs:

        ref_pair = (
            safe_key(ref[a]) + "|" +
            safe_key(ref[b])
        )

        data_pair = (
            safe_key(x[a]) + "|" +
            safe_key(x[b])
        )

        counts = ref_pair.value_counts()

        name = "chain_" + a + "_" + b + "_freq"

        x[name] = (
            data_pair.map(counts).fillna(0)
        )

    # Overall connectivity
    freq_columns = [
        c for c in x.columns
        if c.startswith("chain_") and c.endswith("_freq")
    ]

    x["chain_connection_count"] = (
        (x[freq_columns] > 1).sum(axis=1)
    )

    return x


train_features = build_chain_features(
    train_dna,
    train_dna
)

test_features = build_chain_features(
    test_dna,
    train_dna
)


# ============================================================
# ANOMALY ENGINE
# ============================================================

print("\n[5/10] Training Isolation Forest anomaly engine...")


anomaly_columns = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dna_log_amount",
    "dna_hour",
    "dna_missing_count",
    "dna_amount_vs_median"
]

anomaly_train = train_features[anomaly_columns].copy()
anomaly_test = test_features[anomaly_columns].copy()

anomaly_imputer = SimpleImputer(strategy="median")

anomaly_train_imp = anomaly_imputer.fit_transform(
    anomaly_train
)

anomaly_test_imp = anomaly_imputer.transform(
    anomaly_test
)

iso = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

iso.fit(anomaly_train_imp)

train_anomaly_raw = iso.decision_function(
    anomaly_train_imp
)

test_anomaly_raw = iso.decision_function(
    anomaly_test_imp
)

# Lower IsolationForest score = more anomalous.
# Convert into a 0-100 anomaly risk score.

min_score = train_anomaly_raw.min()
max_score = train_anomaly_raw.max()

def normalize_anomaly(values):
    normalized = (
        1 -
        (
            (values - min_score) /
            (max_score - min_score + 1e-9)
        )
    ) * 100

    return np.clip(normalized, 0, 100)


train_features["anomaly_score"] = normalize_anomaly(
    train_anomaly_raw
)

test_features["anomaly_score"] = normalize_anomaly(
    test_anomaly_raw
)


# ============================================================
# PREPARE SUPERVISED FEATURES
# ============================================================

print("\n[6/10] Preparing supervised features...")

drop_columns = [
    TARGET,
    "TransactionID"
]

X_train = train_features.drop(
    columns=drop_columns,
    errors="ignore"
)

X_test = test_features.drop(
    columns=drop_columns,
    errors="ignore"
)

y_train = train_df[TARGET]
y_test = test_df[TARGET]


# Explicitly define columns
categorical_features = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

numeric_features = [
    c for c in X_train.columns
    if c not in categorical_features
]

print("Total features    :", len(X_train.columns))
print("Numeric features  :", len(numeric_features))
print("Categorical       :", len(categorical_features))


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n[7/10] Training Random Forest...")


rf = RandomForestClassifier(
    n_estimators=250,
    class_weight="balanced",
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            rf
        )
    ]
)

rf_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n[8/10] Training Logistic Regression ensemble member...")


lr_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=0.5,
    solver="liblinear",
    random_state=RANDOM_STATE
)

lr_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            lr_preprocessor
        ),
        (
            "model",
            lr
        )
    ]
)

lr_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n[9/10] Generating ensemble predictions...")


rf_probability = rf_pipeline.predict_proba(
    X_test
)[:, 1]

lr_probability = lr_pipeline.predict_proba(
    X_test
)[:, 1]

anomaly_probability = (
    test_features["anomaly_score"].values / 100.0
)


# ------------------------------------------------------------
# MULTI-SIGNAL FUSION
# ------------------------------------------------------------
#
# RF = primary supervised detector
# LR = complementary linear detector
# Anomaly = unsupervised novelty detector
#
# The anomaly engine receives a smaller weight because
# "unusual" does not automatically mean "fraud".
# ------------------------------------------------------------

ensemble_probability = (
    0.65 * rf_probability +
    0.25 * lr_probability +
    0.10 * anomaly_probability
)

ensemble_probability = np.clip(
    ensemble_probability,
    0,
    1
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n[10/10] Evaluating thresholds...")


thresholds = np.arange(
    0.20,
    0.81,
    0.05
)

results = []

for threshold in thresholds:

    predictions = (
        ensemble_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    # False negative is treated as 5x the cost
    # of a false positive.
    estimated_cost = (
        fp * 1 +
        fn * 5
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "fraud_detection_rate": round(float(recall), 6),
        "estimated_cost": int(estimated_cost)
    })


results_df = pd.DataFrame(results)


# ============================================================
# BEST THRESHOLDS
# ============================================================

best_f1_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_cost_row = results_df.loc[
    results_df["estimated_cost"].idxmin()
]


print("\n")
print("=" * 70)
print("                 SENTINEL V3 RESULTS")
print("=" * 70)

print("\nBest F1 threshold:")
print(
    "Threshold :",
    best_f1_row["threshold"]
)

print(
    "Precision :",
    f"{best_f1_row['precision']:.4f}"
)

print(
    "Recall    :",
    f"{best_f1_row['recall']:.4f}"
)

print(
    "F1 Score  :",
    f"{best_f1_row['f1']:.4f}"
)

print(
    "FP        :",
    best_f1_row["false_positives"]
)

print(
    "FN        :",
    best_f1_row["false_negatives"]
)


print("\nCost-aware threshold:")

print(
    "Threshold :",
    best_cost_row["threshold"]
)

print(
    "Precision :",
    f"{best_cost_row['precision']:.4f}"
)

print(
    "Recall    :",
    f"{best_cost_row['recall']:.4f}"
)

print(
    "F1 Score  :",
    f"{best_cost_row['f1']:.4f}"
)

print(
    "FP        :",
    best_cost_row["false_positives"]
)

print(
    "FN        :",
    best_cost_row["false_negatives"]
)

print(
    "Estimated Cost:",
    best_cost_row["estimated_cost"]
)


print("\nThreshold table:")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# CHOOSE OPERATING THRESHOLD
# ============================================================

OPERATING_THRESHOLD = float(
    best_f1_row["threshold"]
)


# ============================================================
# MODEL ARTIFACT
# ============================================================

model_data = {

    "version": "3.0",

    "rf_model": rf_pipeline,

    "lr_model": lr_pipeline,

    "anomaly_model": iso,

    "anomaly_imputer": anomaly_imputer,

    "anomaly_min_score": float(min_score),

    "anomaly_max_score": float(max_score),

    "threshold": OPERATING_THRESHOLD,

    "ensemble_weights": {
        "random_forest": 0.65,
        "logistic_regression": 0.25,
        "anomaly_engine": 0.10
    },

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "chain_columns": chain_columns,

    "reference_amount_median": float(
        train_df["TransactionAmt"].median()
    )
}


joblib.dump(
    model_data,
    MODEL_PATH
)

results_df.to_csv(
    EVAL_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SENTINEL V3 TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(MODEL_PATH)

print("\nEvaluation saved:")
print(EVAL_PATH)

print("\nOperating threshold:")
print(OPERATING_THRESHOLD)

print("\nArchitecture:")
print("  FraudDNA")
print("      +")
print("  FraudChain")
print("      +")
print("  Isolation Forest")
print("      +")
print("  Random Forest")
print("      +")
print("  Logistic Regression")
print("      ↓")
print("  Risk Intelligence Ensemble")
print("=" * 70)