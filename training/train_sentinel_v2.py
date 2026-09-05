import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# SENTINEL V2
# FraudDNA + FraudChain + Anomaly Intelligence
# ============================================================

DATA_PATH = "data/processed/sentinel_transactions.csv"

MODEL_PATH = "models/sentinel_v2_model.pkl"
RESULT_PATH = "data/processed/sentinel_v2_threshold_analysis.csv"

RANDOM_STATE = 42


print("=" * 70)
print("        SENTINEL V2 - AI FRAUD INTELLIGENCE")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/9] Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)

y = data["isFraud"].astype(int)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

print("\n[2/9] Creating held-out test set...")

train_df, test_df = train_test_split(
    data,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

train_df = train_df.copy()
test_df = test_df.copy()

print("Training samples:", len(train_df))
print("Testing samples :", len(test_df))


# ============================================================
# 3. FRAUDDNA FEATURE ENGINEERING
# ============================================================

print("\n[3/9] Building FraudDNA features...")


def build_fraud_dna(df):
    df = df.copy()

    # --------------------------------------------------------
    # Transaction amount intelligence
    # --------------------------------------------------------

    df["dna_log_amount"] = np.log1p(
        df["TransactionAmt"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Transaction time intelligence
    #
    # TransactionDT is seconds from a reference point.
    # Convert to approximate hour/day-cycle features.
    # --------------------------------------------------------

    seconds_in_day = 86400

    df["dna_time_of_day"] = (
        df["TransactionDT"] % seconds_in_day
    )

    df["dna_hour"] = (
        df["dna_time_of_day"] // 3600
    )

    df["dna_is_night"] = (
        (df["dna_hour"] < 6) |
        (df["dna_hour"] >= 23)
    ).astype(int)

    df["dna_day_index"] = (
        df["TransactionDT"] // seconds_in_day
    )

    # --------------------------------------------------------
    # Missing-value fingerprint
    # --------------------------------------------------------

    feature_columns = [
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "P_emaildomain",
        "R_emaildomain",
        "ProductCD"
    ]

    df["dna_missing_count"] = (
        df[feature_columns].isnull().sum(axis=1)
    )

    df["dna_missing_email"] = (
        df["P_emaildomain"].isnull() |
        df["R_emaildomain"].isnull()
    ).astype(int)

    df["dna_missing_address"] = (
        df["addr1"].isnull() |
        df["addr2"].isnull()
    ).astype(int)

    df["dna_missing_distance"] = (
        df["dist1"].isnull()
    ).astype(int)

    # --------------------------------------------------------
    # Email relationship
    # --------------------------------------------------------

    p_email = df["P_emaildomain"].fillna("__MISSING__").astype(str)
    r_email = df["R_emaildomain"].fillna("__MISSING__").astype(str)

    df["dna_email_match"] = (
        (p_email == r_email) &
        (p_email != "__MISSING__")
    ).astype(int)

    df["dna_email_mismatch"] = (
        (p_email != r_email) &
        (p_email != "__MISSING__") &
        (r_email != "__MISSING__")
    ).astype(int)

    # --------------------------------------------------------
    # Distance behavior
    # --------------------------------------------------------

    df["dna_distance_missing"] = (
        df["dist1"].isnull()
    ).astype(int)

    df["dna_distance_log"] = np.log1p(
        df["dist1"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Amount relative to broad transaction distribution
    # --------------------------------------------------------

    amount_median = train_df["TransactionAmt"].median()

    df["dna_amount_vs_median"] = (
        df["TransactionAmt"] /
        (amount_median + 1e-6)
    )

    return df


train_dna = build_fraud_dna(train_df)
test_dna = build_fraud_dna(test_df)


# ============================================================
# 4. FRAUDCHAIN FEATURES
# ============================================================

print("\n[4/9] Building FraudChain relationship features...")


def build_chain_features(train_data, target_data):

    target_data = target_data.copy()

    # --------------------------------------------------------
    # Frequency maps
    # These are learned ONLY from training data.
    # --------------------------------------------------------

    entity_columns = [
        "card1",
        "card2",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain"
    ]

    for col in entity_columns:

        counts = (
            train_data[col]
            .value_counts(dropna=False)
        )

        target_data["chain_" + col + "_frequency"] = (
            target_data[col]
            .map(counts)
            .fillna(0)
        )

    # --------------------------------------------------------
    # Relationship frequencies
    # --------------------------------------------------------

    relationship_pairs = [
        ("card1", "addr1"),
        ("card1", "addr2"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("addr1", "P_emaildomain"),
        ("addr1", "R_emaildomain")
    ]

    for col1, col2 in relationship_pairs:

        train_keys = (
            train_data[col1].fillna("__NA__").astype(str)
            + "||" +
            train_data[col2].fillna("__NA__").astype(str)
        )

        target_keys = (
            target_data[col1].fillna("__NA__").astype(str)
            + "||" +
            target_data[col2].fillna("__NA__").astype(str)
        )

        counts = train_keys.value_counts()

        name = (
            "chain_" +
            col1 +
            "_" +
            col2 +
            "_links"
        )

        target_data[name] = (
            target_keys
            .map(counts)
            .fillna(0)
        )

    # --------------------------------------------------------
    # Number of connected entities
    # --------------------------------------------------------

    chain_columns = [
        c for c in target_data.columns
        if c.startswith("chain_")
    ]

    target_data["chain_connection_count"] = (
        target_data[chain_columns]
        .gt(0)
        .sum(axis=1)
    )

    return target_data


train_features = build_chain_features(
    train_dna,
    train_dna
)

test_features = build_chain_features(
    train_dna,
    test_dna
)


# ============================================================
# 5. ANOMALY ENGINE
# ============================================================

print("\n[5/9] Training Isolation Forest anomaly engine...")


anomaly_features = [
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


anomaly_imputer = SimpleImputer(strategy="median")

train_anomaly_matrix = anomaly_imputer.fit_transform(
    train_features[anomaly_features]
)

test_anomaly_matrix = anomaly_imputer.transform(
    test_features[anomaly_features]
)


anomaly_scaler = StandardScaler()

train_anomaly_matrix = anomaly_scaler.fit_transform(
    train_anomaly_matrix
)

test_anomaly_matrix = anomaly_scaler.transform(
    test_anomaly_matrix
)


anomaly_model = IsolationForest(
    n_estimators=150,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

anomaly_model.fit(train_anomaly_matrix)


train_anomaly_raw = anomaly_model.decision_function(
    train_anomaly_matrix
)

test_anomaly_raw = anomaly_model.decision_function(
    test_anomaly_matrix
)


# Lower Isolation Forest score = more abnormal.
# Convert into a 0-100 anomaly risk score.

train_min = train_anomaly_raw.min()
train_max = train_anomaly_raw.max()

train_anomaly_score = (
    1 -
    (
        (train_anomaly_raw - train_min) /
        (train_max - train_min + 1e-9)
    )
) * 100

test_anomaly_score = (
    1 -
    (
        (test_anomaly_raw - train_min) /
        (train_max - train_min + 1e-9)
    )
) * 100

train_anomaly_score = np.clip(
    train_anomaly_score,
    0,
    100
)

test_anomaly_score = np.clip(
    test_anomaly_score,
    0,
    100
)


train_features["anomaly_score"] = train_anomaly_score
test_features["anomaly_score"] = test_anomaly_score


# ============================================================
# 6. PREPARE SUPERVISED FEATURES
# ============================================================

print("\n[6/9] Preparing supervised model features...")


DROP_COLUMNS = [
    "isFraud",
    "TransactionID"
]


X_train = train_features.drop(
    columns=DROP_COLUMNS
)

X_test = test_features.drop(
    columns=DROP_COLUMNS
)

y_train = train_df["isFraud"].astype(int)
y_test = test_df["isFraud"].astype(int)


numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


print("Total features:", X_train.shape[1])
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. TRAIN IMPROVED RANDOM FOREST
# ============================================================

print("\n[7/9] Training improved supervised model...")


classifier = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_features="sqrt",
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)


model.fit(
    X_train,
    y_train
)


train_probability = model.predict_proba(
    X_train
)[:, 1]

test_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. THRESHOLD EVALUATION
# ============================================================

print("\n[8/9] Evaluating thresholds...")


thresholds = np.arange(
    0.20,
    0.81,
    0.05
)


results = []


for threshold in thresholds:

    predictions = (
        test_probability >= threshold
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

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fraud_detection_rate": recall
    })


results_df = pd.DataFrame(results)


best_row = results_df.loc[
    results_df["f1"].idxmax()
]


# Cost model:
# False negative = 5x false positive.

results_df["estimated_cost"] = (
    results_df["false_positives"] * 1
    +
    results_df["false_negatives"] * 5
)


best_cost_row = results_df.loc[
    results_df["estimated_cost"].idxmin()
]


os.makedirs(
    "data/processed",
    exist_ok=True
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

print("\n[9/9] Saving SENTINEL V2...")


model_data = {

    "model": model,

    "anomaly_model": anomaly_model,

    "anomaly_imputer": anomaly_imputer,

    "anomaly_scaler": anomaly_scaler,

    "anomaly_features": anomaly_features,

    "anomaly_train_min": train_min,

    "anomaly_train_max": train_max,

    "threshold": float(
        best_row["threshold"]
    ),

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "dna_features": [
        "dna_log_amount",
        "dna_time_of_day",
        "dna_hour",
        "dna_is_night",
        "dna_day_index",
        "dna_missing_count",
        "dna_missing_email",
        "dna_missing_address",
        "dna_missing_distance",
        "dna_email_match",
        "dna_email_mismatch",
        "dna_distance_missing",
        "dna_distance_log",
        "dna_amount_vs_median"
    ],

    "version": "2.0"

}


os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model_data,
    MODEL_PATH
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("                 SENTINEL V2 RESULTS")
print("=" * 70)

print("\nBest F1 threshold:")
print(
    "Threshold :",
    round(best_row["threshold"], 2)
)

print(
    "Precision :",
    f"{best_row['precision']:.4f}"
)

print(
    "Recall    :",
    f"{best_row['recall']:.4f}"
)

print(
    "F1 Score  :",
    f"{best_row['f1']:.4f}"
)

print(
    "FP        :",
    int(best_row["false_positives"])
)

print(
    "FN        :",
    int(best_row["false_negatives"])
)


print("\nCost-aware threshold:")
print(
    "Threshold :",
    round(best_cost_row["threshold"], 2)
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
    int(best_cost_row["false_positives"])
)

print(
    "FN        :",
    int(best_cost_row["false_negatives"])
)

print(
    "Estimated Cost:",
    int(best_cost_row["estimated_cost"])
)


print("\nThreshold table:")
print(
    results_df.to_string(index=False)
)


print("\nModel saved:")
print(MODEL_PATH)

print("\nEvaluation saved:")
print(RESULT_PATH)

print("\n" + "=" * 70)
print("SENTINEL V2 TRAINING COMPLETE")
print("=" * 70)