import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_PATH = "data/processed/sentinel_transactions.csv"
MODEL_PATH = "models/sentinel_v4_model.pkl"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain"
]

NUMERIC_FEATURES = [
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

BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CHAIN_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain"
]


# ============================================================
# UTILITIES
# ============================================================

def clean_value(v):
    if pd.isna(v):
        return "__MISSING__"
    return str(v)


def safe_log1p(v):
    try:
        return np.log1p(max(float(v), 0))
    except:
        return 0.0


# ============================================================
# FRAUD DNA
# ============================================================

def build_dna(df, reference=None):

    out = pd.DataFrame(index=df.index)

    amount = pd.to_numeric(
        df["TransactionAmt"],
        errors="coerce"
    ).fillna(0)

    transaction_dt = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce"
    ).fillna(0)

    dist = pd.to_numeric(
        df["dist1"],
        errors="coerce"
    )

    # Time
    seconds_day = transaction_dt % 86400
    hour = seconds_day // 3600

    out["dna_log_amount"] = amount.apply(safe_log1p)
    out["dna_hour"] = hour
    out["dna_is_night"] = (
        (hour < 6) | (hour >= 23)
    ).astype(int)

    out["dna_day_index"] = (
        transaction_dt // 86400
    )

    # Missingness
    out["dna_missing_count"] = df.isna().sum(axis=1)

    out["dna_missing_email"] = (
        df["P_emaildomain"].isna() |
        df["R_emaildomain"].isna()
    ).astype(int)

    out["dna_missing_address"] = (
        df["addr1"].isna() |
        df["addr2"].isna()
    ).astype(int)

    out["dna_missing_distance"] = (
        df["dist1"].isna()
    ).astype(int)

    # Email relationship
    p_email = df["P_emaildomain"].fillna(
        "__MISSING__"
    ).astype(str)

    r_email = df["R_emaildomain"].fillna(
        "__MISSING__"
    ).astype(str)

    out["dna_email_match"] = (
        (p_email == r_email) &
        (p_email != "__MISSING__")
    ).astype(int)

    out["dna_email_mismatch"] = (
        (p_email != r_email) &
        (p_email != "__MISSING__") &
        (r_email != "__MISSING__")
    ).astype(int)

    # Distance
    out["dna_distance_log"] = (
        dist.fillna(0).apply(safe_log1p)
    )

    # Amount relative to training median
    if reference is None:
        median_amount = float(
            amount.median()
        )
    else:
        median_amount = reference["amount_median"]

    if median_amount <= 0:
        median_amount = 1

    out["dna_amount_vs_median"] = (
        amount / median_amount
    )

    return out


# ============================================================
# FRAUD CHAIN
# ============================================================

def build_chain_stats(train_df, y_train):

    work = train_df.copy()
    y = pd.Series(
        y_train,
        index=train_df.index
    )

    global_rate = float(y.mean())

    stats = {
        "global_rate": global_rate,
        "entities": {},
        "pairs": {}
    }

    # --------------------------------------------------------
    # Individual entities
    # --------------------------------------------------------

    for col in CHAIN_COLUMNS:

        values = work[col].map(clean_value)

        temp = pd.DataFrame({
            "key": values,
            "fraud": y
        })

        grouped = temp.groupby("key")["fraud"].agg(
            ["count", "sum"]
        )

        entity_stats = {}

        for key, row in grouped.iterrows():

            count = int(row["count"])
            fraud_count = int(row["sum"])

            # Bayesian smoothing
            alpha = 20

            rate = (
                fraud_count +
                alpha * global_rate
            ) / (
                count + alpha
            )

            entity_stats[key] = {
                "count": count,
                "fraud_count": fraud_count,
                "rate": float(rate)
            }

        stats["entities"][col] = entity_stats

    # --------------------------------------------------------
    # Entity pairs
    # --------------------------------------------------------

    pairs = [
        ("card1", "addr1"),
        ("card1", "addr2"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("addr1", "P_emaildomain"),
        ("addr1", "R_emaildomain")
    ]

    for c1, c2 in pairs:

        s1 = work[c1].map(clean_value)
        s2 = work[c2].map(clean_value)

        keys = s1 + "||" + s2

        temp = pd.DataFrame({
            "key": keys,
            "fraud": y
        })

        grouped = temp.groupby("key")["fraud"].agg(
            ["count", "sum"]
        )

        pair_stats = {}

        for key, row in grouped.iterrows():

            count = int(row["count"])
            fraud_count = int(row["sum"])

            alpha = 20

            rate = (
                fraud_count +
                alpha * global_rate
            ) / (
                count + alpha
            )

            pair_stats[key] = {
                "count": count,
                "fraud_count": fraud_count,
                "rate": float(rate)
            }

        stats["pairs"][c1 + "||" + c2] = pair_stats

    return stats


def chain_features(df, stats):

    global_rate = stats["global_rate"]

    result = []

    pairs = [
        ("card1", "addr1"),
        ("card1", "addr2"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("addr1", "P_emaildomain"),
        ("addr1", "R_emaildomain")
    ]

    for idx, row in df.iterrows():

        rates = []
        counts = []

        # Individual entities
        for col in CHAIN_COLUMNS:

            key = clean_value(row[col])

            info = (
                stats["entities"]
                .get(col, {})
                .get(key)
            )

            if info:
                rates.append(info["rate"])
                counts.append(info["count"])
            else:
                rates.append(global_rate)
                counts.append(0)

        # Pairs
        for c1, c2 in pairs:

            key = (
                clean_value(row[c1])
                + "||"
                + clean_value(row[c2])
            )

            pair_name = c1 + "||" + c2

            info = (
                stats["pairs"]
                .get(pair_name, {})
                .get(key)
            )

            if info:
                rates.append(info["rate"])
                counts.append(info["count"])
            else:
                rates.append(global_rate)
                counts.append(0)

        max_rate = max(rates)
        avg_rate = float(np.mean(rates))

        strong_count = sum(
            1 for r in rates
            if r > global_rate * 2
        )

        max_support = max(counts)

        result.append([
            max_rate,
            avg_rate,
            strong_count,
            np.log1p(max_support)
        ])

    return np.asarray(result)


# ============================================================
# BASE MODEL
# ============================================================

def create_base_model():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocess = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", rf)
    ])


# ============================================================
# ANOMALY MODEL
# ============================================================

def create_anomaly_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])


def anomaly_score(model, df):

    raw = model.decision_function(
        df[NUMERIC_FEATURES]
    )

    # Convert anomaly score to 0-100.
    # Lower decision_function = more anomalous.
    score = 50 - (raw * 50)

    return np.clip(score, 0, 100)


# ============================================================
# META FEATURES
# ============================================================

def build_meta_features(
    df,
    base_probability,
    chain,
    dna,
    anomaly
):

    meta = pd.DataFrame(index=df.index)

    meta["base_probability"] = base_probability

    meta["chain_max_rate"] = chain[:, 0]
    meta["chain_avg_rate"] = chain[:, 1]
    meta["chain_strong_count"] = chain[:, 2]
    meta["chain_support"] = chain[:, 3]

    meta["dna_log_amount"] = dna["dna_log_amount"].values
    meta["dna_is_night"] = dna["dna_is_night"].values
    meta["dna_missing_count"] = dna["dna_missing_count"].values
    meta["dna_email_mismatch"] = dna["dna_email_mismatch"].values
    meta["dna_distance_missing"] = dna["dna_missing_distance"].values
    meta["dna_amount_vs_median"] = dna["dna_amount_vs_median"].values

    meta["anomaly_score"] = anomaly

    return meta


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("SENTINEL V4 TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.reset_index(drop=True)

X = df[BASE_FEATURES].copy()
y = df["isFraud"].astype(int)

print("Dataset:", df.shape)
print("Fraud:", int(y.sum()))
print("Normal:", int((y == 0).sum()))

# ------------------------------------------------------------
# FINAL HELD-OUT TEST SET
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# ------------------------------------------------------------
# OOF META TRAINING
# ------------------------------------------------------------

print("\nBuilding OOF meta features...")

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_meta = np.zeros(
    (len(X_train), 12)
)

oof_y = y_train.values

X_train_reset = X_train.reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

for fold, (tr_idx, val_idx) in enumerate(
    skf.split(X_train_reset, y_train_reset),
    1
):

    print(
        "\nFold",
        fold,
        "/ 3"
    )

    X_tr = X_train_reset.iloc[tr_idx]
    X_val = X_train_reset.iloc[val_idx]

    y_tr = y_train_reset.iloc[tr_idx]

    # Base model
    base_model = create_base_model()

    base_model.fit(
        X_tr,
        y_tr
    )

    base_prob = base_model.predict_proba(
        X_val
    )[:, 1]

    # Chain
    chain_stats = build_chain_stats(
        X_tr,
        y_tr
    )

    chain = chain_features(
        X_val,
        chain_stats
    )

    # DNA
    dna_reference = {
        "amount_median": float(
            pd.to_numeric(
                X_tr["TransactionAmt"],
                errors="coerce"
            ).median()
        )
    }

    dna = build_dna(
        X_val,
        dna_reference
    )

    # Anomaly
    anomaly_model = create_anomaly_model()

    anomaly_model.fit(
        X_tr[NUMERIC_FEATURES]
    )

    anomaly = anomaly_score(
        anomaly_model,
        X_val
    )

    meta = build_meta_features(
        X_val,
        base_prob,
        chain,
        dna,
        anomaly
    )

    oof_meta[val_idx] = meta.values

    print(
        "Fold base F1:",
        round(
            f1_score(
                y_train_reset.iloc[val_idx],
                base_prob >= 0.45
            ),
            4
        )
    )


# ------------------------------------------------------------
# META MODEL
# ------------------------------------------------------------

print("\nTraining V4 calibration model...")

meta_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])

meta_model.fit(
    oof_meta,
    oof_y
)

# ------------------------------------------------------------
# FINAL BASE MODEL
# ------------------------------------------------------------

print("\nTraining final base model...")

base_model = create_base_model()

base_model.fit(
    X_train,
    y_train
)

test_base_probability = (
    base_model.predict_proba(X_test)[:, 1]
)

# ------------------------------------------------------------
# FINAL CHAIN
# ------------------------------------------------------------

print("Building final FraudChain...")

chain_stats = build_chain_stats(
    X_train,
    y_train
)

test_chain = chain_features(
    X_test,
    chain_stats
)

# ------------------------------------------------------------
# FINAL DNA
# ------------------------------------------------------------

dna_reference = {
    "amount_median": float(
        pd.to_numeric(
            X_train["TransactionAmt"],
            errors="coerce"
        ).median()
    )
}

test_dna = build_dna(
    X_test,
    dna_reference
)

# ------------------------------------------------------------
# FINAL ANOMALY
# ------------------------------------------------------------

print("Training anomaly detector...")

anomaly_model = create_anomaly_model()

anomaly_model.fit(
    X_train[NUMERIC_FEATURES]
)

test_anomaly = anomaly_score(
    anomaly_model,
    X_test
)

# ------------------------------------------------------------
# FINAL META FEATURES
# ------------------------------------------------------------

test_meta = build_meta_features(
    X_test,
    test_base_probability,
    test_chain,
    test_dna,
    test_anomaly
)

# ------------------------------------------------------------
# FINAL V4 PROBABILITY
# ------------------------------------------------------------

v4_probability = meta_model.predict_proba(
    test_meta
)[:, 1]

# ------------------------------------------------------------
# THRESHOLD ANALYSIS
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("V4 THRESHOLD ANALYSIS")
print("=" * 60)

results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    pred = (
        v4_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    fp = int(
        ((pred == 1) & (y_test.values == 0)).sum()
    )

    fn = int(
        ((pred == 0) & (y_test.values == 1)).sum()
    )

    cost = fp * 1 + fn * 5

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "estimated_cost": cost
    })

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False
    )
)

best = results_df.loc[
    results_df["f1"].idxmax()
]

cost_best = results_df.loc[
    results_df["estimated_cost"].idxmin()
]

print("\nBEST F1")
print(best)

print("\nBEST COST")
print(cost_best)

# ------------------------------------------------------------
# OPERATING THRESHOLD
# ------------------------------------------------------------

OPERATING_THRESHOLD = float(
    best["threshold"]
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

artifact = {
    "version": "4.0",

    "base_model": base_model,

    "meta_model": meta_model,

    "anomaly_model": anomaly_model,

    "chain_stats": chain_stats,

    "dna_reference": dna_reference,

    "threshold": OPERATING_THRESHOLD,

    "numeric_features": NUMERIC_FEATURES,

    "categorical_features": CATEGORICAL_FEATURES,

    "base_features": BASE_FEATURES,

    "test_metrics": {
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "false_positives": int(best["false_positives"]),
        "false_negatives": int(best["false_negatives"]),
        "estimated_cost": int(best["estimated_cost"])
    }
}

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    artifact,
    MODEL_PATH
)

results_df.to_csv(
    "data/processed/sentinel_v4_threshold_analysis.csv",
    index=False
)

print("\n")
print("=" * 60)
print("V4 COMPLETE")
print("=" * 60)

print(
    "\nModel saved:",
    MODEL_PATH
)

print(
    "Threshold:",
    OPERATING_THRESHOLD
)

print(
    "Precision:",
    round(best["precision"], 4)
)

print(
    "Recall:",
    round(best["recall"], 4)
)

print(
    "F1:",
    round(best["f1"], 4)
)

print(
    "FP:",
    int(best["false_positives"])
)

print(
    "FN:",
    int(best["false_negatives"])
)