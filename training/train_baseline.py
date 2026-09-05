import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("Loading SENTINEL dataset...")

data = pd.read_csv(
    "data/processed/sentinel_transactions.csv"
)

print("Dataset shape:", data.shape)


# --------------------------------
# TARGET
# --------------------------------

X = data.drop(
    columns=["isFraud", "TransactionID"]
)

y = data["isFraud"]


# --------------------------------
# IDENTIFY FEATURE TYPES
# --------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Updated to avoid the Pandas warning
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------
# NUMERIC PIPELINE
# --------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------
# CATEGORICAL PIPELINE
# --------------------------------

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


# --------------------------------
# PREPROCESSING
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------
# MODEL
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# --------------------------------
# FULL PIPELINE
# --------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------
# TRAIN
# --------------------------------

print("\nTraining SENTINEL baseline model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------
# PREDICT PROBABILITIES
# --------------------------------

print("\nGenerating fraud probabilities...")

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------
# DEFAULT THRESHOLD = 0.50
# --------------------------------

y_pred = (
    y_probability >= 0.50
).astype(int)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# --------------------------------
# BASELINE RESULTS
# --------------------------------

print("\n================================")
print("SENTINEL BASELINE RESULTS")
print("================================")

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------
# THRESHOLD ANALYSIS
# --------------------------------

print("\n================================")
print("SENTINEL THRESHOLD ANALYSIS")
print("================================")

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


results = []


for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    precision_t = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall_t = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1_t = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    tn_t, fp_t, fn_t, tp_t = matrix.ravel()

    false_positive_rate = (
        fp_t / (fp_t + tn_t)
        if (fp_t + tn_t) > 0
        else 0
    )

    fraud_detection_rate = (
        tp_t / (tp_t + fn_t)
        if (tp_t + fn_t) > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "precision": precision_t,
        "recall": recall_t,
        "f1": f1_t,
        "false_positives": fp_t,
        "false_negatives": fn_t,
        "true_positives": tp_t,
        "true_negatives": tn_t,
        "false_positive_rate": false_positive_rate,
        "fraud_detection_rate": fraud_detection_rate
    })


threshold_df = pd.DataFrame(results)


# --------------------------------
# DISPLAY THRESHOLD TABLE
# --------------------------------

print(
    threshold_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positives",
            "false_negatives",
            "fraud_detection_rate"
        ]
    ].round(4).to_string(index=False)
)


# --------------------------------
# FIND BEST F1 THRESHOLD
# --------------------------------

best_f1_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print("\n================================")
print("BEST F1 THRESHOLD")
print("================================")

print(
    "Threshold:",
    best_f1_row["threshold"]
)

print(
    "Precision:",
    round(best_f1_row["precision"], 4)
)

print(
    "Recall:",
    round(best_f1_row["recall"], 4)
)

print(
    "F1:",
    round(best_f1_row["f1"], 4)
)

print(
    "False Positives:",
    int(best_f1_row["false_positives"])
)

print(
    "False Negatives:",
    int(best_f1_row["false_negatives"])
)


# --------------------------------
# COST-AWARE ANALYSIS
# --------------------------------
#
# Example cost assumptions:
#
# False Positive:
# legitimate transaction incorrectly flagged
#
# False Negative:
# fraudulent transaction incorrectly allowed
#
# These are deliberately explicit and can
# later be changed to merchant-specific values.
# --------------------------------

FP_COST = 1
FN_COST = 5


threshold_df["total_cost"] = (
    threshold_df["false_positives"] * FP_COST
    +
    threshold_df["false_negatives"] * FN_COST
)


best_cost_row = threshold_df.loc[
    threshold_df["total_cost"].idxmin()
]


print("\n================================")
print("COST-AWARE THRESHOLD")
print("================================")

print(
    "Assumed false-positive cost:",
    FP_COST
)

print(
    "Assumed false-negative cost:",
    FN_COST
)

print(
    "Recommended threshold:",
    best_cost_row["threshold"]
)

print(
    "Precision:",
    round(best_cost_row["precision"], 4)
)

print(
    "Recall:",
    round(best_cost_row["recall"], 4)
)

print(
    "F1:",
    round(best_cost_row["f1"], 4)
)

print(
    "False Positives:",
    int(best_cost_row["false_positives"])
)

print(
    "False Negatives:",
    int(best_cost_row["false_negatives"])
)

print(
    "Estimated relative cost:",
    int(best_cost_row["total_cost"])
)


# --------------------------------
# SAVE THRESHOLD RESULTS
# --------------------------------

threshold_df.to_csv(
    "data/processed/sentinel_threshold_analysis.csv",
    index=False
)

print(
    "\nThreshold analysis saved to:"
)

print(
    "data/processed/sentinel_threshold_analysis.csv"
)


print("\n================================")
print("SENTINEL EVALUATION COMPLETE")
print("================================")