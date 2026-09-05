
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


print("Loading SENTINEL dataset...")

data = pd.read_csv(
    "data/processed/sentinel_transactions.csv"
)


# ------------------------------
# FEATURES AND TARGET
# ------------------------------

X = data.drop(
    columns=["isFraud", "TransactionID"]
)

y = data["isFraud"]


# ------------------------------
# FEATURE TYPES
# ------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ------------------------------
# TRAIN / TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------
# NUMERIC PROCESSING
# ------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ------------------------------
# CATEGORICAL PROCESSING
# ------------------------------

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


# ------------------------------
# PREPROCESSOR
# ------------------------------

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


# ------------------------------
# MODEL
# ------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ------------------------------
# PIPELINE
# ------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("Training model...")

pipeline.fit(
    X_train,
    y_train
)


# ------------------------------
# GET FRAUD PROBABILITIES
# ------------------------------

fraud_probabilities = pipeline.predict_proba(X_test)[:, 1]


# ------------------------------
# TEST THRESHOLDS
# ------------------------------

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


print("\n========================================")
print("SENTINEL THRESHOLD OPTIMIZATION")
print("========================================\n")


best_f1 = 0
best_threshold = 0


for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
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

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("----------------------------------------")
    print("Threshold:", threshold)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("Confusion Matrix:")
    print(matrix)

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold
print("\n========================================")
print("BEST THRESHOLD")
print("========================================")

print("Threshold:", best_threshold)
print("Best F1 Score:", round(best_f1, 4))
