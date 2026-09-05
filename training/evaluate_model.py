import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_PATH = "data/processed/sentinel_transactions.csv"
MODEL_PATH = "models/sentinel_fraud_model.pkl"
OUTPUT_PATH = "evaluation/metrics.json"

os.makedirs("evaluation", exist_ok=True)

def evaluate():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact.get("threshold", 0.30)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    precision = precision_score(y_test, y_preds, zero_division=0)
    recall = recall_score(y_test, y_preds, zero_division=0)
    f1 = f1_score(y_test, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    fp_cost = fp * 15.0  # $15 per false alert

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "false_positive_cost": f"${fp_cost:,.2f}",
        "sample_count": len(y_test)
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model Evaluation Completed Successfully:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    evaluate()