import pandas as pd
import joblib


# =========================================
# LOAD SAVED SENTINEL MODEL
# =========================================

print("Loading SENTINEL fraud model...")

model_data = joblib.load(
    "models/sentinel_fraud_model.pkl"
)

sentinel_model = model_data["model"]
threshold = model_data["threshold"]


# =========================================
# ANALYZE TRANSACTION
# =========================================

def analyze_transaction(transaction):

    # =====================================
    # AI FRAUD PREDICTION
    # =====================================

    transaction_df = pd.DataFrame([transaction])

    fraud_probability = sentinel_model.predict_proba(
        transaction_df
    )[0][1]

    # Convert probability to percentage
    ai_risk_score = fraud_probability * 100


    # =====================================
    # RULE-BASED RISK ANALYSIS
    # =====================================

    rule_risk_score = 0
    risk_signals = []


    # Very high / high transaction amount
    if transaction["TransactionAmt"] > 1000:

        rule_risk_score += 30

        risk_signals.append(
            "Very high transaction amount"
        )

    elif transaction["TransactionAmt"] > 500:

        rule_risk_score += 15

        risk_signals.append(
            "High transaction amount"
        )


    # Transaction distance analysis
    dist1 = transaction.get("dist1", 0)

    if pd.notna(dist1):

        if dist1 > 500:

            rule_risk_score += 25

            risk_signals.append(
                "Very unusual transaction distance"
            )

        elif dist1 > 100:

            rule_risk_score += 15

            risk_signals.append(
                "Unusual transaction distance"
            )


    # Email domain mismatch
    p_email = transaction.get("P_emaildomain")
    r_email = transaction.get("R_emaildomain")

    if (
        pd.notna(p_email)
        and pd.notna(r_email)
        and p_email != r_email
    ):

        rule_risk_score += 10

        risk_signals.append(
            "Purchaser and recipient email domains differ"
        )


    # Debit card signal
    if transaction.get("card6") == "debit":

        rule_risk_score += 5

        risk_signals.append(
            "Debit card transaction"
        )


    # Limit maximum rule score
    rule_risk_score = min(
        rule_risk_score,
        100
    )


    # =====================================
    # HYBRID RISK SCORE
    # =====================================

    combined_risk_score = (
        ai_risk_score * 0.70
        +
        rule_risk_score * 0.30
    )


    # =====================================
    # RISK LEVEL
    # =====================================

    if combined_risk_score >= 70:

        risk_level = "CRITICAL"

    elif combined_risk_score >= 50:

        risk_level = "HIGH"

    elif combined_risk_score >= 30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # =====================================
    # FINAL DECISION
    # =====================================

    if fraud_probability >= threshold:

        decision = "FRAUD"

    elif combined_risk_score >= 50:

        decision = "REVIEW"

    else:

        decision = "SAFE"


    # =====================================
    # DEFAULT SIGNAL
    # =====================================

    if not risk_signals:

        risk_signals.append(
            "No major rule-based risk signals detected"
        )


    # =====================================
    # EXPLAINABLE AI SUMMARY
    # =====================================

    explanation = []

    if decision == "FRAUD":

        explanation.append(
            "The AI model detected a high probability of fraud."
        )

        explanation.append(
            "The fraud probability exceeded the optimized detection threshold."
        )


    elif decision == "REVIEW":

        explanation.append(
            "The AI model and rule-based risk signals indicate that the transaction requires manual review."
        )

        explanation.append(
            "The combined risk score exceeded the manual review threshold."
        )


    else:

        explanation.append(
            "The AI model found a low probability of fraud."
        )

        if (
            risk_signals
            and risk_signals[0]
            != "No major rule-based risk signals detected"
        ):

            explanation.append(
                "Some risk signals were detected, but they were not strong enough to classify the transaction as fraudulent."
            )

        else:

            explanation.append(
                "No major risk signals were detected."
            )


    # =====================================
    # RETURN RESULT
    # =====================================

    return {

        "ai_fraud_probability":
            round(ai_risk_score, 2),

        "rule_risk_score":
            round(rule_risk_score, 2),

        "combined_risk_score":
            round(combined_risk_score, 2),

        "risk_level":
            risk_level,

        "decision":
            decision,

        "risk_signals":
            risk_signals,

        "explanation":
            explanation
    }


# =========================================
# TEST TRANSACTION
# =========================================

test_transaction = {

    "TransactionDT": 500000,

    "TransactionAmt": 1200.50,

    "card1": 15000,

    "card2": 500,

    "card3": 150,

    "card5": 200,

    "addr1": 300,

    "addr2": 87,

    "dist1": 500,

    "ProductCD": "W",

    "card4": "visa",

    "card6": "credit",

    "P_emaildomain": "gmail.com",

    "R_emaildomain": "gmail.com"
}


# =========================================
# ANALYZE TRANSACTION
# =========================================

result = analyze_transaction(
    test_transaction
)


# =========================================
# DISPLAY RESULTS
# =========================================

print("\n" + "=" * 45)

print(
    "SENTINEL HYBRID FRAUD RISK ANALYSIS"
)

print("=" * 45)


print(
    "AI Fraud Probability:",
    str(result["ai_fraud_probability"]) + "%"
)


print(
    "Rule Risk Score:",
    str(result["rule_risk_score"]) + "/100"
)


print(
    "Combined Risk Score:",
    str(result["combined_risk_score"]) + "/100"
)


print(
    "Risk Level:",
    result["risk_level"]
)


print(
    "Decision:",
    result["decision"]
)


# =========================================
# RISK SIGNALS
# =========================================

print("\nRisk Signals:")

for signal in result["risk_signals"]:

    print("-", signal)


# =========================================
# AI EXPLANATION
# =========================================

print("\nExplanation:")

for item in result["explanation"]:

    print("-", item)