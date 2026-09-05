

import os
import joblib
import pandas as pd
import numpy as np

from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

V1_MODEL_PATH = "models/sentinel_fraud_model.pkl"
V4_MODEL_PATH = "models/sentinel_v4_model.pkl"


# ============================================================
# LOAD V1 MODEL — CHAMPION MODEL
# ============================================================

print("=" * 60)
print("SENTINEL AI RISK INTELLIGENCE")
print("=" * 60)

print("Loading champion V1 fraud model...")

v1_data = joblib.load(V1_MODEL_PATH)

model = v1_data["model"]
THRESHOLD = float(v1_data["threshold"])

NUMERIC_FEATURES = v1_data["numeric_features"]
CATEGORICAL_FEATURES = v1_data["categorical_features"]

print("V1 model loaded successfully!")
print("Operating threshold:", THRESHOLD)


# ============================================================
# LOAD V4 INTELLIGENCE ARTIFACTS
# ============================================================

chain_stats = None
anomaly_model = None
dna_reference = None

if os.path.exists(V4_MODEL_PATH):

    try:

        v4_data = joblib.load(V4_MODEL_PATH)

        chain_stats = v4_data.get("chain_stats")
        anomaly_model = v4_data.get("anomaly_model")
        dna_reference = v4_data.get("dna_reference")

        print("FraudChain intelligence loaded.")
        print("Anomaly intelligence loaded.")

    except Exception as e:

        print(
            "Warning: V4 intelligence unavailable:",
            str(e)
        )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SENTINEL AI Fraud Intelligence",
    description=(
        "Explainable AI-powered transaction "
        "fraud risk intelligence system."
    ),
    version="4.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5503",
        "http://localhost:5503",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class Transaction(BaseModel):

    TransactionDT: float
    TransactionAmt: float

    card1: float
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None

    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None

    ProductCD: str
    card4: str
    card6: str

    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "SENTINEL V1 Champion",
        "intelligence_layer": "FraudDNA + FraudChain",
        "threshold": THRESHOLD
    }


# ============================================================
# FRAUD DNA
# ============================================================

def calculate_dna(transaction):

    signals = []
    dna_score = 0

    amount = transaction.TransactionAmt

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    if amount > 1000:

        dna_score += 25

        signals.append(
            "High transaction amount"
        )

    elif amount > 500:

        dna_score += 12

        signals.append(
            "Elevated transaction amount"
        )

    # --------------------------------------------------------
    # Transaction time
    # --------------------------------------------------------

    seconds_day = (
        transaction.TransactionDT % 86400
    )

    hour = int(
        seconds_day // 3600
    )

    if hour < 6 or hour >= 23:

        dna_score += 15

        signals.append(
            "Unusual transaction time"
        )

    # --------------------------------------------------------
    # Email relationship
    # --------------------------------------------------------

    p_email = transaction.P_emaildomain
    r_email = transaction.R_emaildomain

    if (
        p_email
        and r_email
        and p_email.lower() != r_email.lower()
    ):

        dna_score += 15

        signals.append(
            "Purchaser and recipient email domains differ"
        )

    # --------------------------------------------------------
    # Missing distance
    # --------------------------------------------------------

    if transaction.dist1 is None:

        dna_score += 5

        signals.append(
            "Transaction distance unavailable"
        )

    elif transaction.dist1 > 500:

        dna_score += 20

        signals.append(
            "Unusual transaction distance"
        )

    elif transaction.dist1 > 100:

        dna_score += 10

        signals.append(
            "Elevated transaction distance"
        )

    # --------------------------------------------------------
    # Missing information
    # --------------------------------------------------------

    missing_count = 0

    values = [
        transaction.card2,
        transaction.card3,
        transaction.card5,
        transaction.addr1,
        transaction.addr2,
        transaction.dist1,
        transaction.P_emaildomain,
        transaction.R_emaildomain
    ]

    for value in values:

        if value is None:
            missing_count += 1

    if missing_count >= 4:

        dna_score += 10

        signals.append(
            "Multiple transaction attributes unavailable"
        )

    return min(dna_score, 100), signals


# ============================================================
# FRAUD CHAIN
# ============================================================

def clean_value(value):

    if value is None:
        return "__MISSING__"

    return str(value)


def calculate_chain_risk(transaction):

    if chain_stats is None:

        return 0, [], {
            "max_historical_fraud_rate": 0,
            "known_entity_links": 0
        }

    global_rate = chain_stats.get(
        "global_rate",
        0.027
    )

    entities = chain_stats.get(
        "entities",
        {}
    )

    pairs = chain_stats.get(
        "pairs",
        {}
    )

    rates = []
    supports = []

    linked_entities = []

    entity_columns = [
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain"
    ]

    # --------------------------------------------------------
    # Individual entity links
    # --------------------------------------------------------

    for column in entity_columns:

        value = getattr(
            transaction,
            column
        )

        key = clean_value(value)

        info = (
            entities
            .get(column, {})
            .get(key)
        )

        if info:

            rates.append(
                float(info["rate"])
            )

            supports.append(
                int(info["count"])
            )

            linked_entities.append(
                column
            )

    # --------------------------------------------------------
    # Pair links
    # --------------------------------------------------------

    pair_definitions = [
        ("card1", "addr1"),
        ("card1", "addr2"),
        ("card1", "P_emaildomain"),
        ("card1", "R_emaildomain"),
        ("addr1", "P_emaildomain"),
        ("addr1", "R_emaildomain")
    ]

    for c1, c2 in pair_definitions:

        v1 = clean_value(
            getattr(transaction, c1)
        )

        v2 = clean_value(
            getattr(transaction, c2)
        )

        key = v1 + "||" + v2

        pair_name = (
            c1 + "||" + c2
        )

        info = (
            pairs
            .get(pair_name, {})
            .get(key)
        )

        if info:

            rates.append(
                float(info["rate"])
            )

            supports.append(
                int(info["count"])
            )

            linked_entities.append(
                pair_name
            )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if not rates:

        return 0, [], {
            "max_historical_fraud_rate": global_rate,
            "known_entity_links": 0
        }

    max_rate = max(rates)

    # Convert historical fraud rate into a risk score.
    #
    # Global fraud rate is approximately 2-3%.
    # We deliberately cap this layer so it does not override
    # the champion ML model.

    chain_risk = min(
        100,
        (max_rate / max(global_rate, 0.001)) * 20
    )

    signals = []

    if max_rate >= global_rate * 5:

        signals.append(
            "Strong historical fraud association detected"
        )

    elif max_rate >= global_rate * 2:

        signals.append(
            "Elevated historical fraud association detected"
        )

    if len(linked_entities) >= 3:

        signals.append(
            "Multiple transaction entities are historically linked"
        )

    details = {
        "max_historical_fraud_rate": round(
            max_rate,
            4
        ),
        "known_entity_links": len(
            linked_entities
        ),
        "maximum_entity_support": max(
            supports
        ) if supports else 0
    }

    return (
        min(chain_risk, 100),
        signals,
        details
    )


# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

def calculate_anomaly_risk(transaction):

    if anomaly_model is None:

        return 0

    try:

        row = pd.DataFrame([{
            feature: getattr(
                transaction,
                feature
            )
            for feature in NUMERIC_FEATURES
        }])

        raw = anomaly_model.decision_function(
            row
        )[0]

        score = 50 - (
            float(raw) * 50
        )

        return float(
            np.clip(
                score,
                0,
                100
            )
        )

    except Exception:

        return 0


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score >= 70:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


# ============================================================
# DECISION
# ============================================================

def get_decision(
    fraud_probability,
    combined_score
):

    # Champion V1 model remains authoritative
    # for BLOCK decisions.

    if fraud_probability >= THRESHOLD:

        return "BLOCK"

    # Intelligence layer can route elevated
    # but sub-threshold transactions to review.

    if combined_score >= 50:

        return "REVIEW"

    return "APPROVE"


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze_transaction(
    transaction: Transaction
):

    # --------------------------------------------------------
    # Convert request
    # --------------------------------------------------------

    data = transaction.model_dump()

    df = pd.DataFrame([
        data
    ])

    # --------------------------------------------------------
    # V1 AI MODEL
    # --------------------------------------------------------

    fraud_probability = float(
        model.predict_proba(df)[0][1]
    )

    ai_risk_score = (
        fraud_probability * 100
    )

    # --------------------------------------------------------
    # FRAUD DNA
    # --------------------------------------------------------

    dna_score, dna_signals = (
        calculate_dna(
            transaction
        )
    )

    # --------------------------------------------------------
    # FRAUD CHAIN
    # --------------------------------------------------------

    chain_score, chain_signals, chain_details = (
        calculate_chain_risk(
            transaction
        )
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    anomaly_score = (
        calculate_anomaly_risk(
            transaction
        )
    )

    # --------------------------------------------------------
    # INTELLIGENCE SCORE
    # --------------------------------------------------------

    # AI remains dominant.
    #
    # DNA / Chain / anomaly are bounded supporting signals.

    combined_score = (
        ai_risk_score * 0.75
        + dna_score * 0.10
        + chain_score * 0.10
        + anomaly_score * 0.05
    )

    combined_score = float(
        np.clip(
            combined_score,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    risk_level = get_risk_level(
        combined_score
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision = get_decision(
        fraud_probability,
        combined_score
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    signals = []

    if fraud_probability >= THRESHOLD:

        signals.append(
            "AI fraud probability exceeded the operating threshold"
        )

    elif fraud_probability >= 0.30:

        signals.append(
            "AI model detected elevated fraud probability"
        )

    signals.extend(
        dna_signals
    )

    signals.extend(
        chain_signals
    )

    if anomaly_score >= 70:

        signals.append(
            "Transaction exhibits anomalous characteristics"
        )

    # Remove duplicates
    signals = list(
        dict.fromkeys(signals)
    )

    # Limit to strongest explanations
    signals = signals[:6]

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if decision == "BLOCK":

        explanation = (
            "SENTINEL blocked the transaction because "
            "the champion AI fraud model exceeded its "
            "validated operating threshold."
        )

    elif decision == "REVIEW":

        explanation = (
            "SENTINEL detected elevated risk through "
            "combined AI, behavioral, and transaction-link "
            "signals. Additional verification is recommended."
        )

    else:

        explanation = (
            "SENTINEL found no sufficiently strong fraud "
            "evidence. The transaction remains below the "
            "configured intervention thresholds."
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "ai_fraud_probability": round(
            fraud_probability,
            6
        ),

        "ai_risk_score": round(
            ai_risk_score,
            2
        ),

        "behavioral_risk": round(
            dna_score,
            2
        ),

        "chain_risk": round(
            chain_score,
            2
        ),

        "anomaly_risk": round(
            anomaly_score,
            2
        ),

        "combined_risk_score": round(
            combined_score,
            2
        ),

        "risk_level": risk_level,

        "decision": decision,

        "risk_signals": signals,

        "explanation": explanation,

        "fraud_chain": chain_details,

        "model": "SENTINEL-V1-CHAMPION",

        "threshold": THRESHOLD
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
    name="static"
)


@app.get("/")
def frontend():

    return FileResponse(
        "static/index.html"
    )