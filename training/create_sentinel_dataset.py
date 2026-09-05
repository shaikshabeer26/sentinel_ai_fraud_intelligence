import pandas as pd

# Features selected for the first SENTINEL version
columns = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",

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
    "R_emaildomain"
]

print("Creating compact SENTINEL dataset...")

chunks = []

# Read the original file in small chunks
for chunk in pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=columns,
    chunksize=5000
):
    chunks.append(chunk)

    # Stop after collecting 50,000 transactions
    if sum(len(x) for x in chunks) >= 50000:
        break

print("Combining data...")

sentinel_data = pd.concat(chunks, ignore_index=True)

# Keep exactly 50,000 rows
sentinel_data = sentinel_data.iloc[:50000]

print("Final dataset shape:", sentinel_data.shape)

# Create processed-data folder
import os
os.makedirs("data/processed", exist_ok=True)

# Save compact dataset
sentinel_data.to_csv(
    "data/processed/sentinel_transactions.csv",
    index=False
)

print("\nDataset saved successfully!")
print("Location: data/processed/sentinel_transactions.csv")

print("\nFraud distribution:")
print(sentinel_data["isFraud"].value_counts())

print("\nFraud percentage:")
print(
    sentinel_data["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
)