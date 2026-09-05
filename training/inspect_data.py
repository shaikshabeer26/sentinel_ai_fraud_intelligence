import pandas as pd

# Only load a small, important set of columns
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

print("Loading transaction sample...")

transactions = pd.read_csv(
    "data/raw/train_transaction.csv",
    usecols=columns,
    nrows=10000
)

print("\n--- DATASET SHAPE ---")
print("Transactions:", transactions.shape)

print("\n--- FRAUD DISTRIBUTION ---")
print(transactions["isFraud"].value_counts())

print("\n--- FRAUD PERCENTAGE ---")
print(
    transactions["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\n--- FIRST 5 ROWS ---")
print(transactions.head())

print("\n--- MISSING VALUES ---")
missing = transactions.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n--- DATA TYPES ---")
print(transactions.dtypes)

print("\nData inspection completed successfully.")