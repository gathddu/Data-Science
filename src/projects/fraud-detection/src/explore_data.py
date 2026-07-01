import pandas as pd
import numpy as np

# load raw data
print("Loading data...")
train_tx = pd.read_csv('../data/raw/train_transaction.csv')
train_id = pd.read_csv('../data/raw/train_identity.csv')

print("\n=== TRANSACTION DATA ===")
print(f"Shape: {train_tx.shape}")
print(f"Columns: {train_tx.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_tx.head())
print(f"\nData types:")
print(train_tx.dtypes)
print(f"\nMissing values:")
print(train_tx.isnull().sum())

print("\n=== IDENTITY DATA ===")
print(f"Shape: {train_id.shape}")
print(f"Columns: {train_id.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_id.head())

print("\n=== FRAUD DISTRIBUTION ===")
fraud_counts = train_tx['isFraud'].value_counts()
print(fraud_counts)
print(f"\nFraud rate: {train_tx['isFraud'].mean()*100:.2f}%")

print("\n=== MERGE DATA ===")
merged = train_tx.merge(train_id, on='TransactionID', how='left')
print(f"Merged shape: {merged.shape}")
print(f"Merged columns: {len(merged.columns)}")
