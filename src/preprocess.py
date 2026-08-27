import pandas as pd
import numpy as np

MERCHANT_MAP = {
    "grocery":0,"restaurant":1,"retail":2,"pharmacy":3,
    "fuel":4,"electronics":5,"online":6,"jewellery":7,
}

def load_data(filepath="../data/transactions.csv"):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    return df

def encode_features(df):
    df = df.copy()
    df["merchant_type"] = df["merchant_type"].map(MERCHANT_MAP).fillna(2)
    df["is_night"]      = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["amount_vs_avg"] = (df["amount"] / (df["avg_txn_amount"] + 1)).round(2)
    df["amount_log"]    = np.log1p(df["amount"]).round(4)
    return df

def get_features_and_target(df):
    drop_cols = ["transaction_id", "is_fraud"]
    X = df.drop(columns=drop_cols)
    y = df["is_fraud"]
    return X, y

def preprocess(filepath="../data/transactions.csv"):
    df = load_data(filepath)
    df = encode_features(df)
    X, y = get_features_and_target(df)
    print(f"Features: {list(X.columns)}")
    return X, y

if __name__ == "__main__":
    X, y = preprocess()
    print(f"X shape: {X.shape}")
