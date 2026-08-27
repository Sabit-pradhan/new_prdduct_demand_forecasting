import os, sys, joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

MERCHANT_MAP = {
    "grocery":0,"restaurant":1,"retail":2,"pharmacy":3,
    "fuel":4,"electronics":5,"online":6,"jewellery":7,
}

def load_model(model_path="../models/fraud_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)

def prepare_transaction(txn: dict) -> pd.DataFrame:
    df = pd.DataFrame([txn])
    df["merchant_type"] = df["merchant_type"].map(MERCHANT_MAP).fillna(2)
    df["is_night"]      = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["amount_vs_avg"] = (df["amount"] / (df["avg_txn_amount"] + 1)).round(2)
    df["amount_log"]    = np.log1p(df["amount"]).round(4)
    return df

def predict(model, txn: dict) -> dict:
    features    = prepare_transaction(txn)
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    risk = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.40 else "LOW"
    return {
        "prediction" : "FRAUD" if prediction == 1 else "LEGITIMATE",
        "confidence" : round(probability * 100, 1),
        "risk_level" : risk,
    }

if __name__ == "__main__":
    model = load_model()
    tests = [
        {"amount":450,   "hour":14,"day_of_week":2,"merchant_type":"grocery",     "customer_age":35,"num_prev_txns":120,"avg_txn_amount":500},
        {"amount":45000, "hour":3, "day_of_week":6,"merchant_type":"electronics", "customer_age":22,"num_prev_txns":1,  "avg_txn_amount":300},
        {"amount":1200,  "hour":19,"day_of_week":5,"merchant_type":"restaurant",  "customer_age":42,"num_prev_txns":85, "avg_txn_amount":1100},
        {"amount":120000,"hour":1, "day_of_week":0,"merchant_type":"jewellery",   "customer_age":21,"num_prev_txns":1,  "avg_txn_amount":300},
    ]
    print("\n" + "="*55)
    print("  Testing Predictions")
    print("="*55)
    for i, txn in enumerate(tests, 1):
        r = predict(model, txn)
        print(f"\nTransaction {i}: Rs.{txn['amount']:,} | {txn['merchant_type']} | Hour:{txn['hour']}")
        print(f"  Prediction : {r['prediction']}")
        print(f"  Confidence : {r['confidence']}%")
        print(f"  Risk Level : {r['risk_level']}")
