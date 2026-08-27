import pandas as pd
import numpy as np
import os

def generate_transactions(n_samples=1000, fraud_rate=0.10, random_state=42):
    print("Generating sample transaction data...")
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    legit = {
        "transaction_id" : [f"TXN{str(i).zfill(5)}" for i in range(n_legit)],
        "amount"         : np.random.lognormal(6.5, 1.0, n_legit).round(2),
        "hour"           : np.random.choice(range(8, 22), n_legit),
        "day_of_week"    : np.random.randint(0, 7, n_legit),
        "merchant_type"  : np.random.choice(["grocery","restaurant","retail","pharmacy","fuel"], n_legit),
        "customer_age"   : np.random.randint(22, 65, n_legit),
        "num_prev_txns"  : np.random.randint(5, 200, n_legit),
        "avg_txn_amount" : np.random.lognormal(6.2, 0.8, n_legit).round(2),
        "is_fraud"       : [0] * n_legit,
    }
    fraud = {
        "transaction_id" : [f"TXN{str(i).zfill(5)}" for i in range(n_legit, n_samples)],
        "amount"         : np.random.lognormal(8.5, 1.5, n_fraud).round(2),
        "hour"           : np.random.choice([0,1,2,3,4,22,23], n_fraud),
        "day_of_week"    : np.random.randint(0, 7, n_fraud),
        "merchant_type"  : np.random.choice(["electronics","online","jewellery","fuel","retail"], n_fraud),
        "customer_age"   : np.random.randint(18, 70, n_fraud),
        "num_prev_txns"  : np.random.randint(0, 10, n_fraud),
        "avg_txn_amount" : np.random.lognormal(5.5, 0.5, n_fraud).round(2),
        "is_fraud"       : [1] * n_fraud,
    }
    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_transactions()
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/transactions.csv", index=False)
    print(f"Created {len(df)} transactions")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print("Saved to: ../data/transactions.csv")
