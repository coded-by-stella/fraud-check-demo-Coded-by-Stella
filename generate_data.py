# create a tiny, fake transactions CSV
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

data = pd.DataFrame({
    "transaction_id": np.arange(1, N+1),
    "amount": rng.integers(5, 800, size=N),                      
    "country": rng.choice(["NO", "SE", "DK", "IT", "DE", "FR"], N),
    "channel": rng.choice(["web", "app", "pos"], N, p=[0.5, 0.4, 0.1]),
    "merchant_id": rng.integers(100, 200, size=N),
    "hour_of_day": rng.integers(0, 24, size=N),
    "is_new_device": rng.choice([0, 1], N, p=[0.8, 0.2]),
    "ip_risk_score": rng.uniform(0, 1, size=N).round(3)
})

# Simple hidden rule to label some fraud
fraud_mask = (
    ((data["amount"] > 500) & (data["is_new_device"] == 1) & (data["ip_risk_score"] > 0.7))
    | ((data["hour_of_day"].isin([0, 1, 2, 3])) & (data["ip_risk_score"] > 0.85))
)

data["is_fraud"] = fraud_mask.astype(int)
data.to_csv("transactions.csv", index=False)
print("Saved transactions.csv with", data["is_fraud"].sum(), "fraud rows")
