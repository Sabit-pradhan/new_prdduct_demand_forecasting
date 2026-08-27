import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from predict import load_model, predict

MODEL_PATH = "models/fraud_model.pkl"
LEGIT = {"amount":500,  "hour":14,"day_of_week":2,"merchant_type":"grocery",  "customer_age":35,"num_prev_txns":120,"avg_txn_amount":500}
FRAUD = {"amount":95000,"hour":2, "day_of_week":6,"merchant_type":"jewellery","customer_age":21,"num_prev_txns":1,  "avg_txn_amount":300}

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model not found. Run train.py first."

def test_model_loads():
    assert load_model(MODEL_PATH) is not None

def test_prediction_keys():
    model = load_model(MODEL_PATH)
    r = predict(model, LEGIT.copy())
    assert "prediction" in r and "confidence" in r and "risk_level" in r

def test_prediction_is_binary():
    model = load_model(MODEL_PATH)
    r = predict(model, LEGIT.copy())
    assert r["prediction"] in ["FRAUD", "LEGITIMATE"]

def test_confidence_range():
    model = load_model(MODEL_PATH)
    r = predict(model, LEGIT.copy())
    assert 0 <= r["confidence"] <= 100

def test_risk_level_valid():
    model = load_model(MODEL_PATH)
    r = predict(model, LEGIT.copy())
    assert r["risk_level"] in ["HIGH", "MEDIUM", "LOW"]

def test_legitimate_prediction():
    model = load_model(MODEL_PATH)
    assert predict(model, LEGIT.copy())["prediction"] == "LEGITIMATE"

def test_fraud_prediction():
    model = load_model(MODEL_PATH)
    assert predict(model, FRAUD.copy())["prediction"] == "FRAUD"
