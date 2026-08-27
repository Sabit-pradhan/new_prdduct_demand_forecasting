import os, sys
import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import load_model, predict

st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="centered")

@st.cache_resource
def get_model():
    return load_model("models/fraud_model.pkl")

st.title("🔍 Fraud Detection System")
st.markdown("**MLOps Course — Suresh D R | AI Product Developer**")
st.markdown("---")

st.subheader("📋 Transaction Details")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (Rs.)", min_value=1, max_value=1000000, value=5000, step=100)
    hour = st.selectbox("Hour of Transaction", options=list(range(0, 24)), index=14,
        format_func=lambda x: f"{x:02d}:00  {'🌙 Late Night' if x < 5 or x >= 22 else '☀️ Daytime' if 8 <= x < 20 else '🌆 Evening'}")
    day_of_week = st.selectbox("Day of Week",
        options=[0,1,2,3,4,5,6],
        format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x], index=2)
    merchant_type = st.selectbox("Merchant Type",
        options=["grocery","restaurant","retail","pharmacy","fuel","electronics","online","jewellery"], index=0)

with col2:
    customer_age    = st.slider("Customer Age", min_value=18, max_value=80, value=35)
    num_prev_txns   = st.number_input("Number of Previous Transactions", min_value=0, max_value=500, value=50, step=5)
    avg_txn_amount  = st.number_input("Customer's Avg Transaction Amount (Rs.)", min_value=1, max_value=500000, value=3000, step=100)

st.markdown("---")
st.subheader("📊 Risk Signals")
sig1, sig2, sig3 = st.columns(3)
is_night   = (hour >= 22) or (hour <= 5)
is_weekend = day_of_week >= 5
amount_ratio = round(amount / (avg_txn_amount + 1), 2)

with sig1:
    st.metric("Late Night", "⚠️ YES" if is_night else "✅ NO")
with sig2:
    st.metric("Weekend", "⚠️ YES" if is_weekend else "✅ NO")
with sig3:
    st.metric("Amount vs Avg", f"{amount_ratio}x", delta="HIGH" if amount_ratio > 5 else "NORMAL", delta_color="inverse")

st.markdown("---")

if st.button("🔍 Predict — Is This Fraud?", use_container_width=True, type="primary"):
    try:
        model = get_model()
        transaction = {
            "amount": amount, "hour": hour, "day_of_week": day_of_week,
            "merchant_type": merchant_type, "customer_age": customer_age,
            "num_prev_txns": num_prev_txns, "avg_txn_amount": avg_txn_amount,
        }
        result = predict(model, transaction)
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        if result["prediction"] == "FRAUD":
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        r1, r2, r3 = st.columns(3)
        with r1: st.metric("Prediction", result["prediction"])
        with r2: st.metric("Fraud Probability", f"{result['confidence']}%")
        with r3:
            emoji = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}
            st.metric("Risk Level", f"{emoji[result['risk_level']]} {result['risk_level']}")

        st.markdown("**Fraud Probability:**")
        st.progress(result["confidence"] / 100)

        st.markdown("---")
        st.subheader("💡 Why This Prediction?")
        reasons = []
        if is_night:    reasons.append("🌙 Late night transaction — high fraud signal")
        if is_weekend:  reasons.append("📅 Weekend transaction")
        if amount_ratio > 5: reasons.append(f"💰 Amount is {amount_ratio}x higher than customer average")
        if num_prev_txns < 5: reasons.append("👤 Very few previous transactions")
        if merchant_type in ["electronics","jewellery","online"]: reasons.append(f"🏪 {merchant_type.capitalize()} has higher fraud rates")
        if reasons:
            for r in reasons: st.markdown(f"- {r}")
        else:
            st.markdown("- No major risk signals detected")
    except FileNotFoundError:
        st.error("⚠️ Model not found! Run: python src/train.py")

st.markdown("---")
st.markdown("*MLOps Syllabus — Deploy and Retrain ML Models on AWS | Suresh D R*")
