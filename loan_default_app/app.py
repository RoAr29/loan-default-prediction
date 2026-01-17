import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------- Load artifacts ----------
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

@st.cache_resource
def load_features():
    return joblib.load("models/feature_names.pkl")

model = load_model()
feature_names = load_features()

# ---------- Page Config (better layout) ----------
st.set_page_config(
    page_title="Loan Default Predictor",
    layout="centered",
    page_icon="💰"
)

# ---------- App Header ----------
st.title("💳 Loan Default Risk Predictor")
st.markdown("""
This tool estimates the probability that a borrower will default on a loan.  
It uses a Random Forest model trained on historical lending data and your  
previously chosen **0.40 risk threshold** (risk-averse strategy).
""")

st.divider()

# ---------- Sidebar Inputs (clean layout) ----------
st.sidebar.header("Borrower Details")

amount = st.sidebar.number_input(
    "Loan Amount (₹)",
    min_value=10000,
    max_value=5000000,      # increased limit ✅
    value=150000,
    step=10000
)

income = st.sidebar.number_input(
    "Annual Income (₹)",
    min_value=50000,
    max_value=10000000,     # increased limit ✅
    value=650000,
    step=25000
)

rate = st.sidebar.slider("Interest Rate", 0.05, 0.30, 0.13)
debtIncRat = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 60.0, 18.0)

grade = st.sidebar.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
home = st.sidebar.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
term = st.sidebar.selectbox("Loan Term", ["36 months","60 months"])
verified = st.sidebar.selectbox("Income Verified?", ["Verified","Not Verified"])

st.divider()

# ---------- Create input dataframe ----------
input_df = pd.DataFrame({
    "amount":[amount],
    "income":[income],
    "rate":[rate],
    "debtIncRat":[debtIncRat],
    "grade":[grade],
    "home":[home],
    "term":[term],
    "verified":[verified]
})

# Encode and align with training features
input_encoded = pd.get_dummies(input_df)
input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)

# ---------- Prediction Button ----------
if st.button("🔍 Predict Default Risk"):

    prob = model.predict_proba(input_aligned)[0][1]   # probability of default
    pred = (prob >= 0.40)  # your chosen threshold

    st.subheader("📌 Prediction Result")

    if pred:
        st.error(f"🚨 HIGH DEFAULT RISK — Probability: {prob:.2f}")
    else:
        st.success(f"✅ LOW DEFAULT RISK — Probability: {prob:.2f}")

    # --------- RISK EXPLANATION (new feature) ---------
    st.subheader("🧠 Why this prediction makes sense")

    reasons = []

    if debtIncRat > 30:
        reasons.append("High debt-to-income ratio increases default risk.")
    if rate > 0.18:
        reasons.append("Higher interest rates make repayment harder.")
    if income < 300000:
        reasons.append("Lower income reduces repayment capacity.")
    if grade in ["E","F","G"]:
        reasons.append("Lower credit grade is associated with higher risk.")
    if home == "RENT":
        reasons.append("Renting (vs owning) is slightly riskier on average.")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("Borrower profile looks relatively stable based on key indicators.")

    # --------- BUSINESS NOTE ---------
    st.info("""
    **Decision rule used:**  
    - If probability ≥ 0.40 → Classified as *High Risk*  
    - This favors catching more defaulters (risk-averse lending).
    """)

st.divider()
st.caption(f"Model trained with {len(feature_names)} encoded features.")
