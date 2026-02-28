import streamlit as st
st.set_page_config(page_title="Streaming Churn AI", layout="wide")

st.markdown("""
<style>

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #121212;
    color: white;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #000000;
}

/* Sidebar heading (User Details Input) */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1DB954 !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: white !important;
}

/* Fix sidebar collapse arrow */
button[kind="header"] {
    color: white !important;
}

/* Headings in main area */
h1, h2, h3 {
    color: #1DB954;
}

/* Buttons */
.stButton>button {
    background-color: #1DB954;
    color: white;
    border-radius: 8px;
    height: 3em;
}

.stButton>button:hover {
    background-color: #17a74a;
}

</style>
""", unsafe_allow_html=True)



import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("# 🎧 Analytics Dashboard")
st.markdown("AI-powered churn prediction system for streaming platforms.")
st.markdown("---")



import pandas as pd

df = pd.read_csv("data/streaming_churn.csv")

st.write("### Dataset Overview")
st.write(f"Total Users: {len(df)}")
st.write(f"Overall Churn Rate: {round(df['churn'].mean()*100,2)}%")

import matplotlib.pyplot as plt





st.sidebar.header("User Details Input")

subscription_months = st.sidebar.number_input("Subscription Months", 1, 60, 6)
monthly_fee = st.sidebar.selectbox("Monthly Fee", [199, 299, 499])
watch_time = st.sidebar.number_input("Watch Time Last 30 Days (minutes)", 0, 5000, 300)
login_frequency = st.sidebar.number_input("Login Frequency (Last 30 Days)", 0, 60, 5)
complaints = st.sidebar.number_input("Number of Complaints", 0, 10, 0)
payment_label = st.sidebar.selectbox("Payment Failure", ["No", "Yes"])
payment_failure = 1 if payment_label == "Yes" else 0
subscription_label = st.sidebar.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Premium Platinum"]
)

if subscription_label == "Basic":
    subscription_type = 0
else:
    subscription_type = 1


if st.button("Predict Churn"):

    input_data = np.array([[subscription_months, monthly_fee,
                            watch_time, login_frequency,
                            complaints, payment_failure,
                            subscription_type]])

    probability = model.predict_proba(input_data)[0][1]

    # --- Risk Progress Bar ---
    st.markdown("### 📊 Risk Score")
    st.progress(int(probability * 100))

    # --- Risk Category ---
    if probability > 0.7:
        risk_color = "#FF4B4B"
        risk_label = "High Risk"
    elif probability > 0.4:
        risk_color = "#FFA500"
        risk_label = "Medium Risk"
    else:
        risk_color = "#1DB954"
        risk_label = "Low Risk"

    st.markdown("## 🎯 Prediction Summary")

    st.markdown(
        f"""
        <div style="
            background-color:#181818;
            padding:25px;
            border-radius:15px;
            text-align:center;
            border-left: 8px solid {risk_color};
        ">
            <h2 style="color:{risk_color};">{risk_label}</h2>
            <h1>{round(probability*100,2)}%</h1>
            <p>Probability of churn</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Risk Indicators ---
    st.write("### Key Risk Indicators")

    risk_found = False

    if watch_time < 300:
        st.write("- Low engagement detected")
        risk_found = True

    if login_frequency < 5:
        st.write("- Low login frequency")
        risk_found = True

    if payment_failure == 1:
        st.write("- Recent payment failure")
        risk_found = True

    if complaints > 2:
        st.write("- Multiple complaints registered")
        risk_found = True

    if not risk_found:
        st.success("No major churn risk factors detected.")