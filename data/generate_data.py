import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

subscription_months = np.random.randint(1, 36, n)
monthly_fee = np.random.choice([199, 299, 499], n)
watch_time_last_30_days = np.random.randint(0, 2000, n)
login_frequency = np.random.randint(0, 30, n)
complaints = np.random.randint(0, 5, n)
payment_failure = np.random.choice([0, 1], n, p=[0.85, 0.15])
subscription_label = st.sidebar.selectbox("Subscription Type", ["Basic", "Premium"])

# Convert to numeric for model
subscription_type = 0 if subscription_label == "Basic" else 1

# Simple churn logic
churn_probability = (
    (watch_time_last_30_days < 300).astype(int) * 0.4 +
    (login_frequency < 5).astype(int) * 0.3 +
    payment_failure * 0.5 +
    (complaints > 2).astype(int) * 0.2
)

churn_probability = np.clip(churn_probability, 0, 1)

churn = np.random.binomial(1, churn_probability)

df = pd.DataFrame({
    "subscription_months": subscription_months,
    "monthly_fee": monthly_fee,
    "watch_time_last_30_days": watch_time_last_30_days,
    "login_frequency": login_frequency,
    "complaints": complaints,
    "payment_failure": payment_failure,
    "subscription_type": subscription_type,
    "churn": churn
})

df.to_csv("streaming_churn.csv", index=False)

print("Dataset created successfully!")
print(df.head())
