import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    """
Anonymized features ($V1-V339$, $C1-C14$, $D1-D15$) and engineered features 
are used to evaluate transaction risk in real time.
"""
)


# Load Model and Feature list
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("fraud_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except Exception as e:
        st.error(
            "Could not load model files. Please ensure 'fraud_model.pkl' and 'model_features.pkl' are in the working directory."
        )
        return None, None


model, feature_names = load_artifacts()

# Fallback feature list if not loaded from file
if feature_names is None:
    feature_names = [
        "TransactionAmt",
        "Transaction_hour_of_day",
        "Transaction_velocity_24hr",
        "Card_age_days",
        "is_late_night",
        "email_domain_match",
        "TransactionAmt_log",
        "V258",
        "V70",
        "V294",
        "C1",
        "C13",
    ]


# Helper function to generate synthetic anonymized data
def generate_synthetic_data(num_samples=1):
    data = {}
    for col in feature_names:
        # Custom logic for known engineered columns
        if col == "TransactionAmt":
            data[col] = np.round(np.random.exponential(scale=100) + 0.5, 2)
        elif col == "Transaction_hour_of_day":
            data[col] = np.random.randint(0, 24, size=num_samples)
        elif col == "is_late_night":
            data[col] = np.random.choice([0, 1], p=[0.88, 0.12])
        elif col == "email_domain_match":
            data[col] = np.random.choice([0, 1], p=[0.82, 0.18])
        elif col == "Transaction_velocity_24hr":
            data[col] = np.random.poisson(lam=15, size=num_samples)
        elif col == "Card_age_days":
            data[col] = np.random.randint(0, 600, size=num_samples)
        elif col.startswith("V"):
            # Anonymized V-features modeled as standard normal or uniform distribution
            data[col] = np.random.normal(loc=0.0, scale=1.5, size=num_samples)
        elif col.startswith("C") or col.startswith("D"):
            data[col] = np.random.randint(0, 50, size=num_samples)
        else:
            data[col] = np.random.uniform(0, 100, size=num_samples)

    df = pd.DataFrame(data)
    if "TransactionAmt_log" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df


# Sidebar Setup
st.sidebar.header("🕹️ Controls & Data Synthesizer")

generator_type = st.sidebar.radio(
    "Select Input Mode:",
    [
        "🎲 Single Random Transaction",
        "🎛️ Interactive Feature Synthesizer",
        "📊 Batch Synthetic Data",
    ],
)

if generator_type == "🎲 Single Random Transaction":
    st.subheader("Random Transaction Generator")

    if st.button("🎲 Generate & Predict New Transaction"):
        st.session_state["current_sample"] = generate_synthetic_data(1)

    if "current_sample" in st.session_state:
        sample = st.session_state["current_sample"]

        st.markdown("### Generated Synthetic Feature Profile")
        # Display top 10 key features for readability
        st.dataframe(sample.iloc[:, :12].style.highlight_max(axis=0))

        if model is not None:
            # Predict
            prob = model.predict_proba(sample)[0][1]
            pred = int(prob > 0.5)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability Score", f"{prob:.2%}")
            with col2:
                if pred == 1:
                    st.error("🚨 **Prediction: HIGH RISK (Fraud Alert)**")
                else:
                    st.success("✅ **Prediction: LOW RISK (Legitimate)**")

elif generator_type == "🎛️ Interactive Feature Synthesizer":
    st.subheader("Customize Key Inputs (Rest Auto-Generated)")

    amt = st.slider("Transaction Amount ($)", 0.5, 5000.0, 125.0)
    hour = st.slider("Hour of Day", 0, 23, 14)
    late_night = 1 if (2 <= hour <= 5) else 0
    velocity = st.number_input("24-Hour Velocity (# Transactions)", 1, 100, 5)
    domain_match = st.selectbox(
        "Purchaser & Recipient Email Match?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No"
    )

    if st.button("Predict Fraud Risk"):
        sample = generate_synthetic_data(1)
        # Override key features
        sample["TransactionAmt"] = amt
        sample["Transaction_hour_of_day"] = hour
        sample["is_late_night"] = late_night
        sample["Transaction_velocity_24hr"] = velocity
        sample["email_domain_match"] = domain_match
        sample["TransactionAmt_log"] = np.log1p(amt)

        prob = (
            model.predict_proba(sample)[0][1] if model is not None else np.random.rand()
        )

        st.markdown("---")
        st.metric("Calculated Fraud Risk", f"{prob:.2%}")
        if prob > 0.5:
            st.error("🚨 Flagged as Potential Fraud")
        else:
            st.success("✅ Transaction Safe")

elif generator_type == "📊 Batch Synthetic Data":
    st.subheader("Batch Synthetic Transaction Evaluator")
    n_samples = st.slider("Number of Random Transactions", 5, 100, 20)

    if st.button("Generate Batch"):
        batch_df = generate_synthetic_data(n_samples)

        if model is not None:
            probs = model.predict_proba(batch_df)[:, 1]
            batch_df["Fraud Probability Score"] = probs
            batch_df["Prediction"] = np.where(probs > 0.5, "🚨 Fraud", "✅ Legitimate")

        st.dataframe(batch_df[["TransactionAmt", "Transaction_hour_of_day", "Fraud Probability Score", "Prediction"]])
