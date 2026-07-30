import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('amount_scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a credit card transaction is *fraudulent* or *genuine* using a Random Forest model trained on real transaction data.")

st.markdown("---")

st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=1.0)

with col2:
    st.write("")
    st.write("")
    use_random = st.checkbox("Use random V1-V28 values (demo mode)")

st.caption("Note: V1–V28 are anonymized PCA features from the original dataset (they represent hidden transaction patterns like location, time-of-day behavior, spending pattern etc., but their exact meaning is masked for privacy). In a real production system, these would be computed automatically from raw transaction data.")

if use_random:
    v_features = np.random.normal(0, 1, 28)
else:
    v_features = []
    with st.expander("Manually set V1–V28 values (optional)"):
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                val = st.number_input(f"V{i+1}", value=0.0, format="%.4f", key=f"v{i+1}")
                v_features.append(val)
    v_features = np.array(v_features)

st.markdown("---")

if st.button("🔍 Check Transaction", type="primary"):
    scaled_amount = scaler.transform([[amount]])[0][0]

    
    input_data = np.append(v_features, scaled_amount).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=[f'V{i+1}' for i in range(28)] + ['scaled_amount'])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.markdown("### Result")

    if prediction == 1:
        st.error(f"🚨 *FRAUD DETECTED* — Confidence: {probability[1]*100:.2f}%")
        st.write("This transaction shows patterns consistent with fraudulent activity. It would typically be flagged for manual review or automatically blocked.")
    else:
        st.success(f"✅ *GENUINE TRANSACTION* — Confidence: {probability[0]*100:.2f}%")
        st.write("This transaction appears normal based on the learned patterns.")

    st.markdown("#### Prediction Probability")
    prob_df = pd.DataFrame({
        'Class': ['Genuine', 'Fraud'],
        'Probability': [probability[0], probability[1]]
    })
    st.bar_chart(prob_df.set_index('Class'))

st.markdown("---")
st.caption("Model: Random Forest | Trained on Kaggle Credit Card Fraud Detection dataset (284,807 transactions, SMOTE-balanced) | Built by Subhajit Poddar")
