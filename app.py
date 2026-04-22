import streamlit as st
import numpy as np
import joblib

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# PAGE SETTINGS
# ==============================
st.set_page_config(page_title="Financial Inclusion App", layout="centered")

st.title("🌍 Financial Inclusion Analyzer")
st.write("This app predicts financial inclusion level based on user inputs.")

# ==============================
# INPUT SECTION
# ==============================
st.subheader("Enter Financial Indicators")

account = st.slider("Account Ownership (%)", 0, 100, 50)
digital = st.slider("Digital Access (%)", 0, 100, 40)
saved = st.slider("Savings (%)", 0, 100, 30)
borrowed = st.slider("Borrowing (%)", 0, 100, 20)
fin11 = st.slider("Digital Payments (%)", 0, 100, 50)

# ==============================
# PREDICTION BUTTON
# ==============================
if st.button("Predict Financial Inclusion"):

    # Prepare input
    input_data = np.array([[account, digital, saved, borrowed, fin11]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = model.predict(input_scaled)[0]

    # ==============================
    # OUTPUT
    # ==============================
    st.subheader(f"Predicted Cluster: {cluster}")

    if cluster == 0:
        st.error("🔴 Low Financial Inclusion")
        st.write("Limited access to banking and digital services.")
        
    elif cluster == 1:
        st.warning("🟡 Medium Financial Inclusion")
        st.write("Moderate usage of financial services, developing economy.")
        
    else:
        st.success("🟢 High Financial Inclusion")
        st.write("Strong banking access and digital financial usage.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.write("Model built using financial behavior data.")