import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title and Description
st.markdown(
    """
    <h1 style='text-align: center; color: #5A5A5A;'>Customer Segmentation App 🚀</h1>
    <div style='text-align: center; font-size:18px; color: #444;'>Use this app to predict customer segments based on their purchasing behaviors!</div>
    <hr style="border-color: #F63366;">
    """, unsafe_allow_html=True
)

# Input Section in Columns
st.subheader("Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧑 Age", min_value=18, max_value=100, value=35)
    income = st.number_input("💰 Income", min_value=0, max_value=200000, value=50000)
    total_spending = st.number_input("🛒 Total Spending", min_value=0, max_value=5000, value=1000)
    recency = st.number_input("⏱ Recency (days)", min_value=0, max_value=365, value=30)

with col2:
    num_web_purchases = st.number_input("🌐 Web Purchases", min_value=0, max_value=100, value=10)
    num_store_purchases = st.number_input("🏬 Store Purchases", min_value=0, max_value=100, value=10)
    num_web_visite = st.number_input("📈 Web Visits/month", min_value=0, max_value=50, value=3)

# Prepare input
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visite],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

# Prediction Button with added spacing
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🎯 Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]

    # Custom segment info
    segment_desc = {
        0: "High budget, frequent web visitors",
        1: "High Spending customers",
        2: "Frequent web visitors but moderate spenders"
    }
    st.success(f"Predicted Segment: **Cluster {cluster}**")
    st.info(segment_desc.get(cluster, "Unknown Segment"))

    # Display all clusters
    st.markdown("""
        <hr>
        <h3>Cluster Descriptions</h3>
        <ul>
            <li><b>Cluster 0:</b> High budget, Web visitors</li>
            <li><b>Cluster 1:</b> High Spending</li>
            <li><b>Cluster 2:</b> Web visitors</li>
        </ul>
    """, unsafe_allow_html=True)

# UI Tips:
# - Use icons with inputs for a lively feel.
# - st.markdown with unsafe_allow_html enables HTML/CSS support for colors and styles.
# - Splitting inputs into columns makes the layout clear.
# - Custom segment descriptions improve informativeness.
# - Use emojis in buttons or instructions for an appealing interface.
