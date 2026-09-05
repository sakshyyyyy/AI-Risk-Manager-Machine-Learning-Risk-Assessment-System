import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("risk_model.pkl")

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI Risk Manager")
st.write("AI-powered e-commerce return risk detection system")

st.divider()

# Order Details
st.subheader("Enter Order Details")

customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=25)
product_price = st.number_input("Product Price", min_value=1.0, value=1500.0)
discount_percent = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=60.0)
product_rating = st.number_input("Product Rating", min_value=0.0, max_value=5.0, value=4.2)

past_purchase_count = st.number_input(
    "Past Purchase Count",
    min_value=0,
    value=10
)

past_return_rate = st.number_input(
    "Past Return Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

delivery_delay_days = st.number_input(
    "Delivery Delay (Days)",
    min_value=0.0,
    value=5.0
)

session_length_minutes = st.number_input(
    "Session Length (Minutes)",
    min_value=0.0,
    value=3.0
)

num_product_views = st.number_input(
    "Number of Product Views",
    min_value=0,
    value=35
)

device_type = st.selectbox(
    "Device Type",
    ["mobile", "desktop", "tablet"]
)

product_category = st.selectbox(
    "Product Category",
    ["toys", "beauty", "electronics", "home", "clothing", "sports"]
)

shipping_method = st.selectbox(
    "Shipping Method",
    ["standard", "express", "same_day"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["debit_card", "credit_card", "apple_pay", "paypal"]
)

used_coupon = st.selectbox(
    "Used Coupon",
    [0, 1]
)

# Prediction button
if st.button("🔍 Assess Risk", use_container_width=True):

    # Create order dataframe
    order_df = pd.DataFrame([{
        "customer_age": customer_age,
        "product_price": product_price,
        "discount_percent": discount_percent,
        "product_rating": product_rating,
        "past_purchase_count": past_purchase_count,
        "past_return_rate": past_return_rate,
        "delivery_delay_days": delivery_delay_days,
        "session_length_minutes": session_length_minutes,
        "num_product_views": num_product_views,
        "device_type": device_type,
        "product_category": product_category,
        "shipping_method": shipping_method,
        "payment_method": payment_method,
        "used_coupon": used_coupon
    }])

    # Feature engineering
    order_df["discounted_price"] = (
        order_df["product_price"] *
        (1 - order_df["discount_percent"] / 100)
    )

    order_df["is_high_discount"] = (
        order_df["discount_percent"] > 50
    ).astype(int)

    order_df["is_frequent_returner"] = (
        order_df["past_return_rate"] > 0.5
    ).astype(int)

    order_df["is_delayed_delivery"] = (
        order_df["delivery_delay_days"] > 3
    ).astype(int)

    # Prediction
    probability = model.predict_proba(order_df)[0][1]
    risk_score = probability * 100

    # Risk level and action
    if probability >= 0.60:
        risk_level = "HIGH"
        action = "Manual Review"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
        action = "Additional Verification"
    else:
        risk_level = "LOW"
        action = "Allow Normally"

    # Risk factors
    risk_factors = []

    if past_return_rate > 0.5:
        risk_factors.append("High past return rate")

    if delivery_delay_days > 3:
        risk_factors.append("High delivery delay")

    if discount_percent > 50:
        risk_factors.append("High discount")

    if num_product_views > 30:
        risk_factors.append("High number of product views")

    if session_length_minutes < 5:
        risk_factors.append("Very short session")

    # Display result
    st.divider()
    st.subheader("AI Risk Assessment")

    st.metric(
        "Return Risk Probability",
        f"{risk_score:.2f}%"
    )

    st.write(f"### Risk Level: **{risk_level}**")
    st.write(f"### Recommended Action: **{action}**")

    st.subheader("Risk Factors")

    if risk_factors:
        for factor in risk_factors:
            st.write(f"⚠️ {factor}")
    else:
        st.write("✅ No major risk factors detected")
        
        