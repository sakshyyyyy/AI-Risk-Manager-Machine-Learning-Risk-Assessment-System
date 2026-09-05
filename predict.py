import joblib

# Load trained model
model = joblib.load("risk_model.pkl")

print("Risk model loaded successfully!")

# Example order
order = {
    "customer_age": 25,
    "product_price": 1500,
    "discount_percent": 60,
    "product_rating": 4.2,
    "past_purchase_count": 10,
    "past_return_rate": 0.7,
    "delivery_delay_days": 5,
    "session_length_minutes": 3,
    "num_product_views": 35,
    "device_type": "mobile",
    "product_category": "electronics",
    "shipping_method": "standard",
    "payment_method": "credit_card",
    "used_coupon": 1
}

print(order)

import pandas as pd

# Convert order into DataFrame
order_df = pd.DataFrame([order])

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

# Make prediction
probability = model.predict_proba(order_df)[0][1]

print("\n========== AI RISK RESULT ==========")
print(f"Return Risk Probability: {probability * 100:.2f}%")

# Risk level
if probability >= 0.60:
    risk_level = "HIGH"
    action = "Manual Review"
elif probability >= 0.30:
    risk_level = "MEDIUM"
    action = "Additional Verification"
else:
    risk_level = "LOW"
    action = "Allow Normally"

print(f"Risk Score: {probability * 100:.2f}%")
print(f"Risk Level: {risk_level}")
print(f"Recommended Action: {action}")

# Risk factors
risk_factors = []

if order["past_return_rate"] > 0.5:
    risk_factors.append("High past return rate")

if order["delivery_delay_days"] > 3:
    risk_factors.append("High delivery delay")

if order["discount_percent"] > 50:
    risk_factors.append("High discount")

if order["num_product_views"] > 30:
    risk_factors.append("High number of product views")

if order["session_length_minutes"] < 5:
    risk_factors.append("Very short session")

print("\nRisk Factors:")
if risk_factors:
    for factor in risk_factors:
        print(f"- {factor}")
else:
    print("- No major risk factors detected")