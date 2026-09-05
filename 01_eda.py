import pandas as pd
import numpy as np  
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load dataset
df = pd.read_csv("train.csv")

# First 5 rows
print(df.head())

# Shape
print(df.shape)

# Column names
print(df.columns)

# Information
print(df.info())

# Statistics
print(df.describe())

# Missing values
print(df.isnull().sum())

# Target distribution
print(df["returned"].value_counts())

# Target percentage
print(df["returned"].value_counts(normalize=True) * 100)

# ============================
# STEP 2 - EDA
# ============================

print("\n--- UNIQUE VALUES ---")

for col in df.columns:
    print(f"\n{col}:")
    print(df[col].nunique())


# Categorical columns
categorical_columns = [
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method"
]

for col in categorical_columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())


# Numerical columns
numerical_columns = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views"
]

for col in numerical_columns:
    print(f"\n--- {col} ---")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())


# Suspicious values
print("\n--- SUSPICIOUS VALUES ---")

print("Negative prices:",
      (df["product_price"] < 0).sum())

print("Negative discounts:",
      (df["discount_percent"] < 0).sum())

print("Negative product views:",
      (df["num_product_views"] < 0).sum())

print("Negative session length:",
      (df["session_length_minutes"] < 0).sum())

print("Negative delivery delay:",
      (df["delivery_delay_days"] < 0).sum())


# Numerical features by return status
print("\n--- NUMERICAL FEATURES BY RETURN STATUS ---")

print(
    df.groupby("returned")[numerical_columns].mean().T
)


# Categorical features by return status
for col in categorical_columns:
    print(f"\n--- Return rate by {col} ---")

    print(
        df.groupby(col)["returned"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n--- RETURN RATE FOR SUSPICIOUS VALUES ---")

# Negative product price
print("\nNegative product price:")
print(df.loc[df["product_price"] < 0, "returned"].mean())

# Negative discount
print("\nNegative discount:")
print(df.loc[df["discount_percent"] < 0, "returned"].mean())

# Negative product views
print("\nNegative product views:")
print(df.loc[df["num_product_views"] < 0, "returned"].mean())

# Negative session length
print("\nNegative session length:")
print(df.loc[df["session_length_minutes"] < 0, "returned"].mean())

# Negative delivery delay
print("\nNegative delivery delay:")
print(df.loc[df["delivery_delay_days"] < 0, "returned"].mean())

print("\n--- VALID vs INVALID RETURN RATES ---")

print("\nProduct price:")
print("Valid:",
      df.loc[df["product_price"] >= 0, "returned"].mean())
print("Invalid:",
      df.loc[df["product_price"] < 0, "returned"].mean())


print("\nDiscount:")
print("Valid:",
      df.loc[df["discount_percent"] >= 0, "returned"].mean())
print("Invalid:",
      df.loc[df["discount_percent"] < 0, "returned"].mean())


print("\nProduct views:")
print("Valid:",
      df.loc[df["num_product_views"] >= 0, "returned"].mean())
print("Invalid:",
      df.loc[df["num_product_views"] < 0, "returned"].mean())


print("\nSession length:")
print("Valid:",
      df.loc[df["session_length_minutes"] >= 0, "returned"].mean())
print("Invalid:",
      df.loc[df["session_length_minutes"] < 0, "returned"].mean())

print("\n--- INVALID PRODUCT RATINGS ---")

print("Below 0:",
      (df["product_rating"] < 0).sum())

print("Above 5:",
      (df["product_rating"] > 5).sum())

print("\n--- INVALID PAST RETURN RATE ---")

print("Below 0:",
      (df["past_return_rate"] < 0).sum())

print("Above 1:",
      (df["past_return_rate"] > 1).sum())
print("\n--- AGE DISTRIBUTION ---")

print("Below 18:",
      (df["customer_age"] < 18).sum())

print("Above 100:",
      (df["customer_age"] > 100).sum())

print("\nDelivery delay:")

print("Negative:",
      df.loc[df["delivery_delay_days"] < 0, "returned"].mean())

print("Non-negative:",
      df.loc[df["delivery_delay_days"] >= 0, "returned"].mean())

# ============================
# STEP 4 - DATA CLEANING
# ============================

df_clean = df.copy()

print("\nOriginal shape:", df.shape)
print("Clean dataframe shape:", df_clean.shape)

# Invalid product price
df_clean.loc[
    df_clean["product_price"] < 0,
    "product_price"
] = np.nan

# Invalid discount
df_clean.loc[
    df_clean["discount_percent"] < 0,
    "discount_percent"
] = np.nan


# Invalid product views
df_clean.loc[
    df_clean["num_product_views"] < 0,
    "num_product_views"
] = np.nan


# Invalid session length
df_clean.loc[
    df_clean["session_length_minutes"] < 0,
    "session_length_minutes"
] = np.nan


# Invalid product rating
df_clean.loc[
    df_clean["product_rating"] > 5,
    "product_rating"
] = np.nan


# Invalid past return rate
df_clean.loc[
    df_clean["past_return_rate"] < 0,
    "past_return_rate"
] = np.nan

print("\n--- AFTER INVALID VALUE HANDLING ---")

print(df_clean.isnull().sum())


numerical_columns = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views"
]

imputer = SimpleImputer(strategy="median")

# df_clean[numerical_columns] = imputer.fit_transform(
#     df_clean[numerical_columns]
# )

print("\n--- AFTER IMPUTATION ---")
print(df_clean[numerical_columns].isnull().sum())

df_clean["discounted_price"] = (
    df_clean["product_price"] *
    (1 - df_clean["discount_percent"] / 100)
)
df_clean["is_high_discount"] = (
    df_clean["discount_percent"] >= 50
).astype(int)

df_clean["is_frequent_returner"] = (
    df_clean["past_return_rate"] >= 0.30
).astype(int)

df_clean["is_delayed_delivery"] = (
    df_clean["delivery_delay_days"] > 0
).astype(int)

print("\n--- NEW FEATURES ---")

print(
    df_clean[
        [
            "product_price",
            "discount_percent",
            "discounted_price",
            "is_high_discount",
            "past_return_rate",
            "is_frequent_returner",
            "delivery_delay_days",
            "is_delayed_delivery"
        ]
    ].head()
)

print("\n--- ENGINEERED FEATURES CHECK ---")
print("\n--- ENGINEERED FEATURES CHECK ---")

print(
    df_clean[
        [
            "discounted_price",
            "is_high_discount",
            "is_frequent_returner",
            "is_delayed_delivery"
        ]
    ].head().to_string(index=False)
)

# ============================
# STEP 6 - X AND y
# ============================

X = df_clean.drop(
    columns=["returned", "order_id"]
)

y = df_clean["returned"]

print("\n--- X AND y ---")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\ny distribution:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN TEST SPLIT ---")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

numerical_features = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate",
    "delivery_delay_days",
    "session_length_minutes",
    "num_product_views",
    "discounted_price",
    "is_high_discount",
    "is_frequent_returner",
    "is_delayed_delivery"
]

categorical_features = [
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])
print("\n--- PREPROCESSOR CREATED ---")
print(preprocessor)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\n--- TRAINING MODEL ---")

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

print("\n--- PREDICTIONS ---")
print(y_pred[:20])

print("\n--- MODEL PERFORMANCE ---")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("\n--- TRAINING RANDOM FOREST ---")

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")
rf_pred = rf_model.predict(X_test)

print("\n--- RANDOM FOREST PERFORMANCE ---")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

print("\n--- MODEL COMPARISON ---")

print("Logistic Regression F1:", f1_score(y_test, y_pred))
print("Random Forest F1      :", f1_score(y_test, rf_pred))

# So this broke. Both our models logistics regression and random forest are performing poorly. 
# The F1 scores are low, indicating that the models are not effectively capturing the patterns in the data. 
# This could be due to several reasons such as class imbalance, irrelevant features, or insufficient feature engineering.S.
# Hence we will now check the importance of the features in 
# the Random Forest model to understand which features are contributing the most to the predictions.
# This will help us identify any irrelevant features and potentially improve our model's performance.

# Get feature names after preprocessing
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

# Get importance from Random Forest
importances = rf_model.named_steps["classifier"].feature_importances_

# Create a table
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n--- TOP 20 FEATURE IMPORTANCES ---")
print(feature_importance.head(20).to_string(index=False))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_pred)

print("\n--- CONFUSION MATRIX ---")
print(cm)


from sklearn.metrics import precision_score, recall_score, f1_score
rf_prob = rf_model.predict_proba(X_test)[:, 1]
print("\n--- THRESHOLD ANALYSIS ---")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    predictions = (rf_prob >= threshold).astype(int)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(
        f"Threshold: {threshold} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )
 

print("\n--- THRESHOLD ANALYSIS ---")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    predictions = (rf_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    TN, FP, FN, TP = cm.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:")
    print(cm)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n--- COST ANALYSIS ---")

FP_COST = 50
FN_COST = 500

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    predictions = (rf_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    TN, FP, FN, TP = cm.ravel()

    total_cost = (FP * FP_COST) + (FN * FN_COST)

    print(f"\nThreshold: {threshold}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"Total Cost: {total_cost:}")
    
    print("\n--- FINAL MODEL PERFORMANCE ---")

final_threshold = 0.30

final_pred = (rf_prob >= final_threshold).astype(int)

final_cm = confusion_matrix(y_test, final_pred)

TN, FP, FN, TP = final_cm.ravel()

final_precision = TP / (TP + FP)
final_recall = TP / (TP + FN)
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)

print("Final Threshold:", final_threshold)
print("Confusion Matrix:")
print(final_cm)

print(f"Precision: {final_precision:.4f}")
print(f"Recall:    {final_recall:.4f}")
print(f"F1 Score:  {final_f1:.4f}")

def calculate_risk_level(probability):

    risk_score = probability * 100

    if probability >= 0.60:
        risk_level = "HIGH"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_score, risk_level

def get_recommended_action(risk_level):

    if risk_level == "HIGH":
        return "Manual Review"

    elif risk_level == "MEDIUM":
        return "Additional Verification"

    else:
        return "Allow Normally"
    
    
    

# sample_probability = rf_prob[0]

# risk_score, risk_level = calculate_risk_level(sample_probability)

# # print("\n--- SAMPLE RISK ASSESSMENT ---")
# # print(f"Risk Score: {risk_score:.2f}%")
# # print(f"Risk Level: {risk_level}")

# action = get_recommended_action(risk_level)

# print(f"Recommended Action: {action}")

def get_risk_factors(row):

    factors = []

    if row["past_return_rate"] > 0.5:
        factors.append("High past return rate")

    if row["delivery_delay_days"] > 3:
        factors.append("High delivery delay")

    if row["discount_percent"] > 50:
        factors.append("High discount")

    if row["num_product_views"] > 30:
        factors.append("High number of product views")

    if row["session_length_minutes"] < 5:
        factors.append("Very short session")

    return factors


def assess_order_risk(row, probability):

    # Risk score
    risk_score = probability * 100

    # Risk level
    if probability >= 0.60:
        risk_level = "HIGH"
    elif probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Recommended action
    action = get_recommended_action(risk_level)

    # Risk factors
    risk_factors = get_risk_factors(row)

    return risk_score, risk_level, action, risk_factors

# # Use actual model probability
# sample_probability = rf_prob[0]

# risk_score, risk_level, action, risk_factors = assess_order_risk(
#     sample_order,
#     sample_probability
# )

# # Take a real order from test data
# sample_order = X_test.iloc[0]

# # Get model probability for the same order
# sample_probability = rf_model.predict_proba(X_test.iloc[[0]])[0][1]

# # Final risk assessment
# risk_score, risk_level, action, risk_factors = assess_order_risk(
#     sample_order,
#     sample_probability
# )

# print("\n--- FINAL RISK ASSESSMENT ---")
# print(f"Risk Score: {risk_score:.2f}%")
# print(f"Risk Level: {risk_level}")
# print(f"Recommended Action: {action}")

# print("Risk Factors:")
# if risk_factors:
#     for factor in risk_factors:
#         print(f"- {factor}")
# else:
#     print("- No major risk factors detected")



# Test multiple real orders
print("\n========== MULTIPLE ORDER RISK TEST ==========")

for i in range(5):

    # Take a real order
    order = X_test.iloc[i]

    # Model prediction
    probability = rf_model.predict_proba(X_test.iloc[[i]])[0][1]

    # Risk assessment
    risk_score, risk_level, action, risk_factors = assess_order_risk(
        order,
        probability
    )

    print(f"\n--- ORDER {i+1} ---")
    print(f"Risk Score: {risk_score:.2f}%")
    print(f"Risk Level: {risk_level}")
    print(f"Recommended Action: {action}")

    print("Risk Factors:")
    if risk_factors:
        for factor in risk_factors:
            print(f"- {factor}")
    else:
        print("- No major risk factors detected")
        
        
        
# Overall risk distribution
all_probabilities = rf_model.predict_proba(X_test)[:, 1]

low = sum(all_probabilities < 0.30)
medium = sum((all_probabilities >= 0.30) & (all_probabilities < 0.60))
high = sum(all_probabilities >= 0.60)

total = len(all_probabilities)

print("\n========== RISK DISTRIBUTION ==========")
print(f"LOW: {low} ({low/total*100:.2f}%)")
print(f"MEDIUM: {medium} ({medium/total*100:.2f}%)")
print(f"HIGH: {high} ({high/total*100:.2f}%)")        


import joblib

joblib.dump(rf_model, "risk_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("\nModel and preprocessor saved successfully!")