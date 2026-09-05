# AI Risk Manager

An AI-powered e-commerce return risk detection system developed for the Razorpay AI Buildathon.

## Problem Statement

E-commerce businesses face losses due to product returns and risky orders.
This project uses machine learning to estimate the probability that an order will be returned and assigns an appropriate risk level.

## Features

- E-commerce return risk prediction
- Machine learning based probability score
- LOW, MEDIUM and HIGH risk classification
- Recommended action for each risk level
- Explainable risk factors
- Interactive Streamlit dashboard
- Business cost based threshold analysis

## Machine Learning

The project uses:

- Logistic Regression
- Random Forest Classifier

Random Forest was selected as the final model based on its performance.

### Evaluation

Final threshold: 0.30

- Precision: 0.4777
- Recall: 0.9791
- F1 Score: 0.6421

The threshold was selected using an illustrative business cost analysis where false negatives were assigned a higher cost than false positives.

## Risk Levels

| Risk Probability | Risk Level | Recommended Action |
|---|---|---|
| < 30% | LOW | Allow Normally |
| 30% - 59% | MEDIUM | Additional Verification |
| >= 60% | HIGH | Manual Review |

## Risk Factors

The system identifies factors such as:

- High past return rate
- High delivery delay
- High discount
- High number of product views
- Very short session

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
Razorpay_ai_risk_manager/
│
├── 01_eda.py
├── predict.py
├── app.py
├── risk_model.pkl
├── preprocessor.pkl
├── requirements.txt
├── README.md
├── train.csv
└── test.csv