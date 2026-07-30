# Credit Card Fraud Detection

An end-to-end machine learning project to detect fraudulent credit card transactions using a highly imbalanced real-world dataset.

## Problem Statement

Credit card fraud costs billions of dollars annually. This project builds a classification model to identify fraudulent transactions in real time, helping financial institutions minimize losses while reducing false alarms.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions from European cardholders
- Only 492 (0.17%) are fraudulent — a highly imbalanced dataset
- Features V1–V28 are PCA-transformed for confidentiality; Time and Amount are raw

## Approach

1. *EDA* — Explored class imbalance and transaction amount distributions between fraud and genuine transactions
2. *Preprocessing* — Scaled the Amount feature, dropped Time, and performed a stratified train-test split
3. *Handling Imbalance* — Applied SMOTE (Synthetic Minority Oversampling) on the training set only, to avoid data leakage
4. *Modeling* — Trained and compared three models: Logistic Regression, Random Forest, and XGBoost
5. *Evaluation* — Used Precision, Recall, and F1-score instead of accuracy, since accuracy is misleading on imbalanced data
6. *Explainability* — Used SHAP to identify which features most influence fraud predictions

## Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| *Random Forest* | *0.87* | 0.83 | *0.85* |
| XGBoost | 0.69 | 0.86 | 0.76 |

*Random Forest* was selected as the final model based on the best F1-score, offering the most practical balance between catching fraud (recall) and minimizing false alarms (precision).

## Business Impact

- Missing a fraud case (false negative) directly costs the bank money
- Flagging genuine transactions as fraud (false positive) frustrates customers and increases manual review workload
- The final model catches 83% of fraud cases while keeping false alarms low, striking a practical business trade-off

## Tech Stack

- Python, Pandas, Scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- SHAP (model explainability)
- Streamlit (deployment)

## How to Run

bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb


## Project Structure


credit-card-fraud-detection/
├── fraud_detection.ipynb    # Full analysis and model training notebook
├── fraud_model.pkl          # Saved trained model
├── amount_scaler.pkl        # Saved feature scaler
├── app.py                   # Streamlit web app for live predictions
├── requirements.txt
└── README.md


## Live Demo

[Add Streamlit app link here once deployed]

## Author
[Subhajit podddar] — [https://www.linkedin.com/in/subhajit-poddar-b7b949326?utm_source=share_via&utm_content=profile&utm_medium=member_android]
