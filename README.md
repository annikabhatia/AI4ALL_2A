# Credit Card Fraud Detection

A machine learning project built during the AI4ALL AI Fellowship to detect fraudulent credit card transactions using real-world financial data.

## Overview

This project analyzes 1M+ credit card transactions across two linked datasets to identify anomalous spending patterns indicative of fraud. Built collaboratively with a team of 3 AI4ALL fellows, the model captures 82.8% of fraudulent transactions (0.95 ROC-AUC) while minimizing false-positive review costs.

## Dataset

- **Source:** IEEE-CIS Fraud Detection dataset (transaction + identity data)
- **Size:** 590,540 transactions, 394 raw features
- **Target:** `isFraud` (binary), with a heavily imbalanced positive class (~3.5% fraud rate)
- **Features:** transaction amount, product code, card/address/distance info, email domains, count features (C1–C14), time-delta features (D1–D15), match features (M1–M9), and 300+ anonymized engineered features (V1–V339)

## Approach

1. **EDA** — explored transaction structure, class imbalance, and feature distributions
2. **Missing value handling** — identified and dropped 55 columns with >80% missing data (394 → 339 features)
3. **Feature engineering** — built features to surface anomalous spending signals
4. **Classification modeling** — trained models to flag fraudulent transactions
5. **Evaluation** — tuned for recall on the fraud class while controlling false positives

## Results

- **82.8%** of fraudulent transactions correctly identified
- **0.95 ROC-AUC**
- False-positive rate minimized to reduce downstream review costs

## Tech Stack

Python · Pandas · NumPy · Seaborn · scikit-learn · Google Colab

## Usage

Open `credit_card_fraud_detection_updated.ipynb` in Google Colab (badge linked in the notebook) or Jupyter. Update the data path to point to your local copy of `train_transaction.csv`, then run cells sequentially.
