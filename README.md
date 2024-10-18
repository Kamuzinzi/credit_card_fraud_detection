# credit_card_fraud_detection

## Project Overview
The goal of this project is to build a machine learning model to detect fraudulent credit card transactions using transaction data. Identifying fraudulent transactions is crucial for credit card companies to protect customers from unauthorized charges.

The dataset is sourced from a Kaggle dataset containing credit card transactions made in September 2013 by European cardholders. The challenge in this project is the highly imbalanced nature of the dataset, where fraudulent transactions only account for 0.172% of all transactions.

We will explore various techniques to handle this imbalance, build machine learning models, and use interpretability tools such as SHAP and LIME to explain how the model makes decisions.

Dataset
The dataset contains a total of 284,807 transactions from 2 days, with 492 fraudulent transactions. The features of the dataset are as follows:

V1 to V28: Principal components obtained via PCA (the original features are not available due to confidentiality).
Time: Seconds elapsed between this transaction and the first transaction in the dataset.
Amount: The transaction amount.
Class: The response variable (1 = fraud, 0 = non-fraud).
Since the dataset is highly imbalanced, Area Under the Precision-Recall Curve (AUPRC) is used as the primary evaluation metric, with additional metrics like Precision, Recall, and F1-Score to assess model performance.

