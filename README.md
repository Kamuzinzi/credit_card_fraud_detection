# Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent credit card transactions using advanced classification techniques and model interpretability tools.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Project Overview

This project addresses the critical challenge of credit card fraud detection by building and evaluating multiple machine learning models. The primary challenge lies in the highly imbalanced nature of the dataset, where fraudulent transactions account for only **0.172%** of all transactions.

The project demonstrates:
- **Data exploration and preprocessing** techniques for imbalanced datasets
- **Multiple ML algorithms** including Logistic Regression, Random Forest, Gradient Boosting, SVM, and Neural Networks
- **Model interpretability** using SHAP and LIME to explain model decisions
- **Comprehensive evaluation** using appropriate metrics for imbalanced classification

## ✨ Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of transaction patterns and feature distributions
- **Data Preprocessing**: Handling class imbalance using various techniques (SMOTE, undersampling, class weights)
- **Multiple Model Comparison**: Evaluation of traditional ML models and deep learning approaches
- **Model Interpretability**: SHAP and LIME analysis to understand feature importance and model decisions
- **Performance Evaluation**: Focus on precision-recall metrics suitable for imbalanced datasets

## 📊 Dataset

The dataset contains credit card transactions made in September 2013 by European cardholders.

**Dataset Statistics:**
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Time Period**: 2 days
- **Features**: 31 (28 PCA components + Time + Amount + Class)

**Feature Description:**
- **V1 to V28**: Principal components obtained via PCA (original features anonymized for confidentiality)
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset
- **Amount**: The transaction amount
- **Class**: Target variable (1 = fraud, 0 = non-fraud)

**Evaluation Metrics:**
Due to the highly imbalanced nature of the dataset, the primary evaluation metric is:
- **Area Under the Precision-Recall Curve (AUPRC)**
- Additional metrics: Precision, Recall, F1-Score, Confusion Matrix

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── data/
│   └── row/                    # Raw dataset (not tracked in git - download from Kaggle)
│       └── creditcard.csv
│
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01-data_exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline_model.ipynb
│   ├── 04-advanced_models.ipynb
│   ├── 05-model_interpretation.ipynb
│   └── 06-final_model_evaluation.ipynb
│
├── src/                         # Source code modules
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── shap_analysis.py
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the dataset from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the `data/row/` directory

## 💻 Usage

### Running the Notebooks

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Execute notebooks in order:**
   - `01-data_exploration.ipynb` - Explore the dataset
   - `02-preprocessing.ipynb` - Preprocess and prepare data
   - `03-baseline_model.ipynb` - Train baseline models
   - `04-advanced_models.ipynb` - Train advanced models
   - `05-model_interpretation.ipynb` - Interpret model decisions
   - `06-final_model_evaluation.ipynb` - Final evaluation

### Using the Source Code

The `src/` directory contains reusable modules:

```python
from src.data_preprocessing import preprocess_data
from src.model import train_model
from src.evaluation import evaluate_model
from src.shap_analysis import analyze_with_shap

# Example usage
X_train, X_test, y_train, y_test = preprocess_data('data/row/creditcard.csv')
model = train_model(X_train, y_train, model_type='random_forest')
results = evaluate_model(model, X_test, y_test)
```

## 🔬 Methodology

### 1. Data Exploration
- Statistical summary of features
- Distribution analysis
- Correlation analysis
- Class imbalance visualization

### 2. Data Preprocessing
- Feature scaling/normalization
- Handling class imbalance:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Random undersampling
  - Class weight adjustment

### 3. Model Training
- **Baseline Models:**
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  
- **Advanced Models:**
  - Deep Neural Networks (TensorFlow/Keras)
  - Ensemble methods

### 4. Model Evaluation
- Precision-Recall Curve
- ROC Curve
- Confusion Matrix
- Classification Report

### 5. Model Interpretability
- SHAP (SHapley Additive exPlanations) values
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Model Interpretability**: SHAP, LIME
- **Development**: Jupyter Notebook

## 📈 Results

The project evaluates multiple models on the imbalanced credit card fraud dataset. Key findings include:

- Models are evaluated using **AUPRC** as the primary metric due to class imbalance
- Various techniques are applied to handle the imbalance problem
- Model interpretability tools provide insights into which features drive fraud detection

*Note: Specific results and metrics are available in the evaluation notebooks.*

## 🔮 Future Improvements

- [ ] Implement additional advanced models (XGBoost, LightGBM, CatBoost)
- [ ] Experiment with autoencoders for anomaly detection
- [ ] Deploy model as a REST API
- [ ] Create a web dashboard for real-time fraud detection
- [ ] Implement real-time monitoring and alerting system
- [ ] Add more comprehensive cross-validation strategies
- [ ] Explore feature engineering techniques

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- European cardholders dataset from Worldline and the Machine Learning Group

---

⭐ If you find this project helpful, please consider giving it a star!
