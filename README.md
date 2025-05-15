# 💳🔍 Credit Card Fraud Detection System 🚨🤖

A powerful machine learning project to **detect potentially fraudulent credit card transactions** using advanced classification algorithms. Built to help financial institutions **protect their customers and minimize financial loss**.

---

## 📝 Project Summary

This project implements a **comprehensive fraud detection system** for credit card transactions, empowering banks and financial institutions to identify and stop fraudulent activities in real-time. Using a dataset of anonymized transactions, the system applies various **machine learning models** to accurately classify transactions as **fraudulent or legitimate**.

🔁 The complete data science workflow is followed:

- 🧪 **Exploratory Data Analysis**
- 🧼 **Data Preprocessing**
- 🤖 **Model Training (Multiple Algorithms)**
- 📊 **Performance Evaluation**
- 🚀 **Model Deployment**

🎯 **Random Forest** emerged as the top-performing model and was chosen for **production deployment** due to its excellent results.

---

## 🧰 Tech Stack & Tools

- 🐍 **Python 3.x**
- 🧮 **Pandas, NumPy** – Data Manipulation
- 📊 **Matplotlib, Seaborn** – Data Visualization
- 🤖 **Scikit-learn** – Machine Learning
- 🧪 **Jupyter Notebook** – Development Environment
- 💾 **Joblib** – Model Serialization
- ⚖️ **StandardScaler** – Feature Scaling

---

## 📈 Model Performance Comparison

| 📊 Model | ✅ Accuracy | 🎯 Precision | 🔍 Recall | 📏 F1-Score | 📉 AUC Score |
|---------|------------|--------------|------------|-------------|---------------|
| Logistic Regression | ~97% | High | Moderate | Moderate | ❌ Not Calculated |
| 🌲 **Random Forest** | **~99%** | **Very High** | **High** | **High** | **✅ Above 0.9** |
| Decision Tree | ~98% | High | Moderate | Moderate | ❌ Not Calculated |
| Gradient Boosting | ~98% | High | Moderate | High | ❌ Not Calculated |

---

## 🤔 Why Random Forest?

🎉 The **Random Forest** classifier was chosen as the final production model due to several standout advantages:

1. 🔝 **Top Performance**: Achieved the highest accuracy (~99%) and best balance between **precision** and **recall**.
2. 🛡️ **Robust to Overfitting**: Ensemble method generalizes well to unseen data.
3. 🔍 **Feature Importance**: Offers insights into which features are most indicative of fraud.
4. 📈 **High AUC Score**: Over 0.9, showing excellent ability to distinguish between fraud and legitimate cases.
5. ⚖️ **Handles Imbalanced Data**: Well-suited for datasets with few fraud cases compared to legitimate ones.

✅ These strengths make it ideal for real-world fraud detection systems!

---

## 🌟 Key Highlights

- ✅ **High Accuracy Fraud Detection** with Random Forest
- 🧠 **Multiple Model Evaluations**: Logistic Regression, Decision Tree, Gradient Boosting
- 🔬 **Feature Importance Analysis**
- 📉 **ROC Curve & Confusion Matrix Visualizations**
- 💡 **Production-Ready** with serialized model and scaler
- ⚖️ **Smart Handling of Imbalanced Datasets**
- 🚫 **No Missing Values** in the dataset

---

## 📂 Project Structure

```bash
📁 credit-card-fraud-analysis/
├── 📓 fraud detect.ipynb       # Data analysis, modeling, evaluation
├── 📄 creditcard_2023.csv     # Credit card transaction dataset
├── 🧠 model.pkl               # Trained Random Forest model
├── ⚖️ scaler.pkl              # StandardScaler used for data preprocessing
├── 🌐 app.py                  # Flask web application for predictions
├── 📦 requirements.txt        # Dependencies
└── 📜 LICENSE                 # MIT License
```


## 🚀 Features

✨ **End-to-End Machine Learning Pipeline** built for real-world fraud detection:

---

### 🧹 Data Cleaning & Preprocessing
- ✅ Removal of irrelevant or redundant features
- 🔢 Standardization using `StandardScaler`
- ⚖️ Handling of class imbalance using sampling techniques or class weights

---

### 🧠 Model Training (Multiple Classifiers)
Train and compare a variety of supervised learning models:
- 📈 **Logistic Regression**
- 🌲 **Random Forest**
- 🌳 **Decision Tree**
- 🚀 **Gradient Boosting**

---

### 📊 Performance Evaluation
Robust model evaluation using multiple metrics:
- ✅ **Accuracy**
- 🎯 **Precision**
- 🔍 **Recall**
- 📏 **F1-Score**
- 📉 **AUC (Area Under ROC Curve)**

---

### 🔍 Feature Importance Analysis
- 🧠 Gain insights into which features contribute most to the prediction of fraud
- 📈 Visualizations of feature importance for model interpretability

---

### 🎯 ROC Curves & Confusion Matrices
- 📉 Plot ROC curves to evaluate model discrimination
- 🧾 Visualize confusion matrices to inspect false positives/negatives

---

### 💾 Model Serialization
- 💼 Save trained models using `Joblib`
- 🔁 Easy reusability and deployment with `.pkl` files

---

### 🌐 Flask Web Application
- ⚡ Real-time fraud prediction via REST API
- 💻 Simple web interface for submitting transaction data
- 📡 Instant display of prediction results and fraud probability

---
## ⚙️ Installation & Setup
## 🔧 Clone the Repository
```bash
 git clone https://github.com/yourusername/credit-card-fraud-analysis.git
cd credit-card-fraud-analysis
```
##📦 Install Required Packages
```bash
pip install -r requirements.txt
```

### Running the Notebook

To explore the data analysis and model development process:

```bash
jupyter notebook "fraud detect.ipynb"
```

### Using the Model for Predictions

```python
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare your data (must have the same features as training data)
# Note: Remove 'id' and 'Class' columns if present
data = pd.read_csv('your_transactions.csv').drop(['id', 'Class'], axis=1, errors='ignore')

# Preprocess the data
scaled_data = scaler.transform(data)

# Make predictions
predictions = model.predict(scaled_data)
probability_scores = model.predict_proba(scaled_data)[:, 1]

# Results
results = pd.DataFrame({
    'is_fraud': predictions,
    'fraud_probability': probability_scores
})
```

### Running the Web Application

```bash
python app.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Divyansh Gupta
