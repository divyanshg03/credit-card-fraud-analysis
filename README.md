# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using various classification algorithms.

## Project Summary

This project implements a comprehensive fraud detection system for credit card transactions, helping financial institutions identify and prevent fraudulent activities. Using a dataset of credit card transactions with anonymized features, the system employs machine learning models to classify transactions as fraudulent or legitimate with high accuracy.

The project follows a complete data science workflow: exploratory data analysis, data preprocessing, model training with multiple algorithms, performance evaluation, and model deployment. The Random Forest model achieved the best performance and was selected as the final model for production use.

## Tech Stack 🛠️

- **Programming Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Environment:** Jupyter Notebook
- **Model Serialization:** Joblib
- **Preprocessing:** StandardScaler

## Model Performance Comparison 📊

| Model | Accuracy | Precision | Recall | F1-Score | AUC Score |
|-------|----------|-----------|--------|----------|-----------|
| Logistic Regression | ~97% | High | Moderate | Moderate | Not calculated |
| **Random Forest** | **~99%** | **Very High** | **High** | **High** | **Above 0.9** |
| Decision Tree | ~98% | High | Moderate | Moderate | Not calculated |
| Gradient Boosting | ~98% | High | Moderate | High | Not calculated |

## Why Random Forest? 🤔

Why was Random Forest chosen as the final model among all the classifiers implemented?

💡 The Random Forest classifier was selected as the final model for several compelling reasons:

1. **Superior Performance**: It demonstrated the highest accuracy (~99%) and best balance between precision and recall, crucial for fraud detection where both false positives and false negatives have significant consequences.

2. **Robust to Overfitting**: Random Forest's ensemble approach helps it generalize well to unseen data, important when dealing with the complex patterns of fraudulent transactions.

3. **Feature Importance Insights**: The model provides valuable insights into which transaction characteristics are most indicative of fraud, enhancing interpretability.

4. **Excellent AUC Score**: With an AUC score above 0.9, the Random Forest model shows exceptional discriminatory power between fraudulent and legitimate transactions.

5. **Handles Imbalanced Data**: Random Forest performs well even with the highly imbalanced nature of fraud detection datasets (very few fraudulent transactions compared to legitimate ones).

The combination of these factors made Random Forest the optimal choice for deployment in the production fraud detection system. 🚀

## Key Highlights

- **High Accuracy Detection**: The Random Forest model achieves exceptional accuracy in identifying fraudulent transactions
- **Multiple Models Comparison**: Evaluation of Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting classifiers
- **Feature Importance Analysis**: Identification of the most significant transaction attributes that indicate potential fraud
- **Advanced Visualization**: Comprehensive visualizations including confusion matrices, ROC curves, and feature importance plots
- **Production-Ready Implementation**: Model saved with joblib for easy deployment in production environments
- **Performance Metrics**: Detailed evaluation using accuracy, precision, recall, F1-score, and AUC measures
- **Data Imbalance Handling**: Effective strategies to handle the highly imbalanced nature of fraud detection datasets
- **No Missing Values**: Clean dataset with no missing values, ensuring reliable model training

## Overview

This project implements multiple machine learning models to detect fraudulent credit card transactions with high accuracy. The system analyzes transaction data and flags suspicious activities that may indicate fraud, helping financial institutions protect their customers.

## Project Structure

- `fraud detect.ipynb` - Jupyter notebook containing the complete data analysis, model development and evaluation
- `creditcard_2023.csv` - Dataset containing credit card transaction information
- `model.pkl` - Trained Random Forest model serialized with joblib
- `scaler.pkl` - StandardScaler for preprocessing new data
- `app.py` - Flask application for real-time fraud detection
- `requirements.txt` - List of required Python packages
- `LICENSE` - MIT License

## Features

- Data preprocessing and exploration
- Training of multiple classification models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Gradient Boosting
- Model evaluation with metrics like accuracy, precision, recall, and F1-score
- Feature importance analysis
- ROC curve and AUC score calculation
- A deployable prediction pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-analysis.git
cd credit-card-fraud-analysis

# Install required packages
pip install -r requirements.txt
```

## Usage

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
