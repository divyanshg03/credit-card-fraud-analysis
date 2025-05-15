# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions.

## Overview

This project implements a machine learning model to identify potentially fraudulent credit card transactions. The model is trained on transaction data and can be used to flag suspicious activities in real-time or batch processing scenarios.

## Project Structure

- `creditcard.csv` - Dataset containing credit card transactions
- `fraud detect.ipynb` - Jupyter notebook with data analysis and model development
- `model_detect.pkl` - Serialized machine learning model
- `scaler.pkl` - Serialized data scaler for preprocessing new data
- `LICENSE` - MIT License
- `README.md` - Project documentation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Open the `fraud detect.ipynb` notebook to see the data analysis and model training process
2. To use the pre-trained model:

```python
import pickle
import pandas as pd

# Load the model and scaler
with open('model_detect.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Preprocess your data
# Replace 'your_data.csv' with your transaction data
data = pd.read_csv('your_data.csv')
features = data.drop(['Class'], axis=1)  # Adjust column names as needed
scaled_features = scaler.transform(features)

# Make predictions
predictions = model.predict(scaled_features)
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

Divyansh Gupta
