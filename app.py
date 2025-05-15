import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Credit Card Fraud Detection Model")
st.divider()

st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
st.divider()


feat  =("Enter the features: v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28")
st.write(feat)
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
st.divider()
st.write("Note: The values of v1 to v28 are PCA-encoded values.")
Amount = st.number_input("Amount", min_value=50.00, max_value = 25000.00, step=0.01, format="%.9f")
input_df_lst.append(Amount)
st.write("Note: The value of Amount is the transaction amount.")

submit = st.button("Submit")
if submit:
    # get input feature values
    # create a list of input feature values

    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction(0)")
    else:
        st.write("Fraudulent transaction(1)")