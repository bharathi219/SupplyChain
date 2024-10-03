############## Libraries/Modules ##############

import pandas as pd
import numpy as np

# ModelLoading Libraries
import joblib

# UI & Logic Library
import streamlit as st

####################### Loading Trained Model Files #########
sc = joblib.load("sc.pkl")  # converting numeric cols under one scale
model = joblib.load("log_multi.pkl")  # trained poly regression file

########################## UI Code ################################

st.header("Product Estimation for the Given Features.")

# Dividing Row into columns in streamlit window
p1, p2, p3 = st.columns(3)

with p2:
    st.image("PIC.jpg")

st.write("This app is built on the below features to estimate Order Priority.")

userinpdata = pd.read_excel(r"C:\Users\mekal\Supply_Chain_FinalData.xlsx")
st.dataframe(userinpdata.head(5))

st.subheader("Enter Product Details to Estimate Order Priority:")

# General Input
# sqft = st.number_input("Enter Sqft Value:")

# Form Type Input
col1, col2, col3, col4 = st.columns(4)
with col1:
    Lead_Time = st.number_input("Lead_Time:")
with col2:
    Demand_Forecast = st.number_input("Demand_Forecast:")
with col3:
    Inventory_Level = st.number_input("Inventory_Level:")
with col4:
    Stockout_Flag = st.number_input("Stockout_Flag:")

col5, col6, col7, col8 = st.columns(4)
with col5:
    Backorder_Flag = st.number_input("Backorder_Flag:")
with col6:
    Order_Quantity = st.number_input("Order_Quantity:")
with col7:
    Shipment_Quantity = st.number_input("Shipment_Quantity No:")
with col8:
    Product_Price = st.number_input("Product_Price:")

###################### Logic Code #############################

if st.button("Estimateorderpriority"):

    row = pd.DataFrame([[Lead_Time, Demand_Forecast, Inventory_Level, Stockout_Flag, Backorder_Flag, Order_Quantity, Shipment_Quantity, Product_Price]], columns=userinpdata.columns)
    st.write("Given Input Data:")
    st.dataframe(row)
    
    # Applying Feature Modification steps before giving it to model

    print("********** Prediction ***********")
    print()

    # Accessing the number of classes
    num_classes = model.classes_.size
    print("Number of classes:", num_classes)
    
    # Predicting probabilities
    prob0 = round(model.predict_proba(row)[0][0], 2)
    prob1 = round(model.predict_proba(row)[0][1], 2)
    prob2 = round(model.predict_proba(row)[0][2], 2)
    print("Predicted Probabilities: low 0 - {}, medium 1 - {}, high 2 - {}".format(prob0, prob1, prob2))
    
    # Predicting the outcome
    out = model.predict(row)[0]
    print("Prediction:", out)
    
    st.write(f"Estimated Order Priority: {out} ")
