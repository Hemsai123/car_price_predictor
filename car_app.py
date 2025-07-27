import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Page title
st.title("🚗 Quikr Car Price Predictor")

# Extract categories from the pipeline
name_categories = pipe.named_steps['columntransformer'].transformers_[0][1].categories_[0]
company_categories = pipe.named_steps['columntransformer'].transformers_[0][1].categories_[1]
fuel_categories = pipe.named_steps['columntransformer'].transformers_[0][1].categories_[2]

#  Step 1: Select Company ---
selected_company = st.selectbox("Select Company", sorted(company_categories))

# Step 2: Filter Car Names by Selected Company ---
# Show only names that start with or contain the selected company
filtered_names = [name for name in name_categories if selected_company.lower() in name.lower()]

# Fallback in case no match
if not filtered_names:
    filtered_names = name_categories

selected_name = st.selectbox("Select Car Model", sorted(filtered_names))

#  Step 3: Fuel Type, Year, KMs ---
selected_fuel = st.selectbox("Fuel Type", fuel_categories)
year = st.slider("Year of Purchase", 1990, 2025, 2015)
kms_driven = st.number_input("Kilometers Driven", 0, 1000000, 10000)

#  Step 4: Predict Button ---
if st.button("Predict Price"):
    input_df = pd.DataFrame([[selected_name, selected_company, year, kms_driven, selected_fuel]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = pipe.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

