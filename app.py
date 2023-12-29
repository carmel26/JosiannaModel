import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

def preprocess_uploaded_file(uploaded_df, scaler, num_scans=15):
    uploaded_df.columns = ['wavelengths'] + [f'SCAN{i}' for i in range(1, uploaded_df.shape[1])]
    template_data = {'wavelengths': uploaded_df['wavelengths']}
    for i in range(1, num_scans + 1):
        template_data[f'SCAN{i}'] = 0
    template_df = pd.DataFrame(template_data)
    for column in uploaded_df.columns:
        if column in template_df.columns:
            template_df[column] = uploaded_df[column]
    template_df_cleaned = template_df.dropna(axis=1)
    preprocessed_df = scaler.transform(template_df_cleaned)
    return preprocessed_df

st.image('logo.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("<h3 style='text-align: center; color: grey;'>Hyperspectral Based System For Identification Of Common Bean Genotypes Resistant To Foliar Diseases</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    preprocessed_input = preprocess_uploaded_file(input_df, scaler)
    predictions = model.predict(preprocessed_input)
    most_common_class = Counter(predictions).most_common(1)[0][0]
    if most_common_class == 'Resistant':
        st.write('The Plant is resistant to foliar diseases.')
    elif most_common_class == 'Medium':
        st.write('The Plant shows medium resistance to foliar diseases.')
    elif most_common_class == 'Susceptible':
        st.write('The Plant is susceptible to foliar diseases.')
