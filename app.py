import streamlit as st
import pandas as pd
import os

import ydata_profiling as yp
from ydata_profiling.profile_report import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image(r"C:\Users\hp\Downloads\th.jpg", width=300)
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This Application allows you to build an automated ML pipeline")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling !")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = yp.ProfileReport(df)
    st_profile_report(profile_report)

if choice == "ML":
    st.title(" Machine learning :) ")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train Model"):
        if df[target].dtype.name == 'category':
            df[target] = df[target].cat.codes
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.write(best_model)  
        save_model(best_model, 'best_model') 

if choice == "Download":
    best_model = 'best_model'  # Charger le meilleur modèle
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
