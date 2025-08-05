import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit page
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

# Header
st.title("🧠 Mental Health Treatment Predictor")
st.markdown("Use this app to predict whether a person might need mental health treatment based on workplace and personal attributes.")

# Load the trained model and feature columns
model = joblib.load("treatment_predictor.pkl")
features = joblib.load("feature_columns.pkl")

# --- Input Form ---
st.header("📋 Input Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Self-employed", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

with col2:
    work_interfere = st.selectbox("Work Interference with Mental Health", ["Never", "Rarely", "Sometimes", "Often"])
    no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Do You Work Remotely?", ["Yes", "No"])

# Combine inputs into a DataFrame
input_dict = {
    "Gender": gender,
    "self_employed": self_employed,
    "family_history": family_history,
    "work_interfere": work_interfere,
    "no_employees": no_employees,
    "remote_work": remote_work
}

input_df = pd.DataFrame([input_dict])

# Encode input using one-hot encoding to match training features
input_encoded = pd.get_dummies(input_df)

# Ensure all required feature columns are present
for col in features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0  # add missing columns with default 0

input_encoded = input_encoded[features]  # reorder columns

# --- Prediction ---
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_encoded)
        if prediction[0] == 1:
            st.success("✅ You may need mental health treatment.")
        else:
            st.info("🟢 You may not require mental health treatment.")
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")

# --- Insights Section ---
with st.expander("📊 Show Data Insights"):
    try:
        df = pd.read_csv("survey.csv")
        df = df.dropna(subset=["Gender", "treatment", "family_history", "work_interfere"])

        st.subheader("👤 Gender Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x="Gender", ax=ax1)
        st.pyplot(fig1)

        st.subheader("🧠 Treatment Need by Gender")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="Gender", hue="treatment", ax=ax2)
        st.pyplot(fig2)

        st.subheader("💼 Work Interference vs Treatment")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df, x="work_interfere", hue="treatment", ax=ax3)
        st.pyplot(fig3)

        st.subheader("👪 Family History vs Treatment")
        fig4, ax4 = plt.subplots()
        sns.countplot(data=df, x="family_history", hue="treatment", ax=ax4)
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Could not load insights: {e}")
