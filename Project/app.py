import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
from pathlib import Path


st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")

RAW_NUMERIC_AND_BINARY_COLS = [
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Admission grade', 'Age at enrollment',
    'Scholarship holder', 'Tuition fees up to date', 'Debtor', 'Gender'
]

try:
    # Example Path: r'C:\Users\xlr8m\PyCharmMiscProject\Project\final_model.joblib'
    model = joblib.load(r'\Project\final_model.joblib')
    feature_names = joblib.load(r'\Project\feature_names.joblib')
    label_encoder = joblib.load(r'\Project\label_encoder.joblib')
    scaler = joblib.load(r'\Project\scaler.joblib')

except FileNotFoundError:
    st.error("FATAL ERROR: Deployment file not found. Please verify the file paths.")
    st.stop()
except Exception as e:
    st.error(f"FATAL ERROR during model loading. Check dependencies (imbalanced-learn, joblib) and paths. Error: {e}")
    st.stop()


def perform_feature_engineering(data):
    """Calculates the Total Failed Units, Grade Delta, and Approval Rate."""

    data['Total_Failed_Units'] = (
                                         data['Curricular units 1st sem (enrolled)'] - data[
                                     'Curricular units 1st sem (approved)']
                                 ) + (
                                         data['Curricular units 2nd sem (enrolled)'] - data[
                                     'Curricular units 2nd sem (approved)']
                                 )

    data['Grade_Delta'] = data['Curricular units 2nd sem (grade)'] - data['Curricular units 1st sem (grade)']

    data['Total_Enrolled'] = data['Curricular units 1st sem (enrolled)'] + data['Curricular units 2nd sem (enrolled)']
    data['Total_Approved'] = data['Curricular units 1st sem (approved)'] + data['Curricular units 2nd sem (approved)']


    data['Approval_Rate'] = data.apply(
        lambda row: row['Total_Approved'] / row['Total_Enrolled'] if row['Total_Enrolled'] > 0 else 0,
        axis=1
    )

    data = data.drop(columns=['Total_Enrolled', 'Total_Approved'])
    return data


def predict_outcome(raw_data):

    df_raw = pd.DataFrame([raw_data])


    df_engineered = perform_feature_engineering(df_raw.copy())


    df_encoded = pd.get_dummies(df_engineered,
                                columns=['Marital Status', 'Application mode', 'Daytime/evening attendance'],
                                drop_first=True)

    df_final = pd.DataFrame(OrderedDict([(col, [0]) for col in feature_names]))

    for col in df_encoded.columns:
        if col in df_final.columns:
            df_final[col] = df_encoded[col].values

    transformed_values = scaler.transform(df_final.values)


    df_transformed = pd.DataFrame(transformed_values, columns=feature_names)

    prediction_num = model.predict(df_transformed.values)

    prediction_label = label_encoder.inverse_transform(prediction_num)

    prediction_proba = model.predict_proba(df_transformed.values).flatten()

    return prediction_label[0], prediction_proba



st.title("🎓 Student Outcome Early Warning System")
st.markdown(
    "Use this tool to predict a student's final outcome based on their first-year academic and personal data. Predictions are made using the optimized **Random Forest + SMOTE** model.")

with st.form("student_input_form"):
 
    st.header("1. Academic Performance (Semester Grades and Units)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Semester 1")
        sem1_enrolled = st.number_input("Units Enrolled (Sem 1)", min_value=0, value=6, key="se1")
        sem1_approved = st.number_input("Units Approved (Sem 1)", min_value=0, max_value=sem1_enrolled, value=6,
                                        key="sa1")
        sem1_grade = st.number_input("Average Grade (Sem 1)", min_value=0.0, max_value=20.0, value=13.0, step=0.1,
                                     key="sg1")

    with col2:
        st.subheader("Semester 2")
        sem2_enrolled = st.number_input("Units Enrolled (Sem 2)", min_value=0, value=6, key="se2")
        sem2_approved = st.number_input("Units Approved (Sem 2)", min_value=0, max_value=sem2_enrolled, value=5,
                                        key="sa2")
        sem2_grade = st.number_input("Average Grade (Sem 2)", min_value=0.0, max_value=20.0, value=11.5, step=0.1,
                                     key="sg2")

    with col3:
        st.subheader("Initial Metrics")
        admission_grade = st.number_input("Admission Grade (Entrance Score)", min_value=0.0, max_value=200.0,
                                          value=140.0, step=1.0, key="ag")
        age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, value=19, key="age")

    st.markdown("---")


    st.header("2. Financial and Status Indicators")
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        marital_status = st.selectbox("Marital Status",
                                      ["Single", "Married", "Divorced", "Widower", "Separated", "Other"], key="ms")
        application_mode = st.selectbox("Application Mode",
                                        ["1st phase", "2nd phase", "Special regime", "Transfer", "Change"], key="am")

    with col5:
        daytime_attendance = st.selectbox("Attendance Type", ["Daytime", "Evening"], key="dt")
        scholarship_holder = st.selectbox("Scholarship Holder?", ["Yes", "No"], key="sh")

    with col6:
        tuition_up_to_date = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"], key="tf")
        debtor = st.selectbox("Debtor?", ["Yes", "No"], key="db")

    with col7:
        gender = st.selectbox("Gender", ["Male", "Female"], key="ge")

-
    st.markdown("---")
    submitted = st.form_submit_button("Predict Student Outcome")

if submitted:
  
    raw_input_data = {
        'Curricular units 1st sem (enrolled)': sem1_enrolled,
        'Curricular units 1st sem (approved)': sem1_approved,
        'Curricular units 1st sem (grade)': sem1_grade,
        'Curricular units 2nd sem (enrolled)': sem2_enrolled,
        'Curricular units 2nd sem (approved)': sem2_approved,
        'Curricular units 2nd sem (grade)': sem2_grade,
        'Admission grade': admission_grade,
        'Age at enrollment': age_at_enrollment,


        'Scholarship holder': 1 if scholarship_holder == 'Yes' else 0,
        'Tuition fees up to date': 1 if tuition_up_to_date == 'Yes' else 0,
        'Debtor': 1 if debtor == 'Yes' else 0,
        'Gender': 1 if gender == 'Female' else 0,
      
        'Marital Status': marital_status,
        'Application mode': application_mode,
        'Daytime/evening attendance': daytime_attendance,
    }


    result_label, result_proba = predict_outcome(raw_input_data)


    proba_dict = dict(zip(label_encoder.classes_, result_proba))


    st.header(f"Prediction Result: {result_label}")

    if result_label == "Dropout":
        st.error(
            f"⚠️ **CRITICAL RISK:** The model predicts the student will **{result_label}** (Confidence: {proba_dict['Dropout'] * 100:.1f}%)")
        st.markdown("**Action:** Immediate academic and personal intervention is highly recommended.")
    elif result_label == "Enrolled":
        st.warning(
            f"🔔 **Stable:** The model predicts the student will remain **{result_label}** (Confidence: {proba_dict['Enrolled'] * 100:.1f}%)")
        st.markdown("This student is currently on track, but continuous monitoring is advised.")
    else: 
        st.success(
            f"✅ **Success:** The model predicts the student will **{result_label}** (Confidence: {proba_dict['Graduate'] * 100:.1f}%)")
        st.markdown("The student is performing well and is not considered a primary risk.")

    st.subheader("Probability Distribution")
    st.dataframe(pd.DataFrame(proba_dict.items(), columns=["Outcome", "Probability"]))
