import streamlit as st
import pandas as pd
import joblib
import datetime

# Load model
model = joblib.load('depression_model.pkl')
scaler = joblib.load('scaler.pkl')
st.title("Depression Prediction")
# ['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'Depression']
# Input Form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 34)
academic_pressure = st.selectbox("Academic Pressure (1: No academic pressure, 5: High academic pressure)", [1, 2, 3, 4, 5])
study_satisfaction = st.selectbox("Study Satisfaction (1: No satisfied, 5: Very satisfied)", [1, 2, 3, 4, 5])
sleep = st.selectbox("Sleep Duration", ["<5h", "5-6h", "7-8h", ">8h"])
diet_habit = st.selectbox("Dietary Habits", ["Healthy", "Moderate","Unhealthy"])
sud_thought = st.selectbox("Have you ever have suicidal thought?", ["Yes", "No"])
stu_hour = st.slider("Study Hours / Day", 0, 10)
fin_stress = st.selectbox("Financial Stress (1: No financial stress, 5: High financial stress)", [1, 2, 3, 4, 5])
fam_his = st.selectbox("Family History of Mental Illness", ["Yes", "No"])


if st.button("Predict"):
    # --- Mapping categorical input ---
    gender_val = 0 if gender == "Male" else 1
    sleep_map = {"<5h":2, "5-6h":1, "7-8h":0, ">8h":2}
    sleep_val = sleep_map[sleep]
    diet_habit_map = {"Healthy": 0, "Moderate":1, "Unhealthy":2}
    diet_val = diet_habit_map[diet_habit]
    sud_val = 0 if sud_thought == "No" else 1
    fam_his_val = 0 if fam_his == "No" else 1

    # DataFrame
    input_data = pd.DataFrame([[gender_val, age, academic_pressure, study_satisfaction, sleep_val, diet_val, sud_val, stu_hour, fin_stress,fam_his_val]],
                              columns=['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Study Hours', 'Financial Stress', 'Family History of Mental Illness'])
    input_scaled = scaler.transform(input_data)
    # predict
    prediction = model.predict(input_scaled)[0]
    st.success("Depression: Yes " if prediction==1 else "Depression: No ")

    # save user data
    input_data['Prediction'] = prediction
    input_data['Timestamp'] = datetime.datetime.now()
    input_data.to_csv('user_data.csv', mode='a', header=False, index=False)

