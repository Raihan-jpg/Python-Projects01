from multiprocessing import reduction
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# --- STEP 1: Load and Prepare Data ---
@st.cache_data
def load_data():
    # Fixes the path issue dynamically
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "student_grades.csv")
    data = pd.read_csv(file_path)
    return data


data = load_data()

# Define features (X) and target (Y)
X = data[["Attendance", "Assignment", "Internal", "Study hours"]]
Y = data["Final marks"]

# --- STEP 2: Train Model ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

# --- STEP 3: Streamlit UI ---
st.title("🎓 Student Grade Predictor")
st.write("Predict a student's final exam grade based on key academic factors.")

# Sidebar / User Inputs
st.sidebar.header("Enter Student Details")
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
assignment = st.sidebar.slider("Assignment Marks", 0, 100, 80)
internal = st.sidebar.slider("Internal Exam Marks", 0, 100, 75)
study_hours = st.sidebar.slider("Study Hours Per Day", 0, 12, 3)

# Predict Button
if st.button("Predict Final Marks"):
    input_data = np.array([[attendance, assignment, internal, study_hours]])
    raw_prediction = model.predict(input_data)[0]

    capped_prediction=np.clip(raw_prediction, 0, 100)  # Ensure prediction is between 0 and 100

    st.success(f"🎯 **Predicted Final Mark:** {capped_prediction:.2f}")
    if raw_prediction > 100:
        st.info("note: The predicted mark exceeds 100. It has been capped at 100 for display purposes.*")

# Display raw dataset option
if st.checkbox("Show Raw Dataset"):
    st.subheader("Training Data Preview")
    st.dataframe(data)

# Display Model Performance
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Fixed the syntax error at the bottom here
with st.expander("Model Statistics"):
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
