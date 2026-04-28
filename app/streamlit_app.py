import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "models/tube_disruption_model.pkl"

# ---------------------------
# Helper Functions (same as model)
# ---------------------------

def get_day_peak_weight(day):
    weights = {
        "Monday": 0.6,
        "Tuesday": 1.0,
        "Wednesday": 1.0,
        "Thursday": 0.9,
        "Friday": 0.4,
        "Saturday": 0.3,
        "Sunday": 0.2
    }
    return weights.get(day, 0.5)


def get_time_category(hour):
    if 7 <= hour <= 10:
        return "morning_peak"
    elif 10 < hour < 12:
        return "mid_morning"
    elif 12 <= hour <= 14:
        return "noon_peak"
    elif 14 < hour < 17:
        return "afternoon"
    elif 17 <= hour <= 20:
        return "evening_peak"
    else:
        return "night"


def get_line_status(line):
    improving_lines = ["Central", "District", "Metropolitan", "Hammersmith & City", "Circle"]
    extended_lines = ["Northern"]
    upgrading_lines = ["Piccadilly"]

    if line in improving_lines:
        return 2
    elif line in extended_lines:
        return 2
    elif line in upgrading_lines:
        return 1
    else:
        return 1


def convert_time_to_hour(time_str):
    return pd.to_datetime(time_str).hour


# ---------------------------
# UI START
# ---------------------------

st.set_page_config(
    page_title="London Tube Disruption Predictor",
    page_icon="🚇",
    layout="centered"
)

st.title("🚇 London Tube Disruption Predictor")
st.write("Predict disruption level based on travel time, demand, and real-world patterns.")

model = joblib.load(MODEL_FILE)

# ---------------------------
# User Inputs
# ---------------------------

line = st.selectbox(
    "Select Tube Line",
    [
        "Bakerloo", "Central", "Circle", "District", "Hammersmith & City",
        "Jubilee", "Metropolitan", "Northern", "Piccadilly",
        "Victoria", "Waterloo & City"
    ]
)

# User-friendly time dropdown
time_option = st.selectbox(
    "Select Travel Time",
    [
        "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM",
        "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM",
        "02:00 PM", "03:00 PM", "04:00 PM", "05:00 PM",
        "06:00 PM", "07:00 PM", "08:00 PM", "09:00 PM",
        "10:00 PM", "11:00 PM"
    ]
)

day_of_week = st.selectbox(
    "Select Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

month = st.selectbox(
    "Select Month",
    [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
)

is_strike = st.selectbox("Strike expected?", ["No", "Yes"])
is_strike = 1 if is_strike == "Yes" else 0

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Disruption"):

    hour = convert_time_to_hour(time_option)

    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    is_noon_peak = 1 if 12 <= hour <= 14 else 0
    is_peak_hour = 1 if (7 <= hour <= 10 or 12 <= hour <= 14 or 16 <= hour <= 19) else 0
    is_peak_day = 1 if day_of_week in ["Tuesday", "Wednesday", "Thursday"] else 0
    day_peak_weight = get_day_peak_weight(day_of_week)
    office_occupancy = 0.415
    time_category = get_time_category(hour)
    line_status = get_line_status(line)

    input_data = pd.DataFrame([{
        "line": line,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "time_category": time_category,
        "is_peak_hour": is_peak_hour,
        "is_noon_peak": is_noon_peak,
        "is_peak_day": is_peak_day,
        "day_peak_weight": day_peak_weight,
        "office_occupancy": office_occupancy,
        "is_strike": is_strike,
        "line_status": line_status
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "High":
        st.error("🔴 High Disruption Expected")
        st.write("Strong chance of delays or service disruption.")
    elif prediction == "Medium":
        st.warning("🟠 Moderate Disruption Expected")
        st.write("Possible delays during peak demand.")
    else:
        st.success("🟢 Low Disruption Expected")
        st.write("Travel likely smooth.")
