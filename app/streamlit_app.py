import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = "data/processed/tfl_training_data.csv"


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    features = [
        "line",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "time_category",
        "is_peak_hour",
        "is_peak_day",
        "day_peak_weight",
        "office_occupancy",
        "is_strike",
        "line_status",
    ]

    target = "disruption_level"

    X = df[features]
    y = df[target]

    categorical_features = ["line", "day_of_week", "month", "time_category"]
    numeric_features = [
        "hour",
        "is_weekend",
        "is_peak_hour",
        "is_peak_day",
        "day_peak_weight",
        "office_occupancy",
        "is_strike",
        "line_status",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    model.fit(X, y)
    return model


def get_time_category(hour):
    if 7 <= hour <= 9:
        return "morning_peak"
    elif 10 <= hour <= 15:
        return "daytime"
    elif 16 <= hour <= 19:
        return "evening_peak"
    else:
        return "night"


def get_day_peak_weight(day):
    weights = {
        "Monday": 0.6,
        "Tuesday": 1.0,
        "Wednesday": 1.0,
        "Thursday": 0.9,
        "Friday": 0.4,
        "Saturday": 0.3,
        "Sunday": 0.2,
    }
    return weights.get(day, 0.5)


def get_office_occupancy(day):
    occupancy = {
        "Monday": 0.28,
        "Tuesday": 0.43,
        "Wednesday": 0.39,
        "Thursday": 0.36,
        "Friday": 0.18,
        "Saturday": 0.10,
        "Sunday": 0.08,
    }
    return occupancy.get(day, 0.28)


def get_line_status(line):
    busy_lines = ["Central", "District", "Northern", "Piccadilly", "Victoria"]
    medium_lines = ["Circle", "Jubilee", "Metropolitan", "Hammersmith & City"]

    if line in busy_lines:
        return 2
    elif line in medium_lines:
        return 1
    else:
        return 0


def format_time(hour):
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


st.set_page_config(
    page_title="London Tube Disruption Predictor",
    page_icon="🚇",
    layout="centered",
)

st.title("🚇 London Tube Disruption Predictor")

st.write(
    "Predict disruption level based on travel time, demand, strike patterns, "
    "office occupancy, and Tube line behaviour."
)

model = train_model()

lines = [
    "Bakerloo",
    "Central",
    "Circle",
    "District",
    "Hammersmith & City",
    "Jubilee",
    "Metropolitan",
    "Northern",
    "Piccadilly",
    "Victoria",
    "Waterloo & City",
]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

selected_line = st.selectbox("Select Tube Line", lines)

time_options = {format_time(hour): hour for hour in range(0, 24)}
selected_time_label = st.selectbox("Select Travel Time", list(time_options.keys()), index=8)
selected_hour = time_options[selected_time_label]

selected_day = st.selectbox("Select Day", days)
selected_month = st.selectbox("Select Month", months)

strike_expected = st.selectbox("Strike expected?", ["No", "Yes"])
is_strike = 1 if strike_expected == "Yes" else 0

is_weekend = 1 if selected_day in ["Saturday", "Sunday"] else 0
time_category = get_time_category(selected_hour)
is_peak_hour = 1 if time_category in ["morning_peak", "evening_peak"] else 0
is_peak_day = 1 if selected_day in ["Tuesday", "Wednesday", "Thursday"] else 0
day_peak_weight = get_day_peak_weight(selected_day)
office_occupancy = get_office_occupancy(selected_day)
line_status = get_line_status(selected_line)

input_data = pd.DataFrame(
    [
        {
            "line": selected_line,
            "hour": selected_hour,
            "day_of_week": selected_day,
            "month": selected_month,
            "is_weekend": is_weekend,
            "time_category": time_category,
            "is_peak_hour": is_peak_hour,
            "is_peak_day": is_peak_day,
            "day_peak_weight": day_peak_weight,
            "office_occupancy": office_occupancy,
            "is_strike": is_strike,
            "line_status": line_status,
        }
    ]
)

if st.button("Predict Disruption"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("🟢 Low Disruption Expected")
    elif prediction == "Medium":
        st.warning("🟠 Moderate Disruption Expected")
    else:
        st.error("🔴 High Disruption Expected")

    with st.expander("Why this prediction?"):
        st.write("The prediction is based on:")
        st.write(f"- Tube line: **{selected_line}**")
        st.write(f"- Travel time: **{selected_time_label}**")
        st.write(f"- Day: **{selected_day}**")
        st.write(f"- Month: **{selected_month}**")
        st.write(f"- Time category: **{time_category}**")
        st.write(f"- Office occupancy estimate: **{office_occupancy}**")
        st.write(f"- Strike expected: **{strike_expected}**")
        st.write(f"- Line behaviour score: **{line_status}**")