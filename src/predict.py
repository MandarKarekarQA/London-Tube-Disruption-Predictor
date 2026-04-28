import pandas as pd
import joblib

MODEL_FILE = "models/tube_disruption_model.pkl"


def get_day_peak_weight(day):
    weights = {
        "Monday": 0.5,
        "Tuesday": 1.0,
        "Wednesday": 0.9,
        "Thursday": 0.8,
        "Friday": 0.3,
        "Saturday": 0.2,
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


def predict_disruption(line, hour, day_of_week, month, is_strike=0):
    model = joblib.load(MODEL_FILE)

    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    time_category = get_time_category(hour)
    is_noon_peak = 1 if 12 <= hour <= 14 else 0
    is_peak_hour = 1 if (7 <= hour <= 10 or 12 <= hour <= 14 or 16 <= hour <= 19) else 0
    is_peak_day = 1 if day_of_week in ["Tuesday", "Wednesday", "Thursday"] else 0
    day_peak_weight = get_day_peak_weight(day_of_week)
    office_occupancy = 0.415
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

    print("Prediction:", prediction)
    return prediction


if __name__ == "__main__":
    predict_disruption(
        line="District",
        hour=8,
        day_of_week="Tuesday",
        month="April",
        is_strike=0
    )