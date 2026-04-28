import pandas as pd
from itertools import product

OUTPUT_FILE = "data/processed/tfl_training_data.csv"

LINES = [
    "Bakerloo", "Central", "Circle", "District", "Hammersmith & City",
    "Jubilee", "Metropolitan", "Northern", "Piccadilly",
    "Victoria", "Waterloo & City"
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

HOURS = [6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22]


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
    # lower value = weaker reliability / more risk
    improving_lines = ["Central", "District", "Metropolitan", "Hammersmith & City", "Circle"]
    extended_lines = ["Northern"]
    higher_risk_lines = ["Piccadilly", "Bakerloo", "Waterloo & City"]

    if line in improving_lines:
        return 2
    elif line in extended_lines:
        return 2
    elif line in higher_risk_lines:
        return 1
    else:
        return 2


def get_disruption_level(row):
    score = 0

    # Strike has biggest impact
    if row["is_strike"] == 1:
        score += 3

    # Peak time demand
    if row["is_peak_hour"] == 1:
        score += 1

    # Tuesday/Wednesday/Thursday office demand
    if row["is_peak_day"] == 1:
        score += 1

    # High-risk infrastructure / older line factor
    if row["line_status"] == 1:
        score += 1

    # Some lines historically busier / more complex
    if row["line"] in ["Central", "District", "Northern", "Piccadilly"]:
        score += 1

    # Quiet time reduces risk
    if row["hour"] in [6, 20, 22]:
        score -= 1

    # Weekend usually lower commuter pressure
    if row["is_weekend"] == 1:
        score -= 1

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def create_training_data():
    rows = []

    for line, day, month, hour, strike in product(LINES, DAYS, MONTHS, HOURS, [0, 1]):

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        is_peak_hour = 1 if (7 <= hour <= 10 or 12 <= hour <= 14 or 16 <= hour <= 19) else 0
        is_noon_peak = 1 if 12 <= hour <= 14 else 0
        is_peak_day = 1 if day in ["Tuesday", "Wednesday", "Thursday"] else 0
        day_peak_weight = get_day_peak_weight(day)
        office_occupancy = 0.415
        time_category = get_time_category(hour)
        line_status = get_line_status(line)

        row = {
            "line": line,
            "hour": hour,
            "day_of_week": day,
            "month": month,
            "is_weekend": is_weekend,
            "time_category": time_category,
            "is_peak_hour": is_peak_hour,
            "is_noon_peak": is_noon_peak,
            "is_peak_day": is_peak_day,
            "day_peak_weight": day_peak_weight,
            "office_occupancy": office_occupancy,
            "is_strike": strike,
            "line_status": line_status
        }

        row["disruption_level"] = get_disruption_level(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(df.head())
    print("Rows created:", len(df))
    print("Class distribution:")
    print(df["disruption_level"].value_counts())
    print(f"Training data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    create_training_data()