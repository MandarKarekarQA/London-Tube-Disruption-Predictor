import pandas as pd

INPUT_FILE = "data/raw/tfl_status.csv"
OUTPUT_FILE = "data/processed/tfl_training_data.csv"


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
    """
    Simple infrastructure/reliability factor.
    Higher = stronger/improving infrastructure.
    Lower = more likely to face disruption.
    """
    improved_lines = ["Elizabeth"]
    improving_lines = ["Central", "District", "Metropolitan", "Hammersmith & City", "Circle"]
    upgrading_lines = ["Piccadilly"]
    extended_lines = ["Northern"]

    if line in improved_lines:
        return 3
    elif line in improving_lines:
        return 2
    elif line in extended_lines:
        return 2
    elif line in upgrading_lines:
        return 1
    else:
        return 1


def get_disruption_level(row):
    """
    Creates simple target label:
    Low = normal / low disruption
    Medium = some disruption risk
    High = strong disruption risk
    """

    score = 0

    # TfL status signal
    if row["status"] != "Good Service":
        score += 2

    # Peak demand signal
    if row["is_peak_hour"] == 1:
        score += 1

    # Noon hybrid-working signal
    if row["is_noon_peak"] == 1:
        score += 1

    # Busy office day signal
    if row["is_peak_day"] == 1:
        score += 1

    # Infrastructure/reliability signal
    if row["line_status"] == 1:
        score += 1

    # Final label
    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def prepare_dataset():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month_name()

    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    df["time_category"] = df["hour"].apply(get_time_category)

    df["is_noon_peak"] = df["hour"].between(12, 14).astype(int)

    df["is_peak_hour"] = df["hour"].apply(
        lambda x: 1 if (7 <= x <= 10 or 12 <= x <= 14 or 16 <= x <= 19) else 0
    )

    df["is_peak_day"] = df["day_of_week"].isin(
        ["Tuesday", "Wednesday", "Thursday"]
    ).astype(int)

    df["day_peak_weight"] = df["day_of_week"].apply(get_day_peak_weight)

    # London office occupancy estimate based on hybrid working trend
    df["office_occupancy"] = 0.415

    # For now, default strike flag is 0.
    # Later we will connect this to strikes.csv.
    df["is_strike"] = 0

    df["line_status"] = df["line"].apply(get_line_status)

    df["disruption_level"] = df.apply(get_disruption_level, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(df.head())
    print(f"Processed dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_dataset()