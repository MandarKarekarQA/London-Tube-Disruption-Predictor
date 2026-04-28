import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "data/processed/tfl_training_data.csv"
MODEL_FILE = "models/tube_disruption_model.pkl"


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
        "is_noon_peak",
        "is_peak_day",
        "day_peak_weight",
        "office_occupancy",
        "is_strike",
        "line_status"
    ]

    target = "disruption_level"

    X = df[features]
    y = df[target]

    categorical_features = [
        "line",
        "day_of_week",
        "month",
        "time_category"
    ]

    numeric_features = [
        "hour",
        "is_weekend",
        "is_peak_hour",
        "is_noon_peak",
        "is_peak_day",
        "day_peak_weight",
        "office_occupancy",
        "is_strike",
        "line_status"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if len(df) < 20:
        print("WARNING: Dataset is very small. Training demo model on full dataset.")
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()