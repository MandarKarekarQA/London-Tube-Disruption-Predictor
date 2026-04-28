import requests
import pandas as pd
from datetime import datetime

URL = "https://api.tfl.gov.uk/Line/Mode/tube/Status"

def fetch_tfl_data():
    response = requests.get(URL)

    if response.status_code != 200:
        print("Error fetching data")
        print(response.status_code)
        return

    data = response.json()
    records = []

    for line in data:
        line_name = line["name"]

        for status in line["lineStatuses"]:
            records.append({
                "line": line_name,
                "status": status["statusSeverityDescription"],
                "severity": status["statusSeverity"],
                "timestamp": datetime.now()
            })

    df = pd.DataFrame(records)
    print(df.head())

    df.to_csv("data/raw/tfl_status.csv", index=False)
    print("Data saved to data/raw/tfl_status.csv")

if __name__ == "__main__":
    fetch_tfl_data()