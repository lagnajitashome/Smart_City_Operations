import json
from datetime import datetime

import pandas as pd

from config.config import (
    TRAFFIC_RAW,
    TRAFFIC_PROCESSED
)


# ==========================================
# Create Output Folder
# ==========================================

TRAFFIC_PROCESSED.mkdir(parents=True, exist_ok=True)


# ==========================================
# Find All Raw JSON Files
# ==========================================

json_files = sorted(
    TRAFFIC_RAW.glob("*.json"),
    key=lambda x: x.stat().st_mtime
)

if not json_files:
    raise FileNotFoundError(
        "No traffic JSON files found."
    )


print(f"Found {len(json_files)} traffic JSON file(s).")


# ==========================================
# Process Each JSON File
# ==========================================

records = []

for json_file in json_files:

    print(f"Processing: {json_file.name}")

    with open(json_file, "r") as file:
        data = json.load(file)

    # ======================================
    # Check for New Multi-Location Format
    # ======================================

    if "monitoring_location" not in data:

        print(
            f"Skipping old-format file: "
            f"{json_file.name}"
        )

        continue


    location = data["monitoring_location"]

    location_id = location["location_id"]
    latitude = location["latitude"]
    longitude = location["longitude"]


    # ======================================
    # Traffic Data
    # ======================================

    traffic = data["flowSegmentData"]


    # ======================================
    # Feature Engineering
    # ======================================

    delay = (
        traffic["currentTravelTime"]
        - traffic["freeFlowTravelTime"]
    )


    if traffic["freeFlowSpeed"] > 0:

        speed_reduction = (
            (
                traffic["freeFlowSpeed"]
                - traffic["currentSpeed"]
            )
            /
            traffic["freeFlowSpeed"]
        ) * 100

    else:

        speed_reduction = 0


    # ======================================
    # Timestamp
    # ======================================

    # Get timestamp from filename
    # Example:
    # PARK_STREET_20260814_190309.json

    try:

        timestamp_text = (
            json_file.stem
            .split("_")[-2:]
        )

        timestamp_text = "_".join(timestamp_text)

        timestamp = datetime.strptime(
            timestamp_text,
            "%Y%m%d_%H%M%S"
        )

    except ValueError:

        # Fallback if filename format is unexpected
        timestamp = datetime.fromtimestamp(
            json_file.stat().st_mtime
        )


    # ======================================
    # Create Record
    # ======================================

    records.append({

        "Timestamp": timestamp,

        "Location ID": location_id,

        "Latitude": latitude,

        "Longitude": longitude,

        "Current Speed (km/h)": traffic["currentSpeed"],

        "Free Flow Speed (km/h)": traffic["freeFlowSpeed"],

        "Current Travel Time (sec)": traffic["currentTravelTime"],

        "Free Flow Travel Time (sec)": traffic["freeFlowTravelTime"],

        "Traffic Delay (sec)": delay,

        "Speed Reduction (%)": round(
            speed_reduction,
            2
        ),

        "Confidence": traffic["confidence"],

        "Road Closed": traffic["roadClosure"],

        "Road Class": traffic["frc"]

    })


# ==========================================
# Check Records
# ==========================================

if not records:

    raise ValueError(
        "No new multi-location traffic records found."
    )


# ==========================================
# Create Historical DataFrame
# ==========================================

df = pd.DataFrame(records)


# ==========================================
# Sort by Timestamp and Location
# ==========================================

df = df.sort_values(
    by=[
        "Timestamp",
        "Location ID"
    ]
).reset_index(drop=True)


# ==========================================
# Save Historical CSV
# ==========================================

filename = (
    TRAFFIC_PROCESSED
    /
    "traffic_processed.csv"
)

df.to_csv(
    filename,
    index=False
)


# ==========================================
# Summary
# ==========================================

print("=" * 60)

print("Traffic processing completed.")

print(
    f"Records processed: {len(df)}"
)

print(
    f"Locations processed: "
    f"{df['Location ID'].nunique()}"
)

print(
    f"Output: {filename}"
)

print("=" * 60)


# ==========================================
# Location Summary
# ==========================================

print("\nRecords by Location:")

print(
    df["Location ID"]
    .value_counts()
)


# ==========================================
# Latest Traffic Records
# ==========================================

print("\nLatest Traffic Records:")

print(
    df.tail(8).to_string(index=False)
)