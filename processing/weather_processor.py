import json
from datetime import datetime

import pandas as pd

from config.config import (
    WEATHER_RAW,
    WEATHER_PROCESSED
)


# ============================================================
# Create Output Folder
# ============================================================

WEATHER_PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Find All Weather JSON Files
# ============================================================

json_files = sorted(
    WEATHER_RAW.glob("*.json"),
    key=lambda x: x.stat().st_mtime
)

if not json_files:

    raise FileNotFoundError(
        "No weather JSON files found."
    )


print(
    f"Found {len(json_files)} weather JSON file(s)."
)


# ============================================================
# Process Weather Files
# ============================================================

records = []

skipped_files = []


for json_file in json_files:

    print(
        f"Processing: {json_file.name}"
    )

    # ========================================================
    # Safely Read JSON File
    # ========================================================

    try:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except (json.JSONDecodeError, OSError) as e:

        print(
            f"WARNING: Skipping invalid JSON file: "
            f"{json_file.name}"
        )

        print(
            f"Reason: {e}"
        )

        skipped_files.append(
            json_file.name
        )

        continue


    # ========================================================
    # Check Location Metadata
    # ========================================================

    if "monitoring_location" not in data:

        print(
            f"Skipping file without location metadata: "
            f"{json_file.name}"
        )

        skipped_files.append(
            json_file.name
        )

        continue


    location = data["monitoring_location"]

    location_id = location["location_id"]

    latitude = location["latitude"]

    longitude = location["longitude"]


    # ========================================================
    # Extract Weather Information
    # ========================================================

    main = data.get(
        "main",
        {}
    )

    wind = data.get(
        "wind",
        {}
    )

    weather = data.get(
        "weather",
        []
    )

    clouds = data.get(
        "clouds",
        {}
    )

    rain = data.get(
        "rain",
        {}
    )


    # ========================================================
    # Weather Description
    # ========================================================

    if weather:

        weather_main = weather[0].get(
            "main"
        )

        weather_description = weather[0].get(
            "description"
        )

    else:

        weather_main = None

        weather_description = None


    # ========================================================
    # Rain
    # ========================================================

    rain_1h = rain.get(
        "1h",
        0
    )


    # ========================================================
    # Timestamp
    # ========================================================

    try:

        timestamp_text = (
            json_file.stem
            .split("_")[-2:]
        )

        timestamp_text = "_".join(
            timestamp_text
        )

        timestamp = datetime.strptime(
            timestamp_text,
            "%Y%m%d_%H%M%S"
        )

    except ValueError:

        timestamp = datetime.fromtimestamp(
            json_file.stat().st_mtime
        )


    # ========================================================
    # Create Record
    # ========================================================

    records.append({

        "Timestamp": timestamp,

        "Location ID": location_id,

        "Latitude": latitude,

        "Longitude": longitude,

        "Temperature (C)": main.get(
            "temp"
        ),

        "Feels Like (C)": main.get(
            "feels_like"
        ),

        "Min Temperature (C)": main.get(
            "temp_min"
        ),

        "Max Temperature (C)": main.get(
            "temp_max"
        ),

        "Pressure (hPa)": main.get(
            "pressure"
        ),

        "Humidity (%)": main.get(
            "humidity"
        ),

        "Visibility (m)": data.get(
            "visibility"
        ),

        "Wind Speed (m/s)": wind.get(
            "speed"
        ),

        "Wind Direction (deg)": wind.get(
            "deg"
        ),

        "Cloudiness (%)": clouds.get(
            "all"
        ),

        "Rain (mm)": rain_1h,

        "Weather Main": weather_main,

        "Weather Description": weather_description

    })


# ============================================================
# Validate Records
# ============================================================

if not records:

    raise ValueError(
        "No valid weather records found."
    )


# ============================================================
# Create DataFrame
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# Sort Data
# ============================================================

df = df.sort_values(
    by=[
        "Timestamp",
        "Location ID"
    ]
).reset_index(
    drop=True
)


# ============================================================
# Save Processed CSV
# ============================================================

output_file = (
    WEATHER_PROCESSED /
    "weather_processed.csv"
)


df.to_csv(
    output_file,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("=" * 60)

print(
    "Weather processing completed."
)

print(
    f"Records processed: {len(df)}"
)

print(
    f"Files skipped: {len(skipped_files)}"
)

print(
    f"Locations processed: "
    f"{df['Location ID'].nunique()}"
)

print(
    f"Output: {output_file}"
)

print("=" * 60)


# ============================================================
# Skipped Files
# ============================================================

if skipped_files:

    print(
        "\nSkipped Files:"
    )

    for file_name in skipped_files:

        print(
            f" - {file_name}"
        )


# ============================================================
# Location Summary
# ============================================================

print(
    "\nRecords by Location:"
)

print(
    df["Location ID"].value_counts()
)


# ============================================================
# Latest Records
# ============================================================

print(
    "\nLatest Weather Records:"
)

print(
    df.tail(8).to_string(
        index=False
    )
)