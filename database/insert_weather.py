import pandas as pd

from sqlalchemy import text

from database.connection import engine
from config.config import WEATHER_PROCESSED


# ============================================
# Find Processed Weather CSV
# ============================================

csv_file = WEATHER_PROCESSED / "weather_processed.csv"

if not csv_file.exists():
    raise FileNotFoundError(
        "Processed weather CSV not found."
    )

print("=" * 60)
print(f"Reading: {csv_file.name}")
print("=" * 60)


# ============================================
# Read CSV
# ============================================

df = pd.read_csv(csv_file)

print(f"Records in CSV: {len(df)}")
print()


# ============================================
# Insert into PostgreSQL
# ============================================

inserted = 0
skipped = 0

with engine.begin() as conn:

    for _, row in df.iterrows():

        result = conn.execute(
            text("""
                INSERT INTO weather_data
                (
                    timestamp,
                    location_id,
                    latitude,
                    longitude,
                    temperature,
                    feels_like,
                    min_temperature,
                    max_temperature,
                    pressure,
                    humidity,
                    visibility,
                    wind_speed,
                    wind_direction,
                    cloudiness,
                    rain_mm,
                    weather_main,
                    weather_description
                )

                VALUES
                (
                    :timestamp,
                    :location_id,
                    :latitude,
                    :longitude,
                    :temperature,
                    :feels_like,
                    :min_temperature,
                    :max_temperature,
                    :pressure,
                    :humidity,
                    :visibility,
                    :wind_speed,
                    :wind_direction,
                    :cloudiness,
                    :rain_mm,
                    :weather_main,
                    :weather_description
                )

                ON CONFLICT (timestamp, location_id)
                DO NOTHING
            """),

            {
                "timestamp": row["Timestamp"],

                "location_id": row["Location ID"],

                "latitude": row["Latitude"],

                "longitude": row["Longitude"],

                "temperature": row["Temperature (C)"],

                "feels_like": row["Feels Like (C)"],

                "min_temperature": row["Min Temperature (C)"],

                "max_temperature": row["Max Temperature (C)"],

                "pressure": row["Pressure (hPa)"],

                "humidity": row["Humidity (%)"],

                "visibility": row["Visibility (m)"],

                "wind_speed": row["Wind Speed (m/s)"],

                "wind_direction": row["Wind Direction (deg)"],

                "cloudiness": row["Cloudiness (%)"],

                "rain_mm": row["Rain (mm)"],

                "weather_main": row["Weather Main"],

                "weather_description": row["Weather Description"]
            }
        )

        if result.rowcount == 1:
            inserted += 1
        else:
            skipped += 1


# ============================================
# Summary
# ============================================

print("=" * 60)

print("Weather database ingestion completed.")

print(
    f"New records inserted : {inserted}"
)

print(
    f"Duplicate records skipped : {skipped}"
)

print("=" * 60)