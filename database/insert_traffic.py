import pandas as pd

from sqlalchemy import text

from database.connection import engine
from config.config import TRAFFIC_PROCESSED


# ============================================
# Find Processed CSV
# ============================================

csv_file = TRAFFIC_PROCESSED / "traffic_processed.csv"

if not csv_file.exists():
    raise FileNotFoundError(
        "Processed traffic CSV not found."
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
                INSERT INTO traffic_data
                (
                    timestamp,
                    location_id,
                    latitude,
                    longitude,
                    current_speed,
                    free_flow_speed,
                    current_travel_time,
                    free_flow_travel_time,
                    traffic_delay,
                    speed_reduction,
                    confidence,
                    road_closure,
                    frc
                )

                VALUES
                (
                    :timestamp,
                    :location_id,
                    :latitude,
                    :longitude,
                    :current_speed,
                    :free_flow_speed,
                    :current_travel_time,
                    :free_flow_travel_time,
                    :traffic_delay,
                    :speed_reduction,
                    :confidence,
                    :road_closure,
                    :frc
                )

                ON CONFLICT (timestamp, location_id)
                DO NOTHING
            """),

            {
                "timestamp": row["Timestamp"],
                "location_id": row["Location ID"],
                "latitude": row["Latitude"],
                "longitude": row["Longitude"],
                "current_speed": row["Current Speed (km/h)"],
                "free_flow_speed": row["Free Flow Speed (km/h)"],
                "current_travel_time": row["Current Travel Time (sec)"],
                "free_flow_travel_time": row["Free Flow Travel Time (sec)"],
                "traffic_delay": row["Traffic Delay (sec)"],
                "speed_reduction": row["Speed Reduction (%)"],
                "confidence": row["Confidence"],
                "road_closure": row["Road Closed"],
                "frc": row["Road Class"]
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
print("Traffic database ingestion completed.")
print(f"New records inserted : {inserted}")
print(f"Duplicate records skipped : {skipped}")
print("=" * 60)