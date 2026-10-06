import os
import json
import time
import requests
import subprocess
import sys

from datetime import datetime

from config.settings import OPENWEATHER_API_KEY
from config.config import WEATHER_RAW


# ============================================================
# Kolkata Weather Monitoring Locations
# ============================================================

WEATHER_LOCATIONS = {

    "PARK_STREET": {
        "latitude": 22.5535,
        "longitude": 88.3520
    },

    "ESPLANADE": {
        "latitude": 22.5660,
        "longitude": 88.3510
    },

    "EM_BYPASS": {
        "latitude": 22.5140,
        "longitude": 88.3970
    },

    "SALT_LAKE": {
        "latitude": 22.5800,
        "longitude": 88.4170
    },

    "NEW_TOWN": {
        "latitude": 22.5958,
        "longitude": 88.4497
    },

    "VIP_ROAD": {
        "latitude": 22.6250,
        "longitude": 88.4050
    },

    "RASHBEHARI": {
        "latitude": 22.5190,
        "longitude": 88.3520
    },

    "HOWRAH_APPROACH": {
        "latitude": 22.5850,
        "longitude": 88.3300
    }
}


# ============================================================
# Settings
# ============================================================

# 10 minutes = 600 seconds
INTERVAL_SECONDS = 600

os.makedirs(WEATHER_RAW, exist_ok=True)


# ============================================================
# Collect Weather for One Location
# ============================================================

def collect_location(location_id, latitude, longitude):

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={latitude}"
        f"&lon={longitude}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )

    try:

        response = requests.get(
            url,
            timeout=30
        )

        if response.status_code == 200:

            weather_data = response.json()

            # Add monitoring metadata
            weather_data["monitoring_location"] = {

                "location_id": location_id,

                "latitude": latitude,

                "longitude": longitude
            }

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                WEATHER_RAW /
                f"{location_id}_{timestamp}.json"
            )

            with open(
                filename,
                "w"
            ) as file:

                json.dump(
                    weather_data,
                    file,
                    indent=4
                )

            print(
                f"✓ {location_id} | "
                f"Weather data collected"
            )

            return True

        else:

            print(
                f"✗ {location_id} | "
                f"API Error: {response.status_code}"
            )

            return False

    except requests.exceptions.RequestException as e:

        print(
            f"✗ {location_id} | "
            f"Connection Error: {e}"
        )

        return False


# ============================================================
# Collect All Locations
# ============================================================

def collect_all_locations():

    print()
    print("=" * 65)

    print(
        f"Weather collection started at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print("=" * 65)

    successful = 0

    for location_id, location in WEATHER_LOCATIONS.items():

        success = collect_location(

            location_id,

            location["latitude"],

            location["longitude"]

        )

        if success:
            successful += 1

    print("=" * 65)

    print(
        f"Weather collection completed: "
        f"{successful}/{len(WEATHER_LOCATIONS)} locations"
    )

    print("=" * 65)

    return successful


# ============================================================
# Run Weather Processing
# ============================================================

def process_weather_data():

    print()
    print("=" * 65)
    print("STEP 2: PROCESSING WEATHER DATA")
    print("=" * 65)

    try:

        subprocess.run(
            [
                sys.executable,
                "-m",
                "processing.weather_processor"
            ],
            check=True
        )

        print("✓ Weather processing completed.")

        return True

    except subprocess.CalledProcessError as e:

        print(
            f"✗ Weather processing failed: {e}"
        )

        return False


# ============================================================
# Insert Weather Data into PostgreSQL
# ============================================================

def insert_weather_data():

    print()
    print("=" * 65)
    print("STEP 3: INSERTING WEATHER DATA INTO POSTGRESQL")
    print("=" * 65)

    try:

        subprocess.run(
            [
                sys.executable,
                "-m",
                "database.insert_weather"
            ],
            check=True
        )

        print("✓ Weather database insertion completed.")

        return True

    except subprocess.CalledProcessError as e:

        print(
            f"✗ Weather database insertion failed: {e}"
        )

        return False


# ============================================================
# Complete Weather Pipeline
# ============================================================

def run_weather_cycle():

    print()
    print("#" * 70)
    print("STARTING WEATHER PIPELINE CYCLE")
    print("#" * 70)

    # STEP 1
    successful = collect_all_locations()

    if successful == 0:

        print(
            "No weather data collected. "
            "Skipping processing and database insertion."
        )

        return

    # STEP 2
    processed = process_weather_data()

    if not processed:

        print(
            "Weather processing failed. "
            "Database insertion skipped."
        )

        return

    # STEP 3
    inserted = insert_weather_data()

    if not inserted:

        print(
            "Weather database insertion failed."
        )

        return

    print()
    print("#" * 70)
    print("WEATHER PIPELINE CYCLE COMPLETED")
    print("#" * 70)


# ============================================================
# Continuous Weather Pipeline
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("SMART CITY WEATHER PIPELINE")

    print("=" * 70)

    print(
        f"Monitoring {len(WEATHER_LOCATIONS)} locations"
    )

    print(
        "Collection + Processing + PostgreSQL insertion"
    )

    print(
        "Interval: 10 minutes"
    )

    print(
        "Press CTRL+C to stop"
    )

    print("=" * 70)


    try:

        while True:

            # Run complete pipeline
            run_weather_cycle()

            print()
            print("=" * 70)

            print(
                f"Next weather pipeline run in "
                f"{INTERVAL_SECONDS // 60} minutes..."
            )

            print("=" * 70)

            # Wait 10 minutes
            time.sleep(
                INTERVAL_SECONDS
            )


    except KeyboardInterrupt:

        print()
        print("=" * 70)

        print(
            "Weather pipeline stopped."
        )

        print("=" * 70)