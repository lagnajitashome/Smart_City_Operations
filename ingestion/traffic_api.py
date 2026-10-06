import os
import json
import time
import requests
import subprocess
import sys

from datetime import datetime

from config.settings import TOMTOM_API_KEY
from config.config import TRAFFIC_RAW


# ============================================================
# Kolkata Traffic Monitoring Locations
# ============================================================

TRAFFIC_LOCATIONS = {

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

INTERVAL_SECONDS = 300   # 5 minutes

os.makedirs(TRAFFIC_RAW, exist_ok=True)


# ============================================================
# Collect Traffic for One Location
# ============================================================

def collect_location(location_id, latitude, longitude):

    url = (
        "https://api.tomtom.com/traffic/services/4/"
        "flowSegmentData/absolute/10/json"
        f"?point={latitude},{longitude}"
        f"&key={TOMTOM_API_KEY}"
    )

    try:

        response = requests.get(
            url,
            timeout=30
        )

        if response.status_code == 200:

            traffic_data = response.json()

            # Add our own location metadata
            traffic_data["monitoring_location"] = {
                "location_id": location_id,
                "latitude": latitude,
                "longitude": longitude
            }

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                TRAFFIC_RAW /
                f"{location_id}_{timestamp}.json"
            )

            with open(filename, "w") as file:

                json.dump(
                    traffic_data,
                    file,
                    indent=4
                )

            print(
                f"✓ {location_id} | "
                f"Traffic data collected"
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
        f"Traffic collection started at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 65)

    successful = 0

    for location_id, location in TRAFFIC_LOCATIONS.items():

        success = collect_location(
            location_id,
            location["latitude"],
            location["longitude"]
        )

        if success:
            successful += 1

    print("=" * 65)
    print(
        f"Collection completed: "
        f"{successful}/{len(TRAFFIC_LOCATIONS)} locations"
    )
    print("=" * 65)

    return successful


# ============================================================
# Process Traffic Data
# ============================================================

def process_traffic_data():

    print()
    print("=" * 65)
    print("PROCESSING TRAFFIC DATA")
    print("=" * 65)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "processing.traffic_processor"
        ],
        check=False
    )

    if result.returncode != 0:

        print("✗ Traffic processing failed.")

        return False

    print("✓ Traffic processing completed.")

    return True


# ============================================================
# Insert Traffic Data into PostgreSQL
# ============================================================

def insert_traffic_data():

    print()
    print("=" * 65)
    print("INSERTING TRAFFIC DATA INTO POSTGRESQL")
    print("=" * 65)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "database.insert_traffic"
        ],
        check=False
    )

    if result.returncode != 0:

        print("✗ PostgreSQL ingestion failed.")

        return False

    print("✓ PostgreSQL ingestion completed.")

    return True


# ============================================================
# Complete Traffic Pipeline
# ============================================================

def run_traffic_pipeline():

    successful = collect_all_locations()

    # Only continue if at least one location was collected
    if successful == 0:

        print(
            "✗ No traffic data collected. "
            "Skipping processing and database ingestion."
        )

        return

    # Process newly collected + historical raw files
    processed = process_traffic_data()

    if not processed:
        return

    # Insert processed data into PostgreSQL
    insert_traffic_data()


# ============================================================
# Continuous Collection
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("SMART CITY TRAFFIC INGESTION PIPELINE")
    print("=" * 65)

    print(
        f"Monitoring {len(TRAFFIC_LOCATIONS)} locations"
    )

    print("Collection interval: 5 minutes")

    print(
        "Pipeline: "
        "API → Processing → PostgreSQL"
    )

    print("Press CTRL+C to stop")

    print("=" * 65)

    try:

        while True:

            # Run complete pipeline
            run_traffic_pipeline()

            print()
            print("=" * 65)

            print(
                f"Next collection in "
                f"{INTERVAL_SECONDS // 60} minutes..."
            )

            print("=" * 65)

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:

        print()

        print("=" * 65)

        print("Traffic ingestion pipeline stopped.")

        print("=" * 65)