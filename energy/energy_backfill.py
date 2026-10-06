import os
import math
import random
from datetime import timedelta

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DAYS = 14
BATCH_SIZE = 5000


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "smart_city_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


# IMPORTANT:
# Use URL.create() instead of manually constructing:
# postgresql://user:password@host:port/database
#
# This safely handles special characters such as @ in passwords.

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# GET EARLIEST EXISTING ENERGY TIMESTAMP
# ============================================================

def get_earliest_timestamp():

    query = text(
        """
        SELECT MIN(timestamp)
        FROM energy_data
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).scalar()

    return result


# ============================================================
# GET ENERGY LOCATIONS
# ============================================================

def get_locations():

    query = text(
        """
        SELECT
            location_id,
            location_type,
            capacity_mw
        FROM energy_locations
        ORDER BY location_id
        """
    )

    with engine.connect() as connection:

        rows = connection.execute(
            query
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# ELECTRICITY DEMAND MULTIPLIER
# ============================================================

def get_demand_multiplier(hour):

    if 0 <= hour < 5:
        return 0.55

    elif 5 <= hour < 7:
        return 0.70

    elif 7 <= hour < 10:
        return 1.05

    elif 10 <= hour < 12:
        return 0.95

    elif 12 <= hour < 16:
        return 0.90

    elif 16 <= hour < 18:
        return 1.00

    elif 18 <= hour < 21:
        return 1.15

    elif 21 <= hour < 24:
        return 0.85

    return 0.55


# ============================================================
# SOLAR GENERATION
# ============================================================

def get_solar_generation(hour, capacity):

    # No solar during night

    if hour < 6 or hour >= 18:
        return 0.0

    # Solar increases after sunrise,
    # peaks around noon,
    # decreases toward sunset.

    solar_factor = math.sin(
        math.pi * (hour - 6) / 12
    )

    solar_factor = max(
        0.0,
        solar_factor
    )

    # Solar capacity = 18% of location capacity

    solar_capacity = (
        capacity * 0.18
    )

    noise = random.uniform(
        0.90,
        1.05
    )

    solar = (
        solar_capacity
        * solar_factor
        * noise
    )

    return max(
        0.0,
        solar
    )


# ============================================================
# EV CHARGING DEMAND
# ============================================================

def get_ev_demand(hour, capacity):

    if 0 <= hour < 6:
        factor = 0.02

    elif 6 <= hour < 9:
        factor = 0.08

    elif 9 <= hour < 17:
        factor = 0.05

    elif 17 <= hour < 22:
        factor = 0.12

    else:
        factor = 0.04

    noise = random.uniform(
        0.85,
        1.15
    )

    ev_demand = (
        capacity
        * factor
        * noise
    )

    return max(
        0.0,
        ev_demand
    )


# ============================================================
# SIMULATED TEMPERATURE
# ============================================================

def get_temperature(hour):

    if 0 <= hour < 6:
        base = 25

    elif 6 <= hour < 10:
        base = 27

    elif 10 <= hour < 15:
        base = 32

    elif 15 <= hour < 18:
        base = 31

    else:
        base = 28

    return (
        base
        + random.uniform(
            -1.5,
            1.5
        )
    )


# ============================================================
# LOCATION FACTOR
# ============================================================

def get_location_factor(location_type):

    if location_type == "Commercial":

        return random.uniform(
            0.90,
            1.10
        )

    elif location_type == "Residential":

        return random.uniform(
            0.75,
            0.95
        )

    else:

        return random.uniform(
            0.85,
            1.05
        )


# ============================================================
# GENERATE ONE ENERGY RECORD
# ============================================================

def generate_record(timestamp, location):

    hour = timestamp.hour

    capacity = float(
        location["capacity_mw"]
    )

    location_type = location[
        "location_type"
    ]

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    demand_multiplier = (
        get_demand_multiplier(hour)
    )

    temperature = get_temperature(
        hour
    )

    location_factor = (
        get_location_factor(
            location_type
        )
    )

    # --------------------------------------------------------
    # Weekday / Weekend
    # --------------------------------------------------------

    if timestamp.weekday() >= 5:

        weekend_factor = 0.88

    else:

        weekend_factor = 1.00

    # --------------------------------------------------------
    # Temperature effect
    # --------------------------------------------------------

    temperature_factor = 1.00

    if temperature > 30:

        temperature_factor += (
            (temperature - 30)
            * 0.025
        )

    # --------------------------------------------------------
    # Random short-term variation
    # --------------------------------------------------------

    random_factor = random.uniform(
        0.96,
        1.04
    )

    # --------------------------------------------------------
    # Electricity demand
    # --------------------------------------------------------

    demand = (
        capacity
        * 0.65
        * demand_multiplier
        * location_factor
        * weekend_factor
        * temperature_factor
        * random_factor
    )

    demand = max(
        0.0,
        demand
    )

    # --------------------------------------------------------
    # Solar generation
    # --------------------------------------------------------

    solar = get_solar_generation(
        hour,
        capacity
    )

    # --------------------------------------------------------
    # EV charging demand
    # --------------------------------------------------------

    ev_demand = get_ev_demand(
        hour,
        capacity
    )

    # --------------------------------------------------------
    # Renewable percentage
    # --------------------------------------------------------

    if demand > 0:

        renewable_percentage = (
            solar / demand
        ) * 100

    else:

        renewable_percentage = 0.0

    renewable_percentage = min(
        100.0,
        renewable_percentage
    )

    # --------------------------------------------------------
    # Streetlights
    # --------------------------------------------------------

    if 6 <= hour < 18:

        streetlights_total = 100
        streetlights_online = 100

    else:

        streetlights_total = 100

        online_percentage = (
            random.uniform(
                96,
                99.5
            )
        )

        streetlights_online = int(
            streetlights_total
            * online_percentage
            / 100
        )

    # --------------------------------------------------------
    # Power outage
    # --------------------------------------------------------

    power_outage = (
        random.random() < 0.02
    )

    # --------------------------------------------------------
    # Transformer load
    # --------------------------------------------------------

    transformer_load = (
        demand
        / capacity
        * 100
    )

    transformer_load += (
        random.uniform(
            -3,
            3
        )
    )

    transformer_load = max(
        0.0,
        transformer_load
    )

    # --------------------------------------------------------
    # Return record
    # --------------------------------------------------------

    return {

        "timestamp": timestamp,

        "location_id":
            location["location_id"],

        "electricity_demand_mw":
            round(
                demand,
                3
            ),

        "solar_generation_mw":
            round(
                solar,
                3
            ),

        "ev_charging_demand_mw":
            round(
                ev_demand,
                3
            ),

        "renewable_percentage":
            round(
                renewable_percentage,
                2
            ),

        "transformer_load_pct":
            round(
                transformer_load,
                2
            ),

        "streetlights_online":
            streetlights_online,

        "streetlights_total":
            streetlights_total,

        "power_outage":
            power_outage,
    }


# ============================================================
# INSERT QUERY
# ============================================================

INSERT_SQL = text(
    """
    INSERT INTO energy_data (
        timestamp,
        location_id,
        electricity_demand_mw,
        solar_generation_mw,
        ev_charging_demand_mw,
        renewable_percentage,
        transformer_load_pct,
        streetlights_online,
        streetlights_total,
        power_outage
    )
    VALUES (
        :timestamp,
        :location_id,
        :electricity_demand_mw,
        :solar_generation_mw,
        :ev_charging_demand_mw,
        :renewable_percentage,
        :transformer_load_pct,
        :streetlights_online,
        :streetlights_total,
        :power_outage
    )
    ON CONFLICT (
        timestamp,
        location_id
    )
    DO NOTHING
    """
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print()
    print("=" * 65)
    print(
        "       ENERGY HISTORICAL DATA BACKFILL"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Test database connection
    # --------------------------------------------------------

    try:

        with engine.connect() as connection:

            connection.execute(
                text("SELECT 1")
            )

        print(
            "✓ PostgreSQL connection successful"
        )

    except Exception as e:

        print()
        print(
            "✗ PostgreSQL connection failed"
        )

        print()
        print(
            "Check your .env database settings:"
        )

        print(
            "DB_HOST"
        )

        print(
            "DB_PORT"
        )

        print(
            "DB_NAME"
        )

        print(
            "DB_USER"
        )

        print(
            "DB_PASSWORD"
        )

        print()
        print(
            "Error:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # Find existing energy data
    # --------------------------------------------------------

    earliest_timestamp = (
        get_earliest_timestamp()
    )

    if earliest_timestamp is None:

        print()
        print(
            "✗ No existing energy data found."
        )

        print(
            "Run the energy generator first."
        )

        return

    print()
    print(
        "Existing data starts at:"
    )

    print(
        earliest_timestamp
    )

    # --------------------------------------------------------
    # Create historical period
    # --------------------------------------------------------

    # Backfill ends one minute BEFORE
    # the earliest existing record.

    historical_end = (
        earliest_timestamp
        - timedelta(
            minutes=1
        )
    )

    # Generate exactly 14 days.

    historical_start = (
        historical_end
        - timedelta(
            days=DAYS
        )
        + timedelta(
            minutes=1
        )
    )

    # --------------------------------------------------------
    # Load locations
    # --------------------------------------------------------

    locations = get_locations()

    if not locations:

        print(
            "✗ No locations found in "
            "energy_locations."
        )

        return

    print()
    print(
        f"Locations found: "
        f"{len(locations)}"
    )

    print()
    print(
        "Historical start:"
    )

    print(
        historical_start
    )

    print()
    print(
        "Historical end:"
    )

    print(
        historical_end
    )

    # --------------------------------------------------------
    # Calculate expected records
    # --------------------------------------------------------

    total_minutes = (
        int(
            (
                historical_end
                - historical_start
            ).total_seconds()
            / 60
        )
        + 1
    )

    expected_records = (
        total_minutes
        * len(locations)
    )

    print()
    print(
        f"Expected timestamps: "
        f"{total_minutes:,}"
    )

    print(
        f"Expected records: "
        f"{expected_records:,}"
    )

    print()
    print("=" * 65)

    # --------------------------------------------------------
    # Fixed random seed
    # --------------------------------------------------------

    random.seed(42)

    # --------------------------------------------------------
    # Generate and insert data
    # --------------------------------------------------------

    batch = []

    processed = 0

    current_timestamp = (
        historical_start
    )

    while current_timestamp <= historical_end:

        for location in locations:

            record = generate_record(
                current_timestamp,
                location
            )

            batch.append(record)

            # Insert every 5,000 records

            if len(batch) >= BATCH_SIZE:

                with engine.begin() as connection:

                    connection.execute(
                        INSERT_SQL,
                        batch
                    )

                processed += len(batch)

                print(
                    f"Processed: "
                    f"{processed:,} / "
                    f"{expected_records:,}"
                )

                batch = []

        current_timestamp += (
            timedelta(
                minutes=1
            )
        )

    # --------------------------------------------------------
    # Insert remaining records
    # --------------------------------------------------------

    if batch:

        with engine.begin() as connection:

            connection.execute(
                INSERT_SQL,
                batch
            )

        processed += len(batch)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 65)

    print(
        "✓ BACKFILL COMPLETE"
    )

    print(
        f"Processed records: "
        f"{processed:,}"
    )

    print("=" * 65)

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()