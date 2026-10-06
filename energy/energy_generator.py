import time
import random
from datetime import datetime

import numpy as np
from sqlalchemy import text

from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

SLEEP_SECONDS = 60

# Base demand characteristics for each location
LOCATION_PROFILES = {

    "PARK_STREET": {
        "base_demand": 15.0,
        "solar_capacity": 4.0,
        "ev_capacity": 3.0,
        "streetlights": 850
    },

    "ESPLANADE": {
        "base_demand": 18.0,
        "solar_capacity": 5.0,
        "ev_capacity": 3.5,
        "streetlights": 950
    },

    "EM_BYPASS": {
        "base_demand": 21.0,
        "solar_capacity": 6.0,
        "ev_capacity": 4.0,
        "streetlights": 1100
    },

    "SALT_LAKE": {
        "base_demand": 17.0,
        "solar_capacity": 5.0,
        "ev_capacity": 3.0,
        "streetlights": 1000
    },

    "NEW_TOWN": {
        "base_demand": 24.0,
        "solar_capacity": 8.0,
        "ev_capacity": 5.0,
        "streetlights": 1200
    },

    "VIP_ROAD": {
        "base_demand": 14.0,
        "solar_capacity": 4.0,
        "ev_capacity": 2.5,
        "streetlights": 800
    },

    "RASHBEHARI": {
        "base_demand": 16.0,
        "solar_capacity": 4.5,
        "ev_capacity": 3.0,
        "streetlights": 900
    },

    "HOWRAH_APPROACH": {
        "base_demand": 19.0,
        "solar_capacity": 5.0,
        "ev_capacity": 3.5,
        "streetlights": 1000
    }
}


# ============================================================
# TIME-BASED DEMAND PROFILE
# ============================================================

def get_demand_multiplier(hour):
    """
    Simulates realistic electricity demand throughout the day.
    """

    # Night
    if 0 <= hour < 5:
        return 0.55

    # Early morning
    elif 5 <= hour < 7:
        return 0.70

    # Morning peak
    elif 7 <= hour < 10:
        return 1.05

    # Late morning
    elif 10 <= hour < 12:
        return 0.95

    # Afternoon
    elif 12 <= hour < 16:
        return 0.90

    # Evening increase
    elif 16 <= hour < 18:
        return 1.00

    # Evening peak
    elif 18 <= hour < 21:
        return 1.15

    # Late evening
    else:
        return 0.85


# ============================================================
# SOLAR GENERATION
# ============================================================

def generate_solar_generation(hour, solar_capacity):
    """
    Simulates solar generation based on time of day.
    """

    # No solar during night
    if hour < 6 or hour >= 18:
        return 0.0

    # Solar generation profile
    solar_profile = {

        6: 0.10,
        7: 0.25,
        8: 0.40,
        9: 0.55,
        10: 0.70,
        11: 0.85,
        12: 1.00,
        13: 1.00,
        14: 0.95,
        15: 0.80,
        16: 0.60,
        17: 0.35
    }

    multiplier = solar_profile.get(hour, 0.0)

    # Small natural variation
    variation = random.uniform(0.90, 1.05)

    generation = (
        solar_capacity
        * multiplier
        * variation
    )

    return round(
        max(0.0, generation),
        2
    )


# ============================================================
# EV CHARGING DEMAND
# ============================================================

def generate_ev_demand(hour, ev_capacity):
    """
    Simulates EV charging demand.
    """

    # Very low overnight
    if 0 <= hour < 6:
        multiplier = 0.15

    # Morning charging
    elif 6 <= hour < 10:
        multiplier = 0.65

    # Daytime
    elif 10 <= hour < 17:
        multiplier = 0.45

    # Evening charging peak
    elif 17 <= hour < 22:
        multiplier = 0.85

    # Late night
    else:
        multiplier = 0.30

    variation = random.uniform(0.85, 1.10)

    demand = (
        ev_capacity
        * multiplier
        * variation
    )

    return round(
        max(0.0, demand),
        2
    )


# ============================================================
# TEMPERATURE SIMULATION
# ============================================================

def generate_temperature(hour):
    """
    Simulates approximate daily temperature.
    """

    temperature_profile = {

        0: 25,
        1: 25,
        2: 24,
        3: 24,
        4: 24,
        5: 24,
        6: 25,
        7: 26,
        8: 27,
        9: 29,
        10: 30,
        11: 31,
        12: 32,
        13: 33,
        14: 33,
        15: 32,
        16: 31,
        17: 30,
        18: 29,
        19: 28,
        20: 27,
        21: 27,
        22: 26,
        23: 26
    }

    base_temperature = temperature_profile.get(
        hour,
        28
    )

    variation = random.uniform(
        -1.5,
        1.5
    )

    return round(
        base_temperature + variation,
        1
    )


# ============================================================
# STREETLIGHT STATUS
# ============================================================

def generate_streetlight_status(hour, total_streetlights):
    """
    Simulates streetlight availability.

    During nighttime, most lights are expected to be online.
    During daytime, lights are mostly offline.
    """

    if 6 <= hour < 18:

        # Daytime
        online_percentage = random.uniform(
            0.01,
            0.05
        )

    else:

        # Night
        online_percentage = random.uniform(
            0.96,
            0.995
        )

    online = int(
        total_streetlights
        * online_percentage
    )

    # At night, calculate actual operational lights
    if hour < 6 or hour >= 18:

        online = int(
            total_streetlights
            * random.uniform(
                0.96,
                0.995
            )
        )

    else:

        online = 0

    return max(
        0,
        min(
            total_streetlights,
            online
        )
    )


# ============================================================
# POWER OUTAGE SIMULATION
# ============================================================

def generate_power_outage():
    """
    Small probability of a simulated outage.
    """

    return random.random() < 0.02


# ============================================================
# ENERGY RECORD GENERATION
# ============================================================

def generate_energy_record(
    location_id,
    profile,
    timestamp
):

    hour = timestamp.hour

    # --------------------------------------------------------
    # Basic demand
    # --------------------------------------------------------

    demand_multiplier = get_demand_multiplier(
        hour
    )

    base_demand = profile["base_demand"]

    electricity_demand = (
        base_demand
        * demand_multiplier
    )

    # --------------------------------------------------------
    # Temperature effect
    # --------------------------------------------------------

    temperature = generate_temperature(
        hour
    )

    # Higher temperature → higher electricity demand
    if temperature > 30:

        temperature_factor = (
            1
            + (temperature - 30)
            * 0.025
        )

        electricity_demand *= (
            temperature_factor
        )

    # --------------------------------------------------------
    # EV charging
    # --------------------------------------------------------

    ev_demand = generate_ev_demand(
        hour,
        profile["ev_capacity"]
    )

    # --------------------------------------------------------
    # Solar generation
    # --------------------------------------------------------

    solar_generation = (
        generate_solar_generation(
            hour,
            profile["solar_capacity"]
        )
    )

    # --------------------------------------------------------
    # Random natural variation
    # --------------------------------------------------------

    electricity_demand *= random.uniform(
        0.97,
        1.03
    )

    electricity_demand = round(
        max(
            0.1,
            electricity_demand
        ),
        2
    )

    # --------------------------------------------------------
    # Renewable percentage
    # --------------------------------------------------------

    renewable_percentage = (
        solar_generation
        / electricity_demand
    ) * 100

    renewable_percentage = round(
        min(
            100,
            max(
                0,
                renewable_percentage
            )
        ),
        2
    )

    # --------------------------------------------------------
    # Transformer load
    # --------------------------------------------------------

    result = engine.connect()

    try:

        query = text("""
            SELECT capacity_mw
            FROM energy_locations
            WHERE location_id = :location_id
        """)

        row = result.execute(
            query,
            {
                "location_id": location_id
            }
        ).fetchone()

    finally:

        result.close()

    if row:

        capacity_mw = float(
            row[0]
        )

    else:

        capacity_mw = (
            electricity_demand
            * 1.30
        )

    transformer_load_pct = (
        electricity_demand
        / capacity_mw
    ) * 100

    # Add small variation
    transformer_load_pct += random.uniform(
        -2,
        2
    )

    transformer_load_pct = round(
        max(
            0,
            transformer_load_pct
        ),
        2
    )

    # --------------------------------------------------------
    # Streetlights
    # --------------------------------------------------------

    streetlights_total = (
        profile["streetlights"]
    )

    streetlights_online = (
        generate_streetlight_status(
            hour,
            streetlights_total
        )
    )

    # --------------------------------------------------------
    # Power outage
    # --------------------------------------------------------

    power_outage = (
        generate_power_outage()
    )

    # --------------------------------------------------------
    # Return record
    # --------------------------------------------------------

    return {

        "timestamp": timestamp,

        "location_id": location_id,

        "electricity_demand_mw":
            electricity_demand,

        "solar_generation_mw":
            solar_generation,

        "ev_charging_demand_mw":
            ev_demand,

        "renewable_percentage":
            renewable_percentage,

        "transformer_load_pct":
            transformer_load_pct,

        "streetlights_online":
            streetlights_online,

        "streetlights_total":
            streetlights_total,

        "power_outage":
            power_outage
    }


# ============================================================
# INSERT INTO POSTGRESQL
# ============================================================

def insert_energy_record(record):

    query = text("""
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

        DO NOTHING;
    """)

    with engine.begin() as conn:

        conn.execute(
            query,
            record
        )


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_all_locations():

    timestamp = datetime.now().replace(
        second=0,
        microsecond=0
    )

    for location_id, profile in (
        LOCATION_PROFILES.items()
    ):

        record = generate_energy_record(
            location_id,
            profile,
            timestamp
        )

        insert_energy_record(
            record
        )

        print(
            f"✓ {location_id:<20}"
            f" Demand: "
            f"{record['electricity_demand_mw']:>6.2f} MW |"
            f" Solar: "
            f"{record['solar_generation_mw']:>5.2f} MW |"
            f" EV: "
            f"{record['ev_charging_demand_mw']:>5.2f} MW |"
            f" Transformer: "
            f"{record['transformer_load_pct']:>6.2f}%"
        )


def main():

    print("=" * 80)
    print("SMART CITY ENERGY MANAGEMENT")
    print("LIVE ENERGY DATA GENERATOR")
    print("=" * 80)

    print()
    print(
        f"Updating every "
        f"{SLEEP_SECONDS} seconds..."
    )
    print()

    while True:

        try:

            generate_all_locations()

            print()
            print(
                "✓ Energy data updated."
            )
            print(
                "Waiting for next update..."
            )
            print()

            time.sleep(
                SLEEP_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print(
                "Energy generator stopped."
            )

            break

        except Exception as e:

            print()
            print(
                "ERROR:",
                e
            )

            print(
                "Retrying in 10 seconds..."
            )

            time.sleep(10)


if __name__ == "__main__":

    main()