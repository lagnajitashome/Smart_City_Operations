from sqlalchemy import text

from database.connection import engine


CREATE_LOCATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS energy_locations (

    location_id VARCHAR(50) PRIMARY KEY,

    location_name VARCHAR(100) NOT NULL,

    latitude DOUBLE PRECISION,

    longitude DOUBLE PRECISION,

    location_type VARCHAR(50),

    capacity_mw DOUBLE PRECISION NOT NULL
);
"""


CREATE_ENERGY_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS energy_data (

    timestamp TIMESTAMP NOT NULL,

    location_id VARCHAR(50) NOT NULL,

    electricity_demand_mw DOUBLE PRECISION,

    solar_generation_mw DOUBLE PRECISION,

    ev_charging_demand_mw DOUBLE PRECISION,

    renewable_percentage DOUBLE PRECISION,

    transformer_load_pct DOUBLE PRECISION,

    streetlights_online INTEGER,

    streetlights_total INTEGER,

    power_outage BOOLEAN,

    PRIMARY KEY (timestamp, location_id),

    FOREIGN KEY (location_id)
        REFERENCES energy_locations(location_id)
);
"""


LOCATIONS = [

    {
        "location_id": "PARK_STREET",
        "location_name": "Park Street",
        "latitude": 22.5535,
        "longitude": 88.3520,
        "location_type": "Commercial",
        "capacity_mw": 25.0
    },

    {
        "location_id": "ESPLANADE",
        "location_name": "Esplanade",
        "latitude": 22.5650,
        "longitude": 88.3510,
        "location_type": "Commercial",
        "capacity_mw": 30.0
    },

    {
        "location_id": "EM_BYPASS",
        "location_name": "EM Bypass",
        "latitude": 22.5120,
        "longitude": 88.4000,
        "location_type": "Mixed",
        "capacity_mw": 35.0
    },

    {
        "location_id": "SALT_LAKE",
        "location_name": "Salt Lake",
        "latitude": 22.5800,
        "longitude": 88.4170,
        "location_type": "Residential",
        "capacity_mw": 30.0
    },

    {
        "location_id": "NEW_TOWN",
        "location_name": "New Town",
        "latitude": 22.5958,
        "longitude": 88.4797,
        "location_type": "Mixed",
        "capacity_mw": 40.0
    },

    {
        "location_id": "VIP_ROAD",
        "location_name": "VIP Road",
        "latitude": 22.6100,
        "longitude": 88.4200,
        "location_type": "Residential",
        "capacity_mw": 25.0
    },

    {
        "location_id": "RASHBEHARI",
        "location_name": "Rashbehari",
        "latitude": 22.5186,
        "longitude": 88.3650,
        "location_type": "Commercial",
        "capacity_mw": 25.0
    },

    {
        "location_id": "HOWRAH_APPROACH",
        "location_name": "Howrah Approach",
        "latitude": 22.5840,
        "longitude": 88.3370,
        "location_type": "Mixed",
        "capacity_mw": 30.0
    }
]


def create_tables():

    with engine.begin() as conn:

        conn.execute(
            text(CREATE_LOCATIONS_TABLE)
        )

        conn.execute(
            text(CREATE_ENERGY_DATA_TABLE)
        )

        for location in LOCATIONS:

            conn.execute(
                text("""
                    INSERT INTO energy_locations (
                        location_id,
                        location_name,
                        latitude,
                        longitude,
                        location_type,
                        capacity_mw
                    )

                    VALUES (
                        :location_id,
                        :location_name,
                        :latitude,
                        :longitude,
                        :location_type,
                        :capacity_mw
                    )

                    ON CONFLICT (location_id)
                    DO UPDATE SET

                        location_name = EXCLUDED.location_name,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        location_type = EXCLUDED.location_type,
                        capacity_mw = EXCLUDED.capacity_mw;
                """),
                location
            )


if __name__ == "__main__":

    print("=" * 60)
    print("ENERGY MANAGEMENT - DATABASE SETUP")
    print("=" * 60)

    create_tables()

    print()
    print("✓ energy_locations table created")
    print("✓ energy_data table created")
    print("✓ Energy locations inserted")
    print()
    print("Energy database setup complete.")