import requests
import time
from sqlalchemy import text
from database.connection import engine


# ============================================================
# SMART CITY - PARKING DATA INGESTION
# ============================================================

# These are the SAME geographic zones already used
# by your traffic/weather system.
LOCATIONS = {
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


# Search radius around each existing location
SEARCH_RADIUS_METERS = 3000


# ============================================================
# CREATE TABLE
# ============================================================

def create_parking_table():

    query = text("""
        CREATE TABLE IF NOT EXISTS parking_locations (

            id SERIAL PRIMARY KEY,

            osm_id VARCHAR(100) UNIQUE,

            location_id VARCHAR(100),

            parking_name VARCHAR(255),

            parking_type VARCHAR(100),

            latitude DOUBLE PRECISION,

            longitude DOUBLE PRECISION,

            address TEXT,

            capacity INTEGER,

            source VARCHAR(50),

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    with engine.begin() as conn:
        conn.execute(query)

    print("Parking table ready.")


# ============================================================
# GET REAL PARKING DATA FROM OPENSTREETMAP
# ============================================================

def get_osm_parking(latitude, longitude):

    overpass_query = f"""
    [out:json][timeout:60];

    (
        nwr["amenity"="parking"]
            (around:{SEARCH_RADIUS_METERS},{latitude},{longitude});

        nwr["amenity"="parking_space"]
            (around:{SEARCH_RADIUS_METERS},{latitude},{longitude});
    );

    out center tags;
    """

    url = "https://overpass-api.de/api/interpreter"

    response = requests.post(
        url,
        data=overpass_query,
        timeout=90
    )

    response.raise_for_status()

    return response.json().get("elements", [])


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def get_coordinates(element):

    # Node
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    # Way / Relation
    if "center" in element:
        return (
            element["center"]["lat"],
            element["center"]["lon"]
        )

    return None, None


# ============================================================
# SAVE PARKING LOCATION
# ============================================================

def save_parking(parking):

    query = text("""
        INSERT INTO parking_locations
        (
            osm_id,
            location_id,
            parking_name,
            parking_type,
            latitude,
            longitude,
            address,
            capacity,
            source
        )
        VALUES
        (
            :osm_id,
            :location_id,
            :parking_name,
            :parking_type,
            :latitude,
            :longitude,
            :address,
            :capacity,
            :source
        )

        ON CONFLICT (osm_id)
        DO UPDATE SET

            location_id = EXCLUDED.location_id,
            parking_name = EXCLUDED.parking_name,
            parking_type = EXCLUDED.parking_type,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            address = EXCLUDED.address,
            capacity = EXCLUDED.capacity
    """)

    with engine.begin() as conn:

        conn.execute(
            query,
            parking
        )


# ============================================================
# PROCESS ONE LOCATION
# ============================================================

def collect_parking_for_location(
    location_id,
    latitude,
    longitude
):

    print()
    print("=" * 70)
    print(f"SEARCHING PARKING NEAR {location_id}")
    print(f"Anchor: {latitude}, {longitude}")
    print("=" * 70)

    elements = get_osm_parking(
        latitude,
        longitude
    )

    print(
        f"OpenStreetMap returned "
        f"{len(elements)} parking objects."
    )

    saved = 0

    for element in elements:

        tags = element.get("tags", {})

        lat, lon = get_coordinates(element)

        if lat is None or lon is None:
            continue

        osm_id = (
            f"{element.get('type')}_"
            f"{element.get('id')}"
        )

        name = tags.get("name")

        # Keep unnamed parking as a legitimate OSM facility.
        # We do NOT invent a fake name.
        if not name:
            name = "Unnamed parking facility"

        parking_type = tags.get(
            "parking",
            "GENERAL"
        )

        address_parts = []

        for key in [
            "addr:housenumber",
            "addr:street",
            "addr:suburb",
            "addr:city"
        ]:

            if key in tags:
                address_parts.append(
                    tags[key]
                )

        address = ", ".join(address_parts)

        capacity = None

        if "capacity" in tags:

            try:
                capacity = int(
                    tags["capacity"]
                )

            except (ValueError, TypeError):
                capacity = None

        parking_data = {

            "osm_id": osm_id,

            "location_id": location_id,

            "parking_name": name,

            "parking_type": parking_type,

            "latitude": float(lat),

            "longitude": float(lon),

            "address": address,

            "capacity": capacity,

            "source": "OpenStreetMap"
        }

        save_parking(
            parking_data
        )

        saved += 1

    print(
        f"Saved/updated {saved} parking facilities "
        f"for {location_id}."
    )


# ============================================================
# COLLECT ALL LOCATIONS
# ============================================================

def collect_all_parking():

    create_parking_table()

    print()
    print("=" * 70)
    print("SMART CITY - REAL PARKING DATA COLLECTION")
    print("=" * 70)

    for location_id, coordinates in LOCATIONS.items():

        try:

            collect_parking_for_location(
                location_id,
                coordinates["latitude"],
                coordinates["longitude"]
            )

            # Respect Overpass server
            time.sleep(2)

        except Exception as e:

            print()
            print(
                f"ERROR collecting parking "
                f"for {location_id}"
            )

            print(e)

    print()
    print("=" * 70)
    print("PARKING DATA COLLECTION COMPLETED")
    print("=" * 70)


# ============================================================
# SHOW COLLECTED DATA
# ============================================================

def show_parking_data():

    query = text("""
        SELECT
            id,
            location_id,
            parking_name,
            parking_type,
            latitude,
            longitude,
            capacity,
            source
        FROM parking_locations
        ORDER BY location_id, parking_name
    """)

    with engine.connect() as conn:

        result = conn.execute(query)

        rows = result.mappings().all()

    print()
    print("=" * 100)
    print("COLLECTED REAL PARKING FACILITIES")
    print("=" * 100)

    if not rows:

        print("No parking facilities found.")

        return

    for row in rows:

        print()
        print(
            f"Zone       : {row['location_id']}"
        )

        print(
            f"Parking    : {row['parking_name']}"
        )

        print(
            f"Type       : {row['parking_type']}"
        )

        print(
            f"Coordinates: "
            f"{row['latitude']}, "
            f"{row['longitude']}"
        )

        print(
            f"Capacity   : "
            f"{row['capacity']}"
        )

        print(
            f"Source     : {row['source']}"
        )

        print("-" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    collect_all_parking()

    show_parking_data()