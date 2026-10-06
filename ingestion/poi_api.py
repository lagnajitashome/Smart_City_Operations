import requests
import time
import math

from sqlalchemy import text
from database.connection import engine


# ============================================================
# SMART CITY - REAL POI DATA INGESTION
# ============================================================
# Uses the same 8 geographic anchor locations already used by
# the traffic/weather system.
#
# Finds REAL:
#   - Hospitals
#   - ATMs
#   - Schools
#   - Banks
#   - Shopping malls
#
# Data source: OpenStreetMap via Overpass API
#
# No POI names or coordinates are invented.
# ============================================================


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


# ============================================================
# SEARCH RADII
# ============================================================
# Most zones use 2 km.
# EM_BYPASS gets 4 km because the initial 2 km search returned
# no POIs even though it is an important corridor.
# ============================================================

DEFAULT_SEARCH_RADIUS_METERS = 2000

LOCATION_SEARCH_RADIUS = {
    "PARK_STREET": 2000,
    "ESPLANADE": 2000,
    "EM_BYPASS": 4000,
    "SALT_LAKE": 2000,
    "NEW_TOWN": 2000,
    "VIP_ROAD": 2000,
    "RASHBEHARI": 2000,
    "HOWRAH_APPROACH": 2000
}


# ============================================================
# OVERPASS CONFIGURATION
# ============================================================

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter"
]

HEADERS = {
    "User-Agent": "SmartCityOperations/1.0",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded"
}


# ============================================================
# POI CATEGORIES
# ============================================================

CATEGORY_QUERIES = {
    "HOSPITAL": 'nwr["amenity"="hospital"]',
    "ATM": 'nwr["amenity"="atm"]',
    "SCHOOL": 'nwr["amenity"="school"]',
    "BANK": 'nwr["amenity"="bank"]',
    "SHOPPING_MALL": 'nwr["shop"="mall"]'
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometres between two coordinates."""

    radius = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


# ============================================================
# CREATE TABLE
# ============================================================

def create_poi_table():
    query = text("""
        CREATE TABLE IF NOT EXISTS city_pois (

            id SERIAL PRIMARY KEY,

            osm_id VARCHAR(100) UNIQUE,

            location_id VARCHAR(100),

            poi_name VARCHAR(255),

            poi_type VARCHAR(50),

            latitude DOUBLE PRECISION,

            longitude DOUBLE PRECISION,

            distance_from_anchor_km DOUBLE PRECISION,

            address TEXT,

            source VARCHAR(50),

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    with engine.begin() as conn:
        conn.execute(query)


# ============================================================
# ENSURE DISTANCE COLUMN EXISTS
# ============================================================

def ensure_distance_column():
    query = text("""
        ALTER TABLE city_pois
        ADD COLUMN IF NOT EXISTS
        distance_from_anchor_km DOUBLE PRECISION
    """)

    with engine.begin() as conn:
        conn.execute(query)


# ============================================================
# BUILD OVERPASS QUERY
# ============================================================

def build_overpass_query(latitude, longitude, search_radius):
    category_blocks = []

    for category_query in CATEGORY_QUERIES.values():
        category_blocks.append(
            f"""
            {category_query}
            (around:{search_radius},{latitude},{longitude});
            """
        )

    joined_blocks = "\n".join(category_blocks)

    return f"""
    [out:json][timeout:90];

    (
        {joined_blocks}
    );

    out center tags;
    """


# ============================================================
# GET REAL POIs FROM OPENSTREETMAP
# ============================================================

def get_osm_pois(latitude, longitude, search_radius):
    overpass_query = build_overpass_query(
        latitude,
        longitude,
        search_radius
    )

    last_error = None

    for overpass_url in OVERPASS_URLS:
        try:
            print()
            print("Querying Overpass:")
            print(overpass_url)

            response = requests.post(
                overpass_url,
                data={"data": overpass_query},
                headers=HEADERS,
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            return data.get("elements", [])

        except (
            requests.exceptions.RequestException,
            ValueError
        ) as exc:
            last_error = exc
            print(f"Overpass request failed: {exc}")
            print("Trying the next Overpass instance...")

    raise RuntimeError(
        "All Overpass endpoints failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# DETERMINE POI TYPE
# ============================================================

def get_poi_type(tags):
    if tags.get("amenity") == "hospital":
        return "HOSPITAL"

    if tags.get("amenity") == "atm":
        return "ATM"

    if tags.get("amenity") == "school":
        return "SCHOOL"

    if tags.get("amenity") == "bank":
        return "BANK"

    if tags.get("shop") == "mall":
        return "SHOPPING_MALL"

    return None


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def get_coordinates(element):
    # Node
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])

    # Way / Relation returned with "out center"
    if "center" in element:
        return (
            float(element["center"]["lat"]),
            float(element["center"]["lon"])
        )

    return None, None


# ============================================================
# BUILD ADDRESS
# ============================================================

def build_address(tags):
    address_parts = []

    for key in (
        "addr:housenumber",
        "addr:street",
        "addr:suburb",
        "addr:city",
        "addr:postcode"
    ):
        value = tags.get(key)

        if value:
            address_parts.append(str(value))

    return ", ".join(address_parts)


# ============================================================
# FIND NEAREST EXISTING ANCHOR
# ============================================================

def find_nearest_anchor(poi_lat, poi_lon):
    """
    Assign each genuine POI to the geographically nearest
    existing traffic/weather anchor.
    """

    nearest_location = None
    nearest_distance = float("inf")

    for location_id, coordinates in LOCATIONS.items():
        distance = haversine_distance(
            poi_lat,
            poi_lon,
            coordinates["latitude"],
            coordinates["longitude"]
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_location = location_id

    return nearest_location, nearest_distance


# ============================================================
# SAVE POI
# ============================================================

def save_poi(poi):
    query = text("""
        INSERT INTO city_pois
        (
            osm_id,
            location_id,
            poi_name,
            poi_type,
            latitude,
            longitude,
            distance_from_anchor_km,
            address,
            source
        )
        VALUES
        (
            :osm_id,
            :location_id,
            :poi_name,
            :poi_type,
            :latitude,
            :longitude,
            :distance_from_anchor_km,
            :address,
            :source
        )
        ON CONFLICT (osm_id)
        DO UPDATE SET

            location_id =
                EXCLUDED.location_id,

            poi_name =
                EXCLUDED.poi_name,

            poi_type =
                EXCLUDED.poi_type,

            latitude =
                EXCLUDED.latitude,

            longitude =
                EXCLUDED.longitude,

            distance_from_anchor_km =
                EXCLUDED.distance_from_anchor_km,

            address =
                EXCLUDED.address,

            source =
                EXCLUDED.source
    """)

    with engine.begin() as conn:
        conn.execute(query, poi)


# ============================================================
# PROCESS ONE ANCHOR
# ============================================================

def collect_pois_for_location(location_id, latitude, longitude):
    search_radius = LOCATION_SEARCH_RADIUS.get(
        location_id,
        DEFAULT_SEARCH_RADIUS_METERS
    )

    print()
    print("=" * 75)
    print(f"SEARCHING REAL POIs NEAR {location_id}")
    print(f"Anchor       : {latitude}, {longitude}")
    print(f"Search radius: {search_radius / 1000:.1f} km")
    print("=" * 75)

    elements = get_osm_pois(
        latitude,
        longitude,
        search_radius
    )

    print(
        f"OpenStreetMap returned {len(elements)} objects."
    )

    saved = 0

    category_counts = {
        "HOSPITAL": 0,
        "ATM": 0,
        "SCHOOL": 0,
        "BANK": 0,
        "SHOPPING_MALL": 0
    }

    for element in elements:
        tags = element.get("tags", {})

        poi_type = get_poi_type(tags)

        if poi_type is None:
            continue

        poi_lat, poi_lon = get_coordinates(element)

        if poi_lat is None or poi_lon is None:
            continue

        osm_id = (
            f"{element.get('type')}_{element.get('id')}"
        )

        poi_name = tags.get("name")

        # We never invent the real place name.
        if not poi_name:
            poi_name = "Unnamed OSM facility"

        address = build_address(tags)

        nearest_anchor, anchor_distance = find_nearest_anchor(
            poi_lat,
            poi_lon
        )

        poi_data = {
            "osm_id": osm_id,
            "location_id": nearest_anchor,
            "poi_name": poi_name,
            "poi_type": poi_type,
            "latitude": poi_lat,
            "longitude": poi_lon,
            "distance_from_anchor_km": round(
                anchor_distance,
                3
            ),
            "address": address,
            "source": "OpenStreetMap"
        }

        save_poi(poi_data)

        saved += 1
        category_counts[poi_type] += 1

    print(f"Saved/updated {saved} genuine POIs.")
    print("Category breakdown:")

    for category, count in category_counts.items():
        print(f"  {category:<18}: {count}")


# ============================================================
# COLLECT ALL LOCATIONS
# ============================================================

def collect_all_pois():
    create_poi_table()
    ensure_distance_column()

    print()
    print("=" * 75)
    print("SMART CITY - REAL POI COLLECTION")
    print("=" * 75)

    for location_id, coordinates in LOCATIONS.items():
        try:
            collect_pois_for_location(
                location_id,
                coordinates["latitude"],
                coordinates["longitude"]
            )

        except Exception as exc:
            print()
            print(
                f"ERROR collecting POIs for {location_id}"
            )
            print(exc)

        # Respect public Overpass services.
        time.sleep(3)

    print()
    print("=" * 75)
    print("POI COLLECTION COMPLETED")
    print("=" * 75)


# ============================================================
# SUMMARY BY LOCATION
# ============================================================

def show_poi_summary():
    query = text("""
        SELECT
            location_id,
            poi_type,
            COUNT(*) AS total
        FROM city_pois
        GROUP BY
            location_id,
            poi_type
        ORDER BY
            location_id,
            poi_type
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.mappings().all()

    print()
    print("=" * 80)
    print("POI SUMMARY BY LOCATION")
    print("=" * 80)

    current_location = None

    for row in rows:
        location = row["location_id"]

        if location != current_location:
            print()
            print(location)
            print("-" * 55)
            current_location = location

        print(
            f"{row['poi_type']:<18}: "
            f"{row['total']}"
        )

    print()
    print("=" * 80)


# ============================================================
# DETAILED POI LIST
# ============================================================

def show_poi_details():
    query = text("""
        SELECT
            location_id,
            poi_type,
            poi_name,
            latitude,
            longitude,
            distance_from_anchor_km
        FROM city_pois
        ORDER BY
            location_id,
            poi_type,
            poi_name
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.mappings().all()

    print()
    print("=" * 120)
    print("REAL POI DETAILS")
    print("=" * 120)

    if not rows:
        print("No POIs found.")
        return

    for row in rows:
        name = str(row["poi_name"])
        if len(name) > 45:
            name = name[:42] + "..."

        print(
            f"{row['location_id']:<18} | "
            f"{row['poi_type']:<15} | "
            f"{name:<45} | "
            f"{row['latitude']:.6f}, "
            f"{row['longitude']:.6f} | "
            f"{row['distance_from_anchor_km']:.2f} km"
        )

    print()
    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        collect_all_pois()
        show_poi_summary()

        print()
        input(
            "Press Enter to show detailed POIs..."
        )

        show_poi_details()

    except Exception as exc:
        print()
        print("=" * 75)
        print("POI INGESTION ERROR")
        print("=" * 75)
        print(exc)