import math
import time
import requests

from sqlalchemy import text

from database.connection import engine


# ============================================================
# SMART CITY - ON-DEMAND PARKING API
# ============================================================
#
# FINAL USER FLOW
#
# 1. User selects a LOCATION
# 2. User selects CATEGORY
#       Hospital / ATM / School / Bank / Shopping Mall
# 3. User selects the REAL destination from city_pois
# 4. ONLY THEN we query OpenStreetMap for nearby parking
# 5. All genuine parking facilities within the search radius
#    are returned
# 6. Parking is saved in parking_locations
# 7. POI <-> parking distance is saved in poi_parking_map
#
# This avoids querying hundreds of POIs in advance.
#
# IMPORTANT:
# - No fake parking names
# - No fake coordinates
# - No fake capacity
# - No parking availability is invented here
# - Live/ML parking availability will be added later
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PARKING_RADIUS_KM = 1.5
MIN_PARKING_RADIUS_KM = 0.5
MAX_PARKING_RADIUS_KM = 2.5

MAX_API_RETRIES = 3
BACKOFF_SECONDS = 5

# Current public global-data Overpass instances.
# Private.coffee is listed by OpenStreetMap as a global instance
# and requests a User-Agent for application traffic.
OVERPASS_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

HEADERS = {
    "User-Agent": "SmartCityOperations/1.0",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}


# ============================================================
# USER-FACING CATEGORIES
# ============================================================

CATEGORY_TYPES = {
    "HOSPITAL": "HOSPITAL",
    "ATM": "ATM",
    "SCHOOL": "SCHOOL",
    "BANK": "BANK",
    "SHOPPING_MALL": "SHOPPING_MALL",
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """Return distance between two coordinates in kilometres."""

    earth_radius_km = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))

    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# CREATE TABLES
# ============================================================

def create_parking_tables():

    parking_table = text("""
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

    mapping_table = text("""
        CREATE TABLE IF NOT EXISTS poi_parking_map (

            id SERIAL PRIMARY KEY,

            poi_id INTEGER NOT NULL,

            parking_id INTEGER NOT NULL,

            distance_from_poi_km DOUBLE PRECISION NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT uq_poi_parking
                UNIQUE (poi_id, parking_id)
        )
    """)

    with engine.begin() as conn:
        conn.execute(parking_table)
        conn.execute(mapping_table)

    print("Parking tables ready.")


# ============================================================
# LOAD LOCATIONS
# ============================================================

def get_locations():

    query = text("""
        SELECT DISTINCT
            location_id
        FROM city_pois
        ORDER BY location_id
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return [
        row["location_id"]
        for row in rows
    ]


# ============================================================
# LOAD CATEGORIES FOR A LOCATION
# ============================================================

def get_categories_for_location(
    location_id
):

    query = text("""
        SELECT DISTINCT
            poi_type
        FROM city_pois
        WHERE location_id = :location_id
        ORDER BY poi_type
    """)

    with engine.connect() as conn:

        rows = conn.execute(
            query,
            {
                "location_id": location_id
            }
        ).mappings().all()

    return [
        row["poi_type"]
        for row in rows
    ]


# ============================================================
# LOAD REAL DESTINATIONS
# ============================================================

def get_destinations(
    location_id,
    poi_type
):

    if poi_type not in CATEGORY_TYPES:
        raise ValueError(
            f"Unsupported category: {poi_type}"
        )

    query = text("""
        SELECT

            id,

            location_id,

            poi_name,

            poi_type,

            latitude,

            longitude,

            address

        FROM city_pois

        WHERE location_id = :location_id
          AND poi_type = :poi_type

        ORDER BY poi_name
    """)

    with engine.connect() as conn:

        return conn.execute(
            query,
            {
                "location_id": location_id,
                "poi_type": poi_type
            }
        ).mappings().all()


# ============================================================
# LOAD ONE POI BY ID
# ============================================================

def get_poi_by_id(
    poi_id
):

    query = text("""
        SELECT

            id,

            location_id,

            poi_name,

            poi_type,

            latitude,

            longitude,

            address

        FROM city_pois

        WHERE id = :poi_id
    """)

    with engine.connect() as conn:

        row = conn.execute(
            query,
            {
                "poi_id": poi_id
            }
        ).mappings().first()

    return row


# ============================================================
# BUILD OVERPASS PARKING QUERY
# ============================================================

def build_overpass_query(
    latitude,
    longitude,
    radius_km
):

    radius_meters = int(
        radius_km * 1000
    )

    return f"""
    [out:json][timeout:60];

    nwr
        ["amenity"="parking"]
        (around:{radius_meters},{latitude},{longitude});

    out center tags;
    """


# ============================================================
# QUERY OVERPASS
# ============================================================

def query_overpass(
    latitude,
    longitude,
    radius_km
):
    """
    Query one selected destination only.

    This is intentionally small because parking is now
    fetched on demand after the user chooses a destination.
    """

    query = build_overpass_query(
        latitude,
        longitude,
        radius_km
    )

    last_error = None

    for url in OVERPASS_URLS:

        for attempt in range(
            1,
            MAX_API_RETRIES + 1
        ):

            try:

                response = requests.post(

                    url,

                    data={
                        "data": query
                    },

                    headers=HEADERS,

                    timeout=90
                )

                response.raise_for_status()

                data = response.json()

                return data.get(
                    "elements",
                    []
                )

            except (
                requests.exceptions.RequestException,
                ValueError
            ) as exc:

                last_error = exc

                hostname = (
                    url.split("//")[1]
                    .split("/")[0]
                )

                print(
                    f"Overpass failed "
                    f"({hostname}) "
                    f"attempt "
                    f"{attempt}/{MAX_API_RETRIES}: "
                    f"{exc}"
                )

                if attempt < MAX_API_RETRIES:

                    wait_time = (
                        BACKOFF_SECONDS * attempt
                    )

                    print(
                        f"Retrying in "
                        f"{wait_time}s..."
                    )

                    time.sleep(
                        wait_time
                    )

        print(
            f"Endpoint {url} failed. "
            f"Trying next endpoint..."
        )

    raise RuntimeError(
        "All Overpass endpoints failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# OSM COORDINATES
# ============================================================

def get_coordinates(
    element
):

    if (
        "lat" in element
        and
        "lon" in element
    ):

        return (
            float(element["lat"]),
            float(element["lon"])
        )

    if "center" in element:

        return (
            float(
                element["center"]["lat"]
            ),
            float(
                element["center"]["lon"]
            )
        )

    return None, None


# ============================================================
# ADDRESS
# ============================================================

def build_address(
    tags
):

    parts = []

    for key in (
        "addr:housenumber",
        "addr:street",
        "addr:suburb",
        "addr:city",
        "addr:postcode"
    ):

        value = tags.get(key)

        if value:
            parts.append(
                str(value)
            )

    return ", ".join(parts)


# ============================================================
# CAPACITY
# ============================================================

def parse_capacity(
    tags
):

    value = tags.get(
        "capacity"
    )

    if value is None:
        return None

    try:
        return int(value)

    except (
        ValueError,
        TypeError
    ):
        return None


# ============================================================
# SAVE PARKING
# ============================================================

def save_parking(
    parking
):

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

            location_id =
                EXCLUDED.location_id,

            parking_name =
                EXCLUDED.parking_name,

            parking_type =
                EXCLUDED.parking_type,

            latitude =
                EXCLUDED.latitude,

            longitude =
                EXCLUDED.longitude,

            address =
                EXCLUDED.address,

            capacity =
                EXCLUDED.capacity,

            source =
                EXCLUDED.source

        RETURNING id
    """)

    with engine.begin() as conn:

        result = conn.execute(
            query,
            parking
        )

        return result.scalar_one()


# ============================================================
# SAVE POI-PARKING LINK
# ============================================================

def save_mapping(
    poi_id,
    parking_id,
    distance_km
):

    query = text("""
        INSERT INTO poi_parking_map
        (
            poi_id,
            parking_id,
            distance_from_poi_km
        )

        VALUES
        (
            :poi_id,
            :parking_id,
            :distance_km
        )

        ON CONFLICT (poi_id, parking_id)

        DO UPDATE SET

            distance_from_poi_km =
                EXCLUDED.distance_from_poi_km
    """)

    with engine.begin() as conn:

        conn.execute(
            query,
            {
                "poi_id": poi_id,
                "parking_id": parking_id,
                "distance_km": round(
                    distance_km,
                    3
                )
            }
        )


# ============================================================
# SEARCH NEARBY PARKING FOR A SELECTED POI
# ============================================================

def find_nearby_parking(
    poi_id,
    radius_km=DEFAULT_PARKING_RADIUS_KM
):
    """
    Main parking search function.

    Search policy:
        1. Search within the requested radius (default 1.5 km).
        2. If the default 1.5 km search returns no parking,
           automatically expand the search to 2.5 km.
        3. If parking is still not found, return an empty list.

    For an explicitly supplied radius other than 1.5 km, only that
    radius is used.
    """

    if not (
        MIN_PARKING_RADIUS_KM
        <= radius_km
        <= MAX_PARKING_RADIUS_KM
    ):
        raise ValueError(
            f"radius_km must be between "
            f"{MIN_PARKING_RADIUS_KM} and "
            f"{MAX_PARKING_RADIUS_KM}"
        )

    poi = get_poi_by_id(poi_id)

    if poi is None:
        raise ValueError(
            f"POI {poi_id} not found."
        )

    print()
    print("=" * 80)
    print(f"DESTINATION : {poi['poi_name']}")
    print(f"TYPE        : {poi['poi_type']}")
    print(f"ZONE        : {poi['location_id']}")
    print(
        f"COORDINATES : "
        f"{poi['latitude']}, {poi['longitude']}"
    )
    print("=" * 80)

    def search_at_radius(search_radius_km):
        """Run one OSM parking search and return sorted results."""

        print(
            f"PARKING SEARCH RADIUS : "
            f"{search_radius_km:.1f} km"
        )
        print("-" * 80)

        elements = query_overpass(
            float(poi["latitude"]),
            float(poi["longitude"]),
            search_radius_km
        )

        print(
            f"OSM returned {len(elements)} parking objects."
        )

        unique = {}

        for element in elements:

            tags = element.get("tags", {})

            parking_lat, parking_lon = get_coordinates(
                element
            )

            if parking_lat is None or parking_lon is None:
                continue

            osm_id = (
                f"{element.get('type')}_"
                f"{element.get('id')}"
            )

            if osm_id in unique:
                continue

            distance_km = haversine_distance(
                float(poi["latitude"]),
                float(poi["longitude"]),
                parking_lat,
                parking_lon
            )

            # Final exact distance check.
            if distance_km > search_radius_km:
                continue

            parking = {
                "osm_id": osm_id,
                "location_id": poi["location_id"],
                "parking_name": tags.get(
                    "name",
                    "Unnamed OSM parking facility"
                ),
                "parking_type": tags.get(
                    "parking",
                    "GENERAL"
                ),
                "latitude": parking_lat,
                "longitude": parking_lon,
                "address": build_address(tags),
                "capacity": parse_capacity(tags),
                "source": "OpenStreetMap"
            }

            parking_id = save_parking(parking)

            save_mapping(
                poi_id,
                parking_id,
                distance_km
            )

            parking["id"] = parking_id
            parking["poi_id"] = poi_id
            parking["distance_km"] = round(
                distance_km,
                3
            )

            unique[osm_id] = parking

        results = list(unique.values())

        results.sort(
            key=lambda x: x["distance_km"]
        )

        return results

    # --------------------------------------------------------
    # FIRST SEARCH: 1.5 KM BY DEFAULT
    # --------------------------------------------------------
    results = search_at_radius(radius_km)

    # --------------------------------------------------------
    # FALLBACK: 2.5 KM
    # --------------------------------------------------------
    # Only the normal 1.5 km search expands automatically.
    if (
        not results
        and abs(radius_km - DEFAULT_PARKING_RADIUS_KM) < 1e-9
    ):

        print()
        print(
            "No mapped parking found within 1.5 km."
        )
        print(
            "Expanding search to 2.5 km..."
        )

        results = search_at_radius(2.5)

        if results:
            print()
            print(
                f"Found {len(results)} nearby parking "
                "facilities within 2.5 km."
            )
        else:
            print()
            print(
                "No mapped parking found within 2.5 km."
            )

    # --------------------------------------------------------
    # FINAL DISPLAY
    # --------------------------------------------------------
    print()

    if not results:
        print(
            "No real parking facilities found "
            "within the final search radius."
        )
        return []

    print(
        f"Found {len(results)} nearby parking facilities."
    )

    print()

    for index, parking in enumerate(
        results,
        start=1
    ):

        capacity = (
            parking["capacity"]
            if parking["capacity"] is not None
            else "Not mapped"
        )

        print(
            f"{index}. {parking['parking_name']}"
        )

        print(
            f"   Distance : "
            f"{parking['distance_km']:.2f} km"
        )

        print(
            f"   Type     : "
            f"{parking['parking_type']}"
        )

        print(
            f"   Capacity : {capacity}"
        )

        print()

    return results


# ============================================================
# SHOW DESTINATIONS
# ============================================================

def show_destinations(
    location_id,
    poi_type
):

    destinations = get_destinations(
        location_id,
        poi_type
    )

    print()
    print("=" * 80)

    print(
        f"{poi_type} IN {location_id}"
    )

    print("=" * 80)

    if not destinations:

        print(
            "No destinations found."
        )

        return []

    for index, destination in enumerate(
        destinations,
        start=1
    ):

        address = (
            destination["address"]
            or
            "Address not available"
        )

        print(
            f"{index}. "
            f"{destination['poi_name']}"
        )

        print(
            f"   Coordinates: "
            f"{destination['latitude']}, "
            f"{destination['longitude']}"
        )

        print(
            f"   Address    : "
            f"{address}"
        )

    return destinations


# ============================================================
# INTERACTIVE DESTINATION SELECTION
# ============================================================

def interactive_parking_search():

    locations = get_locations()

    if not locations:

        print(
            "No locations found in city_pois."
        )

        return

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "SMART CITY PARKING ASSISTANT"
    )
    print("=" * 80)

    print()
    print(
        "Where do you want to go?"
    )

    for index, location in enumerate(
        locations,
        start=1
    ):

        print(
            f"{index}. {location}"
        )

    while True:

        try:

            location_choice = int(
                input(
                    "\nEnter location number: "
                ).strip()
            )

            if (
                1
                <= location_choice
                <= len(locations)
            ):
                break

        except ValueError:
            pass

        print(
            "Invalid choice. Try again."
        )

    location_id = locations[
        location_choice - 1
    ]

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    available_categories = (
        get_categories_for_location(
            location_id
        )
    )

    available_categories = [
        category
        for category in available_categories
        if category in CATEGORY_TYPES
    ]

    if not available_categories:

        print(
            "No supported destination "
            "categories found."
        )

        return

    print()
    print(
        "What are you visiting?"
    )

    for index, category in enumerate(
        available_categories,
        start=1
    ):

        print(
            f"{index}. {category}"
        )

    while True:

        try:

            category_choice = int(
                input(
                    "\nEnter category number: "
                ).strip()
            )

            if (
                1
                <= category_choice
                <= len(available_categories)
            ):
                break

        except ValueError:
            pass

        print(
            "Invalid choice. Try again."
        )

    poi_type = available_categories[
        category_choice - 1
    ]

    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    destinations = show_destinations(
        location_id,
        poi_type
    )

    if not destinations:
        return

    while True:

        try:

            destination_choice = int(
                input(
                    "\nEnter destination number: "
                ).strip()
            )

            if (
                1
                <= destination_choice
                <= len(destinations)
            ):
                break

        except ValueError:
            pass

        print(
            "Invalid choice. Try again."
        )

    selected_poi = destinations[
        destination_choice - 1
    ]

    # --------------------------------------------------------
    # SEARCH RADIUS
    # --------------------------------------------------------

    print()
    print(
        f"Selected destination: "
        f"{selected_poi['poi_name']}"
    )

    print(
        f"Default parking search radius: "
        f"{DEFAULT_PARKING_RADIUS_KM:.1f} km"
    )

    radius_input = input(
        "Press Enter to use default, "
        "or enter radius (0.5-2.0 km): "
    ).strip()

    if radius_input:

        try:

            radius_km = float(
                radius_input
            )

        except ValueError:

            print(
                "Invalid radius. "
                "Using default."
            )

            radius_km = (
                DEFAULT_PARKING_RADIUS_KM
            )

    else:

        radius_km = (
            DEFAULT_PARKING_RADIUS_KM
        )

    # --------------------------------------------------------
    # FINAL PARKING SEARCH
    # --------------------------------------------------------

    results = find_nearby_parking(
        selected_poi["id"],
        radius_km
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        create_parking_tables()

        interactive_parking_search()

    except KeyboardInterrupt:

        print()
        print(
            "Parking search cancelled."
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "PARKING API ERROR"
        )
        print("=" * 80)
        print(exc)