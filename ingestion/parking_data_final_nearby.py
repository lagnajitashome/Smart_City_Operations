import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import pyogrio
from sqlalchemy import text

from database.connection import engine


# ============================================================
# SMART CITY - FINAL PARKING DATA PIPELINE
# ============================================================
#
# Geofabrik OpenStreetMap GeoPackage
#              ↓
#       Real parking facilities
#              ↓
#       parking_locations
#              ↓
#          city_pois
#              ↓
#    ALL nearby parking <= 4 km
#              ↓
#       poi_parking_map
#
# NO OVERPASS
# NO FAKE PARKING
# NO FAKE CAPACITY
#
# For every POI:
#   - keep ALL real parking within 4 km
#   - 1.5 km / 2.5 km / 4 km are used only as distance bands
#
# Later Streamlit will rank ALL candidates using:
#   availability
#   predicted availability
#   traffic / ETA
#   walking distance
#   destination proximity
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_RAW = Path("data/raw")

EXPECTED_GPKG_NAME = "eastern-zone-latest-free.gpkg"

SOURCE = "OpenStreetMap-Geofabrik"

FIRST_RADIUS_KM = 1.5
SECOND_RADIUS_KM = 2.5
MAX_RADIUS_KM = 4.0

# Only used for verification/reporting.
# It does NOT limit parking records.
MIN_EXPECTED_OPTIONS = 5


# ============================================================
# EIGHT PROJECT LOCATIONS
# ============================================================

LOCATIONS = {
    "PARK_STREET": {
        "latitude": 22.5535,
        "longitude": 88.3520,
    },

    "ESPLANADE": {
        "latitude": 22.5660,
        "longitude": 88.3510,
    },

    "EM_BYPASS": {
        "latitude": 22.5140,
        "longitude": 88.3970,
    },

    "SALT_LAKE": {
        "latitude": 22.5800,
        "longitude": 88.4170,
    },

    "NEW_TOWN": {
        "latitude": 22.5958,
        "longitude": 88.4497,
    },

    "VIP_ROAD": {
        "latitude": 22.6250,
        "longitude": 88.4050,
    },

    "RASHBEHARI": {
        "latitude": 22.5190,
        "longitude": 88.3520,
    },

    "HOWRAH_APPROACH": {
        "latitude": 22.5850,
        "longitude": 88.3300,
    },
}


# ============================================================
# PARKING CLASSES IN GEOFABRIK
# ============================================================

PARKING_CLASSES = {
    "parking",
    "parking_surface",
    "parking_multistorey",
    "parking_underground",
    "parking_street",
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Return distance between two coordinates in kilometres."""

    radius = 6371.0

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
        math.sqrt(1 - a),
    )

    return radius * c


# ============================================================
# FIND ACTUAL GPKG FILE
# ============================================================

def find_gpkg():

    expected = DATA_RAW / EXPECTED_GPKG_NAME

    if expected.exists() and expected.is_file():

        print(
            f"Using GeoPackage: {expected}"
        )

        return expected

    if expected.exists() and expected.is_dir():

        print(
            f"WARNING: {expected} is a folder, "
            "not a GeoPackage file."
        )

    candidates = [
        path
        for path in DATA_RAW.glob("*.gpkg")
        if path.is_file()
    ]

    if not candidates:

        raise FileNotFoundError(
            "\nNo .gpkg file found in:\n"
            f"{DATA_RAW.resolve()}"
        )

    # Prefer the largest actual GeoPackage.
    candidates.sort(
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    selected = candidates[0]

    print(
        f"Using GeoPackage: {selected}"
    )

    return selected


# ============================================================
# PROJECT BOUNDING BOX
# ============================================================

def get_project_bbox():
    """
    Create a bounding box around all eight project locations,
    expanded by 4 km.
    """

    latitudes = [
        value["latitude"]
        for value in LOCATIONS.values()
    ]

    longitudes = [
        value["longitude"]
        for value in LOCATIONS.values()
    ]

    min_lat = min(latitudes)
    max_lat = max(latitudes)

    min_lon = min(longitudes)
    max_lon = max(longitudes)

    middle_lat = (
        min_lat + max_lat
    ) / 2

    lat_buffer = (
        MAX_RADIUS_KM / 111.0
    )

    lon_buffer = (
        MAX_RADIUS_KM
        /
        (
            111.0
            *
            max(
                math.cos(
                    math.radians(middle_lat)
                ),
                0.2,
            )
        )
    )

    extra_buffer = 0.01

    return (
        min_lon - lon_buffer - extra_buffer,
        min_lat - lat_buffer - extra_buffer,
        max_lon + lon_buffer + extra_buffer,
        max_lat + lat_buffer + extra_buffer,
    )


# ============================================================
# CREATE / UPGRADE DATABASE TABLES
# ============================================================

def prepare_tables():

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS parking_locations (

                    id SERIAL PRIMARY KEY,

                    osm_id VARCHAR(200) UNIQUE,

                    location_id VARCHAR(100),

                    parking_name VARCHAR(255),

                    parking_type VARCHAR(100),

                    latitude DOUBLE PRECISION,

                    longitude DOUBLE PRECISION,

                    address TEXT,

                    capacity INTEGER,

                    source VARCHAR(100),

                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    last_verified_at TIMESTAMP
                )
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS poi_parking_map (

                    id SERIAL PRIMARY KEY,

                    poi_id INTEGER NOT NULL,

                    parking_id INTEGER NOT NULL,

                    distance_from_poi_km
                        DOUBLE PRECISION NOT NULL,

                    parking_rank INTEGER,

                    search_radius_used_km
                        DOUBLE PRECISION,

                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    CONSTRAINT uq_poi_parking
                        UNIQUE (poi_id, parking_id)
                )
                """
            )
        )

        conn.execute(
            text(
                """
                ALTER TABLE parking_locations
                ADD COLUMN IF NOT EXISTS
                last_verified_at TIMESTAMP
                """
            )
        )

        conn.execute(
            text(
                """
                ALTER TABLE poi_parking_map
                ADD COLUMN IF NOT EXISTS
                parking_rank INTEGER
                """
            )
        )

        conn.execute(
            text(
                """
                ALTER TABLE poi_parking_map
                ADD COLUMN IF NOT EXISTS
                search_radius_used_km
                DOUBLE PRECISION
                """
            )
        )

    print(
        "Parking PostgreSQL tables ready."
    )


# ============================================================
# GET GPKG LAYERS
# ============================================================

def get_layers(gpkg_path):

    layer_info = pyogrio.list_layers(
        gpkg_path
    )

    layers = [
        row[0]
        for row in layer_info
    ]

    print()
    print("=" * 80)
    print("GEOPACKAGE LAYERS")
    print("=" * 80)

    for layer in layers:

        print(
            f" - {layer}"
        )

    print("=" * 80)

    return layers


# ============================================================
# FIND TRAFFIC LAYERS
# ============================================================

def find_traffic_layers(layers):

    traffic_layers = [
        layer
        for layer in layers
        if "traffic" in layer.lower()
    ]

    if not traffic_layers:

        raise RuntimeError(
            "No traffic layer found in GeoPackage.\n"
            "Available layers:\n"
            +
            "\n".join(layers)
        )

    print(
        "Traffic layer(s): "
        +
        ", ".join(traffic_layers)
    )

    return traffic_layers


# ============================================================
# READ PARKING FROM GPKG
# ============================================================

def read_parking_layer(
    gpkg_path,
    layer,
):
    """
    Read the project-area part of the traffic layer and keep
    only parking classes.
    """

    bbox = get_project_bbox()

    print()
    print(
        f"Reading parking from: {layer}"
    )

    try:

        frame = pyogrio.read_dataframe(
            gpkg_path,
            layer=layer,
            bbox=bbox,
            columns=[
                "osm_id",
                "fclass",
                "name",
            ],
            where=(
                "fclass IN "
                "('parking',"
                "'parking_surface',"
                "'parking_multistorey',"
                "'parking_underground',"
                "'parking_street')"
            ),
        )

        print(
            f"Parking features returned: "
            f"{len(frame)}"
        )

        return frame

    except Exception as fast_error:

        print(
            "Filtered read failed; "
            "trying spatial read + Python filtering."
        )

        print(
            f"Reason: {fast_error}"
        )

        frame = pyogrio.read_dataframe(
            gpkg_path,
            layer=layer,
            bbox=bbox,
        )

        if frame.empty:

            return frame

        if "fclass" not in frame.columns:

            raise RuntimeError(
                "Traffic layer does not contain "
                "the expected fclass column."
            )

        frame["fclass"] = (
            frame["fclass"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        frame = frame[
            frame["fclass"].isin(
                PARKING_CLASSES
            )
        ].copy()

        print(
            f"Parking features after filtering: "
            f"{len(frame)}"
        )

        return frame


# ============================================================
# REPRESENTATIVE POINT
# ============================================================

def representative_point(geometry):

    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if geometry.geom_type == "Point":
        return geometry

    return geometry.centroid


# ============================================================
# FIND NEAREST PROJECT LOCATION
# ============================================================

def nearest_project_location(
    latitude,
    longitude,
):

    best_location = None
    best_distance = float("inf")

    for location_id, coords in (
        LOCATIONS.items()
    ):

        distance = haversine_distance(
            latitude,
            longitude,
            coords["latitude"],
            coords["longitude"],
        )

        if distance < best_distance:

            best_distance = distance
            best_location = location_id

    return (
        best_location,
        best_distance,
    )


# ============================================================
# BUILD PARKING RECORDS
# ============================================================

def build_parking_records(
    frame,
    layer,
):

    if frame.empty:
        return []

    frame = frame.copy()

    # Ensure WGS84 coordinates.
    if frame.crs is not None:

        try:

            if frame.crs.to_epsg() != 4326:

                frame = frame.to_crs(
                    4326
                )

        except Exception:
            pass

    records = []

    for _, row in frame.iterrows():

        point = representative_point(
            row.geometry
        )

        if point is None:
            continue

        latitude = float(point.y)
        longitude = float(point.x)

        (
            location_id,
            distance_to_anchor,
        ) = nearest_project_location(
            latitude,
            longitude
        )

        # Keep parking only if it is within 4 km of at least
        # one of our eight project locations.
        if distance_to_anchor > MAX_RADIUS_KM:
            continue

        raw_osm_id = row.get(
            "osm_id"
        )

        if (
            raw_osm_id is None
            or pd.isna(raw_osm_id)
        ):

            raw_osm_id = (
                f"{latitude:.7f}_"
                f"{longitude:.7f}"
            )

        osm_id = (
            f"{layer}:{raw_osm_id}"
        )

        name = row.get(
            "name"
        )

        if (
            name is None
            or pd.isna(name)
            or not str(name).strip()
        ):

            name = (
                "Unnamed OSM parking facility"
            )

        else:

            name = str(name).strip()

        parking_type = str(
            row.get(
                "fclass",
                "parking"
            )
        ).strip()

        records.append(
            {
                "osm_id":
                    osm_id,

                "location_id":
                    location_id,

                "parking_name":
                    name,

                "parking_type":
                    parking_type,

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "address":
                    "",

                # Do not invent capacity.
                "capacity":
                    None,

                "source":
                    SOURCE,
            }
        )

    return records


# ============================================================
# REFRESH PARKING MASTER DATABASE
# ============================================================

def refresh_parking_database():

    gpkg_path = find_gpkg()

    prepare_tables()

    layers = get_layers(
        gpkg_path
    )

    traffic_layers = (
        find_traffic_layers(
            layers
        )
    )

    all_records = []

    for layer in traffic_layers:

        frame = read_parking_layer(
            gpkg_path,
            layer
        )

        if frame.empty:
            continue

        records = build_parking_records(
            frame,
            layer
        )

        print(
            f"Valid project-area parking: "
            f"{len(records)}"
        )

        all_records.extend(
            records
        )

    if not all_records:

        raise RuntimeError(
            "No real parking facilities were "
            "found in the project area."
        )

    # Deduplicate.
    unique_records = {}

    for record in all_records:

        unique_records[
            record["osm_id"]
        ] = record

    records = list(
        unique_records.values()
    )

    print()
    print(
        f"Unique project-area parking facilities: "
        f"{len(records)}"
    )

    verified_at = datetime.now()

    # Delete previous local snapshot + its mappings.
    with engine.begin() as conn:

        conn.execute(
            text(
                """
                DELETE FROM poi_parking_map

                WHERE parking_id IN (

                    SELECT id
                    FROM parking_locations

                    WHERE source = :source
                )
                """
            ),
            {
                "source": SOURCE
            },
        )

        conn.execute(
            text(
                """
                DELETE FROM parking_locations

                WHERE source = :source
                """
            ),
            {
                "source": SOURCE
            },
        )

    insert_sql = text(
        """
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
            source,
            last_verified_at
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
            :source,
            :last_verified_at
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
                EXCLUDED.source,

            last_verified_at =
                EXCLUDED.last_verified_at
        """
    )

    with engine.begin() as conn:

        for record in records:

            conn.execute(
                insert_sql,
                {
                    **record,
                    "last_verified_at":
                        verified_at,
                },
            )

    print()
    print("=" * 80)

    print(
        "LOCAL OSM PARKING REFRESH COMPLETED"
    )

    print(
        f"Parking facilities loaded: "
        f"{len(records)}"
    )

    print(
        f"Verified at: {verified_at}"
    )

    print(
        "No Overpass request was made."
    )

    print("=" * 80)

    return len(records)


# ============================================================
# GET ALL REAL POIs
# ============================================================

def get_all_pois():

    query = text(
        """
        SELECT
            id,
            location_id,
            poi_name,
            poi_type,
            latitude,
            longitude,
            address

        FROM city_pois

        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL

        ORDER BY
            location_id,
            poi_type,
            poi_name
        """
    )

    with engine.connect() as conn:

        return (
            conn.execute(query)
            .mappings()
            .all()
        )


# ============================================================
# GET LOCAL PARKING
# ============================================================

def get_all_parking():

    query = text(
        """
        SELECT
            id,
            osm_id,
            location_id,
            parking_name,
            parking_type,
            latitude,
            longitude,
            address,
            capacity,
            source,
            last_verified_at

        FROM parking_locations

        WHERE source = :source
        """
    )

    with engine.connect() as conn:

        return (
            conn.execute(
                query,
                {
                    "source":
                        SOURCE
                },
            )
            .mappings()
            .all()
        )


# ============================================================
# ALL PARKING FOR ONE POI
# ============================================================

def find_parking_for_poi(
    poi,
    parking_rows,
):
    """
    Return ALL genuine parking facilities within 4 km.

    The database is NOT capped at five.
    """

    poi_lat = float(
        poi["latitude"]
    )

    poi_lon = float(
        poi["longitude"]
    )

    candidates = []

    for parking in parking_rows:

        distance = haversine_distance(
            poi_lat,
            poi_lon,
            parking["latitude"],
            parking["longitude"],
        )

        if distance <= MAX_RADIUS_KM:

            item = dict(
                parking
            )

            item[
                "distance_from_poi_km"
            ] = round(
                distance,
                3
            )

            # Distance band only.
            if distance <= FIRST_RADIUS_KM:

                radius_used = (
                    FIRST_RADIUS_KM
                )

            elif distance <= SECOND_RADIUS_KM:

                radius_used = (
                    SECOND_RADIUS_KM
                )

            else:

                radius_used = (
                    MAX_RADIUS_KM
                )

            item[
                "search_radius_used_km"
            ] = radius_used

            candidates.append(
                item
            )

    candidates.sort(
        key=lambda row:
            row[
                "distance_from_poi_km"
            ]
    )

    # Temporary distance rank.
    # Later Streamlit will replace this with the intelligent
    # recommendation score.
    for rank, item in enumerate(
        candidates,
        start=1,
    ):

        item[
            "distance_rank"
        ] = rank

    return candidates


# ============================================================
# BUILD POI -> PARKING MAP
# ============================================================

def rebuild_poi_parking_map():

    pois = get_all_pois()

    parking_rows = get_all_parking()

    if not pois:

        raise RuntimeError(
            "city_pois contains no usable POIs."
        )

    if not parking_rows:

        raise RuntimeError(
            "No local Geofabrik parking records "
            "exist in parking_locations."
        )

    # Remove old mappings for our local parking source.
    with engine.begin() as conn:

        conn.execute(
            text(
                """
                DELETE FROM poi_parking_map

                WHERE parking_id IN (

                    SELECT id
                    FROM parking_locations

                    WHERE source = :source
                )
                """
            ),
            {
                "source":
                    SOURCE
            },
        )

    insert_sql = text(
        """
        INSERT INTO poi_parking_map
        (
            poi_id,
            parking_id,
            distance_from_poi_km,
            parking_rank,
            search_radius_used_km
        )

        VALUES
        (
            :poi_id,
            :parking_id,
            :distance_from_poi_km,
            :parking_rank,
            :search_radius_used_km
        )

        ON CONFLICT (poi_id, parking_id)

        DO UPDATE SET

            distance_from_poi_km =
                EXCLUDED.distance_from_poi_km,

            parking_rank =
                EXCLUDED.parking_rank,

            search_radius_used_km =
                EXCLUDED.search_radius_used_km
        """
    )

    total_links = 0
    fewer_than_five = []

    with engine.begin() as conn:

        for poi in pois:

            candidates = (
                find_parking_for_poi(
                    poi,
                    parking_rows,
                )
            )

            # ONLY for reporting.
            # We still store every candidate.
            if len(candidates) < (
                MIN_EXPECTED_OPTIONS
            ):

                fewer_than_five.append(
                    {
                        "location_id":
                            poi["location_id"],

                        "poi_name":
                            poi["poi_name"],

                        "poi_type":
                            poi["poi_type"],

                        "parking_count":
                            len(candidates),
                    }
                )

            for parking in candidates:

                conn.execute(
                    insert_sql,
                    {
                        "poi_id":
                            poi["id"],

                        "parking_id":
                            parking["id"],

                        "distance_from_poi_km":
                            parking[
                                "distance_from_poi_km"
                            ],

                        "parking_rank":
                            parking[
                                "distance_rank"
                            ],

                        "search_radius_used_km":
                            parking[
                                "search_radius_used_km"
                            ],
                    },
                )

                total_links += 1

    print()
    print("=" * 80)

    print(
        "POI -> ALL PARKING MAPPING COMPLETED"
    )

    print(
        f"POIs processed: {len(pois)}"
    )

    print(
        f"POI-parking links created: "
        f"{total_links}"
    )

    print(
        f"POIs with fewer than "
        f"{MIN_EXPECTED_OPTIONS} mapped parking "
        f"facilities within {MAX_RADIUS_KM:.1f} km: "
        f"{len(fewer_than_five)}"
    )

    print("=" * 80)

    return fewer_than_five


# ============================================================
# EIGHT-LOCATION SUMMARY
# ============================================================

def print_location_summary():

    query = text(
        """
        SELECT
            p.location_id,
            p.poi_type,

            COUNT(DISTINCT p.id)
                AS poi_count,

            COUNT(DISTINCT ppm.parking_id)
                AS parking_links

        FROM city_pois p

        LEFT JOIN poi_parking_map ppm
            ON ppm.poi_id = p.id

        WHERE p.location_id IN (
            'PARK_STREET',
            'ESPLANADE',
            'EM_BYPASS',
            'SALT_LAKE',
            'NEW_TOWN',
            'VIP_ROAD',
            'RASHBEHARI',
            'HOWRAH_APPROACH'
        )

        GROUP BY
            p.location_id,
            p.poi_type

        ORDER BY
            p.location_id,
            p.poi_type
        """
    )

    with engine.connect() as conn:

        rows = (
            conn.execute(query)
            .mappings()
            .all()
        )

    print()
    print("=" * 105)

    print(
        "EIGHT-LOCATION POI + PARKING SUMMARY"
    )

    print("=" * 105)

    current_location = None

    for row in rows:

        location = row[
            "location_id"
        ]

        if location != current_location:

            print()
            print(
                f"LOCATION: {location}"
            )

            print(
                "-" * 70
            )

            current_location = location

        print(
            f"  {row['poi_type']:<18}"
            f" POIs: {row['poi_count']:<4}"
            f" Parking links: "
            f"{row['parking_links']}"
        )

    print()
    print("=" * 105)


# ============================================================
# FULL POI VERIFICATION
# ============================================================

def print_poi_verification():

    query = text(
        """
        SELECT

            p.location_id,
            p.poi_type,
            p.poi_name,

            COUNT(ppm.parking_id)
                AS parking_count,

            COALESCE(
                MIN(
                    ppm.search_radius_used_km
                ),
                0
            ) AS smallest_radius_used_km

        FROM city_pois p

        LEFT JOIN poi_parking_map ppm
            ON ppm.poi_id = p.id

        WHERE p.latitude IS NOT NULL
          AND p.longitude IS NOT NULL

        GROUP BY
            p.location_id,
            p.poi_type,
            p.poi_name

        ORDER BY
            p.location_id,
            p.poi_type,
            p.poi_name
        """
    )

    with engine.connect() as conn:

        rows = (
            conn.execute(query)
            .mappings()
            .all()
        )

    print()
    print("=" * 120)

    print(
        "EVERY POI -> ALL REAL PARKING WITHIN 4 KM"
    )

    print("=" * 120)

    fewer = 0

    for row in rows:

        count = int(
            row["parking_count"]
        )

        if count >= MIN_EXPECTED_OPTIONS:

            status = "OK"

        else:

            status = "FEWER THAN 5"
            fewer += 1

        radius = float(
            row[
                "smallest_radius_used_km"
            ]
        )

        print(
            f"{status:<13} | "
            f"{row['location_id']:<18} | "
            f"{row['poi_type']:<18} | "
            f"{str(row['poi_name'])[:45]:<45} | "
            f"Parking: {count:<2} | "
            f"Radius: {radius:.1f} km"
        )

    print()
    print(
        f"POIs with fewer than "
        f"{MIN_EXPECTED_OPTIONS} genuine "
        f"parking facilities within "
        f"{MAX_RADIUS_KM:.1f} km: {fewer}"
    )

    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        print()
        print("=" * 100)

        print(
            "SMART CITY - COMPLETE PARKING DATABASE BUILD"
        )

        print("=" * 100)

        # 1. Load real local OSM parking.
        refresh_parking_database()

        # 2. Link ALL nearby parking to every real POI.
        rebuild_poi_parking_map()

        # 3. Show the eight locations.
        print_location_summary()

        # 4. Verify every POI.
        print_poi_verification()

        print()
        print("=" * 100)

        print(
            "PARKING DATABASE BUILD COMPLETED"
        )

        print("=" * 100)

    except Exception as exc:

        print()
        print("=" * 100)

        print(            "PARKING DATABASE BUILD ERROR"
        )

        print("=" * 100)

        print(
            exc
        )