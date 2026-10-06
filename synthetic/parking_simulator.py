from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from sqlalchemy import text
from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_DAYS = 30
HISTORY_INTERVAL_HOURS = 1
POI_MAPPING_RADIUS_KM = 2.0
S_SOURCE = "S"
REAL_SOURCES = ("OpenStreetMap", "OpenStreetMap-Geofabrik")

LOT_OFFSETS = [
    (1, 0.0090, 0.0060),
    (2, -0.0070, 0.0100),
    (3, 0.0100, -0.0080),
]

LOT_CAPACITIES = {1: 80, 2: 100, 3: 120}


# ============================================================
# MODELLING-ONLY CAPACITY FOR ORIGINAL PARKING
# ============================================================

def model_capacity(parking_type: str | None) -> int:
    """Return a modelling-only capacity when original capacity is NULL."""
    if parking_type is None:
        return 80

    ptype = str(parking_type).strip().lower()

    if ptype in {"street_side", "street"}:
        return 40
    if ptype in {"surface", "surface parking"}:
        return 80
    if ptype == "general":
        return 60
    if ptype in {"parking", "car", "car_park"}:
        return 100
    return 80


# ============================================================
# DATABASE SETUP
# ============================================================

def create_tables() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS parking_occupancy_history (
                id BIGSERIAL PRIMARY KEY,
                parking_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                capacity INTEGER NOT NULL,
                occupied_spaces INTEGER NOT NULL,
                available_spaces INTEGER NOT NULL,
                occupancy_pct DOUBLE PRECISION NOT NULL,
                source VARCHAR(50) NOT NULL,
                CONSTRAINT uq_parking_occupancy_lot_time
                    UNIQUE (parking_id, timestamp)
            );
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_parking_occupancy_parking_time
            ON parking_occupancy_history (parking_id, timestamp DESC);
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW parking_current_status AS
            SELECT DISTINCT ON (parking_id)
                parking_id,
                timestamp,
                capacity,
                occupied_spaces,
                available_spaces,
                occupancy_pct,
                source
            FROM parking_occupancy_history
            ORDER BY parking_id, timestamp DESC;
        """))


# ============================================================
# GET MONITORING LOCATIONS
# ============================================================

def get_monitoring_locations() -> list[dict]:
    query = text("""
        SELECT DISTINCT ON (location_id)
            location_id,
            latitude,
            longitude
        FROM traffic_data
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
        ORDER BY location_id, timestamp DESC;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    locations = [dict(row) for row in rows]

    if not locations:
        raise RuntimeError("No monitoring locations found in traffic_data.")

    return locations


# ============================================================
# CREATE / FIND S PARKING LOTS
# ============================================================

def create_s_parking_locations(monitoring_locations: list[dict]) -> list[dict]:
    lots = []

    with engine.begin() as conn:
        existing_rows = conn.execute(text("""
            SELECT id, parking_name, latitude, longitude, capacity
            FROM parking_locations
            WHERE source = 'S'
            ORDER BY id;
        """)).mappings().all()

        existing_by_name = {r["parking_name"]: dict(r) for r in existing_rows}
        lot_counter = 1

        for location in sorted(monitoring_locations, key=lambda r: r["location_id"]):
            base_lat = float(location["latitude"])
            base_lon = float(location["longitude"])

            for lot_number, lat_offset, lon_offset in LOT_OFFSETS:
                parking_name = f"S DATA {lot_counter}"
                latitude = base_lat + lat_offset
                longitude = base_lon + lon_offset
                capacity = LOT_CAPACITIES[lot_number]

                if parking_name in existing_by_name:
                    row = existing_by_name[parking_name]
                    lots.append({
                        "id": int(row["id"]),
                        "parking_name": parking_name,
                        "parking_type": "parking",
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "capacity": int(row["capacity"] or capacity),
                        "source": S_SOURCE,
                        "location_id": location["location_id"],
                    })
                else:
                    new_id = conn.execute(text("""
                        INSERT INTO parking_locations
                        (parking_name, parking_type, latitude, longitude,
                         address, capacity, source, last_verified_at)
                        VALUES
                        (:parking_name, 'parking', :latitude, :longitude,
                         :address, :capacity, 'S', CURRENT_TIMESTAMP)
                        RETURNING id;
                    """), {
                        "parking_name": parking_name,
                        "latitude": latitude,
                        "longitude": longitude,
                        "address": f"S DATA parking facility - {location['location_id']}",
                        "capacity": capacity,
                    }).scalar_one()

                    lots.append({
                        "id": int(new_id),
                        "parking_name": parking_name,
                        "parking_type": "parking",
                        "latitude": latitude,
                        "longitude": longitude,
                        "capacity": capacity,
                        "source": S_SOURCE,
                        "location_id": location["location_id"],
                    })

                lot_counter += 1

    return lots


# ============================================================
# GET ORIGINAL OSM / GEOfABRIK PARKING
# ============================================================

def get_real_parking_locations() -> list[dict]:
    query = text("""
        SELECT
            id,
            parking_name,
            parking_type,
            latitude,
            longitude,
            capacity,
            source
        FROM parking_locations
        WHERE source IN ('OpenStreetMap', 'OpenStreetMap-Geofabrik')
        ORDER BY id;
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# REALISTIC OCCUPANCY PATTERN
# ============================================================

def base_occupancy_fraction(
    timestamp: datetime,
    parking_type: str | None,
    parking_id: int,
) -> float:
    hour = timestamp.hour
    weekend = timestamp.weekday() >= 5

    base = 0.20

    if 7 <= hour <= 9:
        base += 0.15
    if 10 <= hour <= 13:
        base += 0.25
    if 14 <= hour <= 16:
        base += 0.27
    if 17 <= hour <= 20:
        base += 0.30
    if hour >= 21:
        base -= 0.08
    if hour <= 5:
        base -= 0.10

    if weekend:
        if 10 <= hour <= 20:
            base += 0.10
        else:
            base -= 0.04

    ptype = str(parking_type or "").strip().lower()

    if ptype == "street_side":
        base += 0.03
    elif ptype == "surface":
        base += 0.02
    elif ptype == "general":
        base += 0.04
    elif ptype in {"parking", "car", "car_park"}:
        base += 0.05

    base += ((parking_id % 11) - 5) * 0.01

    return max(0.03, min(0.97, base))


# ============================================================
# GENERATE HISTORY FOR ONE LOT
# ============================================================

def generate_lot_history(
    parking_id: int,
    capacity: int,
    parking_type: str | None,
    source: str,
    start_time: datetime,
    end_time: datetime,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    current = start_time
    previous_occupancy = None

    while current <= end_time:
        target_fraction = base_occupancy_fraction(
            current,
            parking_type,
            parking_id,
        )

        target_fraction = max(
            0.02,
            min(
                0.98,
                target_fraction + rng.gauss(0.0, 0.035),
            ),
        )

        target_occupied = int(round(capacity * target_fraction))

        if previous_occupancy is not None:
            maximum_change = max(2, int(capacity * 0.08))
            target_occupied = max(
                previous_occupancy - maximum_change,
                min(
                    previous_occupancy + maximum_change,
                    target_occupied,
                ),
            )

        occupied = max(0, min(capacity, int(target_occupied)))
        available = capacity - occupied
        occupancy_pct = (occupied / capacity * 100.0) if capacity else 0.0

        rows.append({
            "parking_id": parking_id,
            "timestamp": current,
            "capacity": capacity,
            "occupied_spaces": occupied,
            "available_spaces": available,
            "occupancy_pct": round(occupancy_pct, 2),
            "source": source,
        })

        previous_occupancy = occupied
        current += timedelta(hours=HISTORY_INTERVAL_HOURS)

    return rows


# ============================================================
# INSERT HISTORY IN BATCHES
# ============================================================

def insert_history(parking_lots: list[dict]) -> int:
    end_time = datetime.now().replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    start_time = (
        end_time
        - timedelta(days=HISTORY_DAYS)
        + timedelta(hours=1)
    )

    insert_sql = text("""
        INSERT INTO parking_occupancy_history
        (
            parking_id,
            timestamp,
            capacity,
            occupied_spaces,
            available_spaces,
            occupancy_pct,
            source
        )
        VALUES
        (
            :parking_id,
            :timestamp,
            :capacity,
            :occupied_spaces,
            :available_spaces,
            :occupancy_pct,
            :source
        )
        ON CONFLICT (parking_id, timestamp)
        DO NOTHING;
    """)

    total_inserted = 0
    batch = []
    batch_size = 10000

    with engine.begin() as conn:
        for lot in parking_lots:
            lot_number = 1
            if str(lot["parking_name"]).startswith("S DATA "):
                try:
                    n = int(str(lot["parking_name"]).replace("S DATA ", ""))
                    lot_number = ((n - 1) % 3) + 1
                except ValueError:
                    lot_number = 1

            rows = generate_lot_history(
                parking_id=int(lot["id"]),
                capacity=int(lot["capacity"]),
                parking_type=lot.get("parking_type"),
                source=lot["source"],
                start_time=start_time,
                end_time=end_time,
                seed=int(lot["id"]) * 1009 + lot_number,
            )

            batch.extend(rows)

            if len(batch) >= batch_size:
                result = conn.execute(insert_sql, batch)
                total_inserted += result.rowcount or 0
                batch.clear()

        if batch:
            result = conn.execute(insert_sql, batch)
            total_inserted += result.rowcount or 0

    return total_inserted


# ============================================================
# MAP S PARKING TO POIs
# ============================================================

def map_s_parking_to_pois(radius_km: float = POI_MAPPING_RADIUS_KM) -> int:
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO poi_parking_map
            (
                poi_id,
                parking_id,
                distance_from_poi_km
            )
            SELECT
                p.id,
                s.id,
                (
                    6371.0 * 2.0 * ASIN(
                        SQRT(
                            POWER(
                                SIN(
                                    RADIANS(s.latitude - p.latitude) / 2.0
                                ), 2
                            )
                            +
                            COS(RADIANS(p.latitude))
                            * COS(RADIANS(s.latitude))
                            * POWER(
                                SIN(
                                    RADIANS(s.longitude - p.longitude) / 2.0
                                ), 2
                            )
                        )
                    )
                )
            FROM city_pois p
            CROSS JOIN parking_locations s
            WHERE s.source = 'S'
              AND p.latitude IS NOT NULL
              AND p.longitude IS NOT NULL
              AND (
                    6371.0 * 2.0 * ASIN(
                        SQRT(
                            POWER(
                                SIN(
                                    RADIANS(s.latitude - p.latitude) / 2.0
                                ), 2
                            )
                            +
                            COS(RADIANS(p.latitude))
                            * COS(RADIANS(s.latitude))
                            * POWER(
                                SIN(
                                    RADIANS(s.longitude - p.longitude) / 2.0
                                ), 2
                            )
                        )
                    )
                  ) <= :radius_km
            ON CONFLICT (poi_id, parking_id)
            DO NOTHING;
        """), {"radius_km": radius_km})

        return result.rowcount or 0


# ============================================================
# SUMMARY
# ============================================================

def print_summary() -> None:
    with engine.connect() as conn:
        parking_counts = conn.execute(text("""
            SELECT source, COUNT(*) AS lots
            FROM parking_locations
            WHERE source IN ('S', 'OpenStreetMap', 'OpenStreetMap-Geofabrik')
            GROUP BY source
            ORDER BY source;
        """)).all()

        history_counts = conn.execute(text("""
            SELECT
                source,
                COUNT(*) AS rows,
                COUNT(DISTINCT parking_id) AS lots,
                MIN(timestamp) AS history_start,
                MAX(timestamp) AS history_end
            FROM parking_occupancy_history
            GROUP BY source
            ORDER BY source;
        """)).all()

        status = conn.execute(text("""
            SELECT
                COUNT(*) AS lots,
                ROUND(AVG(occupancy_pct)::numeric, 2) AS avg_occupancy
            FROM parking_current_status;
        """)).one()

    print("\n" + "=" * 80)
    print("UNIFIED PARKING DATASET")
    print("=" * 80)

    print("\nPARKING LOCATIONS")
    print("-" * 80)
    for row in parking_counts:
        print(f"{row[0]:<30} : {row[1]}")

    print("\nOCCUPANCY HISTORY")
    print("-" * 80)
    for row in history_counts:
        print(
            f"{row[0]:<30} : rows={row[1]}"
            f" | lots={row[2]}"
            f" | {row[3]} -> {row[4]}"
        )

    print(f"\nCurrent parking lots with status : {status[0]}")
    print(f"Average current occupancy        : {status[1]}%")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("UNIFIED PARKING OCCUPANCY PIPELINE")
    print("=" * 80)

    print("\n1. Preparing database...")
    create_tables()

    print("2. Loading monitoring locations...")
    monitoring_locations = get_monitoring_locations()
    print(f"   Found {len(monitoring_locations)} monitoring locations.")

    print("3. Ensuring S DATA parking...")
    s_lots = create_s_parking_locations(monitoring_locations)
    print(f"   S parking lots: {len(s_lots)}")

    print("4. Loading original OSM parking...")
    real_lots = get_real_parking_locations()
    print(f"   Original parking lots: {len(real_lots)}")

    print("5. Preparing modelling capacities...")
    real_for_history = []
    actual_capacity_count = 0
    modelled_capacity_count = 0

    for lot in real_lots:
        if lot["capacity"] is not None:
            effective_capacity = int(lot["capacity"])
            actual_capacity_count += 1
        else:
            effective_capacity = model_capacity(lot["parking_type"])
            modelled_capacity_count += 1

        real_for_history.append({
            "id": int(lot["id"]),
            "parking_name": lot["parking_name"],
            "parking_type": lot["parking_type"],
            "capacity": effective_capacity,
            "source": lot["source"],
        })

    print(f"   Original lots with existing capacity : {actual_capacity_count}")
    print(f"   Original lots with modelled capacity  : {modelled_capacity_count}")

    s_for_history = [
        {
            "id": int(lot["id"]),
            "parking_name": lot["parking_name"],
            "parking_type": lot.get("parking_type", "parking"),
            "capacity": int(lot["capacity"]),
            "source": S_SOURCE,
        }
        for lot in s_lots
    ]

    print("6. Generating unified occupancy history...")
    all_lots = real_for_history + s_for_history
    print(f"   Lots receiving history: {len(all_lots)}")
    inserted = insert_history(all_lots)
    print(f"   New history rows inserted: {inserted}")
    print("   Existing parking/timestamp records were skipped.")

    print("7. Mapping S parking to POIs...")
    mapped = map_s_parking_to_pois()
    print(f"   New POI -> S parking mappings: {mapped}")

    print("8. Final verification...")
    print_summary()

    print("\nDONE")
    print("Original OSM/Geofabrik and S DATA parking now share one occupancy-history system.")


if __name__ == "__main__":
    main()
