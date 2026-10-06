from __future__ import annotations

import os
import random
import time
from datetime import timedelta

from sqlalchemy import text

from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

# One real minute = one simulated parking hour.
# Change to 3600 for true real-hour updates.
SLEEP_SECONDS = int(
    os.getenv(
        "PARKING_UPDATE_SECONDS",
        "60"
    )
)

# How much occupancy can change in one simulated hour.
MAX_CHANGE_FRACTION = 0.08

MIN_OCCUPANCY = 0.03
MAX_OCCUPANCY = 0.97

RANDOM_SEED = 20260830


# ============================================================
# DATABASE SETUP
# ============================================================

def ensure_table_exists() -> None:

    with engine.begin() as conn:

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS
                parking_occupancy_history
                (
                    id BIGSERIAL PRIMARY KEY,

                    parking_id INTEGER NOT NULL,

                    timestamp TIMESTAMP NOT NULL,

                    capacity INTEGER NOT NULL,

                    occupied_spaces INTEGER NOT NULL,

                    available_spaces INTEGER NOT NULL,

                    occupancy_pct DOUBLE PRECISION NOT NULL,

                    source VARCHAR(50) NOT NULL,

                    CONSTRAINT
                    uq_parking_occupancy_lot_time
                    UNIQUE
                    (
                        parking_id,
                        timestamp
                    )
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                idx_parking_occupancy_parking_time
                ON parking_occupancy_history
                (
                    parking_id,
                    timestamp DESC
                );
                """
            )
        )

        conn.execute(
            text(
                """
                CREATE OR REPLACE VIEW
                parking_current_status AS

                SELECT DISTINCT ON
                (
                    parking_id
                )

                    parking_id,
                    timestamp,
                    capacity,
                    occupied_spaces,
                    available_spaces,
                    occupancy_pct,
                    source

                FROM parking_occupancy_history

                ORDER BY
                    parking_id,
                    timestamp DESC;
                """
            )
        )


# ============================================================
# GET CURRENT PARKING STATE
# ============================================================

def get_current_parking_state() -> list[dict]:

    query = text(
        """
        SELECT
            pcs.parking_id,
            p.parking_name,
            p.parking_type,
            p.latitude,
            p.longitude,

            pcs.timestamp,
            pcs.capacity,
            pcs.occupied_spaces,
            pcs.available_spaces,
            pcs.occupancy_pct,
            pcs.source

        FROM parking_current_status pcs

        JOIN parking_locations p
            ON p.id = pcs.parking_id

        WHERE pcs.source IN
        (
            'S',
            'OpenStreetMap',
            'OpenStreetMap-Geofabrik'
        )

        ORDER BY pcs.parking_id;
        """
    )

    with engine.connect() as conn:

        rows = (
            conn.execute(query)
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# OCCUPANCY TARGET
# ============================================================

def occupancy_target(
    timestamp,
    parking_type,
    parking_id,
    source,
) -> float:
    """
    Synthetic but structured demand pattern.
    """

    hour = timestamp.hour

    weekday = timestamp.weekday()

    weekend = weekday >= 5

    # --------------------------------------------------------
    # Base demand
    # --------------------------------------------------------

    base = 0.20

    # Morning
    if 7 <= hour <= 9:

        base += 0.15

    # Midday
    elif 10 <= hour <= 13:

        base += 0.25

    # Afternoon
    elif 14 <= hour <= 16:

        base += 0.27

    # Evening peak
    elif 17 <= hour <= 20:

        base += 0.30

    # Late evening
    elif 21 <= hour <= 23:

        base -= 0.08

    # Night
    elif 0 <= hour <= 5:

        base -= 0.10

    # --------------------------------------------------------
    # Weekend effect
    # --------------------------------------------------------

    if weekend:

        if 10 <= hour <= 20:
            base += 0.10

        else:
            base -= 0.04

    # --------------------------------------------------------
    # Parking type effect
    # --------------------------------------------------------

    ptype = str(
        parking_type or ""
    ).strip().lower()

    if ptype == "street_side":

        base += 0.03

    elif ptype == "surface":

        base += 0.02

    elif ptype == "general":

        base += 0.04

    elif ptype in {
        "parking",
        "car",
        "car_park",
    }:

        base += 0.05

    # --------------------------------------------------------
    # S DATA variation
    # --------------------------------------------------------

    if source == "S":

        base += 0.01

    # --------------------------------------------------------
    # Lot-specific deterministic variation
    # --------------------------------------------------------

    lot_factor = (
        (int(parking_id) % 11) - 5
    ) * 0.01

    base += lot_factor

    return max(
        MIN_OCCUPANCY,
        min(
            MAX_OCCUPANCY,
            base
        )
    )


# ============================================================
# GENERATE NEXT OCCUPANCY
# ============================================================

def generate_next_state(
    lot: dict,
    next_timestamp,
    rng: random.Random,
) -> dict:

    capacity = int(
        lot["capacity"]
    )

    previous_occupied = int(
        lot["occupied_spaces"]
    )

    target_fraction = occupancy_target(
        timestamp=next_timestamp,
        parking_type=lot.get(
            "parking_type"
        ),
        parking_id=lot["parking_id"],
        source=lot["source"],
    )

    # --------------------------------------------------------
    # Random variation
    # --------------------------------------------------------

    noise = rng.gauss(
        0.0,
        0.025
    )

    desired_fraction = (
        target_fraction
        + noise
    )

    desired_fraction = max(
        MIN_OCCUPANCY,
        min(
            MAX_OCCUPANCY,
            desired_fraction
        )
    )

    desired_occupied = int(
        round(
            capacity
            * desired_fraction
        )
    )

    # --------------------------------------------------------
    # Limit hourly change
    # --------------------------------------------------------

    maximum_change = max(
        2,
        int(
            capacity
            * MAX_CHANGE_FRACTION
        )
    )

    lower_bound = (
        previous_occupied
        - maximum_change
    )

    upper_bound = (
        previous_occupied
        + maximum_change
    )

    occupied = max(
        lower_bound,
        min(
            upper_bound,
            desired_occupied
        )
    )

    occupied = max(
        0,
        min(
            capacity,
            int(occupied)
        )
    )

    available = (
        capacity
        - occupied
    )

    occupancy_pct = (
        occupied
        / capacity
        * 100.0
    )

    return {
        "parking_id":
            int(
                lot["parking_id"]
            ),

        "timestamp":
            next_timestamp,

        "capacity":
            capacity,

        "occupied_spaces":
            occupied,

        "available_spaces":
            available,

        "occupancy_pct":
            round(
                occupancy_pct,
                2
            ),

        "source":
            lot["source"],
    }


# ============================================================
# INSERT ONE SIMULATED HOUR
# ============================================================

def insert_next_hour(
    lots: list[dict],
    next_timestamp,
) -> int:

    rng = random.Random(
        RANDOM_SEED
        + int(
            next_timestamp.timestamp()
        )
    )

    rows = []

    for lot in lots:

        row = generate_next_state(
            lot=lot,
            next_timestamp=next_timestamp,
            rng=rng,
        )

        rows.append(
            row
        )

    if not rows:
        return 0

    with engine.begin() as conn:

        result = conn.execute(
            text(
                """
                INSERT INTO
                parking_occupancy_history
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

                ON CONFLICT
                (
                    parking_id,
                    timestamp
                )

                DO NOTHING;
                """
            ),
            rows,
        )

    return (
        result.rowcount
        or 0
    )


# ============================================================
# NEXT SIMULATED TIMESTAMP
# ============================================================

def get_next_timestamp(
    lots: list[dict],
):

    if not lots:

        raise RuntimeError(
            "No parking lots with current status found."
        )

    latest = max(
        lot["timestamp"]
        for lot in lots
        if lot["timestamp"] is not None
    )

    latest = latest.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    return (
        latest
        + timedelta(hours=1)
    )


# ============================================================
# PRINT STATUS
# ============================================================

def print_cycle_summary(
    lots: list[dict],
    next_timestamp,
    inserted: int,
):

    print()
    print(
        "=" * 80
    )

    print(
        "PARKING LIVE UPDATE"
    )

    print(
        "=" * 80
    )

    print(
        f"Simulated timestamp : "
        f"{next_timestamp}"
    )

    print(
        f"Parking lots        : "
        f"{len(lots)}"
    )

    print(
        f"Rows inserted       : "
        f"{inserted}"
    )

    print()

    print(
        f"{'Parking':<30}"
        f"{'Capacity':>10}"
        f"{'Occupied':>10}"
        f"{'Available':>11}"
        f"{'Occupancy':>12}"
    )

    print(
        "-" * 80
    )

    for lot in lots:

        capacity = int(
            lot["capacity"]
        )

        occupied = int(
            lot["occupied_spaces"]
        )

        available = int(
            lot["available_spaces"]
        )

        occupancy = float(
            lot["occupancy_pct"]
        )

        print(
            f"{str(lot['parking_name'])[:29]:<30}"
            f"{capacity:>10}"
            f"{occupied:>10}"
            f"{available:>11}"
            f"{occupancy:>11.1f}%"
        )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "CONTINUOUS PARKING OCCUPANCY UPDATER"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"Update interval : "
        f"{SLEEP_SECONDS} seconds"
    )

    print(
        "Simulation      : "
        "1 real minute = 1 simulated hour"
    )

    print()

    ensure_table_exists()

    while True:

        try:

            lots = (
                get_current_parking_state()
            )

            if not lots:

                print(
                    "No parking status records found."
                )

                print(
                    "Run parking_simulator.py "
                    "once first."
                )

                time.sleep(
                    SLEEP_SECONDS
                )

                continue

            # ------------------------------------------------
            # Determine next hour
            # ------------------------------------------------

            next_timestamp = (
                get_next_timestamp(
                    lots
                )
            )

            # ------------------------------------------------
            # Insert next state
            # ------------------------------------------------

            inserted = (
                insert_next_hour(
                    lots,
                    next_timestamp,
                )
            )

            # ------------------------------------------------
            # Reload current status
            # ------------------------------------------------

            updated_lots = (
                get_current_parking_state()
            )

            print_cycle_summary(
                updated_lots,
                next_timestamp,
                inserted,
            )

            print()
            print(
                f"Next update in "
                f"{SLEEP_SECONDS} seconds..."
            )

            print(
                "Press Ctrl+C to stop."
            )

            time.sleep(
                SLEEP_SECONDS
            )

        except KeyboardInterrupt:

            print()
            print(
                "Parking live updater stopped."
            )

            break

        except Exception as exc:

            print()
            print(
                "ERROR:"
            )

            print(
                str(exc)
            )

            print()
            print(
                "Retrying after "
                f"{SLEEP_SECONDS} seconds..."
            )

            time.sleep(
                SLEEP_SECONDS
            )


if __name__ == "__main__":
    main()