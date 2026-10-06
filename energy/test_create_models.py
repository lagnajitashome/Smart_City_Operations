import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# DATABASE
# ============================================================

db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)

engine = create_engine(db_url)


# ============================================================
# LOAD DATA
# ============================================================

query = """
SELECT
    timestamp,
    location_id,
    electricity_demand_mw,
    solar_generation_mw,
    ev_charging_demand_mw,
    transformer_load_pct
FROM energy_data
ORDER BY timestamp;
"""

df = pd.read_sql(
    query,
    engine
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# ============================================================
# CITY LEVEL
# ============================================================

city_df = (

    df.groupby("timestamp")

    .agg(

        electricity_demand_mw=(
            "electricity_demand_mw",
            "sum"
        ),

        solar_generation_mw=(
            "solar_generation_mw",
            "sum"
        ),

        ev_charging_demand_mw=(
            "ev_charging_demand_mw",
            "sum"
        ),

        transformer_load_pct=(
            "transformer_load_pct",
            "mean"
        )

    )

    .sort_index()
)


# ============================================================
# BACKFILL BOUNDARY
# ============================================================

first_timestamp = city_df.index.min()

backfill_end = (
    first_timestamp
    +
    pd.Timedelta(days=14)
)


# ============================================================
# RAW LIVE DATA
# ============================================================

raw_live = city_df[
    city_df.index >= backfill_end
].copy()


print("\n" + "=" * 70)
print("LIVE DATA DIAGNOSTIC")
print("=" * 70)

print(
    f"\nFirst timestamp : "
    f"{city_df.index.min()}"
)

print(
    f"Backfill ends   : "
    f"{backfill_end}"
)

print(
    f"Latest timestamp: "
    f"{city_df.index.max()}"
)

print(
    f"\nRAW LIVE TIMESTAMPS: "
    f"{len(raw_live):,}"
)


# ============================================================
# SHOW RAW LIVE RANGE
# ============================================================

if len(raw_live) > 0:

    print(
        f"Raw live start: "
        f"{raw_live.index.min()}"
    )

    print(
        f"Raw live end  : "
        f"{raw_live.index.max()}"
    )


# ============================================================
# CHECK NEXT-HOUR ACTUAL
# ============================================================

city_df["next_hour_ev_demand"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(-60)
)


live_with_target = city_df[
    city_df.index >= backfill_end
].copy()


print(
    f"\nLIVE ROWS WITH NEXT-HOUR ACTUAL: "
    f"{live_with_target['next_hour_ev_demand'].notna().sum():,}"
)


# ============================================================
# CHECK BASIC DATA
# ============================================================

print("\nMissing values in LIVE data:")

print(
    live_with_target[
        [
            "electricity_demand_mw",
            "solar_generation_mw",
            "ev_charging_demand_mw",
            "transformer_load_pct",
            "next_hour_ev_demand"
        ]
    ]
    .isna()
    .sum()
)


# ============================================================
# TIMESTAMP GAP ANALYSIS
# ============================================================

live_timestamps = raw_live.index.sort_values()

if len(live_timestamps) > 1:

    gaps = (
        live_timestamps[1:]
        -
        live_timestamps[:-1]
    )

    print("\nLIVE TIMESTAMP GAP ANALYSIS")

    print(
        f"Smallest gap : "
        f"{gaps.min()}"
    )

    print(
        f"Largest gap  : "
        f"{gaps.max()}"
    )

    print(
        f"1-minute gaps: "
        f"{(gaps == pd.Timedelta(minutes=1)).sum():,}"
    )


# ============================================================
# LAST 20 LIVE ROWS
# ============================================================

print("\n" + "=" * 70)
print("LAST 20 RAW LIVE OBSERVATIONS")
print("=" * 70)

print(
    raw_live[
        [
            "electricity_demand_mw",
            "solar_generation_mw",
            "ev_charging_demand_mw"
        ]
    ]
    .tail(20)
    .to_string()
)


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)