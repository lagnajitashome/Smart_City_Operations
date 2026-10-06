from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
)

from database.connection import engine


metadata = MetaData()


traffic_table = Table(
    "traffic_data",
    metadata,

    Column("id", Integer, primary_key=True, autoincrement=True),

    # Time and location
    Column("timestamp", DateTime),
    Column("location_id", String(50)),
    Column("latitude", Float),
    Column("longitude", Float),

    # Traffic measurements
    Column("current_speed", Float),
    Column("free_flow_speed", Float),

    Column("current_travel_time", Float),
    Column("free_flow_travel_time", Float),

    # Derived traffic features
    Column("traffic_delay", Float),
    Column("speed_reduction", Float),

    # API information
    Column("confidence", Float),
    Column("road_closure", Boolean),
    Column("frc", String(20))
)


# ------------------------------------------------
# Recreate table
# ------------------------------------------------

traffic_table.drop(engine, checkfirst=True)

metadata.create_all(engine)


print("=" * 60)
print("Traffic Table Created Successfully")
print("=" * 60)