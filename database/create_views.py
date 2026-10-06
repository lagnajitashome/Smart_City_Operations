from sqlalchemy import text

from database.connection import engine


# ============================================================
# TRAFFIC + WEATHER VIEW
# ============================================================

VIEW_SQL = """
CREATE OR REPLACE VIEW traffic_weather_view AS

SELECT

    -- ========================================================
    -- TRAFFIC
    -- ========================================================

    t.timestamp AS traffic_timestamp,
    t.location_id,
    t.latitude,
    t.longitude,

    t.current_speed,
    t.free_flow_speed,
    t.current_travel_time,
    t.free_flow_travel_time,
    t.traffic_delay,
    t.speed_reduction,
    t.confidence,
    t.road_closure,
    t.frc,

    -- ========================================================
    -- WEATHER
    -- ========================================================

    w.timestamp AS weather_timestamp,
    w.temperature,
    w.feels_like,
    w.humidity,
    w.wind_speed,
    w.wind_direction,
    w.cloudiness,
    w.rain_mm,
    w.weather_main,
    w.weather_description,

    -- ========================================================
    -- WEATHER DATA AGE
    -- ========================================================

    EXTRACT(
        EPOCH FROM (t.timestamp - w.timestamp)
    ) AS weather_age_seconds

FROM traffic_data t

LEFT JOIN LATERAL (

    SELECT

        weather_data.timestamp,
        weather_data.temperature,
        weather_data.feels_like,
        weather_data.humidity,
        weather_data.wind_speed,
        weather_data.wind_direction,
        weather_data.cloudiness,
        weather_data.rain_mm,
        weather_data.weather_main,
        weather_data.weather_description

    FROM weather_data

    WHERE weather_data.location_id = t.location_id

      AND weather_data.timestamp <= t.timestamp

      AND weather_data.timestamp >=
          t.timestamp - INTERVAL '30 minutes'

    ORDER BY weather_data.timestamp DESC

    LIMIT 1

) w ON TRUE;
"""


# ============================================================
# CREATE VIEW
# ============================================================

with engine.begin() as conn:

    conn.execute(
        text(VIEW_SQL)
    )


print("=" * 60)
print("Traffic + Weather view created successfully.")
print("View name: traffic_weather_view")
print("=" * 60)