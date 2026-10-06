import time

from sqlalchemy import text

from database.connection import engine


MERGE_SQL = """
INSERT INTO traffic_weather_data (

    traffic_timestamp,
    weather_timestamp,
    location_id,

    latitude,
    longitude,

    current_speed,
    free_flow_speed,
    current_travel_time,
    free_flow_travel_time,

    traffic_delay,
    speed_reduction,
    confidence,
    road_closure,
    frc,

    temperature,
    feels_like,
    humidity,

    wind_speed,
    wind_direction,
    cloudiness,
    rain_mm,

    weather_main,
    weather_description,

    weather_age_seconds
)

SELECT

    t.timestamp,
    w.timestamp,
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

    w.temperature,
    w.feels_like,
    w.humidity,

    w.wind_speed,
    w.wind_direction,
    w.cloudiness,
    w.rain_mm,

    w.weather_main,
    w.weather_description,

    EXTRACT(
        EPOCH FROM (t.timestamp - w.timestamp)
    )

FROM traffic_data t

LEFT JOIN LATERAL (

    SELECT
        timestamp,
        temperature,
        feels_like,
        humidity,
        wind_speed,
        wind_direction,
        cloudiness,
        rain_mm,
        weather_main,
        weather_description

    FROM weather_data

    WHERE location_id = t.location_id

      AND timestamp <= t.timestamp

    ORDER BY timestamp DESC

    LIMIT 1

) w ON TRUE

ON CONFLICT (location_id, traffic_timestamp)
DO NOTHING;
"""


def merge_data():

    with engine.begin() as conn:

        result = conn.execute(text(MERGE_SQL))

        return result.rowcount


print("=" * 60)
print("SMART CITY MERGE PROCESSOR")
print("=" * 60)
print("Waiting for traffic and weather data...")
print()


while True:

    try:

        inserted = merge_data()

        if inserted > 0:
            print(
                f"✓ Merged {inserted} new traffic-weather record(s)"
            )
        else:
            print("Waiting for new traffic data...")

        time.sleep(5)

    except Exception as e:

        print("ERROR:", e)

        time.sleep(5)