import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import folium
import streamlit as st
from streamlit_folium import st_folium
from sqlalchemy import text


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT DATABASE
# ============================================================

from database.connection import engine

from utils.route_congestion_predictor import (
    get_location_predictions,
    get_route_congestion_predictions,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart City Mobility",
    page_icon="🚦",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Smart City Mobility Dashboard")

st.caption(
    "Live Kolkata mobility monitoring, route intelligence "
    "and parking visualization"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Keep the same weights used by the working routing version.
TRAFFIC_WEIGHT = 0.50
WEATHER_WEIGHT = 0.30
TIME_WEIGHT = 0.20


# ============================================================
# DATABASE — LATEST TRAFFIC + WEATHER
# ============================================================

@st.cache_data(ttl=30)
def get_latest_traffic_weather():

    query = text("""
        SELECT DISTINCT ON (location_id)

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

            traffic_timestamp,
            weather_timestamp

        FROM traffic_weather_data

        ORDER BY
            location_id,
            traffic_timestamp DESC
    """)

    with engine.connect() as conn:

        rows = (
            conn.execute(query)
            .mappings()
            .all()
        )

    return [dict(row) for row in rows]


# ============================================================
# DATABASE — PARKING
# ============================================================

@st.cache_data(ttl=30)
def get_parking_locations():
    """Return parking locations with the latest occupancy snapshot."""

    query = text("""
        SELECT
            pl.id,
            pl.parking_name,
            pl.parking_type,
            pl.latitude,
            pl.longitude,
            pl.address,
            COALESCE(pl.capacity, pcs.capacity) AS capacity,
            pcs.occupied_spaces,
            pcs.available_spaces,
            pcs.occupancy_pct,
            pcs.timestamp AS occupancy_timestamp,
            pl.source,
            pl.last_verified_at
        FROM parking_locations pl
        LEFT JOIN parking_current_status pcs
            ON pcs.parking_id = pl.id
        WHERE pl.latitude IS NOT NULL
          AND pl.longitude IS NOT NULL
        ORDER BY pl.id
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# POI / DESTINATION / PARKING HELPERS
# ============================================================

@st.cache_data(ttl=60)
def get_poi_categories(location_id):
    """Return destination categories available at one monitoring zone."""

    query = text("""
        SELECT DISTINCT poi_type
        FROM city_pois
        WHERE location_id = :location_id
          AND poi_type IS NOT NULL
        ORDER BY poi_type
    """)

    with engine.connect() as conn:
        rows = (
            conn.execute(
                query,
                {"location_id": location_id}
            )
            .mappings()
            .all()
        )

    return [
        row["poi_type"]
        for row in rows
    ]


@st.cache_data(ttl=60)
def get_pois(location_id, poi_type):
    """Return real POIs for a location and selected category."""

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
        rows = (
            conn.execute(
                query,
                {
                    "location_id": location_id,
                    "poi_type": poi_type
                }
            )
            .mappings()
            .all()
        )

    return [dict(row) for row in rows]


@st.cache_data(ttl=60)
def get_parking_for_poi(poi_id):
    """Return mapped parking plus the latest occupancy snapshot."""

    query = text("""
        SELECT
            p.id AS poi_id,
            p.location_id,
            p.poi_name,
            p.poi_type,

            pl.id AS parking_id,
            pl.parking_name,
            pl.parking_type,
            pl.latitude,
            pl.longitude,
            pl.address,
            COALESCE(pl.capacity, pcs.capacity) AS capacity,
            pcs.occupied_spaces,
            pcs.available_spaces,
            pcs.occupancy_pct,
            pcs.timestamp AS occupancy_timestamp,
            pl.source,
            pl.last_verified_at,

            ppm.distance_from_poi_km

        FROM poi_parking_map ppm

        JOIN city_pois p
            ON p.id = ppm.poi_id

        JOIN parking_locations pl
            ON pl.id = ppm.parking_id

        LEFT JOIN parking_current_status pcs
            ON pcs.parking_id = pl.id

        WHERE p.id = :poi_id

        ORDER BY ppm.distance_from_poi_km ASC
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"poi_id": poi_id}
        ).mappings().all()

    return [dict(row) for row in rows]


# ============================================================
# PARKING ML — NEXT-HOUR OCCUPANCY PREDICTION
# ============================================================

PARKING_MODEL_PATH = PROJECT_ROOT / "models" / "parking_occupancy_model.pkl"

@st.cache_resource
def load_parking_occupancy_model():
    if not PARKING_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(PARKING_MODEL_PATH)
    except Exception:
        return None

@st.cache_data(ttl=30)
def get_parking_prediction_history(parking_id, hours=180):
    query = text("""
        SELECT parking_id, timestamp, capacity, occupied_spaces,
               available_spaces, occupancy_pct
        FROM parking_occupancy_history
        WHERE parking_id = :parking_id
        ORDER BY timestamp DESC
        LIMIT :hours
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"parking_id": parking_id, "hours": hours}).mappings().all()
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    history = pd.DataFrame([dict(r) for r in rows])
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    return history.sort_values("timestamp").reset_index(drop=True)

def create_prediction_features(history, source):
    data = history.sort_values("timestamp").copy().reset_index(drop=True)
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["day_of_month"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    for lag in [1,2,3,6,12,24,48,168]:
        data[f"lag_{lag}"] = data["occupancy_pct"].shift(lag)
    data["change_1h"] = data["occupancy_pct"] - data["lag_1"]
    data["change_3h"] = data["occupancy_pct"] - data["lag_3"]
    shifted = data["occupancy_pct"].shift(1)
    data["rolling_mean_3"] = shifted.rolling(3).mean()
    data["rolling_mean_6"] = shifted.rolling(6).mean()
    data["rolling_mean_24"] = shifted.rolling(24).mean()
    data["rolling_std_24"] = shifted.rolling(24).std()
    data["is_s_source"] = 1 if source == "S" else 0
    return data

def predict_next_hour_occupancy(parking):
    artifact = load_parking_occupancy_model()
    if not artifact:
        return None
    parking_id = parking.get("parking_id")
    capacity = parking.get("capacity")
    source = parking.get("source")
    if parking_id is None or capacity is None:
        return None
    try:
        capacity = int(float(capacity))
    except (TypeError, ValueError):
        return None
    if capacity <= 0:
        return None
    history = get_parking_prediction_history(parking_id, 180)
    if len(history) < 169:
        return None
    features = create_prediction_features(history, source)
    latest = features.iloc[-1]
    columns = artifact.get("features", [])
    if not columns or any(c not in latest.index for c in columns):
        return None
    X = pd.DataFrame([[latest[c] for c in columns]], columns=columns)
    if X.isna().any().any():
        return None
    try:
        prediction = float(artifact["model"].predict(X)[0])
    except Exception:
        return None
    prediction = float(np.clip(prediction, 0, 100))
    predicted_occupied = int(round(capacity * prediction / 100))
    predicted_occupied = max(0, min(capacity, predicted_occupied))
    return {
        "predicted_occupancy_pct": round(prediction, 1),
        "predicted_occupied_spaces": predicted_occupied,
        "predicted_available_spaces": capacity - predicted_occupied,
    }

def attach_predictions(parking_rows):
    updated=[]
    for parking in parking_rows:
        row=dict(parking)
        pred=predict_next_hour_occupancy(row)
        if pred:
            row.update(pred)
        else:
            row["predicted_occupancy_pct"]=None
            row["predicted_occupied_spaces"]=None
            row["predicted_available_spaces"]=None
        updated.append(row)
    return updated


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in kilometres."""

    import math

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

    return radius * (
        2
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )


def get_fallback_nearby_parking(
    poi,
    parking_data,
    radius_km=1.5
):
    """
    Fallback when no explicit POI-parking mapping exists.

    Uses the real parking_locations table only.
    No parking records are fabricated.
    """

    nearby = []

    for parking in parking_data:

        if (
            parking.get("latitude") is None
            or parking.get("longitude") is None
        ):
            continue

        distance = haversine_km(
            poi["latitude"],
            poi["longitude"],
            parking["latitude"],
            parking["longitude"]
        )

        if distance <= radius_km:

            record = dict(parking)

            record["distance_from_poi_km"] = round(
                distance,
                3
            )

            nearby.append(record)

    nearby.sort(
        key=lambda row:
        float(row["distance_from_poi_km"])
    )

    return nearby


def calculate_parking_recommendation(
    parking_rows
):
    """
    Rank real parking options.

    Current production-safe scoring:
        - distance is the primary factor
        - known capacity receives a small tie-break advantage

    We do NOT claim live occupancy because the current
    parking database does not contain live occupancy.
    """

    if not parking_rows:
        return []

    max_distance = max(
        float(
            row.get(
                "distance_from_poi_km",
                0
            )
        )
        for row in parking_rows
    )

    ranked = []

    for row in parking_rows:

        distance = float(
            row.get(
                "distance_from_poi_km",
                0
            )
        )

        if max_distance > 0:

            distance_score = (
                1
                -
                distance / max_distance
            ) * 90

        else:

            distance_score = 90.0


        capacity = row.get(
            "capacity"
        )

        # Capacity is only a tie-breaker because
        # capacity != current availability.
        capacity_bonus = (
            10.0
            if capacity is not None
            else 0.0
        )


        row_copy = dict(row)

        row_copy[
            "parking_score"
        ] = round(
            distance_score
            + capacity_bonus,
            2
        )

        ranked.append(
            row_copy
        )


    ranked.sort(
        key=lambda row:
        (
            -float(
                row["parking_score"]
            ),
            float(
                row[
                    "distance_from_poi_km"
                ]
            )
        )
    )


    for rank, row in enumerate(
        ranked,
        start=1
    ):

        row[
            "parking_rank"
        ] = rank


    return ranked


# ============================================================
# CONGESTION
# ============================================================

def calculate_congestion(row):

    current_speed = row.get("current_speed")
    free_flow_speed = row.get("free_flow_speed")

    if (
        current_speed is None
        or free_flow_speed is None
        or float(free_flow_speed) <= 0
    ):
        return 0.0

    congestion = (
        (
            float(free_flow_speed)
            - float(current_speed)
        )
        / float(free_flow_speed)
    ) * 100

    return max(
        0.0,
        min(100.0, congestion)
    )


def congestion_color(value):

    if value < 20:
        return "green"

    if value < 50:
        return "orange"

    if value < 70:
        return "red"

    return "darkred"


def congestion_label(value):

    if value < 20:
        return "LOW"

    if value < 50:
        return "MODERATE"

    if value < 70:
        return "HIGH"

    return "SEVERE"


# ============================================================
# WEATHER LABEL
# ============================================================

def weather_label(risk):

    if risk <= 20:
        return "LOW"

    if risk <= 40:
        return "MODERATE"

    if risk <= 70:
        return "HIGH"

    return "SEVERE"


# ============================================================
# LOAD DATA
# ============================================================

try:

    traffic_data = (
        get_latest_traffic_weather()
    )

    parking_data = (
        get_parking_locations()
    )

except Exception as exc:

    st.error(
        f"Database error: {exc}"
    )

    st.stop()


if not traffic_data:

    st.warning(
        "No traffic/weather data available."
    )

    st.stop()


df = pd.DataFrame(
    traffic_data
)


# ============================================================
# CALCULATED TRAFFIC FIELDS
# ============================================================

df["congestion_pct"] = (
    df.apply(
        calculate_congestion,
        axis=1
    )
)

df["congestion_level"] = (
    df["congestion_pct"].apply(
        congestion_label
    )
)


# ============================================================
# TOP KPI CARDS
# ============================================================

avg_congestion = (
    df["congestion_pct"].mean()
)

avg_speed = (
    pd.to_numeric(
        df["current_speed"],
        errors="coerce"
    ).mean()
)

avg_delay = (
    pd.to_numeric(
        df["traffic_delay"],
        errors="coerce"
    ).mean()
)

active_road_closures = int(
    df["road_closure"]
    .fillna(False)
    .astype(bool)
    .sum()
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Avg Congestion",
        f"{avg_congestion:.1f}%"
    )

    st.caption(
        "8 monitored zones"
    )


with col2:

    st.metric(
        "Avg Speed",
        f"{avg_speed:.1f} km/h"
    )


with col3:

    st.metric(
        "Avg Delay",
        f"{avg_delay:.0f} sec"
    )


with col4:

    st.metric(
        "Active Road Closures",
        active_road_closures
    )


st.divider()


# ============================================================
# SIDEBAR — MAP LAYERS
# ============================================================

st.sidebar.header(
    "🗺️ Map Layers"
)


show_traffic = st.sidebar.checkbox(
    "🚦 Traffic",
    value=True
)


show_weather = st.sidebar.checkbox(
    "🌦 Weather",
    value=True
)


show_parking = st.sidebar.checkbox(
    "🅿️ Parking",
    value=True
)


show_routes = st.sidebar.checkbox(
    "🚗 Routes",
    value=True
)


# ============================================================
# SIDEBAR — ROUTE PLANNER
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "🚗 Route Planner"
)


location_ids = sorted(
    df["location_id"]
    .dropna()
    .unique()
    .tolist()
)


origin = st.sidebar.selectbox(
    "Start location",
    location_ids
)


destination_options = [
    location
    for location in location_ids
    if location != origin
]


destination = st.sidebar.selectbox(
    "Destination",
    destination_options
)


find_route = st.sidebar.button(
    "🚗 Find Best Route",
    use_container_width=True
)


refresh_data = st.sidebar.button(
    "🔄 Refresh Data",
    use_container_width=True
)


if refresh_data:

    st.cache_data.clear()

    st.session_state.pop(
        "evaluated_routes",
        None
    )

    st.session_state.pop(
        "route_congestion_predictions",
        None
    )

    st.rerun()


# ============================================================
# SIDEBAR — DESTINATION & PARKING PLANNER
# ============================================================

st.sidebar.divider()

st.sidebar.header(
    "📍 Destination & Parking"
)


poi_location = st.sidebar.selectbox(
    "1. Choose city location",
    location_ids,
    key="poi_location"
)


try:

    poi_categories = get_poi_categories(
        poi_location
    )

except Exception as exc:

    st.sidebar.error(
        f"Could not load POI categories: {exc}"
    )

    poi_categories = []


if poi_categories:

    poi_type = st.sidebar.selectbox(
        "2. Choose destination type",
        poi_categories,
        key="poi_type"
    )

else:

    poi_type = None

    st.sidebar.warning(
        "No POI categories available "
        "for this location."
    )


poi_records = []

if poi_type:

    try:

        poi_records = get_pois(
            poi_location,
            poi_type
        )

    except Exception as exc:

        st.sidebar.error(
            f"Could not load destinations: {exc}"
        )


if poi_records:

    poi_labels = [
        f"{poi['poi_name']} (ID {poi['id']})"
        for poi in poi_records
    ]

    selected_poi_label = st.sidebar.selectbox(
        "3. Choose exact destination",
        poi_labels,
        key="selected_poi_label"
    )

    selected_poi = poi_records[
        poi_labels.index(
            selected_poi_label
        )
    ]

else:

    selected_poi = None

    if poi_type:
        st.sidebar.info(
            "No destinations found."
        )


find_parking = st.sidebar.button(
    "🅿️ Find Nearby Parking",
    use_container_width=True
)


if find_parking:

    if selected_poi is None:

        st.sidebar.error(
            "Please select an exact destination first."
        )

    else:

        mapped_parking = (
            get_parking_for_poi(
                selected_poi["id"]
            )
        )

        if mapped_parking:

            nearby_parking = (
                calculate_parking_recommendation(
                    mapped_parking
                )
            )

            nearby_parking = attach_predictions(nearby_parking)

            st.session_state[
                "selected_poi"
            ] = selected_poi

            st.session_state[
                "nearby_parking"
            ] = nearby_parking

            st.session_state[
                "parking_source_type"
            ] = "POI → parking mapping"

        else:

            fallback_parking = (
                get_fallback_nearby_parking(
                    selected_poi,
                    parking_data,
                    radius_km=1.5
                )
            )

            nearby_parking = (
                calculate_parking_recommendation(
                    fallback_parking
                )
            )

            nearby_parking = attach_predictions(nearby_parking)

            st.session_state[
                "selected_poi"
            ] = selected_poi

            st.session_state[
                "nearby_parking"
            ] = nearby_parking

            st.session_state[
                "parking_source_type"
            ] = "Geographic proximity fallback"


# ============================================================
# MAP CENTER
# ============================================================

map_center = [

    df["latitude"].mean(),

    df["longitude"].mean()

]


city_map = folium.Map(

    location=map_center,

    zoom_start=12,

    control_scale=True

)


# ============================================================
# TRAFFIC LAYER
# ============================================================

if show_traffic:

    traffic_group = (
        folium.FeatureGroup(
            name="Traffic"
        )
    )

    for _, row in df.iterrows():

        congestion = float(
            row["congestion_pct"]
        )

        marker_color = (
            congestion_color(
                congestion
            )
        )

        popup_html = f"""
        <div style="width:250px">

            <h4>🚦 {row['location_id']}</h4>

            <b>Current speed:</b>
            {row['current_speed']} km/h
            <br>

            <b>Free-flow speed:</b>
            {row['free_flow_speed']} km/h
            <br>

            <b>Congestion:</b>
            {congestion:.2f}%
            <br>

            <b>Status:</b>
            {row['congestion_level']}
            <br>

            <b>Delay:</b>
            {row['traffic_delay']} sec
            <br><br>

            <b>Traffic updated:</b>
            {row['traffic_timestamp']}

        </div>
        """

        folium.CircleMarker(

            location=[
                float(row["latitude"]),
                float(row["longitude"])
            ],

            radius=9,

            color=marker_color,

            fill=True,

            fill_color=marker_color,

            fill_opacity=0.8,

            popup=folium.Popup(
                popup_html,
                max_width=320
            ),

            tooltip=(
                f"{row['location_id']} | "
                f"{congestion:.1f}% congestion"
            )

        ).add_to(
            traffic_group
        )


    traffic_group.add_to(
        city_map
    )


# ============================================================
# WEATHER LAYER
# ============================================================

if show_weather:

    weather_group = (
        folium.FeatureGroup(
            name="Weather"
        )
    )

    for _, row in df.iterrows():

        popup_html = f"""
        <div style="width:240px">

            <h4>🌦 {row['location_id']}</h4>

            <b>Condition:</b>
            {row['weather_main']}
            <br>

            <b>Description:</b>
            {row['weather_description']}
            <br>

            <b>Temperature:</b>
            {row['temperature']} °C
            <br>

            <b>Humidity:</b>
            {row['humidity']} %
            <br>

            <b>Cloudiness:</b>
            {row['cloudiness']} %
            <br>

            <b>Rain:</b>
            {row['rain_mm']} mm
            <br><br>

            <b>Weather updated:</b>
            {row['weather_timestamp']}

        </div>
        """

        folium.Marker(

            location=[
                float(row["latitude"]),
                float(row["longitude"])
            ],

            icon=folium.Icon(
                color="blue",
                icon="cloud",
                prefix="fa"
            ),

            popup=folium.Popup(
                popup_html,
                max_width=320
            ),

            tooltip=(
                f"{row['location_id']} | "
                f"{row['weather_main']}"
            )

        ).add_to(
            weather_group
        )


    weather_group.add_to(
        city_map
    )


# ============================================================
# PARKING LAYER
# ============================================================

if show_parking and parking_data:

    parking_group = (
        folium.FeatureGroup(
            name="Parking"
        )
    )

    for parking in parking_data:

        capacity = parking.get(
            "capacity"
        )

        address = parking.get(
            "address"
        )

        popup_html = f"""
        <div style="width:270px">

            <h4>🅿️ {parking['parking_name']}</h4>

            <b>Type:</b>
            {parking['parking_type']}
            <br>

            <b>Capacity:</b>
            {
                capacity
                if capacity is not None
                else "Not mapped"
            }
            <br>

            <b>Address:</b>
            {
                address
                if address
                else "Not available"
            }
            <br>

            <b>Source:</b>
            {parking['source']}
            <br>

            <b>Verified:</b>
            {parking['last_verified_at']}

        </div>
        """

        folium.CircleMarker(

            location=[
                float(
                    parking["latitude"]
                ),
                float(
                    parking["longitude"]
                )
            ],

            radius=5,

            color="blue",

            fill=True,

            fill_color="blue",

            fill_opacity=0.7,

            popup=folium.Popup(
                popup_html,
                max_width=340
            ),

            tooltip=(
                parking["parking_name"]
            )

        ).add_to(
            parking_group
        )


    parking_group.add_to(
        city_map
    )


# ============================================================
# SELECTED DESTINATION + NEARBY PARKING
# ============================================================

selected_poi_for_map = (
    st.session_state.get(
        "selected_poi"
    )
)

nearby_parking_for_map = (
    st.session_state.get(
        "nearby_parking",
        []
    )
)


if selected_poi_for_map:

    destination_group = (
        folium.FeatureGroup(
            name="Selected Destination"
        )
    )

    poi_popup = f"""
    <div style="width:300px">

        <h4>📍 Selected Destination</h4>

        <b>Name:</b>
        {selected_poi_for_map['poi_name']}
        <br>

        <b>Type:</b>
        {selected_poi_for_map['poi_type']}
        <br>

        <b>Location:</b>
        {selected_poi_for_map['location_id']}
        <br>

        <b>Address:</b>
        {
            selected_poi_for_map.get("address")
            or "Not available"
        }

    </div>
    """

    folium.Marker(

        location=[
            float(
                selected_poi_for_map[
                    "latitude"
                ]
            ),
            float(
                selected_poi_for_map[
                    "longitude"
                ]
            )
        ],

        icon=folium.Icon(
            color="red",
            icon="flag",
            prefix="fa"
        ),

        popup=folium.Popup(
            poi_popup,
            max_width=340
        ),

        tooltip=(
            "📍 "
            +
            selected_poi_for_map[
                "poi_name"
            ]
        )

    ).add_to(
        destination_group
    )

    destination_group.add_to(
        city_map
    )


if nearby_parking_for_map:

    nearby_group = (
        folium.FeatureGroup(
            name="Nearby Parking"
        )
    )

    for parking in nearby_parking_for_map:

        rank = parking.get(
            "parking_rank",
            0
        )

        distance = float(
            parking.get(
                "distance_from_poi_km",
                0
            )
        )

        capacity = parking.get(
            "capacity"
        )

        capacity_text = (
            str(capacity)
            if capacity is not None
            else "Not mapped"
        )

        popup_html = f"""
        <div style="width:300px">

            <h4>🅿️ Parking #{rank}</h4>

            <b>Name:</b>
            {parking['parking_name']}
            <br>

            <b>Distance:</b>
            {distance:.2f} km
            <br>

            <b>Parking score:</b>
            {parking.get('parking_score', 0):.2f}/100
            <br>

            <b>Type:</b>
            {parking.get('parking_type', 'Not available')}
            <br>

            <b>Capacity:</b>
            {capacity_text}
            <br>

            <b>Address:</b>
            {
                parking.get("address")
                or "Not available"
            }
            <br>

            <b>Source:</b>
            {parking.get('source', 'Not available')}

        </div>
        """

        marker_color = (
            "green"
            if rank == 1
            else "blue"
        )

        folium.CircleMarker(

            location=[
                float(
                    parking["latitude"]
                ),
                float(
                    parking["longitude"]
                )
            ],

            radius=9 if rank == 1 else 7,

            color=marker_color,

            fill=True,

            fill_color=marker_color,

            fill_opacity=0.85,

            popup=folium.Popup(
                popup_html,
                max_width=350
            ),

            tooltip=(
                f"🅿️ #{rank} "
                f"{parking['parking_name']} | "
                f"{distance:.2f} km"
            )

        ).add_to(
            nearby_group
        )


    nearby_group.add_to(
        city_map
    )


# ============================================================
# ROUTING IMPORT
# ============================================================

def run_route_engine(
    origin,
    destination
):

    """
    Supports both routing versions we have used:
        evaluate_all_routes()
        evaluate_routes()
    """

    from graphhopper_routes_v2 import (
        find_routes
    )

    routes = find_routes(
        origin,
        destination
    )

    if not routes:
        return []


    try:

        from route_conditions_v2 import (
            evaluate_all_routes
        )

        return evaluate_all_routes(
            routes
        )

    except ImportError:

        from route_conditions_v2 import (
            evaluate_routes
        )

        return evaluate_routes(
            routes
        )


# ============================================================
# ROUTE SCORING FALLBACK
# ============================================================

def get_route_value(
    route,
    *keys,
    default=0.0
):

    for key in keys:

        value = route.get(key)

        if value is not None:

            try:
                return float(value)
            except (
                TypeError,
                ValueError
            ):
                continue

    return float(default)


def calculate_display_score(
    route,
    all_routes
):

    """
    Use the routing module's score when it is valid.

    If its score is zero/missing while traffic,
    weather and time scores are valid, calculate the
    same weighted score here for dashboard display.

    This prevents the dashboard from displaying 0
    merely because of a version mismatch between
    dashboard and routing module.
    """

    existing_score = get_route_value(
        route,
        "overall_score",
        default=0
    )

    traffic_score = get_route_value(
        route,
        "traffic_score",
        default=0
    )

    weather_score = get_route_value(
        route,
        "weather_score",
        default=0
    )

    time_score = get_route_value(
        route,
        "travel_time_score",
        "time_score",
        default=0
    )


    # --------------------------------------------------------
    # If routing module already supplied a meaningful score
    # --------------------------------------------------------

    if existing_score > 0:

        return round(
            existing_score,
            2
        )


    # --------------------------------------------------------
    # Otherwise calculate it from the component scores
    # --------------------------------------------------------

    calculated_score = (

        traffic_score
        * TRAFFIC_WEIGHT

        +

        weather_score
        * WEATHER_WEIGHT

        +

        time_score
        * TIME_WEIGHT

    )


    return round(
        max(
            0.0,
            min(
                100.0,
                calculated_score
            )
        ),
        2
    )


# ============================================================
# CONGESTION ML — STREAMLIT INTEGRATION
# ============================================================

def get_cached_location_predictions():
    """Return current and +30m/+60m congestion by monitored location."""
    return get_location_predictions()


def get_cached_route_congestion_predictions(
    origin,
    destination
):
    """Return route-specific current/+30m/+60m congestion predictions."""
    return get_route_congestion_predictions(
        origin,
        destination
    )


# ============================================================
# ROAD NAME EXTRACTION
# ============================================================

def get_instruction_road_name(
    instruction
):

    if not isinstance(
        instruction,
        dict
    ):
        return None


    possible_keys = [

        "street_name",

        "streetName",

        "name",

        "road_name",

        "roadName"

    ]


    for key in possible_keys:

        value = instruction.get(
            key
        )

        if value is not None:

            value = str(
                value
            ).strip()

            if value:

                return value


    return None


def get_route_road_names(
    route
):

    road_names = []

    instructions = route.get(
        "instructions",
        []
    )


    for instruction in instructions:

        road_name = (
            get_instruction_road_name(
                instruction
            )
        )

        if (
            road_name
            and road_name not in road_names
        ):

            road_names.append(
                road_name
            )


    return road_names


# ============================================================
# GET ROUTE GEOMETRY
# ============================================================

def get_route_points(
    route
):

    points = route.get(
        "geometry_points"
    )

    if points:

        return points


    geometry = route.get(
        "geometry"
    )


    if isinstance(
        geometry,
        dict
    ):

        coordinates = geometry.get(
            "coordinates",
            []
        )

        return coordinates


    return []


# ============================================================
# FIND ROUTE
# ============================================================

if find_route:

    with st.spinner(
        "Generating and evaluating routes..."
    ):

        try:

            evaluated_routes = (
                run_route_engine(
                    origin,
                    destination
                )
            )


            # ------------------------------------------------
            # Add ML route-specific congestion forecasting.
            # The existing GraphHopper/route-condition results
            # remain unchanged; ML fields are attached to them.
            # ------------------------------------------------

            route_congestion_predictions = []

            if evaluated_routes:

                try:

                    route_congestion_predictions = (
                        get_route_congestion_predictions(
                            origin,
                            destination
                        )
                    )

                    prediction_lookup = {
                        str(route.get("route_id")): route
                        for route in route_congestion_predictions
                    }

                    for index, route in enumerate(
                        evaluated_routes
                    ):

                        prediction = prediction_lookup.get(
                            str(route.get("route_id"))
                        )

                        # Fallback to position when GraphHopper
                        # returns stable route ordering but an ID
                        # differs between the two calls.
                        if (
                            prediction is None
                            and
                            index < len(route_congestion_predictions)
                        ):
                            prediction = (
                                route_congestion_predictions[index]
                            )

                        if prediction is not None:

                            route[
                                "route_current_congestion"
                            ] = prediction.get(
                                "route_current_congestion"
                            )

                            route[
                                "route_predicted_congestion_30m"
                            ] = prediction.get(
                                "route_predicted_congestion_30m"
                            )

                            route[
                                "route_predicted_congestion_60m"
                            ] = prediction.get(
                                "route_predicted_congestion_60m"
                            )

                            route[
                                "route_max_predicted_30m"
                            ] = prediction.get(
                                "route_max_predicted_30m"
                            )

                            route[
                                "route_max_predicted_60m"
                            ] = prediction.get(
                                "route_max_predicted_60m"
                            )

                            route[
                                "prediction_locations_count"
                            ] = prediction.get(
                                "prediction_locations_count",
                                0
                            )

                            route[
                                "prediction_coverage"
                            ] = prediction.get(
                                "prediction_coverage",
                                "UNKNOWN"
                            )

                            route[
                                "route_sensor_matches"
                            ] = prediction.get(
                                "route_sensor_matches",
                                []
                            )

                            route[
                                "route_location_predictions"
                            ] = prediction.get(
                                "route_location_predictions",
                                []
                            )

                    st.session_state[
                        "route_congestion_predictions"
                    ] = route_congestion_predictions

                except Exception as prediction_error:

                    # Keep route planning functional even if the ML
                    # forecast layer is temporarily unavailable.
                    st.warning(
                        "Route congestion forecast is temporarily "
                        f"unavailable: {prediction_error}"
                    )

                    st.session_state.pop(
                        "route_congestion_predictions",
                        None
                    )


            if not evaluated_routes:

                st.error(
                    "No routes were returned."
                )

                st.session_state.pop(
                    "evaluated_routes",
                    None
                )

            else:

                # ------------------------------------------------
                # Calculate safe dashboard display score
                # ------------------------------------------------

                for route in evaluated_routes:

                    route[
                        "_display_score"
                    ] = calculate_display_score(
                        route,
                        evaluated_routes
                    )


                # ------------------------------------------------
                # Rank highest score first
                # ------------------------------------------------

                evaluated_routes = sorted(

                    evaluated_routes,

                    key=lambda route:
                        route[
                            "_display_score"
                        ],

                    reverse=True
                )


                # ------------------------------------------------
                # Store rank
                # ------------------------------------------------

                for rank, route in enumerate(
                    evaluated_routes,
                    start=1
                ):

                    route[
                        "_dashboard_rank"
                    ] = rank


                st.session_state[
                    "evaluated_routes"
                ] = evaluated_routes


                st.session_state[
                    "route_origin"
                ] = origin


                st.session_state[
                    "route_destination"
                ] = destination


        except Exception as exc:

            st.error(
                f"Route calculation failed: {exc}"
            )

            st.session_state.pop(
                "evaluated_routes",
                None
            )


# ============================================================
# GET STORED ROUTES
# ============================================================

evaluated_routes = (
    st.session_state.get(
        "evaluated_routes"
    )
)


# ============================================================
# DRAW ROUTES
# ============================================================

if (
    evaluated_routes
    and show_routes
):

    route_group = (
        folium.FeatureGroup(
            name="Routes"
        )
    )


    best_route = (
        evaluated_routes[0]
    )


    # --------------------------------------------------------
    # Draw alternatives first
    # --------------------------------------------------------

    for route in evaluated_routes:

        if (
            route["route_id"]
            ==
            best_route["route_id"]
        ):
            continue


        points = get_route_points(
            route
        )


        if not points:
            continue


        folium.PolyLine(

            locations=[
                [
                    float(point[1]),
                    float(point[0])
                ]

                for point in points

                if len(point) >= 2
            ],

            color="gray",

            weight=5,

            opacity=0.45,

            tooltip=(

                f"Alternative Route "
                f"{route['route_id']}"

                f" | "

                f"{route['distance_km']:.2f} km"

                f" | "

                f"{route['duration_minutes']:.1f} min"

            )

        ).add_to(
            route_group
        )


    # --------------------------------------------------------
    # Draw BEST route
    # --------------------------------------------------------

    best_points = get_route_points(
        best_route
    )


    if best_points:

        road_names = (
            get_route_road_names(
                best_route
            )
        )


        road_text = (
            " → ".join(
                road_names
            )
            if road_names
            else
            "Recommended route"
        )


        folium.PolyLine(

            locations=[
                [
                    float(point[1]),
                    float(point[0])
                ]

                for point in best_points

                if len(point) >= 2
            ],

            color="green",

            weight=9,

            opacity=0.95,

            tooltip=(
                "🏆 Recommended Route"
                "<br>"
                f"{road_text}"
            ),

            popup=folium.Popup(

                f"""
                <div style="width:300px">

                    <h4>🏆 Recommended Route</h4>

                    <b>Distance:</b>
                    {best_route['distance_km']:.2f} km
                    <br>

                    <b>ETA:</b>
                    {best_route['duration_minutes']:.1f} min
                    <br>

                    <b>Current Congestion:</b>
                    {best_route.get('route_current_congestion', 'Unavailable')}%
                    <br>

                    <b>Predicted +30 min:</b>
                    {best_route.get('route_predicted_congestion_30m', 'Unavailable')}%
                    <br>

                    <b>Predicted +60 min:</b>
                    {best_route.get('route_predicted_congestion_60m', 'Unavailable')}%
                    <br><br>

                    <b>Overall Score:</b>
                    {best_route['_display_score']:.2f}/100
                    <br><br>

                    <b>Roads:</b><br>
                    {road_text}

                </div>
                """,

                max_width=350
            )

        ).add_to(
            route_group
        )


    route_group.add_to(
        city_map
    )


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(
    city_map
)


# ============================================================
# MAP
# ============================================================

st.subheader(
    "📍 Live Kolkata Mobility Map"
)


st_folium(

    city_map,

    width=None,

    height=650,

    returned_objects=[]

)



# ============================================================
# ENERGY INTELLIGENCE — DATABASE + ML
# ============================================================

ENERGY_MODEL_DIR = PROJECT_ROOT / "models" / "energy"


def _load_energy_artifact(candidates, default_features=None):
    """Load a compatible energy model artifact.

    Prefer artifacts that contain their saved feature list. This is
    important because an older plain .pkl model can otherwise be found
    first and the dashboard would have no way to construct its input.
    """

    fallback = None

    for filename in candidates:
        path = ENERGY_MODEL_DIR / filename
        if not path.exists():
            continue

        try:
            artifact = joblib.load(path)

            if isinstance(artifact, dict) and "model" in artifact:
                artifact["_path"] = str(path)

                saved_features = artifact.get("features", [])

                if saved_features:
                    return artifact

                # Keep as fallback, but do not choose it if another
                # compatible artifact contains the feature list.
                if default_features:
                    artifact["features"] = default_features.copy()
                    return artifact

                fallback = artifact
                continue

            # Plain model artifact. Keep it only as a last resort.
            plain = {
                "model": artifact,
                "features": default_features.copy() if default_features else [],
                "_path": str(path),
            }

            if plain["features"]:
                return plain

            fallback = plain

        except Exception:
            continue

    return fallback


# Exact feature order used by the electricity-demand models.
ELECTRICITY_DEMAND_FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "demand_lag_1",
    "demand_lag_5",
    "demand_lag_15",
    "demand_lag_30",
    "demand_lag_60",
    "solar_lag_60",
    "ev_lag_60",
    "demand_rolling_mean_15",
    "demand_rolling_mean_30",
    "demand_rolling_mean_60",
    "demand_std_60",
    "solar_generation_mw",
    "ev_charging_demand_mw",
    "transformer_load_pct",
]


@st.cache_resource
def load_energy_models():
    return {
        "demand": _load_energy_artifact(
            [
                # Prefer the current model artifacts that store
                # their feature list.
                "energy_demand_xgboost.pkl",
                "energy_demand_lightgbm.pkl",
                "energy_demand_random_forest.pkl",
                "energy_demand_model.pkl",
            ],
            default_features=ELECTRICITY_DEMAND_FEATURES,
        ),
        "peak": _load_energy_artifact([
            "peak_load_lightgbm.pkl",
            "peak_load_xgboost.pkl",
            "peak_load_classifier.pkl",
        ]),
        "ev": _load_energy_artifact([
            "ev_demand_final_model.pkl",
            "ev_demand_random_forest.pkl",
            "ev_demand_xgboost.pkl",
            "ev_demand_lightgbm.pkl",
            "ev_charging_demand_random_forest.pkl",
        ]),
    }


@st.cache_data(ttl=30)
def get_energy_data():
    query = text("""
        SELECT
            timestamp,
            location_id,
            electricity_demand_mw,
            solar_generation_mw,
            ev_charging_demand_mw,
            renewable_percentage,
            transformer_load_pct,
            streetlights_online,
            streetlights_total,
            power_outage
        FROM energy_data
        ORDER BY timestamp
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    if not rows:
        return pd.DataFrame()

    data = pd.DataFrame([dict(r) for r in rows])
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data


def create_city_energy_data(energy_df):
    if energy_df.empty:
        return pd.DataFrame()

    city = (
        energy_df.groupby("timestamp")
        .agg(
            electricity_demand_mw=("electricity_demand_mw", "sum"),
            solar_generation_mw=("solar_generation_mw", "sum"),
            ev_charging_demand_mw=("ev_charging_demand_mw", "sum"),
            renewable_percentage=("renewable_percentage", "mean"),
            transformer_load_pct=("transformer_load_pct", "mean"),
            streetlights_online=("streetlights_online", "sum"),
            streetlights_total=("streetlights_total", "sum"),
            power_outage=("power_outage", "sum"),
        )
        .sort_index()
    )

    return city


def create_energy_features(city):
    """Build the feature families used by the energy forecasting models."""
    df = city.copy()

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)

    # Electricity-demand features
    for lag in [1, 5, 15, 30, 60]:
        df[f"demand_lag_{lag}"] = df["electricity_demand_mw"].shift(lag)

    df["solar_lag_60"] = df["solar_generation_mw"].shift(60)
    df["ev_lag_60"] = df["ev_charging_demand_mw"].shift(60)

    previous_demand = df["electricity_demand_mw"].shift(1)
    df["demand_rolling_mean_15"] = previous_demand.rolling(15).mean()
    df["demand_rolling_mean_30"] = previous_demand.rolling(30).mean()
    df["demand_rolling_mean_60"] = previous_demand.rolling(60).mean()
    df["demand_std_60"] = previous_demand.rolling(60).std()

    # EV-demand features
    for lag in [1, 5, 15, 30, 60, 120, 180, 360, 1440]:
        df[f"ev_demand_lag_{lag}"] = df["ev_charging_demand_mw"].shift(lag)

    df["electricity_demand_lag_60"] = df["electricity_demand_mw"].shift(60)
    df["electricity_demand_lag_1440"] = df["electricity_demand_mw"].shift(1440)
    df["solar_lag_1440"] = df["solar_generation_mw"].shift(1440)

    df["ev_rolling_mean_15"] = df["ev_charging_demand_mw"].rolling(15).mean()
    df["ev_rolling_mean_30"] = df["ev_charging_demand_mw"].rolling(30).mean()
    df["ev_rolling_mean_60"] = df["ev_charging_demand_mw"].rolling(60).mean()
    df["ev_rolling_std_60"] = df["ev_charging_demand_mw"].rolling(60).std()
    df["ev_change_15"] = df["ev_charging_demand_mw"] - df["ev_charging_demand_mw"].shift(15)
    df["ev_change_60"] = df["ev_charging_demand_mw"] - df["ev_charging_demand_mw"].shift(60)

    return df


def _model_input(features_df, artifact):
    """Return a one-row model input using the artifact's saved feature order."""
    features = artifact.get("features", [])
    if not features:
        return None
    missing = [c for c in features if c not in features_df.columns]
    if missing:
        return None
    valid = features_df[features].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    return valid.iloc[[-1]]


@st.cache_data(ttl=30)
def get_energy_forecasts():
    raw = get_energy_data()
    if raw.empty:
        return {}, pd.DataFrame()

    city = create_city_energy_data(raw)
    if city.empty:
        return {}, city

    features = create_energy_features(city)
    models = load_energy_models()

    result = {
        "timestamp": city.index[-1],
        "current_demand": float(city["electricity_demand_mw"].iloc[-1]),
        "current_solar": float(city["solar_generation_mw"].iloc[-1]),
        "current_ev": float(city["ev_charging_demand_mw"].iloc[-1]),
        "renewable": float(city["renewable_percentage"].iloc[-1]),
        "transformer": float(city["transformer_load_pct"].iloc[-1]),
        "streetlights_online": int(city["streetlights_online"].iloc[-1]),
        "streetlights_total": int(city["streetlights_total"].iloc[-1]),
        "outages": int(city["power_outage"].iloc[-1]),
        "predicted_demand": None,
        "predicted_ev": None,
        "peak_status": None,
        "peak_probability": None,
        "peak_threshold": None,
        "demand_model": None,
        "demand_error": None,
        "ev_model": None,
        "peak_model": None,
    }

    # Next-hour electricity demand
    demand_artifact = models["demand"]
    if demand_artifact:
        X = _model_input(features, demand_artifact)
        if X is not None:
            try:
                result["predicted_demand"] = max(
                    0.0,
                    float(demand_artifact["model"].predict(X)[0])
                )
                result["demand_model"] = Path(demand_artifact["_path"]).stem
            except Exception as demand_error:
                result["demand_error"] = str(demand_error)

    # Next-hour EV charging demand
    ev_artifact = models["ev"]
    if ev_artifact:
        X = _model_input(features, ev_artifact)
        if X is not None:
            try:
                result["predicted_ev"] = max(
                    0.0,
                    float(ev_artifact["model"].predict(X)[0])
                )
                result["ev_model"] = Path(ev_artifact["_path"]).stem
            except Exception:
                pass

    # Next-hour peak-load classification
    peak_artifact = models["peak"]
    if peak_artifact:
        X = _model_input(features, peak_artifact)
        if X is not None:
            try:
                model = peak_artifact["model"]
                prediction = int(model.predict(X)[0])
                result["peak_status"] = "PEAK" if prediction == 1 else "NORMAL"

                if hasattr(model, "predict_proba"):
                    result["peak_probability"] = float(model.predict_proba(X)[0][1] * 100)

                result["peak_threshold"] = peak_artifact.get("peak_threshold_mw")
                result["peak_model"] = Path(peak_artifact["_path"]).stem
            except Exception:
                pass

    return result, city


# ============================================================
# ENERGY INTELLIGENCE DASHBOARD
# ============================================================

try:
    energy_forecast, energy_city_df = get_energy_forecasts()
except Exception as energy_error:
    energy_forecast = {}
    energy_city_df = pd.DataFrame()
    st.warning(f"Energy module temporarily unavailable: {energy_error}")


if not energy_city_df.empty:

    st.divider()
    st.header("⚡ Energy Intelligence")
    st.caption(
        "Live city-level energy monitoring with next-hour electricity-demand, "
        "peak-load and EV charging forecasts."
    )

    current_demand = energy_forecast.get("current_demand")
    predicted_demand = energy_forecast.get("predicted_demand")
    current_solar = energy_forecast.get("current_solar")
    current_ev = energy_forecast.get("current_ev")
    renewable = energy_forecast.get("renewable")
    transformer = energy_forecast.get("transformer")
    predicted_ev = energy_forecast.get("predicted_ev")
    peak_status = energy_forecast.get("peak_status")
    peak_probability = energy_forecast.get("peak_probability")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "⚡ Current Demand",
            f"{current_demand:.2f} MW" if current_demand is not None else "Unavailable"
        )

    with c2:
        if predicted_demand is not None:
            st.metric(
                "🔮 Next-Hour Demand",
                f"{predicted_demand:.2f} MW",
                delta=f"{predicted_demand - current_demand:+.2f} MW"
            )
        else:
            st.metric("🔮 Next-Hour Demand", "Unavailable")

    with c3:
        st.metric(
            "☀️ Solar Generation",
            f"{current_solar:.2f} MW" if current_solar is not None else "Unavailable"
        )

    with c4:
        st.metric(
            "🔋 EV Charging",
            f"{current_ev:.2f} MW" if current_ev is not None else "Unavailable"
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("🌱 Renewable", f"{renewable:.1f}%" if renewable is not None else "Unavailable")

    with c6:
        st.metric("🔌 Transformer Load", f"{transformer:.1f}%" if transformer is not None else "Unavailable")

    with c7:
        total = energy_forecast.get("streetlights_total", 0)
        online = energy_forecast.get("streetlights_online", 0)
        light_pct = (online / total * 100) if total else 0
        st.metric("💡 Streetlights Online", f"{light_pct:.1f}%")

    with c8:
        st.metric("🚨 Power Outages", str(energy_forecast.get("outages", 0)))

    st.markdown("### 🤖 Energy ML Predictions")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("#### ⚡ Electricity Demand")
        if predicted_demand is not None:
            st.metric("Next Hour", f"{predicted_demand:.2f} MW")
            st.caption(f"Model: {energy_forecast.get('demand_model', 'Unknown')}")
        else:
            st.info("Demand forecast unavailable.")
            if energy_forecast.get("demand_error"):
                st.caption(
                    "Demand model error: "
                    f"{energy_forecast['demand_error']}"
                )

    with p2:
        st.markdown("#### 🚨 Peak Load")
        if peak_status == "PEAK":
            st.error("🔴 PEAK LOAD EXPECTED")
        elif peak_status == "NORMAL":
            st.success("🟢 NORMAL LOAD")
        else:
            st.info("Peak-load prediction unavailable.")

        if peak_probability is not None:
            st.metric("Peak Probability", f"{peak_probability:.1f}%")

        threshold = energy_forecast.get("peak_threshold")
        if threshold is not None:
            st.caption(f"Peak threshold: {float(threshold):.2f} MW")
        if energy_forecast.get("peak_model"):
            st.caption(f"Model: {energy_forecast['peak_model']}")

    with p3:
        st.markdown("#### 🔋 EV Charging Demand")
        if predicted_ev is not None:
            st.metric("Next Hour", f"{predicted_ev:.2f} MW")
            st.caption(f"Model: {energy_forecast.get('ev_model', 'Unknown')}")
        else:
            st.info("EV demand forecast unavailable.")

    st.markdown("### 📈 Energy Trends")

    trend = energy_city_df[[
        "electricity_demand_mw",
        "solar_generation_mw",
        "ev_charging_demand_mw",
    ]].tail(720).rename(columns={
        "electricity_demand_mw": "Electricity Demand (MW)",
        "solar_generation_mw": "Solar Generation (MW)",
        "ev_charging_demand_mw": "EV Charging (MW)",
    })

    st.line_chart(trend, use_container_width=True)

    trend2 = energy_city_df[[
        "transformer_load_pct",
        "renewable_percentage",
    ]].tail(720).rename(columns={
        "transformer_load_pct": "Transformer Load (%)",
        "renewable_percentage": "Renewable (%)",
    })

    st.line_chart(trend2, use_container_width=True)

    st.caption(
        f"Energy data updated through {energy_forecast.get('timestamp')} | "
        "Energy models use the same feature definitions used during training."
    )



# ============================================================
# LOCATION-WISE CONGESTION FORECAST
# ============================================================

@st.fragment(run_every="30s")
def render_location_forecast():

    st.divider()

    st.subheader(
        "🔮 Location-wise Congestion Forecast"
    )

    st.caption(
        "ML forecast at each monitored traffic location. "
        "+30 min and +60 min use the fine-tuned congestion models."
    )

    try:

        # Fetch fresh location predictions from the database.
        # This function is intentionally not cached because traffic
        # ingestion changes the underlying data continuously.
        location_predictions = (
            get_cached_location_predictions()
        )

        location_forecast_df = location_predictions.copy()

        # Round current congestion
        location_forecast_df[
            "current_congestion"
        ] = location_forecast_df[
            "current_congestion"
        ].round(2)

        # Round +30 minute prediction
        location_forecast_df[
            "predicted_congestion_30m"
        ] = location_forecast_df[
            "predicted_congestion_30m"
        ].round(2)

        # Round +60 minute prediction
        location_forecast_df[
            "predicted_congestion_60m"
        ] = location_forecast_df[
            "predicted_congestion_60m"
        ].round(2)

        # Rename columns for dashboard display
        location_forecast_df = (
            location_forecast_df.rename(
                columns={
                    "location_id": "Location",
                    "current_congestion": "Current (%)",
                    "predicted_congestion_30m": "+30 min (%)",
                    "predicted_congestion_60m": "+60 min (%)",
                }
            )
        )

        # Display the latest forecast
        st.dataframe(
            location_forecast_df[
                [
                    "Location",
                    "Current (%)",
                    "+30 min (%)",
                    "+60 min (%)",
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    except Exception as forecast_error:

        st.warning(
            "Location congestion forecast is temporarily unavailable: "
            f"{forecast_error}"
        )


# Automatically refresh ONLY this forecast section every 30 seconds.
render_location_forecast()


# ============================================================
# NEARBY PARKING RESULTS
# ============================================================

selected_poi_for_results = (
    st.session_state.get(
        "selected_poi"
    )
)

nearby_parking = (
    st.session_state.get(
        "nearby_parking",
        []
    )
)

parking_source_type = (
    st.session_state.get(
        "parking_source_type",
        ""
    )
)


if selected_poi_for_results:

    st.divider()

    st.subheader(
        "🅿️ Nearby Parking"
    )

    st.write(
        "Destination: "
        f"**{selected_poi_for_results['poi_name']}**"
    )

    if parking_source_type:

        st.caption(
            f"Data source: {parking_source_type}"
        )


    if nearby_parking:

        # ----------------------------------------------------
        # BEST CURRENT RECOMMENDATION
        # ----------------------------------------------------

        best_parking = nearby_parking[0]

        st.success(
            "🏆 Recommended Parking: "
            f"**{best_parking['parking_name']}** "
            f"— "
            f"{float(best_parking['distance_from_poi_km']):.2f} km "
            "from the destination."
        )

        st.caption(
            "Current occupancy comes from the latest parking occupancy snapshot. "
            "The next-hour forecast is generated by the tuned parking ML model."
        )


        # ----------------------------------------------------
        # FULL PARKING TABLE
        # ----------------------------------------------------

        parking_rows = []

        for parking in nearby_parking:

            parking_rows.append({

                "Rank":
                    parking.get(
                        "parking_rank",
                        ""
                    ),

                "Parking":
                    parking["parking_name"],

                "Type":
                    parking.get(
                        "parking_type",
                        "Not available"
                    ),

                "Distance (km)":
                    round(
                        float(
                            parking[
                                "distance_from_poi_km"
                            ]
                        ),
                        2
                    ),

                "Parking Score":
                    round(
                        float(
                            parking.get(
                                "parking_score",
                                0
                            )
                        ),
                        2
                    ),

                "Capacity": (
                    int(parking["capacity"])
                    if parking.get("capacity") is not None
                    else "Unavailable"
                ),

                "Occupied": (
                    int(parking["occupied_spaces"])
                    if parking.get("occupied_spaces") is not None
                    else "Unavailable"
                ),

                "Available": (
                    int(parking["available_spaces"])
                    if parking.get("available_spaces") is not None
                    else "Unavailable"
                ),

                "Occupancy": (
                    f"{float(parking['occupancy_pct']):.1f}%"
                    if parking.get("occupancy_pct") is not None
                    else "Unavailable"
                ),

                "Predicted Occupancy": (
                    f"{float(parking['predicted_occupancy_pct']):.1f}%"
                    if parking.get("predicted_occupancy_pct") is not None
                    else "Unavailable"
                ),

                "Predicted Available": (
                    int(parking["predicted_available_spaces"])
                    if parking.get("predicted_available_spaces") is not None
                    else "Unavailable"
                ),

                "Address": (
                    parking.get("address")
                    or "Not available"
                )
            })


        parking_df = pd.DataFrame(
            parking_rows
        )


        st.dataframe(
            parking_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # DETAILED CARDS
        # ----------------------------------------------------

        st.markdown(
            "### Parking Details"
        )


        for parking in nearby_parking[:10]:

            rank = parking.get(
                "parking_rank",
                ""
            )

            distance = float(
                parking[
                    "distance_from_poi_km"
                ]
            )

            capacity = (
                int(parking["capacity"])
                if parking.get("capacity") is not None
                else "Unavailable"
            )

            occupied = (
                int(parking["occupied_spaces"])
                if parking.get("occupied_spaces") is not None
                else "Unavailable"
            )

            available = (
                int(parking["available_spaces"])
                if parking.get("available_spaces") is not None
                else "Unavailable"
            )

            occupancy = (
                f"{float(parking['occupancy_pct']):.1f}%"
                if parking.get("occupancy_pct") is not None
                else "Unavailable"
            )

            title = (
                f"#{rank} "
                f"{parking['parking_name']} "
                f"— {distance:.2f} km"
            )


            with st.expander(
                title,
                expanded=(rank == 1)
            ):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Distance",
                        f"{distance:.2f} km"
                    )

                with c2:

                    st.metric(
                        "Parking Score",
                        (
                            f"{float(parking.get('parking_score', 0)):.2f}"
                            "/100"
                        )
                    )

                with c3:

                    st.metric(
                        "Capacity",
                        str(capacity)
                    )

                c4, c5, c6 = st.columns(3)

                with c4:

                    st.metric(
                        "Occupied",
                        str(occupied)
                    )

                with c5:

                    st.metric(
                        "Available",
                        str(available)
                    )

                with c6:

                    st.metric(
                        "Occupancy",
                        str(occupancy)
                    )

                prediction = predict_next_hour_occupancy(parking)

                st.markdown("#### 🔮 Next-Hour Forecast")

                if prediction is not None:
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.metric("Predicted Occupancy", f"{prediction['predicted_occupancy_pct']:.1f}%")
                    with p2:
                        st.metric("Predicted Occupied", str(prediction['predicted_occupied_spaces']))
                    with p3:
                        st.metric("Predicted Available", str(prediction['predicted_available_spaces']))
                else:
                    st.info("Next-hour forecast is unavailable for this parking lot.")


                st.write(
                    f"**Type:** "
                    f"{parking.get('parking_type', 'Not available')}"
                )

                st.write(
                    f"**Address:** "
                    f"{parking.get('address') or 'Not available'}"
                )


    else:

        st.warning(
            "No real parking facility was found "
            "within the current 1.5 km fallback search radius."
        )


# ============================================================
# ROUTE RESULTS — LIVE ML REFRESH
# ============================================================

@st.fragment(run_every="60s")
def render_route_results():
    """
    Render the stored route geometry while refreshing only the
    traffic-congestion ML values from the latest database data.

    Route geometry/road names remain stable. Current/+30m/+60m
    congestion are recalculated every 60 seconds from the latest
    traffic history, so new traffic ingestion is reflected without
    requiring the user to press "Find Best Route" again.
    """

    evaluated_routes = st.session_state.get(
        "evaluated_routes"
    )

    route_origin = st.session_state.get(
        "route_origin"
    )

    route_destination = st.session_state.get(
        "route_destination"
    )

    if not evaluated_routes:
        return

    # ------------------------------------------------------------
    # LIVE REFRESH OF ROUTE-SPECIFIC ML CONGESTION
    # ------------------------------------------------------------

    if route_origin and route_destination:
        try:
            fresh_predictions = get_route_congestion_predictions(
                route_origin,
                route_destination
            )

            prediction_lookup = {
                str(route.get("route_id")): route
                for route in fresh_predictions
            }

            for index, route in enumerate(evaluated_routes):
                prediction = prediction_lookup.get(
                    str(route.get("route_id"))
                )

                # GraphHopper can return a different route_id on a
                # later request even when route ordering is unchanged.
                if (
                    prediction is None
                    and index < len(fresh_predictions)
                ):
                    prediction = fresh_predictions[index]

                if prediction is None:
                    continue

                route["route_current_congestion"] = prediction.get(
                    "route_current_congestion"
                )

                route["route_predicted_congestion_30m"] = prediction.get(
                    "route_predicted_congestion_30m"
                )

                route["route_predicted_congestion_60m"] = prediction.get(
                    "route_predicted_congestion_60m"
                )

                route["route_max_predicted_30m"] = prediction.get(
                    "route_max_predicted_30m"
                )

                route["route_max_predicted_60m"] = prediction.get(
                    "route_max_predicted_60m"
                )

                route["prediction_locations_count"] = prediction.get(
                    "prediction_locations_count",
                    0
                )

                route["prediction_coverage"] = prediction.get(
                    "prediction_coverage",
                    "UNKNOWN"
                )

                route["route_sensor_matches"] = prediction.get(
                    "route_sensor_matches",
                    []
                )

                route["route_location_predictions"] = prediction.get(
                    "route_location_predictions",
                    []
                )

            # Save the refreshed congestion values back to session state.
            st.session_state["evaluated_routes"] = evaluated_routes
            st.session_state["route_congestion_predictions"] = fresh_predictions

        except Exception as refresh_error:
            st.warning(
                "Live route congestion refresh is temporarily unavailable: "
                f"{refresh_error}"
            )

    # ============================================================
    # EXISTING ROUTE DISPLAY
    # ============================================================

    if evaluated_routes:

        best_route = (
            evaluated_routes[0]
        )


        st.subheader(
            "🏆 Recommended Route"
        )


        # --------------------------------------------------------
        # TOP ROUTE METRICS
        # --------------------------------------------------------

        best_current_congestion = best_route.get(
            "route_current_congestion"
        )

        if best_current_congestion is None:
            best_current_congestion = get_route_value(
                best_route,
                "average_congestion_percentage",
                "congestion_percentage"
            )


        best_predicted_30 = best_route.get(
            "route_predicted_congestion_30m"
        )

        best_predicted_60 = best_route.get(
            "route_predicted_congestion_60m"
        )


        col1, col2, col3, col4, col5, col6 = (
            st.columns(6)
        )


        with col1:

            st.metric(
                "Distance",
                f"{best_route['distance_km']:.2f} km"
            )


        with col2:

            st.metric(
                "Current ETA",
                f"{best_route['duration_minutes']:.1f} min"
            )


        with col3:

            st.metric(
                "Current Congestion",
                f"{float(best_current_congestion):.1f}%"
                if best_current_congestion is not None
                else "Unavailable"
            )


        with col4:

            st.metric(
                "Predicted +30 min",
                f"{float(best_predicted_30):.1f}%"
                if best_predicted_30 is not None
                else "Unavailable"
            )


        with col5:

            st.metric(
                "Predicted +60 min",
                f"{float(best_predicted_60):.1f}%"
                if best_predicted_60 is not None
                else "Unavailable"
            )


        with col6:

            st.metric(
                "Overall Score",
                f"{best_route['_display_score']:.2f}/100"
            )


        st.success(
            "🏆 Recommended route selected"
        )


        # --------------------------------------------------------
        # ROAD-BY-ROAD DESCRIPTION
        # --------------------------------------------------------

        st.markdown(
            "### 🛣️ Road-by-Road Route"
        )


        road_names = (
            get_route_road_names(
                best_route
            )
        )


        if road_names:

            for index, road in enumerate(
                road_names,
                start=1
            ):

                st.write(
                    f"**{index}.** {road}"
                )


        else:

            st.warning(
                "GraphHopper did not provide named "
                "roads in the returned instructions "
                "for this route."
            )


        # --------------------------------------------------------
        # SCORE BREAKDOWN
        # --------------------------------------------------------

        st.markdown(
            "### 📊 Score Breakdown"
        )


        score1, score2, score3 = (
            st.columns(3)
        )


        with score1:

            st.metric(
                "Traffic",
                (
                    f"{get_route_value(best_route, 'traffic_score'):.2f}"
                    "/100"
                )
            )


        with score2:

            st.metric(
                "Weather",
                (
                    f"{get_route_value(best_route, 'weather_score'):.2f}"
                    "/100"
                )
            )


        with score3:

            time_score = get_route_value(
                best_route,
                "travel_time_score",
                "time_score"
            )

            st.metric(
                "Travel Time",
                f"{time_score:.2f}/100"
            )


        # --------------------------------------------------------
        # ML CONGESTION FORECAST
        # --------------------------------------------------------

        st.markdown(
            "### 🔮 Route Congestion Forecast"
        )


        forecast1, forecast2, forecast3, forecast4 = (
            st.columns(4)
        )


        with forecast1:

            st.metric(
                "Current Route Congestion",
                (
                    f"{float(best_current_congestion):.2f}%"
                    if best_current_congestion is not None
                    else "Unavailable"
                )
            )


        with forecast2:

            st.metric(
                "+30 min",
                (
                    f"{float(best_predicted_30):.2f}%"
                    if best_predicted_30 is not None
                    else "Unavailable"
                )
            )


        with forecast3:

            st.metric(
                "+60 min",
                (
                    f"{float(best_predicted_60):.2f}%"
                    if best_predicted_60 is not None
                    else "Unavailable"
                )
            )


        with forecast4:

            st.metric(
                "Sensors Used",
                str(
                    best_route.get(
                        "prediction_locations_count",
                        0
                    )
                )
            )


        if best_route.get(
            "route_location_predictions"
        ):

            with st.expander(
                "View sensor-level predictions used for this route"
            ):

                sensor_forecast_df = pd.DataFrame(
                    best_route[
                        "route_location_predictions"
                    ]
                )

                st.dataframe(
                    sensor_forecast_df,
                    use_container_width=True,
                    hide_index=True
                )


        # --------------------------------------------------------
        # CONDITION DETAILS
        # --------------------------------------------------------

        congestion = best_route.get(
            "route_current_congestion"
        )

        if congestion is None:
            congestion = get_route_value(
                best_route,
                "average_congestion_percentage",
                "congestion_percentage"
            )


        weather_risk = get_route_value(
            best_route,
            "average_weather_risk",
            "weather_risk"
        )


        st.write(
            f"**Average route congestion:** "
            f"{congestion:.2f}%"
        )


        st.write(
            f"**Average weather risk:** "
            f"{weather_risk:.1f}/100"
        )


        # --------------------------------------------------------
        # ROUTE COMPARISON
        # --------------------------------------------------------

        st.markdown(
            "### Route Comparison"
        )


        comparison_rows = []


        for route in evaluated_routes:

            route_roads = (
                get_route_road_names(
                    route
                )
            )


            road_summary = (
                " → ".join(
                    route_roads[:5]
                )
                if route_roads
                else
                "Road names unavailable"
            )


            comparison_rows.append({

                "Rank":
                    route["_dashboard_rank"],

                "Route":
                    route["route_id"],

                "Roads":
                    road_summary,

                "Distance (km)":
                    round(
                        float(
                            route["distance_km"]
                        ),
                        2
                    ),

                "ETA (min)":
                    round(
                        float(
                            route["duration_minutes"]
                        ),
                        1
                    ),

                "Current Congestion (%)":
                    round(
                        float(
                            route.get(
                                "route_current_congestion",
                                get_route_value(
                                    route,
                                    "average_congestion_percentage",
                                    "congestion_percentage"
                                )
                            )
                        ),
                        2
                    ),

                "Predicted +30m (%)": (
                    round(
                        float(
                            route[
                                "route_predicted_congestion_30m"
                            ]
                        ),
                        2
                    )
                    if route.get(
                        "route_predicted_congestion_30m"
                    ) is not None
                    else None
                ),

                "Predicted +60m (%)": (
                    round(
                        float(
                            route[
                                "route_predicted_congestion_60m"
                            ]
                        ),
                        2
                    )
                    if route.get(
                        "route_predicted_congestion_60m"
                    ) is not None
                    else None
                ),

                "Sensors":
                    route.get(
                        "prediction_locations_count",
                        0
                    ),

                "Traffic Score":
                    round(
                        get_route_value(
                            route,
                            "traffic_score"
                        ),
                        2
                    ),

                "Weather Score":
                    round(
                        get_route_value(
                            route,
                            "weather_score"
                        ),
                        2
                    ),

                "Time Score":
                    round(
                        get_route_value(
                            route,
                            "travel_time_score",
                            "time_score"
                        ),
                        2
                    ),

                "Overall Score":
                    round(
                        route[
                            "_display_score"
                        ],
                        2
                    )
            })


        comparison_df = pd.DataFrame(
            comparison_rows
        )


        st.dataframe(

            comparison_df,

            use_container_width=True,

            hide_index=True

        )


        # --------------------------------------------------------
        # ALTERNATIVE ROUTES
        # --------------------------------------------------------

        st.markdown(
            "### Alternative Routes"
        )


        for route in evaluated_routes[1:]:

            roads = (
                get_route_road_names(
                    route
                )
            )


            road_summary = (
                " → ".join(
                    roads
                )
                if roads
                else
                "Road names unavailable"
            )


            with st.expander(

                f"Route {route['route_id']} | "
                f"{route['distance_km']:.2f} km | "
                f"{route['duration_minutes']:.1f} min | "
                f"Score {route['_display_score']:.2f}"

            ):

                st.write(
                    f"**Roads:** {road_summary}"
                )


                current_route_congestion = route.get(
                    "route_current_congestion"
                )

                if current_route_congestion is None:
                    current_route_congestion = get_route_value(
                        route,
                        "average_congestion_percentage",
                        "congestion_percentage"
                    )


                st.write(
                    f"**Current congestion:** "
                    f"{float(current_route_congestion):.2f}%"
                )


                st.write(
                    f"**Predicted +30 min:** "
                    f"{route.get('route_predicted_congestion_30m', 'Unavailable')}%"
                )


                st.write(
                    f"**Predicted +60 min:** "
                    f"{route.get('route_predicted_congestion_60m', 'Unavailable')}%"
                )


                st.write(
                    f"**Sensors used:** "
                    f"{route.get('prediction_locations_count', 0)}"
                )


                st.write(
                    f"**Traffic Score:** "
                    f"{get_route_value(route, 'traffic_score'):.2f}/100"
                )


                st.write(
                    f"**Weather:** "
                    f"{get_route_value(route, 'weather_score'):.2f}/100"
                )


                st.write(
                    f"**Travel Time:** "
                    f"{get_route_value(route, 'travel_time_score', 'time_score'):.2f}/100"
                )


                st.write(
                    f"**Overall:** "
                    f"{route['_display_score']:.2f}/100"
                )




# Render route results and refresh congestion automatically.
render_route_results()


# ============================================================
# CURRENT DATA STATUS
# ============================================================

st.divider()


latest_traffic = (
    df["traffic_timestamp"]
    .max()
)


latest_weather = (
    df["weather_timestamp"]
    .max()
)


parking_model = load_parking_occupancy_model()
model_status = "Tuned ML model loaded" if parking_model is not None else "Parking ML model unavailable"

st.caption(
    "PostgreSQL live data | "
    f"Latest traffic: {latest_traffic} | "
    f"Latest weather: {latest_weather} | "
    f"Parking forecast: {model_status}"
)


st.caption(
    "Traffic/weather values come from the "
    "8 monitored Kolkata zones. "
    "Routes are generated using GraphHopper/OpenStreetMap."
)