# ============================================================
# SMART CITY OPERATIONS
# Route-Specific Traffic Congestion Prediction
#
# Uses the EXACT feature-engineering function from
# ml.congestion_model.py so training and inference stay aligned.
#
# Output per route:
#   - Current congestion
#   - Predicted +30 min congestion
#   - Predicted +60 min congestion
#   - GraphHopper ETA
#   - Sensors actually representing the route
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from database.connection import engine
from graphhopper_routes_v2 import find_routes
from ml.congestion_model import create_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_30_FILE = os.path.join(
    MODEL_DIR,
    "tuned_xgboost_congestion_30m.pkl"
)

MODEL_60_FILE = os.path.join(
    MODEL_DIR,
    "tuned_xgboost_congestion_60m.pkl"
)


# ============================================================
# ROUTE-SENSOR CONFIGURATION
# ============================================================

# A sensor can represent a route only when the sensor is
# physically close to the route geometry.
MAX_SENSOR_DISTANCE_KM = 0.50

# GraphHopper geometry points are dense. Sampling every 5th
# point gives sufficiently dense route coverage while keeping
# matching fast.
ROUTE_SAMPLE_STEP = 5


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Distance between two latitude/longitude points in km.
    """

    radius = 6371.0

    lat1 = np.radians(
        float(lat1)
    )

    lon1 = np.radians(
        float(lon1)
    )

    lat2 = np.radians(
        float(lat2)
    )

    lon2 = np.radians(
        float(lon2)
    )

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    a = (
        np.sin(d_lat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(d_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return float(
        radius * c
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not os.path.exists(
        MODEL_30_FILE
    ):
        raise FileNotFoundError(
            f"30-minute model not found:\n"
            f"{MODEL_30_FILE}"
        )

    if not os.path.exists(
        MODEL_60_FILE
    ):
        raise FileNotFoundError(
            f"60-minute model not found:\n"
            f"{MODEL_60_FILE}"
        )

    model_30_package = joblib.load(
        MODEL_30_FILE
    )

    model_60_package = joblib.load(
        MODEL_60_FILE
    )

    print(
        "\n30-minute model:",
        model_30_package.get(
            "model_name",
            "Unknown"
        )
    )

    print(
        "60-minute model:",
        model_60_package.get(
            "model_name",
            "Unknown"
        )
    )

    return (
        model_30_package,
        model_60_package
    )


# ============================================================
# GET RECENT TRAFFIC HISTORY
# ============================================================

def get_recent_history(
    rows_per_location=20
):
    """
    Load recent history for every monitored traffic location.
    """

    query = """
        WITH ranked AS (
            SELECT
                location_id,
                traffic_timestamp,

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

                ROW_NUMBER() OVER (
                    PARTITION BY location_id
                    ORDER BY traffic_timestamp DESC
                ) AS rn

            FROM traffic_weather_data
        )

        SELECT
            location_id,
            traffic_timestamp,

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
            frc

        FROM ranked

        WHERE rn <= %(n)s

        ORDER BY
            location_id,
            traffic_timestamp
    """

    history = pd.read_sql(
        query,
        engine,
        params={
            "n": rows_per_location
        }
    )

    if history.empty:
        raise RuntimeError(
            "No recent traffic history found."
        )

    return history


# ============================================================
# PREPARE DATA IN THE EXACT TRAINING SCHEMA
# ============================================================

def prepare_history_for_model(
    history
):
    """
    The uploaded congestion_model.py expects:
        Location ID
        Timestamp
        current_speed
        free_flow_speed
        current_travel_time
        free_flow_travel_time
        traffic_delay
        speed_reduction
        confidence
        road_closed
        road_class
        congestion_pct
    """

    df = history.copy()

    # --------------------------------------------------------
    # Rename database identifiers to training identifiers.
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "location_id":
                "Location ID",

            "traffic_timestamp":
                "Timestamp",

            "road_closure":
                "road_closed",

            "frc":
                "road_class",
        }
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Numeric traffic columns
    # --------------------------------------------------------

    numeric_columns = [
        "latitude",
        "longitude",
        "current_speed",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "traffic_delay",
        "speed_reduction",
        "confidence",
        "road_closed",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Road closed
    # --------------------------------------------------------

    if "road_closed" not in df.columns:
        df["road_closed"] = 0

    df["road_closed"] = (
        pd.to_numeric(
            df["road_closed"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Required traffic records
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Location ID",
            "Timestamp",
            "current_speed",
            "free_flow_speed",
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate congestion BEFORE create_features().
    #
    # This is exactly the target/current state definition used
    # by the uploaded training code.
    # --------------------------------------------------------

    df["congestion_pct"] = np.where(
        df["free_flow_speed"] > 0,

        (
            (
                df["free_flow_speed"]
                - df["current_speed"]
            )
            /
            df["free_flow_speed"]
        ) * 100,

        0
    )

    df["congestion_pct"] = (
        pd.to_numeric(
            df["congestion_pct"],
            errors="coerce"
        )
        .clip(0, 100)
    )

    df = df.dropna(
        subset=[
            "congestion_pct"
        ]
    )

    df = df.sort_values(
        [
            "Location ID",
            "Timestamp"
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE EXACT TRAINING FEATURES
# ============================================================

def build_location_features(
    history
):
    """
    Reuse the exact create_features() from the trained model.
    """

    prepared = (
        prepare_history_for_model(
            history
        )
    )

    # --------------------------------------------------------
    # This is the exact function used by training.
    # --------------------------------------------------------

    feature_data = create_features(
        prepared
    )

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Keep the newest observation at each location.
    # --------------------------------------------------------

    latest = (
        feature_data
        .sort_values(
            [
                "Location ID",
                "Timestamp"
            ]
        )
        .groupby(
            "Location ID",
            as_index=False
        )
        .tail(1)
        .copy()
    )

    return latest.reset_index(
        drop=True
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    feature_data,
    model_package
):
    """
    Match inference features exactly to the saved training
    feature order.
    """

    feature_columns = (
        model_package[
            "features"
        ]
    )

    X = feature_data.copy()

    # --------------------------------------------------------
    # Missing columns are added as zero.
    #
    # This matters if a road/location category is absent from
    # the current prediction batch.
    # --------------------------------------------------------

    for feature in feature_columns:

        if feature not in X.columns:

            X[feature] = 0

    X = X[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Use medians learned from TRAIN + VALIDATION.
    # --------------------------------------------------------

    medians = model_package.get(
        "train_validation_medians",
        model_package.get(
            "train_medians",
            {}
        )
    )

    for column, value in medians.items():

        if column in X.columns:

            X[column] = (
                X[column]
                .fillna(value)
            )

    # Final fallback.
    X = X.fillna(0)

    return X


# ============================================================
# LOCATION-LEVEL PREDICTIONS
# ============================================================

def get_location_predictions():

    package_30, package_60 = (
        load_models()
    )

    history = get_recent_history(
        rows_per_location=20
    )

    latest = build_location_features(
        history
    )

    # --------------------------------------------------------
    # 30-minute
    # --------------------------------------------------------

    X30 = prepare_model_input(
        latest,
        package_30
    )

    predictions_30 = (
        package_30[
            "model"
        ].predict(
            X30
        )
    )

    # --------------------------------------------------------
    # 60-minute
    # --------------------------------------------------------

    X60 = prepare_model_input(
        latest,
        package_60
    )

    predictions_60 = (
        package_60[
            "model"
        ].predict(
            X60
        )
    )

    latest[
        "predicted_congestion_30m"
    ] = np.clip(
        predictions_30,
        0,
        100
    )

    latest[
        "predicted_congestion_60m"
    ] = np.clip(
        predictions_60,
        0,
        100
    )

    latest[
        "current_congestion"
    ] = (
        latest[
            "congestion_pct"
        ]
        .clip(0, 100)
    )

    result = latest[
        [
            "Location ID",
            "current_congestion",
            "predicted_congestion_30m",
            "predicted_congestion_60m",
        ]
    ].copy()

    result = result.rename(
        columns={
            "Location ID":
                "location_id"
        }
    )

    return result


# ============================================================
# SENSOR LOCATIONS
# ============================================================

def get_sensor_locations():

    query = """
        SELECT DISTINCT ON (location_id)
            location_id,
            latitude,
            longitude
        FROM traffic_weather_data

        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL

        ORDER BY
            location_id,
            traffic_timestamp DESC
    """

    sensors = pd.read_sql(
        query,
        engine
    )

    if sensors.empty:
        raise RuntimeError(
            "No monitored traffic sensors found."
        )

    return sensors


# ============================================================
# MATCH ONE ROUTE TO TRAFFIC SENSORS
# ============================================================

def match_route_to_sensors(
    route,
    sensors
):
    """
    For every sampled GraphHopper route point, find the closest
    traffic sensor.

    A sensor is accepted only when it lies within
    MAX_SENSOR_DISTANCE_KM.

    Each sensor is kept once.

    segment_share represents the share of sampled route points
    assigned to that sensor. It is used as the route aggregation
    weight.
    """

    geometry_points = route.get(
        "geometry_points",
        []
    )

    if not geometry_points:
        return []

    sampled_points = (
        geometry_points[
            ::ROUTE_SAMPLE_STEP
        ]
    )

    if (
        geometry_points
        and geometry_points[-1]
        not in sampled_points
    ):

        sampled_points.append(
            geometry_points[-1]
        )

    assignments = []

    for point in sampled_points:

        if len(point) < 2:
            continue

        # GraphHopper format:
        # [longitude, latitude]
        route_lon = float(
            point[0]
        )

        route_lat = float(
            point[1]
        )

        best_sensor_id = None
        best_distance = float(
            "inf"
        )

        for _, sensor in sensors.iterrows():

            distance = haversine_distance(
                route_lat,
                route_lon,
                sensor["latitude"],
                sensor["longitude"]
            )

            if (
                distance
                < best_distance
            ):

                best_distance = distance

                best_sensor_id = (
                    sensor[
                        "location_id"
                    ]
                )

        if (
            best_sensor_id is not None
            and
            best_distance
            <= MAX_SENSOR_DISTANCE_KM
        ):

            assignments.append(
                {
                    "location_id":
                        best_sensor_id,

                    "distance_km":
                        best_distance,
                }
            )

    if not assignments:
        return []

    # --------------------------------------------------------
    # Unique sensors + route coverage share.
    # --------------------------------------------------------

    sensor_groups = {}

    for assignment in assignments:

        sensor_groups.setdefault(
            assignment[
                "location_id"
            ],
            []
        ).append(
            assignment
        )

    total_points = len(
        assignments
    )

    matches = []

    for location_id, group in (
        sensor_groups.items()
    ):

        distances = [
            x["distance_km"]
            for x in group
        ]

        point_count = len(
            group
        )

        matches.append(
            {
                "location_id":
                    location_id,

                "distance_km":
                    float(
                        min(
                            distances
                        )
                    ),

                "point_count":
                    point_count,

                "segment_share":
                    point_count
                    /
                    total_points,
            }
        )

    return sorted(
        matches,
        key=lambda x:
        x["segment_share"],
        reverse=True
    )


# ============================================================
# ROUTE AGGREGATION
# ============================================================

def aggregate_route_predictions(
    matches,
    location_predictions
):
    """
    Convert sensor-level predictions into route-level values.
    """

    if not matches:

        return {
            "current":
                None,

            "predicted_30m":
                None,

            "predicted_60m":
                None,

            "max_30m":
                None,

            "max_60m":
                None,

            "sensor_count":
                0,

            "coverage":
                "NO SENSOR COVERAGE",
        }

    lookup = (
        location_predictions
        .set_index(
            "location_id"
        )
        .to_dict(
            orient="index"
        )
    )

    usable = []

    for match in matches:

        location_id = match[
            "location_id"
        ]

        prediction = lookup.get(
            location_id
        )

        if prediction is None:
            continue

        usable.append(
            {
                "location_id":
                    location_id,

                "distance_km":
                    match[
                        "distance_km"
                    ],

                "segment_share":
                    match[
                        "segment_share"
                    ],

                "current":
                    float(
                        prediction[
                            "current_congestion"
                        ]
                    ),

                "predicted_30m":
                    float(
                        prediction[
                            "predicted_congestion_30m"
                        ]
                    ),

                "predicted_60m":
                    float(
                        prediction[
                            "predicted_congestion_60m"
                        ]
                    ),
            }
        )

    if not usable:

        return {
            "current":
                None,

            "predicted_30m":
                None,

            "predicted_60m":
                None,

            "max_30m":
                None,

            "max_60m":
                None,

            "sensor_count":
                0,

            "coverage":
                "NO SENSOR COVERAGE",
        }

    weights = np.asarray(
        [
            x["segment_share"]
            for x in usable
        ],
        dtype=float
    )

    current = np.asarray(
        [
            x["current"]
            for x in usable
        ],
        dtype=float
    )

    pred30 = np.asarray(
        [
            x["predicted_30m"]
            for x in usable
        ],
        dtype=float
    )

    pred60 = np.asarray(
        [
            x["predicted_60m"]
            for x in usable
        ],
        dtype=float
    )

    if weights.sum() <= 0:
        weights = np.ones_like(
            weights
        )

    weights = (
        weights
        /
        weights.sum()
    )

    return {
        "current":
            round(
                float(
                    np.average(
                        current,
                        weights=weights
                    )
                ),
                2
            ),

        "predicted_30m":
            round(
                float(
                    np.average(
                        pred30,
                        weights=weights
                    )
                ),
                2
            ),

        "predicted_60m":
            round(
                float(
                    np.average(
                        pred60,
                        weights=weights
                    )
                ),
                2
            ),

        "max_30m":
            round(
                float(
                    pred30.max()
                ),
                2
            ),

        "max_60m":
            round(
                float(
                    pred60.max()
                ),
                2
            ),

        "sensor_count":
            len(usable),

        "coverage":
            "COVERED",
    }


# ============================================================
# COMPLETE ROUTE-SPECIFIC PIPELINE
# ============================================================

def get_route_congestion_predictions(
    origin,
    destination
):

    # --------------------------------------------------------
    # 1. Generate genuine GraphHopper alternatives.
    # --------------------------------------------------------

    routes = find_routes(
        origin,
        destination
    )

    if not routes:
        return []

    # --------------------------------------------------------
    # 2. Get traffic sensor coordinates.
    # --------------------------------------------------------

    sensors = (
        get_sensor_locations()
    )

    # --------------------------------------------------------
    # 3. Generate location-level ML predictions.
    # --------------------------------------------------------

    location_predictions = (
        get_location_predictions()
    )

    # --------------------------------------------------------
    # 4. Convert location predictions into route predictions.
    # --------------------------------------------------------

    results = []

    for route in routes:

        matches = (
            match_route_to_sensors(
                route,
                sensors
            )
        )

        summary = (
            aggregate_route_predictions(
                matches,
                location_predictions
            )
        )

        result = dict(
            route
        )

        result[
            "route_current_congestion"
        ] = summary[
            "current"
        ]

        result[
            "route_predicted_congestion_30m"
        ] = summary[
            "predicted_30m"
        ]

        result[
            "route_predicted_congestion_60m"
        ] = summary[
            "predicted_60m"
        ]

        result[
            "route_max_predicted_30m"
        ] = summary[
            "max_30m"
        ]

        result[
            "route_max_predicted_60m"
        ] = summary[
            "max_60m"
        ]

        result[
            "prediction_locations_count"
        ] = summary[
            "sensor_count"
        ]

        result[
            "prediction_coverage"
        ] = summary[
            "coverage"
        ]

        # ----------------------------------------------------
        # Detailed sensor information.
        # This is useful for Streamlit drill-down.
        # ----------------------------------------------------

        location_lookup = (
            location_predictions
            .set_index(
                "location_id"
            )
            .to_dict(
                orient="index"
            )
        )

        details = []

        for match in matches:

            location_id = match[
                "location_id"
            ]

            prediction = (
                location_lookup.get(
                    location_id
                )
            )

            if prediction is None:
                continue

            details.append(
                {
                    "location_id":
                        location_id,

                    "distance_km":
                        round(
                            match[
                                "distance_km"
                            ],
                            3
                        ),

                    "segment_share_pct":
                        round(
                            match[
                                "segment_share"
                            ] * 100,
                            2
                        ),

                    "current_congestion":
                        round(
                            prediction[
                                "current_congestion"
                            ],
                            2
                        ),

                    "predicted_30m":
                        round(
                            prediction[
                                "predicted_congestion_30m"
                            ],
                            2
                        ),

                    "predicted_60m":
                        round(
                            prediction[
                                "predicted_congestion_60m"
                            ],
                            2
                        ),
                }
            )

        result[
            "route_location_predictions"
        ] = details

        results.append(
            result
        )

    return results


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def display_route_predictions(
    routes
):

    print("\n")
    print("=" * 100)

    print(
        "ROUTE-SPECIFIC CONGESTION PREDICTION"
    )

    print("=" * 100)

    for route in routes:

        print("\n")
        print(
            f"ROUTE {route['route_id']}"
        )

        print(
            "-" * 85
        )

        print(
            f"Distance              : "
            f"{route['distance_km']} km"
        )

        print(
            f"Current GraphHopper ETA: "
            f"{route['duration_minutes']} min"
        )

        print(
            f"Current congestion     : "
            f"{route['route_current_congestion']}%"
        )

        print(
            f"Predicted +30 min      : "
            f"{route['route_predicted_congestion_30m']}%"
        )

        print(
            f"Predicted +60 min      : "
            f"{route['route_predicted_congestion_60m']}%"
        )

        print(
            f"Max predicted +30 min  : "
            f"{route['route_max_predicted_30m']}%"
        )

        print(
            f"Max predicted +60 min  : "
            f"{route['route_max_predicted_60m']}%"
        )

        print(
            f"Sensor coverage        : "
            f"{route['prediction_coverage']}"
        )

        print(
            f"Unique sensors         : "
            f"{route['prediction_locations_count']}"
        )

        print(
            "\nSensors representing this route:"
        )

        for item in route.get(
            "route_location_predictions",
            []
        ):

            print(
                f"  {item['location_id']}"
                f" | distance={item['distance_km']:.3f} km"
                f" | segment={item['segment_share_pct']:.2f}%"
                f" | current={item['current_congestion']:.2f}%"
                f" | +30m={item['predicted_30m']:.2f}%"
                f" | +60m={item['predicted_60m']:.2f}%"
            )

    print("\n")
    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    locations = (
        pd.read_sql(
            """
            SELECT DISTINCT location_id
            FROM traffic_weather_data
            ORDER BY location_id
            """,
            engine
        )[
            "location_id"
        ]
        .tolist()
    )

    print(
        "\nAvailable locations:"
    )

    for location in locations:
        print(
            "-",
            location
        )

    origin = input(
        "\nEnter origin: "
    ).strip().upper()

    destination = input(
        "Enter destination: "
    ).strip().upper()

    if origin == destination:

        raise ValueError(
            "Origin and destination "
            "cannot be the same."
        )

    results = (
        get_route_congestion_predictions(
            origin,
            destination
        )
    )

    if not results:

        print(
            "\nNo route predictions available."
        )

    else:

        display_route_predictions(
            results
        )
