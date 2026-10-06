from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from sqlalchemy import text

from database.connection import engine


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "parking_occupancy_model.pkl"
)

# Forecast up to 72 hours
FORECAST_HOURS = 72

# Operational thresholds
THRESHOLD_90 = 90.0
THRESHOLD_FULL = 100.0


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    return artifact


# ============================================================
# CURRENT PARKING STATUS
# ============================================================

def get_current_parking():

    query = text("""
        SELECT
            pcs.parking_id,
            p.parking_name,
            p.parking_type,
            pcs.timestamp,
            pcs.capacity,
            pcs.occupied_spaces,
            pcs.available_spaces,
            pcs.occupancy_pct,
            pcs.source

        FROM parking_current_status pcs

        JOIN parking_locations p
            ON p.id = pcs.parking_id

        WHERE pcs.source IN (
            'S',
            'OpenStreetMap',
            'OpenStreetMap-Geofabrik'
        )

        ORDER BY pcs.parking_id;
    """)

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
# PARKING HISTORY
# ============================================================

def get_history(
    parking_id,
    hours=180
):

    query = text("""
        SELECT
            parking_id,
            timestamp,
            capacity,
            occupied_spaces,
            available_spaces,
            occupancy_pct

        FROM parking_occupancy_history

        WHERE parking_id = :parking_id

        ORDER BY timestamp DESC

        LIMIT :hours;
    """)

    with engine.connect() as conn:

        rows = (
            conn.execute(
                query,
                {
                    "parking_id": parking_id,
                    "hours": hours,
                }
            )
            .mappings()
            .all()
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [dict(row) for row in rows]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# EXACT SAME FEATURES USED BY DASHBOARD
# ============================================================

def create_prediction_features(
    history,
    source
):

    data = (
        history
        .sort_values("timestamp")
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["day_of_month"] = (
        data["timestamp"].dt.day
    )

    data["month"] = (
        data["timestamp"].dt.month
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclic
    # --------------------------------------------------------

    data["hour_sin"] = np.sin(
        2 * np.pi
        * data["hour"]
        / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi
        * data["hour"]
        / 24
    )

    data["dow_sin"] = np.sin(
        2 * np.pi
        * data["day_of_week"]
        / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi
        * data["day_of_week"]
        / 7
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        168,
    ]:

        data[
            f"lag_{lag}"
        ] = (
            data["occupancy_pct"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Changes
    # --------------------------------------------------------

    data["change_1h"] = (
        data["occupancy_pct"]
        - data["lag_1"]
    )

    data["change_3h"] = (
        data["occupancy_pct"]
        - data["lag_3"]
    )

    # --------------------------------------------------------
    # Rolling
    # --------------------------------------------------------

    shifted = (
        data["occupancy_pct"]
        .shift(1)
    )

    data["rolling_mean_3"] = (
        shifted
        .rolling(3)
        .mean()
    )

    data["rolling_mean_6"] = (
        shifted
        .rolling(6)
        .mean()
    )

    data["rolling_mean_24"] = (
        shifted
        .rolling(24)
        .mean()
    )

    data["rolling_std_24"] = (
        shifted
        .rolling(24)
        .std()
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    data["is_s_source"] = (
        1 if source == "S" else 0
    )

    return data


# ============================================================
# ONE-HOUR FORECAST
# ============================================================

def predict_one_hour(
    working_history,
    source,
    artifact
):

    data = create_prediction_features(
        working_history,
        source
    )

    latest = data.iloc[-1]

    feature_columns = artifact.get(
        "features",
        []
    )

    if not feature_columns:
        return None

    # Make sure every model feature exists
    missing = [
        feature
        for feature in feature_columns
        if feature not in latest.index
    ]

    if missing:
        print(
            "Missing model features:",
            missing
        )
        return None

    X = pd.DataFrame(
        [
            [
                latest[feature]
                for feature in feature_columns
            ]
        ],
        columns=feature_columns
    )

    if X.isna().any().any():
        return None

    try:

        prediction = float(
            artifact["model"].predict(X)[0]
        )

    except Exception as exc:

        print(
            "Prediction error:",
            exc
        )

        return None

    prediction = float(
        np.clip(
            prediction,
            0,
            100
        )
    )

    return prediction


# ============================================================
# 72-HOUR RECURSIVE FORECAST
# ============================================================

def forecast_72_hours(
    parking,
    history,
    artifact
):

    parking_id = parking[
        "parking_id"
    ]

    source = parking[
        "source"
    ]

    capacity = int(
        parking["capacity"]
    )

    working = (
        history[
            [
                "parking_id",
                "timestamp",
                "capacity",
                "occupied_spaces",
                "available_spaces",
                "occupancy_pct",
            ]
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    forecasts = []

    for hour in range(
        1,
        FORECAST_HOURS + 1
    ):

        prediction = predict_one_hour(
            working,
            source,
            artifact
        )

        if prediction is None:
            break

        next_timestamp = (
            pd.to_datetime(
                working["timestamp"].iloc[-1]
            )
            + pd.Timedelta(
                hours=1
            )
        )

        predicted_occupied = int(
            round(
                capacity
                * prediction
                / 100
            )
        )

        predicted_occupied = max(
            0,
            min(
                capacity,
                predicted_occupied
            )
        )

        predicted_available = (
            capacity
            - predicted_occupied
        )

        row = {
            "hour_ahead": hour,
            "timestamp": next_timestamp,
            "occupancy_pct": prediction,
            "occupied_spaces": predicted_occupied,
            "available_spaces": predicted_available,
        }

        forecasts.append(row)

        # ----------------------------------------------------
        # Recursive prediction:
        # predicted hour becomes input to next hour
        # ----------------------------------------------------

        working = pd.concat(
            [
                working,
                pd.DataFrame(
                    [
                        {
                            "parking_id":
                                parking_id,

                            "timestamp":
                                next_timestamp,

                            "capacity":
                                capacity,

                            "occupied_spaces":
                                predicted_occupied,

                            "available_spaces":
                                predicted_available,

                            "occupancy_pct":
                                prediction,
                        }
                    ]
                )
            ],
            ignore_index=True
        )

    return forecasts


# ============================================================
# FIND EXACT ESTIMATED CROSSING
# ============================================================

def find_crossing(
    parking,
    forecasts,
    threshold
):

    current_occupancy = float(
        parking[
            "occupancy_pct"
        ]
    )

    current_time = pd.to_datetime(
        parking[
            "timestamp"
        ]
    )

    # Already at/above threshold
    if current_occupancy >= threshold:

        return {
            "minutes": 0.0,
            "hours": 0.0,
            "estimated_time": current_time,
            "crossed": True
        }

    previous_value = (
        current_occupancy
    )

    previous_time = (
        current_time
    )

    for point in forecasts:

        point_value = float(
            point["occupancy_pct"]
        )

        point_time = pd.to_datetime(
            point["timestamp"]
        )

        if point_value >= threshold:

            change = (
                point_value
                - previous_value
            )

            # ------------------------------------------------
            # Interpolate between previous hour and current
            # forecast to estimate the threshold crossing.
            # ------------------------------------------------

            if change > 0:

                fraction = (
                    threshold
                    - previous_value
                ) / change

                fraction = max(
                    0.0,
                    min(
                        1.0,
                        fraction
                    )
                )

                estimated_time = (
                    previous_time
                    + (
                        point_time
                        - previous_time
                    ) * fraction
                )

            else:

                estimated_time = (
                    point_time
                )

            minutes = (
                estimated_time
                - current_time
            ).total_seconds() / 60

            return {
                "minutes":
                    round(
                        minutes,
                        1
                    ),

                "hours":
                    round(
                        minutes / 60,
                        2
                    ),

                "estimated_time":
                    estimated_time,

                "crossed":
                    True
            }

        previous_value = point_value

        previous_time = point_time

    # Did not reach threshold within 72 h
    return {
        "minutes": None,
        "hours": None,
        "estimated_time": None,
        "crossed": False
    }


# ============================================================
# ANALYZE ONE PARKING LOT
# ============================================================

def analyze_parking(
    parking,
    artifact
):

    parking_id = parking[
        "parking_id"
    ]

    print(
        f"Processing "
        f"{parking['parking_name']} "
        f"(ID {parking_id})"
    )

    history = get_history(
        parking_id,
        hours=180
    )

    # Need lag_168
    if len(history) < 169:

        print(
            "  Insufficient history."
        )

        return {
            **parking,
            "prediction_available":
                False
        }

    forecasts = forecast_72_hours(
        parking,
        history,
        artifact
    )

    result = dict(
        parking
    )

    # --------------------------------------------------------
    # Next-hour prediction
    # --------------------------------------------------------

    if forecasts:

        first = forecasts[0]

        result[
            "predicted_occupancy_pct"
        ] = round(
            first["occupancy_pct"],
            1
        )

        result[
            "predicted_occupied_spaces"
        ] = first[
            "occupied_spaces"
        ]

        result[
            "predicted_available_spaces"
        ] = first[
            "available_spaces"
        ]

    else:

        result[
            "predicted_occupancy_pct"
        ] = None

        result[
            "predicted_occupied_spaces"
        ] = None

        result[
            "predicted_available_spaces"
        ] = None

    # --------------------------------------------------------
    # Time to 90%
    # --------------------------------------------------------

    crossing_90 = find_crossing(
        parking,
        forecasts,
        THRESHOLD_90
    )

    result[
        "time_to_90_hours"
    ] = crossing_90["hours"]

    result[
        "time_to_90_minutes"
    ] = crossing_90["minutes"]

    result[
        "time_to_90_timestamp"
    ] = crossing_90["estimated_time"]

    # --------------------------------------------------------
    # Time to FULL
    # --------------------------------------------------------

    crossing_full = find_crossing(
        parking,
        forecasts,
        THRESHOLD_FULL
    )

    result[
        "time_to_full_hours"
    ] = crossing_full["hours"]

    result[
        "time_to_full_minutes"
    ] = crossing_full["minutes"]

    result[
        "time_to_full_timestamp"
    ] = crossing_full["estimated_time"]

    # --------------------------------------------------------
    # Fill risk
    # --------------------------------------------------------

    current = float(
        parking[
            "occupancy_pct"
        ]
    )

    if current >= 90:

        risk = "HIGH"

    elif crossing_90["crossed"]:

        hours_to_90 = (
            crossing_90["hours"]
        )

        if hours_to_90 <= 3:
            risk = "HIGH"

        elif hours_to_90 <= 8:
            risk = "MEDIUM"

        else:
            risk = "LOW"

    else:

        risk = "LOW"

    result[
        "fill_risk"
    ] = risk

    result[
        "prediction_available"
    ] = True

    return result


# ============================================================
# RUN ALL PARKING LOTS
# ============================================================

def analyze_all():

    artifact = load_model()

    parking_lots = (
        get_current_parking()
    )

    print()
    print(
        "=" * 110
    )

    print(
        "PARKING 72-HOUR TIME-TO-FULL ANALYSIS"
    )

    print(
        "=" * 110
    )

    print(
        f"Parking lots found: "
        f"{len(parking_lots)}"
    )

    results = []

    for index, parking in enumerate(
        parking_lots,
        start=1
    ):

        print(
            f"\n[{index}/{len(parking_lots)}]"
        )

        try:

            result = analyze_parking(
                parking,
                artifact
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

    return results


# ============================================================
# FORMATTING
# ============================================================

def format_hours(hours):

    if hours is None:
        return ">72 h"

    if hours < 1:
        return (
            f"{hours * 60:.0f} min"
        )

    return (
        f"{hours:.1f} h"
    )


def format_datetime(value):

    if value is None:
        return "Not within 72 h"

    return pd.to_datetime(
        value
    ).strftime(
        "%d-%b-%Y %I:%M %p"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print(
        "=" * 125
    )

    print(
        "FINAL 72-HOUR PARKING FORECAST"
    )

    print(
        "=" * 125
    )

    print()

    print(
        f"{'Parking':<32}"
        f"{'Current':>10}"
        f"{'Next Hr':>10}"
        f"{'To 90%':>12}"
        f"{'90% At':>25}"
        f"{'To Full':>12}"
        f"{'Full At':>25}"
    )

    print(
        "-" * 125
    )

    for row in results:

        name = str(
            row.get(
                "parking_name",
                "Unknown"
            )
        )[:31]

        current = row.get(
            "occupancy_pct"
        )

        next_hour = row.get(
            "predicted_occupancy_pct"
        )

        to_90 = row.get(
            "time_to_90_hours"
        )

        to_90_time = row.get(
            "time_to_90_timestamp"
        )

        to_full = row.get(
            "time_to_full_hours"
        )

        to_full_time = row.get(
            "time_to_full_timestamp"
        )

        current_text = (
            f"{float(current):.1f}%"
            if current is not None
            else "-"
        )

        next_text = (
            f"{float(next_hour):.1f}%"
            if next_hour is not None
            else "-"
        )

        print(
            f"{name:<32}"
            f"{current_text:>10}"
            f"{next_text:>10}"
            f"{format_hours(to_90):>12}"
            f"{format_datetime(to_90_time):>25}"
            f"{format_hours(to_full):>12}"
            f"{format_datetime(to_full_time):>25}"
        )

    # --------------------------------------------------------
    # FIRST TO 90%
    # --------------------------------------------------------

    candidates_90 = [
        row
        for row in results
        if row.get(
            "time_to_90_hours"
        ) is not None
    ]

    if candidates_90:

        first_90 = min(
            candidates_90,
            key=lambda row:
                row[
                    "time_to_90_minutes"
                ]
        )

        print()
        print(
            "=" * 90
        )

        print(
            "FIRST PARKING LOT TO REACH 90%"
        )

        print(
            "=" * 90
        )

        print(
            f"Parking       : "
            f"{first_90['parking_name']}"
        )

        print(
            f"Current       : "
            f"{first_90['occupancy_pct']:.1f}%"
        )

        print(
            f"Time to 90%   : "
            f"{first_90['time_to_90_hours']:.2f} hours"
        )

        print(
            f"Estimated time: "
            f"{format_datetime(first_90['time_to_90_timestamp'])}"
        )

    else:

        print()
        print(
            "No parking lot reaches 90% "
            "within the 72-hour forecast."
        )

    # --------------------------------------------------------
    # FIRST TO FULL
    # --------------------------------------------------------

    candidates_full = [
        row
        for row in results
        if row.get(
            "time_to_full_hours"
        ) is not None
    ]

    if candidates_full:

        first_full = min(
            candidates_full,
            key=lambda row:
                row[
                    "time_to_full_minutes"
                ]
        )

        print()
        print(
            "=" * 90
        )

        print(
            "FIRST PARKING LOT TO REACH FULL CAPACITY"
        )

        print(
            "=" * 90
        )

        print(
            f"Parking       : "
            f"{first_full['parking_name']}"
        )

        print(
            f"Current       : "
            f"{first_full['occupancy_pct']:.1f}%"
        )

        print(
            f"Time to full  : "
            f"{first_full['time_to_full_hours']:.2f} hours"
        )

        print(
            f"Estimated time: "
            f"{format_datetime(first_full['time_to_full_timestamp'])}"
        )

    else:

        print()
        print(
            "No parking lot reaches full capacity "
            "within the 72-hour forecast."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    results = analyze_all()

    print_results(
        results
    )

    print()
    print(
        "=" * 90
    )

    print(
        "72-HOUR TIME-TO-FULL ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()