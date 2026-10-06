# ============================================================
# SMART CITY OPERATIONS
# Fine-Tuned Traffic Congestion Prediction
#
# Forecast horizons:
#   - 30 minutes
#   - 60 minutes
#
# Models:
#   1. Baseline
#   2. Random Forest
#   3. Fine-tuned Random Forest
#   4. XGBoost
#   5. Fine-tuned XGBoost
#
# Evaluation design:
#   70% chronological TRAIN
#   15% chronological VALIDATION
#   15% chronological TEST
#
# Tuning:
#   Stage 1 -> RandomizedSearchCV on TRAIN only
#   Stage 2 -> GridSearchCV around the best Stage-1 params
#   Validation -> choose the final tuned algorithm
#   Final fit -> TRAIN + VALIDATION
#   TEST -> used only once for final reporting
#
# IMPORTANT:
# Do not try to force a high training R2. A large train/test gap
# is overfitting. The goal is good unseen test performance.
# ============================================================

import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV,
    GridSearchCV,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAFFIC_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "traffic",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_traffic_data():
    files = glob.glob(
        os.path.join(
            TRAFFIC_DIR,
            "*.csv",
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No traffic CSV files found in: {TRAFFIC_DIR}"
        )

    latest_file = max(
        files,
        key=os.path.getmtime,
    )

    print("\nUsing traffic file:")
    print(latest_file)

    df = pd.read_csv(
        latest_file
    )

    print(
        "\nRaw shape:",
        df.shape,
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_traffic_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    required_columns = [
        "Location ID",
        "Timestamp",
        "Current Speed (km/h)",
        "Free Flow Speed (km/h)",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    column_mapping = {
        "Current Speed (km/h)": "current_speed",
        "Free Flow Speed (km/h)": "free_flow_speed",
        "Current Travel Time (sec)": "current_travel_time",
        "Free Flow Travel Time (sec)": "free_flow_travel_time",
        "Traffic Delay (sec)": "traffic_delay",
        "Speed Reduction (%)": "speed_reduction",
        "Confidence": "confidence",
        "Road Closed": "road_closed",
        "Road Class": "road_class",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }

    df = df.rename(
        columns=column_mapping
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

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
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "road_closed" in df.columns:
        if not pd.api.types.is_numeric_dtype(
            df["road_closed"]
        ):
            df["road_closed"] = (
                df["road_closed"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": 1,
                        "false": 0,
                        "yes": 1,
                        "no": 0,
                        "1": 1,
                        "0": 0,
                    }
                )
            )

        df["road_closed"] = (
            pd.to_numeric(
                df["road_closed"],
                errors="coerce",
            )
            .fillna(0)
            .clip(0, 1)
        )
    else:
        df["road_closed"] = 0

    df = df.dropna(
        subset=[
            "Location ID",
            "Timestamp",
            "current_speed",
            "free_flow_speed",
        ]
    )

    # Congestion percentage from current/free-flow speed.
    df["congestion_pct"] = np.where(
        df["free_flow_speed"] > 0,
        (
            (
                df["free_flow_speed"]
                - df["current_speed"]
            )
            / df["free_flow_speed"]
        ) * 100,
        0,
    )

    df["congestion_pct"] = (
        pd.to_numeric(
            df["congestion_pct"],
            errors="coerce",
        )
        .clip(0, 100)
    )

    df = df.dropna(
        subset=["congestion_pct"]
    )

    df = df.sort_values(
        [
            "Location ID",
            "Timestamp",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\nRows before duplicate removal:",
        len(df),
    )

    aggregation = {}

    for column in df.columns:

        if column in [
            "Location ID",
            "Timestamp",
            "road_class",
        ]:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            aggregation[column] = "mean"

    if "road_class" in df.columns:
        aggregation["road_class"] = "first"

    df = (
        df.groupby(
            [
                "Location ID",
                "Timestamp",
            ],
            as_index=False,
        )
        .agg(aggregation)
    )

    df = df.sort_values(
        [
            "Location ID",
            "Timestamp",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Rows after duplicate removal:",
        len(df),
    )

    return df


# ============================================================
# FUTURE TARGETS
# ============================================================

def _create_future_target(
    df,
    horizon_minutes,
    target_name,
    tolerance_minutes=8,
):
    """
    API data is approximately every 5 minutes, so exact
    timestamp matching is too strict.

    For each row, find the first observation at or after
    current_time + horizon, provided it is within tolerance.
    """

    result = df.copy()

    target_values = pd.Series(
        np.nan,
        index=result.index,
        dtype=float,
    )

    for _, group in result.groupby(
        "Location ID",
        sort=False,
    ):

        group = group.sort_values(
            "Timestamp"
        )

        timestamps = group[
            "Timestamp"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        congestion = group[
            "congestion_pct"
        ].to_numpy(
            dtype=float
        )

        target_times = (
            timestamps
            + np.timedelta64(
                horizon_minutes,
                "m",
            )
        )

        positions = np.searchsorted(
            timestamps,
            target_times,
            side="left",
        )

        valid_position = (
            positions < len(timestamps)
        )

        values = np.full(
            len(group),
            np.nan,
            dtype=float,
        )

        valid_indices = np.where(
            valid_position
        )[0]

        if len(valid_indices) > 0:

            matched_positions = positions[
                valid_indices
            ]

            differences = (
                timestamps[
                    matched_positions
                ]
                - target_times[
                    valid_indices
                ]
            )

            within_tolerance = (
                differences
                <= np.timedelta64(
                    tolerance_minutes,
                    "m",
                )
            )

            usable_indices = (
                valid_indices[
                    within_tolerance
                ]
            )

            usable_positions = positions[
                usable_indices
            ]

            values[
                usable_indices
            ] = congestion[
                usable_positions
            ]

        target_values.loc[
            group.index
        ] = values

    result[target_name] = target_values

    return result


def create_future_targets(df):
    print(
        "\nCreating future congestion targets..."
    )

    df = _create_future_target(
        df,
        horizon_minutes=30,
        target_name="target_congestion_30m",
        tolerance_minutes=8,
    )

    df = _create_future_target(
        df,
        horizon_minutes=60,
        target_name="target_congestion_60m",
        tolerance_minutes=8,
    )

    print(
        "\nTarget rows created:"
    )

    print(
        "30-minute:",
        df[
            "target_congestion_30m"
        ].notna().sum(),
    )

    print(
        "60-minute:",
        df[
            "target_congestion_60m"
        ].notna().sum(),
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    df = df.sort_values(
        [
            "Location ID",
            "Timestamp",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["hour"] = (
        df["Timestamp"].dt.hour
    )

    df["minute"] = (
        df["Timestamp"].dt.minute
    )

    df["day_of_week"] = (
        df["Timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["Timestamp"].dt.day
    )

    df["month"] = (
        df["Timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_morning_peak"] = (
        df["hour"].between(7, 10)
    ).astype(int)

    df["is_evening_peak"] = (
        df["hour"].between(16, 20)
    ).astype(int)

    # --------------------------------------------------------
    # Cyclic time features
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # One-hot encode location and road class
    # --------------------------------------------------------

    location_dummies = pd.get_dummies(
        df["Location ID"],
        prefix="location",
        dtype=int,
    )

    df = pd.concat(
        [
            df,
            location_dummies,
        ],
        axis=1,
    )

    if "road_class" in df.columns:

        road_dummies = pd.get_dummies(
            df["road_class"].astype(str),
            prefix="road_class",
            dtype=int,
        )

        df = pd.concat(
            [
                df,
                road_dummies,
            ],
            axis=1,
        )

    # --------------------------------------------------------
    # Current traffic ratios
    # --------------------------------------------------------

    df["speed_ratio"] = np.where(
        df["free_flow_speed"] > 0,
        df["current_speed"]
        / df["free_flow_speed"],
        0,
    )

    if (
        "current_travel_time" in df.columns
        and
        "free_flow_travel_time" in df.columns
    ):
        df["travel_time_ratio"] = np.where(
            df["free_flow_travel_time"] > 0,
            df["current_travel_time"]
            / df["free_flow_travel_time"],
            1,
        )

    else:
        df["travel_time_ratio"] = 1

    if (
        "traffic_delay" in df.columns
        and
        "free_flow_travel_time" in df.columns
    ):
        df["delay_ratio"] = np.where(
            df["free_flow_travel_time"] > 0,
            df["traffic_delay"]
            / df["free_flow_travel_time"],
            0,
        )

    else:
        df["delay_ratio"] = 0

    # --------------------------------------------------------
    # Historical congestion features
    #
    # Data is approximately 5 minutes per observation:
    # 1 = ~5 min, 2 = ~10 min, 3 = ~15 min,
    # 6 = ~30 min, 12 = ~60 min.
    # --------------------------------------------------------

    grouped_congestion = df.groupby(
        "Location ID"
    )["congestion_pct"]

    for lag in [
        1,
        2,
        3,
        6,
        12,
    ]:

        df[
            f"congestion_lag_{lag}"
        ] = grouped_congestion.shift(
            lag
        )

    # --------------------------------------------------------
    # Historical speed features
    # --------------------------------------------------------

    grouped_speed = df.groupby(
        "Location ID"
    )["current_speed"]

    for lag in [
        1,
        3,
        6,
        12,
    ]:

        df[
            f"speed_lag_{lag}"
        ] = grouped_speed.shift(
            lag
        )

    # --------------------------------------------------------
    # Change features
    # --------------------------------------------------------

    df["congestion_change_1"] = (
        df["congestion_pct"]
        - df["congestion_lag_1"]
    )

    df["congestion_change_3"] = (
        df["congestion_pct"]
        - df["congestion_lag_3"]
    )

    # --------------------------------------------------------
    # Rolling congestion statistics
    # --------------------------------------------------------

    df[
        "congestion_roll_mean_3"
    ] = grouped_congestion.transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )

    df[
        "congestion_roll_mean_6"
    ] = grouped_congestion.transform(
        lambda x:
        x.shift(1)
        .rolling(6)
        .mean()
    )

    df[
        "congestion_roll_mean_12"
    ] = grouped_congestion.transform(
        lambda x:
        x.shift(1)
        .rolling(12)
        .mean()
    )

    df[
        "congestion_roll_std_12"
    ] = grouped_congestion.transform(
        lambda x:
        x.shift(1)
        .rolling(12)
        .std()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# ============================================================
# FEATURE LIST
# ============================================================

def get_feature_columns(df):

    base_features = [
        "congestion_pct",

        "current_speed",
        "free_flow_speed",
        "current_travel_time",
        "free_flow_travel_time",
        "traffic_delay",
        "speed_reduction",
        "confidence",
        "road_closed",

        "speed_ratio",
        "travel_time_ratio",
        "delay_ratio",

        "hour",
        "minute",
        "day_of_week",
        "day_of_month",
        "month",

        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",

        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",

        "congestion_lag_1",
        "congestion_lag_2",
        "congestion_lag_3",
        "congestion_lag_6",
        "congestion_lag_12",

        "speed_lag_1",
        "speed_lag_3",
        "speed_lag_6",
        "speed_lag_12",

        "congestion_change_1",
        "congestion_change_3",

        "congestion_roll_mean_3",
        "congestion_roll_mean_6",
        "congestion_roll_mean_12",
        "congestion_roll_std_12",
    ]

    location_features = [
        column
        for column in df.columns
        if column.startswith("location_")
    ]

    road_features = [
        column
        for column in df.columns
        if column.startswith("road_class_")
    ]

    all_features = (
        base_features
        + location_features
        + road_features
    )

    return [
        column
        for column in all_features
        if column in df.columns
    ]


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

def split_data(
    model_data,
    feature_columns,
):
    """
    70% train, 15% validation, 15% test by unique time.
    """

    model_data = model_data.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    unique_times = np.sort(
        model_data[
            "Timestamp"
        ]
        .drop_duplicates()
        .to_numpy()
    )

    if len(unique_times) < 20:
        raise ValueError(
            "Not enough unique timestamps "
            "for train/validation/test split."
        )

    train_cut = int(
        len(unique_times) * 0.70
    )

    valid_cut = int(
        len(unique_times) * 0.85
    )

    train_end = unique_times[
        train_cut
    ]

    valid_end = unique_times[
        valid_cut
    ]

    train_data = model_data[
        model_data["Timestamp"]
        < train_end
    ].copy()

    valid_data = model_data[
        (
            model_data["Timestamp"]
            >= train_end
        )
        &
        (
            model_data["Timestamp"]
            < valid_end
        )
    ].copy()

    test_data = model_data[
        model_data["Timestamp"]
        >= valid_end
    ].copy()

    X_train = train_data[
        feature_columns
    ].copy()

    X_valid = valid_data[
        feature_columns
    ].copy()

    X_test = test_data[
        feature_columns
    ].copy()

    y_train = train_data[
        "target"
    ].copy()

    y_valid = valid_data[
        "target"
    ].copy()

    y_test = test_data[
        "target"
    ].copy()

    # Replace infinity values.
    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_valid = X_valid.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Impute using training medians only.
    train_medians = X_train.median(
        numeric_only=True
    )

    X_train = X_train.fillna(
        train_medians
    )

    X_valid = X_valid.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    return (
        train_data,
        valid_data,
        test_data,
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_medians,
        train_end,
        valid_end,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):
    predictions = np.clip(
        predictions,
        0,
        100,
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        ),
        "R2": r2_score(
            y_true,
            predictions,
        ),
    }


def print_metrics(
    model_name,
    train_metrics,
    valid_metrics=None,
    test_metrics=None,
):
    print(
        "\n" + "=" * 70
    )

    print(
        model_name
    )

    print(
        "=" * 70
    )

    print(
        f"Train MAE : "
        f"{train_metrics['MAE']:.4f}"
    )

    print(
        f"Train RMSE: "
        f"{train_metrics['RMSE']:.4f}"
    )

    print(
        f"Train R2  : "
        f"{train_metrics['R2']:.4f}"
    )

    if valid_metrics is not None:

        print(
            f"Valid MAE : "
            f"{valid_metrics['MAE']:.4f}"
        )

        print(
            f"Valid RMSE: "
            f"{valid_metrics['RMSE']:.4f}"
        )

        print(
            f"Valid R2  : "
            f"{valid_metrics['R2']:.4f}"
        )

    if test_metrics is not None:

        print(
            f"Test MAE  : "
            f"{test_metrics['MAE']:.4f}"
        )

        print(
            f"Test RMSE : "
            f"{test_metrics['RMSE']:.4f}"
        )

        print(
            f"Test R2   : "
            f"{test_metrics['R2']:.4f}"
        )


# ============================================================
# BASELINE
# ============================================================

def evaluate_baseline(
    train_data,
    valid_data,
    test_data,
):
    train_pred = np.clip(
        train_data[
            "congestion_pct"
        ].to_numpy(),
        0,
        100,
    )

    valid_pred = np.clip(
        valid_data[
            "congestion_pct"
        ].to_numpy(),
        0,
        100,
    )

    test_pred = np.clip(
        test_data[
            "congestion_pct"
        ].to_numpy(),
        0,
        100,
    )

    train_metrics = calculate_metrics(
        train_data["target"],
        train_pred,
    )

    valid_metrics = calculate_metrics(
        valid_data["target"],
        valid_pred,
    )

    test_metrics = calculate_metrics(
        test_data["target"],
        test_pred,
    )

    print_metrics(
        "Baseline",
        train_metrics,
        valid_metrics,
        test_metrics,
    )

    return {
        "model": "Baseline",
        "CV_MAE": np.nan,
        "train_MAE": train_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "train_R2": train_metrics["R2"],
        "valid_MAE": valid_metrics["MAE"],
        "valid_RMSE": valid_metrics["RMSE"],
        "valid_R2": valid_metrics["R2"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
    }


# ============================================================
# FINE-TUNING HELPERS
# ============================================================

def fine_tune_random_forest(
    X_train,
    y_train,
):
    """
    Two-stage RF search:
      Stage 1: broad randomized search
      Stage 2: small grid around the best parameters
    """

    cv = TimeSeriesSplit(
        n_splits=4
    )

    rf_stage1_grid = {
        "n_estimators": [
            300,
            500,
            700,
            900,
        ],

        "max_depth": [
            6,
            8,
            10,
            12,
            16,
            20,
            None,
        ],

        "min_samples_split": [
            2,
            5,
            10,
            15,
            20,
        ],

        "min_samples_leaf": [
            1,
            2,
            4,
            6,
            8,
            10,
        ],

        "max_features": [
            "sqrt",
            "log2",
            0.5,
            0.7,
            0.9,
        ],

        "bootstrap": [
            True,
        ],
    }

    stage1 = RandomizedSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),

        param_distributions=rf_stage1_grid,

        n_iter=25,

        scoring="neg_mean_absolute_error",

        cv=cv,

        random_state=42,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    print(
        "\nRF Stage 1: broad hyperparameter search..."
    )

    stage1.fit(
        X_train,
        y_train,
    )

    best_rf = stage1.best_params_

    print(
        "\nRF Stage 1 best parameters:"
    )

    print(
        best_rf
    )

    # --------------------------------------------------------
    # Fine tuning around the Stage-1 solution.
    # --------------------------------------------------------

    n_estimators = int(
        best_rf["n_estimators"]
    )

    max_depth = best_rf[
        "max_depth"
    ]

    min_samples_split = int(
        best_rf[
            "min_samples_split"
        ]
    )

    min_samples_leaf = int(
        best_rf[
            "min_samples_leaf"
        ]
    )

    fine_n_estimators = sorted(
        set(
            [
                max(
                    200,
                    n_estimators - 100,
                ),
                n_estimators,
                n_estimators + 100,
            ]
        )
    )

    fine_min_split = sorted(
        set(
            [
                max(
                    2,
                    min_samples_split - 2,
                ),
                min_samples_split,
                min_samples_split + 2,
            ]
        )
    )

    fine_min_leaf = sorted(
        set(
            [
                max(
                    1,
                    min_samples_leaf - 1,
                ),
                min_samples_leaf,
                min_samples_leaf + 1,
            ]
        )
    )

    if max_depth is None:
        fine_depth = [
            None,
            16,
            20,
        ]
    else:
        fine_depth = sorted(
            set(
                [
                    max(
                        4,
                        int(max_depth) - 2,
                    ),
                    int(max_depth),
                    int(max_depth) + 2,
                ]
            )
        )

    rf_stage2_grid = {
        "n_estimators": fine_n_estimators,
        "max_depth": fine_depth,
        "min_samples_split": fine_min_split,
        "min_samples_leaf": fine_min_leaf,
        "max_features": [
            best_rf[
                "max_features"
            ]
        ],
        "bootstrap": [
            True
        ],
    }

    stage2 = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),

        param_grid=rf_stage2_grid,

        scoring="neg_mean_absolute_error",

        cv=cv,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    print(
        "\nRF Stage 2: fine tuning around the best region..."
    )

    stage2.fit(
        X_train,
        y_train,
    )

    print(
        "\nRF final parameters:"
    )

    print(
        stage2.best_params_
    )

    return (
        stage2.best_estimator_,
        -stage2.best_score_,
        stage1.best_params_,
        stage2.best_params_,
    )


def fine_tune_xgboost(
    X_train,
    y_train,
):
    """
    Two-stage XGBoost search:
      Stage 1: broad randomized search
      Stage 2: small grid around the best parameters
    """

    cv = TimeSeriesSplit(
        n_splits=4
    )

    xgb_stage1_grid = {
        "n_estimators": [
            200,
            300,
            400,
            500,
            700,
            900,
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
            6,
            8,
        ],

        "learning_rate": [
            0.01,
            0.015,
            0.02,
            0.03,
            0.05,
            0.08,
        ],

        "subsample": [
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "colsample_bytree": [
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],

        "min_child_weight": [
            2,
            3,
            5,
            8,
            12,
        ],

        "gamma": [
            0,
            0.05,
            0.10,
            0.20,
            0.50,
            1.00,
        ],

        "reg_alpha": [
            0,
            0.01,
            0.05,
            0.10,
            0.50,
            1.00,
        ],

        "reg_lambda": [
            2,
            5,
            10,
            20,
        ],
    }

    stage1 = RandomizedSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        ),

        param_distributions=xgb_stage1_grid,

        n_iter=30,

        scoring="neg_mean_absolute_error",

        cv=cv,

        random_state=42,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    print(
        "\nXGB Stage 1: broad hyperparameter search..."
    )

    stage1.fit(
        X_train,
        y_train,
    )

    best_xgb = stage1.best_params_

    print(
        "\nXGB Stage 1 best parameters:"
    )

    print(
        best_xgb
    )

    # --------------------------------------------------------
    # Fine tuning around the Stage-1 solution.
    # --------------------------------------------------------

    n_estimators = int(
        best_xgb[
            "n_estimators"
        ]
    )

    max_depth = int(
        best_xgb[
            "max_depth"
        ]
    )

    learning_rate = float(
        best_xgb[
            "learning_rate"
        ]
    )

    min_child_weight = int(
        best_xgb[
            "min_child_weight"
        ]
    )

    gamma = float(
        best_xgb[
            "gamma"
        ]
    )

    reg_alpha = float(
        best_xgb[
            "reg_alpha"
        ]
    )

    reg_lambda = float(
        best_xgb[
            "reg_lambda"
        ]
    )

    fine_xgb_grid = {
        "n_estimators": sorted(
            set(
                [
                    max(
                        150,
                        n_estimators - 100,
                    ),
                    n_estimators,
                    n_estimators + 100,
                ]
            )
        ),

        "max_depth": sorted(
            set(
                [
                    max(
                        2,
                        max_depth - 1,
                    ),
                    max_depth,
                    max_depth + 1,
                ]
            )
        ),

        "learning_rate": sorted(
            set(
                [
                    max(
                        0.005,
                        learning_rate * 0.75,
                    ),
                    learning_rate,
                    min(
                        0.15,
                        learning_rate * 1.25,
                    ),
                ]
            )
        ),

        "subsample": [
            max(
                0.60,
                min(
                    1.0,
                    best_xgb[
                        "subsample"
                    ],
                ),
            )
        ],

        "colsample_bytree": [
            max(
                0.60,
                min(
                    1.0,
                    best_xgb[
                        "colsample_bytree"
                    ],
                ),
            )
        ],

        "min_child_weight": sorted(
            set(
                [
                    max(
                        1,
                        min_child_weight - 2,
                    ),
                    min_child_weight,
                    min_child_weight + 2,
                ]
            )
        ),

        "gamma": sorted(
            set(
                [
                    max(
                        0,
                        gamma - 0.1,
                    ),
                    gamma,
                    gamma + 0.1,
                ]
            )
        ),

        "reg_alpha": sorted(
            set(
                [
                    max(
                        0,
                        reg_alpha * 0.5,
                    ),
                    reg_alpha,
                    reg_alpha * 1.5,
                ]
            )
        ),

        "reg_lambda": sorted(
            set(
                [
                    max(
                        1,
                        reg_lambda - 2,
                    ),
                    reg_lambda,
                    reg_lambda + 2,
                ]
            )
        ),
    }

    stage2 = GridSearchCV(
        estimator=XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        ),

        param_grid=fine_xgb_grid,

        scoring="neg_mean_absolute_error",

        cv=cv,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    print(
        "\nXGB Stage 2: fine tuning around the best region..."
    )

    stage2.fit(
        X_train,
        y_train,
    )

    print(
        "\nXGB final parameters:"
    )

    print(
        stage2.best_params_
    )

    return (
        stage2.best_estimator_,
        -stage2.best_score_,
        stage1.best_params_,
        stage2.best_params_,
    )


# ============================================================
# TRAIN ONE HORIZON
# ============================================================

def train_one_horizon(
    df,
    target_column,
    horizon_name,
):

    print("\n")
    print("=" * 80)
    print(
        f"TRAINING {horizon_name.upper()} "
        "CONGESTION MODEL"
    )
    print("=" * 80)

    feature_columns = get_feature_columns(
        df
    )

    model_data = df[
        [
            "Location ID",
            "Timestamp",
            *feature_columns,
            target_column,
        ]
    ].copy()

    model_data = model_data.rename(
        columns={
            target_column: "target"
        }
    )

    model_data = model_data.dropna(
        subset=["target"]
    )

    model_data = model_data.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    print(
        "\nUsable rows:",
        len(model_data)
    )

    if len(model_data) < 300:
        raise ValueError(
            f"Only {len(model_data)} usable rows "
            f"are available for {horizon_name}. "
            "Collect more history before training."
        )

    (
        train_data,
        valid_data,
        test_data,
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        train_medians,
        train_end,
        valid_end,
    ) = split_data(
        model_data,
        feature_columns,
    )

    print(
        "\nTrain rows:",
        len(train_data),
    )

    print(
        "Validation rows:",
        len(valid_data),
    )

    print(
        "Test rows:",
        len(test_data),
    )

    print(
        "Training ends:",
        train_end,
    )

    print(
        "Validation ends:",
        valid_end,
    )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_result = evaluate_baseline(
        train_data,
        valid_data,
        test_data,
    )

    # ========================================================
    # BASIC RANDOM FOREST
    # ========================================================

    print(
        "\nTraining basic Random Forest..."
    )

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(
        X_train,
        y_train,
    )

    rf_train_pred = np.clip(
        rf.predict(X_train),
        0,
        100,
    )

    rf_valid_pred = np.clip(
        rf.predict(X_valid),
        0,
        100,
    )

    rf_test_pred = np.clip(
        rf.predict(X_test),
        0,
        100,
    )

    rf_train_metrics = calculate_metrics(
        y_train,
        rf_train_pred,
    )

    rf_valid_metrics = calculate_metrics(
        y_valid,
        rf_valid_pred,
    )

    rf_test_metrics = calculate_metrics(
        y_test,
        rf_test_pred,
    )

    print_metrics(
        "Random Forest",
        rf_train_metrics,
        rf_valid_metrics,
        rf_test_metrics,
    )

    # ========================================================
    # TUNED RANDOM FOREST
    # ========================================================

    (
        tuned_rf,
        rf_cv_mae,
        rf_stage1_params,
        rf_final_params,
    ) = fine_tune_random_forest(
        X_train,
        y_train,
    )

    tuned_rf_train_pred = np.clip(
        tuned_rf.predict(X_train),
        0,
        100,
    )

    tuned_rf_valid_pred = np.clip(
        tuned_rf.predict(X_valid),
        0,
        100,
    )

    tuned_rf_test_pred = np.clip(
        tuned_rf.predict(X_test),
        0,
        100,
    )

    tuned_rf_train_metrics = calculate_metrics(
        y_train,
        tuned_rf_train_pred,
    )

    tuned_rf_valid_metrics = calculate_metrics(
        y_valid,
        tuned_rf_valid_pred,
    )

    tuned_rf_test_metrics = calculate_metrics(
        y_test,
        tuned_rf_test_pred,
    )

    print_metrics(
        "Tuned Random Forest",
        tuned_rf_train_metrics,
        tuned_rf_valid_metrics,
        tuned_rf_test_metrics,
    )

    # ========================================================
    # BASIC XGBOOST
    # ========================================================

    print(
        "\nTraining basic XGBoost..."
    )

    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(
        X_train,
        y_train,
    )

    xgb_train_pred = np.clip(
        xgb.predict(X_train),
        0,
        100,
    )

    xgb_valid_pred = np.clip(
        xgb.predict(X_valid),
        0,
        100,
    )

    xgb_test_pred = np.clip(
        xgb.predict(X_test),
        0,
        100,
    )

    xgb_train_metrics = calculate_metrics(
        y_train,
        xgb_train_pred,
    )

    xgb_valid_metrics = calculate_metrics(
        y_valid,
        xgb_valid_pred,
    )

    xgb_test_metrics = calculate_metrics(
        y_test,
        xgb_test_pred,
    )

    print_metrics(
        "XGBoost",
        xgb_train_metrics,
        xgb_valid_metrics,
        xgb_test_metrics,
    )

    # ========================================================
    # TUNED XGBOOST
    # ========================================================

    (
        tuned_xgb,
        xgb_cv_mae,
        xgb_stage1_params,
        xgb_final_params,
    ) = fine_tune_xgboost(
        X_train,
        y_train,
    )

    tuned_xgb_train_pred = np.clip(
        tuned_xgb.predict(X_train),
        0,
        100,
    )

    tuned_xgb_valid_pred = np.clip(
        tuned_xgb.predict(X_valid),
        0,
        100,
    )

    tuned_xgb_test_pred = np.clip(
        tuned_xgb.predict(X_test),
        0,
        100,
    )

    tuned_xgb_train_metrics = calculate_metrics(
        y_train,
        tuned_xgb_train_pred,
    )

    tuned_xgb_valid_metrics = calculate_metrics(
        y_valid,
        tuned_xgb_valid_pred,
    )

    tuned_xgb_test_metrics = calculate_metrics(
        y_test,
        tuned_xgb_test_pred,
    )

    print_metrics(
        "Tuned XGBoost",
        tuned_xgb_train_metrics,
        tuned_xgb_valid_metrics,
        tuned_xgb_test_metrics,
    )

    # ========================================================
    # COMPARISON ON VALIDATION SET
    # ========================================================

    results = [
        baseline_result,

        {
            "model": "Random Forest",
            "CV_MAE": np.nan,
            "train_MAE": rf_train_metrics["MAE"],
            "train_RMSE": rf_train_metrics["RMSE"],
            "train_R2": rf_train_metrics["R2"],
            "valid_MAE": rf_valid_metrics["MAE"],
            "valid_RMSE": rf_valid_metrics["RMSE"],
            "valid_R2": rf_valid_metrics["R2"],
            "test_MAE": rf_test_metrics["MAE"],
            "test_RMSE": rf_test_metrics["RMSE"],
            "test_R2": rf_test_metrics["R2"],
        },

        {
            "model": "Tuned Random Forest",
            "CV_MAE": rf_cv_mae,
            "train_MAE": tuned_rf_train_metrics["MAE"],
            "train_RMSE": tuned_rf_train_metrics["RMSE"],
            "train_R2": tuned_rf_train_metrics["R2"],
            "valid_MAE": tuned_rf_valid_metrics["MAE"],
            "valid_RMSE": tuned_rf_valid_metrics["RMSE"],
            "valid_R2": tuned_rf_valid_metrics["R2"],
            "test_MAE": tuned_rf_test_metrics["MAE"],
            "test_RMSE": tuned_rf_test_metrics["RMSE"],
            "test_R2": tuned_rf_test_metrics["R2"],
        },

        {
            "model": "XGBoost",
            "CV_MAE": np.nan,
            "train_MAE": xgb_train_metrics["MAE"],
            "train_RMSE": xgb_train_metrics["RMSE"],
            "train_R2": xgb_train_metrics["R2"],
            "valid_MAE": xgb_valid_metrics["MAE"],
            "valid_RMSE": xgb_valid_metrics["RMSE"],
            "valid_R2": xgb_valid_metrics["R2"],
            "test_MAE": xgb_test_metrics["MAE"],
            "test_RMSE": xgb_test_metrics["RMSE"],
            "test_R2": xgb_test_metrics["R2"],
        },

        {
            "model": "Tuned XGBoost",
            "CV_MAE": xgb_cv_mae,
            "train_MAE": tuned_xgb_train_metrics["MAE"],
            "train_RMSE": tuned_xgb_train_metrics["RMSE"],
            "train_R2": tuned_xgb_train_metrics["R2"],
            "valid_MAE": tuned_xgb_valid_metrics["MAE"],
            "valid_RMSE": tuned_xgb_valid_metrics["RMSE"],
            "valid_R2": tuned_xgb_valid_metrics["R2"],
            "test_MAE": tuned_xgb_test_metrics["MAE"],
            "test_RMSE": tuned_xgb_test_metrics["RMSE"],
            "test_R2": tuned_xgb_test_metrics["R2"],
        },
    ]

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 85)
    print(
        f"{horizon_name.upper()} MODEL COMPARISON"
    )
    print("=" * 85)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT MODEL USING VALIDATION MAE
    # ========================================================
    # IMPORTANT:
    # Test set is NOT used for model selection.
    # ========================================================

    tuned_results = results_df[
        results_df["model"].isin(
            [
                "Tuned Random Forest",
                "Tuned XGBoost",
            ]
        )
    ].copy()

    selected_name = (
        tuned_results
        .sort_values("valid_MAE")
        .iloc[0]["model"]
    )

    if selected_name == "Tuned Random Forest":
        selected_model = tuned_rf
        selected_params = rf_final_params
        selected_stage1_params = rf_stage1_params
        selected_cv_mae = rf_cv_mae
    else:
        selected_model = tuned_xgb
        selected_params = xgb_final_params
        selected_stage1_params = xgb_stage1_params
        selected_cv_mae = xgb_cv_mae

    print(
        "\nSelected model using VALIDATION MAE:"
    )

    print(
        selected_name
    )

    # ========================================================
    # FINAL REFIT ON TRAIN + VALIDATION
    # ========================================================

    train_valid_data = pd.concat(
        [
            train_data,
            valid_data,
        ],
        ignore_index=True,
    )

    X_train_valid = train_valid_data[
        feature_columns
    ].copy()

    y_train_valid = train_valid_data[
        "target"
    ].copy()

    X_train_valid = X_train_valid.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test_final = test_data[
        feature_columns
    ].copy()

    X_test_final = X_test_final.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    train_valid_medians = (
        X_train_valid.median(
            numeric_only=True
        )
    )

    X_train_valid = X_train_valid.fillna(
        train_valid_medians
    )

    X_test_final = X_test_final.fillna(
        train_valid_medians
    )

    if selected_name == "Tuned Random Forest":

        final_model = (
            RandomForestRegressor(
                **selected_params,
                random_state=42,
                n_jobs=-1,
            )
        )

    else:

        final_model = (
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                **selected_params,
                random_state=42,
                n_jobs=-1,
            )
        )

    final_model.fit(
        X_train_valid,
        y_train_valid,
    )

    final_test_pred = np.clip(
        final_model.predict(
            X_test_final
        ),
        0,
        100,
    )

    final_test_metrics = calculate_metrics(
        test_data["target"],
        final_test_pred,
    )

    # Training metric after final refit.
    final_train_pred = np.clip(
        final_model.predict(
            X_train_valid
        ),
        0,
        100,
    )

    final_train_metrics = calculate_metrics(
        y_train_valid,
        final_train_pred,
    )

    print("\n")
    print("=" * 80)
    print(
        f"FINAL {horizon_name.upper()} MODEL "
        "AFTER TRAIN+VALIDATION REFIT"
    )
    print("=" * 80)

    print(
        f"Model     : {selected_name}"
    )

    print(
        f"Train+Val MAE : "
        f"{final_train_metrics['MAE']:.4f}"
    )

    print(
        f"Train+Val RMSE: "
        f"{final_train_metrics['RMSE']:.4f}"
    )

    print(
        f"Train+Val R2  : "
        f"{final_train_metrics['R2']:.4f}"
    )

    print(
        f"UNSEEN TEST MAE : "
        f"{final_test_metrics['MAE']:.4f}"
    )

    print(
        f"UNSEEN TEST RMSE: "
        f"{final_test_metrics['RMSE']:.4f}"
    )

    print(
        f"UNSEEN TEST R2  : "
        f"{final_test_metrics['R2']:.4f}"
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_output = test_data[
        [
            "Location ID",
            "Timestamp",
            "congestion_pct",
            "target",
        ]
    ].copy()

    test_output = test_output.rename(
        columns={
            "target":
                "actual_future_congestion"
        }
    )

    test_output[
        "predicted_congestion"
    ] = final_test_pred

    test_output[
        "absolute_error"
    ] = (
        test_output[
            "predicted_congestion"
        ]
        - test_output[
            "actual_future_congestion"
        ]
    ).abs()

    test_file = os.path.join(
        MODEL_DIR,
        f"congestion_test_predictions_{horizon_name}_fine_tuned.csv",
    )

    test_output.to_csv(
        test_file,
        index=False,
    )

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_file = os.path.join(
        MODEL_DIR,
        f"congestion_model_comparison_{horizon_name}_fine_tuned.csv",
    )

    results_df.to_csv(
        comparison_file,
        index=False,
    )

    # ========================================================
    # SAVE FINAL MODEL PACKAGE
    # ========================================================

    final_package = {
        "model": final_model,
        "model_name": selected_name,
        "features": feature_columns,
        "target": target_column,
        "horizon": horizon_name,

        "train_end": str(train_end),
        "validation_end": str(valid_end),

        "train_validation_medians":
            train_valid_medians.to_dict(),

        "cv_mae":
            float(selected_cv_mae),

        "stage1_best_params":
            selected_stage1_params,

        "fine_tuned_params":
            selected_params,

        "final_train_validation_metrics":
            {
                "MAE":
                    float(
                        final_train_metrics[
                            "MAE"
                        ]
                    ),
                "RMSE":
                    float(
                        final_train_metrics[
                            "RMSE"
                        ]
                    ),
                "R2":
                    float(
                        final_train_metrics[
                            "R2"
                        ]
                    ),
            },

        "unseen_test_metrics":
            {
                "MAE":
                    float(
                        final_test_metrics[
                            "MAE"
                        ]
                    ),
                "RMSE":
                    float(
                        final_test_metrics[
                            "RMSE"
                        ]
                    ),
                "R2":
                    float(
                        final_test_metrics[
                            "R2"
                        ]
                    ),
            },

        "model_comparison":
            results_df.to_dict(
                orient="records"
            ),
    }

    model_file = os.path.join(
        MODEL_DIR,
        f"congestion_model_{horizon_name}_fine_tuned.pkl",
    )

    joblib.dump(
        final_package,
        model_file,
    )

    print(
        "\nSaved:"
    )

    print(
        model_file
    )

    print(
        test_file
    )

    print(
        comparison_file
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 90)
    print(
        "SMART CITY - FINE-TUNED TRAFFIC "
        "CONGESTION PREDICTION"
    )
    print("=" * 90)

    traffic = load_traffic_data()

    traffic = clean_traffic_data(
        traffic
    )

    traffic = create_future_targets(
        traffic
    )

    traffic = create_features(
        traffic
    )

    target_30_availability = (
        traffic[
            "target_congestion_30m"
        ]
        .notna()
        .mean()
        * 100
    )

    target_60_availability = (
        traffic[
            "target_congestion_60m"
        ]
        .notna()
        .mean()
        * 100
    )

    print(
        "\nTarget availability:"
    )

    print(
        f"30-minute: "
        f"{target_30_availability:.2f}%"
    )

    print(
        f"60-minute: "
        f"{target_60_availability:.2f}%"
    )

    # --------------------------------------------------------
    # 30-minute
    # --------------------------------------------------------

    train_one_horizon(
        traffic,
        "target_congestion_30m",
        "30m",
    )

    # --------------------------------------------------------
    # 60-minute
    # --------------------------------------------------------

    train_one_horizon(
        traffic,
        "target_congestion_60m",
        "60m",
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        traffic
    )

    feature_file = os.path.join(
        MODEL_DIR,
        "congestion_features_fine_tuned.pkl",
    )

    joblib.dump(
        feature_columns,
        feature_file,
    )

    print(
        "\nFeature list saved to:"
    )

    print(
        feature_file
    )

    print("\n")
    print("=" * 90)
    print(
        "FINE-TUNED CONGESTION TRAINING COMPLETED"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()