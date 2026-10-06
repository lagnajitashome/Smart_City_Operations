import os
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "smart_city_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


# ============================================================
# DATABASE CONNECTION
# ============================================================

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = os.path.join(
    "models",
    "energy"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "energy_demand_model.pkl"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_energy_data():

    query = text(
        """
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
        """
    )

    with engine.connect() as connection:

        df = pd.read_sql(
            query,
            connection
        )

    return df


# ============================================================
# CITY-LEVEL AGGREGATION
# ============================================================

def prepare_city_data(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    city = (
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
            ),

            streetlights_online=(
                "streetlights_online",
                "sum"
            ),

            streetlights_total=(
                "streetlights_total",
                "sum"
            ),

            power_outage=(
                "power_outage",
                "max"
            )
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Create a continuous 1-minute index.
    #
    # Missing timestamps remain NaN.
    # We do NOT fill them artificially.
    # --------------------------------------------------------

    full_index = pd.date_range(
        start=city.index.min(),
        end=city.index.max(),
        freq="min"
    )

    city = city.reindex(
        full_index
    )

    city.index.name = "timestamp"

    return city


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["hour"] = data.index.hour

    data["minute"] = data.index.minute

    data["day_of_week"] = (
        data.index.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclic time features
    # --------------------------------------------------------

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["minute_sin"] = np.sin(
        2 * np.pi * data["minute"] / 60
    )

    data["minute_cos"] = np.cos(
        2 * np.pi * data["minute"] / 60
    )

    # --------------------------------------------------------
    # Demand lag features
    # --------------------------------------------------------

    for lag in [
        1,
        5,
        15,
        30,
        60
    ]:

        data[
            f"demand_lag_{lag}"
        ] = (
            data[
                "electricity_demand_mw"
            ].shift(lag)
        )

    # --------------------------------------------------------
    # 1-hour lag for solar and EV
    # --------------------------------------------------------

    data["solar_lag_60"] = (
        data[
            "solar_generation_mw"
        ].shift(60)
    )

    data["ev_lag_60"] = (
        data[
            "ev_charging_demand_mw"
        ].shift(60)
    )

    # --------------------------------------------------------
    # Rolling demand
    #
    # shift(1) prevents the current demand from
    # leaking into the feature.
    # --------------------------------------------------------

    previous_demand = (
        data[
            "electricity_demand_mw"
        ].shift(1)
    )

    data["demand_rolling_15"] = (
        previous_demand
        .rolling(15)
        .mean()
    )

    data["demand_rolling_30"] = (
        previous_demand
        .rolling(30)
        .mean()
    )

    data["demand_rolling_60"] = (
        previous_demand
        .rolling(60)
        .mean()
    )

    data["demand_std_60"] = (
        previous_demand
        .rolling(60)
        .std()
    )

    # --------------------------------------------------------
    # Target
    #
    # Demand exactly 60 minutes into the future.
    # --------------------------------------------------------

    data["target_next_hour"] = (
        data[
            "electricity_demand_mw"
        ].shift(-60)
    )

    return data


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

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

    "demand_rolling_15",
    "demand_rolling_30",
    "demand_rolling_60",

    "demand_std_60",

    "solar_generation_mw",
    "ev_charging_demand_mw",

    "transformer_load_pct",
]


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(data):

    first_timestamp = data.index.min()

    # The first 14 days are our historical backfill.

    live_start = (
        first_timestamp
        + pd.Timedelta(days=14)
    )

    validation_start = (
        live_start
        - pd.Timedelta(days=2)
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train = data[
        data.index < validation_start
    ].copy()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation = data[
        (
            data.index >= validation_start
        )
        &
        (
            data.index < live_start
        )
    ].copy()

    # --------------------------------------------------------
    # LIVE TEST
    # --------------------------------------------------------

    test = data[
        data.index >= live_start
    ].copy()

    print()
    print("=" * 60)
    print("TIME SERIES SPLIT")
    print("=" * 60)

    print(
        f"Training period   : "
        f"{train.index.min()} → "
        f"{train.index.max()}"
    )

    print(
        f"Validation period : "
        f"{validation.index.min()} → "
        f"{validation.index.max()}"
    )

    print(
        f"Test period       : "
        f"{test.index.min()} → "
        f"{test.index.max()}"
    )

    print("=" * 60)

    return (
        train,
        validation,
        test
    )


# ============================================================
# PREPARE X AND y
# ============================================================

def prepare_xy(df):

    model_data = (
        df[
            FEATURES
            + ["target_next_hour"]
        ]
        .dropna()
    )

    X = model_data[
        FEATURES
    ]

    y = model_data[
        "target_next_hour"
    ]

    return X, y


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(
    X_train,
    y_train
):

    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=18,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(
        X
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print()
    print("=" * 60)

    print(
        f"{dataset_name} PERFORMANCE"
    )

    print("=" * 60)

    print(
        f"MAE  : {mae:.3f} MW"
    )

    print(
        f"RMSE : {rmse:.3f} MW"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model
):

    importance = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print()
    print("=" * 60)

    print(
        "TOP FEATURE IMPORTANCE"
    )

    print("=" * 60)

    print(
        importance
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    validation_metrics,
    test_metrics
):

    artifact = {

        "model":
            model,

        "features":
            FEATURES,

        "target":
            "electricity_demand_mw",

        "forecast_horizon_minutes":
            60,

        "model_type":
            "RandomForestRegressor",

        "validation_metrics":
            validation_metrics,

        "test_metrics":
            test_metrics
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print()
    print(
        "✓ Model saved:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "NEXT-HOUR ELECTRICITY DEMAND FORECASTING"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print(
        "Loading energy data..."
    )

    raw_data = (
        load_energy_data()
    )

    print(
        f"Raw records: "
        f"{len(raw_data):,}"
    )

    if raw_data.empty:

        print(
            "✗ No energy data found."
        )

        return

    # --------------------------------------------------------
    # CITY AGGREGATION
    # --------------------------------------------------------

    city_data = (
        prepare_city_data(
            raw_data
        )
    )

    print(
        f"Continuous city timestamps: "
        f"{len(city_data):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Creating forecasting features..."
    )

    featured_data = (
        create_features(
            city_data
        )
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = split_data(
        featured_data
    )

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    X_train, y_train = (
        prepare_xy(
            train_df
        )
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    X_validation, y_validation = (
        prepare_xy(
            validation_df
        )
    )

    # --------------------------------------------------------
    # LIVE TEST DATA
    # --------------------------------------------------------

    X_test, y_test = (
        prepare_xy(
            test_df
        )
    )

    print()
    print("=" * 60)

    print(
        "USABLE ML ROWS"
    )

    print("=" * 60)

    print(
        f"Training   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation : "
        f"{len(X_validation):,}"
    )

    print(
        f"Live test  : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(X_train) < 1000:

        print()
        print(
            "✗ Training dataset is too small."
        )

        return

    if len(X_validation) < 100:

        print()
        print(
            "✗ Validation dataset is too small."
        )

        return

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print(
        "Training Random Forest..."
    )

    model = train_model(
        X_train,
        y_train
    )

    print(
        "✓ Training complete"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_metrics = (
        evaluate_model(
            model,
            X_validation,
            y_validation,
            "VALIDATION"
        )
    )

    # --------------------------------------------------------
    # LIVE TEST
    # --------------------------------------------------------

    test_metrics = None

    if len(X_test) >= 100:

        test_metrics = (
            evaluate_model(
                model,
                X_test,
                y_test,
                "LIVE TEST"
            )
        )

    else:

        print()
        print(
            "⚠ Live test set is currently small."
        )

        print(
            "The model will still be saved."
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    show_feature_importance(
        model
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_model(
        model,
        validation_metrics,
        test_metrics
    )

    print()
    print("=" * 70)

    print(
        "✓ FORECASTING MODEL COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()




