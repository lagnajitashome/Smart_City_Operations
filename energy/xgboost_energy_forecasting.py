# ============================================================
# XGBOOST ENERGY DEMAND FORECASTING
# Smart City Operations Project
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# 2. CHECK DATABASE VARIABLES
# ============================================================

required_variables = {
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD
}

missing_variables = [
    name
    for name, value in required_variables.items()
    if not value
]

if missing_variables:
    raise ValueError(
        f"Missing database environment variables: "
        f"{missing_variables}"
    )


# ============================================================
# 3. DATABASE CONNECTION
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
# 4. LOAD ENERGY DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING ENERGY DATA")
print("=" * 60)

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

df = pd.read_sql(query, engine)

if df.empty:
    raise ValueError("No data found in energy_data table.")

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Total records : {len(df):,}")
print(f"Time range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Locations     : {df['location_id'].nunique()}")


# ============================================================
# 5. AGGREGATE ALL LOCATIONS
#    CITY-LEVEL ENERGY DATA
# ============================================================

print("\n" + "=" * 60)
print("CREATING CITY-LEVEL ENERGY DATA")
print("=" * 60)

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

print(f"City-level records : {len(city_df):,}")


# ============================================================
# 6. CREATE CONTINUOUS 1-MINUTE TIMELINE
# ============================================================

print("\n" + "=" * 60)
print("CREATING CONTINUOUS TIMELINE")
print("=" * 60)

full_index = pd.date_range(
    start=city_df.index.min(),
    end=city_df.index.max(),
    freq="min"
)

city_df = city_df.reindex(full_index)

city_df.index.name = "timestamp"

print(f"Expected timestamps : {len(full_index):,}")

print(
    f"Available timestamps: "
    f"{city_df['electricity_demand_mw'].notna().sum():,}"
)

print(
    f"Missing timestamps   : "
    f"{city_df['electricity_demand_mw'].isna().sum():,}"
)


# ============================================================
# 7. TIME FEATURES
# ============================================================

print("\nCreating time features...")

city_df["hour"] = city_df.index.hour

city_df["minute"] = city_df.index.minute

city_df["day_of_week"] = city_df.index.dayofweek

city_df["is_weekend"] = (
    city_df["day_of_week"] >= 5
).astype(int)


# ============================================================
# 8. CYCLICAL TIME FEATURES
# ============================================================

city_df["hour_sin"] = np.sin(
    2 * np.pi * city_df["hour"] / 24
)

city_df["hour_cos"] = np.cos(
    2 * np.pi * city_df["hour"] / 24
)

city_df["minute_sin"] = np.sin(
    2 * np.pi * city_df["minute"] / 60
)

city_df["minute_cos"] = np.cos(
    2 * np.pi * city_df["minute"] / 60
)


# ============================================================
# 9. LAG FEATURES
# ============================================================

print("Creating lag features...")

lag_values = [1, 5, 15, 30, 60]

for lag in lag_values:

    city_df[f"demand_lag_{lag}"] = (
        city_df["electricity_demand_mw"]
        .shift(lag)
    )


# Solar lag

city_df["solar_lag_60"] = (
    city_df["solar_generation_mw"]
    .shift(60)
)


# EV charging lag

city_df["ev_lag_60"] = (
    city_df["ev_charging_demand_mw"]
    .shift(60)
)


# ============================================================
# 10. ROLLING FEATURES
# ============================================================

print("Creating rolling features...")

city_df["demand_rolling_mean_15"] = (
    city_df["electricity_demand_mw"]
    .shift(1)
    .rolling(15)
    .mean()
)

city_df["demand_rolling_mean_30"] = (
    city_df["electricity_demand_mw"]
    .shift(1)
    .rolling(30)
    .mean()
)

city_df["demand_rolling_mean_60"] = (
    city_df["electricity_demand_mw"]
    .shift(1)
    .rolling(60)
    .mean()
)

city_df["demand_std_60"] = (
    city_df["electricity_demand_mw"]
    .shift(1)
    .rolling(60)
    .std()
)


# ============================================================
# 11. TARGET
#    NEXT-HOUR ELECTRICITY DEMAND
# ============================================================

print("Creating next-hour target...")

city_df["target_demand"] = (
    city_df["electricity_demand_mw"]
    .shift(-60)
)


# ============================================================
# 12. FEATURE LIST
# ============================================================

features = [

    # Time features
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",

    # Cyclical features
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",

    # Demand lags
    "demand_lag_1",
    "demand_lag_5",
    "demand_lag_15",
    "demand_lag_30",
    "demand_lag_60",

    # Other lags
    "solar_lag_60",
    "ev_lag_60",

    # Rolling features
    "demand_rolling_mean_15",
    "demand_rolling_mean_30",
    "demand_rolling_mean_60",
    "demand_std_60",

    # Current known values
    "solar_generation_mw",
    "ev_charging_demand_mw",
    "transformer_load_pct"
]


# ============================================================
# 13. CREATE MODEL DATASET
# ============================================================

print("\n" + "=" * 60)
print("PREPARING MODEL DATASET")
print("=" * 60)

model_df = city_df[
    features + ["target_demand"]
].copy()


# Replace infinity values with NaN

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove rows with missing values

model_df = model_df.dropna()


print(f"Model rows  : {len(model_df):,}")
print(f"Features    : {len(features)}")


# ============================================================
# 14. CHECK DATA QUALITY
# ============================================================

print("\nChecking data quality...")

print(
    f"Remaining NaN values: "
    f"{model_df.isna().sum().sum()}"
)

print(
    f"Remaining Inf values: "
    f"{np.isinf(model_df.to_numpy()).sum()}"
)


if model_df.empty:
    raise ValueError(
        "Model dataset is empty after removing NaN/Inf values."
    )


# ============================================================
# 15. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
#
# Historical backfill = 14 days
#
# First 12 days  -> Training
# Last 2 days    -> Validation
# After backfill -> Live Test
# ============================================================

print("\n" + "=" * 60)
print("CREATING CHRONOLOGICAL DATA SPLIT")
print("=" * 60)

first_timestamp = model_df.index.min()

backfill_end = (
    first_timestamp +
    pd.Timedelta(days=14)
)

validation_start = (
    backfill_end -
    pd.Timedelta(days=2)
)


# -------------------------------
# Training
# -------------------------------

train_df = model_df[
    model_df.index < validation_start
]


# -------------------------------
# Validation
# -------------------------------

validation_df = model_df[
    (model_df.index >= validation_start) &
    (model_df.index < backfill_end)
]


# -------------------------------
# Live test
# -------------------------------

test_df = model_df[
    model_df.index >= backfill_end
]


print("\nDataset split:")
print("-" * 50)

print(
    f"Training   : {len(train_df):,} rows"
)

print(
    f"Validation : {len(validation_df):,} rows"
)

print(
    f"Live Test  : {len(test_df):,} rows"
)

print("-" * 50)

print(
    f"Training period   : "
    f"{train_df.index.min()} → {train_df.index.max()}"
)

print(
    f"Validation period : "
    f"{validation_df.index.min()} → {validation_df.index.max()}"
)

if len(test_df) > 0:

    print(
        f"Test period       : "
        f"{test_df.index.min()} → {test_df.index.max()}"
    )


# ============================================================
# 16. CHECK TRAINING / VALIDATION DATA
# ============================================================

if train_df.empty:
    raise ValueError("Training dataset is empty.")

if validation_df.empty:
    raise ValueError(
        "Validation dataset is empty. "
        "Check whether at least 14 days of historical data exist."
    )


# ============================================================
# 17. CREATE X AND y
# ============================================================

X_train = train_df[features]
y_train = train_df["target_demand"]

X_val = validation_df[features]
y_val = validation_df["target_demand"]


if len(test_df) > 0:

    X_test = test_df[features]
    y_test = test_df["target_demand"]

else:

    X_test = None
    y_test = None


# ============================================================
# 18. FINAL DATA QUALITY CHECK
# ============================================================

print("\nFinal data quality check...")

print(
    f"X_train shape : {X_train.shape}"
)

print(
    f"X_val shape   : {X_val.shape}"
)

if X_test is not None:
    print(
        f"X_test shape  : {X_test.shape}"
    )


print(
    f"\nNaN in X_train: "
    f"{X_train.isna().sum().sum()}"
)

print(
    f"NaN in X_val: "
    f"{X_val.isna().sum().sum()}"
)

print(
    f"NaN in y_train: "
    f"{y_train.isna().sum()}"
)

print(
    f"NaN in y_val: "
    f"{y_val.isna().sum()}"
)


# ============================================================
# 19. CREATE XGBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)

model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 20. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_val, y_val)
    ],

    verbose=False
)

print("XGBoost training completed.")


# ============================================================
# 21. VALIDATION PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION EVALUATION")
print("=" * 60)

y_val_pred = model.predict(X_val)


# Extra safety check

y_val_pred = np.asarray(y_val_pred)

valid_validation_rows = (
    np.isfinite(y_val.to_numpy()) &
    np.isfinite(y_val_pred)
)


y_val_clean = y_val.to_numpy()[
    valid_validation_rows
]

y_val_pred_clean = y_val_pred[
    valid_validation_rows
]


if len(y_val_clean) == 0:
    raise ValueError(
        "No valid validation predictions available."
    )


# ============================================================
# 22. VALIDATION METRICS
# ============================================================

val_mae = mean_absolute_error(
    y_val_clean,
    y_val_pred_clean
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val_clean,
        y_val_pred_clean
    )
)

val_r2 = r2_score(
    y_val_clean,
    y_val_pred_clean
)


print("\n==============================")
print("XGBOOST VALIDATION RESULTS")
print("==============================")

print(
    f"MAE  : {val_mae:.3f} MW"
)

print(
    f"RMSE : {val_rmse:.3f} MW"
)

print(
    f"R²   : {val_r2:.4f}"
)


# ============================================================
# 23. LIVE TEST EVALUATION
# ============================================================

test_mae = None
test_rmse = None
test_r2 = None

if X_test is not None and len(X_test) > 0:

    print("\n" + "=" * 60)
    print("LIVE TEST EVALUATION")
    print("=" * 60)

    y_test_pred = model.predict(X_test)

    y_test_pred = np.asarray(
        y_test_pred
    )

    valid_test_rows = (
        np.isfinite(y_test.to_numpy()) &
        np.isfinite(y_test_pred)
    )

    y_test_clean = y_test.to_numpy()[
        valid_test_rows
    ]

    y_test_pred_clean = y_test_pred[
        valid_test_rows
    ]

    if len(y_test_clean) > 0:

        test_mae = mean_absolute_error(
            y_test_clean,
            y_test_pred_clean
        )

        test_rmse = np.sqrt(
            mean_squared_error(
                y_test_clean,
                y_test_pred_clean
            )
        )

        test_r2 = r2_score(
            y_test_clean,
            y_test_pred_clean
        )

        print("\n==============================")
        print("XGBOOST LIVE TEST RESULTS")
        print("==============================")

        print(
            f"MAE  : {test_mae:.3f} MW"
        )

        print(
            f"RMSE : {test_rmse:.3f} MW"
        )

        print(
            f"R²   : {test_r2:.4f}"
        )

    else:

        print(
            "\nNo valid live test observations available."
        )

else:

    print(
        "\nNo live test data available yet."
    )


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

importance_df = pd.DataFrame({

    "feature": features,

    "importance": model.feature_importances_

}).sort_values(
    "importance",
    ascending=False
)


print(
    importance_df.to_string(index=False)
)


# ============================================================
# 25. CREATE MODEL DIRECTORY
# ============================================================

model_directory = "models/energy"

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 26. SAVE MODEL
# ============================================================

model_path = os.path.join(
    model_directory,
    "energy_demand_xgboost.pkl"
)


artifact = {

    "model": model,

    "features": features,

    "model_name": "XGBoost",

    "target": "next_hour_electricity_demand_mw"

}


joblib.dump(
    artifact,
    model_path
)


print(
    f"\nModel saved to: {model_path}"
)


# ============================================================
# 27. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    model_directory,
    "xgboost_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)


print(
    f"Feature importance saved to: "
    f"{importance_path}"
)


# ============================================================
# 28. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "timestamp": validation_df.index,

    "actual_demand_mw": y_val.to_numpy(),

    "predicted_demand_mw": y_val_pred

})


prediction_path = os.path.join(
    model_directory,
    "xgboost_validation_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


print(
    f"Validation predictions saved to: "
    f"{prediction_path}"
)


# ============================================================
# 29. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "model": ["XGBoost"],

    "validation_mae": [val_mae],

    "validation_rmse": [val_rmse],

    "validation_r2": [val_r2],

    "test_mae": [test_mae],

    "test_rmse": [test_rmse],

    "test_r2": [test_r2]

})


metrics_path = os.path.join(
    model_directory,
    "xgboost_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


print(
    f"Metrics saved to: {metrics_path}"
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST ENERGY FORECASTING COMPLETE")
print("=" * 60)

print(
    f"Model      : XGBoost Regressor"
)

print(
    f"Features   : {len(features)}"
)

print(
    f"Training   : {len(X_train):,} rows"
)

print(
    f"Validation : {len(X_val):,} rows"
)

print(
    f"Live Test  : "
    f"{0 if X_test is None else len(X_test):,} rows"
)

print("\nValidation Metrics")

print(
    f"MAE  = {val_mae:.3f} MW"
)

print(
    f"RMSE = {val_rmse:.3f} MW"
)

print(
    f"R²   = {val_r2:.4f}"
)

if test_mae is not None:

    print("\nLive Test Metrics")

    print(
        f"MAE  = {test_mae:.3f} MW"
    )

    print(
        f"RMSE = {test_rmse:.3f} MW"
    )

    print(
        f"R²   = {test_r2:.4f}"
    )

print("\nDone.")