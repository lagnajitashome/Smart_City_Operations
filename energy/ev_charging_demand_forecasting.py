# ============================================================
# EV CHARGING DEMAND FORECASTING
# Smart City Operations Project
#
# Models:
# 1. Random Forest Regressor
# 2. XGBoost Regressor
# 3. LightGBM Regressor
#
# Target:
# Next-hour EV charging demand
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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
        f"Missing database variables: {missing_variables}"
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

print("\n" + "=" * 65)
print("LOADING ENERGY DATA")
print("=" * 65)

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

if df.empty:

    raise ValueError(
        "No data found in energy_data table."
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


print(
    f"Total records : {len(df):,}"
)

print(
    f"Time range    : "
    f"{df['timestamp'].min()} → "
    f"{df['timestamp'].max()}"
)

print(
    f"Locations     : "
    f"{df['location_id'].nunique()}"
)


# ============================================================
# 5. CITY-LEVEL AGGREGATION
# ============================================================

print("\n" + "=" * 65)
print("CREATING CITY-LEVEL EV DEMAND")
print("=" * 65)

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


print(
    f"City-level records: "
    f"{len(city_df):,}"
)


# ============================================================
# 6. CREATE CONTINUOUS TIMELINE
# ============================================================

print("\nCreating continuous timeline...")

full_index = pd.date_range(

    start=city_df.index.min(),

    end=city_df.index.max(),

    freq="min"
)


city_df = city_df.reindex(
    full_index
)

city_df.index.name = "timestamp"


print(
    f"Expected timestamps : "
    f"{len(full_index):,}"
)

print(
    f"Available timestamps: "
    f"{city_df['ev_charging_demand_mw'].notna().sum():,}"
)


# ============================================================
# 7. TIME FEATURES
# ============================================================

city_df["hour"] = (
    city_df.index.hour
)

city_df["minute"] = (
    city_df.index.minute
)

city_df["day_of_week"] = (
    city_df.index.dayofweek
)

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
# 9. EV DEMAND LAG FEATURES
# ============================================================

print("\nCreating EV demand lag features...")

for lag in [
    1,
    5,
    15,
    30,
    60,
    120,
    180,
    360,
    1440
]:

    city_df[
        f"ev_demand_lag_{lag}"
    ] = (
        city_df[
            "ev_charging_demand_mw"
        ]
        .shift(lag)
    )


# ============================================================
# 10. ELECTRICITY DEMAND LAGS
# ============================================================

print(
    "Creating electricity demand lag features..."
)

for lag in [
    60,
    1440
]:

    city_df[
        f"electricity_demand_lag_{lag}"
    ] = (
        city_df[
            "electricity_demand_mw"
        ]
        .shift(lag)
    )


# ============================================================
# 11. SOLAR LAGS
# ============================================================

city_df["solar_lag_60"] = (
    city_df[
        "solar_generation_mw"
    ]
    .shift(60)
)

city_df["solar_lag_1440"] = (
    city_df[
        "solar_generation_mw"
    ]
    .shift(1440)
)


# ============================================================
# 12. EV ROLLING FEATURES
# ============================================================

print(
    "Creating EV rolling features..."
)

city_df["ev_rolling_mean_15"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(1)
    .rolling(15)
    .mean()
)

city_df["ev_rolling_mean_30"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(1)
    .rolling(30)
    .mean()
)

city_df["ev_rolling_mean_60"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(1)
    .rolling(60)
    .mean()
)

city_df["ev_rolling_std_60"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(1)
    .rolling(60)
    .std()
)


# ============================================================
# 13. EV DEMAND CHANGE FEATURES
# ============================================================

city_df["ev_change_15"] = (

    city_df[
        "ev_charging_demand_mw"
    ]

    - city_df[
        "ev_charging_demand_mw"
    ].shift(15)

)


city_df["ev_change_60"] = (

    city_df[
        "ev_charging_demand_mw"
    ]

    - city_df[
        "ev_charging_demand_mw"
    ].shift(60)

)


# ============================================================
# 14. NEXT-HOUR EV DEMAND TARGET
# ============================================================

city_df["next_hour_ev_demand"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(-60)
)


# ============================================================
# 15. FEATURE LIST
# ============================================================

features = [

    # Time
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",

    # Cyclical time
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",

    # EV demand lags
    "ev_demand_lag_1",
    "ev_demand_lag_5",
    "ev_demand_lag_15",
    "ev_demand_lag_30",
    "ev_demand_lag_60",
    "ev_demand_lag_120",
    "ev_demand_lag_180",
    "ev_demand_lag_360",
    "ev_demand_lag_1440",

    # Electricity demand
    "electricity_demand_lag_60",
    "electricity_demand_lag_1440",

    # Solar
    "solar_lag_60",
    "solar_lag_1440",

    # EV rolling
    "ev_rolling_mean_15",
    "ev_rolling_mean_30",
    "ev_rolling_mean_60",
    "ev_rolling_std_60",

    # EV changes
    "ev_change_15",
    "ev_change_60",

    # Current known variables
    "electricity_demand_mw",
    "solar_generation_mw",
    "transformer_load_pct"

]


# ============================================================
# 16. IDENTIFY 14-DAY HISTORICAL BACKFILL
# ============================================================

first_timestamp = city_df.index.min()

backfill_end = (
    first_timestamp +
    pd.Timedelta(days=14)
)


# ============================================================
# 17. CREATE MODEL DATASET
# ============================================================

model_df = city_df[
    features + [
        "ev_charging_demand_mw",
        "next_hour_ev_demand"
    ]
].copy()


model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


model_df = model_df.dropna()


# ============================================================
# 18. CHRONOLOGICAL SPLIT
#
# First 12 days → Training
# Last 2 days   → Validation
# After 14 days → Live Test
# ============================================================

validation_start = (
    backfill_end -
    pd.Timedelta(days=2)
)


train_df = model_df[
    model_df.index < validation_start
].copy()


validation_df = model_df[
    (model_df.index >= validation_start)
    &
    (model_df.index < backfill_end)
].copy()


test_df = model_df[
    model_df.index >= backfill_end
].copy()


print("\n" + "=" * 65)
print("DATASET SPLIT")
print("=" * 65)

print(
    f"Training   : "
    f"{len(train_df):,}"
)

print(
    f"Validation : "
    f"{len(validation_df):,}"
)

print(
    f"Live Test  : "
    f"{len(test_df):,}"
)


# ============================================================
# 19. CREATE X AND y
# ============================================================

X_train = train_df[
    features
]

y_train = train_df[
    "next_hour_ev_demand"
]


X_val = validation_df[
    features
]

y_val = validation_df[
    "next_hour_ev_demand"
]


if len(test_df) > 0:

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "next_hour_ev_demand"
    ]

else:

    X_test = None
    y_test = None


# ============================================================
# 20. MODEL DEFINITIONS
# ============================================================

print("\n" + "=" * 65)
print("CREATING MODELS")
print("=" * 65)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest = RandomForestRegressor(

    n_estimators=300,

    max_depth=18,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgboost_model = XGBRegressor(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=7,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------

lightgbm_model = LGBMRegressor(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=8,

    num_leaves=31,

    min_child_samples=20,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="regression",

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


models = {

    "Random Forest":
        random_forest,

    "XGBoost":
        xgboost_model,

    "LightGBM":
        lightgbm_model

}


# ============================================================
# 21. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 65)

    print(
        f"TRAINING {model_name.upper()}"
    )

    print("=" * 65)


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    if model_name == "XGBoost":

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ],

            verbose=False
        )

    elif model_name == "LightGBM":

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ]

        )

    else:

        model.fit(
            X_train,
            y_train
        )


    print(
        f"{model_name} training completed."
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_val
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_val,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )


    r2 = r2_score(
        y_val,
        predictions
    )


    print(
        f"\n{model_name} VALIDATION RESULTS"
    )

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

        "model": model_name,

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "predictions": predictions

    }


# ============================================================
# 22. TRAIN ALL MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    result = evaluate_model(

        model_name,

        model,

        X_train,

        y_train,

        X_val,

        y_val

    )

    results.append(
        result
    )

    trained_models[
        model_name
    ] = model


# ============================================================
# 23. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("EV CHARGING DEMAND MODEL COMPARISON")
print("=" * 70)


comparison_df = pd.DataFrame({

    "Model": [

        result["model"]

        for result in results

    ],

    "MAE": [

        result["mae"]

        for result in results

    ],

    "RMSE": [

        result["rmse"]

        for result in results

    ],

    "R2": [

        result["r2"]

        for result in results

    ]

})


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 24. VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({

    "timestamp":
        validation_df.index,

    "current_ev_demand_mw":
        validation_df[
            "ev_charging_demand_mw"
        ].values,

    "actual_next_hour_ev_demand_mw":
        y_val.values

})


for result in results:

    model_name = result["model"]

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    validation_predictions[
        f"{safe_name}_prediction_mw"
    ] = result[
        "predictions"
    ]


# ============================================================
# 25. MODEL DIRECTORY
# ============================================================

model_directory = "models/energy"

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 26. SAVE EACH MODEL
# ============================================================

print("\n" + "=" * 65)
print("SAVING MODELS")
print("=" * 65)


model_file_names = {

    "Random Forest":
        "ev_demand_random_forest.pkl",

    "XGBoost":
        "ev_demand_xgboost.pkl",

    "LightGBM":
        "ev_demand_lightgbm.pkl"

}


for model_name, model in trained_models.items():

    model_path = os.path.join(

        model_directory,

        model_file_names[
            model_name
        ]

    )


    artifact = {

        "model": model,

        "features": features,

        "model_name": model_name,

        "target":
            "next_hour_ev_charging_demand"

    }


    joblib.dump(
        artifact,
        model_path
    )


    print(
        f"{model_name} saved to:"
        f"\n{model_path}"
    )


# ============================================================
# 27. SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(

    model_directory,

    "ev_charging_demand_model_comparison.csv"

)


comparison_df.to_csv(

    comparison_path,

    index=False

)


print(
    f"\nModel comparison saved to:"
    f"\n{comparison_path}"
)


# ============================================================
# 28. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_path = os.path.join(

    model_directory,

    "ev_charging_demand_validation_predictions.csv"

)


validation_predictions.to_csv(

    prediction_path,

    index=False

)


print(
    f"Validation predictions saved to:"
    f"\n{prediction_path}"
)


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("SAVING FEATURE IMPORTANCE")
print("=" * 65)


for model_name, model in trained_models.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    importance_df = pd.DataFrame({

        "feature":
            features,

        "importance":
            model.feature_importances_

    }).sort_values(

        "importance",

        ascending=False

    )


    importance_path = os.path.join(

        model_directory,

        f"ev_demand_{safe_name}"
        "_feature_importance.csv"

    )


    importance_df.to_csv(

        importance_path,

        index=False

    )


    print(

        f"{model_name} feature importance saved:"
        f"\n{importance_path}"

    )


# ============================================================
# 30. CURRENT EV DEMAND FORECAST
# ============================================================

print("\n" + "=" * 65)
print("CURRENT EV CHARGING DEMAND FORECAST")
print("=" * 65)


latest_data = city_df[
    features
].replace(

    [np.inf, -np.inf],

    np.nan

).dropna()


if len(latest_data) > 0:

    latest_features = (
        latest_data
        .iloc[[-1]]
    )


    latest_timestamp = (
        latest_features.index[0]
    )


    current_ev_demand = float(

        city_df.loc[
            latest_timestamp,
            "ev_charging_demand_mw"
        ]

    )


    print(
        f"Timestamp          : "
        f"{latest_timestamp}"
    )

    print(
        f"Current EV Demand  : "
        f"{current_ev_demand:.2f} MW"
    )


    current_forecast_rows = []


    for model_name, model in trained_models.items():

        prediction = float(

            model.predict(
                latest_features
            )[0]

        )


        # EV demand cannot be negative

        prediction = max(
            0,
            prediction
        )


        print(
            f"\n{model_name}"
        )

        print(
            f"Next-hour EV demand:"
            f" {prediction:.2f} MW"
        )


        current_forecast_rows.append({

            "timestamp":
                latest_timestamp,

            "model":
                model_name,

            "current_ev_demand_mw":
                current_ev_demand,

            "predicted_next_hour_ev_demand_mw":
                prediction

        })


    current_forecast_df = pd.DataFrame(

        current_forecast_rows

    )


    current_forecast_path = os.path.join(

        model_directory,

        "current_ev_charging_demand_forecast.csv"

    )


    current_forecast_df.to_csv(

        current_forecast_path,

        index=False

    )


    print(
        f"\nCurrent forecast saved to:"
        f"\n{current_forecast_path}"
    )


else:

    print(
        "No valid latest observation "
        "available for prediction."
    )


# ============================================================
# 31. CURRENT EV DEMAND FORECAST
# ============================================================

print("\n" + "=" * 65)
print("CURRENT EV CHARGING DEMAND FORECAST")
print("=" * 65)

latest_data = city_df[features].replace(
    [np.inf, -np.inf], np.nan
).dropna()

if len(latest_data) > 0:

    latest_features = latest_data.iloc[[-1]]
    latest_timestamp = latest_features.index[0]

    current_ev_demand = float(
        city_df.loc[
            latest_timestamp,
            "ev_charging_demand_mw"
        ]
    )

    print(f"Timestamp          : {latest_timestamp}")
    print(f"Current EV Demand  : {current_ev_demand:.2f} MW")

    current_forecast_rows = []

    for model_name, model in trained_models.items():

        prediction = float(
            model.predict(latest_features)[0]
        )

        prediction = max(0, prediction)

        print(f"\n{model_name}")
        print(
            f"Next-hour EV demand: {prediction:.2f} MW"
        )

        current_forecast_rows.append({
            "timestamp": latest_timestamp,
            "model": model_name,
            "current_ev_demand_mw": current_ev_demand,
            "predicted_next_hour_ev_demand_mw": prediction
        })

    current_forecast_df = pd.DataFrame(
        current_forecast_rows
    )

    current_forecast_path = os.path.join(
        model_directory,
        "current_ev_charging_demand_forecast.csv"
    )

    current_forecast_df.to_csv(
        current_forecast_path,
        index=False
    )

    print(
        f"\nCurrent forecast saved to:"
        f"\n{current_forecast_path}"
    )

else:

    print(
        "No valid latest observation available for prediction."
    )


# ============================================================
# 32. SELECT FINAL PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 65)
print("SELECTING FINAL EV DEMAND MODEL")
print("=" * 65)

# Primary selection metric: MAE.
# Secondary metrics: RMSE and R2.

best_model_row = comparison_df.sort_values(
    by=["MAE", "RMSE", "R2"],
    ascending=[True, True, False]
).iloc[0]

best_model_name = best_model_row["Model"]

final_model = trained_models[best_model_name]

final_artifact = {
    "model": final_model,
    "features": features,
    "model_name": best_model_name,
    "target": "next_hour_ev_charging_demand",
    "forecast_horizon": "60 minutes",
    "selection_metric": "MAE",
    "validation_mae": float(best_model_row["MAE"]),
    "validation_rmse": float(best_model_row["RMSE"]),
    "validation_r2": float(best_model_row["R2"])
}

final_model_path = os.path.join(
    model_directory,
    "ev_demand_final_model.pkl"
)

joblib.dump(
    final_artifact,
    final_model_path
)

print(
    f"\nProduction model selected: "
    f"{best_model_name}"
)

print(
    f"Validation MAE : "
    f"{best_model_row['MAE']:.6f} MW"
)

print(
    f"Validation RMSE: "
    f"{best_model_row['RMSE']:.6f} MW"
)

print(
    f"Validation R2  : "
    f"{best_model_row['R2']:.6f}"
)

print(
    f"\nFinal model saved to:"
    f"\n{final_model_path}"
)


# ============================================================
# 33. SAVE FINAL MODEL METADATA
# ============================================================

final_metadata = pd.DataFrame([{
    "model": best_model_name,
    "target": "next_hour_ev_charging_demand",
    "forecast_horizon_minutes": 60,
    "validation_mae_mw": float(best_model_row["MAE"]),
    "validation_rmse_mw": float(best_model_row["RMSE"]),
    "validation_r2": float(best_model_row["R2"]),
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "live_test_rows": len(test_df)
}])

metadata_path = os.path.join(
    model_directory,
    "ev_demand_final_model_metadata.csv"
)

final_metadata.to_csv(
    metadata_path,
    index=False
)

print(
    f"Final model metadata saved to:"
    f"\n{metadata_path}"
)


# ============================================================
# 34. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EV CHARGING DEMAND FORECASTING COMPLETE")
print("=" * 70)

print(f"Features       : {len(features)}")
print(f"Training rows  : {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Live Test rows : {len(test_df):,}")

print("\nModel Comparison:")

print(
    comparison_df.to_string(index=False)
)

print(
    f"\nFINAL PRODUCTION MODEL: "
    f"{best_model_name}"
)

print(
    f"MAE  : {best_model_row['MAE']:.6f} MW"
)

print(
    f"RMSE : {best_model_row['RMSE']:.6f} MW"
)

print(
    f"R2   : {best_model_row['R2']:.6f}"
)

print("\nFinal model file:")
print(final_model_path)

print("\nDone.")
