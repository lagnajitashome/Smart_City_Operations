import os
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# 2. DATABASE
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
# 3. LOAD DATA
# ============================================================

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

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ============================================================
# 4. CITY LEVEL AGGREGATION
# ============================================================

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


# ============================================================
# 5. CONTINUOUS TIMELINE
# ============================================================

full_index = pd.date_range(
    start=city_df.index.min(),
    end=city_df.index.max(),
    freq="min"
)

city_df = city_df.reindex(full_index)

city_df.index.name = "timestamp"


# ============================================================
# 6. TIME FEATURES
# ============================================================

city_df["hour"] = city_df.index.hour

city_df["minute"] = city_df.index.minute

city_df["day_of_week"] = (
    city_df.index.dayofweek
)

city_df["is_weekend"] = (
    city_df["day_of_week"] >= 5
).astype(int)


# ============================================================
# 7. CYCLICAL FEATURES
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
# 8. EV LAGS
# ============================================================

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
        ].shift(lag)
    )


# ============================================================
# 9. ELECTRICITY DEMAND LAGS
# ============================================================

for lag in [60, 1440]:

    city_df[
        f"electricity_demand_lag_{lag}"
    ] = (
        city_df[
            "electricity_demand_mw"
        ].shift(lag)
    )


# ============================================================
# 10. SOLAR LAGS
# ============================================================

city_df["solar_lag_60"] = (
    city_df[
        "solar_generation_mw"
    ].shift(60)
)

city_df["solar_lag_1440"] = (
    city_df[
        "solar_generation_mw"
    ].shift(1440)
)


# ============================================================
# 11. ROLLING FEATURES
#
# IMPORTANT:
# EXACTLY SAME AS TRAINING
# ============================================================

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
# 12. CHANGE FEATURES
# ============================================================

city_df["ev_change_15"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    -
    city_df[
        "ev_charging_demand_mw"
    ].shift(15)
)

city_df["ev_change_60"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    -
    city_df[
        "ev_charging_demand_mw"
    ].shift(60)
)


# ============================================================
# 13. TARGET
# ============================================================

city_df["next_hour_ev_demand"] = (
    city_df[
        "ev_charging_demand_mw"
    ].shift(-60)
)


# ============================================================
# 14. EXACT TRAINING FEATURES
#
# 30 FEATURES
# ============================================================

features = [

    # Time - 4
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",

    # Cyclical - 4
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",

    # EV lags - 9
    "ev_demand_lag_1",
    "ev_demand_lag_5",
    "ev_demand_lag_15",
    "ev_demand_lag_30",
    "ev_demand_lag_60",
    "ev_demand_lag_120",
    "ev_demand_lag_180",
    "ev_demand_lag_360",
    "ev_demand_lag_1440",

    # Electricity lags - 2
    "electricity_demand_lag_60",
    "electricity_demand_lag_1440",

    # Solar lags - 2
    "solar_lag_60",
    "solar_lag_1440",

    # Rolling - 4
    "ev_rolling_mean_15",
    "ev_rolling_mean_30",
    "ev_rolling_mean_60",
    "ev_rolling_std_60",

    # Changes - 2
    "ev_change_15",
    "ev_change_60",

    # Current known variables - 3
    "electricity_demand_mw",
    "solar_generation_mw",
    "transformer_load_pct"
]


print()
print("=" * 75)
print("EV CHARGING DEMAND - LIVE MODEL TEST")
print("=" * 75)

print()
print("Training feature count :", len(features))


# ============================================================
# 15. LIVE PERIOD
# ============================================================

historical_start = city_df.index.min()

live_start = (
    historical_start
    + pd.Timedelta(days=14)
)

print()
print("Historical start :", historical_start)
print("Live period      :", live_start)
print("Latest data      :", city_df.index.max())


# ============================================================
# 16. CREATE TEST DATA
# ============================================================

test_data = city_df[
    features
    +
    [
        "next_hour_ev_demand"
    ]
].copy()


test_data = test_data[
    test_data.index >= live_start
]


# Replace infinite values

test_data = test_data.replace(
    [np.inf, -np.inf],
    np.nan
)


print()
print(
    "Raw live rows :",
    len(test_data)
)


# ============================================================
# 17. REQUIRE NEXT-HOUR ACTUAL
# ============================================================

test_data = test_data[
    test_data[
        "next_hour_ev_demand"
    ].notna()
]

print(
    "Live rows with next-hour actual :",
    len(test_data)
)


# ============================================================
# 18. REMOVE ROWS WITH MISSING FEATURES
#
# NO INTERPOLATION
# NO FORWARD FILL
# NO BACKWARD FILL
# ============================================================

test_data = test_data.dropna(
    subset=features
)


print(
    "Valid live test rows :",
    len(test_data)
)


if len(test_data) == 0:

    print()
    print(
        "NO VALID LIVE TEST ROWS."
    )

    raise SystemExit


# ============================================================
# 19. MODEL PATHS
# ============================================================

model_paths = {

    "Random Forest":
        "models/energy/ev_demand_final_model.pkl",

    "XGBoost":
        "models/energy/ev_demand_xgboost.pkl",

    "LightGBM":
        "models/energy/ev_demand_lightgbm.pkl"
}


# ============================================================
# 20. TEST MODELS
# ============================================================

results = []

all_predictions = []


for model_name, model_path in model_paths.items():

    print()
    print("=" * 75)
    print(
        f"TESTING {model_name}"
    )
    print("=" * 75)

    print(
        "Model file:",
        model_path
    )


    if not os.path.exists(model_path):

        print(
            "MODEL FILE NOT FOUND"
        )

        continue


    # --------------------------------------------------------
    # LOAD ARTIFACT
    # --------------------------------------------------------

    artifact = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # SAVED FILE IS A DICTIONARY
    # --------------------------------------------------------

    if isinstance(
        artifact,
        dict
    ):

        model = artifact[
            "model"
        ]

        saved_features = artifact[
            "features"
        ]

    else:

        model = artifact

        saved_features = features


    print()
    print(
        "Saved feature count :",
        len(saved_features)
    )


    # --------------------------------------------------------
    # CHECK FEATURE LIST
    # --------------------------------------------------------

    if list(saved_features) != list(features):

        print()
        print(
            "WARNING: FEATURE LIST DIFFERENCE"
        )

        print()
        print(
            "Saved features:"
        )

        print(
            saved_features
        )

        print()
        print(
            "Expected features:"
        )

        print(
            features
        )

        print()
        print(
            "Skipping this model."
        )

        continue


    print(
        "Feature alignment : OK"
    )


    # --------------------------------------------------------
    # CREATE X AND y
    # --------------------------------------------------------

    X_test = test_data[
        saved_features
    ].copy()

    y_test = test_data[
        "next_hour_ev_demand"
    ].copy()


    print(
        "Test rows :",
        len(X_test)
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(
        X_test
    )

    prediction = np.asarray(
        prediction
    ).reshape(-1)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )


    # MAPE

    non_zero = (
        np.abs(
            y_test.values
        ) > 0.01
    )

    if np.any(non_zero):

        mape = np.mean(
            np.abs(
                (
                    y_test.values[
                        non_zero
                    ]
                    -
                    prediction[
                        non_zero
                    ]
                )
                /
                y_test.values[
                    non_zero
                ]
            )
        ) * 100

    else:

        mape = np.nan


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"{model_name.upper()} LIVE TEST RESULTS"
    )

    print(
        f"Test rows : {len(X_test)}"
    )

    print(
        f"MAE       : {mae:.4f} MW"
    )

    print(
        f"RMSE      : {rmse:.4f} MW"
    )

    print(
        f"R²        : {r2:.4f}"
    )

    if np.isnan(mape):

        print(
            "MAPE      : N/A"
        )

    else:

        print(
            f"MAPE      : {mape:.4f}%"
        )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model":
            model_name,

        "Test_Rows":
            len(X_test),

        "MAE_MW":
            mae,

        "RMSE_MW":
            rmse,

        "R2":
            r2,

        "MAPE_percent":
            mape
    })


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    model_predictions = pd.DataFrame({

        "timestamp":
            X_test.index,

        "actual_next_hour_ev_demand_mw":
            y_test.values,

        "predicted_next_hour_ev_demand_mw":
            prediction,

        "absolute_error_mw":
            np.abs(
                y_test.values
                -
                prediction
            ),

        "model":
            model_name
    })


    all_predictions.append(
        model_predictions
    )


# ============================================================
# 21. FINAL RESULTS
# ============================================================

print()
print("=" * 80)
print("FINAL LIVE TEST RESULTS")
print("=" * 80)


if len(results) == 0:

    print(
        "No models were successfully tested."
    )

else:

    results_df = pd.DataFrame(
        results
    )

    print()
    print(
        results_df.to_string(
            index=False
        )
    )


# ============================================================
# 22. SAVE RESULTS
# ============================================================

os.makedirs(
    "models/energy",
    exist_ok=True
)


if len(results) > 0:

    results_df.to_csv(

        "models/energy/"
        "ev_demand_live_test_metrics.csv",

        index=False
    )


if len(all_predictions) > 0:

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df.to_csv(

        "models/energy/"
        "ev_demand_live_test_predictions.csv",

        index=False
    )


# ============================================================
# 23. COMPLETE
# ============================================================

print()
print("=" * 80)
print("LIVE TEST COMPLETE")
print("=" * 80)

print()
print(
    "Metrics:"
)

print(
    "models/energy/"
    "ev_demand_live_test_metrics.csv"
)

print()
print(
    "Predictions:"
)

print(
    "models/energy/"
    "ev_demand_live_test_predictions.csv"
)

print()
print(
    "Only genuine live observations were used."
)

print(
    "No interpolation or artificial values were introduced."
)