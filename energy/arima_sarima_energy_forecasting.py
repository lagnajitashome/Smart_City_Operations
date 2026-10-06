# ============================================================
# ARIMA + SARIMA ENERGY DEMAND FORECASTING
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

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

print("\n" + "=" * 60)
print("LOADING ENERGY DATA")
print("=" * 60)

query = """
SELECT
    timestamp,
    location_id,
    electricity_demand_mw
FROM energy_data
ORDER BY timestamp;
"""

df = pd.read_sql(query, engine)

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
# 5. CREATE CITY-LEVEL DEMAND
# ============================================================

print("\n" + "=" * 60)
print("CREATING CITY-LEVEL DEMAND")
print("=" * 60)

# IMPORTANT:
# Keep this as a DATAFRAME rather than a Series.

city_df = (
    df.groupby("timestamp")
    .agg(
        electricity_demand_mw=(
            "electricity_demand_mw",
            "sum"
        )
    )
    .sort_index()
)

print(
    f"City-level minute records: "
    f"{len(city_df):,}"
)


# ============================================================
# 6. IDENTIFY 14-DAY HISTORICAL BACKFILL
# ============================================================

first_timestamp = city_df.index.min()

historical_end = (
    first_timestamp +
    pd.Timedelta(days=14)
)

historical_df = city_df[
    city_df.index < historical_end
].copy()


print("\nHistorical period:")

print(
    f"{historical_df.index.min()} → "
    f"{historical_df.index.max()}"
)

print(
    f"Historical records: "
    f"{len(historical_df):,}"
)


# ============================================================
# 7. CHECK HISTORICAL DATA COMPLETENESS
# ============================================================

print("\n" + "=" * 60)
print("CHECKING HISTORICAL DATA")
print("=" * 60)

expected_minutes = 14 * 24 * 60

print(
    f"Expected historical minutes : "
    f"{expected_minutes:,}"
)

print(
    f"Actual historical minutes   : "
    f"{len(historical_df):,}"
)


# ============================================================
# 8. CREATE HOURLY ENERGY DEMAND
# ============================================================

print("\n" + "=" * 60)
print("CREATING HOURLY ENERGY DEMAND")
print("=" * 60)

# Average demand within each hour

hourly_df = (
    historical_df
    .resample("1h")
    .mean()
)


# ============================================================
# 9. CHECK NUMBER OF MINUTES PER HOUR
# ============================================================

minute_count = (
    historical_df
    .resample("1h")
    .count()
)


# Keep only complete hours

complete_hours = (
    minute_count["electricity_demand_mw"] >= 60
)

hourly_df = hourly_df[
    complete_hours
].copy()


# Remove missing/infinite values

hourly_df = hourly_df.replace(
    [np.inf, -np.inf],
    np.nan
)

hourly_df = hourly_df.dropna()


print(
    f"Complete hourly observations: "
    f"{len(hourly_df):,}"
)

print(
    f"Hourly range: "
    f"{hourly_df.index.min()} → "
    f"{hourly_df.index.max()}"
)


# ============================================================
# 10. CHECK DATA SUFFICIENCY
# ============================================================

if len(hourly_df) < 100:

    raise ValueError(
        "Not enough hourly observations "
        "for ARIMA/SARIMA."
    )


# ============================================================
# 11. TRAIN / VALIDATION SPLIT
#
# Last 48 hours = validation
# Everything before = training
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 60)

VALIDATION_HOURS = 48

if len(hourly_df) <= VALIDATION_HOURS:

    raise ValueError(
        "Not enough hourly observations "
        "for 48-hour validation."
    )


train_df = hourly_df.iloc[
    :-VALIDATION_HOURS
].copy()

validation_df = hourly_df.iloc[
    -VALIDATION_HOURS:
].copy()


print(
    f"Training hours   : "
    f"{len(train_df)}"
)

print(
    f"Validation hours : "
    f"{len(validation_df)}"
)

print(
    f"\nTraining period:"
)

print(
    f"{train_df.index.min()} → "
    f"{train_df.index.max()}"
)

print(
    f"\nValidation period:"
)

print(
    f"{validation_df.index.min()} → "
    f"{validation_df.index.max()}"
)


# ============================================================
# 12. EXTRACT TARGET
# ============================================================

y_train = train_df[
    "electricity_demand_mw"
].copy()

y_validation = validation_df[
    "electricity_demand_mw"
].copy()


# ============================================================
# 13. ARIMA
#
# ARIMA(2,1,2)
#
# p = 2
# d = 1
# q = 2
# ============================================================

print("\n" + "=" * 60)
print("TRAINING ARIMA")
print("=" * 60)

print(
    "ARIMA order: (2, 1, 2)"
)

arima_model = ARIMA(
    y_train,
    order=(2, 1, 2)
)

arima_result = arima_model.fit()

print(
    "ARIMA training completed."
)


# ============================================================
# 14. ARIMA FORECAST
# ============================================================

print(
    "\nGenerating ARIMA validation forecast..."
)

arima_forecast = (
    arima_result
    .forecast(
        steps=VALIDATION_HOURS
    )
)

arima_forecast = np.asarray(
    arima_forecast
)


# ============================================================
# 15. ARIMA METRICS
# ============================================================

arima_mae = mean_absolute_error(
    y_validation,
    arima_forecast
)

arima_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        arima_forecast
    )
)

arima_r2 = r2_score(
    y_validation,
    arima_forecast
)


print("\n==============================")
print("ARIMA VALIDATION RESULTS")
print("==============================")

print(
    f"MAE  : {arima_mae:.3f} MW"
)

print(
    f"RMSE : {arima_rmse:.3f} MW"
)

print(
    f"R²   : {arima_r2:.4f}"
)


# ============================================================
# 16. SARIMA
#
# SARIMA:
#
# (p,d,q)(P,D,Q,s)
#
# Non-seasonal:
# (1,1,1)
#
# Seasonal:
# (1,1,1,24)
#
# 24 = 24-hour daily seasonality
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SARIMA")
print("=" * 60)

print(
    "SARIMA order   : (1,1,1)"
)

print(
    "Seasonal order : (1,1,1,24)"
)


sarima_model = SARIMAX(

    y_train,

    order=(1, 1, 1),

    seasonal_order=(1, 1, 1, 24),

    enforce_stationarity=False,

    enforce_invertibility=False
)


sarima_result = sarima_model.fit(
    disp=False
)


print(
    "SARIMA training completed."
)


# ============================================================
# 17. SARIMA FORECAST
# ============================================================

print(
    "\nGenerating SARIMA validation forecast..."
)

sarima_forecast = (
    sarima_result
    .forecast(
        steps=VALIDATION_HOURS
    )
)

sarima_forecast = np.asarray(
    sarima_forecast
)


# ============================================================
# 18. SARIMA METRICS
# ============================================================

sarima_mae = mean_absolute_error(
    y_validation,
    sarima_forecast
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        sarima_forecast
    )
)

sarima_r2 = r2_score(
    y_validation,
    sarima_forecast
)


print("\n==============================")
print("SARIMA VALIDATION RESULTS")
print("==============================")

print(
    f"MAE  : {sarima_mae:.3f} MW"
)

print(
    f"RMSE : {sarima_rmse:.3f} MW"
)

print(
    f"R²   : {sarima_r2:.4f}"
)


# ============================================================
# 19. ARIMA VS SARIMA COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("ARIMA VS SARIMA")
print("=" * 60)

comparison_df = pd.DataFrame({

    "model": [
        "ARIMA",
        "SARIMA"
    ],

    "MAE": [
        arima_mae,
        sarima_mae
    ],

    "RMSE": [
        arima_rmse,
        sarima_rmse
    ],

    "R2": [
        arima_r2,
        sarima_r2
    ]

})


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "timestamp":
        validation_df.index,

    "actual_demand_mw":
        y_validation.values,

    "arima_prediction_mw":
        arima_forecast,

    "sarima_prediction_mw":
        sarima_forecast

})


# ============================================================
# 21. CREATE MODEL DIRECTORY
# ============================================================

model_directory = "models/energy"

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 22. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    model_directory,
    "arima_sarima_validation_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    f"\nValidation predictions saved to:"
    f"\n{prediction_path}"
)


# ============================================================
# 23. SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    model_directory,
    "arima_sarima_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"Model comparison saved to:"
    f"\n{comparison_path}"
)


# ============================================================
# 24. REFIT FINAL ARIMA ON ALL HISTORICAL DATA
# ============================================================

print("\n" + "=" * 60)
print("REFITTING FINAL ARIMA MODEL")
print("=" * 60)

final_arima_model = ARIMA(
    hourly_df["electricity_demand_mw"],
    order=(2, 1, 2)
)

final_arima_result = (
    final_arima_model.fit()
)


print(
    "Final ARIMA model trained."
)


# ============================================================
# 25. REFIT FINAL SARIMA
# ============================================================

print("\n" + "=" * 60)
print("REFITTING FINAL SARIMA MODEL")
print("=" * 60)

final_sarima_model = SARIMAX(

    hourly_df["electricity_demand_mw"],

    order=(1, 1, 1),

    seasonal_order=(1, 1, 1, 24),

    enforce_stationarity=False,

    enforce_invertibility=False
)

final_sarima_result = (
    final_sarima_model.fit(
        disp=False
    )
)


print(
    "Final SARIMA model trained."
)


# ============================================================
# 26. NEXT-HOUR FORECAST
# ============================================================

print("\n" + "=" * 60)
print("NEXT-HOUR ENERGY DEMAND FORECAST")
print("=" * 60)

next_hour_arima = (
    final_arima_result
    .forecast(
        steps=1
    )
)

next_hour_sarima = (
    final_sarima_result
    .forecast(
        steps=1
    )
)


next_hour_arima_value = float(
    next_hour_arima.iloc[0]
)

next_hour_sarima_value = float(
    next_hour_sarima.iloc[0]
)


print(
    f"ARIMA  : "
    f"{next_hour_arima_value:.2f} MW"
)

print(
    f"SARIMA : "
    f"{next_hour_sarima_value:.2f} MW"
)


# ============================================================
# 27. SAVE ARIMA MODEL
# ============================================================

arima_model_path = os.path.join(
    model_directory,
    "energy_demand_arima.pkl"
)

joblib.dump(
    final_arima_result,
    arima_model_path
)

print(
    f"\nARIMA model saved to:"
    f"\n{arima_model_path}"
)


# ============================================================
# 28. SAVE SARIMA MODEL
# ============================================================

sarima_model_path = os.path.join(
    model_directory,
    "energy_demand_sarima.pkl"
)

joblib.dump(
    final_sarima_result,
    sarima_model_path
)

print(
    f"SARIMA model saved to:"
    f"\n{sarima_model_path}"
)


# ============================================================
# 29. SAVE NEXT-HOUR FORECAST
# ============================================================

forecast_df = pd.DataFrame({

    "model": [
        "ARIMA",
        "SARIMA"
    ],

    "forecast_demand_mw": [
        next_hour_arima_value,
        next_hour_sarima_value
    ]

})


forecast_path = os.path.join(
    model_directory,
    "arima_sarima_next_hour_forecast.csv"
)

forecast_df.to_csv(
    forecast_path,
    index=False
)

print(
    f"Next-hour forecast saved to:"
    f"\n{forecast_path}"
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ARIMA + SARIMA ENERGY FORECASTING COMPLETE")
print("=" * 60)

print(
    f"Historical hourly observations : "
    f"{len(hourly_df)}"
)

print(
    f"Training observations          : "
    f"{len(y_train)}"
)

print(
    f"Validation observations        : "
    f"{len(y_validation)}"
)

print("\nARIMA")

print(
    f"MAE  = {arima_mae:.3f} MW"
)

print(
    f"RMSE = {arima_rmse:.3f} MW"
)

print(
    f"R²   = {arima_r2:.4f}"
)

print("\nSARIMA")

print(
    f"MAE  = {sarima_mae:.3f} MW"
)

print(
    f"RMSE = {sarima_rmse:.3f} MW"
)

print(
    f"R²   = {sarima_r2:.4f}"
)

print("\nNext-hour forecast")

print(
    f"ARIMA  = "
    f"{next_hour_arima_value:.2f} MW"
)

print(
    f"SARIMA = "
    f"{next_hour_sarima_value:.2f} MW"
)

print("\nDone.")