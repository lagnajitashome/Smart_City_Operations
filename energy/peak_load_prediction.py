# ============================================================
# PEAK LOAD PREDICTION
# Smart City Operations Project
#
# Models:
# 1. Random Forest Classifier
# 2. XGBoost Classifier
# 3. LightGBM Classifier
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, URL

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
    f"{city_df['electricity_demand_mw'].notna().sum():,}"
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
# 8. CYCLICAL FEATURES
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
# 9. DEMAND LAG FEATURES
# ============================================================

print("Creating demand lag features...")

for lag in [1, 5, 15, 30, 60]:

    city_df[
        f"demand_lag_{lag}"
    ] = (
        city_df[
            "electricity_demand_mw"
        ]
        .shift(lag)
    )


# ============================================================
# 10. SOLAR / EV LAGS
# ============================================================

city_df["solar_lag_60"] = (
    city_df[
        "solar_generation_mw"
    ]
    .shift(60)
)

city_df["ev_lag_60"] = (
    city_df[
        "ev_charging_demand_mw"
    ]
    .shift(60)
)


# ============================================================
# 11. ROLLING FEATURES
# ============================================================

print("Creating rolling demand features...")

city_df["demand_rolling_mean_15"] = (
    city_df[
        "electricity_demand_mw"
    ]
    .shift(1)
    .rolling(15)
    .mean()
)

city_df["demand_rolling_mean_30"] = (
    city_df[
        "electricity_demand_mw"
    ]
    .shift(1)
    .rolling(30)
    .mean()
)

city_df["demand_rolling_mean_60"] = (
    city_df[
        "electricity_demand_mw"
    ]
    .shift(1)
    .rolling(60)
    .mean()
)

city_df["demand_std_60"] = (
    city_df[
        "electricity_demand_mw"
    ]
    .shift(1)
    .rolling(60)
    .std()
)


# ============================================================
# 12. NEXT-HOUR DEMAND
# ============================================================

city_df["next_hour_demand"] = (
    city_df[
        "electricity_demand_mw"
    ]
    .shift(-60)
)


# ============================================================
# 13. FEATURE LIST
# ============================================================

features = [

    # Time
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",

    # Cyclical
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

    # Rolling
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
# 14. IDENTIFY HISTORICAL BACKFILL
# ============================================================

first_timestamp = city_df.index.min()

backfill_end = (
    first_timestamp +
    pd.Timedelta(days=14)
)


# ============================================================
# 15. CREATE MODEL DATASET
# ============================================================

model_df = city_df[
    features + [
        "electricity_demand_mw",
        "next_hour_demand"
    ]
].copy()


# Replace infinity

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove missing values

model_df = model_df.dropna()


# ============================================================
# 16. CHRONOLOGICAL SPLIT
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
    (model_df.index >= validation_start) &
    (model_df.index < backfill_end)
].copy()


test_df = model_df[
    model_df.index >= backfill_end
].copy()


print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

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
# 17. CALCULATE PEAK THRESHOLD
#
# 90th percentile of TRAINING demand.
# ============================================================

peak_threshold = (
    train_df[
        "electricity_demand_mw"
    ]
    .quantile(0.90)
)


print("\n" + "=" * 60)
print("PEAK LOAD THRESHOLD")
print("=" * 60)

print(
    f"90th percentile threshold: "
    f"{peak_threshold:.2f} MW"
)


# ============================================================
# 18. CREATE PEAK TARGET
#
# 1 = Peak
# 0 = Normal
# ============================================================

train_df["peak_target"] = (
    train_df[
        "next_hour_demand"
    ] >= peak_threshold
).astype(int)


validation_df["peak_target"] = (
    validation_df[
        "next_hour_demand"
    ] >= peak_threshold
).astype(int)


if len(test_df) > 0:

    test_df["peak_target"] = (
        test_df[
            "next_hour_demand"
        ] >= peak_threshold
    ).astype(int)


# ============================================================
# 19. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTraining:")
print(
    train_df[
        "peak_target"
    ]
    .value_counts()
    .sort_index()
)

print(
    f"\nTraining peak percentage: "
    f"{train_df['peak_target'].mean() * 100:.2f}%"
)

print(
    f"Validation peak percentage: "
    f"{validation_df['peak_target'].mean() * 100:.2f}%"
)


# ============================================================
# 20. CREATE X AND y
# ============================================================

X_train = train_df[
    features
]

y_train = train_df[
    "peak_target"
]


X_val = validation_df[
    features
]

y_val = validation_df[
    "peak_target"
]


if len(test_df) > 0:

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "peak_target"
    ]

else:

    X_test = None
    y_test = None


# ============================================================
# 21. CLASS BALANCE
# ============================================================

positive_count = (
    y_train == 1
).sum()

negative_count = (
    y_train == 0
).sum()


if positive_count == 0:

    raise ValueError(
        "Training data contains no peak observations."
    )


scale_pos_weight = (
    negative_count /
    positive_count
)


print("\nClass balance:")

print(
    f"Normal samples : {negative_count:,}"
)

print(
    f"Peak samples   : {positive_count:,}"
)

print(
    f"XGBoost scale_pos_weight: "
    f"{scale_pos_weight:.3f}"
)


# ============================================================
# 22. MODEL DEFINITIONS
# ============================================================

print("\n" + "=" * 60)
print("CREATING MODELS")
print("=" * 60)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgboost_model = XGBClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=7,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------

lightgbm_model = LGBMClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=8,

    num_leaves=31,

    min_child_samples=20,

    subsample=0.8,

    colsample_bytree=0.8,

    class_weight="balanced",

    objective="binary",

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


models = {

    "Random Forest": random_forest,

    "XGBoost": xgboost_model,

    "LightGBM": lightgbm_model

}


# ============================================================
# 23. METRIC FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val
):

    print("\n" + "=" * 60)

    print(
        f"TRAINING {model_name.upper()}"
    )

    print("=" * 60)


    # Train

    if model_name == "XGBoost":

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ],

            verbose=False
        )

    else:

        model.fit(
            X_train,
            y_train
        )


    print(
        f"{model_name} training completed."
    )


    # Predictions

    y_pred = model.predict(
        X_val
    )

    y_probability = (
        model.predict_proba(
            X_val
        )[:, 1]
    )


    # Metrics

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )


    if len(
        np.unique(y_val)
    ) == 2:

        roc_auc = roc_auc_score(
            y_val,
            y_probability
        )

    else:

        roc_auc = np.nan


    # Confusion matrix

    cm = confusion_matrix(
        y_val,
        y_pred
    )


    print(
        f"\n{model_name} RESULTS"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "Normal Load",
                "Peak Load"
            ],
            zero_division=0
        )
    )


    return {

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc,

        "predictions": y_pred,

        "probabilities": y_probability,

        "confusion_matrix": cm

    }


# ============================================================
# 24. TRAIN ALL MODELS
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
# 25. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PEAK LOAD MODEL COMPARISON")
print("=" * 70)


comparison_df = pd.DataFrame({

    "Model": [
        result["model"]
        for result in results
    ],

    "Accuracy": [
        result["accuracy"]
        for result in results
    ],

    "Precision": [
        result["precision"]
        for result in results
    ],

    "Recall": [
        result["recall"]
        for result in results
    ],

    "F1 Score": [
        result["f1_score"]
        for result in results
    ],

    "ROC-AUC": [
        result["roc_auc"]
        for result in results
    ]

})


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 26. CREATE VALIDATION PREDICTION TABLE
# ============================================================

validation_predictions = pd.DataFrame({

    "timestamp":
        validation_df.index,

    "current_demand_mw":
        validation_df[
            "electricity_demand_mw"
        ].values,

    "actual_next_hour_demand_mw":
        validation_df[
            "next_hour_demand"
        ].values,

    "actual_peak":
        y_val.values

})


# Add predictions from each model

for result in results:

    model_name = result["model"]

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    validation_predictions[
        f"{safe_name}_prediction"
    ] = result[
        "predictions"
    ]

    validation_predictions[
        f"{safe_name}_probability"
    ] = result[
        "probabilities"
    ]


# ============================================================
# 27. CREATE MODEL DIRECTORY
# ============================================================

model_directory = "models/energy"

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 28. SAVE EACH MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)


model_file_names = {

    "Random Forest":
        "peak_load_random_forest.pkl",

    "XGBoost":
        "peak_load_xgboost.pkl",

    "LightGBM":
        "peak_load_lightgbm.pkl"

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

        "model_name":
            model_name,

        "peak_threshold_mw":
            float(peak_threshold),

        "peak_percentile":
            90,

        "target":
            "next_hour_peak_load"

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
# 29. SAVE MODEL COMPARISON
# ============================================================

comparison_df["peak_threshold_mw"] = (
    peak_threshold
)


comparison_path = os.path.join(
    model_directory,
    "peak_load_model_comparison.csv"
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
# 30. SAVE VALIDATION PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    model_directory,
    "peak_load_validation_predictions.csv"
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
# 31. SAVE FEATURE IMPORTANCE FOR EACH MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING FEATURE IMPORTANCE")
print("=" * 60)


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

        f"peak_load_{safe_name}"
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
# 32. CURRENT PEAK-LOAD STATUS
# ============================================================

print("\n" + "=" * 60)
print("CURRENT PEAK-LOAD STATUS")
print("=" * 60)


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


    current_demand = float(

        city_df.loc[
            latest_timestamp,
            "electricity_demand_mw"
        ]

    )


    print(
        f"Timestamp       : "
        f"{latest_timestamp}"
    )

    print(
        f"Current Demand  : "
        f"{current_demand:.2f} MW"
    )

    print(
        f"Peak Threshold  : "
        f"{peak_threshold:.2f} MW"
    )


    current_status_rows = []


    # --------------------------------------------------------
    # Predictions from all three models
    # --------------------------------------------------------

    for model_name, model in trained_models.items():

        prediction = model.predict(
            latest_features
        )[0]


        probability = (
            model.predict_proba(
                latest_features
            )[0, 1]
        )


        if prediction == 1:

            status = "PEAK EXPECTED"

        else:

            status = "NORMAL LOAD"


        print(
            f"\n{model_name}"
        )

        print(
            f"Probability : "
            f"{probability * 100:.2f}%"
        )

        print(
            f"Prediction  : "
            f"{status}"
        )


        current_status_rows.append({

            "timestamp":
                latest_timestamp,

            "model":
                model_name,

            "current_demand_mw":
                current_demand,

            "peak_threshold_mw":
                peak_threshold,

            "peak_probability":
                probability,

            "peak_status":
                status

        })


    # --------------------------------------------------------
    # Save current status
    # --------------------------------------------------------

    current_status_df = pd.DataFrame(
        current_status_rows
    )


    current_status_path = os.path.join(

        model_directory,

        "current_peak_load_status.csv"
    )


    current_status_df.to_csv(
        current_status_path,
        index=False
    )


    print(
        f"\nCurrent status saved to:"
        f"\n{current_status_path}"
    )


else:

    print(
        "No valid latest observation "
        "available for prediction."
    )


# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PEAK LOAD MODEL COMPARISON COMPLETE")
print("=" * 70)

print(
    f"Peak threshold : "
    f"{peak_threshold:.2f} MW"
)

print(
    f"Training rows  : "
    f"{len(X_train):,}"
)

print(
    f"Validation rows: "
    f"{len(X_val):,}"
)

print("\nModel Comparison:")

print(
    comparison_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nNo final peak-load model has been selected yet."
)

print(
    "We will select the production model "
    "after reviewing these results."
)

print("\nDone.")