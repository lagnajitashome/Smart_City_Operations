from __future__ import annotations

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV

from database.connection import engine

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "parking_occupancy_model.pkl"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "parking_model_tuning_comparison.csv"
)

TEST_FRACTION = 0.20

RANDOM_STATE = 42

CV_SPLITS = 3

RF_N_ITER = 20

XGB_N_ITER = 25


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    "capacity",
    "occupied_spaces",
    "available_spaces",
    "occupancy_pct",

    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    "hour_sin",
    "hour_cos",

    "dow_sin",
    "dow_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_168",

    "change_1h",
    "change_3h",

    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",

    "rolling_std_24",

    "is_s_source",
]


# ============================================================
# LOAD UNIFIED PARKING DATA
# ============================================================

def load_data() -> pd.DataFrame:

    query = text(
        """
        SELECT
            h.parking_id,
            h.timestamp,
            h.capacity,
            h.occupied_spaces,
            h.available_spaces,
            h.occupancy_pct,
            h.source,

            p.parking_name,
            p.parking_type,
            p.latitude,
            p.longitude

        FROM parking_occupancy_history h

        JOIN parking_locations p
            ON p.id = h.parking_id

        WHERE h.source IN
        (
            'S',
            'OpenStreetMap',
            'OpenStreetMap-Geofabrik'
        )

        ORDER BY
            h.timestamp,
            h.parking_id;
        """
    )

    with engine.connect() as conn:

        df = pd.read_sql(
            query,
            conn
        )

    if df.empty:

        raise RuntimeError(
            "No unified parking occupancy history found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = (
        df
        .sort_values(
            [
                "parking_id",
                "timestamp"
            ]
        )
        .copy()
    )

    group = data.groupby(
        "parking_id",
        sort=False
    )

    # --------------------------------------------------------
    # Calendar features
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
    # Cyclic time features
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

        data[f"lag_{lag}"] = (
            group["occupancy_pct"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Change features
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
    # Rolling features
    #
    # Shift first so only historical observations are used.
    # --------------------------------------------------------

    shifted = (
        data
        .groupby(
            "parking_id",
            sort=False
        )["occupancy_pct"]
        .shift(1)
    )

    shifted_group = shifted.groupby(
        data["parking_id"],
        sort=False
    )

    data["rolling_mean_3"] = (
        shifted_group
        .rolling(3)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    data["rolling_mean_6"] = (
        shifted_group
        .rolling(6)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    data["rolling_mean_24"] = (
        shifted_group
        .rolling(24)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    data["rolling_std_24"] = (
        shifted_group
        .rolling(24)
        .std()
        .reset_index(
            level=0,
            drop=True
        )
    )

    # --------------------------------------------------------
    # Source feature
    # --------------------------------------------------------

    data["is_s_source"] = (
        data["source"] == "S"
    ).astype(int)

    # --------------------------------------------------------
    # Target
    #
    # Occupancy one hour into the future.
    # --------------------------------------------------------

    data["target_occupancy"] = (
        group["occupancy_pct"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    return data


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def time_split(
    data: pd.DataFrame
):

    data = (
        data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    unique_times = (
        data["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(unique_times)
        * (1 - TEST_FRACTION)
    )

    cutoff = unique_times[
        split_index
    ]

    train = data[
        data["timestamp"] < cutoff
    ].copy()

    test = data[
        data["timestamp"] >= cutoff
    ].copy()

    return (
        train,
        test
    )


# ============================================================
# TIME-AWARE CV SPLITS
# ============================================================

def create_time_cv_splits(
    train_data: pd.DataFrame,
    n_splits: int = 3,
):

    train_data = (
        train_data
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    times = (
        train_data["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_times = len(times)

    # Initial history block plus n_splits validation blocks
    block_size = (
        n_times
        // (n_splits + 1)
    )

    splits = []

    for fold in range(
        n_splits
    ):

        train_end = (
            block_size
            * (fold + 1)
        )

        valid_start = train_end

        valid_end = (
            block_size
            * (fold + 2)
        )

        if fold == n_splits - 1:

            valid_end = n_times

        train_times = set(
            times.iloc[
                :train_end
            ]
        )

        valid_times = set(
            times.iloc[
                valid_start:valid_end
            ]
        )

        train_idx = np.flatnonzero(
            train_data["timestamp"]
            .isin(train_times)
            .to_numpy()
        )

        valid_idx = np.flatnonzero(
            train_data["timestamp"]
            .isin(valid_times)
            .to_numpy()
        )

        if (
            len(train_idx) > 0
            and len(valid_idx) > 0
        ):

            splits.append(
                (
                    train_idx,
                    valid_idx
                )
            )

    if not splits:

        raise RuntimeError(
            "Unable to create time-aware CV folds."
        )

    return splits


# ============================================================
# METRICS
# ============================================================

def evaluate(
    model_name,
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    y_pred = np.clip(
        y_pred,
        0,
        100
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    denominator = np.where(
        np.abs(y_true) < 1e-8,
        1,
        np.abs(y_true)
    )

    mape = (
        np.mean(
            np.abs(
                (
                    y_true
                    - y_pred
                )
                / denominator
            )
        )
        * 100
    )

    print()
    print(
        f"{model_name}"
    )
    print(
        "-" * 60
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"MAPE : {mape:.4f}%"
    )

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


# ============================================================
# BASELINE: CURRENT RANDOM FOREST
# ============================================================

def train_current_rf(
    X_train,
    y_train,
    X_test,
    y_test,
):

    print()
    print(
        "1. Current Random Forest..."
    )

    model = RandomForestRegressor(

        n_estimators=300,

        max_depth=15,

        min_samples_leaf=2,

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    result = evaluate(
        "Current Random Forest",
        y_test,
        prediction
    )

    return (
        model,
        result
    )


# ============================================================
# BASELINE: CURRENT XGBOOST
# ============================================================

def train_current_xgb(
    X_train,
    y_train,
    X_test,
    y_test,
):

    print()
    print(
        "2. Current XGBoost..."
    )

    try:

        from xgboost import (
            XGBRegressor
        )

    except ImportError:

        print(
            "XGBoost is not installed."
        )

        print(
            "Run: pip install xgboost"
        )

        return (
            None,
            None
        )

    model = XGBRegressor(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    result = evaluate(
        "Current XGBoost",
        y_test,
        prediction
    )

    return (
        model,
        result
    )


# ============================================================
# TUNE RANDOM FOREST
# ============================================================

def tune_rf(
    X_train,
    y_train,
    cv_splits,
    X_test,
    y_test,
):

    print()
    print("=" * 80)
    print(
        "3. RANDOM FOREST HYPERPARAMETER TUNING"
    )
    print("=" * 80)

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    parameters = {

        "n_estimators": [
            200,
            300,
            500,
            700,
        ],

        "max_depth": [
            8,
            12,
            15,
            20,
            None,
        ],

        "min_samples_split": [
            2,
            5,
            10,
        ],

        "min_samples_leaf": [
            1,
            2,
            4,
        ],

        "max_features": [
            "sqrt",
            "log2",
            0.7,
            1.0,
        ],
    }

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=RF_N_ITER,

        scoring="neg_mean_absolute_error",

        cv=cv_splits,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    search.fit(
        X_train,
        y_train
    )

    print()
    print(
        "Best RF parameters:"
    )

    print(
        search.best_params_
    )

    print(
        f"Best CV MAE: "
        f"{-search.best_score_:.4f}"
    )

    prediction = search.best_estimator_.predict(
        X_test
    )

    result = evaluate(
        "Tuned Random Forest",
        y_test,
        prediction
    )

    return (
        search.best_estimator_,
        result
    )


# ============================================================
# TUNE XGBOOST
# ============================================================

def tune_xgb(
    X_train,
    y_train,
    cv_splits,
    X_test,
    y_test,
):

    print()
    print("=" * 80)
    print(
        "4. XGBOOST HYPERPARAMETER TUNING"
    )
    print("=" * 80)

    try:

        from xgboost import (
            XGBRegressor
        )

    except ImportError:

        print(
            "XGBoost is not installed."
        )

        print(
            "Run: pip install xgboost"
        )

        return (
            None,
            None
        )

    model = XGBRegressor(

        objective="reg:squarederror",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    parameters = {

        "n_estimators": [
            200,
            300,
            500,
            700,
            900,
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            8,
            10,
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.08,
            0.10,
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],

        "min_child_weight": [
            1,
            3,
            5,
            8,
        ],

        "reg_alpha": [
            0.0,
            0.01,
            0.1,
            1.0,
        ],

        "reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
        ],
    }

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=XGB_N_ITER,

        scoring="neg_mean_absolute_error",

        cv=cv_splits,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    search.fit(
        X_train,
        y_train
    )

    print()
    print(
        "Best XGBoost parameters:"
    )

    print(
        search.best_params_
    )

    print(
        f"Best CV MAE: "
        f"{-search.best_score_:.4f}"
    )

    prediction = search.best_estimator_.predict(
        X_test
    )

    result = evaluate(
        "Tuned XGBoost",
        y_test,
        prediction
    )

    return (
        search.best_estimator_,
        result
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
    model_name,
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return

    importance = (
        pd.DataFrame(
            {
                "feature":
                    FEATURE_COLUMNS,

                "importance":
                    model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False
        )
        .head(20)
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"TOP FEATURES — {model_name}"
    )

    print(
        "=" * 80
    )

    print(
        importance.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "PARKING OCCUPANCY — HYPERPARAMETER TUNING"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load unified data
    # --------------------------------------------------------

    print()
    print(
        "1. Loading unified parking history..."
    )

    raw = load_data()

    print(
        f"Rows       : {len(raw):,}"
    )

    print(
        f"Parking lots: "
        f"{raw['parking_id'].nunique()}"
    )

    print()
    print(
        "Rows by source:"
    )

    print(
        raw["source"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------

    print()
    print(
        "2. Creating forecasting features..."
    )

    data = create_features(
        raw
    )

    print(
        f"Modeling rows: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # 3. Chronological train/test
    # --------------------------------------------------------

    print()
    print(
        "3. Chronological train/test split..."
    )

    train, test = time_split(
        data
    )

    print(
        f"Train rows : "
        f"{len(train):,}"
    )

    print(
        f"Test rows  : "
        f"{len(test):,}"
    )

    print(
        f"Train period: "
        f"{train['timestamp'].min()} "
        f"→ "
        f"{train['timestamp'].max()}"
    )

    print(
        f"Test period : "
        f"{test['timestamp'].min()} "
        f"→ "
        f"{test['timestamp'].max()}"
    )

    # --------------------------------------------------------
    # 4. Matrices
    # --------------------------------------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target_occupancy"
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "target_occupancy"
    ]

    # --------------------------------------------------------
    # 5. Current Random Forest
    # --------------------------------------------------------

    current_rf, current_rf_result = (
        train_current_rf(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # 6. Current XGBoost
    # --------------------------------------------------------

    current_xgb, current_xgb_result = (
        train_current_xgb(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # 7. Time-aware CV
    # --------------------------------------------------------

    print()
    print(
        "5. Creating time-aware CV folds..."
    )

    cv_splits = (
        create_time_cv_splits(
            train,
            CV_SPLITS
        )
    )

    print(
        f"CV folds created: "
        f"{len(cv_splits)}"
    )

    for i, (
        train_idx,
        valid_idx
    ) in enumerate(
        cv_splits,
        start=1
    ):

        print(
            f"Fold {i}: "
            f"train={len(train_idx):,}, "
            f"validation={len(valid_idx):,}"
        )

    # --------------------------------------------------------
    # 8. Tune RF
    # --------------------------------------------------------

    tuned_rf, tuned_rf_result = (
        tune_rf(
            X_train,
            y_train,
            cv_splits,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # 9. Tune XGB
    # --------------------------------------------------------

    tuned_xgb, tuned_xgb_result = (
        tune_xgb(
            X_train,
            y_train,
            cv_splits,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # 10. Comparison
    # --------------------------------------------------------

    results = []

    if current_rf_result:

        results.append(
            current_rf_result
        )

    if current_xgb_result:

        results.append(
            current_xgb_result
        )

    if tuned_rf_result:

        results.append(
            tuned_rf_result
        )

    if tuned_xgb_result:

        results.append(
            tuned_xgb_result
        )

    comparison = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "MAE"
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 80)
    print(
        "FINAL TUNING COMPARISON"
    )
    print("=" * 80)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 11. Best model
    # --------------------------------------------------------

    best_name = (
        comparison.iloc[0][
            "model"
        ]
    )

    if best_name == (
        "Current Random Forest"
    ):

        best_model = current_rf

    elif best_name == (
        "Tuned Random Forest"
    ):

        best_model = tuned_rf

    elif best_name == (
        "Current XGBoost"
    ):

        best_model = current_xgb

    elif best_name == (
        "Tuned XGBoost"
    ):

        best_model = tuned_xgb

    else:

        raise RuntimeError(
            f"Unknown best model: "
            f"{best_name}"
        )

    # --------------------------------------------------------
    # 12. Feature importance
    # --------------------------------------------------------

    print_feature_importance(
        best_model,
        best_name
    )

    # --------------------------------------------------------
    # 13. Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    artifact = {

        "model":
            best_model,

        "features":
            FEATURE_COLUMNS,

        "target":
            "target_occupancy",

        "horizon_hours":
            1,

        "model_name":
            best_name,

        "data_sources":
            [
                "S",
                "OpenStreetMap",
                "OpenStreetMap-Geofabrik",
            ],

        "forecast_type":
            "next_hour_parking_occupancy",

        "occupancy_unit":
            "percentage",
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print()
    print("=" * 80)

    print(
        f"BEST FINAL MODEL: "
        f"{best_name}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Comparison saved to: "
        f"{COMPARISON_PATH}"
    )

    print("=" * 80)

    print()
    print(
        "PARKING MODEL TUNING COMPLETE"
    )

    print(
        "Target: occupancy percentage one hour ahead."
    )


if __name__ == "__main__":
    main()