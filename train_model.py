from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "traffic_data.csv"
MODEL_PATH = BASE_DIR / "traffic_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "day", "hour", "route", "weather", "congestion_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "congestion_ratio", "weather", "route"])

    # Features known before/during a trip. We intentionally do NOT use
    # travel_time_s or traffic_delay_s because they directly determine the target
    # and would cause data leakage in a real prediction app.
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["day"] = data["timestamp"].dt.day_name()
    data["month"] = data["timestamp"].dt.month
    data["day_of_month"] = data["timestamp"].dt.day
    data["is_weekend"] = (data["timestamp"].dt.dayofweek >= 5).astype(int)

    # Cyclical time features help the model understand that 23:00 and 00:00 are close.
    minutes_since_midnight = data["hour"] * 60 + data["minute"]
    data["time_sin"] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    data["time_cos"] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    return data.sort_values("timestamp").reset_index(drop=True)


def congestion_level(ratio: float) -> str:
    if ratio < 1.15:
        return "Low"
    if ratio < 1.40:
        return "Moderate"
    if ratio < 1.80:
        return "High"
    return "Severe"


def main():
    df = pd.read_csv(DATA_PATH)
    data = prepare_data(df)

    feature_cols = [
        "route", "weather", "day", "hour", "minute", "month",
        "day_of_month", "is_weekend", "time_sin", "time_cos"
    ]
    target_col = "congestion_ratio"

    # Chronological split: first 80% for training, last 20% for testing.
    split_idx = int(len(data) * 0.80)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    categorical = ["route", "weather", "day"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "features": feature_cols,
        "routes": sorted(data["route"].dropna().unique().tolist()),
        "weather_conditions": sorted(data["weather"].dropna().unique().tolist()),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "data_start": str(data["timestamp"].min()),
        "data_end": str(data["timestamp"].max()),
        "metrics": {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)},
        "target_min": float(data[target_col].min()),
        "target_max": float(data[target_col].max()),
        "level_rules": {
            "Low": "ratio < 1.15",
            "Moderate": "1.15 <= ratio < 1.40",
            "High": "1.40 <= ratio < 1.80",
            "Severe": "ratio >= 1.80"
        }
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model training complete.")
    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
