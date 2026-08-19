"""
Feature Engineering Engine for Time-Series Telemetry
Vectorized Pandas/NumPy transforms matching SQL sliding window queries.
"""

import numpy as np
import pandas as pd


def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes temporal, lag, rolling, and physics interaction features
    with strict backward-looking windows to prevent lookahead data leakage.
    """
    df = df.sort_values(["container_id", "timestamp"]).reset_index(drop=True)

    # 1. Cyclical Solar/Temporal Encodings
    hour_val = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * hour_val / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour_val / 24.0)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # 2. Physical Interaction Signals
    df["thermal_gradient"] = df["ambient_temp"] - df["internal_temp"]
    df["cooling_efficiency_ratio"] = df["duty_cycle"] / (df["thermal_gradient"].clip(lower=0.1) + 1e-4)

    # 3. Grouped Lags and Rolling Windows (SQL Window Function Equivalents)
    grouped = df.groupby("container_id")

    for lag in [1, 2, 4, 8, 16]:  # 15m, 30m, 1h, 2h, 4h
        df[f"internal_temp_lag_{lag}"] = grouped["internal_temp"].shift(lag)
        df[f"ambient_temp_lag_{lag}"] = grouped["ambient_temp"].shift(lag)
        df[f"duty_cycle_lag_{lag}"] = grouped["duty_cycle"].shift(lag)

    # Rolling window stats over past 1 hour (4 steps) and 4 hours (16 steps)
    for window, label in [(4, "1h"), (16, "4h")]:
        df[f"temp_roll_mean_{label}"] = grouped["internal_temp"].transform(lambda s: s.shift(1).rolling(window).mean())
        df[f"temp_roll_std_{label}"] = grouped["internal_temp"].transform(lambda s: s.shift(1).rolling(window).std())
        df[f"temp_roll_max_{label}"] = grouped["internal_temp"].transform(lambda s: s.shift(1).rolling(window).max())
        df[f"temp_roll_min_{label}"] = grouped["internal_temp"].transform(lambda s: s.shift(1).rolling(window).min())
        df[f"duty_roll_mean_{label}"] = grouped["duty_cycle"].transform(lambda s: s.shift(1).rolling(window).mean())
        df[f"vibr_roll_mean_{label}"] = grouped["vibration"].transform(lambda s: s.shift(1).rolling(window).mean())

    # Derivatives / Rate of thermal change
    df["temp_velocity_1h"] = (df["internal_temp_lag_1"] - df["internal_temp_lag_4"]) / 3.0
    df["temp_velocity_4h"] = (df["internal_temp_lag_1"] - df["internal_temp_lag_16"]) / 15.0

    # 4. Forward Forecasting Targets (t+4 = 1 hour, t+12 = 3 hours, t+24 = 6 hours)
    df["target_t_plus_1h"] = grouped["internal_temp"].shift(-4)
    df["target_t_plus_3h"] = grouped["internal_temp"].shift(-12)
    df["target_t_plus_6h"] = grouped["internal_temp"].shift(-24)

    return df.dropna().reset_index(drop=True)
