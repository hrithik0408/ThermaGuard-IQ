"""
Configuration and Hyperparameters for ThermaGuard-IQ
"""

# Telemetry Generator Parameters
NUM_CONTAINERS = 25
NUM_DAYS = 21
SAMPLING_FREQ_MINUTES = 15
RANDOM_SEED = 42

# Thermal Limits (°C)
COLD_CHAIN_MIN_TEMP = 2.0
COLD_CHAIN_MAX_TEMP = 8.0
COLD_CHAIN_TARGET_TEMP = 4.0

# Quantile Forecast Settings
QUANTILES = (0.10, 0.50, 0.90)
FORECAST_HORIZONS = {
    "1h": {"target_col": "target_t_plus_1h", "label": "1-Hour Horizon (t+4 steps)"},
    "3h": {"target_col": "target_t_plus_3h", "label": "3-Hour Horizon (t+12 steps)"},
    "6h": {"target_col": "target_t_plus_6h", "label": "6-Hour Horizon (t+24 steps)"},
}

# Holdout Test Window (Days)
TEST_HOLDOUT_DAYS = 4

# LightGBM Quantile Regressor Hyperparameters
LGBM_PARAMS = {
    "n_estimators": 130,
    "learning_rate": 0.04,
    "max_depth": 5,
    "num_leaves": 24,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}
