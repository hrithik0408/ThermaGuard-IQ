"""
Multi-Horizon Quantile LightGBM Forecaster
"""

import pandas as pd
import lightgbm as lgb


class MultiHorizonQuantileForecaster:
    """
    LightGBM Pinball Regressor suite for multi-horizon uncertainty intervals.
    Trains independent models for each quantile (P10, P50, P90) across forecast horizons.
    """
    def __init__(self, quantiles=(0.10, 0.50, 0.90), lgbm_params: dict = None):
        self.quantiles = quantiles
        self.lgbm_params = lgbm_params or {
            "n_estimators": 130,
            "learning_rate": 0.04,
            "max_depth": 5,
            "num_leaves": 24,
            "min_child_samples": 25,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "verbosity": -1,
        }
        self.models = {}
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y_dict: dict):
        """
        X: DataFrame of engineered features
        y_dict: dict mapping horizon_key (e.g., '1_hour') to target array
        """
        self.feature_names = list(X.columns)
        for horizon_name, y in y_dict.items():
            self.models[horizon_name] = {}
            for q in self.quantiles:
                params = self.lgbm_params.copy()
                params["objective"] = "quantile"
                params["alpha"] = q
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y)
                self.models[horizon_name][q] = model

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Returns nested dictionary: predictions[horizon_name][quantile] = array of predictions
        """
        predictions = {}
        for horizon_name, q_models in self.models.items():
            predictions[horizon_name] = {
                q: q_models[q].predict(X) for q in self.quantiles
            }
        return predictions
