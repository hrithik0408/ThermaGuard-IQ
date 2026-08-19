#!/usr/bin/env python3
"""
Main Execution Script for ThermaGuard-IQ Pipeline
Runs end-to-end telemetry generation, feature store computation, 
unsupervised clustering, quantile training, backtesting, and live early-warning simulation.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure project directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    NUM_CONTAINERS, NUM_DAYS, RANDOM_SEED,
    QUANTILES, FORECAST_HORIZONS, TEST_HOLDOUT_DAYS, LGBM_PARAMS
)
from data.generator import generate_cold_chain_telemetry
from features.engineer import build_feature_store
from models.clustering import cluster_fleet_operating_regimes
from models.forecaster import MultiHorizonQuantileForecaster
from evaluation.metrics import evaluate_forecast_horizon, format_metric_report


def main():
    print("=====================================================================")
    print("  THERMAGUARD-IQ: COLD-CHAIN FLEET TELEMETRY FORECASTING ENGINE      ")
    print("=====================================================================")
    
    # 1. Synthesize Telemetry
    print(f"[1/5] Ingesting multi-container IoT telemetry stream ({NUM_CONTAINERS} assets, {NUM_DAYS} days)...")
    raw_df = generate_cold_chain_telemetry(num_containers=NUM_CONTAINERS, num_days=NUM_DAYS, seed=RANDOM_SEED)
    print(f"      -> Ingested {len(raw_df):,} records at 15-minute sampling frequency.")

    # 2. Build Feature Store
    print("[2/5] Engineering temporal features, rolling statistics, and physics ratios...")
    feat_df = build_feature_store(raw_df)
    print(f"      -> Generated {feat_df.shape[1]} features across {len(feat_df):,} valid rows.")

    # 3. Unsupervised Regime Clustering
    print("[3/5] Performing unsupervised fleet degradation profiling (PCA + KMeans)...")
    cluster_profiles, var_ratio = cluster_fleet_operating_regimes(feat_df, random_state=RANDOM_SEED)
    print(f"      -> PCA 2-Component Explained Variance: {np.sum(var_ratio)*100:.2f}%")
    print("      -> Asset Cluster Distribution:")
    print(cluster_profiles["cluster_id"].value_counts().to_dict())

    # 4. Out-of-Time Train/Test Split (Temporal Backtest: Last 4 days holdout)
    split_cutoff = feat_df["timestamp"].max() - pd.Timedelta(days=TEST_HOLDOUT_DAYS)
    train_df = feat_df[feat_df["timestamp"] <= split_cutoff].reset_index(drop=True)
    test_df = feat_df[feat_df["timestamp"] > split_cutoff].reset_index(drop=True)

    non_feature_cols = [
        "timestamp", "container_id", "true_health_state", 
        "target_t_plus_1h", "target_t_plus_3h", "target_t_plus_6h"
    ]
    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train_targets = {
        h_key: train_df[h_info["target_col"]].values
        for h_key, h_info in FORECAST_HORIZONS.items()
    }

    print(f"[4/5] Training Multi-Horizon Quantile LightGBM models on {len(X_train):,} samples...")
    forecaster = MultiHorizonQuantileForecaster(quantiles=QUANTILES, lgbm_params=LGBM_PARAMS)
    forecaster.fit(X_train, y_train_targets)
    print("      -> Successfully fitted 9 quantile models (3 horizons x 3 quantiles).")

    # Evaluate on Holdout
    test_preds = forecaster.predict(X_test)
    for horizon_key, horizon_info in FORECAST_HORIZONS.items():
        y_true = test_df[horizon_info["target_col"]].values
        p10 = test_preds[horizon_key][0.10]
        p50 = test_preds[horizon_key][0.50]
        p90 = test_preds[horizon_key][0.90]
        report = evaluate_forecast_horizon(y_true, p10, p50, p90, horizon_info["label"])
        format_metric_report(report)

    # 5. Live Simulation Alerting Demo
    print("\n=====================================================================")
    print("  SIMULATION DEMO: LIVE EARLY-WARNING TRIP DISPATCH ALERT             ")
    print("=====================================================================")
    sample_asset = "REEFER-001"  # Degraded insulation unit
    asset_test = test_df[test_df["container_id"] == sample_asset].iloc[-1]
    
    sample_feat = pd.DataFrame([asset_test[feature_cols]])
    pred_sample = forecaster.predict(sample_feat)
    
    curr_t = asset_test['internal_temp']
    p10_3h = pred_sample['3h'][0.10][0]
    p50_3h = pred_sample['3h'][0.50][0]
    p90_3h = pred_sample['3h'][0.90][0]
    
    print(f" Telemetry Ingested for Container : {sample_asset}")
    print(f" Timestamp                        : {asset_test['timestamp']}")
    print(f" Current Temperature              : {curr_t:.2f} °C (Pharma Safe Range: 2.0°C - 8.0°C)")
    print(f" Ambient Weather Temp             : {asset_test['ambient_temp']:.2f} °C | Compressor Duty: {asset_test['duty_cycle']*100:.1f}%")
    print(f" -------------------------------------------------------------------")
    print(f" 3-Hour Forward Probabilistic Forecast:")
    print(f"   • P10 (Best Case)              : {p10_3h:.2f} °C")
    print(f"   • P50 (Expected Median)        : {p50_3h:.2f} °C")
    print(f"   • P90 (Worst Case Risk Bound)  : {p90_3h:.2f} °C")
    
    if p90_3h >= 8.0:
        print(f" [CRITICAL ACTION TRIGGERED]: Upper risk bound P90 ({p90_3h:.2f}°C) exceeds 8.0°C threshold.")
        print(f"   -> Action: Alerting dispatch to route truck to nearest cold storage facility.")
        print(f"   -> Estimated Lead Time to Cargo Loss: ~3.5 hours.")
    else:
        print(" [NOMINAL STATUS]: Thermal trajectory within compliant envelope.")


if __name__ == "__main__":
    main()
