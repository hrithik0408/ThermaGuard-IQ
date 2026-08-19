#!/usr/bin/env python3
"""
Visualization Script for ThermaGuard-IQ
Generates high-resolution diagnostic plots for GitHub README & LinkedIn:
1. Multi-Horizon Quantile Fan Chart (P10 - P50 - P90) with Critical 8°C Line
2. Unsupervised Operating Regime PCA Cluster Map
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def generate_visualizations(output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating diagnostic plots in ./{output_dir}/ ...")

    # 1. Pipeline Execution
    raw_df = generate_cold_chain_telemetry(num_containers=NUM_CONTAINERS, num_days=NUM_DAYS, seed=RANDOM_SEED)
    feat_df = build_feature_store(raw_df)
    
    # 2. PCA Clustering Plot
    cluster_profiles, var_ratio = cluster_fleet_operating_regimes(feat_df, random_state=RANDOM_SEED)
    
    plt.figure(figsize=(9, 6), dpi=300)
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    labels = ['Healthy Nominal Fleet', 'Degraded Insulation (Seals Leaking)', 'Compressor Mechanical Stress']
    
    for c_id in sorted(cluster_profiles['cluster_id'].unique()):
        subset = cluster_profiles[cluster_profiles['cluster_id'] == c_id]
        label_text = labels[c_id] if c_id < len(labels) else f"Cluster {c_id}"
        plt.scatter(
            subset['pca_1'], subset['pca_2'], 
            s=120, c=colors[c_id % len(colors)], 
            label=f"{label_text} (n={len(subset)})", 
            edgecolor='black', alpha=0.85
        )

    plt.title("ThermaGuard-IQ: Unsupervised Asset Health Profiling (PCA + K-Means)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(f"PCA Component 1 ({var_ratio[0]*100:.1f}% Variance Explained)", fontsize=11)
    plt.ylabel(f"PCA Component 2 ({var_ratio[1]*100:.1f}% Variance Explained)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='best')
    plt.tight_layout()
    cluster_plot_path = os.path.join(output_dir, "asset_health_clusters.png")
    plt.savefig(cluster_plot_path)
    plt.close()
    print(f"  [✓] Saved: {cluster_plot_path}")

    # 3. Model Training for Forecast Fan Chart
    split_cutoff = feat_df["timestamp"].max() - pd.Timedelta(days=TEST_HOLDOUT_DAYS)
    train_df = feat_df[feat_df["timestamp"] <= split_cutoff].reset_index(drop=True)
    test_df = feat_df[feat_df["timestamp"] > split_cutoff].reset_index(drop=True)

    non_feature_cols = [
        "timestamp", "container_id", "true_health_state", 
        "target_t_plus_1h", "target_t_plus_3h", "target_t_plus_6h"
    ]
    feature_cols = [c for c in feat_df.columns if c not in non_feature_cols]

    X_train = train_df[feature_cols]
    y_train_targets = {
        h_key: train_df[h_info["target_col"]].values
        for h_key, h_info in FORECAST_HORIZONS.items()
    }

    forecaster = MultiHorizonQuantileForecaster(quantiles=QUANTILES, lgbm_params=LGBM_PARAMS)
    forecaster.fit(X_train, y_train_targets)

    # Pick a degraded container for demonstration
    sample_container = "REEFER-001"
    container_test = test_df[test_df["container_id"] == sample_container].sort_values("timestamp").reset_index(drop=True)
    
    # Predict 3-hour horizon on this test asset
    sample_X = container_test[feature_cols]
    preds = forecaster.predict(sample_X)
    
    p10 = preds['3h'][0.10]
    p50 = preds['3h'][0.50]
    p90 = preds['3h'][0.90]
    y_actual = container_test['target_t_plus_3h'].values
    timestamps = container_test['timestamp']

    # Plot 3-Day Time Window
    plot_len = min(len(timestamps), 96 * 3)  # 3 days
    t_sub = timestamps.iloc[:plot_len]
    act_sub = y_actual[:plot_len]
    p10_sub = p10[:plot_len]
    p50_sub = p50[:plot_len]
    p90_sub = p90[:plot_len]

    plt.figure(figsize=(13, 6), dpi=300)
    plt.plot(t_sub, act_sub, color='#2c3e50', linewidth=2.2, label='Actual Internal Cargo Temp (t+3h)', zorder=4)
    plt.plot(t_sub, p50_sub, color='#2980b9', linestyle='--', linewidth=1.8, label='P50 Median Forecast', zorder=3)
    plt.fill_between(t_sub, p10_sub, p90_sub, color='#3498db', alpha=0.25, label='80% Prediction Interval [P10, P90]', zorder=2)
    
    # Critical 8.0°C Line
    plt.axhline(8.0, color='#e74c3c', linestyle='-', linewidth=2.0, label='Critical Compliance Threshold (8.0°C)', zorder=5)
    plt.axhline(2.0, color='#16a085', linestyle=':', linewidth=1.5, label='Lower Safe Bound (2.0°C)', zorder=5)
    
    plt.title(f"ThermaGuard-IQ: 3-Hour Horizon Quantile Forecast & Risk Bound ({sample_container})", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Timestamp (15-Minute Telemetry Stream)", fontsize=11)
    plt.ylabel("Internal Temperature (°C)", fontsize=11)
    plt.ylim(0.0, max(act_sub.max(), p90_sub.max()) + 3.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    plt.tight_layout()
    fan_plot_path = os.path.join(output_dir, "quantile_forecast_fan_chart.png")
    plt.savefig(fan_plot_path)
    plt.close()
    print(f"  [✓] Saved: {fan_plot_path}")
    print("\nVisualizations successfully generated! You can include these images in your GitHub README.")


if __name__ == "__main__":
    generate_visualizations()
