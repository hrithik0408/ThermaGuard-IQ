"""
Unsupervised Operating Regime & Asset Degradation Profiler (PCA + KMeans)
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cluster_fleet_operating_regimes(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Extracts asset-level behavioral thermal signatures and clusters
    containers into health regimes without using ground-truth labels.
    """
    asset_profiles = df.groupby("container_id").agg({
        "internal_temp": ["mean", "max"],
        "duty_cycle": "mean",
        "cooling_efficiency_ratio": "mean",
        "vibration": "mean",
        "temp_roll_std_4h": "mean",
        "thermal_gradient": "mean"
    })
    asset_profiles.columns = [f"{col[0]}_{col[1]}" for col in asset_profiles.columns]
    
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(asset_profiles)
    
    pca = PCA(n_components=2, random_state=random_state)
    pca_features = pca.fit_transform(scaled_profiles)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled_profiles)
    
    asset_profiles["cluster_id"] = clusters
    asset_profiles["pca_1"] = pca_features[:, 0]
    asset_profiles["pca_2"] = pca_features[:, 1]
    
    return asset_profiles, pca.explained_variance_ratio_
