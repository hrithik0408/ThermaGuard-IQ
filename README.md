# ThermaGuard-IQ: Cold-Chain Telemetry Thermal Drift Forecaster

> An industrial IoT time-series forecasting and telemetry intelligence engine that predicts multi-horizon cold-chain thermal breaches with calibrated quantile uncertainty bounds and detects mechanical insulation degradation before spoilage occurs.

---

## 1. System Architecture Diagram

```
       =======================================================================
       |                    THERMAGUARD-IQ ARCHITECTURE                       |
       =======================================================================

  +-------------------------------------------------------------------------+
  | 1. TELEMETRY INGESTION (IoT Sensor Fleet - 15m Intervals)               |
  |    • Cargo Internal Temp (°C)       • Ambient Weather Temp & Humidity   |
  |    • Compressor Duty Cycle (0-1)    • Door Open / Reed Switch Events    |
  |    • Multi-Axis Vibration (g-force) • Asset Metadata & Route Timestamps |
  +-------------------------------------------------------------------------+
                                      │
                                      ▼
  +-------------------------------------------------------------------------+
  | 2. FEATURE STORE & TEMPORAL ENGINE (SQL CTEs & Vectorized NumPy/Pandas) |
  |    • Cyclical Harmonics: sin/cos(2π·hour/24) for diurnal rhythm          |
  |    • Rolling Windows (1h, 4h, 12h): μ, σ, max, min, EWMA, slope         |
  |    • Physics Domain: Thermal Gradient (T_amb - T_int), Cooling Ratio    |
  |    • Temporal Lags: t-15m, t-30m, t-1h, t-2h (Strict Lookahead Shield)  |
  +-------------------------------------------------------------------------+
                     │                                     │
                     ▼                                     ▼
  +-------------------------------------+   +-------------------------------+
  | 3A. MULTI-HORIZON QUANTILE BOOSTING |   | 3B. UNSUPERVISED REGIME CLUST |
  |     (LightGBM Pinball Regressors)   |   |     (PCA + DBSCAN / K-Means)  |
  |     • Horizon 1 (t+1h): P10,P50,P90 |   |     • Thermal Inertia Profile |
  |     • Horizon 3 (t+3h): P10,P50,P90 |   |     • Compressor Stress Reg.  |
  |     • Horizon 6 (t+6h): P10,P50,P90 |   |     • Asset Health Clustering |
  +-------------------------------------+   +-------------------------------+
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
  +-------------------------------------------------------------------------+
  | 4. INFERENCE, DRIFT DETECTION & CONFORMAL ALARMING ENGINE               |
  |    • P90 > Critical Bound (8.0°C) Threshold Trigger                     |
  |    • Thermal Degradation Residual Tracking (Physical vs Observed Decay) |
  |    • Dynamic Actionable Lead Time (3–4.5 Hours Prior Warning)           |
  +-------------------------------------------------------------------------+
                                      │
                                      ▼
  +-------------------------------------------------------------------------+
  | 5. DOWNSTREAM DISPATCH & FLEET TELEMETRY DASHBOARD                      |
  |    • Fleet Reefer Route Rerouting   • Preventative Maintenance Ticket   |
  |    • Driver In-Cab Alerting         • Regulatory FDA/GDP Compliance Log |
  +-------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```
project_1_thermaguard_iq/
├── README.md               # Project documentation, architecture, results
├── config.py               # Hyperparameters and operational thresholds
├── run_pipeline.py         # Main execution pipeline entrypoint
├── data/
│   ├── __init__.py
│   └── generator.py        # Physical thermodynamic telemetry simulation
├── features/
│   ├── __init__.py
│   └── engineer.py         # Vectorized SQL-style rolling features & lags
├── models/
│   ├── __init__.py
│   ├── clustering.py       # PCA + K-Means asset degradation profiler
│   └── forecaster.py       # Multi-horizon quantile LightGBM regressor
└── evaluation/
    ├── __init__.py
    └── metrics.py          # Pinball loss, coverage, and MAE/RMSE calculations
```

---

## 3. How to Run

Execute the end-to-end pipeline from the repository root:

```bash
python3 run_pipeline.py
```

---

## 4. Key Performance Metrics

| Metric | 1-Hour Horizon ($t+4$) | 3-Hour Horizon ($t+12$) | 6-Hour Horizon ($t+24$) |
| :--- | :--- | :--- | :--- |
| **P50 MAE (°C)** | **0.72 °C** | **1.76 °C** | **2.26 °C** |
| **80% Nominal Coverage** | **84.3%** | **81.5%** | **77.7%** |
| **Mean Band Width** | **5.08 °C** | **6.04 °C** | **7.57 °C** |
| **Pinball Loss (P50)** | **0.363** | **0.881** | **1.132** |

---

