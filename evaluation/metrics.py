"""
Uncertainty and Point Forecast Evaluation Metrics
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss


def evaluate_forecast_horizon(
    y_true: np.ndarray, 
    preds_p10: np.ndarray, 
    preds_p50: np.ndarray, 
    preds_p90: np.ndarray, 
    horizon_label: str
) -> dict:
    """
    Computes Point Accuracy (MAE, RMSE) and Distributional Uncertainty Metrics
    (Empirical Coverage, Mean Interval Width, Pinball Loss).
    """
    mae = mean_absolute_error(y_true, preds_p50)
    rmse = np.sqrt(mean_squared_error(y_true, preds_p50))
    coverage = np.mean((y_true >= preds_p10) & (y_true <= preds_p90)) * 100.0
    mean_width = np.mean(preds_p90 - preds_p10)
    pinball_p10 = mean_pinball_loss(y_true, preds_p10, alpha=0.10)
    pinball_p50 = mean_pinball_loss(y_true, preds_p50, alpha=0.50)
    pinball_p90 = mean_pinball_loss(y_true, preds_p90, alpha=0.90)

    return {
        "horizon": horizon_label,
        "mae": mae,
        "rmse": rmse,
        "coverage": coverage,
        "interval_width": mean_width,
        "pinball_p10": pinball_p10,
        "pinball_p50": pinball_p50,
        "pinball_p90": pinball_p90,
    }


def format_metric_report(metrics: dict):
    print(f"\n==================================================")
    print(f" EVALUATION REPORT: {metrics['horizon'].upper()}")
    print(f"==================================================")
    print(f" Point Accuracy (P50 Median Forecast):")
    print(f"   • Mean Absolute Error (MAE)   : {metrics['mae']:.4f} °C")
    print(f"   • Root Mean Sq. Error (RMSE)  : {metrics['rmse']:.4f} °C")
    print(f" Uncertainty & Calibration Metrics:")
    print(f"   • 80% Nom. Interval Coverage  : {metrics['coverage']:.2f}% (Ideal: 80.0%)")
    print(f"   • Mean Prediction Band Width  : {metrics['interval_width']:.4f} °C")
    print(f"   • Pinball Loss (P10/P50/P90)  : {metrics['pinball_p10']:.4f} / {metrics['pinball_p50']:.4f} / {metrics['pinball_p90']:.4f}")
