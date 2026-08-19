"""
Physical IoT Cold-Chain Telemetry Simulation Engine
"""

import numpy as np
import pandas as pd


def generate_cold_chain_telemetry(num_containers: int = 25, num_days: int = 21, seed: int = 42) -> pd.DataFrame:
    """
    Simulates high-frequency IoT sensor telemetry with thermodynamic heat transfer:
    - Diurnal solar ambient cycles
    - Thermostat bang-bang / PWM compressor cooling
    - Insulation seal degradation in select assets
    - Stochastic door opening heat ingress
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start="2026-06-01 00:00", periods=num_days * 24 * 4, freq="15min")
    fleet_records = []

    for cid in range(1, num_containers + 1):
        # Asset Health Profile: container 1-3 have failing seals, 4-5 have weak compressors
        if cid in [1, 2, 3]:
            insulation_factor = np.random.uniform(0.35, 0.50)  # Heavy thermal leakage
            compressor_eff = np.random.uniform(0.85, 0.95)
            health_label = "Degraded_Insulation"
        elif cid in [4, 5]:
            insulation_factor = np.random.uniform(0.80, 0.95)
            compressor_eff = np.random.uniform(0.40, 0.55)     # Weak compressor pump
            health_label = "Failing_Compressor"
        else:
            insulation_factor = np.random.uniform(0.85, 1.05)  # Nominal brand new
            compressor_eff = np.random.uniform(0.90, 1.05)
            health_label = "Healthy_Nominal"

        # Ambient diurnal temperature curve (20°C night, 34°C peak afternoon + stochastic weather waves)
        hour_fraction = timestamps.hour + timestamps.minute / 60.0
        weather_drift = np.sin(np.linspace(0, np.pi * 6, len(timestamps))) * 3.5
        ambient_temp = 22.0 + 10.0 * np.sin(2 * np.pi * (hour_fraction - 9) / 24.0) + weather_drift + np.random.normal(0, 0.8, len(timestamps))
        ambient_humidity = np.clip(55.0 + 25.0 * np.cos(2 * np.pi * hour_fraction / 24.0) + np.random.normal(0, 2.0, len(timestamps)), 15.0, 98.0)

        # Simulation state variables
        internal_temp = np.zeros(len(timestamps))
        duty_cycle = np.zeros(len(timestamps))
        door_open = np.zeros(len(timestamps))
        vibration = np.zeros(len(timestamps))
        
        curr_t = 4.0  # Cold chain target: 4.0°C (Pharma safe zone: 2.0°C to 8.0°C)

        for step in range(len(timestamps)):
            ts = timestamps[step]
            # Door events during daylight delivery hours (08:00 to 17:00)
            is_delivery_hrs = 8 <= ts.hour <= 17
            p_door = 0.05 if is_delivery_hrs else 0.002
            is_open = 1.0 if np.random.rand() < p_door else 0.0
            door_open[step] = is_open

            # Thermodynamic heat ingress: Q_in = k * (T_ambient - T_internal)
            heat_leak = (ambient_temp[step] - curr_t) * (0.075 / insulation_factor) * 0.25
            if is_open:
                heat_leak += (ambient_temp[step] - curr_t) * 0.45  # Direct hot air exchange

            # Thermostat controller logic with hysteresis
            if curr_t > 5.2:
                dc = np.clip(0.90 * compressor_eff + np.random.normal(0, 0.03), 0.1, 1.0)
            elif curr_t > 4.0:
                dc = np.clip(0.50 * compressor_eff + np.random.normal(0, 0.03), 0.0, 0.8)
            else:
                dc = np.clip(0.10 * compressor_eff + np.random.normal(0, 0.02), 0.0, 0.3)
            duty_cycle[step] = dc

            # Active cooling: Q_out = dc * Cooling_Capacity
            cooling_power = dc * 2.35 * compressor_eff * 0.25
            
            # Update internal temperature
            curr_t = curr_t + heat_leak - cooling_power + np.random.normal(0, 0.04)
            internal_temp[step] = curr_t

            # Vibration profile (compressor load + road roughness)
            base_vibe = 0.4 + 2.2 * dc + (1.6 if compressor_eff < 0.6 else 0.0)
            vibration[step] = base_vibe + np.random.normal(0, 0.12)

        container_df = pd.DataFrame({
            "timestamp": timestamps,
            "container_id": f"REEFER-{cid:03d}",
            "ambient_temp": np.round(ambient_temp, 2),
            "ambient_humidity": np.round(ambient_humidity, 2),
            "internal_temp": np.round(internal_temp, 3),
            "duty_cycle": np.round(duty_cycle, 3),
            "door_open": door_open.astype(int),
            "vibration": np.round(vibration, 3),
            "true_health_state": health_label
        })
        fleet_records.append(container_df)

    return pd.concat(fleet_records, ignore_index=True)
