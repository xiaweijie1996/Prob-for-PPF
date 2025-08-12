from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_powerflow_scalers(active_power, reactive_power, voltage_magnitudes, voltage_angles):
    """
    Fit four StandardScalers for active power, reactive power, voltage magnitudes, and voltage angles.

    Parameters
    ----------
    active_power : array-like, shape (n_samples, n_features)
        Active power values (P).
    reactive_power : array-like, shape (n_samples, n_features)
        Reactive power values (Q).
    voltage_magnitudes : array-like, shape (n_samples, n_features)
        Voltage magnitude values.
    voltage_angles : array-like, shape (n_samples, n_features)
        Voltage angle values.

    Returns
    -------
    tuple of (StandardScaler, StandardScaler, StandardScaler, StandardScaler)
        Scalers for (P, Q, V_magnitude, V_angle).
    """
    # Ensure NumPy arrays
    active_power = np.array(active_power)
    reactive_power = np.array(reactive_power)
    voltage_magnitudes = np.array(voltage_magnitudes)
    voltage_angles = np.array(voltage_angles)

    # Initialize scalers
    scaler_p = StandardScaler().fit(active_power)
    scaler_q = StandardScaler().fit(reactive_power)
    scaler_vm = StandardScaler().fit(voltage_magnitudes)
    scaler_va = StandardScaler().fit(voltage_angles)

    return scaler_p, scaler_q, scaler_vm, scaler_va
