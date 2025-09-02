import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb
import pickle 
import yaml
from sklearn.mixture import GaussianMixture

from src.powersystems.node34 import Node34Example
from src.powersystems.randomsys import  magnitude_transform, angle_transform

def main():
    # Configureation
    # -----------------------
    num_nodes = 34
    power_factor = 0.2
    std = 10
    save_path = 'src/powersystems/psdistribution/34node'
    batch_size = 20000
    n_components = 5
    # -----------------------
    
    # Initialize the random system
    random_sys = Node34Example()
    mean_vector = [50 + i*1 for i in range(num_nodes)]  # Example mean vector
    mean_vector = np.array(mean_vector)
    print(f"Mean vector: {mean_vector[1:].shape}, {mean_vector[1:]}")
    
    # Power
    _active_power1 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1))
    _reactive_power1 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1)) * power_factor
    _active_power2 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1)) *0.6
    _reactive_power2 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1)) * power_factor*0.6
    _active_power3 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1)) *1.4
    _reactive_power3 = np.random.normal(loc=mean_vector[1:], scale=std, size=(batch_size, num_nodes-1)) * power_factor*1.4
    _active_power = np.vstack([_active_power1, _active_power2, _active_power3])
    _reactive_power = np.vstack([_reactive_power1, _reactive_power2, _reactive_power3])

    # Run power flow
    _solution = random_sys.run(active_power=_active_power, 
                                reactive_power=_reactive_power) 
    
    _voltage_magnitudes = magnitude_transform(_solution['v'])
    _voltage_angles = angle_transform(_solution['v'])
    print(f"Voltage magnitudes shape: {_voltage_magnitudes.shape}, Voltage angles shape: {_voltage_angles.shape}")
    # Fit GMM
    power_data = np.hstack([_active_power, _reactive_power])
    voltage_data = np.hstack([_voltage_magnitudes, _voltage_angles])
    print(f"Power data shape: {power_data.shape}, Voltage data shape: {voltage_data.shape}")
    print(f"voltage magnitude example: {_voltage_magnitudes.shape}")
    
    gmm_power = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm_power.fit(power_data)
    # print covariances and means
    print(f"GMM Power Means: {gmm_power.means_}, Covariances: {gmm_power.covariances_}")
    
    # Save GMM model
    gmm_path = os.path.join(save_path, 'gmm_power.pkl')
    with open(gmm_path, 'wb') as f:
        pickle.dump(gmm_power, f)
    print(f"GMM model saved to {gmm_path}")
    
    # Scalers
    mean_active_power = np.mean(_active_power, axis=0)
    std_active_power = np.std(_active_power, axis=0)
    mean_reactive_power = np.mean(_reactive_power, axis=0)
    std_reactive_power = np.std(_reactive_power, axis=0)
    mean_voltage_magnitude = np.mean(_voltage_magnitudes, axis=0)
    std_voltage_magnitude = np.std(_voltage_magnitudes, axis=0)
    mean_voltage_angle = np.mean(_voltage_angles, axis=0)
    std_voltage_angle = np.std(_voltage_angles, axis=0)
    print(f"Mean active power shape: {mean_active_power.shape}, std active power shape: {std_active_power.shape}")
    print(f"Mean voltage magnitude shape: {mean_voltage_magnitude.shape}, std voltage magnitude shape: {std_voltage_magnitude.shape}")
    scalers = {
        'mean_active_power': mean_active_power,
        'std_active_power': std_active_power,
        'mean_reactive_power': mean_reactive_power,
        'std_reactive_power': std_reactive_power,
        'mean_voltage_magnitude': mean_voltage_magnitude,
        'std_voltage_magnitude': std_voltage_magnitude,
        'mean_voltage_angle': mean_voltage_angle,
        'std_voltage_angle': std_voltage_angle
    }
    
    # Save scalers
    scalers_path = os.path.join(save_path, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {scalers_path}")
    
if __name__ == "__main__":
    main()