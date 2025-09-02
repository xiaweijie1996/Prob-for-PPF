import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import numpy as np
import pickle 
import yaml
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from src.powersystems.node34 import Node34Example
from src.powersystems.randomsys import  magnitude_transform, angle_transform

def main():
    # Configureation
    # -----------------------
    # import config
    with open('src/config34node.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    num_nodes = config['SystemAndDistribution']['node']
    power_factor = config['SystemAndDistribution']['power_factor']
    std = config['SystemAndDistribution']['std']
    save_path = config['SystemAndDistribution']['save_path']
    batch_size = config['SystemAndDistribution']['batch_size']
    n_components = config['SystemAndDistribution']['n_components']
    mean_vector_start = config['SystemAndDistribution']['mean_vector_start']
    # -----------------------
    
    # Initialize the random system
    random_sys = Node34Example()
    mean_vector = [mean_vector_start + i*1 for i in range(num_nodes)]  # Example mean vector
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
    # print(f"GMM Power Means: {gmm_power.means_}, Covariances: {gmm_power.covariances_}")
    samples = gmm_power.sample(batch_size)[0]
    _active_power_sampled = samples[:, :num_nodes-1]
    _reactive_power_sampled = samples[:, num_nodes-1:]
    print("mean of sampled active power:", np.mean(_active_power_sampled))
    
    _solution_sampled = random_sys.run(active_power=_active_power_sampled,
                                        reactive_power=_reactive_power_sampled)
    _voltage_magnitudes = magnitude_transform(_solution_sampled['v'])
    _voltage_angles = angle_transform(_solution_sampled['v'])
    print(f"Sampled Voltage magnitudes shape: {_voltage_magnitudes.shape}, Voltage angles shape: {_voltage_angles.shape}")
    
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
    
    # Plot active and voltage distribution
    _index = 0
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(_reactive_power[:, _index], _active_power[:, _index], alpha=0.01)
    plt.title(f'Active vs Reactive Power at Node {_index+1}')
    plt.xlabel('Reactive Power (Q)')
    plt.ylabel('Active Power (P)')
    plt.grid(True)
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    plt.scatter(_voltage_magnitudes[:, _index], _voltage_angles[:, _index], alpha=0.01)
    plt.title(f'Voltage Magnitude vs Angle at Node {_index+1}')
    plt.xlabel('Voltage Magnitude (|V|)')
    plt.ylabel('Voltage Angle (θ)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(save_path, 'PQ_distribution.png'.format(_index+1)))

    
if __name__ == "__main__":
    main()