import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml

from src.models.cnice.cnicemodel import CNicemModel
from src.powersystems.node34 import Node34Example
from src.powersystems.randomsys import magnitude_transform, angle_transform

# -----------------------
# Configureation
# -----------------------
with open('src/config34node.yaml', 'r') as f:
    config = yaml.safe_load(f)

num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
dis_path = config['SystemAndDistribution']['dis_path']  # Path to the distribution system file
scaler_path = config['SystemAndDistribution']['scaler_path']  # Path to save/load the scalers
power_factor = config['SystemAndDistribution']['power_factor']  # Power factor for reactive power calculation

split_ratio = config['CNice']['split_ratio']
n_blocks = config['CNice']['n_blocks']
hiddemen_dim = config['CNice']['hiddemen_dim']
c_dim = (2 * (num_nodes - 1))  # Condition dimension (P and Q for all nodes except slack)
n_layers = config['CNice']['n_layers']
input_dim = config['CNice']['input_dim']  # Input dimension (P and Q for one node)
hiddemen_dim_condition = config['CNice']['hiddemen_dim_condition']
output_dim_condition = config['CNice']['output_dim_condition']
n_layers_condition = config['CNice']['n_layers_condition']

batch_size = config['CNice']['batch_size']
epochs = config['CNice']['epochs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = config['CNice']['save_path']

# -----------------------
# Initialize the random system　model and scalers
# -----------------------
random_sys = Node34Example()

# Initialize the NICE model
nice_model = CNicemModel(
    input_dim=input_dim,
    hidden_dim=hiddemen_dim,
    condition_dim=c_dim,
    n_layers=n_layers,
    split_ratio=split_ratio,
    n_blocks=n_blocks,
    hidden_dim_condition=hiddemen_dim_condition,
    output_dim_condition=output_dim_condition,
    n_layers_condition=n_layers_condition
).to(device)
print(f"Model Parameters: {sum(p.numel() for p in nice_model.parameters() if p.requires_grad)}")

# -----------------------    
# Load GMM and Scalers
# -----------------------
with open(dis_path, 'rb') as f:
    gmm = pickle.load(f)  
print("Loaded GMM from:", dis_path)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print("Loaded scalers from:", scaler_path)

# -----------------------    
# Define the data
# -----------------------
power_sample = gmm.sample(batch_size)[0]
active_power = power_sample[:, :num_nodes-1]
reactive_power = power_sample[:, num_nodes-1:]

_solution = random_sys.run(active_power=active_power, 
                            reactive_power=reactive_power)
voltage_magnitudes = magnitude_transform(_solution['v'])
voltage_angles = angle_transform(_solution['v'])

# Scale voltage and power using loaded scalers
scaled_voltage_magnitudes = (voltage_magnitudes - scaler['mean_voltage_magnitude']) / scaler['std_voltage_magnitude']
scaled_voltage_angles = (voltage_angles - scaler['mean_voltage_angle']) / scaler['std_voltage_angle']
scaled_active_power = (active_power - scaler['mean_active_power']) / scaler['std_active_power']
scaled_reactive_power = (reactive_power - scaler['mean_reactive_power']) / scaler['std_reactive_power']

# Concat the scaled data
concat_vm_va = np.hstack((scaled_voltage_magnitudes, scaled_voltage_angles))
concat_active_reactive = np.hstack((scaled_active_power, scaled_reactive_power))

# ----------------------- 
# Evaluate the model
# ----------------------- 
input_power = torch.tensor(concat_active_reactive, device=device, dtype=torch.float32)
target_voltage = torch.tensor(concat_vm_va, device=device, dtype=torch.float32)

p_index =  12 # torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index
input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
input_c = input_power.clone()
output_y = torch.cat((target_voltage[:, v_index].unsqueeze(1),
                      target_voltage[:, v_index+num_nodes-1].unsqueeze(1)), dim=1)
print(f"Input shape: {input_x.shape}, Condition shape: {input_c.shape}, Output shape: {output_y.shape}")
# print  mean and std of input_x and output_y
print(f"Input_x mean: {input_x.mean(axis=0).cpu().numpy()}, std: {input_x.std(axis=0).cpu().numpy()}")
print(f"Output_y mean: {output_y.mean(axis=0).cpu().numpy()}, std: {output_y.std(axis=0).cpu().numpy()}")

nice_model.eval()
with torch.no_grad():
    pre_v, _jaf = nice_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
    # fake_output_y = pre_v + torch.randn_like(output_y) * 0.1
    pre_p, _jai = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
    
pre_v = pre_v.cpu().numpy()
pre_p = pre_p.cpu().numpy()
print(f"pre_v: {pre_v.shape}, pre_p: {pre_p.shape}")

# Plot the pre_v and pre_p and real and target
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_v[:, 0], pre_v[:, 1], label='Predicted Voltage', alpha=0.1)
plt.scatter(output_y[:, 0].cpu().numpy(), output_y[:, 1].cpu().numpy(), label='Target Voltage', alpha=0.1)
plt.title('Predicted vs Target Voltage Magnitudes and Angles')
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(pre_p[:, 0], pre_p[:, 1], label='Predicted Power', alpha=0.1)
plt.scatter(input_x[:, 0].cpu().numpy(), input_x[:, 1].cpu().numpy(), label='Target Power', alpha=0.1)
plt.title('Predicted vs Target Active and Reactive Power')
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
# plt.tight_layout()
plt.savefig(f'figures/cnice_{num_nodes}node_accuracy1.png')


# Scale back to original
pre_v_magnitude = pre_v[:, 0] * scaler['std_voltage_magnitude'][p_index] + scaler['mean_voltage_magnitude'][p_index]
pre_v_angle = pre_v[:, 1] * scaler['std_voltage_angle'][p_index] + scaler['mean_voltage_angle'][p_index]
pre_p_active = pre_p[:, 0] * scaler['std_active_power'][p_index] + scaler['mean_active_power'][p_index]
pre_p_reactive = pre_p[:, 1] * scaler['std_reactive_power'][p_index] + scaler['mean_reactive_power'][p_index]

true_v_magnitude = voltage_magnitudes[:, p_index]
true_v_angle = voltage_angles[:, p_index]
true_p_active = active_power[:, p_index]
true_p_reactive = reactive_power[:, p_index]

# Plot the pre_v and pre_p and real and target
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_v_magnitude, pre_v_angle, label='Predicted Voltage', alpha=0.1)
plt.scatter(true_v_magnitude, true_v_angle, label='True Voltage', alpha=0.1)
plt.title('Predicted vs True Voltage Magnitudes and Angles')
plt.xlabel('Voltage Magnitude (p.u.)')
plt.ylabel('Voltage Angle (degrees)')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(pre_p_active, pre_p_reactive, label='Predicted Power', alpha=0.1)
plt.scatter(true_p_active, true_p_reactive, label='True Power', alpha=0.1)
plt.title('Predicted vs True Active and Reactive Power')
plt.xlabel('Active Power (P)')
plt.ylabel('Reactive Power (Q)')
plt.legend()
# plt.tight_layout()
plt.savefig(f'figures/cnice_{num_nodes}node_accuracy2.png')