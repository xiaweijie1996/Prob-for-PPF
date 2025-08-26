import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

from src.models.cnice.cnicemodel import CNicemModel
from src.powersystems.node34 import Node34Example
from src.powersystems.randomsys import magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers
from src.utility.inversepdf import inverse_pdf_gaussian



# -----------------------
# Configureation
# -----------------------
num_nodes = 34
# num_children = 3
power_factor = 0.2

split_ratio = 0.5
n_blocks = 3
hiddemen_dim = 128
c_dim = (num_nodes - 1) * 2
n_layers = 4
input_dim = 2  # Assuming each node has a real and imaginary part
hiddemen_dim_condition = 128
output_dim_condition = 1
n_layers_condition = 2

batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'src/training/cnice/savedmodel'


# -----------------------
# Initialize the random system　model and scalers
# -----------------------
random_sys = Node34Example()
mean_vector = [50 + i*2 for i in range(num_nodes)]  # Example mean vector
mean_vector = np.array(mean_vector).astype(float)
print(f"Mean vector: {mean_vector[1:].shape}, {mean_vector[1:]}")

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

# Load the model and scalers
model_path = os.path.join(save_path, f"cnicemodel_{num_nodes}.pth")
nice_model.load_state_dict(torch.load(model_path, map_location=device))

scalers = {}
path_scalers = os.path.join(save_path, f'scalers_{num_nodes}.pkl')
with open(path_scalers, 'rb') as f:
    scalers = pickle.load(f)
# Load the scalers
scaler_vm = scalers['scaler_vm']
scaler_va = scalers['scaler_va']
scaler_p = scalers['scaler_p']
scaler_q = scalers['scaler_q']

# -----------------------    
# Define the data
# -----------------------
# Fix other nodes, vary one node
p_index = torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index
_active_power_index = np.random.normal(mean_vector[1+p_index], scale=5, size=(batch_size, ))
_active_power = np.tile(mean_vector[1:], (batch_size, 1))
_active_power[:, p_index] = _active_power_index

# _active_power = np.random.normal(mean_vector[1:], scale=5, size=(5000, num_nodes-1))  # Power in kW
_reactive_power = _active_power * power_factor
_solution = random_sys.run(active_power=_active_power, 
                            reactive_power=_reactive_power)
_voltage_magnitudes = magnitude_transform(_solution['v'])
_voltage_angles = angle_transform(_solution['v'])

# Scale the data
scaled_vm = scaler_vm.transform(_voltage_magnitudes)
scaled_va = scaler_va.transform(_voltage_angles)
scaled_active_power = scaler_p.transform(_active_power)
scaled_reactive_power = scaler_q.transform(_reactive_power)

# Concat the data
concat_vm_va = np.concatenate((scaled_vm, scaled_va), axis=1)
concat_active_reactive = np.concatenate((scaled_active_power, scaled_reactive_power), axis=1)

# ----------------------- 
# Evaluate the model
# ----------------------- 
input_power = torch.tensor(concat_active_reactive, device=device, dtype=torch.float32)
target_voltage = torch.tensor(concat_vm_va, device=device, dtype=torch.float32)

input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
input_c = input_power.clone()
output_y = torch.cat((target_voltage[:, v_index].unsqueeze(1),
                      target_voltage[:, v_index+num_nodes-1].unsqueeze(1)), dim=1)


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
plt.tight_layout()
plt.savefig(f'figures/cnice_{num_nodes}node_accuracy1.png', dpi=300)

# Reconstruct the full voltage and power arrays
pre_v_total = target_voltage.clone().cpu().numpy()
pre_p_total = input_power.clone().cpu().numpy()
pre_v_total[:, v_index] = pre_v[:, 0]
pre_v_total[:, v_index+num_nodes-1] = pre_v[:, 1]
pre_p_total[:, p_index] = pre_p[:, 0]
pre_p_total[:, p_index+num_nodes-1] = pre_p[:, 1]

# Inverse transform the scaled data
pre_v_total[:, :num_nodes-1] = scaler_vm.inverse_transform(pre_v_total[:, :num_nodes-1])
pre_v_total[:, num_nodes-1:] = scaler_va.inverse_transform(pre_v_total[:, num_nodes-1:])
pre_p_total[:, :num_nodes-1] = scaler_p.inverse_transform(pre_p_total[:, :num_nodes-1])
pre_p_total[:, num_nodes-1:] = scaler_q.inverse_transform(pre_p_total[:, num_nodes-1:])

# pre_v and pre_p scaled back
pre_vm_scaled = pre_v_total[:, :num_nodes-1]
pre_va_scaled = pre_v_total[:, num_nodes-1:]
pre_p_scaled = pre_p_total[:, :num_nodes-1]
pre_q_scaled = pre_p_total[:, num_nodes-1:]

# Plot the pre_v_scaled and pre_p_scaled and real and target
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_vm_scaled[:, p_index], pre_va_scaled[:, p_index], label='Predicted Voltage', alpha=0.1)
plt.scatter(_voltage_magnitudes[:, p_index], _voltage_angles[:, p_index], label='Target Voltage', alpha=0.1)        
plt.title('Predicted vs Target Voltage Magnitudes and Angles (Scaled Back)')
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(pre_p_scaled[:, p_index], pre_q_scaled[:, p_index], label='Predicted Power', alpha=0.1)
plt.scatter(_active_power[:, p_index], _reactive_power[:, p_index], label='Target Power', alpha=0.1)        
plt.title('Predicted vs Target Active and Reactive Power (Scaled Back)') 
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/cnice_{num_nodes}node_accuracy2.png', dpi=300)
