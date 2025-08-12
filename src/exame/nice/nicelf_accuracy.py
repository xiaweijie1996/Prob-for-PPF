import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.nice.nicemodel import NicemModel
from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers

num_nodes = 34
num_children = 3
power_factor = 0.2

split_ratio = 0.6
n_blocks = 3
hiddemen_dim = 256
n_layers = 2
full_dim = (num_nodes -1) * 2  # Assuming each node has a real and imaginary part

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the random system
random_sys = randomsystem(num_nodes=num_nodes, num_children=num_children)

# Initialize the NICE model
nice_model = NicemModel(
    full_dim=full_dim,
    hiddemen_dim=hiddemen_dim,
    n_layers=n_layers,
    split_ratio=split_ratio,
    n_blocks=n_blocks
).to(device)

print(f"Model Parameters: {sum(p.numel() for p in nice_model.parameters() if p.requires_grad)}")

# Define the scalers
_active_power = np.random.normal(50, scale=5, size=(100, num_nodes-1))  # Power in kW
_reactive_power = _active_power * power_factor
_solution = random_sys.run(active_power=_active_power, 
                            reactive_power=_reactive_power, 
                            plot_graph=False)
_voltage_magnitudes = magnitude_transform(_solution['v'])
_voltage_angles = angle_transform(_solution['v'])

# Fit the scalers
scaler_p, scaler_q, scaler_vm, scaler_va = fit_powerflow_scalers(
    active_power=_active_power,
    reactive_power=_reactive_power,
    voltage_magnitudes=_voltage_magnitudes,
    voltage_angles=_voltage_angles
)

scaled_vm = scaler_vm.transform(_voltage_magnitudes)
scaled_va = scaler_va.transform(_voltage_angles)
concat_vm_va = np.concatenate((scaled_vm, scaled_va), axis=1)

scaled_active_power = scaler_p.transform(_active_power)
scaled_reactive_power = scaler_q.transform(_reactive_power)
concat_active_reactive = np.concatenate((scaled_active_power, scaled_reactive_power), axis=1)

concat_vm_va = torch.tensor(concat_vm_va, device=device, dtype=torch.float32)
concat_active_reactive = torch.tensor(concat_active_reactive, device=device, dtype=torch.float32)

# # Load the nice model
nice_model.load_state_dict(torch.load(f"src/training/nice/savedmodel/nicemodel_34.pth"))
nice_model.eval()

# Perform inference
with torch.no_grad():
    pre_v, _ = nice_model.forward(concat_active_reactive)
    pre_p, _ = nice_model.inverse(concat_vm_va)
    
    
    # pre_v[:, :num_nodes-1] = scaler_vm.inverse_transform(pre_v[:, :num_nodes-1])
    # pre_v[:, num_nodes-1:] = scaler_va.inverse_transform(pre_v[:, num_nodes-1:])
    # pre_p[:, :num_nodes-1] = scaler_p.inverse_transform(pre_p[:, :num_nodes-1])
    # pre_p[:, num_nodes-1:] = scaler_q.inverse_transform(pre_p[:, num_nodes-1:])
    
# Plot the pre_v and pre_p and real
pre_v = pre_v.cpu().numpy()
pre_p = pre_p.cpu().numpy()
concat_vm_va = concat_vm_va.cpu().numpy()
concat_active_reactive = concat_active_reactive.cpu().numpy()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_v[:, 0], pre_v[:, num_nodes], label='Predicted Voltage')
plt.title('Predicted Voltage Magnitudes and Angles')
plt.xlabel('Node Index')
plt.ylabel('Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(concat_vm_va[:, 0], concat_vm_va[:, num_nodes], label='Input Voltage')
plt.title('Input Voltage Magnitudes and Angles')
plt.xlabel('Node Index')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig('figures/predicted_voltage_magnitudes_angles.png')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_p[:, 0], pre_p[:, num_nodes-1], label='Predicted Active Power')
plt.title('Predicted Active Power')
plt.xlabel('Node Index')
plt.ylabel('Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(concat_active_reactive[:, 0], concat_active_reactive[:, num_nodes-1], label='Input Active Power')
plt.title('Input Active Power')
plt.xlabel('Node Index')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig('figures/predicted_active_power.png')