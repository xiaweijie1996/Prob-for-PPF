import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.models.cfcpflow.cfcpflow import SimplifiedFcpflow
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.ieee39 import Case39PF

# Set all torch floats to double precision
torch.set_default_dtype(torch.float64)

# Configureation
# -----------------------
with open('src/config39bus.yaml', 'r') as f:
    config = yaml.safe_load(f)

num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
load_bus_num = config['SystemAndDistribution']['load_bus_num']  # Number of load buses

split_ratio = config['CFCP']['split_ratio']
n_blocks = config['CFCP']['n_blocks']
hiddemen_dim = config['CFCP']['hiddemen_dim']
c_dim = (2 * (load_bus_num))  # Condition dimension (P and Q for all nodes except slack)
n_layers = config['CFCP']['n_layers']
input_dim = config['CFCP']['input_dim']  # Input dimension (P and Q for one node)
hiddemen_dim_condition = config['CFCP']['hiddemen_dim_condition']
output_dim_condition = config['CFCP']['output_dim_condition']
n_layers_condition = config['CFCP']['n_layers_condition']

load_variance_rate = config['SystemAndDistribution']['load_variance_rate']
forward_loss_ratio = config['CFCP'].get('forward_loss_ratio', 1.0)  # Default to 1.0 if not specified
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# Initialize the random system
random_sys = Case39PF()
random_sys.run_pf()
load_index = random_sys.net.load['bus']
load_bus = random_sys.net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']].loc[load_index].reset_index(drop=True)
epsilon = 1e-6
base_ap = load_bus['p_mw'].values + epsilon
base_rp = load_bus['q_mvar'].values
base_vm = load_bus['vm_pu'].values
base_va = load_bus['va_degree'].values
load_rato = base_rp/base_ap
# print('load ratio:', load_rato)
scaler = {
    'mean_active_power': base_ap,
    'mean_reactive_power': base_rp,
    'mean_voltage_magnitude': base_vm,
    'mean_voltage_angle': base_va.mean(),
    
}
print(f"Scaler: {scaler}")

# -----------------------
# Initialize the NICE model
fcp_model = SimplifiedFcpflow(
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
fcp_model.double()
print(f"Model Parameters: {sum(p.numel() for p in fcp_model.parameters() if p.requires_grad)}")

# Load pre-trained model
model_path = 'src/trainingpf/cfcpflow/savedmodel/CFCPmodel_39_new.pth'
fcp_model.load_state_dict(torch.load(model_path, map_location=device))
fcp_model.eval()

# --------------------
# Create the data for evaluation
input_x = torch.empty((0, 2), dtype=torch.float64).to(device)
input_c = torch.empty((0, c_dim), dtype=torch.float64).to(device)
output_y = torch.empty((0, 2), dtype=torch.float64).to(device)
p_index = torch.randint(0, load_bus_num, (1,)).item()  # Random index for the power input for load bus (0 to 20)
v_index = p_index
for _b in range(1000):
    coeff_active = np.random.uniform(-1, 1)
    _active_power = base_ap + coeff_active * load_variance_rate * base_ap
    _reactive_power = _active_power * load_rato
    _active_power = _active_power.reshape(-1)
    _reactive_power = _reactive_power.reshape(-1)
    random_sys.set_loads(_active_power, _reactive_power)
    
    # Run the power flow analysis
    solution = random_sys.run_pf()
    solution = solution.loc[load_index].reset_index(drop=True)
    # Transform the voltage magnitudes
    _voltage_magnitudes = solution['vm_pu'].values
    _voltage_angles = solution['va_degree'].values
    
    
    _voltage_magnitudes = (_voltage_magnitudes / scaler['mean_voltage_magnitude']) - 1  # per unit deviation [-1, 1]
    _voltage_angles = (_voltage_angles / scaler['mean_voltage_angle']) - 1
    
    _voltages = np.hstack((_voltage_magnitudes, _voltage_angles))
    
    # Convert to torch tensor
    _active_power = (_active_power / scaler['mean_active_power']) - 1 
    _reactive_power = (_reactive_power / scaler['mean_reactive_power']) - 1
    
    _input_power = torch.tensor(np.hstack((_active_power, _reactive_power)), dtype=torch.float64).to(device)
    _input_power = _input_power.unsqueeze(0)  # shape (batch_size, 42)
    _target_voltage = torch.tensor(_voltages, dtype=torch.float64).to(device)
    _target_voltage = _target_voltage.unsqueeze(0)  # shape (batch_size, 42)
    # print(input_power.shape, target_voltage.shape)
    
    #-------input and target power flow data preparation-------
    _input_x = torch.cat((_input_power[:, p_index].unsqueeze(1), _input_power[:, p_index+load_bus_num-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
    _input_c = _input_power.clone()
    
    _output_y = torch.cat((_target_voltage[:, v_index].unsqueeze(1),
                        _target_voltage[:, v_index+load_bus_num-1].unsqueeze(1)
                        ), dim=1)
    # print(f"Input shape: {input_x.shape}, Condition shape: {input_c.shape}, Output shape: {output_y.shape}")
    input_x = torch.cat((input_x, _input_x), dim=0)
    input_c = torch.cat((input_c, _input_c), dim=0)
    output_y = torch.cat((output_y, _output_y), dim=0)

#--------------------
# Evaluate the model
with torch.no_grad():
    # Forward pass: x,c -> z
    z, log_det_jacobian = fcp_model.forward(input_x, input_c)
    
    # Inverse pass: z,c -> x_reconstructed
    x_reconstructed, inv_log_det_jacobian = fcp_model.inverse(z, input_c)
    
    # Compute the reconstruction loss
    reconstruction_loss = torch.mean((input_x - x_reconstructed) ** 2).item()
    
    print(f"Reconstruction Loss (MSE): {reconstruction_loss}")