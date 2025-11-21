import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.mixedflow.mixedflow import CMixedModel
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.ieee39 import Case39PF
from src.testingpf.evaluationmetrics import mse_loss, rmse_loss

# Set all torch floats to double precision
torch.set_default_dtype(torch.float64)

# Configureation
# -----------------------
with open('src/config39bus.yaml', 'r') as f:
    config = yaml.safe_load(f)

num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
load_bus_num = config['SystemAndDistribution']['load_bus_num']  # Number of load buses
load_variance_rate = config['SystemAndDistribution']['load_variance_rate']

split_ratio = config['MixedSplineFCP']['split_ratio']
n_blocks_spline = config['MixedSplineFCP']['n_blocks_spline']
n_blocks_fcp = config['MixedSplineFCP']['n_blocks_fcp']
hiddemen_dim = config['MixedSplineFCP']['hiddemen_dim']
c_dim = (2 * load_bus_num)  # Condition dimension (P and Q for all nodes except slack)
n_layers = config['MixedSplineFCP']['n_layers']
input_dim = config['MixedSplineFCP']['input_dim']  # Input dimension (P and Q for one node)
hiddemen_dim_condition = config['MixedSplineFCP']['hiddemen_dim_condition']
n_layers_condition = config['MixedSplineFCP']['n_layers_condition']
b_interval = config['MixedSplineFCP']['b_interval']
k_bins = config['MixedSplineFCP']['k_bins']

load_variance_rate = config['SystemAndDistribution']['load_variance_rate']
batch_size = config['MixedSplineFCP']['batch_size']
epochs = config['MixedSplineFCP']['epochs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = config['MixedSplineFCP']['save_path']
lr = config['MixedSplineFCP']['lr']
forward_loss_ratio = config['MixedSplineFCP'].get('forward_loss_ratio', 1.0)  # Default to 1.0 if not specified
# -----------------------

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
mix_model = CMixedModel(
    input_dim=input_dim,
    hidden_dim=hiddemen_dim,
    condition_dim=c_dim,
    n_layers=n_layers,
    split_ratio=split_ratio,
    n_blocks_fcp=n_blocks_fcp,
    n_blocks_spline=n_blocks_spline,
    hidden_dim_condition=hiddemen_dim_condition,
    n_layers_condition=n_layers_condition,
    b_interval=b_interval,
    k_bins=k_bins
).to(device)
mix_model.double()
print(f"Model Parameters: {sum(p.numel() for p in mix_model.parameters() if p.requires_grad)}")

# Load pre-trained model
model_path = 'src/trainingpf/mixedflow/savedmodel/MixedSplineFCPmodel_39.pth'
mix_model.load_state_dict(torch.load(model_path, map_location=device))
mix_model.eval()

# --------------------
# Create the data for evaluation
power_mse_list = []
power_rmse_list = []
voltage_mse_list = []
voltage_rmse_list = []
for p_index in range(load_bus_num):
    input_x = torch.empty((0, 2), dtype=torch.float64).to(device)
    input_c = torch.empty((0, c_dim), dtype=torch.float64).to(device)
    output_y = torch.empty((0, 2), dtype=torch.float64).to(device)
    v_index = p_index
    print(f"Evaluating for load bus index: {p_index}")
    for _b in tqdm(range(10)):
        coeff_active = np.random.uniform(-1, 1, size=load_bus_num)  # Random coefficient for active power
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
        # Forward pass through the NICE model
        print("Evaluating the model...")
        print('p_index:', p_index, 'v_index:', v_index)
        output_voltage, _ja = mix_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
            
        # Backward pass to get the output power
        output_power, _j = mix_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
            
        # Transform back to original scale
        output_power_np = output_power.cpu().numpy()
        output_voltage_np = output_voltage.cpu().numpy()
        output_power_np[:, 0] = (output_power_np[:, 0] + 1) * scaler['mean_active_power'][p_index]
        output_power_np[:, 1] = (output_power_np[:, 1] + 1) * scaler['mean_reactive_power'][p_index]
        output_voltage_np[:, 0] = (output_voltage_np[:, 0] + 1) * scaler['mean_voltage_magnitude'][v_index]
        output_voltage_np[:, 1] = (output_voltage_np[:, 1] + 1) * scaler['mean_voltage_angle']
        
        # Transform target back to original scale
        target_power_np = input_x.cpu().numpy()
        target_power_np[:, 0] = (target_power_np[:, 0] + 1) * scaler['mean_active_power'][p_index]
        target_power_np[:, 1] = (target_power_np[:, 1] + 1) * scaler['mean_reactive_power'][p_index]
        target_voltage_np = output_y.cpu().numpy()
        target_voltage_np[:, 0] = (target_voltage_np[:, 0] + 1) * scaler['mean_voltage_magnitude'][v_index]
        target_voltage_np[:, 1] = (target_voltage_np[:, 1] + 1) * scaler['mean_voltage_angle']
        
        # Calculate evaluation metrics, numpy
        power_mse = mse_loss(output_power_np, target_power_np)
        power_rmse = rmse_loss(output_power_np, target_power_np)
        voltage_mse = mse_loss(output_voltage_np, target_voltage_np)
        voltage_rmse = rmse_loss(output_voltage_np, target_voltage_np)
        power_mse_list.append(power_mse)
        power_rmse_list.append(power_rmse)
        voltage_mse_list.append(voltage_mse)
        voltage_rmse_list.append(voltage_rmse)

#--------------------
# Save and print the evaluation results as a text file
results_dir = 'src/testingpf/cmixedflowtest/evaluationresults'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_path = os.path.join(results_dir, 'cmixedflow_evaluation_39bus.txt')
with open(results_path, 'w') as f:
    f.write("CFCP-Flow Evaluation Results on IEEE 39-Bus System\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'Bus':<10}{'Power MSE':<20}{'Power RMSE':<20}{'Voltage MSE':<20}{'Voltage RMSE':<20}\n")
    for i in range(load_bus_num):
        f.write(f"{i:<10}{power_mse_list[i]:<20.6f}{power_rmse_list[i]:<20.6f}{voltage_mse_list[i]:<20.6f}{voltage_rmse_list[i]:<20.6f}\n")
    f.write("--------------------------------------------------\n")
    f.write(f"Average Power MSE: {np.mean(power_mse_list):.6f}\n")
    f.write(f"Average Power RMSE: {np.mean(power_rmse_list):.6f}\n")
    f.write(f"Average Voltage MSE: {np.mean(voltage_mse_list):.6f}\n")
    f.write(f"Average Voltage RMSE: {np.mean(voltage_rmse_list):.6f}\n")