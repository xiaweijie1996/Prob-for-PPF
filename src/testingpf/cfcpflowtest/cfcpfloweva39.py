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

from src.models.cfcpflow.cfcpflow import SimplifiedFcpflow
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
power_active_mse_list = []
power_active_rmse_list = []
power_reactive_mse_list = []
power_reactive_rmse_list = []
voltage_magnitude_mse_list = []
voltage_magnitude_rmse_list = []
voltage_angle_mse_list = []
voltage_angle_rmse_list = []
for p_index in range(load_bus_num):
    input_x = torch.empty((0, 2), dtype=torch.float64).to(device)
    input_c = torch.empty((0, c_dim), dtype=torch.float64).to(device)
    output_y = torch.empty((0, 2), dtype=torch.float64).to(device)
    v_index = p_index
    print(f"Evaluating for load bus index: {p_index}")
    for _b in tqdm(range(10)):
        coeff_active = np.random.uniform(-1, 1, size=load_bus_num)
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
        output_voltage, _ja = fcp_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
            
        # Backward pass to get the output power
        output_power, _j = fcp_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
            
        # # Transform back to original scale
        output_power_np = output_power.cpu().numpy()
        output_voltage_np = output_voltage.cpu().numpy()
        # output_power_np[:, 0] = (output_power_np[:, 0] + 1) * scaler['mean_active_power'][p_index]
        # output_power_np[:, 1] = (output_power_np[:, 1] + 1) * scaler['mean_reactive_power'][p_index]
        # output_voltage_np[:, 0] = (output_voltage_np[:, 0] + 1) * scaler['mean_voltage_magnitude'][v_index]
        # output_voltage_np[:, 1] = (output_voltage_np[:, 1] + 1) * scaler['mean_voltage_angle']
        
        # # Transform target back to original scale
        target_power_np = input_x.cpu().numpy()
        # target_power_np[:, 0] = (target_power_np[:, 0] + 1) * scaler['mean_active_power'][p_index]
        # target_power_np[:, 1] = (target_power_np[:, 1] + 1) * scaler['mean_reactive_power'][p_index]
        target_voltage_np = output_y.cpu().numpy()
        # target_voltage_np[:, 0] = (target_voltage_np[:, 0] + 1) * scaler['mean_voltage_magnitude'][v_index]
        # target_voltage_np[:, 1] = (target_voltage_np[:, 1] + 1) * scaler['mean_voltage_angle']
        
        # Calculate evaluation metrics, numpy
        power_mse_active = mse_loss(output_power_np[:, 0], target_power_np[:, 0])
        power_mse_reactive = mse_loss(output_power_np[:, 1], target_power_np[:, 1])
        power_rmse_active = rmse_loss(output_power_np[:, 0], target_power_np[:, 0])
        power_rmse_reactive = rmse_loss(output_power_np[:, 1], target_power_np[:, 1])
        voltage_mse_magnitude = mse_loss(output_voltage_np[:, 0], target_voltage_np[:, 0])
        voltage_mse_angle = mse_loss(output_voltage_np[:, 1], target_voltage_np[:, 1])
        voltage_rmse_magnitude = rmse_loss(output_voltage_np[:, 0], target_voltage_np[:, 0])
        voltage_rmse_angle = rmse_loss(output_voltage_np[:, 1], target_voltage_np[:, 1])    
        
        # Append results
        power_active_mse_list.append(power_mse_active)
        power_active_rmse_list.append(power_rmse_active)
        power_reactive_mse_list.append(power_mse_reactive)
        power_reactive_rmse_list.append(power_rmse_reactive)
        voltage_magnitude_mse_list.append(voltage_mse_magnitude)
        voltage_magnitude_rmse_list.append(voltage_rmse_magnitude)
        voltage_angle_mse_list.append(voltage_mse_angle)
        voltage_angle_rmse_list.append(voltage_rmse_angle)
        
#--------------------
# Save and print the evaluation results as a text file
results_dir = 'src/testingpf/cfcpflowtest/evaluationresults'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_path = os.path.join(results_dir, 'cfcpflow_evaluation_39bus.txt')
with open(results_path, 'w') as f:
    f.write("CFCP-Flow Evaluation Results on IEEE 39-Bus System\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'Bus':<10}{'Active Power MSE':<20}{'Active Power RMSE':<20}{'Reactive Power MSE':<20}{'Reactive Power RMSE':<20}{'Voltage Mag MSE':<20}{'Voltage Mag RMSE':<20}{'Voltage Angle MSE':<20}{'Voltage Angle RMSE':<20}\n")
    for i in range(load_bus_num):
        f.write(f"{i:<10}{power_active_mse_list[i]:<20.6f}{power_active_rmse_list[i]:<20.6f}{power_reactive_mse_list[i]:<20.6f}{power_reactive_rmse_list[i]:<20.6f}{voltage_magnitude_mse_list[i]:<20.6f}{voltage_magnitude_rmse_list[i]:<20.6f}{voltage_angle_mse_list[i]:<20.6f}{voltage_angle_rmse_list[i]:<20.6f}\n")
    f.write("--------------------------------------------------\n")
    f.write(f"Average Active Power MSE: {np.mean(power_active_mse_list):.6f}\n")
    f.write(f"Average Active Power RMSE: {np.mean(power_active_rmse_list):.6f}\n")
    f.write(f"Average Reactive Power MSE: {np.mean(power_reactive_mse_list):.6f}\n")
    f.write(f"Average Reactive Power RMSE: {np.mean(power_reactive_rmse_list):.6f}\n")
    f.write(f"Average Voltage Magnitude MSE: {np.mean(voltage_magnitude_mse_list):.6f}\n")
    f.write(f"Average Voltage Magnitude RMSE: {np.mean(voltage_magnitude_rmse_list):.6f}\n")
    f.write(f"Average Voltage Angle MSE: {np.mean(voltage_angle_mse_list):.6f}\n")
    f.write(f"Average Voltage Angle RMSE: {np.mean(voltage_angle_rmse_list):.6f}\n")