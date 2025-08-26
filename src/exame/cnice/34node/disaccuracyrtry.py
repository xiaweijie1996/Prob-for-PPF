import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from src.models.cnice.cnicemodel import CNicemModel
from src.powersystems.node34 import Node34Example
from src.powersystems.randomsys import magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers
from src.utility.inversepdf import inverse_pdf_gaussian



# -----------------------
# Configureation
# -----------------------
num_nodes = 34
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

_root = 40
batch_size = _root**2
device = 'cpu'
save_path = 'src/training/cnice/savedmodel'

# Fix other nodes, vary one node
p_index = 12 # torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index
n_components = 1
print(f"Target node index: {p_index}, {v_index}")
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
# Define the data and distribution
# -----------------------
_weights = np.random.randint(1, 10, size=(n_components,))
_weights = _weights / np.sum(_weights)
_weights = _weights.round(2)
_active_power_index = np.random.normal(0,
                                      scale=5, 
                                      size=(batch_size, )) + mean_vector[1+p_index]
# for _w in _weights:
#     _active_power_index = np.concatenate((_active_power_index, 
#                                          np.random.normal(mean_vector[1+p_index],
#                                                           scale=5*_w, 
#                                                           size=(int(batch_size*_w), ))))
_active_power_index = _active_power_index[:batch_size]
print(f"p_index: {p_index}, v_index: {v_index}, weights: {_weights}, _active_power_index: {_active_power_index.shape}")

# Replace _active_power_index with sampled_index
_active_power = np.tile(mean_vector[1:], (batch_size, 1))
_active_power[:, p_index] = _active_power_index
_reactive_power = _active_power * power_factor

# Run the power flow to get the voltages
_solution = random_sys.run(active_power=_active_power, 
                            reactive_power=_reactive_power)
_voltage_magnitudes = magnitude_transform(_solution['v'])
_voltage_angles = angle_transform(_solution['v'])

# Scale the data
scaled_vm = scaler_vm.transform(_voltage_magnitudes)
scaled_va = scaler_va.transform(_voltage_angles)
scaled_active_power = scaler_p.transform(_active_power)
scaled_reactive_power = scaler_q.transform(_reactive_power)

# Fig a gmm to the input data
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
_target_samples = np.concatenate((scaled_active_power[:, p_index].reshape(-1, 1),
                                  scaled_reactive_power[:, p_index].reshape(-1, 1)), axis=1)
# gmm.fit(_target_samples[:, 0].reshape(-1, 1))
gmm.fit(_target_samples)
print(f"GMM means: {gmm.means_}, covariances: {gmm.covariances_}, weights: {gmm.weights_}")
print(f"target samples, mean and variance: {_target_samples.mean(axis=0)}, {_target_samples.std(axis=0)}")
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

# ----------------------- 
# Check distribution 
# ----------------------- 
# f(x) = y
# p(x) = p(y) * |det(d f^-1(y)/ dy)
# # Plot the empirical pdf and cdf of x
# y = ((scaled_vm[:, v_index].reshape(-1, 1),
#                     scaled_va[:, v_index].reshape(-1, 1)), axis=1)  # Original output data
y = output_y.cpu().numpy()
print(f"y: {y.shape}")

# Check the min and max if x and y
max_y0, min_y0 = y[:,0].max().item(), y[:,0].min().item()
max_y1, min_y1 = y[:,1].max().item(), y[:,1].min().item()
print(f"y0: min {min_y0}, max {max_y0}, y1: min {min_y1}, max {max_y1}")
n_bins = int(np.sqrt(batch_size))  # Number of bins for the histogram
y0_line = np.linspace(min_y0, max_y0, n_bins)
y1_line = np.linspace(min_y1, max_y1, n_bins)
gap_area = (max_y0 - min_y0) * (max_y1 - min_y1) / (n_bins * n_bins)

grid_yy0, grid_yy1 = np.meshgrid(y0_line, y1_line)
grid_y0, grid_y1 = np.meshgrid(y0_line, y1_line, sparse=True)
density = np.zeros((n_bins, n_bins))
cum_density = np.zeros((n_bins, n_bins))
print("Grid shape:", grid_y0.shape, grid_y1.shape)
# print(grid_y0)
# print(grid_y1)
# compute the density of each bin  
for i in range(n_bins):
    for j in range(n_bins):
        # pdf filter
        filter = (
        (y[:, 1] >= grid_y1[j, 0]) &
        (y[:, 1] <  grid_y1[j, 0] + (max_y1 - min_y1) / n_bins) &  # y in y-bin j
        (y[:, 0] >= grid_y0[0, i]) &
        (y[:, 0] <  grid_y0[0, i] + (max_y0 - min_y0) / n_bins)    # x in x-bin i
        )

        density[j, i] = np.sum(filter) / (batch_size * gap_area)
        
        # cdf filter
        cdf_filter = (
            (y[:, 1] <= grid_y1[j, 0]) &
            (y[:, 0] <= grid_y0[0, i])
        )
        cum_density[j, i] = np.sum(cdf_filter) / batch_size
        
# Plot the pdf and cdf in a 3d plot
fig = plt.figure(figsize=(14, 6))

# First subplot: PDF
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(grid_yy0, grid_yy1, density, cmap='viridis', edgecolor='none')
ax1.set_title('PDF of Output')
ax1.set_xlabel('v_m')
ax1.set_ylabel('v_a')
ax1.set_zlabel('Density')

# Second subplot: CDF
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(grid_yy0, grid_yy1,  cum_density, cmap='viridis', edgecolor='none')
ax2.set_title('CDF of Output')
ax2.set_xlabel('v_m')
ax2.set_ylabel('v_a')
ax2.set_zlabel('Cumulative Density')

plt.tight_layout()
plt.savefig(f'figures/target_node_{v_index}_{num_nodes}_pdf_cdf.png')
plt.close()

# ----------------------- 
# Compute the density of the output using the model
# -----------------------
# Condition input, the same for all does not matter the p_index as it will be replaced in the null token, the scenario is fixed
nice_model.eval()
x_inverse, _ja_inverse = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
# x_inverse[:,1] = x_inverse[:,0]  # only keep the active power
# p_y_compute = gmm.score_samples(x_inverse[:,0].detach().numpy().reshape(-1, 1))
p_y_compute = gmm.score_samples(x_inverse.detach().cpu().numpy())
p_y_compute = torch.tensor(p_y_compute, dtype=torch.float32)
p_y_compute = p_y_compute.exp().cpu() * _ja_inverse.cpu()
print(_ja_inverse.mean().item(), p_y_compute.mean().item())

# Compute the density for each bin
density_y = torch.zeros((n_bins, n_bins))
for i in range(n_bins):
    for j in range(n_bins):
        # pdf filter
        filter = (
        (y[:, 1] >= grid_y1[j, 0]) &
        (y[:, 1] <  grid_y1[j, 0] + (max_y1 - min_y1) / n_bins) &  # y in y-bin j
        (y[:, 0] >= grid_y0[0, i]) &
        (y[:, 0] <  grid_y0[0, i] + (max_y0 - min_y0) / n_bins)    # x in x-bin i
        )
        
        if np.sum(filter) > 0:
            density_y[j, i] = p_y_compute[filter].mean()  # multiply by the area
        else:
            density_y[j, i] = 0.0

# Compute cdf
cum_density_y = torch.zeros((n_bins, n_bins))
for i in range(n_bins):
    for j in range(n_bins):
        # sum over all bins less than or equal to (i, j)
        density_sum = density_y[:j+1, :i+1].sum()
        cum_density_y[j, i] = density_sum * gap_area  # multiply by the area

# max_cum_density_y = cum_density_y.max().item()
# cum_density_y = cum_density_y / max_cum_density_y  # normalize to 1
# density_y = density_y/max_cum_density_y

# plot the density of the output
fig = plt.figure(figsize=(14, 6))
# First subplot: PDF
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(grid_yy0, grid_yy1, density_y.detach().numpy(), cmap='viridis', edgecolor='none')
ax1.set_title('Computed PDF of Output')
ax1.set_xlabel('v_m')
ax1.set_ylabel('v_a')
ax1.set_zlabel('Density')
# Second subplot: CDF
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(grid_yy0, grid_yy1, cum_density_y.detach().numpy(), cmap='viridis', edgecolor='none')
ax2.set_title('Computed CDF of Output')
ax2.set_xlabel('v_m')
ax2.set_ylabel('v_a')
ax2.set_zlabel('Cumulative Density')    
plt.tight_layout()
plt.savefig(f'figures/target_node_{v_index}_{num_nodes}_computed_pdf_cdf.png')
plt.close()

# -----------------------
# Error analysis
# -----------------------
# Plot distribution of inverse_x and input_x
input_x_np = input_x.cpu().numpy()
x_inverse_np = x_inverse.detach().cpu().numpy()
# fig, ax = plt.subplots(subplot_kw={"projection"})
# 2D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(input_x_np[:,0], input_x_np[:,1], c='b', label='Input x', alpha=0.5)
ax.scatter(x_inverse_np[:,0], x_inverse_np[:,1], c='r', label='Inverse x', alpha=0.5)
ax.set_title('Input x and Inverse x Distribution')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.legend()
plt.savefig(f'figures/target_node_x_{v_index}_{num_nodes}_input_inverse_distribution_2d.png')
plt.close()

# plot the density comparison if x_inverse and input_x
log_density_input_x = gmm.score_samples(input_x_np)
log_density_x_inverse = gmm.score_samples(x_inverse_np)
density_input_x = np.exp(log_density_input_x)
density_x_inverse = np.exp(log_density_x_inverse)
# 3d two surface plot
fig = plt.figure(figsize=(14, 6))
# First subplot: Input x density
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(input_x_np[:,0], input_x_np[:,1], density_input_x, c='b', label='Input x Density', alpha=0.5)
ax1.set_title('Input x Density')
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.set_zlabel('Density')
# Second subplot: Inverse x density
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(x_inverse_np[:,0], x_inverse_np[:,1], density_x_inverse, c='r', label='Inverse x Density', alpha=0.5)
ax2.set_title('Inverse x Density')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_zlabel('Density')
plt.tight_layout()
plt.savefig(f'figures/target_node_x_{v_index}_{num_nodes}_input_inverse_density_3d.png')
plt.close()
