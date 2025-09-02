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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = config['CNice']['save_path']
batch_size = 10000
n_bins = 20  # Number of bins for the histogram

# Fix other nodes, vary one node
p_index = 12 # torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index
print(f"Target node index: {p_index}, {v_index}")

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
# Load GMM, MODEL, and Scalers
# -----------------------
with open(dis_path, 'rb') as f:
    gmm = pickle.load(f)  
    
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

model_path = os.path.join(save_path, f"cnicemodel_{num_nodes}.pth")
nice_model.load_state_dict(torch.load(model_path, map_location=device))

# Define another GMM for p_index active and reactive power
gmm_p_index = GaussianMixture(n_components=gmm.n_components, covariance_type=gmm.covariance_type)
gmm_p_index.means_ = gmm.means_[:, [p_index, p_index + num_nodes - 1]]
gmm_p_index.covariances_ = gmm.covariances_[:, [p_index, p_index + num_nodes - 1], :][:, :, [p_index, p_index + num_nodes - 1]]
gmm_p_index.weights_ = gmm.weights_
gmm_p_index.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm_p_index.covariances_))

# -----------------------    
# Define the data and distribution
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

input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
input_c = input_power.clone()
output_y = torch.cat((target_voltage[:, v_index].unsqueeze(1),
                      target_voltage[:, v_index+num_nodes-1].unsqueeze(1)), dim=1)

nice_model.eval()
with torch.no_grad():
    pre_v, _jaf = nice_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
    pre_p, _jai = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
    
pre_v = pre_v.cpu().numpy()
pre_p = pre_p.cpu().numpy()
print(f"pre_v: {pre_v.shape}, pre_p: {pre_p.shape}")

# Scale back to original
pre_v_magnitude = pre_v[:, 0] * scaler['std_voltage_magnitude'][p_index] + scaler['mean_voltage_magnitude'][p_index]
pre_v_angle = pre_v[:, 1] * scaler['std_voltage_angle'][p_index] + scaler['mean_voltage_angle'][p_index]
pre_p_active = pre_p[:, 0] * scaler['std_active_power'][p_index] + scaler['mean_active_power'][p_index]
pre_p_reactive = pre_p[:, 1] * scaler['std_reactive_power'][p_index] + scaler['mean_reactive_power'][p_index]

true_v_magnitude = voltage_magnitudes[:, p_index]
true_v_angle = voltage_angles[:, p_index]
true_p_active = active_power[:, p_index]
true_p_reactive = reactive_power[:, p_index]

# ----------------------- 
# Check distribution 
# ----------------------- 
# f(x) = y
# p(x) = p(y) * |det(d f^-1(y)/ dy)
# # Plot the empirical pdf and cdf of x
# y = np.hstack((true_v_magnitude.reshape(-1,1), true_v_angle.reshape(-1,1)))
y = pre_v

# Check the min and max if x and y
max_y0, min_y0 = y[:,0].max().item(), y[:,0].min().item()
max_y1, min_y1 = y[:,1].max().item(), y[:,1].min().item()
print(f"y0: min {min_y0}, max {max_y0}, y1: min {min_y1}, max {max_y1}")
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
# nice_model.eval()
# with torch.no_grad():
#     x_inverse, _ja_inverse = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)

# Inverse the data
x_inverse = pre_p
x_inverse_active = x_inverse[:, 0] * scaler['std_active_power'][p_index] + scaler['mean_active_power'][p_index]
x_inverse_reactive = x_inverse[:, 1] * scaler['std_reactive_power'][p_index] + scaler['mean_reactive_power'][p_index]
x_inverse_denormalized = np.hstack((x_inverse_active.reshape(-1,1), x_inverse_reactive.reshape(-1,1)))
print(f"x_inverse_active: min {x_inverse_active.min()}, max {x_inverse_active.max()}, x_inverse_reactive: min {x_inverse_reactive.min()}, max {x_inverse_reactive.max()}")
print(f"x_inverse_denormalized shape: {x_inverse_denormalized.shape}")
# # Compute the density p(x) using GMM
# _ja_inverse = _ja_inverse.cpu().numpy()
# p_compute = gmm_p_index.score_samples(x_inverse_denormalized)
# p_compute = np.exp(p_compute)   # p(y) * |det(d f^-1(y)/ dy)
# print("Computed p(y) mean:", p_y_compute.mean().item(), "std:", p_y_compute.std().item())
# # y -> y_scaled ja is 
# ja_scaler_y =  scaler['std_voltage_angle'][v_index] * scaler['std_voltage_magnitude'][v_index]
# # y_scaled -> x x_scaled ja is _ja_inverse
# # x_scaled -> x ja is scaler['std'][[p_index, p_index + num_nodes - 1]]
# ja_scale_x = scaler['std_active_power'][p_index] * scaler['std_reactive_power'][p_index]
# p_y_compute = p_y_compute * _ja_inverse  / (ja_scaler_y * ja_scale_x) * (8/600/0.000001 )
# print('all jas are:', ja_scaler_y, ja_scale_x, _ja_inverse.mean().item())

# print(_ja_inverse.mean().item(), p_y_compute.mean().item())

# # Compute the density for each bin
# density_y = torch.zeros((n_bins, n_bins))
# for i in range(n_bins):
#     for j in range(n_bins):
#         # pdf filter
#         filter = (
#         (y[:, 1] >= grid_y1[j, 0]) &
#         (y[:, 1] <  grid_y1[j, 0] + (max_y1 - min_y1) / n_bins) &  # y in y-bin j
#         (y[:, 0] >= grid_y0[0, i]) &
#         (y[:, 0] <  grid_y0[0, i] + (max_y0 - min_y0) / n_bins)    # x in x-bin i
#         )
   
#         if np.sum(filter) > 0:
#             density_y[j, i] = p_y_compute[filter].mean()  # multiply by the area
#         else:
#             density_y[j, i] = 0.0

# # Compute cdf
# cum_density_y = torch.zeros((n_bins, n_bins))
# for i in range(n_bins):
#     for j in range(n_bins):
#         # sum over all bins less than or equal to (i, j)
#         density_sum = density_y[:j+1, :i+1].sum()
#         cum_density_y[j, i] = density_sum * gap_area  # multiply by the area
        
# # plot the density of the output
# fig = plt.figure(figsize=(14, 6))
# # First subplot: PDF
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.plot_surface(grid_yy0, grid_yy1, density_y.detach().numpy(), cmap='viridis', edgecolor='none')
# ax1.set_title('Computed PDF of Output')
# ax1.set_xlabel('v_m')
# ax1.set_ylabel('v_a')
# ax1.set_zlabel('Density')
# # Second subplot: CDF
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# ax2.plot_surface(grid_yy0, grid_yy1, cum_density_y.detach().numpy(), cmap='viridis', edgecolor='none')
# ax2.set_title('Computed CDF of Output')
# ax2.set_xlabel('v_m')
# ax2.set_ylabel('v_a')
# ax2.set_zlabel('Cumulative Density')    
# plt.tight_layout()
# plt.savefig(f'figures/target_node_{v_index}_{num_nodes}_computed_pdf_cdf.png')
# plt.close()

# -----------------------
# Error analysis
# -----------------------
# Plot distribution of inverse_x and input_x
# input_x_np = np.hstack((true_p_active.reshape(-1,1), true_p_reactive.reshape(-1,1)))
# x_inverse_np = x_inverse_denormalized
input_x_np= input_x.cpu().numpy()
x_inverse_np = pre_p
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
plt.savefig(f'figures/scaled_powrer_distribution.png')
plt.close()

# Same 3D plot to plot the pdf and cdf of input_x and inverse_x using the same method as above
# input_x use MC mehtod
# pdf of inverse_x use gmm to get the p(x) directly
# Check the min and max if x and y
max_x0, min_x0 = input_x_np[:,0].max().item(), input_x_np[:,0].min().item()
max_x1, min_x1 = input_x_np[:,1].max().item(), input_x_np[:,1].min().item()
print(f"x0: min {min_x0}, max {max_x0}, x1: min {min_x1}, max {max_x1}")
x0_line = np.linspace(min_x0, max_x0, n_bins)
x1_line = np.linspace(min_x1, max_x1, n_bins)
gap_area_x = (max_x0 - min_x0) * (max_x1 - min_x1) / (n_bins * n_bins)

grid_xx0, grid_xx1 = np.meshgrid(x0_line, x1_line)
grid_x0, grid_x1 = np.meshgrid(x0_line, x1_line, sparse=True)
density_input = np.zeros((n_bins, n_bins))
density_inverse = np.zeros((n_bins, n_bins))
print("Grid shape:", grid_x0.shape, grid_x1.shape)
# print(grid_x0)
# print(grid_x1)
# compute the density of each bin  
for i in range(n_bins):
    for j in range(n_bins):
        # pdf filter for input_x
        filter_input = (
        (input_x_np[:, 1] >= grid_x1[j, 0]) &
        (input_x_np[:, 1] <  grid_x1[j, 0] + (max_x1 - min_x1) / n_bins) &  # x in x-bin j
        (input_x_np[:, 0] >= grid_x0[0, i]) &
        (input_x_np[:, 0] <  grid_x0[0, i] + (max_x0 - min_x0) / n_bins)    # x in x-bin i
        )

        density_input[j, i] = np.sum(filter_input) / (batch_size * gap_area_x)
        
        # pdf filter for inverse_x
        filter_inverse = (
        (x_inverse_np[:, 1] >= grid_x1[j, 0]) &
        (x_inverse_np[:, 1] <  grid_x1[j, 0] + (max_x1 - min_x1) / n_bins) &  # x in x-bin j
        (x_inverse_np[:, 0] >= grid_x0[0, i]) &
        (x_inverse_np[:, 0] <  grid_x0[0, i] + (max_x0 - min_x0) / n_bins)    # x in x-bin i
        )

        density_inverse[j, i] = np.sum(filter_inverse) / (batch_size * gap_area_x)
        
# Plot the pdf and cdf in a 3d plot
fig = plt.figure(figsize=(14, 6))
# First subplot: PDF
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(grid_xx0, grid_xx1, density_input, cmap='viridis', edgecolor='none')
ax1.set_title('PDF of Input x')
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.set_zlabel('Density')
# Second subplot: PDF of inverse_x
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(grid_xx0, grid_xx1, density_inverse, cmap='viridis', edgecolor='none')
ax2.set_title('PDF of Inverse x')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_zlabel('Density')    
plt.tight_layout()
plt.savefig(f'figures/target_node_x_{v_index}_{num_nodes}_input_inverse_pdf_3d.png')
plt.close()

# Get the density of inverse_x using gmm
# --- IGNORE ---
density = gmm_p_index.score_samples(x_inverse_denormalized)
density = np.exp(density)

# Plot the density of x_inverse_np using gmm using 3d plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_inverse_np[:,0], x_inverse_np[:,1], density, c='r', label='Inverse x Density', alpha=0.5)
ax.set_title('Density of Inverse x using GMM')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('Density')
ax.legend()
plt.savefig(f'figures/target_node_x_{v_index}_{num_nodes}_inverse_density_gmm_3d.png')
plt.close()
# --- IGNORE ---
