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
num_nodes = 4
std = 5

split_ratio = 0.5
n_blocks = 3
hiddemen_dim = 24
c_dim = (num_nodes - 1) * 2
n_layers = 3
input_dim = 2  # Assuming each node has a real and imaginary part
hiddemen_dim_condition = 24
output_dim_condition = 1
n_layers_condition = 2

_root = 100
batch_size = _root**2
n_bins = 20  # Number of bins for the histogram
device = 'cpu'
save_path = 'src/training/cnice/savedmodel'

# Fix other nodes, vary one node
p_index = 2 # torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index
n_components = 3
print(f"Target node index: {p_index}, {v_index}")

# -----------------------
# Initialize the random system　model and scalers
# -----------------------
random_sys = CNicemModel(
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

# Check if toolcase model exists
if os.path.exists(os.path.join(save_path, f"toolcase_target_{num_nodes}.pth")):
    print(f"Loading toolcase model from {os.path.join(save_path, f'toolcase_target_{num_nodes}.pth')}")
    random_sys.load_state_dict(torch.load(os.path.join(save_path, f"toolcase_target_{num_nodes}.pth")))
else:
    # Save current initialized model as toolcase model
    print('Toolcase model not found. Please run toolcase.py first to create the toolcase model.')
    # print(os.path.join(save_path, f"toolcase_target_{num_nodes}.pth"))
    

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
model_path = os.path.join(save_path, f"toolcase_{num_nodes}.pth")
nice_model.load_state_dict(torch.load(model_path, map_location=device))

# -----------------------    
# Define the data and distribution
# -----------------------
active_power_index = np.random.normal(0, scale=std, size=(batch_size, 1))
reactive_power_index = np.random.normal(0, scale=std, size=(batch_size, 1)) * np.random.uniform(0.01, 0.5, size=(batch_size, 1))
_target_samples = np.hstack((active_power_index, reactive_power_index))

# Fig a gmm to the input data
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
gmm.fit(_target_samples)
samples_gmm = gmm.sample(batch_size)[0]
active_power_index = samples_gmm[:, 0].reshape(-1, 1)
reactive_power_index = samples_gmm[:, 1].reshape(-1, 1)

active_power = np.random.normal(0, scale=std, size=(1, num_nodes-1))
active_power = np.repeat(active_power, batch_size, axis=0)
active_power[:, p_index] = active_power_index[:, 0]
reactive_power = np.random.normal(0, scale=std, size=(1, num_nodes-1)) * np.random.uniform(0.01, 0.5, size=(1, num_nodes-1))  # Random power factor between 0.1 and 0.3
reactive_power = np.repeat(reactive_power, batch_size, axis=0)
reactive_power[:, p_index] = reactive_power_index[:, 0]

input_power = torch.tensor(np.hstack((active_power, reactive_power)), dtype=torch.float32).to(device)

input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
input_c = input_power.clone()

with torch.no_grad():
    output_y = random_sys.forward(input_x, input_c, index_p=p_index, index_v=v_index)[0].detach()

# ----------------------- 
# Evaluate the model
# ----------------------- 
nice_model.eval()
with torch.no_grad():
    pre_v, _jaf = nice_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
    # fake_output_y = pre_v + torch.randn_like(output_y) * 0.1
    pre_p, _jai = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
    
pre_v = pre_v.cpu().numpy()
pre_p = pre_p.cpu().numpy()
print(f"pre_v: {pre_v.shape}, pre_p: {pre_p.shape}")

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
plt.savefig(f'figures/toolcase_node_{v_index}_{num_nodes}_pdf_cdf.png')
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

max_cum_density_y = cum_density_y.max().item()
cum_density_y = cum_density_y / max_cum_density_y  # normalize to 1
density_y = density_y/max_cum_density_y

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
plt.savefig(f'figures/toolcase_{v_index}_{num_nodes}_computed_pdf_cdf.png')
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
plt.savefig(f'figures/toolcase_{v_index}_{num_nodes}_input_inverse_distribution_2d.png')
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
plt.savefig(f'figures/toolcase_{v_index}_{num_nodes}_input_inverse_density_3d.png')
plt.close()
