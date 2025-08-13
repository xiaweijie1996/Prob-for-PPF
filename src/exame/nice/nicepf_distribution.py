import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.models.nice.nicemodel import NicemModel
from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers
from src.utility.inversepdf import inverse_pdf_gaussian

# set all np to be float64
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
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

# Define the data
_active_power = np.random.normal(50, scale=5, size=(10000, num_nodes-1))  # Power in kW
_reactive_power = _active_power * power_factor
_solution = random_sys.run(active_power=_active_power, 
                            reactive_power=_reactive_power, 
                            plot_graph=False)
_voltage_magnitudes = magnitude_transform(_solution['v'])
_voltage_angles = angle_transform(_solution['v'])

# Load the scalers
path_scalers = 'src/training/nice/savedmodel/scalers_34.pkl'
with open(path_scalers, 'rb') as f:
    scalers = pickle.load(f)
    
scaler_vm = scalers['scaler_vm']
scaler_va = scalers['scaler_va']
scaler_p = scalers['scaler_p']
scaler_q = scalers['scaler_q']

# Scale the data
scaled_active_power = scaler_p.transform(_active_power)
scaled_reactive_power = scaler_q.transform(_reactive_power)
scaled_voltage_magnitudes = scaler_vm.transform(_voltage_magnitudes)
scaled_voltage_angles = scaler_va.transform(_voltage_angles)

# Concatenate the scaled data
powers_active_reactive = np.concatenate((scaled_active_power, scaled_reactive_power), axis=1)
concat_vm_va = np.concatenate((scaled_voltage_magnitudes, scaled_voltage_angles), axis=1)

# Load the nice model
nice_model.load_state_dict(torch.load(f"src/training/nice/savedmodel/nicemodel_34.pth"))
nice_model.eval()

# Perform inference
# Fit GMM to input power data
gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
gmm.fit(powers_active_reactive)

log_p_y, p_y, inverse_p = inverse_pdf_gaussian(
    y= torch.tensor(concat_vm_va, dtype=torch.float32),
    model=nice_model,
    x_distribution=gmm,
    device=device
)
print("log_p_y :", log_p_y.max(), log_p_y.min())
print("p_y density:", p_y.max(), p_y.min())
print("p_y dtype:", p_y.dtype)

# Use PCA for dimensionality reduction for visualization
pca_v = PCA(n_components=2)
v_reduced = pca_v.fit_transform(concat_vm_va)

pca_p = PCA(n_components=2)
p_reduced = pca_p.fit_transform(powers_active_reactive)

# Check the accuracy of GAussian Mixture Model---------------
samples = gmm.sample(10000)[0]
samples_reduced = pca_p.transform(samples)

inverse_p_reduced = pca_p.transform(inverse_p.cpu().numpy())

# Plot the GMM samples in reduced space
plt.figure(figsize=(8, 6))
plt.subplot(1, 3, 1)
plt.scatter(samples_reduced[:, 0], samples_reduced[:, 1], alpha=0.5, s=1)
plt.title('GMM Samples in Reduced Space')
plt.xlabel('v_x')
plt.ylabel('v_y')
# Plot the PCA reduced data
plt.subplot(1, 3, 2)
plt.scatter(p_reduced[:, 0], p_reduced[:, 1], alpha=0.5, s=1)
plt.title('PCA Reduced Data')
plt.xlabel('v_x')
plt.ylabel('v_y')
# plot inverse_p
plt.subplot(1, 3, 3)
plt.scatter(inverse_p_reduced[:, 0], inverse_p_reduced[:, 1], alpha=0.5, s=1)
plt.title('Inverse PDF Samples')
plt.xlabel('v_x')
plt.ylabel('v_y')
plt.tight_layout()
plt.savefig("figures/gmm_samples_pca_reduced.png")
plt.close()

# Fit a Gaussian Mixture Model to the concat_vm_va data
gmm_v = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm_v.fit(concat_vm_va)

# Plot the cdf of v based on the pca data---------------
v_x_min, v_x_max = v_reduced[:, 0].min(), v_reduced[:, 0].max()
v_y_min, v_y_max = v_reduced[:, 1].min(), v_reduced[:, 1].max()
v_x_range = v_x_max - v_x_min
v_y_range = v_y_max - v_y_min
n_bins = 100

# Creat the 2d grid for the histogram
pad_x = 0.02 * (v_x_range + 1e-12)
pad_y = 0.02 * (v_y_range + 1e-12)
x_edges = np.linspace(v_x_min - pad_x, v_x_max + pad_x, n_bins + 1)
y_edges = np.linspace(v_y_min - pad_y, v_y_max + pad_y, n_bins + 1)
dx = np.diff(x_edges)[0]
dy = np.diff(y_edges)[0]
 
# sample from each bins to get the empirical pdf from the gmm_v
log_emp_pdf = gmm.score_samples(concat_vm_va)
log_emp_pdf = log_emp_pdf.reshape(-1, 1)  # Reshape to
print("log_emp_pdf shape:", log_emp_pdf.shape)
print("log_emp_pdf max:", log_emp_pdf.max(), "min:", log_emp_pdf.min())
emp_pdf = np.exp(log_emp_pdf)
print("emp_pdf max:", emp_pdf.max(), "min:", emp_pdf.min())

# Plot the scatter plot of the reduced data
normalized_log_emp_pdf = (log_emp_pdf - log_emp_pdf.min()) / (log_emp_pdf.max() - log_emp_pdf.min())
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.scatter(v_reduced[:, 0], v_reduced[:, 1], c=normalized_log_emp_pdf , cmap='viridis', alpha=0.5, s=1)
plt.colorbar(label='Log PDF')
plt.title('Scatter Plot of Reduced Data')
plt.xlabel('v_x')
plt.ylabel('v_y')       

normalized_log_p_y = (log_p_y - log_p_y.min()) / (log_p_y.max() - log_p_y.min())
plt.subplot(1, 2, 2)
plt.scatter(v_reduced[:, 0], v_reduced[:, 1], c= normalized_log_p_y, cmap='viridis', alpha=0.5, s=1)
plt.colorbar(label='Log PDF')
plt.title('2D Histogram of Reduced Data')
plt.xlabel('v_x')
plt.ylabel('v_y')
plt.tight_layout()
plt.savefig("figures/reduced_data_histogram.png")
plt.close()