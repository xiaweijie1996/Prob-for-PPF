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
concat_vm_va = np.concatenate((_voltage_magnitudes, _voltage_angles), axis=1)

# Load the scalers
path_scalers = 'src/training/nice/savedmodel/scalers_34.pkl'
with open(path_scalers, 'rb') as f:
    scalers = pickle.load(f)
    
scaler_vm = scalers['scaler_vm']
scaler_va = scalers['scaler_va']
scaler_p = scalers['scaler_p']
scaler_q = scalers['scaler_q']

# Load the nice model
nice_model.load_state_dict(torch.load(f"src/training/nice/savedmodel/nicemodel_34.pth"))
nice_model.eval()

# Perform inference
# Fit GMM to input power data
gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
gmm.fit(_active_power)
_mean = gmm.means_[0]
_cov = gmm.covariances_[0]
print("GMM Mean:", _mean.shape)
print("GMM Covariance:", _cov.shape)

log_p_y, p_y, x = inverse_pdf_gaussian(
    y= torch.tensor(concat_vm_va),
    model=nice_model,
    device=device,
    assume_det_is_log=False
)
print("log_p_y :", log_p_y[:2])
print("p_y density:", p_y[:2])

# Use PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
v_reduced = pca.fit_transform(concat_vm_va)

# Define the network for plotting
v_x_min, v_x_max = v_reduced[:, 0].min(), v_reduced[:, 0].max()
v_y_min, v_y_max = v_reduced[:, 1].min(), v_reduced[:, 1].max()
v_x_range = v_x_max - v_x_min
v_y_range = v_y_max - v_y_min
n_bins = 200


# ----------------Plot pdf and cdf in 3D
# Emperical PDF
# Small padding so edge points are included nicely
pad_x = 0.02 * (v_x_max - v_x_min + 1e-12)
pad_y = 0.02 * (v_y_max - v_y_min + 1e-12)

x_edges = np.linspace(v_x_min - pad_x, v_x_max + pad_x, n_bins + 1)
y_edges = np.linspace(v_y_min - pad_y, v_y_max + pad_y, n_bins + 1)
dx = np.diff(x_edges)[0]
dy = np.diff(y_edges)[0]

# 2D histogram -> empirical counts
H, _, _ = np.histogram2d(
    v_reduced[:, 0],  # v_x
    v_reduced[:, 1],  # v_y
    bins=[x_edges, y_edges],
    density=False
)

# Empirical PDF: normalize so that sum(pdf * dx * dy) = 1
pdf = H / (H.sum() * dx * dy + 1e-12)

# Empirical CDF: cumulative integral up to (x, y)
# Do cumulative sums along both axes, then multiply by bin area
cdf = np.cumsum(np.cumsum(pdf, axis=0), axis=1) * dx * dy
# Numerical guard (last element should be ~1)
cdf = np.minimum(cdf, 1.0)

# Bin centers for plotting
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')  # shapes match pdf/cdf

# --- Plot PDF and CDF in 3D ---
fig = plt.figure(figsize=(14, 6))

# PDF surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(Xc, Yc, pdf, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax1.set_title('Empirical PDF (2D histogram)')
ax1.set_xlabel('v_x')
ax1.set_ylabel('v_y')
ax1.set_zlabel('density')

# CDF surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(Xc, Yc, cdf, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax2.set_title('Empirical CDF')
ax2.set_xlabel('v_x')
ax2.set_ylabel('v_y')
ax2.set_zlabel('cdf')

plt.tight_layout()
plt.savefig("figures/empirical_pdf_cdf_2d_histogram.png")


# ---- Put model predictions on the SAME grid as the empirical histogram ----
# points: (N,2), density: (N,)
points = v_reduced  # already computed above (PCA reduced)
density = p_y.cpu().numpy()  # predicted pdf at each sample point

# Bin indices for each sample (clamped to valid range)
ix = np.digitize(points[:, 0], x_edges) - 1
iy = np.digitize(points[:, 1], y_edges) - 1
ix = np.clip(ix, 0, len(x_edges) - 2)  # because edges are len-1 longer than cells
iy = np.clip(iy, 0, len(y_edges) - 2)

# Accumulate sum of predicted densities and counts per cell
pred_sum = np.zeros_like(H, dtype=float)     # same shape as empirical H
pred_cnt = np.zeros_like(H, dtype=float)

np.add.at(pred_sum, (ix, iy), density)
np.add.at(pred_cnt, (ix, iy), 1.0)

# Cell-wise average predicted density at sample locations (avoid divide-by-zero)
pred_avg = np.zeros_like(pred_sum)
mask = pred_cnt > 0
pred_avg[mask] = pred_sum[mask] / pred_cnt[mask]

# Renormalize to a proper PDF on the SAME grid so ∑ pdf * dx * dy = 1
pdf_pred = pred_avg / (np.sum(pred_avg) * dx * dy + 1e-12)

# CDF from predicted PDF on the SAME grid
cdf_pred = np.cumsum(np.cumsum(pdf_pred, axis=0), axis=1) * dx * dy
cdf_pred = np.minimum(cdf_pred, 1.0)

# ---- Plot predicted PDF/CDF on the SAME Xc, Yc as empirical ----
fig2 = plt.figure(figsize=(14, 6))

ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Xc, Yc, pdf_pred, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax1.set_title('Predicted PDF (on empirical grid)')
ax1.set_xlabel('v_x'); ax1.set_ylabel('v_y'); ax1.set_zlabel('density')

ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Xc, Yc, cdf_pred, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax2.set_title('Predicted CDF (on empirical grid)')
ax2.set_xlabel('v_x'); ax2.set_ylabel('v_y'); ax2.set_zlabel('cdf')

plt.tight_layout()
plt.savefig("figures/predicted_pdf_cdf_on_empirical_grid.png")

# (Optional) quick sanity checks
print("Empirical mass:", np.sum(pdf) * dx * dy)
print("Predicted mass:", np.sum(pdf_pred) * dx * dy)
print("Empirical CDF[-1,-1]:", cdf[-1, -1], " Pred CDF[-1,-1]:", cdf_pred[-1, -1])
