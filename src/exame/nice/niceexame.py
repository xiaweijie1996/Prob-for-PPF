import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
from sklearn.mixture import GaussianMixture

from src.models.nice.nicemodel import NicemModel
from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers
from src.utility.inversepdf import inverse_pdf_gaussian

num_nodes = 34
num_children = 3
power_factor = 0.2

split_ratio = 0.6
n_blocks = 4
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
scaler_p, scaler_q, scaler_vm, scaler_va = fit_powerflow_scalers(
    active_power=_active_power,
    reactive_power=_reactive_power,
    voltage_magnitudes=_voltage_magnitudes,
    voltage_angles=_voltage_angles
)
scaled_vm = scaler_vm.transform(_voltage_magnitudes)
scaled_va = scaler_va.transform(_voltage_angles)
concat_vm_va = np.concatenate((scaled_vm, scaled_va), axis=1)
concat_vm_va = torch.tensor(concat_vm_va, device=device, dtype=torch.float32)

# # Load the nice model
# nice_model.load_state_dict(torch.load(f"src/training/nice/savedmodel/nicemodel_epoch_34.pth"))
# nice_model.eval()

log_p_y, p_y, x = inverse_pdf_gaussian(
    y=concat_vm_va[: 10],
    mean_x=torch.zeros(full_dim, device=device),
    cov_matr=torch.eye(full_dim, device=device),
    model=nice_model,
    device=device,
    assume_det_is_log=False
)
print(f"p_Y(y): {p_y}")
print(p_y.sum().max()) 
print(f"log p_Y(y): {log_p_y}")


# print(concat_vm_va[:10].cpu().numpy())

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(concat_vm_va.cpu().numpy())

# Evaluate the density of the GMM of concat_vm_va[: 10]
gmm_density = gmm.score_samples(concat_vm_va[:10].cpu().numpy())
print(f"GMM Density: {gmm_density}")