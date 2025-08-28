import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.models.cnice.cnicemodel import CNicemModel


# -----------------------
# Configureation
# -----------------------
num_nodes = 4
std = 10

split_ratio = 0.5
n_blocks = 3
hiddemen_dim = 24
c_dim = (num_nodes - 1) * 2
n_layers = 3
input_dim = 2  # Assuming each node has a real and imaginary part
hiddemen_dim_condition = 24
output_dim_condition = 1
n_layers_condition = 2

batch_size = 100
epochs = 100000000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'src/training/cnice/savedmodel'

# -----------------------
# Initialize the random system　model and scalers
# -----------------------
# Initialize the random system
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
# Define the data
# -----------------------
active_power = np.random.normal(0, scale=std, size=(batch_size, num_nodes-1))
reactive_power = np.random.normal(0, scale=std, size=(batch_size, num_nodes-1)) * np.random.uniform(0.01, 0.5, size=(batch_size, num_nodes-1))  # Random power factor between 0.1 and 0.3
input_power = torch.tensor(np.hstack((active_power, reactive_power)), dtype=torch.float32).to(device)

p_index = torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
v_index = p_index

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
print(f"output_y:{output_y[:3]}, pre_v:{pre_v[:3]}")

# Plot the pre_v and pre_p and real and target
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pre_v[:, 0], pre_v[:, 1], label='Predicted Voltage', alpha=0.3)
plt.scatter(output_y[:, 0].cpu().numpy(), output_y[:, 1].cpu().numpy(), label='Target Voltage', alpha=0.3)
plt.title('Predicted vs Target Voltage Magnitudes and Angles')
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(pre_p[:, 0], pre_p[:, 1], label='Predicted Power', alpha=0.3)
plt.scatter(input_x[:, 0].cpu().numpy(), input_x[:, 1].cpu().numpy(), label='Target Power', alpha=0.3)
plt.title('Predicted vs Target Active and Reactive Power')
plt.xlabel('Predicted Value')
plt.ylabel('Target Value')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/cnice_toolcase_{num_nodes}_accuracy1.png', dpi=300)
