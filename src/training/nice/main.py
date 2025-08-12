import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb

from src.models.nice.nicemodel import NicemModel
from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform


def main(): 
    num_nodes = 34
    num_children = 3
    power_factor = 0.2
    
    split_ratio = 0.6
    n_blocks = 4
    hiddemen_dim = 128
    n_layers = 2
    full_dim = (num_nodes -1) * 2  # Assuming each node has a real and imaginary part
    
    batch_size = 500
    epochs = 2000
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
    
    # Define the optimizer
    optimizer = torch.optim.Adam(nice_model.parameters(), lr=0.001)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Initialize Weights and Biases
    wb.init(project="NICE-PowerFlow")
    
    for _ in range(epochs):
        # Generate random active and reactive power inputs
        active_power = torch.randn((batch_size, num_nodes-1)) *  5 + 50
        reactive_power = active_power * power_factor

        # Run the power flow analysis
        solution = random_sys.run(active_power=active_power.numpy(), 
                                reactive_power=reactive_power.numpy(), 
                                plot_graph=False)

        # Transform the voltage magnitudes
        voltage_magnitudes = magnitude_transform(solution['v'])
        voltage_angles = angle_transform(solution['v'])
        
        voltages = np.hstack((voltage_magnitudes, voltage_angles))
        
        # Convert to torch tensor
        input_power = torch.concat((active_power, reactive_power), dim=1).to(device)
        target_voltage = torch.tensor(voltages, dtype=torch.float32).to(device)
        print(f"Input Power Shape: {input_power.shape}, Target Voltage Shape: {target_voltage.shape}")
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the NICE model
        output_voltage, _ja = nice_model.forward(input_power)
        
        # Compute the loss
        loss = loss_function(output_voltage, target_voltage)
        
        # Percentage error
        percentage_error = torch.mean(torch.abs((output_voltage - target_voltage) / target_voltage)) * 100
        
        print(f"Loss: {loss.item()}, epoch: {_+1}/{epochs}, jacobian: {_ja.mean().item()}, percentage error: {percentage_error.item()}%")
        
        # Log to Weights and Biases
        wb.log({
            "loss": loss.item(),
            "jacobian": _ja.mean().item(),
            "percentage_error": percentage_error.item(),
            "epoch": _+1
        })
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        

if __name__ == "__main__":
    main()