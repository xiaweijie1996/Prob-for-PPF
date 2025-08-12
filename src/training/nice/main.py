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
from src.utility.scalers import fit_powerflow_scalers

def main(): 
    # Configureation
    # -----------------------
    num_nodes = 34
    num_children = 3
    power_factor = 0.2
    
    split_ratio = 0.6
    n_blocks = 3
    hiddemen_dim = 256
    n_layers = 2
    full_dim = (num_nodes -1) * 2  # Assuming each node has a real and imaginary part
    
    batch_size = 1000
    epochs = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------
    
    
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
    
    # Define the scalers
    _active_power = np.random.normal(50, scale=5, size=(1000, num_nodes-1))  # Power in kW
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
    
    # Initialize Weights and Biases
    wb.init(project=f"NICE-PowerFlow-node-{num_nodes}")
    
    end_loss = 10000
    for _ in range(epochs):
        # Generate random active and reactive power inputs
        active_power = np.random.normal(50, scale=5, size=(batch_size, num_nodes-1))
        reactive_power = active_power * power_factor

        # Run the power flow analysis
        solution = random_sys.run(active_power=active_power, 
                                reactive_power=reactive_power, 
                                plot_graph=False)

        # Transform the voltage magnitudes
        voltage_magnitudes = magnitude_transform(solution['v'])
        voltage_angles = angle_transform(solution['v'])
        
        scaled_voltage_magnitudes = scaler_vm.transform(voltage_magnitudes)
        scaled_voltage_angles = scaler_va.transform(voltage_angles)
    
        voltages = np.hstack((scaled_voltage_magnitudes, scaled_voltage_angles))
        
        # Convert to torch tensor
        active_power = scaler_p.transform(active_power)
        reactive_power = scaler_q.transform(reactive_power)
        input_power = torch.tensor(np.hstack((active_power, reactive_power)), dtype=torch.float32).to(device)
        target_voltage = torch.tensor(voltages, dtype=torch.float32).to(device)
        print(f"Input Power Shape: {input_power.shape}, Target Voltage Shape: {target_voltage.shape}")
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the NICE model
        output_voltage, _ja = nice_model.forward(input_power)
        
        # Compute the loss
        loss = loss_function(output_voltage, target_voltage)
    
        # Percentage error
        output_voltage = output_voltage.cpu().detach().numpy()
        scaled_output_mag = output_voltage[:, :num_nodes-1]
        scaled_output_mag = scaler_vm.inverse_transform(scaled_output_mag)
        scaled_output_angle = output_voltage[:, num_nodes-1:]
        scaled_output_angle = scaler_va.inverse_transform(scaled_output_angle)
        
        percentage_mag = np.mean(np.abs((scaled_output_mag - voltage_magnitudes) / voltage_magnitudes)) * 100
        percentage_angle = np.mean(np.abs((scaled_output_angle - voltage_angles) / voltage_angles)) * 100
       
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}, epoch: {_+1}/{epochs}, jacobian: {_ja.mean().item()}, "
              f"percentage error magnitude: {percentage_mag.item()}%, "
              f"percentage error angle: {percentage_angle.item()}%")
        
        # Log to Weights and Biases
        wb.log({
            "loss": loss.item(),
            "jacobian": _ja.mean().item(),
            "epoch": _+1
        })

        
        # Save the model every 100 epochs
        if (_ + 1) >200 and end_loss > loss.item():
            end_loss = loss.item()
            torch.save(nice_model.state_dict(), f"src/training/nice/savedmodel/nicemodel_{num_nodes}.pth")
            print(f"saved at epoch {_+1} with loss {end_loss}")
    

if __name__ == "__main__":
    main()