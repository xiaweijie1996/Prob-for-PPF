import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb
import pickle 

from src.models.cnice.cnicemodel import CNicemModel
from src.powersystems.randomsys import randomsystem, magnitude_transform, angle_transform
from src.utility.scalers import fit_powerflow_scalers

def main(): 
    # Configureation
    # -----------------------
    num_nodes = 4
    num_children = 3
    power_factor = 0.2
    
    split_ratio = 0.5
    n_blocks = 3
    hiddemen_dim = 64
    c_dim = (num_nodes - 1) * 2
    n_layers = 4
    input_dim = 2  # Assuming each node has a real and imaginary part
    hiddemen_dim_condition = 32
    output_dim_condition = 1
    n_layers_condition = 2
    
    batch_size = 1000
    epochs = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'src/training/cnice/savedmodel'
    # -----------------------
    
    # Initialize the random system
    random_sys = randomsystem(num_nodes=num_nodes, num_children=num_children)
    mean_vector = random_sys.network_rnd.bus_info['PD']
    mean_vector = np.array(mean_vector)
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
    
    # Define the optimizer
    optimizer = torch.optim.Adam(nice_model.parameters(), lr=0.001)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Define the scalers
    _active_power = np.random.normal(mean_vector[1:], scale=5, size=(5000, num_nodes-1))  # Power in kW
    _reactive_power = _active_power * power_factor
    _solution = random_sys.run(active_power=_active_power, 
                                reactive_power=_reactive_power, 
                                )
    _voltage_magnitudes = magnitude_transform(_solution['v'])
    _voltage_angles = angle_transform(_solution['v'])
    
    scaler_p, scaler_q, scaler_vm, scaler_va = fit_powerflow_scalers(
        active_power=_active_power,
        reactive_power=_reactive_power,
        voltage_magnitudes=_voltage_magnitudes,
        voltage_angles=_voltage_angles
    )
    
    # Save the scalers as pickle files
    scalers = {
        "scaler_p": scaler_p,
        "scaler_q": scaler_q,
        "scaler_vm": scaler_vm,
        "scaler_va": scaler_va
    }
    with open(os.path.join(save_path, f"scalers_{num_nodes}.pkl"), 'wb') as f:
        pickle.dump(scalers, f)
    
    # Initialize Weights and Biases
    # wb.init(project=f"cNICE-PowerFlow-node-{num_nodes}")
    
    # Load already trained model if exists
    model_path = os.path.join(save_path, f"cnicemodel_{num_nodes}.pth")
    if os.path.exists(model_path):
        nice_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    end_loss = 1e6
    for _ in range(epochs):
        
        #-------input and target power flow data preparation-------
        # Generate random active and reactive power inputs
        active_power = np.random.normal(mean_vector[1:], scale=5, size=(batch_size, num_nodes-1))
        reactive_power = active_power * power_factor # np.random.uniform(0.1, 0.3, size=(batch_size, num_nodes-1))  # Random power factor between 0.1 and 0.3
        
        # Run the power flow analysis
        solution = random_sys.run(active_power=active_power, 
                                reactive_power=reactive_power)

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
        
        
        #-------input and target power flow data preparation-------
        p_index = torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
        # q_index = np.arange(0, num_nodes-1)
        v_index = p_index

        input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
        input_c = input_power.clone()
        
        output_y = torch.cat((target_voltage[:, v_index].unsqueeze(1),
                              target_voltage[:, v_index+num_nodes-1].unsqueeze(1)
                            ), dim=1)
        print(f"Input shape: {input_x.shape}, Condition shape: {input_c.shape}, Output shape: {output_y.shape}")
        
        
        # ------- training -------
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the NICE model
        output_voltage, _ja = nice_model.forward(input_x, input_c, index_p=p_index, index_v=v_index)
        
        # Backward pass to get the output power
        output_power, _j = nice_model.inverse(output_y, input_c, index_p=p_index, index_v=v_index)
        
        # Compute the loss
        loss_backward = loss_function(output_power, input_x)
        
        # Compute the loss
        loss_forward = loss_function(output_voltage, output_y)
        
        # Loss
        loss = loss_forward + loss_backward
    
        # Percentage error
        loss_mangitude = loss_function(output_voltage[:, 0], output_y[:, 0])
        loss_angle = loss_function(output_voltage[:, 1], output_y[:, 1])
    
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {_+1}, Loss Forward: {loss_forward.item():.6f}, Loss Backward: {loss_backward.item():.6f}, Jacobean: {_ja.mean().item():.6f}, Percentage Error Magnitude: {loss_mangitude.item():.6f}, Percentage Error Angle: {loss_angle.item():.6f}")
        
        # ----------Log to Weights and Biases
        wb.log({
            "loss_forward": loss_forward.item(),
            "loss_backward": loss_backward.item(),
            "jacobian": _ja.mean().item(),
            "epoch": _+1,
            "percentage_error_magnitude": loss_mangitude.item(),
            "percentage_error_angle": loss_angle.item()
        })
        
        # Save the model every 100 epochs
        if (_ + 1) >200 and end_loss > loss_forward.item():
            end_loss = loss_forward.item()
            torch.save(nice_model.state_dict(), os.path.join(save_path, f"cnicemodel_{num_nodes}.pth"))
            print(f"saved at epoch {_+1} with loss {end_loss}")
    

if __name__ == "__main__":
    main()