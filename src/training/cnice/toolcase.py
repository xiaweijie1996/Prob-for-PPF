import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb

from src.models.cnice.cnicemodel import CNicemModel


def main(): 
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
    
    batch_size = 1000
    epochs = 100000000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = 'src/training/cnice/savedmodel'
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
        torch.save(random_sys.state_dict(), os.path.join(save_path, f"toolcase_target_{num_nodes}.pth"))
        print(f"Saved toolcase model at {os.path.join(save_path, f'toolcase_target_{num_nodes}.pth')}")
    
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
    
    # Initialize Weights and Biases
    wb.init(project=f"ToolCase-node-{num_nodes}")
    
    # Log Model size
    wb.config.update({
        "model_parameters": sum(p.numel() for p in nice_model.parameters() if p.requires_grad)
    })
    
    # Load already trained model if exists
    model_path = os.path.join(save_path, f"toolcase_{num_nodes}.pth")
    if os.path.exists(model_path):
        nice_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    end_loss = 1e6
    for _ in range(epochs):
        
        #-------input and target power flow data preparation-------
        # Generate random active and reactive power inputs
        active_power = np.random.normal(0, scale=std, size=(batch_size, num_nodes-1))
        reactive_power = np.random.normal(0, scale=std, size=(batch_size, num_nodes-1)) * np.random.uniform(0.01, 0.5, size=(batch_size, num_nodes-1))  # Random power factor between 0.1 and 0.3
        input_power = torch.tensor(np.hstack((active_power, reactive_power)), dtype=torch.float32).to(device)
        
        #-------input and target power flow data preparation-------
        p_index = torch.randint(0, num_nodes-1, (1,)).item()  # Random index for the power input
        v_index = p_index

        input_x = torch.cat((input_power[:, p_index].unsqueeze(1), input_power[:, p_index+num_nodes-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
        input_c = input_power.clone()
        
        with torch.no_grad():
            output_y = random_sys.forward(input_x, input_c, index_p=p_index, index_v=v_index)[0].detach()
        
        # print(f"Input shape: {input_x.shape}, Condition shape: {input_c.shape}, Output shape: {output_y.shape}")
        

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
            torch.save(nice_model.state_dict(), os.path.join(save_path, f"toolcase_{num_nodes}.pth"))
            print(f"saved at epoch {_+1} with loss {end_loss}")
    

if __name__ == "__main__":
    main()