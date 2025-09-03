import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb
import pickle 
import yaml
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from src.models.cnice.cnicemodel import CNicemModel
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.node34 import Node34Example
from src.utility.scalers import fit_powerflow_scalers

def main(): 
    # Configureation
    # -----------------------
    with open('src/config34node.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
    dis_path = config['SystemAndDistribution']['dis_path']  # Path to the distribution system file
    scaler_path = config['SystemAndDistribution']['scaler_path']  # Path to save/load the scalers
    
    split_ratio = config['CNice']['split_ratio']
    n_blocks = config['CNice']['n_blocks']
    hiddemen_dim = config['CNice']['hiddemen_dim']
    c_dim = (2 * (num_nodes - 1))  # Condition dimension (P and Q for all nodes except slack)
    n_layers = config['CNice']['n_layers']
    input_dim = config['CNice']['input_dim']  # Input dimension (P and Q for one node)
    hiddemen_dim_condition = config['CNice']['hiddemen_dim_condition']
    output_dim_condition = config['CNice']['output_dim_condition']
    n_layers_condition = config['CNice']['n_layers_condition']
    
    batch_size = config['CNice']['batch_size']
    epochs = config['CNice']['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = config['CNice']['save_path']
    lr = config['CNice']['lr']
    # -----------------------
    
    # Initialize the random system
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
    
    # Load GMM and Scalers
    # gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    with open(dis_path, 'rb') as f:
        gmm = pickle.load(f)  
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded scalers from: {scaler_path}")
        
    # Load 100 samples, print mean and std
    _samples = gmm.sample(100)[0]
    print(f"Sampled 100 data from GMM, mean: {_samples.mean()}, std: {_samples.std()}, shape: {_samples.shape}")
    print("Loaded GMM from:", dis_path)
    
    # Define the optimizer
    optimizer = torch.optim.Adam(nice_model.parameters(), lr=lr)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Initialize Weights and Biases
    wb.init(project=f"cNICE-node-{num_nodes}")
    
    # Log Model size
    wb.log({"Model Parameters": sum(p.numel() for p in nice_model.parameters() if p.requires_grad)})
    
    # Load already trained model if exists
    model_path = os.path.join(save_path, f"cnicemodel_{num_nodes}.pth")
    if os.path.exists(model_path):
        nice_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    end_loss = 1e6
    for _ in range(epochs):
        
        #-------input and target power flow data preparation-------
        # Generate random active and reactive power inputs
        power_sample = gmm.sample(batch_size)[0]
        active_power = power_sample[:, :num_nodes-1]
        reactive_power = power_sample[:, num_nodes-1:]
     
        # Run the power flow analysis
        solution = random_sys.run(active_power=active_power, 
                                    reactive_power=reactive_power)
        
        # Transform the voltage magnitudes
        voltage_magnitudes = magnitude_transform(solution['v'])
        voltage_angles = angle_transform(solution['v'])
        
        voltage_magnitudes = (voltage_magnitudes - scaler['mean_voltage_magnitude']) / scaler['std_voltage_magnitude']
        voltage_angles = (voltage_angles - scaler['mean_voltage_angle']) / scaler['std_voltage_angle']
        
        voltages = np.hstack((voltage_magnitudes, voltage_angles))
        
        # Convert to torch tensor
        active_power = (active_power - scaler['mean_active_power']) / scaler['std_active_power']
        reactive_power = (reactive_power - scaler['mean_reactive_power']) / scaler['std_reactive_power']
        
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
    
        # Error
        with torch.no_grad():
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
        if (_ + 1) >100 and end_loss > loss_forward.item():
            end_loss = loss_forward.item()
            torch.save(nice_model.state_dict(), os.path.join(save_path, f"cnicemodel_{num_nodes}.pth"))
            print(f"saved at epoch {_+1} with loss {end_loss}")
            
            # Plot the output vs target for power and voltage for the current p_index
            pre_power = output_power.cpu().detach().numpy()
            true_power = input_x.cpu().detach().numpy()
            pre_voltage = output_voltage.cpu().detach().numpy()
            true_voltage = output_y.cpu().detach().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(true_power[:, 0], true_power[:, 1], label='True Power', alpha=0.1)
            axes[0].scatter(pre_power[:, 0], pre_power[:, 1], label='Predicted Power', alpha=0.1)
            axes[0].set_title(f'Active vs Reactive Power at Node {p_index+1}')
            axes[0].set_xlabel('Active Power (P)')
            axes[0].set_ylabel('Reactive Power (Q)')
            axes[0].legend()
            axes[0].axis('equal')

            axes[1].scatter(true_voltage[:, 0], true_voltage[:, 1], label='True Voltage', alpha=0.1)
            axes[1].scatter(pre_voltage[:, 0], pre_voltage[:, 1], label='Predicted Voltage', alpha=0.1)
            axes[1].set_title(f'Voltage Magnitude vs Angle at Node {v_index+1}')
            axes[1].set_xlabel('Voltage Magnitude (|V|)')
            axes[1].set_ylabel('Voltage Angle (θ)')
            axes[1].legend()
            axes[1].axis('equal')

            fig.tight_layout()
            fig.savefig('src/training/cnice/savedmodel/cnice_gen.png')

            # ✅ log the figure object, not `plt`
            wb.log({"cnice_gen": fig})

            plt.close(fig)   # close after logging

           
           
if __name__ == "__main__":
    main()