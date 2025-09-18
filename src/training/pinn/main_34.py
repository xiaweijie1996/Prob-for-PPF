import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import pandas as pd
import numpy as np
import wandb as wb
import pickle 
import yaml
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from src.models.pinn.pinn import PinnEncoder as pinnmodel
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.node34 import Node34Example

# Set all torch floats to double precision
torch.set_default_dtype(torch.float64)

def main(): 
    # Configureation
    # -----------------------
    with open('src/config34node.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
    scaler_path = config['SystemAndDistribution']['scaler_path']  # Path to save/load the scalers
    dis_path = config['SystemAndDistribution']['dis_path']  # Path to the distribution system file
    
    num_blocks = config['Pinn']['num_blocks']
    hidden_channel = config['Pinn']['hidden_channel']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = config['Pinn']['save_path']
    lr = config['Pinn']['lr']
    batch_size = config['Pinn']['batch_size']
    epochs = config['Pinn']['epochs']
    
    # -----------------------
    # Edge index for a fully connected graph excluding self-loops
    system_file = 'src/powersystems/files/Lines_34.csv'
    edge_index = pd.read_csv(system_file, header=None)
    edge_index = edge_index.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna().values.astype(int)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long).to(device) - 1  # Convert to zero-based index 
    
    # cancel column with 0
    edge_index = edge_index[:, edge_index[0, :] != 0] # [1, 33]
    edge_index = edge_index[:, edge_index[1, :] != 0] # [1, 33]
    edge_index -= 1  # Convert to zero-based index [0, 32]
    
    # Initialize the random system
    random_sys = Node34Example()

    # Initialize the NICE model
    pinn_model = pinnmodel(
        # Graph parameters
        num_nodes=num_nodes-1,
        edge_index=edge_index,
        num_block=num_blocks,
        hidden_channel=hidden_channel,
    ).to(device)
    # pinn_model = pinn_model.float()
    print(f"Model Parameters: {sum(p.numel() for p in pinn_model.parameters() if p.requires_grad)}")
    
    # Load GMM and Scalers
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
    optimizer = torch.optim.AdamW(pinn_model.parameters(), lr=lr)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Initialize Weights and Biases
    wb.init(project=f"pinn-{num_nodes}")
    
    # Log Model size
    wb.log({"Model Parameters": sum(p.numel() for p in pinn_model.parameters() if p.requires_grad)})
    
    # Load already trained model if exists
    model_path = os.path.join(save_path, f"pinn_{num_nodes}.pth")
    if os.path.exists(model_path):
        pinn_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    save_loss = 1e6
    for _ in range(epochs):
        pinn_model.train()
        
        #-------input and target power flow data preparation-------
        # Generate random active and reactive power inputs
        power_sample = gmm.sample(batch_size)[0]
        active_power = power_sample[:, :num_nodes-1]
        reactive_power = power_sample[:, num_nodes-1:]
   
        # Run the power flow analysis
        solution = random_sys.run(active_power=active_power, 
                                    reactive_power=reactive_power)
        
        # Transform the voltage magnitudes
        voltage_magnitudes = solution['v'].real
        voltage_angles = solution['v'].imag
        
        voltages = np.vstack((voltage_magnitudes, voltage_angles))
        
        input_power = torch.tensor(np.vstack((active_power, reactive_power))).to(device) # [B, 2*(num_nodes-1)]
        target_voltage = torch.tensor(voltages).to(device) # [B, 2*(num_nodes-1)]
        
        input_power = input_power.reshape(batch_size, num_nodes-1, 2) # [B, N-1, 2]
        target_voltage = target_voltage.reshape(batch_size, num_nodes-1, 2) # [B, N-1, 2]
        # print(f"Input power shape: {input_power.shape}, Target voltage shape: {target_voltage.shape}")

        # ------- training -------
        # Zero the gradients
        optimizer.zero_grad()
        
        # Voltage to Power
        output_power_real, output_power_img = pinn_model(target_voltage[:,:,0], target_voltage[:,:,1]) # [B, N-1, 2]
        output_power = torch.cat((output_power_real.unsqueeze(2), output_power_img.unsqueeze(2)), dim=2) # [B, N-1, 2]
        
        # Compute the loss
        loss = loss_function(output_power, input_power)
        
        # Add weight clipping to avoid NaN
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=0.3)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Scaled loss
        output_power_scaled = output_power.cpu().detach().numpy()
        input_power_scaled = input_power.cpu().detach().numpy()
        output_power_scaled[:, :, 0] = (output_power_scaled[:, :, 0] - scaler['mean_active_power']) / scaler['std_active_power']
        output_power_scaled[:, :, 1] = (output_power_scaled[:, :, 1] - scaler['mean_reactive_power']) / scaler['std_reactive_power']
        input_power_scaled[:, :, 0] = (input_power_scaled[:, :, 0] - scaler['mean_active_power']) / scaler['std_active_power']
        input_power_scaled[:, :, 1] = (input_power_scaled[:, :, 1] - scaler['mean_reactive_power']) / scaler['std_reactive_power']
        scaled_loss = loss_function(torch.tensor(output_power_scaled), torch.tensor(input_power_scaled)).item()

        print(f"Epoch {_+1}, Loss Forward: {loss.item():.6f}, Scaled Loss: {end_loss:.6f}")
         
        # ----------Log to Weights and Biases
        wb.log({
            "loss_vtp_unscaled": loss.item(),
            "epoch": _+1,
            "loss_vtp": scaled_loss,
        
        })
        
        # Save the model every 100 epochs
        if (_ + 1) > 1000 and save_loss > loss.item():
            save_loss = loss.item()
            torch.save(pinn_model.state_dict(), os.path.join(save_path, f"pinnmodel_{num_nodes}.pth"))
            print(f"saved at epoch {_+1} with loss {save_loss}")
            
            # Plot the output vs target for power and voltage for the current p_index
            pre_power = output_power.cpu().detach().numpy()
            true_power = input_power.cpu().detach().numpy()
            
            # fig.tight_layout()
            # fig.savefig(os.path.join(save_path, f"pinn_gen.png"))
            plt.figure(figsize=(12, 5))
            plt.scatter(pre_power[:, 0, 0], pre_power[:, 0, 1], alpha=0.1)
            plt.scatter(true_power[:, 0, 0], true_power[:, 0, 1], alpha=0.1)
            plt.xlabel('True Power P ')
            plt.ylabel('Predicted Power P ')
            plt.title('Predicted vs True Power P')
            plt.legend(['Predicted Power P', 'True Power P'])
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"pinn_power.png"))
            plt.close()
           
           
if __name__ == "__main__":
    main()