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

from src.models.gnn.gnn import GnnModel
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.node34 import Node34Example

# Set all torch floats to double precision
torch.set_default_dtype(torch.float32)

def main(): 
    # Configureation
    # -----------------------
    with open('src/config34node.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
    scaler_path = config['SystemAndDistribution']['scaler_path']  # Path to save/load the scalers
    dis_path = config['SystemAndDistribution']['dis_path']  # Path to the distribution system file
    
    c_dim = (2 * (num_nodes - 1))  # Condition dimension (P and Q for all nodes except slack)
    hiddemen_dim = config['Gnn']['hiddemen_dim']
    n_layers = config['Gnn']['n_layers']
    
    batch_size = config['Gnn']['batch_size']
    epochs = config['Gnn']['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = config['Gnn']['save_path']
    lr = config['Gnn']['lr']
    # -----------------------
    # Initialize the random system
    
    random_sys = Node34Example()

    # Initialize the NICE model
    gnn_model = GnnModel(
        in_dim=c_dim,
        hidden_dim=hiddemen_dim,
        out_dim=c_dim,
        num_layers=n_layers
    ).to(device)
    # gnn_model.double()
    print(f"Model Parameters: {sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)}")
    
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
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=lr)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Initialize Weights and Biases
    wb.init(project=f"Gnn-node-{num_nodes}")
    
    # Log Model size
    wb.log({"Model Parameters": sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)})
    
    # Edge index for a fully connected graph excluding self-loops
    system_file = 'src/powersystems/files/Lines_34.csv'
    edge_index = pd.read_csv(system_file, header=None)
    edge_index = edge_index.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna().values.astype(int)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long).to(device) - 1  # Convert to zero-based index
    
    # Load already trained model if exists
    # model_path = os.path.join(save_path, f"Gnnmodel_{num_nodes}.pth")
    # if os.path.exists(model_path):
    #     gnn_model.load_state_dict(torch.load(model_path))
    #     print(f"Loaded model from {model_path}")
    
    end_loss = 1e6
    for _ in range(epochs):
        gnn_model.train()
        
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

        # ------- training -------
        # Zero the gradients
        optimizer.zero_grad()
        
        # Voltage to Power
        output_power = gnn_model(target_voltage, edge_index)
        
        # Compute the loss
        loss = loss_function(output_power, input_power)
        
        # Add weight clipping to avoid NaN
        torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=0.5)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # print(f"Epoch {_+1}, Loss Forward: {loss.item():.6f}")
        
        # ----------Log to Weights and Biases
        wb.log({
            "loss_vtp": loss.item(),
            "epoch": _+1,
        })
        
        # Save the model every 100 epochs
        if (_ + 1) > 10000 and end_loss > loss.item():
            end_loss = loss.item()
            torch.save(gnn_model.state_dict(), os.path.join(save_path, f"Gnnmodel_{num_nodes}.pth"))
            print(f"saved at epoch {_+1} with loss {end_loss}")
            
            # Plot the output vs target for power and voltage for the current p_index
            pre_power = output_power.cpu().detach().numpy()
            true_power = input_power.cpu().detach().numpy()
            
            # fig.tight_layout()
            # fig.savefig(os.path.join(save_path, f"Gnn_gen.png"))
            plt.figure(figsize=(12, 5))
            plt.scatter(pre_power[:, 0], pre_power[:, num_nodes-1], alpha=0.1)
            plt.scatter(true_power[:, 0], true_power[:, num_nodes-1], alpha=0.1)
            plt.xlabel('True Power P ')
            plt.ylabel('Predicted Power P ')
            plt.title('Predicted vs True Power P')
            plt.legend(['Predicted Power P', 'True Power P'])
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"Gnn_power.png"))
            plt.close()
           
           
if __name__ == "__main__":
    main()