import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import numpy as np
import wandb as wb
import yaml
import matplotlib.pyplot as plt

from src.models.mixedflow.mixedflow import CMixedModel
from src.powersystems.randomsys import  magnitude_transform, angle_transform
from src.powersystems.ieee39 import Case39PF

# Set all torch floats to double precision
torch.set_default_dtype(torch.float64)

def main(): 
    # Configureation
    # -----------------------
    with open('src/config39bus.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    num_nodes = config['SystemAndDistribution']['node']  # Total number of nodes including slack
    load_bus_num = config['SystemAndDistribution']['load_bus_num']  # Number of load buses
    load_variance_rate = config['SystemAndDistribution']['load_variance_rate']
    
    split_ratio = config['MixedSplineFCP']['split_ratio']
    n_blocks_spline = config['MixedSplineFCP']['n_blocks_spline']
    n_blocks_fcp = config['MixedSplineFCP']['n_blocks_fcp']
    hiddemen_dim = config['MixedSplineFCP']['hiddemen_dim']
    c_dim = (2 * load_bus_num)  # Condition dimension (P and Q for all nodes except slack)
    n_layers = config['MixedSplineFCP']['n_layers']
    input_dim = config['MixedSplineFCP']['input_dim']  # Input dimension (P and Q for one node)
    hiddemen_dim_condition = config['MixedSplineFCP']['hiddemen_dim_condition']
    n_layers_condition = config['MixedSplineFCP']['n_layers_condition']
    b_interval = config['MixedSplineFCP']['b_interval']
    k_bins = config['MixedSplineFCP']['k_bins']
    
    load_variance_rate = config['SystemAndDistribution']['load_variance_rate']
    batch_size = config['MixedSplineFCP']['batch_size']
    epochs = config['MixedSplineFCP']['epochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = config['MixedSplineFCP']['save_path']
    lr = config['MixedSplineFCP']['lr']
    forward_loss_ratio = config['MixedSplineFCP'].get('forward_loss_ratio', 1.0)  # Default to 1.0 if not specified
    # -----------------------
    
    # Initialize the random system
    random_sys = Case39PF()
    random_sys.run_pf()
    load_index = random_sys.net.load['bus']
    load_bus = random_sys.net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']].loc[load_index].reset_index(drop=True)
    epsilon = 1e-6
    base_ap = load_bus['p_mw'].values + epsilon
    base_rp = load_bus['q_mvar'].values
    base_vm = load_bus['vm_pu'].values
    base_va = load_bus['va_degree'].values
    load_rato = base_rp/base_ap
    # print('load ratio:', load_rato)
    scaler = {
        'mean_active_power': base_ap,
        'mean_reactive_power': base_rp,
        'mean_voltage_magnitude': base_vm,
        'mean_voltage_angle': base_va.mean(),
        
    }
    print(f"Scaler: {scaler}")

    # Initialize the NICE model
    mix_model = CMixedModel(
        input_dim=input_dim,
        hidden_dim=hiddemen_dim,
        condition_dim=c_dim,
        n_layers=n_layers,
        split_ratio=split_ratio,
        n_blocks_fcp=n_blocks_fcp,
        n_blocks_spline=n_blocks_spline,
        hidden_dim_condition=hiddemen_dim_condition,
        n_layers_condition=n_layers_condition,
        b_interval=b_interval,
        k_bins=k_bins
        
    ).to(device)
    mix_model.double()
    print(f"Model Parameters: {sum(p.numel() for p in mix_model.parameters() if p.requires_grad)}")

    # Define the optimizer
    optimizer = torch.optim.Adam(mix_model.parameters(), lr=lr)
    
    # Define the loss function
    loss_function = torch.nn.MSELoss()
    
    # Initialize Weights and Biases
    wb.init(project=f"MixedSplineFCP-node-{num_nodes}")
    
    # Log Model size
    wb.log({"Model Parameters": sum(p.numel() for p in mix_model.parameters() if p.requires_grad)})
    
    # Load already trained model if exists
    model_path = os.path.join(save_path, f"MixedSplineFCPmodel_{num_nodes}.pth")
    if os.path.exists(model_path):
        mix_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    end_loss = 1e6
    for _ in range(epochs):
        mix_model.train()
        
        #-------input and target power flow data preparation-------
        # Generate random active and reactive power inputs
        # a coefficient follow uniform distribution between (-1, 1)
        
        input_x = torch.empty((0, 2), dtype=torch.float64).to(device)
        input_c = torch.empty((0, c_dim), dtype=torch.float64).to(device)
        output_y = torch.empty((0, 2), dtype=torch.float64).to(device)
        p_index = torch.randint(0, load_bus_num, (1,)).item()  # Random index for the power input for load bus (0 to 20)
        v_index = p_index
        for _b in range(batch_size):
            coeff_active = np.random.uniform(-1, 1, size=load_bus_num)
            _active_power = base_ap + coeff_active * load_variance_rate * base_ap
            _reactive_power = _active_power * load_rato
            _active_power = _active_power.reshape(-1)
            _reactive_power = _reactive_power.reshape(-1)
            random_sys.set_loads(_active_power, _reactive_power)
            
            # Run the power flow analysis
            solution = random_sys.run_pf()
            solution = solution.loc[load_index].reset_index(drop=True)
            # Transform the voltage magnitudes
            _voltage_magnitudes = solution['vm_pu'].values
            _voltage_angles = solution['va_degree'].values
            
            
            _voltage_magnitudes = (_voltage_magnitudes / scaler['mean_voltage_magnitude']) - 1  # per unit deviation [-1, 1]
            _voltage_angles = (_voltage_angles / scaler['mean_voltage_angle']) - 1
            
            _voltages = np.hstack((_voltage_magnitudes, _voltage_angles))
            
            # Convert to torch tensor
            _active_power = (_active_power / scaler['mean_active_power']) - 1 
            _reactive_power = (_reactive_power / scaler['mean_reactive_power']) - 1
            
            _input_power = torch.tensor(np.hstack((_active_power, _reactive_power)), dtype=torch.float64).to(device)
            _input_power = _input_power.unsqueeze(0)  # shape (batch_size, 42)
            _target_voltage = torch.tensor(_voltages, dtype=torch.float64).to(device)
            _target_voltage = _target_voltage.unsqueeze(0)  # shape (batch_size, 42)
            # print(input_power.shape, target_voltage.shape)
            
            #-------input and target power flow data preparation-------
            _input_x = torch.cat((_input_power[:, p_index].unsqueeze(1), _input_power[:, p_index+load_bus_num-1].unsqueeze(1)), dim=1)  # shape (batch_size, 2)
            _input_c = _input_power.clone()
            
            _output_y = torch.cat((_target_voltage[:, v_index].unsqueeze(1),
                                _target_voltage[:, v_index+load_bus_num-1].unsqueeze(1)
                                ), dim=1)
            # print(f"Input shape: {input_x.shape}, Condition shape: {input_c.shape}, Output shape: {output_y.shape}")
            input_x = torch.cat((input_x, _input_x), dim=0)
            input_c = torch.cat((input_c, _input_c), dim=0)
            output_y = torch.cat((output_y, _output_y), dim=0)
      
        # ------- training -------
        # Zero the gradients
        optimizer.zero_grad()
        
        # Power to Voltage
        output_voltage, _ja1 = mix_model.inverse(input_x, input_c, index_p=p_index, index_v=v_index)
        
        # Voltage to Power
        output_power, _ja = mix_model.forward(output_y, input_c, index_p=p_index, index_v=v_index)
        
        # Compute the loss
        loss_vtp = loss_function(output_power, input_x)
        
        # Compute the loss
        loss_ptv = loss_function(output_voltage, output_y)
        
        # Loss
        loss = loss_vtp * forward_loss_ratio + loss_ptv * (1 - forward_loss_ratio) # + distribution_loss
    

        # Error
        with torch.no_grad():
            loss_mangitude = loss_function(output_voltage[:, 0], output_y[:, 0])
            loss_angle = loss_function(output_voltage[:, 1], output_y[:, 1])
    
        # Backward pass and optimization
        loss.backward()
        
        # Add weight clipping to avoid NaN
        torch.nn.utils.clip_grad_norm_(mix_model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        print(f"Epoch {_+1}, Loss VTP: {loss_vtp.item():.6f}, Loss PTV: {loss_ptv.item():.6f}, Total Loss: {loss.item():.6f}, Jac: {_ja.mean().item():.6f}, Mag Err: {loss_mangitude.item():.6f}, Ang Err: {loss_angle.item():.6f}")
              
        # ----------Log to Weights and Biases
        wb.log({
            "loss": loss.item(),
            "loss_ptv": loss_ptv.item(),
            "loss_vtp": loss_vtp.item(),
            "epoch": _+1,
            "percentage_error_magnitude": loss_mangitude.item(),
            "percentage_error_angle": loss_angle.item(),
            "jacobian": _ja.mean().item()
            
        })
        
        # Save the model every 100 epochs
        if (_ + 1) % 50==0 and end_loss > loss_vtp.item():
            end_loss = loss_vtp.item()
            torch.save(mix_model.state_dict(), os.path.join(save_path, f"MixedSplineFCPmodel_{num_nodes}.pth"))
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
            fig.savefig(os.path.join(save_path, f"MixedSplineFCP_gen_{num_nodes}.png"))

            # ✅ log the figure object, not `plt`
            # wb.log({"MixedSplineFCP_gen": fig})
            
            plt.close(fig)   # close after logging

           
           
if __name__ == "__main__":
    main()