from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter
import torch

class Node34Example:
    
    def __init__(self):
        """
        Initialize the Node34Example class with optional reactive power input.
        
        Parameters:
        gpu_mode (bool): Whether to use GPU for computation.
        """
    
        self.gpu = False
        
        # Load the 34 node bus network
        self.network = GridTensor(gpu_mode=self.gpu)
        
        
    def _checkgpu(self):
        """
        Check if a GPU is available for computation.
        
        Returns:
        bool: True if a GPU is available, False otherwise.
        """
        return torch.cuda.is_available()

    def run(self, 
            active_power: np.ndarray = None,
            reactive_power: np.ndarray = None
            ):
        """
        Run the power flow analysis for the 34 node bus network.
        
        Returns:
        dict: A dictionary containing the voltage magnitudes at each node.
        
        dict_keys(['v', 'time_pre_pf', 'time_pf', 'time_algorithm', 'iterations',
        'convergence', 'iterations_log', 'time_pre_pf_log', 'time_pf_log', 'convergence_log'])
        """
        # Check if there is a GPU available and set the device accordingly
       
        
        # Run power flow analysis
        solution = self.network.run_pf(
            active_power=active_power,
            reactive_power=reactive_power
        )
        
        return solution
    
    
if __name__ == "__main__":
    # Example usage of the Node34Example class
    # np.random.seed(42)
    active_power = np.random.normal(50, scale=1, size=(2, 33))  # # Power in kW
    print("Active power shape:", active_power.mean())
    print("Active power:", active_power.shape)
    reactive_power = active_power * 0.1
    system = Node34Example()
    # print("Active Power (P):", system.)
    # mean_vector = system.network_rnd.bus_info
    
    result = system.run(reactive_power=reactive_power,
                        active_power=active_power)
    print("Voltage magnitudes at each node:", len(result))
    print("Voltage magnitudes:", result["v"].mean())
    print("Convergence status:", result.keys())
    