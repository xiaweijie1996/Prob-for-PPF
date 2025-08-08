from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter
import torch

class Node34Example:
    
    def __init__(self, 
                 active_power: np.ndarray = None,
                 reactive_power: np.ndarray = None,
                 
                 ):
        """
        Initialize the Node34Example class with optional reactive power input.
        
        Parameters:
        reactive_power (np.ndarray): Optional array of reactive power values.
        active_power (np.ndarray): Optional array of active power values.
        """
        self.active_power = active_power
        self.reactive_power = reactive_power
        self.gpu = self._checkgpu()
        
    def _checkgpu(self):
        """
        Check if a GPU is available for computation.
        
        Returns:
        bool: True if a GPU is available, False otherwise.
        """
        return torch.cuda.is_available()

    def run(self):
        """
        Run the power flow analysis for the 34 node bus network.
        
        Returns:
        dict: A dictionary containing the voltage magnitudes at each node.
        
        dict_keys(['v', 'time_pre_pf', 'time_pf', 'time_algorithm', 'iterations',
        'convergence', 'iterations_log', 'time_pre_pf_log', 'time_pf_log', 'convergence_log'])
        """
        # Check if there is a GPU available and set the device accordingly
        device = self.gpu
        
        # Load the 34 node bus network
        network = GridTensor(gpu_mode=device)
        
        # Run power flow analysis
        solution = network.run_pf(
            active_power=self.active_power,
            reactive_power=self.reactive_power
        )
        
        return solution
    
    
if __name__ == "__main__":
    # Example usage of the Node34Example class
    active_power = np.random.normal(50, scale=1, size=(100, 33))  # # Power in kW
    print("Active power:", active_power.shape)
    reactive_power = active_power * 0.1
    example = Node34Example(reactive_power= reactive_power,
                            active_power=active_power)
    
    result = example.run()
    print("Voltage magnitudes at each node:", len(result))
    print("Voltage magnitudes:", result["v"][1,:])
    print("Convergence status:", result.keys())
    