from tensorpowerflow import GridTensor
import numpy as np
from time import perf_counter
import torch

class randomsystem:
    def __init__(self, 
                 num_nodes: int = 100,
                 num_children: int = 3,
                 plot_graph: bool = False
                 ):
        """
        Initialize the randomsystem class with optional reactive power input.
        
        Parameters:
        num_nodes (int): Number of nodes in the random system.
        num_children (int): Number of children nodes for each parent node.
        """
        self.num_nodes = num_nodes
        self.num_children = num_children
        self.gpu = self._checkgpu()
        self.plot_graph = plot_graph
    
        # Generate a system
        np.random.seed(42) 
        self.network_rnd = GridTensor.generate_from_graph(
            nodes=self.num_nodes,
            child=self.num_children,
            plot_graph=self.plot_graph,
            gpu_mode=False,
        )
        
    def _checkgpu(self):
        """
        Check if a GPU is available for computation.
        
        Returns:
        bool: True if a GPU is available, False otherwise.
        """
        return torch.cuda.is_available()

    def run(self,
            active_power: np.ndarray = None,
            reactive_power: np.ndarray = None,
            ):
        """
        Run the power flow analysis for the random system.
        
        Returns:
        dict: A dictionary containing the voltage magnitudes at each node.
        
        dict_keys(['v', 'time_pre_pf', 'time_pf', 'time_algorithm', 'iterations',
        'convergence', 'iterations_log', 'time_pre_pf_log', 'time_pf_log', 'convergence_log'])
        """

        # Run power flow analysis
        solution = self.network_rnd.run_pf(
            active_power=active_power,
            reactive_power=reactive_power
        )
        
        return solution
    
def magnitude_transform(v):
    """
    Calculate the magnitude of a complex voltage vector.
    
    Parameters:
    v (np.ndarray): Complex voltage vector.
    
    Returns:
    np.ndarray: Magnitude of the voltage vector.
    """
    return np.sqrt(np.real(v)**2 + np.imag(v)**2)

def angle_transform(v):
    """
    Calculate the angle of a complex voltage vector.
    
    Parameters:
    v (np.ndarray): Complex voltage vector.
    
    Returns:
    np.ndarray: Angle of the voltage vector in radians.
    """
    return np.angle(v)
    
if __name__ == "__main__":
    # Example usage of the randomsystem class
    active_power = np.random.normal(50, scale=1, size=(50, 4))  # Power in kW
    reactive_power = np.random.normal(10, scale=0.5, size=(50, 4))  # Reactive power in kVAR
    
    system = randomsystem(num_nodes=5, num_children=2, plot_graph=True)
    # Print the system initial P and Q
    print("Active Power (P):", system.network_rnd.bus_info['PD'].shape)
    mean_vector = system.network_rnd.bus_info['PD']
    mean_vector = np.array(mean_vector)
    print(f"Mean vector: {mean_vector[1:].shape}, {mean_vector[1:]}")
    
    result = system.run(active_power=active_power, reactive_power=reactive_power)
    
    print("Voltage magnitudes at each node:", len(result))
    print("Voltage magnitudes:", result["v"].shape)
    print("Convergence status:", result.keys())
    print("Active Power (P):", system.network_rnd.bus_info['PD'])