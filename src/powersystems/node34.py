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

def power_mismatch_from_V(edge_index, R, X, V, P_hat=None, Q_hat=None):
    """
    edge_index: LongTensor [2, E] with 0-based node indices (i -> j) for each line (undirected: list each line once)
    R, X:       FloatTensor [E] line resistance/reactance (per-unit preferred)
    V:          Complex tensor [N] bus voltages (per-unit), e.g. from result["v"][k]
    P_hat,Q_hat: optional FloatTensors [N] predicted active/reactive injections

    Returns:
      P, Q:       implied injections from (V,R,X)
      dP, dQ:     mismatches (only if P_hat/Q_hat given), else None
    """
    device = V.device
    E = edge_index.size(1)
    i, j = edge_index[0], edge_index[1]                         # [E], [E]

    # Complex line admittance
    Z = torch.complex(R.to(device), X.to(device))               # [E]
    Y = 1.0 / Z                                                 # [E]  (complex)

    # Branch currents (i -> j). KCL sign convention: sum_{j}(V_i - V_j)Y_ij
    Iij = (V[i] - V[j]) * Y                                     # [E] (complex)

    # Accumulate to nodal currents
    N = V.numel()
    I = torch.zeros(N, dtype=V.dtype, device=device)            # complex
    I.index_add_(0, i, Iij)                                     # + (V_i - V_j)Y_ij at node i
    I.index_add_(0, j, -Iij)                                    # + (V_j - V_i)Y_ij at node j

    # Nodal complex power implied by voltages/lines
    S = V * torch.conj(I)                                       # [N] complex
    P = S.real
    Q = S.imag
    
    return P, Q


if __name__ == "__main__":
    # Example usage of the Node34Example class
    np.random.seed(42)
    active_power = np.random.normal(50, scale=1, size=(2, 33))
    reactive_power = active_power * 0.1
    system = Node34Example()
    
    result = system.run(reactive_power=reactive_power,
                        active_power=active_power)
    print("Voltage magnitudes at each node:", len(result))
    print("Voltage:", result["v"][:4])
  
    # print system details
    print(system.network.branch_info)
    print(system.network.bus_info)
    
    # 1) Build edge_index, R, X from your branch_info DataFrame (0-based)
    bi = system.network.branch_info
    edge_index = torch.tensor(
        np.vstack([bi["FROM"].values - 1, bi["TO"].values - 1]),
        dtype=torch.long)

    R = torch.tensor(bi["R"].values, dtype=torch.float32)
    X = torch.tensor(bi["X"].values, dtype=torch.float32)

    # 2) Pick one scenario’s voltages (complex, per-unit)
    V_np = result["v"][0]                 # shape [N], complex64/complex128 in numpy
    # Add slack bus voltage if not included
    if V_np.shape[0] == 33:
        V_slack = 1.0 + 0.0j
        V_np = np.insert(V_np, 0, V_slack)    # now shape [34]
    V = torch.from_numpy(V_np).to(torch.complex64)

    # 3) (Optional) predicted injections — if you have them; else skip
    P_hat = None
    Q_hat = None

    # 4) Compute P,Q (and ΔP,ΔQ if hats provided)
    P, Q = power_mismatch_from_V(edge_index, R, X, V, P_hat, Q_hat)

    print( system.network.s_base)
    print("P (implied):", P[:5] * system.network.s_base)  # convert to MW
    print("Q (implied):", Q[:5] * system.network.s_base)  # convert to MVAr
    
    dp = P[1:] * system.network.s_base - torch.from_numpy(active_power[0]).float()
    dq = Q[1:] * system.network.s_base - torch.from_numpy(reactive_power[0]).float()
    print("dP (mismatch):", dp[:5])
    print("dQ (mismatch):", dq[:5])

    
    
    
    