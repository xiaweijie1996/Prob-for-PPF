import numpy as np
import pandapower as pp
import pandapower.networks as pn

class Case39PF:
    def __init__(self):
        """
        Initialize the 39-bus network and set P/Q for all loads.
        - p_vec, q_vec: lists or arrays matching the number of loads (net.load).
        """
        self.net = pn.case39()
    
    @property
    def _diagnose(self):
        """
        Print a summary of the network.
        """
        pp.runpp(self.net, max_iteration="auto")
        
        print('Network features', self.net)
        print("Bus data:")
        print(self.net.res_bus)
        # The degault range of p_mw
        print("Default range of p_mw:", self.net.res_load.p_mw.min(), self.net.res_load.p_mw.max())
        # the default range of q_mvar
        print("Default range of q_mvar:", self.net.res_load.q_mvar.min(), self.net.load.q_mvar.max())
        print("Load data:")
        print(self.net.load)
        
    def set_loads(self, 
                p_vec: np.ndarray,
                q_vec: np.ndarray
                ):
        """
        Set the active and reactive power for all loads in the network.
        Parameters:
        p_vec (list or np.ndarray): Active power values for each load.
        q_vec (list or np.ndarray): Reactive power values for each load.
        """
    
        self.net.load["p_mw"] = p_vec
        self.net.load["q_mvar"] = q_vec
        
    def run_pf(self, max_iteration="auto"):
        """
        Runs the power flow analysis on the network.
        Parameters:
        max_iteration (int or str): Maximum number of iterations for the power flow solver.
                                    Default is "auto" which lets pandapower decide.
        """
        
        pp.runpp(self.net, max_iteration=max_iteration)
        
        return self.net.res_bus
        

if __name__ == "__main__":

    case39 = Case39PF()
    case39._diagnose()
    
    # change the loads
    length = len(case39.net.load)
    p_vec = [199 * 1 for i in range(length)]  # Example active power values
    q_vec = [30 * 1 for i in range(length)]
    case39.set_loads(p_vec, q_vec)
    
    # Example usage of the power flow analysis
    print("Running power flow analysis with modified loads...")
    result = case39.run_pf()
    print("Power flow results:", result)