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
        
    def pf_input(self):
        """
        Three types of the bus:
        - PQ bus: input P, Q; output V, θ
        - PV bus: input P, V; output Q, θ
        - Slack bus: input V, θ; output P, Q
        
        return the input of the power flow analysis
        """
        input_dic = {}
        input_dic['bus_p'] = self.net.load.p_mw
        input_dic['bus_q'] = self.net.load.q_mvar
        input_dic["v_bus"] = self.net.ext_grid.vm_pu
        
        # input_dic['pv_buses'] = self.net.sgen.bus.values.tolist()  # PV buses
        # input_dic['slack_buses'] = self.net.ext_grid.bus.values.tolist()  # Slack bus
        return input_dic
        
        
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
    case39._diagnose
    print(case39.net.res_bus)
    # change the loads
    length = len(case39.net.load)
    p_vec = [199 * 1 + np.random.randn() * 1 for i in range(length)]
    q_vec = [30 * 1 + np.random.randn() * 1 for i in range(length)]
    case39.set_loads(p_vec, q_vec)
    
    # Sys input
    # input_dic = case39.pf_input()
    # print("Power flow input:", input_dic)
    
    # Example usage of the power flow analysis
    print("Running power flow analysis with modified loads...")
    result = case39.run_pf()
    print("Power flow results:", result)
    
    # Plot
    pp.plotting.plotly.simple_plotly(case39.net, filename="src/powersystems/ieee39_network.html")