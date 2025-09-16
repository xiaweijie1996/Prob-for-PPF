import numpy as np
import pandapower as pp
import networkx as nx

import pandapower.networks as pn
from pandapower.topology import create_nxgraph

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
    
    
    def bus_index(self):
        """
        Returns the bus indices of the network.
        
        """
        net = self.net
        slack_buses = net.ext_grid.bus.tolist()                 # slack / reference bus(es)
        pv_buses    = sorted(set(net.gen.bus.tolist()) - set(slack_buses))
        pq_buses    = sorted(set(net.bus.index) - set(slack_buses) - set(pv_buses))
        return slack_buses, pv_buses, pq_buses
    
    def get_topology_adj_and_edges(self, respect_switches=True):
        """
        Returns:
        A  : numpy [N,N] binary adjacency (lines+trafos)
        ei : numpy [2,E] undirected edge list (each undirected edge once) firt column "from", second "to"
        neighbors: dict {bus: [neighbor buses]}
        """
        net = self.net
        
        # Build NX graph (simple graph, no multi-edges)
        G = create_nxgraph(net,
                        respect_switches=respect_switches,
                        include_lines=True,
                        include_trafos=True,
                        multi=False)
        nodelist = list(net.bus.index)  # ensure row/col order == bus indices
        A = nx.to_numpy_array(G, nodelist=nodelist, dtype=int)  # 0/1 adjacency
        ei = np.array(list(G.edges()), dtype=int).T if G.number_of_edges() else np.zeros((2,0), dtype=int)
        # The graph is undirected, but pandapower lines/trafo have a direction. 
        # Append reverse edges to get undirected edge list
        if ei.shape[1]>0:
            ei = np.hstack((ei, ei[::-1,:]))  # both directions
            
        # Update adjacency to be symmetric
        A = np.maximum(A, A.T)
        return A, ei
        
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
    # case39._diagnose
    # print(case39.net.res_bus)
    # change the loads
    length = len(case39.net.load)
    p_vec = [199 for i in range(length)]
    q_vec = [30  for i in range(length)]
    case39.set_loads(p_vec, q_vec)
    
    # Sys input
    # input_dic = case39.pf_input()
    # print("Power flow input:", input_dic)
    
    # Example usage of the power flow analysis
    print("Running power flow analysis with modified loads...")
    result = case39.run_pf()
    print("Power flow results:", result)
    
    # Plot
    pp.plotting.to_html(case39.net, filename="src/powersystems/ieee39_network.html")
    
    A, ei = case39.get_topology_adj_and_edges()
    print("Adjacency matrix:\n", A)
    print("Edge list:\n", ei)