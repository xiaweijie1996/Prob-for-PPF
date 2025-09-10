import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx

import src.models.basicnetwork.basicnets as basicnets

torch.set_default_dtype(torch.float64)

class GnnBasic(nn.Module):
    def __init__(self, 
                 # Graph parameters
                 num_nodes :int =34,
                 norm_adj_matrix :torch.Tensor =None,
                 
                 # Graph convolution parameters
                 num_hidden_layers :int =2,
                 hidden_dim :int =16,
                 ):
        super(GnnBasic, self).__init__()
        self.num_nodes = num_nodes # Total number of nodes
        self.norm_adj_matrix = norm_adj_matrix  # [N, N]
        
        self.conv_layer = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.conv_layer.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1))
            self.conv_layer.append(nn.LeakyReLU(0.1))
        self.conv_layer = nn.Sequential(*self.conv_layer)
        
    def forward(self, x):
        """
        x: [B, N, Fin] - Node features for each node in the graph
        """
        # Graph convolution operation
        x = self.norm_adj_matrix @ x # [B, N, Fin] @ [N, N] -> [B, N, Fin]
        
        # Permute to [B, Fin, N], Fin as channels
        x = x.permute(0, 2, 1) # [B, Fin, N]
        
        # Apply convolutional layers
        x = self.conv_layer(x) # [B, Fout, N]
        
        return x.permute(0, 2, 1) # [B, N, Fout]

class Gnn(nn.Module):
    def __init__(self, 
                 # Graph parameters
                 num_nodes :int =34,
                 dim_node_feature :int =2,
                 edge_index :torch.Tensor =None,
                 
                 # Graph convolution parameters
                 gnn_num_hidden_layers :int =2,
                 hidden_dim :int =16,
                 num_block :int =2,
                 
                 # MLP parameters
                 mlp_hidden_dim :int =64,
                 mlp_num_layers :int =2
                 ):
        
        super(Gnn, self).__init__()
        self.num_nodes = num_nodes
        self.dim_node_feature = dim_node_feature
        self.edge_index = edge_index  # [2, E]
        
        # Initialize adjacency and degree matrices
        self.generate_matrix()
        
        # Define GNN layers
        self.first_layer = nn.Linear(dim_node_feature, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [GnnBasic(
            num_nodes=num_nodes,
            norm_adj_matrix=self.norm_adj_matrix,
            num_hidden_layers=gnn_num_hidden_layers,
            hidden_dim=hidden_dim 
            ) for _ in range(num_block)]
        )
        
        # Define the densie layer to map to output features if needed
        self.linear_out = basicnets.BasicFFN(
            input_dim=hidden_dim * num_nodes,
            hidden_dim=mlp_hidden_dim,
            output_dim=dim_node_feature * num_nodes,
            n_layers=mlp_num_layers
        )

    def generate_matrix(self):
        """
        Generate adjacency matrix and degree matrix from edge_index
        """
        self.adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.edge_index.size(1)):
            src = self.edge_index[1, i]
            tgt = self.edge_index[0, i]
            self.adj_matrix[tgt, src] += 1
        
        # Add self-loops
        self.adj_matrix += torch.eye(self.num_nodes)
        
        # Degree matrix
        self.degree_matrix = torch.diag(self.adj_matrix.sum(dim=1))
        
        # Normalize adjacency matrix (symmetric normalization)
        deg_inv_sqrt = torch.pow(self.degree_matrix.diag(), -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        self.norm_adj_matrix = D_inv_sqrt @ self.adj_matrix @ D_inv_sqrt
  
    def plot_graph(self, save_path=None):
        # Figure size
        plt.figure(figsize=(20, 20))
        # Convert adjacency to directed graph
        G = nx.from_numpy_array(self.adj_matrix.numpy(), create_using=nx.DiGraph)

        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes + edges with arrows
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.1"
        )
        # Draw edge labels if weighted
        edge_labels = nx.get_edge_attributes(G, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Directed Graph Visualization")
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()
    
    def forward(self, x):
        """
        x: [B, N, Fin] - Node features for each node in the graph
        """
        # Initial mapping
        x = self.first_layer(x) # [B, N, hidden ]
        
        # Go through GNN blocks
        for gnn in self.gnn_layers: # Apply residual connection
            x = x + gnn(x) # [B, hidden, N]
        
        # Linear layer to map to output features if needed
        x = x.reshape(x.size(0), -1) # [B, N*hidden]
        x = self.linear_out(x) # [B, N*Fin]
        x = x.reshape(x.size(0), self.num_nodes, self.dim_node_feature)
        return x
        
if __name__ == "__main__":
    import pandas as pd
    
    num_nodes = 33
    batch_size = 100
    dim_node_feature = 2
    hidden_dim = 256
    num_hidden_layers = 3
    num_block = 4
    
    # Example usage
    # [[Target nodes], [Source nodes]]
    system_file = 'src/powersystems/files/Lines_34.csv'
    edge_index = pd.read_csv(system_file, header=None)
    edge_index = edge_index.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna().values.astype(int)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)-1  # Convert to zero-based index
    print(edge_index)
    # cancel column with 0
    edge_index = edge_index[:, edge_index[0, :] != 0]
    edge_index = edge_index[:, edge_index[1, :] != 0]
    edge_index -= 1  # Convert to zero-based index
    print(edge_index)
    
    x = torch.randn((batch_size, num_nodes, dim_node_feature))  # [B, N, Fin]
    
    gnn = Gnn(
        num_nodes=num_nodes,
        dim_node_feature=dim_node_feature,
        edge_index=edge_index,
        gnn_num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        num_block=num_block
    )
    gnn.plot_graph('src/models/gnn/graph_visualization.png')
    print("Model Parameters: ", sum(p.numel() for p in gnn.parameters() if p.requires_grad))
    out = gnn(x)  # [B, N, Fout]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
