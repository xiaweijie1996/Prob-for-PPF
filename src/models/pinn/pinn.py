import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import torch.nn as nn


class PinnmodelEncoderBasic(torch.nn.Module):
    def __init__(self, 
                 num_nodes :int =33,
                 edge_index :torch.Tensor =None,
                 
                 # Convolution parameters
                 hidden_channel :int =64,
                ):
        super(PinnmodelEncoderBasic, self).__init__()
       
        # Admittance matrix Real par 
        self.adm_real = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=True)
        self.adm_imag = nn.Parameter(torch.randn(num_nodes, num_nodes), requires_grad=True)
       
        # Bias term
        self.bias_p = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
        self.bias_q = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
        
        # Define a convolution NN layer for [B, 1, N, N] -> [B, 1 , N, N] 
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden_channel, out_channels=1, kernel_size=1),
        )
        
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden_channel, out_channels=1, kernel_size=1),
        )
        
        # Graph structure
        self.num_nodes = num_nodes
        self.edge_index = edge_index  # [2, E]
        self.generate_matrix()
    
    def generate_matrix(self):
        """
        Generate adjacency matrix and degree matrix from edge_index
        """
        self.adj_matrix = torch.zeros((self.num_nodes, self.num_nodes)).to(self.edge_index.device)

        for i in range(self.edge_index.size(1)):
            src = self.edge_index[1, i]
            tgt = self.edge_index[0, i]
            self.adj_matrix[tgt, src] += 1
        
        # Add self-loops
        self.adj_matrix += torch.eye(self.num_nodes).to(self.edge_index.device)
        
        
    def forward(self, y_u, y_w):
        """
        y_u: [B, N] - Node features for each node in the graph, real part of voltage
        y_w: [B, N] - Node features for each node in the graph, imag part of voltage
        """
        
        # --- Construct the bilinear mapping ---
        yuyu = torch.matmul(y_u.unsqueeze(2), y_u.unsqueeze(1)) # [B, N, 1] @ [B, 1, N] -> [B, N, N]
        ywyw = torch.matmul(y_w.unsqueeze(2), y_w.unsqueeze(1)) # [B, N, 1] @ [B, 1, N] -> [B, N, N]
        yuyw = torch.matmul(y_u.unsqueeze(2), y_w.unsqueeze(1)) # [B, N, 1] @ [B, 1, N] -> [B, N, N]
        ywyu = torch.matmul(y_w.unsqueeze(2), y_u.unsqueeze(1)) # [B, N, 1] @ [B, 1, N] -> [B, N, N]

        s1 = (yuyu + ywyw) @ self.adm_real + (ywyu - yuyw) @ self.adm_imag # [B, N, N] @ [N, N] -> [B, N, N]
        s2 = (ywyu - yuyw) @ self.adm_real - (yuyu + ywyw) @ self.adm_imag # [B, N, N] @ [N, N] -> [B, N, N]
        
        # --- Non-linear transformation ---
        s1 = self.convs1(s1.unsqueeze(1)).squeeze(1) # [B, N, N]
        s2 = self.convs2(s2.unsqueeze(1)).squeeze(1) # [B, N, N]

        # --- Leverageing the graph structure ---
        s1 = s1 * self.adj_matrix.unsqueeze(0) # [B, N, N] * [1, N, N] -> [B, N, N]
        s2 = s2 * self.adj_matrix.unsqueeze(0) # [B, N, N] * [1, N, N] -> [B, N, N]
        
        # --- Final linear transformation ---
        one_vector = torch.ones((y_u.size(1), 1), device=y_u.device) # [N, 1]
        y_p = s1 @ one_vector + self.bias_p # [B, N, N] @ [N, 1] -> [B, N, 1] 
        y_q = s2 @ one_vector + self.bias_q # [B, N, N] @ [N, 1] -> [B, N, 1] 
        return y_p.squeeze(-1), y_q.squeeze(-1) # [B, N], [B, N]
    
class PinnEncoder(torch.nn.Module):
    def __init__(self, 
                 num_nodes :int =33,
                 edge_index :torch.Tensor =None,
                 num_block :int =4,
                 hidden_channel :int =64,
                ):
        super(PinnEncoder, self).__init__()
       
        self.num_block = num_block
        self.blocks = nn.ModuleList()
        for _ in range(num_block):
            model_block = PinnmodelEncoderBasic(num_nodes=num_nodes, edge_index=edge_index, hidden_channel=hidden_channel)
            bn = nn.BatchNorm1d(num_nodes)
            activation = nn.LeakyReLU(0.2)
            self.blocks.append(nn.ModuleList([model_block, bn, activation]))

        self.lastblock = PinnmodelEncoderBasic(num_nodes=num_nodes, edge_index=edge_index, hidden_channel=hidden_channel)
        
    def forward(self, y_u, y_w):
        """
        y_u: [B, N] - Node features for each node in the graph, real part of voltage
        y_w: [B, N] - Node features for each node in the graph, imag part of voltage
        """
        for block, bn, activation in self.blocks:
            y_u, y_w = block(y_u, y_w)
            # y_u = y_u + y_u_1 # Residual connection
            # y_w = y_w + y_w_2 # Residual connection
            y_u = bn(y_u)
            y_w = bn(y_w)
            y_u = activation(y_u)
            y_w = activation(y_w)
        # Last block without activation
        y_u, y_w = self.lastblock(y_u, y_w)
        return y_u, y_w


if __name__ == "__main__":
    import pandas as pd
    # Example usage
    # [[Target nodes], [Source nodes]]
    system_file = 'src/powersystems/files/Lines_34.csv'
    edge_index = pd.read_csv(system_file, header=None)
    edge_index = edge_index.iloc[:, :2].apply(pd.to_numeric, errors='coerce').dropna().values.astype(int)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)-1  # Convert to zero-based index
    # cancel column with 0
    edge_index = edge_index[:, edge_index[0, :] != 0]
    edge_index = edge_index[:, edge_index[1, :] != 0]
    edge_index -= 1  # Convert to zero-based index
    
    x = torch.randn(10, 33)
    y = torch.randn(10, 33)
    model = PinnEncoder(num_nodes=33, edge_index=edge_index, num_block=3)
    p, q = model(x, y)
    print(p.shape, q.shape)  # Should print: torch.Size([10,