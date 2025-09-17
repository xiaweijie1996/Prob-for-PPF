import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv  # swap to GCNConv or ChebConv if you prefer

# --- 1) Your one-hop message passing block (edge-aware) ---
from torch_geometric.nn import MessagePassing


class MPlayer(MessagePassing):
    def __init__(self, nfeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j):
        xx_ij = torch.cat([x_i, x_j], dim=-1)  # [N, nodes, 2*F]
        return self.edge_aggr(xx_ij)  # [N, nodes, out_dim]

    def forward(self, x, edge_index):
        # (Optional) degree-based normalization if you want GCN-style scaling
        out = self.propagate(edge_index=edge_index, x=x)
        return out

# --- 2) The model: 1 hop MP + K convs ---
class OneHopThenKConv(nn.Module):
    def __init__(self,
                 # Model params
                 hidden_dim, 
                 K_convs=3,
                 K=3):
        """
        nfeature_dim: node feature size
        efeature_dim: edge feature size
        hidden_dim:   hidden size for both MP and conv stack
        out_dim:      final output dim per node
        K_convs:      number of conv layers (K)
        K:            TAGConv hop parameter (polynomial order). Not to be confused with K_convs
        """
        super().__init__()
        self.edge_aggr = MPlayer(hidden_dim, hidden_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        if K_convs == 1:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            for _ in range(K_convs - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
    
        
        self.act = nn.ReLU()
    def forward(self, x, edge_index):
        # 1) one-hop edge-aware message passing
        x = self.edge_aggr(x, edge_index)
        # 2) K conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)
        
        return x

class PFnet(nn.Module):
    def __init__(self, 
                 # Model params
                 nfeature_dim, 
                 hidden_dim, 
                 out_dim, 
                 n_blocks,
                 K_convs=3,
                 K=3
                 ):
        """
        nfeature_dim: node feature size
        efeature_dim: edge feature size
        hidden_dim:   hidden size for both MP and conv stack
        out_dim:      final output dim per node
        K_convs:      number of conv layers (K)
        K:            TAGConv hop parameter (polynomial order). Not to be confused with K_convs
        """
        super().__init__()
        self.modelist = nn.ModuleList()
        for _ in range(n_blocks):
            self.modelist.append(
                OneHopThenKConv(
                    hidden_dim=hidden_dim,
                    K_convs=K_convs,
                    K=K
                )
            )
        self.firstlayer = MPlayer(nfeature_dim, hidden_dim, hidden_dim)
        self.lastlayer = nn.Linear(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = self.firstlayer(x, edge_index)
        for model in self.modelist:
            x = model(x, edge_index)
        x = self.lastlayer(x)
        return x

# --- 4) Run it ---
if __name__ == "__main__":
    torch.manual_seed(0)
    
   
    x = torch.randn(10, 33, 2)  # [B, N, F]
    edge_index = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 30],
                               [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 30, 1]])
    print(edge_index.shape)
    
    model = PFnet(
        nfeature_dim=2,
        hidden_dim=64,
        out_dim=2,
        n_blocks=2,
        K_convs=3,
        K=3
    )
    out = model(x, edge_index)
    print(out.shape)  # [B, N, out_dim]
    
    
    
    