import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv  # swap to GCNConv or ChebConv if you prefer

# --- 1) Your one-hop message passing block (edge-aware) ---
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch_geometric 

class MPlayer(MessagePassing):
    def __init__(self, nfeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j):
        return self.edge_aggr(torch.cat([x_i, x_j], dim=-1)) 

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
        x = x + self.edge_aggr(x, edge_index)
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
        self.lastlayer = MPlayer(hidden_dim, hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = self.firstlayer(x, edge_index)
        for model in self.modelist:
            x = model(x, edge_index)
        x = self.lastlayer(x, edge_index)
        return x

# --- 4) Run it ---
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # --- 3) Tiny demo graph ---
    def make_toy_graph():
        # 5 nodes, undirected edges (send both directions)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4,
            1, 0, 2, 1, 3, 2, 4, 3],  # reverse edges to make undirected
            [1, 0, 2, 1, 3, 2, 4, 3,
            0, 1, 1, 2, 2, 3, 3, 4]
        ], dtype=torch.long)

        N = 5
        nfeature_dim = 33
        x = torch.ones(N, nfeature_dim)       # node features [N, F]
        
        return Data(x=x, edge_index=edge_index)

    data = make_toy_graph()
    x, edge_index = data.x, data.edge_index
    
    model = PFnet(
        nfeature_dim=33,
        hidden_dim=128,
        out_dim=33,
        n_blocks=6,
        K_convs=3,
        K=3
    )
    out = model(x, edge_index)
    print('parameter count:', sum(p.numel() for p in model.parameters()))
    print(out)