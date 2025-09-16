import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TAGConv  # swap to GCNConv or ChebConv if you prefer

# --- 1) Your one-hop message passing block (edge-aware) ---
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class EdgeAggregation(MessagePassing):
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def message(self, x_i, x_j, edge_attr):
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def forward(self, x, edge_index, edge_attr):
        # (Optional) degree-based normalization if you want GCN-style scaling
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, norm=norm)
        return out

# --- 2) The model: 1 hop MP + K convs ---
class OneHopThenKConv(nn.Module):
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, out_dim, K_convs=3, K=3):
        """
        nfeature_dim: node feature size
        efeature_dim: edge feature size
        hidden_dim:   hidden size for both MP and conv stack
        out_dim:      final output dim per node
        K_convs:      number of conv layers (K)
        K:            TAGConv hop parameter (polynomial order). Not to be confused with K_convs
        """
        super().__init__()
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        if K_convs == 1:
            self.convs.append(TAGConv(hidden_dim, out_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            for _ in range(K_convs - 2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            self.convs.append(TAGConv(hidden_dim, out_dim, K=K))

        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.ReLU()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # 1) one-hop edge-aware message passing
        x = self.edge_aggr(x, edge_index, edge_attr)
        # 2) K conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.dropout(self.act(x))
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
        nfeature_dim = 4
        efeature_dim = 2

        x = torch.randn(N, nfeature_dim)       # node features [N, F]
        edge_attr = torch.randn(edge_index.size(1), efeature_dim)  # edge features [E, Fe]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    nfeature_dim = 4
    efeature_dim = 2
    hidden_dim = 32
    out_dim = 8
    K_convs = 3   # number of conv layers
    K = 2         # TAGConv hop/poly order (not the number of conv layers)

    model = OneHopThenKConv(nfeature_dim, efeature_dim, hidden_dim, out_dim, K_convs=K_convs, K=K)
    data = make_toy_graph()
    out = model(data)  # [N, out_dim]
    print("Output shape:", out.shape)  # should be [5, 8]

    # Example training step (dummy target)
    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    print("Loss:", float(loss))
