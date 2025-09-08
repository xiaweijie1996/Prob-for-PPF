import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(edge_index, num_nodes, device=None):
    # edge_index: [2, E] with self-loops added
    row, col = edge_index
    # degree
    deg = torch.bincount(row, minlength=num_nodes).float()

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    
    # return normalized scalar per edge: D^-1/2 A D^-1/2
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index):
        # x: [N, Fin], edge_index: [2, E]
        N = x.size(0)
        device = x.device

        # add self-loops
        self_loops = torch.arange(N, device=device)
        ei = torch.cat([edge_index,
                        torch.stack([self_loops, self_loops], dim=0)], dim=1)
        norm = normalize_adj(ei, N).to(device)  # [E']
        xW = self.lin(x)  # [N, Fout]

        # sparse message passing: sum_j norm_ij * xW_j
        row, col = ei
        out = torch.zeros_like(xW)
        out.index_add_(0, row, norm.unsqueeze(-1) * xW[col])
        return out

class GnnModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, out_dim))
        
        self.constrain = self.adjusted_function
        
    def adjusted_function(self, x):
        # _output = torch.
        _output = torch.tanh(x) * 5  # Scale to [-0.5, 0.5]
        return _output
    
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = self.constrain(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)  # [2, E]
    
    x = torch.randn((3, 4))  # [N, Fin]
    model = GnnModel(in_dim=4, hidden_dim=8, out_dim=2, num_layers=3)
    print(x)
    out = model(x, edge_index)
    print(out)  # [N, Fout]
    print(out.shape)  # should be [3, 2]