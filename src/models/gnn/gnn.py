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
    print(deg_inv_sqrt, row, col)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    print(norm)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)  # [2, E]
    
    # Plot the graph
    plt.figure()
    plt.scatter([0, 1, 2], [0, 0, 0])
    for i in range(edge_index.size(1)):
        plt.plot([edge_index[0, i], edge_index[1, i]], [0, 0], 'k-')
    plt.title("Graph Structure")
    plt.savefig("src/models/gnn/graph_structure.png")
    plt.close()
    
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=torch.float)  # [N, Fin]
    

    gcn = GCNLayer(2, 2)
    out = gcn(x, edge_index)
    print(out)