import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
from src.models.spline.cspline import CSplineBasic
from src.models.crealnvp.crealnvp import CRealnvpBasic
 
# Set all tensor Double globally dtyepe
torch.set_default_dtype(torch.float64)


class CMixedModel(torch.nn.Module):
    def __init__(self, 
                 # input features
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 12,
                 
                 # model features 
                 n_layers: int = 3,
                 split_ratio: float = 0.5,
                 n_blocks_spline: int = 3,
                 n_blocks_realnvp: int = 2,
                 
                 
                 # model features condition
                 hidden_dim_condition: int = 32,
                 n_layers_condition: int = 2,
                 b_interval: float = 5.0,
                 k_bins: int = 10
                ):
        super(CMixedModel, self).__init__()
        
        self.blocks_spline = torch.nn.ModuleList([
            CSplineBasic(
                # Shared parameters with RealNVP
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                n_layers=n_layers,
                split_ratio=split_ratio,
                hidden_dim_condition=hidden_dim_condition,
                n_layers_condition=n_layers_condition,
                
                # Specific parameters for spline
                b_interval=b_interval,
                k_bins=k_bins
            ) for _ in range(n_blocks_spline)
        ])
        
        self.blocks_realnvp = torch.nn.ModuleList([
            CRealnvpBasic(
                # Shared parameters with spline
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                n_layers=n_layers,
                split_ratio=split_ratio,
                hidden_dim_condition=hidden_dim_condition,
                n_layers_condition=n_layers_condition
            ) for _ in range(n_blocks_realnvp)
        ])

    def forward(self, x, c, index_p, index_v):
        # Forward through blocks of spline
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in self.blocks_spline:
            x, ja_block = block.forward_direction(x, c, index_p=index_p, index_v=index_v)
            ja = ja * ja_block.squeeze(-1)
        
        # Forward through blocks of realnvp
        for block in self.blocks_realnvp:
            x, ja_block = block.forward(x, c, index_p=index_p, index_v=index_v)
            ja = ja * ja_block.squeeze(-1)
        return x, ja
    
    def inverse(self, y, c, index_p, index_v):
        # Inverse through blocks of realnvp
        for block in reversed(self.blocks_realnvp):
            y, _ = block.inverse(y, c, index_p=index_p, index_v=index_v)
            
        # Inverse through blocks of spline
        for block in reversed(self.blocks_spline):
            y, _ = block.inverse_direction(y, c, index_p=index_p, index_v=index_v)
            
        return y, _
                         
if __name__ == "__main__":
    model = CMixedModel()
    batch_size = 10 
    dim_in = 2  
    dim_c = 12
    x = torch.randn(batch_size, dim_in)
    c = torch.randn(batch_size, dim_c)
    index_p = 1
    index_v = 1
    y, ja = model.forward(x, c, index_p, index_v)
    print(y.shape, ja.shape)
    x_recon, _ = model.inverse(y, c, index_p, index_v)
    print(x_recon.shape)
    print('error', torch.norm(x - x_recon))
    print(torch.allclose(x, x_recon))