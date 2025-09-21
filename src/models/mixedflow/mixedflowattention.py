import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
from src.models.mixedflow.splineattention import CSplineBasicAttention
from src.models.mixedflow.fcpattention import CFCPBasicAttention
# Set all tensor Double globally dtyepe
torch.set_default_dtype(torch.float64)


class CMixedModel(torch.nn.Module):
    def __init__(self, 
                 # input features
                 input_dim: int = 2,
                 
                 # model features transformer
                 num_layers_spline: int = 3,
                 num_layers_fcp: int = 3,
                 num_blocks: int = 1,
                 emb_dim: int = 64,
                 num_heads: int = 4,
                 bias: bool = True,
                 num_nodes: int = 33,
                 num_output_nodes: int = 1,  
                 output_dim_fcp: int = 1,
                 
                 # model features spline
                 b_interval: float = 5.0, # better to max of the output data maybe
                 k_bins: int = 10, # number of bins
                 
                ):
        super(CMixedModel, self).__init__()
        
        self.blocks_spline = torch.nn.ModuleList([
            CSplineBasicAttention(
                # input features
                input_dim = input_dim,
                # model features transformer
                num_blocks=num_blocks,
                emb_dim=emb_dim,
                num_heads=num_heads,
                bias=bias,
                num_nodes=num_nodes,
                num_output_nodes=num_output_nodes,
                
                # model features spline
                b_interval=b_interval,
                k_bins=k_bins,
                 
                
            ) for _ in range(num_layers_spline)
        ])
        
        self.blocks_realnvp = torch.nn.ModuleList([
            CFCPBasicAttention(
                input_dim=input_dim,
                num_blocks_encoder=num_blocks,
                output_dim=output_dim_fcp,
                embed_dim=emb_dim,
                num_heads=num_heads,
                bias=bias,
                num_nodes=num_nodes,
                num_output_nodes=num_output_nodes,
                
                
            ) for _ in range(num_layers_fcp)
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
    batch_size = 100
    x = torch.randn(batch_size, 1, 2)
    c = torch.randn(batch_size, 33, 2)
    
    index_p = 1
    index_v = 1
    model = CMixedModel(
        input_dim=2,
        num_layers_spline=2,
        num_layers_fcp=2,
        num_blocks=1,
        emb_dim=64,
        num_heads=4,
        bias=True,
        num_nodes=33,
        num_output_nodes=1,
        b_interval=5.0,
        k_bins=10,
        output_dim_fcp=1
    )
    y, ja = model.forward(x, c, index_p=index_p, index_v=index_v)
    x_recon, _ = model.inverse(y, c, index_p=index_p, index_v=index_v)
    print("Input x:", x.shape)
    print("Transformed y:", y.shape)
    print("Reconstructed x:", x_recon.shape)
    print("Jacobian determinant ja:", ja.shape)
    print("Reconstruction error:", torch.mean((x - x_recon) ** 2).item())
    print("Allclose?", torch.allclose(x, x_recon, atol=1e-8, rtol=1e-8) )