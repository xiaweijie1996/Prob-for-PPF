import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.transformer as transformer


class CTarflowBasic(torch.nn.Module):
    def __init__(self, 
                # Encoder features
                input_dim: int,
                num_blocks: int,
                output_dim: int,
                
                # Attention features
                embed_dim: int,
                num_heads: int,
                bias: bool = True,
                 ):
        
        super(CTarflowBasic, self).__init__()
        
        self.split_dim1 = input_dim // 2
        # self.split_dim2 = input_dim - self.split_dim1
        
        # Define the Transformer encoders
        self.transformer_s1 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        
        self.transformer_s2 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        
        self.transformer_b1 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        
        self.transformer_b2 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        
        self.constrain = self.adjusted_sigmoid
        
    def adjusted_sigmoid(self, x):
        _output =  torch.sigmoid(x/5) * 4 - 2
        # if _output closer to 0, make it 1e-6
        _output = torch.where(torch.abs(_output) < 1e-10, torch.tensor(1e-10, device=x.device), _output)
        return _output
        
       
    def forward(self, x, c, index_p, index_v):

        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1, :], x[:, self.split_dim1:, :]
        print(x11.shape, x12.shape, c.shape)
        
        # Forward pass through the first coupling layer
        x21 = x11 # (B, s_len, split_dim1)
        x11_c = torch.cat([x11, c], dim=-1)
        s1 = torch.exp(self.constrain(self.transformer_s1(x11_c, index_p, index_v)))
        t1 = self.transformer_b1(x11_c, index_p, index_v)
        x22 = x12 * s1 + t1
        _det_ja1 = torch.cumprod(s1, dim=1)[:,-1]

        # Forward pass through the second coupling layer
        x32 = x22
        x22_c = torch.cat([x22, c], dim=-1)
        s2 = torch.exp(self.constrain(self.transformer_s2(x22_c, index_p, index_v)))
        t2 = self.transformer_b2(x22_c, index_p, index_v)
        x31 = x21 * s2 + t2
        _det_ja2 = torch.cumprod(s2, dim=1)[:,-1]

        # Final output
        x3 = torch.cat([x31, x32], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2)
        
        return x3, det_ja
    
    def inverse(self, x3, c, index_p, index_v):
  # Split the input tensor
        x31, x32 = x3[:, :self.split_dim1], x3[:, self.split_dim1:]
        
        # Inverse pass through the second coupling layer
        x22 = x32
        x22_c = torch.cat([x22, c], dim=-1)
        s2 = torch.exp(self.constrain(self.transformer_s2(x22_c, index_p, index_v)))
        t2 = self.transformer_b2(x22_c, index_p, index_v)
        x21 = (x31 - t2) / s2
        _det_ja2 = torch.cumprod(1/s2, dim=1)[:,-1]
        
        # Inverse pass through the first coupling layer
        x11 = x21
        x11_c = torch.cat([x21, c], dim=-1)
        s1 = torch.exp(self.constrain(self.transformer_s1(x11_c, index_p, index_v)))
        t1 = self.transformer_b1(x11_c, index_p, index_v)
        x12 = (x22 - t1) / s1
        _det_ja1 = torch.cumprod(1/s1, dim=1)[:,-1]
        
        # Final output
        x = torch.cat([x11, x12], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2 )
        
        return x, det_ja
    
# class SimplifiedFcpflow(torch.nn.Module):
#     def __init__(self, 
#                 input_dim: int,
#                 condition_dim: int,
#                 n_blocks: int = 4,
#                 hidden_dim: int = 256,
#                 n_layers: int = 2,
#                 split_ratio: float = 0.5,
                
#                 hidden_dim_condition: int = 128,
#                 output_dim_condition: int = 128,
#                 n_layers_condition: int = 2,
#                  ):
#         super(SimplifiedFcpflow, self).__init__()
        
#         self.blocks = torch.nn.ModuleList([
#             CTarflowBasic(
#                 input_dim=input_dim,
#                 num_blocks=n_layers,
#                 output_dim=input_dim,
#                 embed_dim=hidden_dim,
#                 num_heads=4,
#                 bias=True,
#             ) for _ in range(n_blocks)
#         ])
       
    
#     def forward(self, x, c, index_p, index_v, postional_encoding=False):
#         # Forward through blocks
#         ja = torch.ones((x.shape[0]), device=x.device)
#         for block in self.blocks:
#             x, _ja = block.forward(x, c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
#             ja = ja * _ja
            
#         return x, ja
    
#     def inverse(self, x, c, index_p, index_v, postional_encoding=False):
#         # Inverse through blocks
#         ja = torch.ones((x.shape[0]), device=x.device)
#         for block in reversed(self.blocks):
#             x, _ja = block.inverse(x, c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
#             ja = ja * _ja
#         return x, ja
    
    
if __name__ == "__main__":
    x = torch.randn(2, 10, 2)
    c = torch.randn(2, 10, 2)
    index_p = 1
    index_v = 2
    
    model = CTarflowBasic(
        input_dim=2,
        num_blocks=2,
        output_dim=8,
        embed_dim=16,
        num_heads=4,
        bias=True,
    )
    y, ja = model.forward(x, c, index_p, index_v)
    print(y.shape, ja.shape)
    x_recon, ja_inv = model.inverse(y, c, index_p, index_v)
    print(x_recon.shape, ja_inv.shape)
    print(torch.allclose(x, x_recon, atol=1e-5))