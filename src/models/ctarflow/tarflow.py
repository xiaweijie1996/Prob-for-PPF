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
                num_blocks_econder: int,
                output_dim: int,
                
                # Attention features
                embed_dim: int,
                num_heads: int,
                
                # Default features
                num_nodes: int = 33,  # Not used in current implementation
                num_output_nodes: int = 1,  # Not used in current implementation
                bias: bool = True,
                 ):
        
        super(CTarflowBasic, self).__init__()
        
        self.split_dim1 = input_dim // 2
        # self.split_dim2 = input_dim - self.split_dim1
        
        # Define the Transformer encoders
        self.transformer_s1 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks_econder,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            num_nodes=num_nodes + 1 + self.split_dim1,  # because we add one null token and concat one x[split_dim1]
            num_output_nodes=num_output_nodes
        )
        
        self.transformer_s2 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks_econder,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            num_nodes=num_nodes + 1 + self.split_dim1,  # because we add one null token and concat one x[split_dim1]
            num_output_nodes=num_output_nodes
            
        )
        
        self.transformer_b1 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks_econder,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            num_nodes=num_nodes + 1 + self.split_dim1,  # because we add one null token and concat one x[split_dim1]
            num_output_nodes=num_output_nodes
        )
        
        self.transformer_b2 = transformer.TransformerEncoder(
            input_dim=input_dim,
            num_blocks=num_blocks_econder,
            output_dim=output_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            num_nodes=num_nodes + 1 + self.split_dim1,  # because we add one null token and concat one x[split_dim1]
            num_output_nodes=num_output_nodes
        )
        
        self.constrain = self.adjusted_sigmoid
        
    def adjusted_sigmoid(self, x):
        _output =  torch.sigmoid(x/5) * 4 - 2
        # if _output closer to 0, make it 1e-6
        _output = torch.where(torch.abs(_output) < 1e-10, torch.tensor(1e-10, device=x.device), _output)
        return _output
        
    def forward(self, x, c, index_p, index_v):

        # Split the input tensor
        x11, x12 = x[:, :, :self.split_dim1], x[:, :, self.split_dim1:]
        # Forward pass through the first coupling layer
        x21 = x11 # (B, s_len, split_dim1)
        x11_repeat = x11.repeat(1,1,2)  
        x11_c = torch.cat([c, x11_repeat], dim=1)
        s1 = torch.exp(self.constrain(self.transformer_s1(x11_c, index_p, index_v)))
        t1 = self.transformer_b1(x11_c, index_p, index_v)
        x22 = x12 * s1 + t1
        
        _det_ja1 = s1.squeeze()  # (B, 1)
        
        # Forward pass through the second coupling layer
        x32 = x22
        x22_repeat = x22.repeat(1,1,2)  # (B, s_len, orial_dim)
        x22_c = torch.cat([c, x22_repeat], dim=1)
        s2 = torch.exp(self.constrain(self.transformer_s2(x22_c, index_p, index_v)))
        t2 = self.transformer_b2(x22_c, index_p, index_v)
        x31 = x21 * s2 + t2
        _det_ja2 = s2.squeeze()  # (B, s_len)

        # Final output
        x3 = torch.cat([x31, x32], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2)
        
        return x3, det_ja
    
    def inverse(self, x3, c, index_p, index_v):
        # Split the input tensor
        # print("Inverse input shape:", x3.shape)
        x31, x32 = x3[:, :, :self.split_dim1], x3[:, :, self.split_dim1:]
        
        # Inverse pass through the second coupling layer
        x22 = x32
        x22_repeat = x22.repeat(1,1,2)  # (B, s_len, orial_dim)
        x22_c = torch.cat([c, x22_repeat], dim=1)
        s2 = torch.exp(self.constrain(self.transformer_s2(x22_c, index_p, index_v)))
        t2 = self.transformer_b2(x22_c, index_p, index_v)
        x21 = (x31 - t2) / s2
        _det_ja2 = (1/s2).squeeze()  # (B, s_len)
        
        # Inverse pass through the first coupling layer
        x11 = x21
        x11_repeat = x11.repeat(1,1,2)  # (B, s_len, orial_dim)
        x11_c = torch.cat([c, x11_repeat], dim=1)
        s1 = torch.exp(self.constrain(self.transformer_s1(x11_c, index_p, index_v)))
        t1 = self.transformer_b1(x11_c, index_p, index_v)
        x12 = (x22 - t1) / s1
        _det_ja1 = (1/s1).squeeze()  # (B, s_len)
        
        # Final output
        x = torch.cat([x11, x12], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2 )
        
        return x, det_ja
  

class CTarflow(torch.nn.Module):
    def __init__(self, 
                # Flow feature
                num_layers: int,
                 
                # Encoder features
                input_dim: int,
                num_blocks_econder: int,
                output_dim: int,
                
                # Attention features
                embed_dim: int,
                num_heads: int,
                
                # Default features
                num_nodes: int = 33,  # Not used in current implementation
                num_output_nodes: int = 1,  # Not used in current implementation
                bias: bool = True,
                 
                 ):
        super().__init__()
        
        self.layers = torch.nn.ModuleList([
            CTarflowBasic(
                input_dim=input_dim,
                num_blocks_econder=num_blocks_econder,
                output_dim=output_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias,
                num_nodes=num_nodes,
                num_output_nodes=num_output_nodes
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, c, index_p, index_v):
        det_jacobian = 1
        for layer in self.layers:
            x, det_ja = layer(x, c, index_p, index_v)
            det_jacobian = det_jacobian * det_ja
        return x, det_jacobian

    def inverse(self, x, c, index_p, index_v):
        det_jacobian = 1
        for layer in reversed(self.layers):
            x, det_ja = layer.inverse(x, c, index_p, index_v)
            det_jacobian = det_jacobian * det_ja
        return x, det_jacobian
    
if __name__ == "__main__":
    x = torch.randn(2, 1, 2) 
    c = torch.randn(2, 33, 2)
    index_p = 1
    index_v = 2
    
    model = CTarflow(
        num_layers=2,
        input_dim=2,
        num_blocks_econder=2,
        output_dim=1,
        embed_dim=32,
        num_heads=4,
        bias=True
    )
    y, det_jacobian = model(x, c, index_p, index_v)
    print(y.shape)
    x_recon, det_jacobian_inv = model.inverse(y, c, index_p, index_v)
    print(x_recon.shape)
    print(torch.allclose(x, x_recon, atol=1e-5))
    print((x-x_recon).abs().max())
    
    # Draw distribution of x and y