import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets

class CFcpflowBasic(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 128,
                 n_layers: int = 1,
                 split_ratio: float = 0.6,
                 hidden_dim_condition: int = 32,
                 output_dim_condition: int = 1,
                 n_layers_condition: int = 2,
                 ):
        
        super(CFcpflowBasic, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim_condition = hidden_dim_condition
        self.output_dim_condition = output_dim_condition
        self.n_layers_condition = n_layers_condition
        
        self.split_dim1 = int(input_dim * split_ratio)
        # print(f"Split dimensions: {self.split_dim1}, {input_dim - self.split_dim1}")
        self.split_dim2 = input_dim - self.split_dim1
        
        # Define the layers using BasicFFN
        self.fcs1 = basicnets.BasicFFN(
            input_dim=self.split_dim1 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.split_dim2,
            n_layers=self.n_layers
        )

        self.fcs2 = basicnets.BasicFFN(
            input_dim=self.split_dim2 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.split_dim1,
            n_layers=self.n_layers
        )
        
        self.fct1 = basicnets.BasicFFN(
            input_dim=self.split_dim1 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.split_dim2,
            n_layers=self.n_layers
        )

        self.fct2 = basicnets.BasicFFN(
            input_dim=self.split_dim2 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.split_dim1,
            n_layers=self.n_layers
        )
        
        self.fc_add_dim = basicnets.BasicFFN(
            input_dim = 1,
            hidden_dim= self.hidden_dim_condition,
            output_dim = self.hidden_dim_condition,
            n_layers= self.n_layers_condition
        )
        self.fc_min_dim = basicnets.BasicFFN(
            input_dim = self.hidden_dim_condition,
            hidden_dim= self.hidden_dim_condition,
            output_dim = 1,
            n_layers= self.n_layers_condition
        )
        
        # Define a special token for null condition
        self.vector = torch.nn.Parameter(torch.randn(1, self.input_dim))
        self.null_token = torch.nn.Parameter(torch.randn(1, self.hidden_dim_condition))
        self.vectorcontrain = self.adjusted_sigmoid
        self.constrain = self.adjusted_sigmoid
        
    def add_pe_and_null_to_c(self, c, index_p, index_v, postional_encoding=False):
        """
        Add positional encoding and null token to the condition vector.
        
        inpit:
        c (torch.Tensor): Condition vector of shape (batch_size, condition_dim).
        
        after self.fc_add_dim:
        c (torch.Tensor): Condition vector of shape (batch_size, hidden_dim_condition).
        
        after adding null token:
        c (torch.Tensor): Condition vector of shape (batch_size, hidden_dim_condition).
        
        after adding positional encoding:
        c_pe (torch.Tensor): Positional encoding of shape (batch_size, hidden_dim_condition).
        
        after self.fc_min_dim:
        c_add (torch.Tensor): Condition vector of shape (batch_size, 1).
        """
        #  torh.sin(index_v) and expand to match batch size
        v_info = torch.sin(torch.tensor(index_v)).to(c.device)
        v_info = v_info.unsqueeze(0).expand(c.shape[0], -1)  # shape (batch_size, 1)
        
        c = torch.cat([c, v_info], dim=-1) 
        c = c.unsqueeze(-1)  # shape (batch_size, condition_dim+1, 1)
        
        # Map c to hidden_dim_condition
        c_add = self.fc_add_dim(c)
        
        # Replace the index_i-th element with the null token
        c_add[:, index_p, :] = self.null_token.to(c.device)
        num_nodes = int(self.condition_dim / 2) + 1
        c_add[:, index_p + (num_nodes-1), :] = self.null_token
        
        # Add positional encoding (if transformer then use this)
        if postional_encoding:
            c_pe = basicnets.abs_pe(c_add)
            c_add = c_pe.to(c.device) + c_add
            
        # Map c_add to a single dimension
        c_add = self.fc_min_dim(c_add)
        
        return c_add.squeeze(-1)  # shape (batch_size, condition_dim +1)
    
    def adjusted_sigmoid(self, x):
        _output =  torch.sigmoid(x/5) * 4 - 2
        # if _output closer to 0, make it 1e-6
        _output = torch.where(torch.abs(_output) < 1e-10, torch.tensor(1e-10, device=x.device), _output)
        return _output
        
        
    def forward(self, x, c, index_p, index_v, postional_encoding=False):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
        
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # Forward pass through the first coupling layer
        x21 = x11
        s1 = torch.exp(self.constrain(self.fcs1(torch.cat([x11, c_processed], dim=-1))))
        t1 = self.fct1(torch.cat([x11, c_processed], dim=-1))
        x22 = x12 * s1 + t1
        _det_ja1 = torch.cumprod(s1, dim=1)[:,-1]

        # Forward pass through the second coupling layer
        x32 = x22
        s2 = torch.exp(self.constrain(self.fcs2(torch.cat([x22, c_processed], dim=-1))))
        t2 = self.fct2(torch.cat([x22, c_processed], dim=-1))
        x31 = x21 * s2 + t2
        _det_ja2 = torch.cumprod(s2, dim=1)[:,-1]

        # Final output
        x3 = torch.cat([x31, x32], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2)
        
        # Make vector multiplication
        scaler = self.vectorcontrain(self.vector.to(x3.device))
        scaler = scaler.expand(x3.shape[0], -1)  # Expand to match batch size
        x3 = x3 * scaler
        
        det_ja = det_ja * torch.abs(torch.cumprod(scaler, dim=1))[:,-1] # Return the Jacobian determinant as well
        
        return x3, det_ja
    
    def inverse(self, x3, c, index_p, index_v, postional_encoding=False):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
        
        # Make vector multiplication
        scaler = self.vectorcontrain(self.vector.to(x3.device))
        scaler = scaler.expand(x3.shape[0], -1)  # Expand to match batch size
        x3 = x3 / scaler
        _det_ja0 = torch.cumprod(1/scaler, dim=1)[:,-1]
        
        # Split the input tensor
        x31, x32 = x3[:, :self.split_dim1], x3[:, self.split_dim1:]
        
        # Inverse pass through the second coupling layer
        x22 = x32
        s2 = torch.exp(self.constrain(self.fcs2(torch.cat([x22, c_processed], dim=-1))))
        t2 = self.fct2(torch.cat([x22, c_processed], dim=-1))
        x21 = (x31 - t2) / s2
        _det_ja2 = torch.cumprod(1/s2, dim=1)[:,-1]
        
        # Inverse pass through the first coupling layer
        x11 = x21
        s1 = torch.exp(self.constrain(self.fcs1(torch.cat([x21, c_processed], dim=-1))))
        t1 = self.fct1(torch.cat([x21, c_processed], dim=-1))
        x12 = (x22 - t1) / s1
        _det_ja1 = torch.cumprod(1/s1, dim=1)[:,-1]
        
        # Final output
        x = torch.cat([x11, x12], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2 * _det_ja0)
        
        return x, det_ja
    
class SimplifiedFcpflow(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 12,
                 n_layers: int = 1,
                 split_ratio: float = 0.5,
                 n_blocks: int = 2,
                 hidden_dim_condition: int = 32,
                 output_dim_condition: int = 1,
                 n_layers_condition: int = 2
                 ):
        super(SimplifiedFcpflow, self).__init__()
        
        self.blocks = torch.nn.ModuleList([
            CFcpflowBasic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                n_layers=n_layers,
                split_ratio=split_ratio,
                hidden_dim_condition=hidden_dim_condition,
                output_dim_condition=output_dim_condition,
                n_layers_condition=n_layers_condition
            ) for _ in range(n_blocks)
        ])
       
    
    def forward(self, x, c, index_p, index_v, postional_encoding=False):
        # Forward through blocks
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in self.blocks:
            x, _ja = block.forward(x, c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
            ja = ja * _ja
            
        return x, ja
    
    def inverse(self, x, c, index_p, index_v, postional_encoding=False):
        # Inverse through blocks
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in reversed(self.blocks):
            x, _ja = block.inverse(x, c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
            ja = ja * _ja
        return x, ja
    
if __name__ == "__main__":
    # Example usage
    test_dim = 2
    c_dim = 100
    index_v = 1
    index_p = 2
    batch_size = 4000
    
    model = SimplifiedFcpflow(
        input_dim=test_dim,
        condition_dim=c_dim,
        n_blocks=3
    )
    x = torch.randn(batch_size, test_dim)*10
    c = torch.randn(batch_size, c_dim)*10
    
    # Forward pass
    y, ja = model.forward(x, c, index_p=index_p, index_v=index_v)
    
    # Inverse pass
    x_recon, ja_inv = model.inverse(y, c, index_p=index_p, index_v=index_v)

    # Check if reconstruction is close to original
    print("Original x.shape:", x.shape, "Reconstructed x.shape:", x_recon.shape, "Jacobian shape:", ja.shape)
    print("Reconstruction error:", torch.norm(x - x_recon))
    print(torch.allclose(x, x_recon, atol=1e-5))
