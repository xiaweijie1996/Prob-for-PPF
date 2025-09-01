import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets

class CRealnvpBasic(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 128,
                 n_layers: int = 1,
                 split_ratio: float = 0.6,
                 hidden_dim_condition: int = 32,
                 output_dim_condition: int = 1,
                 n_layers_condition: int = 2,
                 affine: bool = True
                 ):
        
        super(CRealnvpBasic, self).__init__()
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
        self.null_token = torch.nn.Parameter(torch.randn(1, self.hidden_dim_condition))
        
        # define a nn.parameter vector 
        self.vector = torch.nn.Parameter(torch.randn(1, self.input_dim))
        self.vectorcontrain = self.adjusted_sigmoid
    
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
        v_info = torch.sin(torch.tensor(index_v, dtype=torch.float32)).to(c.device)
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
        return torch.sigmoid(x) * 4 - 2  # Adjust the range to [-2, 2]
    
    def forward(self, x, c, index_p, index_v, postional_encoding=False):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
        
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # Forward pass through the first coupling layer
        x21 = x11
        s1 = torch.exp(self.fcs1(torch.cat([x11, c_processed], dim=-1)))
        t1 = self.fcs1(torch.cat([x11, c_processed], dim=-1))
        x22 = x12 * s1 + t1
        _det_ja1 = torch.cumprod(s1, dim=1)[:,-1]

        # Forward pass through the second coupling layer
        x32 = x22
        s2 = torch.exp(self.fcs2(torch.cat([x22, c_processed], dim=-1)))
        t2 = self.fcs2(torch.cat([x22, c_processed], dim=-1))
        x31 = x21 * s2 + t2
        _det_ja2 = torch.cumprod(s2, dim=1)[:,-1]

        # Final output
        x3 = torch.cat([x31, x32], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2)
        
        return x3, det_ja
    
    def inverse(self, x3, c, index_p, index_v, postional_encoding=False):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v, postional_encoding=postional_encoding)
        
        # Split the input tensor
        x31, x32 = x3[:, :self.split_dim1], x3[:, self.split_dim1:]
        
        # Inverse pass through the second coupling layer
        x22 = x32
        s2 = torch.exp(self.fcs2(torch.cat([x22, c_processed], dim=-1)))
        t2 = self.fcs2(torch.cat([x22, c_processed], dim=-1))
        x21 = (x31 - t2) / s2
        _det_ja2 = torch.cumprod(1/s2, dim=1)[:,-1]
        
        # Inverse pass through the first coupling layer
        x11 = x21
        s1 = torch.exp(self.fcs1(torch.cat([x21, c_processed], dim=-1)))
        t1 = self.fcs1(torch.cat([x21, c_processed], dim=-1))
        x12 = (x22 - t1) / s1
        _det_ja1 = torch.cumprod(1/s1, dim=1)[:,-1]
        
        # Final output
        x = torch.cat([x11, x12], dim=-1)
        det_ja = torch.abs(_det_ja1 * _det_ja2)
        
        return x, det_ja
    
if __name__ == "__main__":
    # Example usage
    test_dim = 2
    c_dim = 100
    index_v = 1
    index_p = 2
    
    model = CRealnvpBasic(input_dim=test_dim, condition_dim=c_dim)
    x = torch.randn(4, test_dim)
    c = torch.randn(4, c_dim)
    
    # Forward pass
    y, ja = model.forward(x, c, index_p=index_p, index_v=index_v)
    print("Output y:", y)
    print("Jacobian determinant:", ja)
    
    # Inverse pass
    x_recon, ja_inv = model.inverse(y, c, index_p=index_p, index_v=index_v)
    print("Reconstructed x:", x_recon)
    print("Inverse Jacobian determinant:", ja_inv)
    
    # Check if reconstruction is close to original
    print("Reconstruction error:", torch.norm(x - x_recon))
    print(torch.allclose(x, x_recon, atol=1e-5))
