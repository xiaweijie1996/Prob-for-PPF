import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets


class CNiceModelBasic(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 64,
                 condition_dim: int = 128,
                 n_layers: int = 1,
                 split_ratio: float = 0.6,
                 hidden_dim_condition: int = 32,
                 output_dim_condition: int = 1,
                 n_layers_condition: int = 2
                 ):
        
        super(CNiceModelBasic, self).__init__()
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
        self.fc1 = basicnets.BasicFFN(
            input_dim=self.split_dim1 + self.condition_dim +1,
            hidden_dim=self.hidden_dim,
            output_dim=self.split_dim2,
            n_layers=self.n_layers
        )

        self.fc2 = basicnets.BasicFFN(
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
        self.vectorcontrain = torch.nn.Sigmoid()
    
    def add_pe_and_null_to_c(self, c, index_p, index_v):
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
        v_info = torch.sin(torch.tensor(index_v, dtype=torch.float32))
        v_info = v_info.unsqueeze(0).expand(c.shape[0], -1)  # shape (batch_size, 1)
        
        c = torch.cat([c, v_info], dim=-1) 
        c = c.unsqueeze(-1)  # shape (batch_size, condition_dim+1, 1)
        
        # Map c to hidden_dim_condition
        c_add = self.fc_add_dim(c)
        
        # Replace the index_i-th element with the null token
        c_add[:, index_p, :] = self.null_token
        num_nodes = int(self.condition_dim / 2) 
        c_add[:, index_p + num_nodes, :] = self.null_token
        
        # Add positional encoding
        c_pe = basicnets.abs_pe(c_add)
        c_add = c_pe + c_add
        
        # Map c_add to a single dimension
        c_add = self.fc_min_dim(c_add)
        
        return c_add.squeeze(-1)  # shape (batch_size, condition_dim +1)
        
        
    def adjusted_sigmoid(self, x):
        return torch.sigmoid(x) * 4 - 2  # Adjust the range to [-2, 2]
        
    def forward_direction(self, x, c, index_p, index_v):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v)
        
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # x2
        x21 = x11
        x22 = x12 + self.fc1(torch.cat([x11, c_processed], dim=1))
        
        # x3
        x31 = x22
        x32 = x21 + self.fc2(torch.cat([x22, c_processed], dim=1))
        
        # Combine the outputs
        x3 = torch.cat([x31, x32], dim=1)
        
        # Make vector multiplication
        scaler = self.vectorcontrain(self.vector.to(x.device))
        scaler = scaler.expand(x3.shape[0], -1)  # Expand to match batch size
        x3 = x3 * scaler
        return x3, torch.abs(torch.cumprod(scaler, dim=1))[:,-1] # Return the Jacobian determinant as well
    
    def inverse_direction(self, x3, c, index_p, index_v):
        c_processed = self.add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v)
        
        # Adjust the vector to the device
        scaler = self.vectorcontrain(self.vector.to(x.device))
        scaler = scaler.expand(x3.shape[0], -1)  # Expand to match batch size
        
        # Make vector division
        x3 = x3 / scaler
        
        # Split the input tensor
        x31, x32 = x3[:, :self.split_dim2], x3[:, self.split_dim2:]
        
        # x2
        x22 = x31
        x21 = x32 - self.fc2(torch.cat([x31, c_processed], dim=1))
        
        # x1
        x11 = x21
        x12 = x22 - self.fc1(torch.cat([x21, c_processed], dim=1))
        
        # Combine the outputs
        x1 = torch.cat([x11, x12], dim=1)

        return x1, 1/(torch.abs(torch.cumprod(self.vectorcontrain(self.vector), dim=1))[:,-1])  # Return the Jacobian determinant as well
    
    
class CNicemModel(torch.nn.Module):
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
        super(CNicemModel, self).__init__()
        
        self.input_dim = input_dim
        self.basic_collection = torch.nn.ModuleList([
            CNiceModelBasic(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                split_ratio=split_ratio,
                hidden_dim_condition=hidden_dim_condition,
                output_dim_condition=output_dim_condition,
                n_layers_condition=n_layers_condition
            ) for _ in range(n_blocks)
        ])
        
    def forward(self, x, c, index_p, index_v):
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in self.basic_collection:
            x, _ja = block.forward_direction(x, c, index_p=index_p, index_v=index_v)
            ja = ja * _ja
        return x, ja
    
    def inverse(self, x, c, index_p, index_v):
        ja = torch.ones((x.shape[0]), device=x.device)
        for block in reversed(self.basic_collection):
            x, _ja = block.inverse_direction(x, c, index_p=index_p, index_v=index_v)
            ja = ja * _ja
        return x, ja
    
    
if __name__ == "__main__":
    # Example usage
    # Test NicemModel with multiple blocks
    test_dim = 2
    c_dim = 100
    index_v = 1
    index_p = 2
    nicem_model = CNicemModel(input_dim=test_dim, n_layers=1, split_ratio=0.6, n_blocks=2, 
                              hidden_dim=64, condition_dim=c_dim, 
                              hidden_dim_condition=32, output_dim_condition=1, n_layers_condition=2)
    x = torch.randn(5, test_dim)  # Batch size of 2
    c = torch.randn(5, c_dim)  # Condition vector
    output, _ja = nicem_model.forward(x, c, index_p=index_p, index_v=index_v)
    print(_ja.shape)
    
    # Test inverse pass
    inverse_output, _ = nicem_model.inverse(output, c, index_p=index_p, index_v=index_v)
    print(_.shape)
    print(inverse_output.shape)  # Should be [2, 6] for input_dim=6
    # print(torch.max(x- inverse_output))  # Should be True if the inverse is correct
    print(torch.allclose(x, inverse_output))  # Should be True if the inverse is correct

    # # Test add positional encoding and null token
    # index_i = 2  # Example index
    c_proccessed = nicem_model.basic_collection[0].add_pe_and_null_to_c(c, index_p=index_p, index_v=index_v)
    print("Processed condition shape:", c_proccessed.shape)  
    
    #　check _ja
    print("Jacobian determinant:", _ja[0])
    print("scalers:", nicem_model.basic_collection[0].vectorcontrain(nicem_model.basic_collection[0].vector))
    _ja = torch.cumprod(nicem_model.basic_collection[0].vectorcontrain(nicem_model.basic_collection[0].vector), dim=1)
    print("Jacobian determinant shape:", _ja)