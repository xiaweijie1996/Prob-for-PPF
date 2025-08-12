import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets


class NiceModelBasic(torch.nn.Module):
    def __init__(self, 
                 full_dim: int = 128,
                 hiddemen_dim: int = 64,
                 n_layers: int = 1,
                 split_ratio: float = 0.6
                 ):
        
        super(NiceModelBasic, self).__init__()
        self.full_dim = full_dim
        self.n_layers = n_layers
        self.hiddemen_dim = hiddemen_dim
        
        self.split_dim1 = int(full_dim * split_ratio)
        self.split_dim2 = full_dim - self.split_dim1
        
        # Define the layers using BasicFFN
        self.fc1 = basicnets.BasicFFN(
            input_dim=self.split_dim1,
            hiddemen_dim=self.hiddemen_dim,
            output_dim=self.split_dim2,
            n_layers=n_layers
        )

        self.fc2 = basicnets.BasicFFN(
            input_dim=self.split_dim2,
            hiddemen_dim=self.hiddemen_dim,
            output_dim=self.split_dim1,
            n_layers=n_layers
        )
        
        # define a nn.parameter vector 
        self.vector = torch.nn.Parameter(torch.randn(1))
        self.sigmoid = self.adjusted_sigmoid
        
    def adjusted_sigmoid(self, x):
        return torch.sigmoid(x) * 2
        
    def forward_direction(self, x):
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # x2
        x21 = x11
        x22 = x12 + self.fc1(x11)
        
        # x3
        x31 = x22
        x32 = x21 + self.fc2(x22)
        
        # Combine the outputs
        x3 = torch.cat([x31, x32], dim=1)
        
        # Make vector multiplication
        x3 = x3 * self.sigmoid(self.vector)
        return x3, self.sigmoid(self.vector)
    
    def inverse_direction(self, x):
        # Make vector division
        x = x / self.sigmoid(self.vector)
        
        # Split the input tensor
        x31, x32 = x[:, :self.split_dim2], x[:, self.split_dim2:]
        
        # x2
        x22 = x31
        x21 = x32 - self.fc2(x22)
        
        # x1
        x11 = x21
        x12 = x22 - self.fc1(x11)
        
        # Combine the outputs
        x1 = torch.cat([x11, x12], dim=1)

        return x1, 1/self.sigmoid(self.vector)
    
    
class NicemModel(torch.nn.Module):
    def __init__(self, 
                 full_dim: int = 128,
                 hiddemen_dim: int = 64,
                 n_layers: int = 1,
                 split_ratio: float = 0.6,
                 n_blocks: int = 2
                 ):
        super(NicemModel, self).__init__()
        
        self.basic_collection = torch.nn.ModuleList([
            NiceModelBasic(
                full_dim=full_dim,
                hiddemen_dim=hiddemen_dim,
                n_layers=n_layers,
                split_ratio=split_ratio
            ) for _ in range(n_blocks)
        ])
        
    def forward(self, x):
        ja = 1
        for block in self.basic_collection:
            x, _ja = block.forward_direction(x)
            ja *= _ja
        return x, ja
    
    def inverse(self, x):
        ja = torch.ones(x.shape[0], 1)
        for block in reversed(self.basic_collection):
            x, _ja = block.inverse_direction(x)
            ja *= _ja
        return x, ja
    
    
if __name__ == "__main__":
    # Example usage
    # model = NiceModelBasic(full_dim=6, n_layers=1, split_ratio=0.4)
    
    # # Test forward pass with random input
    # x = torch.randn(2, 6)  # Batch size of 2
    # output, _ja = model.forward_direction(x)
    # print(output.shape)  # Should be [2, 20] for full_dim=20
    # print(_ja.shape)
    
    # # Test inverse pass
    # inverse_output, _ = model.inverse_direction(output)
    # print(inverse_output.shape)  # Should be [2, 20] for full
    # print(torch.allclose(x, inverse_output))  # Should be True if the inverse is

    # print("jacobian", torch.cumprod(_ja, dim=0)[-1])  # Should be close to 1 if the jacobian is correct
    
    # Test NicemModel with multiple blocks
    test_dim = 1000
    nicem_model = NicemModel(full_dim=test_dim, n_layers=4, split_ratio=0.4, n_blocks=2)
    x = torch.randn(200, test_dim)  # Batch size of 2
    output, _ja = nicem_model.forward(x)
    print(_ja.shape)
    
    # Test inverse pass
    inverse_output, _ = nicem_model.inverse(output)
    print(inverse_output.shape)  # Should be [2, 6] for full_dim=6
    print(torch.max(x- inverse_output))  # Should be True if the inverse is correct
    print(torch.allclose(x, inverse_output))  # Should be True if the inverse is correct
    print("jacobian", _ja)  # Should be close to 1 if the jacobian is correct