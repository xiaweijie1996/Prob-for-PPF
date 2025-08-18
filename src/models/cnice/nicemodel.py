import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets


class CNiceModelBasic(torch.nn.Module):
    def __init__(self, 
                 full_dim: int = 2,
                 hiddemen_dim: int = 64,
                 condition_dim: int = 128,
                 n_layers: int = 1,
                 split_ratio: float = 0.6
                 ):
        
        super(CNiceModelBasic, self).__init__()
        self.full_dim = full_dim
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.hiddemen_dim = hiddemen_dim
        
        self.split_dim1 = int(full_dim * split_ratio)
        self.split_dim2 = full_dim - self.split_dim1
        
        # Define the layers using BasicFFN
        self.fc1 = basicnets.BasicFFN(
            input_dim=self.split_dim1 + self.condition_dim,
            hiddemen_dim=self.hiddemen_dim,
            output_dim=self.split_dim2,
            n_layers=n_layers
        )

        self.fc2 = basicnets.BasicFFN(
            input_dim=self.split_dim2 + self.condition_dim,
            hiddemen_dim=self.hiddemen_dim,
            output_dim=self.split_dim1,
            n_layers=n_layers
        )
        
        # define a nn.parameter vector 
        self.vector = torch.nn.Parameter(torch.randn(1))
        self.vectorcontrain = torch.nn.Tanh()
        
    def adjusted_sigmoid(self, x):
        return torch.sigmoid(x) * 2
        
    def forward_direction(self, x, c):
        # Split the input tensor
        x11, x12 = x[:, :self.split_dim1], x[:, self.split_dim1:]
        
        # x2
        x21 = x11
        x22 = x12 + self.fc1(torch.cat([x11, c], dim=1))
        
        # x3
        x31 = x22
        x32 = x21 + self.fc2(torch.cat([x22, c], dim=1))
        
        # Combine the outputs
        x3 = torch.cat([x31, x32], dim=1)
        
        # Make vector multiplication
        x3 = x3 * self.vectorcontrain(self.vector.to(x.device))
        return x3, torch.abs(self.vectorcontrain(self.vector))
    
    def inverse_direction(self, x3, c):
        # Make vector division
        x3 = x3 / self.vectorcontrain(self.vector.to(x.device))
        
        # Split the input tensor
        x31, x32 = x3[:, :self.split_dim2], x3[:, self.split_dim2:]
        
        # x2
        x22 = x31
        x21 = x32 - self.fc2(torch.cat([x31, c], dim=1))
        
        # x1
        x11 = x21
        x12 = x22 - self.fc1(torch.cat([x21, c], dim=1))
        
        # Combine the outputs
        x1 = torch.cat([x11, x12], dim=1)

        return x1, 1/  torch.abs(self.vectorcontrain(self.vector))
    
    
class CNicemModel(torch.nn.Module):
    def __init__(self, 
                 full_dim: int = 2,
                 hiddemen_dim: int = 64,
                 conditio_dim: int = 12,
                 n_layers: int = 1,
                 split_ratio: float = 0.6,
                 n_blocks: int = 2
                 ):
        super(CNicemModel, self).__init__()
        
        self.basic_collection = torch.nn.ModuleList([
            CNiceModelBasic(
                full_dim=full_dim,
                condition_dim=conditio_dim,
                hiddemen_dim=hiddemen_dim,
                n_layers=n_layers,
                split_ratio=split_ratio
            ) for _ in range(n_blocks)
        ])
        
    def forward(self, x, c):
        ja = torch.ones(x.shape[0], device=x.device)
        for block in self.basic_collection:
            x, _ja = block.forward_direction(x, c)
            ja *= _ja
        return x, ja
    
    def inverse(self, x, c):
        ja = torch.ones(x.shape[0], device=x.device)
        for block in reversed(self.basic_collection):
            x, _ja = block.inverse_direction(x, c)
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
    test_dim = 20
    c_dim = 1200
    nicem_model = CNicemModel(full_dim=test_dim, n_layers=4, split_ratio=0.6, n_blocks=2, 
                              hiddemen_dim=64, conditio_dim=c_dim)
    x = torch.randn(20, test_dim)  # Batch size of 2
    c = torch.randn(20, c_dim)  # Condition vector
    output, _ja = nicem_model.forward(x, c)
    print(_ja.shape)
    
    # Test inverse pass
    inverse_output, _ = nicem_model.inverse(output, c)
    print(_.shape)
    print(inverse_output.shape)  # Should be [2, 6] for full_dim=6
    # print(torch.max(x- inverse_output))  # Should be True if the inverse is correct
    print(torch.allclose(x, inverse_output))  # Should be True if the inverse is correct
    # print("jacobian", _ja)  # Should be close to 1 if the jacobian is correct