import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets


class NNmodel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(NNmodel, self).__init__()
        self.model = basicnets.BasicFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers
        )
        
        self.constrain = self.adjusted_function
        
    def adjusted_function(self, x):
        # _output = torch.
        _output = torch.tanh(x) * 5  # Scale to [-0.5, 0.5]
        return _output
    
    def forward(self, x):
        
        x = self.model(x)
        x = self.constrain(x)
        return x

if __name__ == "__main__":
    x = torch.randn(10, 6)
    model = NNmodel(input_dim=6, hidden_dim=64, output_dim=6, n_layers=4)
    y = model(x)
    print(y)