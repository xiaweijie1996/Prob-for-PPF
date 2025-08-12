import torch
import types

# Basic FFN model
class BasicFFN(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 128,
                 hiddemen_dim: int = 64,
                 output_dim: int = 10,
                 n_layers: int = 2
                 ):
        super(BasicFFN, self).__init__()
        
        self.layers = n_layers
        self.input_dim = input_dim
        self.hiddemen_dim = hiddemen_dim
        self.output_dim = output_dim
        
        # Define the layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hiddemen_dim),
            torch.nn.ReLU(),
            *[torch.nn.Sequential(
                torch.nn.Linear(hiddemen_dim, hiddemen_dim),
                # Batch normalization can be added if needed
                torch.nn.BatchNorm1d(hiddemen_dim),
                torch.nn.ReLU()
            ) for _ in range(n_layers - 1)],
            torch.nn.Linear(hiddemen_dim, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    # Example usage
    model = BasicFFN(input_dim=20, hiddemen_dim=64, output_dim=10, n_layers=3)
    # Test forward pass with random input
    x = torch.randn(2, 20)  # Batch size of 1
    output = model(x)
    print(output.shape)  # Should be [1, 10] for output_dim=10