import torch
import types

# Basic FFN model
class BasicFFN(torch.nn.Module):
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 64,
                 output_dim: int = 10,
                 n_layers: int = 1
                 ):
        super(BasicFFN, self).__init__()
        
        self.layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            *[torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            ) for _ in range(n_layers - 1)],
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)
    


# Define a positional encoding function
def abs_pe(x, max_len=5000):
    """
    Generate positional encoding for the input tensor.
    
    Parameters:
    x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
    max_len (int): Maximum length of the sequence.
    
    Returns:
    torch.Tensor: Positional encoding of shape (batch_size, seq_len, d_model).
    """
    batch_size, seq_len, d_model = x.size()
    pe = torch.zeros(batch_size, seq_len, d_model)
    
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    
    return pe

if __name__ == "__main__":
    # Example usage
    model = BasicFFN(input_dim=20, hidden_dim=64, output_dim=10, n_layers=3)
    
    # Test forward pass with random input
    x = torch.randn(2, 20)  # Batch size of 1
    output = model(x)
    print(output.shape)  # Should be [1, 10] for output_dim=10