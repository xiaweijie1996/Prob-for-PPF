import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import src.models.basicnetwork.basicnets as basicnets

class Nicemodel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Nicemodel, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode