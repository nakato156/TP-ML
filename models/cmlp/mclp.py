import torch
import torch.nn as nn

class CMLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int):
        assert input_dim > 0, "Input dimension must be greater than 0"
        assert hidden_dim > 0, "Hidden dimension must be greater than 0"
        assert hidden_dim < input_dim, "Hidden dimension must be less than input dimension"
        assert hidden_dim // 2 <= 2, "Hidden dimension must be at least half of the input dimension"

        super(CMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.clasif = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.clasif(x)
        return x
