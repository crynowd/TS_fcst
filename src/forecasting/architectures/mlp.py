import torch
import torch.nn as nn


def chaotic_activation(x: torch.Tensor, r: float = 3.8) -> torch.Tensor:
    """Logistic-map inspired nonlinearity applied elementwise."""
    x = torch.tanh(x)
    x01 = (x + 1.0) / 2.0
    y01 = r * x01 * (1.0 - x01)
    return 2.0 * y01 - 1.0


class VanillaMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ChaoticMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, r: float = 3.8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.r = r
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = chaotic_activation(self.fc1(x), r=self.r)
        return self.fc2(h)
