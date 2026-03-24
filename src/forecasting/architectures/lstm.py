import numpy as np
import torch
import torch.nn as nn


def init_chaotic_(tensor: torch.Tensor, r: float = 3.9) -> None:
    """Fill tensor with a logistic-map sequence scaled similar to Xavier."""
    with torch.no_grad():
        numel = tensor.numel()
        x = np.empty(numel, dtype=np.float32)
        x[0] = 0.123456
        for i in range(1, numel):
            x[i] = r * x[i - 1] * (1.0 - x[i - 1])
        z = 2.0 * x - 1.0
        chaos = torch.from_numpy(z).view_as(tensor)
        if tensor.dim() >= 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            scale = 0.1
        chaos = chaos * scale
        tensor.copy_(chaos)


class LSTMForecast(nn.Module):
    """Two-layer LSTM with linear head."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _c_n) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc(h_last)


class LSTMForecastChaotic(nn.Module):
    """Same architecture but weights initialized via logistic chaos."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        r: float = 3.9,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.r = r
        self.reset_parameters_chaotic()

    def reset_parameters_chaotic(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                init_chaotic_(param, r=self.r)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _c_n) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc(h_last)
