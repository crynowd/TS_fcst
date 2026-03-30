import torch
import torch.nn as nn


class ChaoticLogisticNet(nn.Module):
    """
    Input-driven damped logistic network usable as a reservoir.
    r_layer can be frozen (train_r=False) so only the readout learns.
    """

    def __init__(
        self,
        window_size: int,
        hidden_size: int = 64,
        r_min: float = 2.6,
        r_max: float = 3.2,
        beta: float = 0.1,
        train_r: bool = False,
    ) -> None:
        super().__init__()
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if float(r_min) >= float(r_max):
            raise ValueError("r_min must be smaller than r_max")
        if float(r_min) < 0.0 or float(r_max) > 4.0:
            raise ValueError("r_min/r_max must be within [0, 4] for logistic dynamics")
        if float(beta) <= 0.0 or float(beta) > 1.0:
            raise ValueError("beta must be in (0, 1]")

        self.window_size = int(window_size)
        self.hidden_size = int(hidden_size)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.beta = float(beta)
        self.train_r = train_r

        self.r_layer = nn.Linear(1, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.r_layer.weight)
        nn.init.zeros_(self.r_layer.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

        if not self.train_r:
            for p in self.r_layer.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch, window_size), got {tuple(x.shape)}")
        if x.shape[1] != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {x.shape[1]}")
        batch_size = x.shape[0]
        h0 = torch.full((batch_size, self.hidden_size), 0.5, dtype=x.dtype, device=x.device)
        if not self.train_r:
            with torch.no_grad():
                h = self._forward_states(x, h0)
        else:
            h = self._forward_states(x, h0)
        return self.out_layer(h)

    def _forward_states(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h = h0
        for t in range(self.window_size):
            u_t = x[:, t].unsqueeze(-1)
            r_logits = self.r_layer(u_t)
            r_t = torch.sigmoid(r_logits)
            r_t = self.r_min + (self.r_max - self.r_min) * r_t
            logistic_part = r_t * h * (1.0 - h)
            h = (1.0 - self.beta) * h + self.beta * logistic_part
            h = torch.clamp(h, 1e-6, 1.0 - 1e-6)
        return h
