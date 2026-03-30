import torch
import torch.nn as nn


def chaotic_activation(x: torch.Tensor, r: float = 3.8) -> torch.Tensor:
    """Logistic-map inspired nonlinearity applied elementwise."""
    x = torch.tanh(x)
    x01 = (x + 1.0) / 2.0
    y01 = r * x01 * (1.0 - x01)
    return 2.0 * y01 - 1.0


def _normalize_hidden_dims(hidden_dims: list[int] | tuple[int, ...] | None, hidden_size: int) -> list[int]:
    if hidden_dims is None:
        dims = [int(hidden_size)]
    else:
        dims = [int(v) for v in hidden_dims]
    if not dims:
        raise ValueError("hidden_dims must contain at least one layer size")
    if any(v <= 0 for v in dims):
        raise ValueError("hidden_dims values must be positive")
    return dims


class _BaseMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        dims = _normalize_hidden_dims(hidden_dims=hidden_dims, hidden_size=hidden_size)
        layers = [int(input_size)] + dims
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layers[idx], layers[idx + 1]) for idx in range(len(layers) - 1)]
        )
        self.output_layer = nn.Linear(dims[-1], 1)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden_layers:
            h = self._activate(layer(h))
        return self.output_layer(h)


class VanillaMLP(_BaseMLP):
    def __init__(
        self,
        input_size: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_size: int = 128,
    ) -> None:
        super().__init__(input_size=input_size, hidden_dims=hidden_dims, hidden_size=hidden_size)
        self.act = nn.ReLU()

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class ChaoticMLP(_BaseMLP):
    def __init__(
        self,
        input_size: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        hidden_size: int = 128,
        r: float = 3.8,
    ) -> None:
        super().__init__(input_size=input_size, hidden_dims=hidden_dims, hidden_size=hidden_size)
        self.r = r

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return chaotic_activation(x, r=self.r)
