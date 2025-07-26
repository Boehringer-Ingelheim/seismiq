from collections.abc import Callable

import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden: int,
        output_size: int,
        hidden_activation: Callable[[], nn.Module] = nn.LeakyReLU,
        pdrop: float | None = 0.1,
        norm_inputs: bool = False,
        norm: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        layers: list[nn.Module] = []
        if norm_inputs:
            layers.append(nn.LazyBatchNorm1d())

        if num_hidden > 0:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    hidden_activation(),
                ]
            )
            for _ in range(num_hidden):
                if pdrop is not None:
                    layers.append(nn.Dropout(pdrop))
                layers.append(nn.Linear(hidden_size, hidden_size))
                if norm:
                    layers.append(nn.LazyBatchNorm1d())
                layers.append(hidden_activation())

            layers.append(nn.Linear(hidden_size, output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for ll in self.layers:
            h = ll(h)
        return h


class PositionalEncoder(nn.Module):
    def __init__(self, num_freqs: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = torch.arange(1, num_freqs + 1).float()

    @property
    def output_shape(self) -> int:
        return 2 * self.num_freqs

    def forward(self, ten: Tensor) -> Tensor:
        # adds the positional encoding to the last dimension
        # based on the position on the second-to-last dimension
        pos = self.encode(torch.arange(ten.shape[-2], device=ten.device).unsqueeze(1))
        return ten + pos

    def encode(self, pos_idx: Tensor) -> Tensor:
        if pos_idx.device != self.freqs.device:
            self.freqs = self.freqs.to(pos_idx.device)

        enc = torch.cat(
            [
                torch.sin(pos_idx / self.freqs),
                torch.sin(pos_idx / self.freqs),
            ],
            dim=-1,
        )

        return enc
