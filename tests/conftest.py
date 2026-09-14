from __future__ import annotations

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


class TinyNet(nn.Module):
    """Two-class linear model -- small enough that steps are hand-checkable."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BatchNormNet(nn.Module):
    """Has buffers that the forward pass mutates in training mode."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.bn(x))


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    return inputs, targets


@pytest.fixture
def model() -> TinyNet:
    return TinyNet()


@pytest.fixture
def loss_fn() -> nn.Module:
    return nn.CrossEntropyLoss()


def make_loader(
    n_batches: int = 4, batch_size: int = 8, in_features: int = 4
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A fixed list of batches: deterministic and re-iterable."""
    return [
        (torch.randn(batch_size, in_features), torch.randint(0, 2, (batch_size,)))
        for _ in range(n_batches)
    ]
